#include <MeshFEMSparse/SystemAssembler.hh>
#include <cstdint>
#include <type_traits>
#include <catch2/catch.hpp>

using namespace MeshFEM;

namespace {
struct ThreadCount {
    explicit ThreadCount(int n) : previous(get_max_num_tbb_threads()) { set_max_num_tbb_threads(n); }
    ~ThreadCount() { set_max_num_tbb_threads(previous); }
    int previous;
};

template<typename Index, size_t... Dims>
void checkAssembly(SystemAssembler<Dims...> &assembler) {
    const size_t ne = 4096;
    // 64 leaf partitions all share variable 0. Variable 65 is unused, and the
    // assembler has one additional block outside the partition's variable set.
    std::vector<Index> parent(65, 64), member(66, 64);
    parent.back() = -1;
    for (Index v = 1; v <= 64; ++v) member[v] = v - 1;
    std::vector<std::vector<Index>> stencils(ne);
    for (size_t e = 0; e < ne; ++e) {
        Index v = 1 + Index(e % 64);
        if (e % 17 == 0) stencils[e] = {};
        else if (e % 19 == 0) stencils[e] = {0};
        else stencils[e] = {v, 0, v}; // Duplicate stencil entries must accumulate.
    }
    auto element = [&](size_t e) -> const auto & { return stencils[e]; };
    ElementPartitionFromND<Index> partition(ne, element, parent, member, 1);
    partition.validate(ne, element);
    REQUIRE(partition.numPartitions() == 64);
    REQUIRE(std::count(partition.variableNeedsLock.begin(), partition.variableNeedsLock.end(), 1) == 1);
    auto gradient = [&](size_t e) {
        size_t n = 0;
        for (Index v : stencils[e]) n += assembler.varStructure().blockSize(v);
        Eigen::VectorXd ge(n);
        for (size_t k = 0; k < n; ++k) ge[k] = double((e % 11 + 1) * (k + 1));
        return ge;
    };
    Eigen::VectorXd expected = Eigen::VectorXd::Constant(assembler.numVars(), 7);
    for (size_t e = 0; e < ne; ++e) {
        auto ge = gradient(e);
        size_t lvar = 0;
        for (Index v : stencils[e]) {
            auto [gvar, bs] = assembler.varStructure().blockInfo(v);
            expected.segment(gvar, bs) += ge.segment(lvar, bs);
            lvar += bs;
        }
    }

    // Initialize and exercise the shared lock pool through Hessian assembly
    // first, then reuse it for repeated partitioned gradient calls.
    auto edge = [](size_t e) { return std::array<size_t, 2>{{0, e + 1}}; };
    auto H = assembler.blockSparsityPattern(64, edge);
    H->Ax.assign(H->scalarNNZ(), 0);
    auto hessian = [&](size_t e) -> Eigen::MatrixXd {
        size_t n = assembler.varStructure().blockSize(0) + assembler.varStructure().blockSize(e + 1);
        return Eigen::MatrixXd::Identity(n, n);
    };
    assembler.assembleHessianBlockAccelerated(H->Ax.data(), *H, 64, hessian, edge);
    std::vector<std::atomic<int>> visits(ne);
    for (auto &v : visits) v.store(0);
    auto countedGradient = [&](size_t e) { visits[e].fetch_add(1); return gradient(e); };
    for (int repeat = 0; repeat < 3; ++repeat) {
        Eigen::VectorXd result = Eigen::VectorXd::Constant(assembler.numVars(), 7);
        assembler.assembleGradient(result, partition, countedGradient, element);
        REQUIRE((result - expected).squaredNorm() == 0);
    }
    for (auto &v : visits) REQUIRE(v.load() == 3);
    // Physically reordered elements can omit the explicit identity index list.
    const auto order = partition.elementOrder;
    auto reorderedElement = [&](size_t e) -> const auto & { return element(order[e]); };
    auto reorderedGradient = [&](size_t e) { return gradient(order[e]); };
    auto identity = partition;
    identity.elementOrder.clear();
    REQUIRE(identity.numElements() == ne);
    identity.validate(ne, reorderedElement);
    Eigen::VectorXd reorderedResult = Eigen::VectorXd::Constant(assembler.numVars(), 7);
    assembler.assembleGradient(reorderedResult, identity, reorderedGradient, reorderedElement);
    REQUIRE((reorderedResult - expected).squaredNorm() == 0);
    auto truncated = partition;
    truncated.elementOrder.pop_back();
    REQUIRE_THROWS_AS(truncated.validate(ne, element), std::invalid_argument);
    Eigen::VectorXd regular = Eigen::VectorXd::Constant(assembler.numVars(), 7);
    assembler.assembleGradient(regular, ne, gradient, element);
    REQUIRE((regular - expected).squaredNorm() == 0);
    Eigen::VectorXd tooSmall(assembler.numVars() - 1);
    REQUIRE_THROWS_AS(assembler.assembleGradient(tooSmall, partition, gradient, element), std::invalid_argument);
    SystemAssembler<1> tooFew(65);
    REQUIRE_THROWS_AS(tooFew.assembleGradient(regular, partition, gradient, element), std::invalid_argument);

    // A one-partition call needs no locks; an empty call preserves accumulated values.
    ElementPartitionFromND<Index> single(ne, element, parent, member, 0);
    Eigen::VectorXd result = Eigen::VectorXd::Constant(assembler.numVars(), 7);
    assembler.assembleGradient(result, single, gradient, element);
    REQUIRE((result - expected).squaredNorm() == 0);
    ElementPartitionFromND<Index> empty(0, element, parent, member, 1);
    assembler.assembleGradient(result, empty, gradient, element);
    REQUIRE((result - expected).squaredNorm() == 0);
}
} // namespace

TEST_CASE("Partitioned gradient assembly reuses variable locks", "[partitioned_gradient]") {
    const int threads = GENERATE(1, 14);
    ThreadCount threadCount(threads);
    SystemAssembler<2> uniform(67);
    checkAssembly<int>(uniform);
    checkAssembly<int64_t>(uniform);
    SystemAssembler<3, 1, 1> mixed(33, 17, 17);
    checkAssembly<int>(mixed);
    checkAssembly<int64_t>(mixed);
}

TEST_CASE("ND partition index types", "[nd_partition]") {
    std::vector<std::array<int64_t, 3>> elements{{{0,1,2}}, {{1,0,3}}};
    auto element = [&](size_t e) { return elements[e]; };
    std::vector<int64_t> parent{2,2,-1}, member{2,2,0,1};
    ElementPartitionFromND partition(elements.size(), element, parent, member, 1);
    static_assert(std::is_same_v<decltype(partition)::index_type, int64_t>);
    static_assert(std::is_same_v<decltype(partition.elementOrder)::value_type, int64_t>);
    static_assert(std::is_same_v<decltype(partition.partitionOffsets)::value_type, int64_t>);
    partition.validate(elements.size(), element);
    REQUIRE(partition.numPartitions() == 2);
    REQUIRE(partition.variableNeedsLock == std::vector<uint8_t>{1,1,0,0});
    parent = {1,0};
    REQUIRE_THROWS_AS(ElementPartitionFromND<int64_t>(elements.size(), element, parent, member), std::invalid_argument);
    // Check the bound using a narrow index type without huge allocations.
    std::vector<int8_t> noComponents;
    REQUIRE_THROWS_AS(ElementPartitionFromND<int8_t>(128, element, noComponents, noComponents), std::invalid_argument);
}
