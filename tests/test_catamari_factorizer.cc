#if MESHFEM_WITH_CATAMARI && !defined(MESHFEM_USE_LEGACY_CATAMARI)
#include <MeshFEMSparse/Solvers/CatamariFactorizer.hh>
#include <MeshFEMSparse/Solvers/make_cholesky_factorizer.hh>
#include <MeshFEMSparse/SystemAssembler.hh>
#include <catch2/catch.hpp>
#include <tbb/global_control.h>

using namespace MeshFEM;
#include <catamari/norms.hpp>
#include <catamari/sparse_ldl.hpp>

namespace {
template<class Matrix>
void checkSolves(Matrix &A, const SuiteSparseMatrix &scalarA, bool single,
                 bool left, CatamariFactorizer::OrderingMethod ordering,
                 const std::vector<size_t> &pins, bool blockAccel = true) {
    CatamariFactorizer f(false, single);
    f.orderingMethod = ordering;
    f.setUseLeftLooking(left);
    f.setUseBlockAccel(blockAccel);
    Eigen::MatrixXd expected(scalarA.n, 3);
    for (Eigen::Index j = 0; j < expected.cols(); ++j)
        for (Eigen::Index i = 0; i < expected.rows(); ++i)
            expected(i, j) = std::sin(0.2 * (i + 1) * (j + 1));
    for (size_t i : pins) expected.row(i).setZero();
    Eigen::MatrixXd rhs(expected.rows(), expected.cols());
    for (Eigen::Index j = 0; j < rhs.cols(); ++j) {
        Eigen::VectorXd x = expected.col(j);
        rhs.col(j) = scalarA.apply(x);
    }
    const double tol = single ? 2e-5 : 1e-11;
    int solveCheck = 0;
    auto verify = [&](const Eigen::MatrixXd &b) {
        CAPTURE(solveCheck);
        ++solveCheck;
        Eigen::MatrixXd x;
        f.solveMultiRHS(b, x);
        REQUIRE((x - expected).norm() < tol * expected.norm());
        Eigen::VectorXd b0 = b.col(0), x0 = f.solve(b0);
        REQUIRE((x0 - expected.col(0)).norm() < tol * expected.col(0).norm());
        for (bool permuted : {false, true}) {
            Eigen::VectorXd reduced, reducedX(f.n_reduced()), full;
            f.removeFixedEntries(b0, reduced, permuted);
            f.solveRawReduced(reduced.data(), reducedX.data(), CholeskySys::A, permuted);
            f.extractFullSolution(reducedX, full, permuted);
            REQUIRE((full - expected.col(0)).norm() < tol * expected.col(0).norm());
            f.solveRawReducedInPlace(reduced.data(), CholeskySys::A, permuted);
            f.extractFullSolution(reduced, full, permuted);
            REQUIRE((full - expected.col(0)).norm() < tol * expected.col(0).norm());
        }
    };
    for (int repeat = 0; repeat < 2; ++repeat) {
        CAPTURE(single, left, ordering, pins, blockAccel, repeat);
        f.factorizeSymbolic(A, pins);
        REQUIRE(f.getFactorNNZ() > 0);
        REQUIRE(f.getFlopEstimate() > 0);
        f.factorizeNumeric(A);
        verify(rhs);
        f.stashFactorization();
        REQUIRE(f.hasStashedFactorization());
        for (double sigma : {0.75, 0.25}) {
            f.factorizeNumericWithShift(A, sigma);
            verify(rhs + sigma * expected);
        }
        // A itself is a valid same-pattern shift matrix, including block layout.
        f.factorizeNumericWithShift(A, 0.5, static_cast<const SuiteSparseMatrix &>(A));
        verify(1.5 * rhs);
        f.swapStashedFactorization();
        verify(rhs);
        f.swapStashedFactorization();
        verify(1.5 * rhs);
        f.clearStashedFactorization();
        REQUIRE_FALSE(f.hasStashedFactorization());
        A.data() *= 2.0;
        f.factorizeNumeric(A);
        verify(2.0 * rhs);
        A.data() *= 0.5;
    }
    f.clearFactors();
    REQUIRE_FALSE(f.hasNumericFactorization());
    f.factorizeSymbolic(A, pins);
    f.factorizeNumeric(A);
    verify(rhs);
    // Failure must invalidate the previous numeric factor and allow recovery.
    A.data() *= -1.0;
    REQUIRE_THROWS(f.factorizeNumeric(A));
    REQUIRE_FALSE(f.hasNumericFactorization());
    A.data() *= -1.0;
    f.factorizeNumeric(A);
    verify(rhs);
    f.stashFactorization();
    f.clearFactors();
    f.swapStashedFactorization();
    verify(rhs);
    f.factorizeNumeric(A);
    verify(rhs);
    f.clearStashedFactorization();
}

template<size_t BS>
void checkBlocks(bool single, bool left, CatamariFactorizer::OrderingMethod ordering, int pinMode, bool contiguous, bool blockAccel) {
    SystemAssembler<BS> assembler(20);
    auto pattern = assembler.blockSparsityPattern(19, [](size_t i) { return std::array<size_t, 2>{{i, i + 1}}; });
    pattern->setZero();
    auto scalar = pattern->toScalar();
    for (auto entry : scalar) {
        const double value = entry.i == entry.j ? 10.1 : -0.1 * (1 + (entry.i + entry.j) % 5);
        pattern->addNZScalar(entry.i, entry.j, value);
    }
    scalar = pattern->toScalar();
    std::vector<size_t> pins;
    if (pinMode == 1) for (size_t i = 0; i < BS; ++i) pins.push_back(i);
    if (pinMode == 2) pins.push_back(1);
    if (contiguous) {
        auto A = pattern->template cloneWithLayout<true>();
        checkSolves(*A, scalar, single, left, ordering, pins, blockAccel);
    }
    else {
        auto A = pattern->template cloneWithLayout<false>();
        checkSolves(*A, scalar, single, left, ordering, pins, blockAccel);
    }
}
}

TEST_CASE("Catamari precision preserves scalar and block solver behavior", "[catamari][precision]") {
    const bool single = GENERATE(false, true);
    const bool left = GENERATE(false, true);
    const auto ordering = GENERATE(CatamariFactorizer::OrderingMethod::AMD, CatamariFactorizer::OrderingMethod::CholmodNesdisParallel, CatamariFactorizer::OrderingMethod::Adaptive);
    const int pinMode = GENERATE(0, 1, 2);
    const bool contiguous = GENERATE(false, true);
    SECTION("scalar blocks") { checkBlocks<1>(single, left, ordering, pinMode, contiguous, true); }
    SECTION("blocks of size two") { checkBlocks<2>(single, left, ordering, pinMode, contiguous, true); }
    SECTION("blocks of size three") { checkBlocks<3>(single, left, ordering, pinMode, contiguous, true); }
    SECTION("explicit scalar fallback") { checkBlocks<2>(single, left, ordering, pinMode, contiguous, false); }
    SECTION("mixed block sizes fall back to scalar") {
        SystemAssembler<3, 1, 1> assembler(8, 4, 4);
        auto A = assembler.blockSparsityPattern(15, [](size_t i) { return std::array<size_t, 2>{{i, i + 1}}; });
        A->setZero();
        for (auto entry : A->toScalar())
            A->addNZScalar(entry.i, entry.j, entry.i == entry.j ? 10.1 : -0.125);
        auto scalar = A->toScalar();
        const std::vector<size_t> pins = pinMode == 0 ? std::vector<size_t>{} : pinMode == 1 ? std::vector<size_t>{0, 1, 2} : std::vector<size_t>{1};
        if (contiguous) {
            auto B = A->cloneWithLayout<true>();
            checkSolves(*B, scalar, single, left, ordering, pins);
        }
        else {
            auto B = A->cloneWithLayout<false>();
            checkSolves(*B, scalar, single, left, ordering, pins);
        }
    }
    SECTION("plain scalar matrix") {
        TripletMatrix<> triplets(20, 20);
        for (size_t i = 0; i < 20; ++i) {
            triplets.addNZ(i, i, 4.1);
            if (i + 1 < 20) triplets.addNZ(i, i + 1, -0.7);
        }
        SuiteSparseMatrix A(triplets);
        A.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
        SuiteSparseMatrix reference = A;
        checkSolves(A, reference, single, left, ordering, pinMode ? std::vector<size_t>{1} : std::vector<size_t>{});
    }
}

TEST_CASE("Catamari precision factory rejects legacy mode", "[catamari][precision]") {
    REQUIRE_THROWS_AS(make_cholesky_factorizer(CholeskyProvider::CatamariLegacy, true), std::invalid_argument);
}

TEST_CASE("Catamari solve traversal survives changes to a separator forest", "[catamari][solve_kernels]") {
    tbb::global_control limit(tbb::global_control::max_allowed_parallelism, 4);
    CatamariFactorizer factor;
    factor.orderingMethod = CatamariFactorizer::OrderingMethod::CholmodNesdisParallel;
    constexpr int side = 17, component_size = side * side, n = 2 * component_size;
    for (bool connect : {false, true, false}) {
        TripletMatrix<> triplets(n, n);
        for (int i = 0; i < n; ++i) {
            triplets.addNZ(i, i, 5.0);
            if (i % side + 1 < side) triplets.addNZ(i, i + 1, -0.5);
            if (i % component_size + side < component_size) triplets.addNZ(i, i + side, -0.5);
        }
        if (connect) triplets.addNZ(component_size - 1, component_size, -0.5);
        SuiteSparseMatrix A(triplets);
        A.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
        factor.factorizeSymbolic(A, {});
        factor.factorizeNumeric(A);
        Eigen::MatrixXd expected = Eigen::MatrixXd::Random(n, 3), rhs(n, 3), actual;
        for (int j = 0; j < 3; ++j) rhs.col(j) = A.apply(Eigen::VectorXd(expected.col(j)));
        for (int repeat = 0; repeat < 2; ++repeat) {
            factor.solveMultiRHS(rhs, actual);
            REQUIRE((actual - expected).norm() < 1e-11 * expected.norm());
            Eigen::VectorXd b = rhs.col(0);
            REQUIRE((factor.solve(b) - expected.col(0)).norm() < 1e-11 * expected.col(0).norm());
        }
    }
}

TEST_CASE("Catamari handles rounding-induced loss of positive definiteness", "[catamari][precision]") {
    TripletMatrix<> t(2, 2);
    t.nz = {{0, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0 + 1e-8}};
    SuiteSparseMatrix A(t);
    A.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    CatamariFactorizer single(false, true), dbl;
    single.orderingMethod = dbl.orderingMethod = CatamariFactorizer::OrderingMethod::AMD;
    REQUIRE_NOTHROW(dbl.factorize(A));
    REQUIRE_THROWS(single.factorize(A));
    REQUIRE_FALSE(single.hasNumericFactorization());
    Eigen::VectorXd b = Eigen::VectorXd::Ones(2), x(2);
    REQUIRE_THROWS(single.solveRawReduced(b.data(), x.data()));
    Eigen::MatrixXd B = b, X;
    REQUIRE_THROWS(single.solveMultiRHS(B, X));
}

TEST_CASE("Catamari symbolic reuse resets scalar fallback mappings", "[catamari][precision]") {
    const bool single = GENERATE(false, true);
    TripletMatrix<> t(6, 6);
    for (size_t i = 0; i < 6; ++i) {
        t.addNZ(i, i, 4.1);
        if (i + 1 < 6) t.addNZ(i, i + 1, -0.7);
    }
    SuiteSparseMatrix A(t);
    A.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    auto block = BlockCSCHessianBase::fromScalar(A, 2)->cloneWithContiguousBlocks();
    CatamariFactorizer f(false, single);
    f.orderingMethod = CatamariFactorizer::OrderingMethod::AMD;
    f.factorizeSymbolic(*block, {1});
    f.factorizeNumeric(*block);
    f.factorize(A);
    Eigen::VectorXd expected = Eigen::VectorXd::LinSpaced(6, 0.1, 0.9), b = A.apply(expected);
    REQUIRE((f.solve(b) - expected).norm() < (single ? 1e-5 : 1e-12));
}
#if MESHFEM_WITH_CHOLMOD
TEST_CASE("Catamari temporal ND provider defaults to 32 incremental analyses", "[temporal_nd][catamari][factory]") {
    const bool single = GENERATE(false, true);
    auto base = make_cholesky_factorizer(CholeskyProvider::CatamariNesdisReuse, single);
    auto *f = dynamic_cast<CatamariFactorizer *>(base.get());
    REQUIRE(f != nullptr);
    REQUIRE(f->orderingMethod == CatamariFactorizer::OrderingMethod::CholmodNesdisParallel);
    REQUIRE(f->temporalReusePeriod() == 32);
    REQUIRE(base->provider() == CholeskyProvider::CatamariNesdisReuse);

    TripletMatrix<> triplets(16, 16);
    for (size_t v = 0; v < 16; ++v) {
        triplets.addNZ(v, v, 4);
        if (v + 1 < 16) triplets.addNZ(v, v + 1, -1);
    }
    SuiteSparseMatrix matrix(triplets);
    matrix.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    for (size_t call = 0; call < 34; ++call) {
        CAPTURE(call, single);
        base->factorizeSymbolic(matrix, {});
        const auto &stats = f->temporalReuseStatistics();
        const bool full = call == 0 || call == 33;
        REQUIRE(stats.full_rebuild == full);
        REQUIRE(stats.repartitioned_vertices == (full ? 16 : 0));
    }
    base->factorizeNumeric(matrix);
    Eigen::VectorXd expected = Eigen::VectorXd::LinSpaced(16, -1, 1);
    Eigen::VectorXd rhs = matrix.apply(expected);
    REQUIRE((base->solve(rhs) - expected).norm() < 1e-5 * expected.norm());

    f->setTemporalReusePeriod(8);
    REQUIRE(base->provider() == CholeskyProvider::CatamariNesdisReuse);
    f->setTemporalReusePeriod(0);
    REQUIRE(base->provider() == CholeskyProvider::CatamariNesdisParallel);
    auto plain = make_cholesky_factorizer(CholeskyProvider::CatamariNesdisParallel, single);
    REQUIRE(plain->provider() == CholeskyProvider::CatamariNesdisParallel);
    REQUIRE(dynamic_cast<CatamariFactorizer &>(*plain).temporalReusePeriod() == 0);
}

TEST_CASE("Catamari temporal ND uses the actual block size after scalar fallback", "[temporal_nd][catamari]") {
    SystemAssembler<3> assembler(32);
    auto matrix = assembler.blockSparsityPattern(31, [](size_t i) { return std::array<size_t, 2>{{i, i + 1}}; });
    matrix->setZero();
    auto scalar = matrix->toScalar();
    for (auto entry : scalar) matrix->addNZScalar(entry.i, entry.j, entry.i == entry.j ? 10 : -0.1);
    scalar = matrix->toScalar();
    CatamariFactorizer f(false, GENERATE(false, true));
    f.orderingMethod = CatamariFactorizer::OrderingMethod::CholmodNesdisParallel;
    f.setTemporalReusePeriod(2);
    // Repeated partial-block pins fall back to scalar ordering. Whole-block
    // pins switch back to block ordering and must clear the previous history.
    for (const std::vector<size_t> &pins : {std::vector<size_t>{1}, std::vector<size_t>{0, 1, 2}}) {
        for (int repeat = 0; repeat < 2; ++repeat) {
            CAPTURE(pins, repeat);
            f.factorizeSymbolic(*matrix, pins);
            REQUIRE(f.temporalReuseStatistics().full_rebuild == (repeat == 0));
            REQUIRE(f.temporalReuseStatistics().graph_vertices == (pins.size() == 1 ? 95 : 31));
            if (repeat) REQUIRE(f.temporalReuseStatistics().repartitioned_vertices == 0);
            f.factorizeNumeric(*matrix);
            Eigen::VectorXd expected = Eigen::VectorXd::LinSpaced(96, -1, 1);
            for (size_t v : pins) expected[v] = 0;
            Eigen::VectorXd rhs = scalar.apply(expected);
            REQUIRE((f.solve(rhs) - expected).norm() < 1e-5 * expected.norm());
        }
    }
}
#endif

TEST_CASE("Catamari index arrays store one scalar base per block", "[catamari][block_indices]") {
    using namespace catamari;
    using namespace catamari::supernodal_ldl;
    const Int d = GENERATE(1, 2, 3);
    Buffer<Int> sizes(3), degrees(3), members(4 * d);
    sizes[0] = d; sizes[1] = 2 * d; sizes[2] = d;
    degrees[0] = 2 * d; degrees[1] = d; degrees[2] = 0;
    BlasMatrix<double> storage;
    storage.Resize(4 * d * d, 1);
    LowerFactor<double> lower(sizes, degrees, storage.Submatrix(0, 0, 4 * d * d, 1), d);
    REQUIRE(lower.IndexBlockSize() == d);
    REQUIRE(lower.NumStructureEntries() == 3);
    REQUIRE(lower.StructureEnd(0) - lower.StructureBeg(0) == 2);
    REQUIRE(lower.StructureEnd(1) - lower.StructureBeg(1) == 1);
    REQUIRE(lower.StructureEnd(2) == lower.StructureBeg(2));
    lower.StructureBeg(0)[0] = d; lower.StructureBeg(0)[1] = 3 * d;
    lower.StructureBeg(1)[0] = 3 * d;
    for (Int i = 0; i < 4 * d; ++i) members[i] = i < d ? 0 : i < 3 * d ? 1 : 2;
    lower.FillIntersectionSizes(sizes, members);
    REQUIRE(lower.IntersectionSizesEnd(0) - lower.IntersectionSizesBeg(0) == 2);
    REQUIRE(lower.IntersectionSizesBeg(0)[0] == d);
    REQUIRE(lower.IntersectionSizesBeg(0)[1] == d);
    for (Int c = 0; c < d; ++c) {
        REQUIRE(lower.FindScalarRow(0, d + c) == c);
        REQUIRE(lower.FindScalarRow(0, 3 * d + c) == d + c);
        REQUIRE(lower.ScalarStructureBeg(0)[d + c] == 3 * d + c);
    }
    REQUIRE_THROWS(lower.FindScalarRow(0, 2 * d));
    SymmetricOrdering ordering;
    ordering.supernode_sizes = sizes;
    ordering.supernode_offsets.Resize(4);
    ordering.supernode_offsets[0] = 0; ordering.supernode_offsets[1] = d;
    ordering.supernode_offsets[2] = 3 * d; ordering.supernode_offsets[3] = 4 * d;
    auto &af = ordering.assembly_forest;
    af.parents.Resize(3); af.parents[0] = 1; af.parents[1] = 2; af.parents[2] = -1;
    constructChildToParentMap(ordering, &lower);
    REQUIRE(af.child_rel_indices.Size() == 3);
    REQUIRE(af.child_rel_indices_offsets[3] == 3);
    REQUIRE(af.child_rel_indices[0] == 0);
    REQUIRE(af.child_rel_indices[1] == 2 * d);
    REQUIRE(af.child_rel_indices[2] == 0);
    REQUIRE(af.num_child_diag_indices[0] == d);
    REQUIRE(af.num_child_diag_indices[1] == d);
    REQUIRE(af.num_child_diag_indices[2] == 0);
}

namespace {
template<size_t BS>
void checkBlockForest(bool single, bool left) {
    constexpr int side = 13, component = side * side, n = 2 * component;
    std::vector<std::array<size_t, 2>> edges;
    for (size_t i = 0; i < n; ++i) {
        if (i % side + 1 < side) edges.push_back({{i, i + 1}});
        if (i % component + side < component) edges.push_back({{i, i + side}});
    }
    SystemAssembler<BS> assembler(n);
    auto A = assembler.blockSparsityPattern(edges.size(), [&](size_t i) { return edges[i]; });
    A->setZero();
    for (auto e : A->toScalar())
        A->addNZScalar(e.i, e.j, e.i == e.j ? 10.0 : -0.1);
    const auto scalar = A->toScalar();
    checkSolves(*A, scalar, single, left, CatamariFactorizer::OrderingMethod::CholmodNesdisParallel, {});
}
}

TEST_CASE("Catamari compact block maps survive forest factorization and reuse", "[catamari][block_indices]") {
    const bool single = GENERATE(false, true);
    const bool left = GENERATE(false, true);
    const int threads = GENERATE(1, 4);
    tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
    SECTION("two components per block") { checkBlockForest<2>(single, left); }
    SECTION("three components per block") { checkBlockForest<3>(single, left); }
}

TEST_CASE("Catamari symbolic scaling never expands stored row indices", "[catamari][block_indices]") {
    using namespace catamari;
    using namespace catamari::supernodal_ldl;
    const Int d = GENERATE(2, 3);
    const auto algorithm = GENERATE(kLeftLookingLDL, kRightLookingLDL);
    constexpr Int side = 13, n = side * side;
    CoordinateMatrix<double> matrix;
    matrix.Resize(n, n);
    for (Int i = 0; i < n; ++i) {
        matrix.QueueEntryAddition(i, i, 5.0);
        for (Int j : {i % side + 1 < side ? i + 1 : -1, i + side < n ? i + side : -1}) {
            if (j < 0) continue;
            matrix.QueueEntryAddition(i, j, -0.5);
            matrix.QueueEntryAddition(j, i, -0.5);
        }
    }
    matrix.FlushEntryQueues();
    Factorization<double> symbolic;
    Control<double> control;
    control.algorithm = algorithm;
    symbolic.Factor(matrix, SymmetricOrdering{}, control, true);
    auto &before = *symbolic.lower_factor_;
    REQUIRE(before.NumStructureEntries() > 0);
    constructChildToParentMap(symbolic.ordering_, &before);
    auto scaled = symbolic.ExpandSymbolicFactorizationToScalar(d);
    auto &after = *scaled->lower_factor_;
    REQUIRE(after.IndexBlockSize() == d);
    REQUIRE(after.NumStructureEntries() == before.NumStructureEntries());
    constructChildToParentMap(scaled->ordering_, &after);
    for (Int s = 0; s < Int(before.blocks.Size()); ++s) {
        REQUIRE(after.blocks[s].height == d * before.blocks[s].height);
        REQUIRE(after.blocks[s].width == d * before.blocks[s].width);
        for (Int i = 0; i < before.blocks[s].height; ++i)
            REQUIRE(after.StructureBeg(s)[i] == d * before.StructureBeg(s)[i]);
        REQUIRE(scaled->ordering_.assembly_forest.num_child_diag_indices[s] ==
                d * symbolic.ordering_.assembly_forest.num_child_diag_indices[s]);
    }
    const auto &old_map = symbolic.ordering_.assembly_forest.child_rel_indices;
    const auto &new_map = scaled->ordering_.assembly_forest.child_rel_indices;
    REQUIRE(new_map.Size() == old_map.Size());
    for (Int i = 0; i < Int(old_map.Size()); ++i) REQUIRE(new_map[i] == d * old_map[i]);
}
#endif
