#include <MeshFEMSparse/Solvers/ParallelNestedDissection/hierarchical_camd.hh>
#include <MeshFEMSparse/Solvers/ParallelNestedDissection/cholmod_nesdis_parallel.hh>
#include <MeshFEMSparse/Solvers/ParallelNestedDissection/nesdis_bisector.hh>
#include <MeshFEMSparse/Solvers/ParallelNestedDissection/nesdis_options.hh>
#include <MeshFEMSparse/Solvers/cholmod_ordering.hh>
#if MESHFEM_WITH_CATAMARI && !MESHFEM_USE_LEGACY_CATAMARI
#include <MeshFEMSparse/Solvers/CatamariFactorizer.hh>
#include <tbb/task_group.h>
using namespace MeshFEM;
#include <catamari/apply_sparse.hpp>
#include <catamari/blas_matrix.hpp>
#include <catamari/norms.hpp>
#include <catamari/sparse_ldl.hpp>
#endif
#include <tbb/global_control.h>
#include <algorithm>
#include <initializer_list>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <cstdlib>
#include <type_traits>
#include <catch2/catch.hpp>

using namespace MeshFEM::CholmodParallelNesdis;

namespace {
template<class Int>
struct Graph {
    std::vector<Int> p, i;
    cholmod_sparse view{};
    Graph(size_t n, const std::vector<std::pair<Int, Int>> &edges) : p(n + 1, 0) {
        std::vector<std::set<Int>> adjacency(n);
        for (auto [a, b] : edges) { adjacency[a].insert(b); adjacency[b].insert(a); }
        for (size_t v = 0; v < n; ++v) {
            i.insert(i.end(), adjacency[v].begin(), adjacency[v].end());
            p[v + 1] = (Int)i.size();
        }
        view.nrow = view.ncol = n;
        view.nzmax = i.size();
        // Empty adjacency still needs nonnull storage for CHOLMOD.
        if (i.empty()) i.push_back(0);
        view.p = p.data(); view.i = i.data();
        view.packed = view.sorted = 1;
        view.itype = std::is_same_v<Int, int32_t> ? CHOLMOD_INT : CHOLMOD_LONG;
        view.xtype = CHOLMOD_PATTERN;
    }
};
struct Common {
    cholmod_common value;
    Common(int itype = CHOLMOD_LONG) {
        if (itype == CHOLMOD_INT) cholmod_start(&value);
        else cholmod_l_start(&value);
    }
    ~Common() {
        if (value.itype == CHOLMOD_INT) cholmod_finish(&value);
        else cholmod_l_finish(&value);
    }
};
struct Environment {
    std::string name, old;
    bool existed;
    Environment(const char *name_, const char *value) : name(name_) {
        const char *previous = std::getenv(name_);
        existed = previous != nullptr;
        if (existed) old = previous;
        set(value);
    }
    void set(const char *value) {
#ifdef _WIN32
        _putenv_s(name.c_str(), value ? value : "");
#else
        if (value) setenv(name.c_str(), value, 1);
        else unsetenv(name.c_str());
#endif
    }
    ~Environment() { set(existed ? old.c_str() : nullptr); }
};

// Materialize boundary cliques only in small test references.
template<class Int>
std::set<std::pair<Int, Int>> explicit_upper_edges(const HierarchicalCAMDResult<Int> &result) {
    std::set<std::pair<Int, Int>> edges;
    std::vector<std::vector<Int>> boundaries(result.quotient_elements);
    Int n = (Int)result.upper_vertices.size();
    for (auto [a, b] : result.upper_edges) {
        if (b < n) edges.emplace(a, b);
        else boundaries[b - n].push_back(a);
    }
    for (const auto &boundary : boundaries)
        for (size_t a = 0; a < boundary.size(); ++a)
            for (size_t b = a + 1; b < boundary.size(); ++b)
                edges.insert(std::minmax(boundary[a], boundary[b]));
    return edges;
}
}

#ifdef MESHFEM_WITH_SCOTCH
TEMPLATE_TEST_CASE("Scotch bisector preserves CHOLMOD conventions", "[hierarchical_camd]", int32_t, int64_t) {
    using Int = TestType;
    Environment levels("CHOLMOD_NESDIS_SCOTCH_LEVELS", "1");
    Environment threads("CHOLMOD_NESDIS_SCOTCH_THREADS", "1");
    Environment strategy("CHOLMOD_NESDIS_SCOTCH_STRATEGY", nullptr);
    REQUIRE(NesdisOptions{}.scotch_strategy == "fast");
    Environment verify("CHOLMOD_NESDIS_SCOTCH_VERIFY", GENERATE("0", "1"));
    NesdisBisector bisector;
    Common common(std::is_same_v<Int, int32_t> ? CHOLMOD_INT : CHOLMOD_LONG);
    Graph<Int> graph(4, {{0, 1}, {2, 3}});
    Int weights[]{1, 1, 1, 1}, part[4], metis[4];
    auto sep = bisector(&graph.view, weights, (Int *)nullptr, part, 0, &common.value);
    REQUIRE(sep > 0);
    REQUIRE(sep == std::count(part, part + 4, Int(2)));
    if constexpr (std::is_same_v<Int, int32_t>)
        sep = cholmod_metis_bisector(&graph.view, weights, nullptr, metis, &common.value);
    else sep = cholmod_l_metis_bisector(&graph.view, weights, nullptr, metis, &common.value);
    REQUIRE(common.value.status == CHOLMOD_OK);
    REQUIRE(sep > 0);
    REQUIRE(bisector(&graph.view, weights, (Int *)nullptr, part, 1, &common.value) == sep);
    REQUIRE(std::equal(part, part + 4, metis));
    strategy.set("not_a_strategy");
    REQUIRE_THROWS_AS(NesdisBisector(), std::runtime_error);
}
#endif

TEMPLATE_TEST_CASE("hierarchical CAMD exact fill and deterministic constraints", "[hierarchical_camd]", int32_t, int64_t) {
    using Int = TestType;
    const bool skip_upper = GENERATE(false, true);
    Common common;
    const std::vector<Int> parent{2, 2, 6, 5, 5, 6, -1, -1};
    std::mt19937 rng(19);
    for (int trial = 0; trial < 12; ++trial) {
        std::vector<Int> member(64);
        for (size_t v = 0; v < member.size(); ++v) member[v] = (Int)(v % parent.size());
        auto ancestor = [&](Int a, Int b) {
            for (; b >= 0; b = parent[b]) if (a == b) return true;
            return false;
        };
        std::vector<std::pair<Int, Int>> edges;
        for (Int a = 0; a < (Int)member.size(); ++a)
            for (Int b = a + 1; b < (Int)member.size(); ++b)
                if ((ancestor(member[a], member[b]) || ancestor(member[b], member[a])) && rng() % 7 == 0)
                    edges.emplace_back(a, b);
        Graph<Int> graph(member.size(), edges);
        for (int cut = 0; cut <= 4; ++cut) {
            HierarchicalCAMDOptions options;
            options.cut_depth = cut;
            options.verify = true; // Independent elimination reference and tree constraints.
            options.skip_upper = skip_upper;
            std::vector<Int> serial;
            for (size_t threads : {size_t(1), size_t(4)}) {
                tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
                auto result = hierarchical_camd(graph.view, parent.data(), (Int)parent.size(), member.data(), common.value, options);
                REQUIRE(result.permutation.size() == member.size());
                std::set<std::pair<Int, Int>> unique_edges(result.upper_edges.begin(), result.upper_edges.end());
                REQUIRE(unique_edges.size() == result.upper_edges.size());
                if (skip_upper) {
                    REQUIRE(result.upper_edges.empty());
                    REQUIRE(result.quotient_elements == 0);
                    REQUIRE(result.upper_graph_seconds == 0);
                    REQUIRE(result.upper_camd_seconds == 0);
                    auto expected = result.upper_vertices;
                    std::stable_sort(expected.begin(), expected.end(), [&](Int a, Int b) { return member[a] < member[b]; });
                    REQUIRE(std::equal(expected.begin(), expected.end(), result.permutation.end() - expected.size()));
                    auto regular_options = options;
                    regular_options.skip_upper = false;
                    auto regular = hierarchical_camd(graph.view, parent.data(), (Int)parent.size(), member.data(), common.value, regular_options);
                    REQUIRE(std::equal(result.permutation.begin(), result.permutation.end() - expected.size(), regular.permutation.begin()));
                    for (const auto &task : result.tasks) REQUIRE(task.boundary_components == 0);
                }
                if (threads == 1) serial = result.permutation;
                else REQUIRE(result.permutation == serial);
            }
        }
    }
}

TEMPLATE_TEST_CASE("hierarchical CAMD keeps disconnected boundary elements separate", "[hierarchical_camd]", int32_t, int64_t) {
    using Int = TestType;
    // Vertices 0,1 are disconnected within one lower task. Their separate
    // boundaries {2,3} and {3,4} must not introduce an edge between 2 and 4.
    Graph<Int> graph(5, {{0, 2}, {0, 3}, {1, 3}, {1, 4}});
    const Int parent[]{1, -1}, member[]{0, 0, 1, 1, 1};
    Common common;
    HierarchicalCAMDOptions options;
    options.cut_depth = 1;
    options.verify = true;
    auto result = hierarchical_camd(graph.view, parent, Int(2), member, common.value, options);
    const std::set<std::pair<Int, Int>> expected{{0, 1}, {1, 2}};
    REQUIRE(explicit_upper_edges(result) == expected);
    REQUIRE(result.tasks[0].boundary_components == 2);
    REQUIRE(result.quotient_elements == 2);
    REQUIRE(result.upper_edges.size() == 4);
    // The default depth exceeds this forest's height and must clamp to one.
    options.cut_depth = HierarchicalCAMDOptions{}.cut_depth;
    auto clamped = hierarchical_camd(graph.view, parent, Int(2), member, common.value, options);
    REQUIRE(clamped.permutation == result.permutation);
    REQUIRE(clamped.upper_edges == result.upper_edges);
    REQUIRE(clamped.tasks.size() == result.tasks.size());
}

TEMPLATE_TEST_CASE("hierarchical CAMD quotient elements avoid dense boundary cliques", "[hierarchical_camd]", int32_t, int64_t) {
    using Int = TestType;
    std::vector<std::pair<Int, Int>> edges;
    for (Int v = 1; v <= 50; ++v) edges.emplace_back(0, v);
    Graph<Int> graph(51, edges);
    const Int parent[]{1, -1};
    std::vector<Int> member(51, 1);
    member[0] = 0;
    Common common;
    HierarchicalCAMDOptions options;
    options.cut_depth = 1;
    options.verify = true;
    auto elements = hierarchical_camd(graph.view, parent, Int(2), member.data(), common.value, options);
    std::set<std::pair<Int, Int>> clique;
    for (Int a = 0; a < 50; ++a) for (Int b = a + 1; b < 50; ++b) clique.emplace(a, b);
    REQUIRE(explicit_upper_edges(elements) == clique);
    REQUIRE(elements.upper_edges.size() == 50);
    REQUIRE(elements.quotient_elements == 1);
}

TEMPLATE_TEST_CASE("hierarchical CAMD handles empty and invalid forests", "[hierarchical_camd]", int32_t, int64_t) {
    using Int = TestType;
    Common common;
    HierarchicalCAMDOptions options;
    options.verify = true;
    Graph<Int> empty(0, {});
    REQUIRE(hierarchical_camd<Int>(empty.view, nullptr, 0, nullptr, common.value, options).permutation.empty());
    Graph<Int> disconnected(4, {{0, 1}, {2, 3}});
    const Int roots[]{-1, -1}, root_member[]{0, 0, 1, 1};
    auto clamped = hierarchical_camd(disconnected.view, roots, Int(2), root_member, common.value, options);
    REQUIRE(clamped.tasks.size() == 2);
    REQUIRE(clamped.upper_vertices.empty());
    Graph<Int> graph(2, {{0, 1}});
    const Int parent[]{-1, -1}, member[]{0, 1};
    options.cut_depth = 0;
    REQUIRE_THROWS(hierarchical_camd(graph.view, parent, Int(2), member, common.value, options));
}

TEMPLATE_TEST_CASE("parallel nesdis hierarchical CAMD integration", "[hierarchical_camd]", int32_t, int64_t) {
    using Int = TestType;
    Environment scotch_levels("CHOLMOD_NESDIS_SCOTCH_LEVELS", "0");
    const Int components = GENERATE(1, 4);
    Environment subtree_size("CHOLMOD_NESDIS_SERIAL_SUBTREE_SIZE", "64");
    Environment num_threads("CHOLMOD_NESDIS_NUM_THREADS", "4");
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, 4);
    std::vector<std::pair<Int, Int>> edges;
    const Int side = GENERATE(1, 16);
    const Int component_size = side * side, n = components * component_size;
    const Int diagonal_stride = GENERATE(0, 1, 3);
    if (diagonal_stride) for (Int v = 0; v < n; v += diagonal_stride) edges.emplace_back(v, v);
    for (Int c = 0; c < components; ++c) for (Int v = 0; v < component_size; ++v) {
        Int offset = c * component_size;
        if (v % side + 1 < side) edges.emplace_back(offset + v, offset + v + 1);
        if (v + side < component_size) edges.emplace_back(offset + v, offset + v + side);
    }
    // Visit tags must remain valid when descendant separator IDs decrease.
    if (GENERATE(false, true)) {
        std::vector<Int> labels(n);
        std::iota(labels.begin(), labels.end(), Int(0));
        std::mt19937 rng(42);
        std::shuffle(labels.begin(), labels.end(), rng);
        for (auto &[a, b] : edges) { a = labels[a]; b = labels[b]; }
    }
    Graph<Int> graph(n, edges);
    graph.view.stype = -1;
    cholmod_common common;
    if constexpr (std::is_same_v<Int, int32_t>) cholmod_start(&common);
    else                                       cholmod_l_start(&common);
    common.method[0].nd_small = 8;
    common.method[0].nd_compress = GENERATE(0, 1);
    common.method[0].nd_components = GENERATE(0, 1);
    std::vector<Int> perm(n), parent(n), member(n);
    const auto original_p = graph.p, original_i = graph.i;
    auto order = [&](bool parallel, bool full = false) -> int64_t {
        if (full) {
            auto view = graph.view;
            view.stype = 0;
            return nested_dissection_from_graph<Int>(view, perm.data(), parent.data(), member.data(), &common);
        }
        if constexpr (std::is_same_v<Int, int32_t>) {
            auto routine = parallel ? cholmod_nested_dissection_parallel : cholmod_nested_dissection;
            return routine(&graph.view, nullptr, 0, perm.data(), parent.data(), member.data(), &common);
        }
        else {
            auto routine = parallel ? cholmod_l_nested_dissection_parallel : cholmod_l_nested_dissection;
            return routine(&graph.view, nullptr, 0, perm.data(), parent.data(), member.data(), &common);
        }
    };
    {
        Environment global_camd("CHOLMOD_NESDIS_CAMD_CUT_DEPTH", "-1");
        auto ncomponents = order(false);
        REQUIRE(ncomponents > 0);
        auto upstream = perm;
        auto upstream_parent = parent, upstream_member = member;
        for (const char *size : {"64", "4096"}) for (const char *threads : {"1", "2", "4"}) {
            subtree_size.set(size);
            num_threads.set(threads);
            // Repeated nested discovery joins exercise reuse of worker scratch.
            for (int repeat = 0; repeat < 4; ++repeat) for (bool full : {false, true}) {
                REQUIRE(order(true, full) == ncomponents);
                REQUIRE(perm == upstream);
                REQUIRE(std::equal(parent.begin(), parent.begin() + ncomponents, upstream_parent.begin()));
                REQUIRE(member == upstream_member);
                REQUIRE(graph.p == original_p);
                REQUIRE(graph.i == original_i);
            }
        }
        subtree_size.set("64");
    }
    Environment verify("CHOLMOD_NESDIS_CAMD_VERIFY", "1");
    for (const char *skip : {"0", "1"}) for (const char *cut : {"0", "1", "2", "3", "4", "8"}) {
        Environment skip_upper("CHOLMOD_NESDIS_CAMD_SKIP_UPPER", skip);
        Environment depth("CHOLMOD_NESDIS_CAMD_CUT_DEPTH", cut);
        auto count = order(true);
        REQUIRE(count > 0);
        auto reference_perm = perm, reference_parent = parent, reference_member = member;
        REQUIRE(order(true, true) == count);
        REQUIRE(perm == reference_perm);
        REQUIRE(std::equal(parent.begin(), parent.begin() + count, reference_parent.begin()));
        REQUIRE(member == reference_member);
        auto sorted = perm;
        std::sort(sorted.begin(), sorted.end());
        for (Int v = 0; v < n; ++v) REQUIRE(sorted[v] == v);
    }
    {
        Environment depth("CHOLMOD_NESDIS_CAMD_CUT_DEPTH", "4");
        auto count = order(true);
        REQUIRE(count > 0);
        auto depth_four = perm;
        depth.set(nullptr);
        REQUIRE(order(true) == count);
        REQUIRE(perm == depth_four);
        if (count == 1) {
            depth.set("-1");
            REQUIRE(order(true) == count);
            REQUIRE(perm == depth_four);
        }
    }
#ifdef MESHFEM_WITH_SCOTCH
    for (const char *levels : {"1", "2"}) {
        scotch_levels.set(levels);
        for (const char *strategy : {"recursive", "fast2"}) {
            Environment strat("CHOLMOD_NESDIS_SCOTCH_STRATEGY", strategy);
            for (bool full : {false, true}) {
                REQUIRE(order(true, full) > 0);
                auto sorted = perm;
                std::sort(sorted.begin(), sorted.end());
                for (Int v = 0; v < n; ++v) REQUIRE(sorted[v] == v);
                REQUIRE(graph.p == original_p);
                REQUIRE(graph.i == original_i);
            }
        }
    }
#endif
    if constexpr (std::is_same_v<Int, int32_t>) cholmod_finish(&common);
    else                                       cholmod_l_finish(&common);
}

TEMPLATE_TEST_CASE("parallel nesdis preserves numbering of unequal components", "[hierarchical_camd]", int32_t, int64_t) {
    using Int = TestType;
    Environment scotch("CHOLMOD_NESDIS_SCOTCH_LEVELS", "0");
    Environment camd("CHOLMOD_NESDIS_CAMD_CUT_DEPTH", "-1");
    Environment threads("CHOLMOD_NESDIS_NUM_THREADS", "4");
    Environment subtree("CHOLMOD_NESDIS_SERIAL_SUBTREE_SIZE", GENERATE("16", "4096"));
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, 4);
    // Different BFS depths, indistinguishable vertices, and isolated vertices.
    std::vector<std::pair<Int, Int>> edges;
    for (Int a = 0; a < 12; ++a) for (Int b = a + 1; b < 12; ++b) edges.emplace_back(a, b);
    for (Int a = 12; a < 59; ++a) edges.emplace_back(a, a + 1);
    for (Int a = 61; a < 88; ++a) edges.emplace_back(60, a);
    edges.emplace_back(88, 89);
    // Include a nontrivially numbered dense-node parent of the initial search.
    if (GENERATE(false, true)) {
        std::vector<Int> labels(100);
        std::iota(labels.begin(), labels.end(), Int(0));
        std::mt19937 rng(42);
        std::shuffle(labels.begin(), labels.end(), rng);
        for (auto &[a, b] : edges) { a = labels[a]; b = labels[b]; }
    }
    Graph<Int> graph(100, edges);
    graph.view.stype = -1;
    Common common(std::is_same_v<Int, int32_t> ? CHOLMOD_INT : CHOLMOD_LONG);
    common.value.method[0].nd_small = GENERATE(8, 128);
    common.value.method[0].nd_camd = GENERATE(0, 1, 2, 3);
    common.value.method[0].nd_compress = GENERATE(0, 1);
    common.value.method[0].nd_components = GENERATE(0, 1);
    common.value.method[0].prune_dense = GENERATE(-1.0, 0.0);
    std::vector<Int> perm(100), parent(100), member(100);
    auto order = [&](bool parallel) {
        if constexpr (std::is_same_v<Int, int32_t>) {
            auto routine = parallel ? cholmod_nested_dissection_parallel : cholmod_nested_dissection;
            return routine(&graph.view, nullptr, 0, perm.data(), parent.data(), member.data(), &common.value);
        }
        else {
            auto routine = parallel ? cholmod_l_nested_dissection_parallel : cholmod_l_nested_dissection;
            return routine(&graph.view, nullptr, 0, perm.data(), parent.data(), member.data(), &common.value);
        }
    };
    auto count = order(false);
    REQUIRE(count > 0);
    auto reference_perm = perm, reference_parent = parent, reference_member = member;
    REQUIRE(order(true) == count);
    REQUIRE(perm == reference_perm);
    REQUIRE(std::equal(parent.begin(), parent.begin() + count, reference_parent.begin()));
    REQUIRE(member == reference_member);
    auto full_view = graph.view;
    full_view.stype = 0;
    REQUIRE(nested_dissection_from_graph<Int>(full_view, perm.data(), parent.data(), member.data(), &common.value) == count);
    REQUIRE(perm == reference_perm);
    REQUIRE(std::equal(parent.begin(), parent.begin() + count, reference_parent.begin()));
    REQUIRE(member == reference_member);
}

TEMPLATE_TEST_CASE("parallel nesdis rejects invalid settings and remains reusable", "[hierarchical_camd]", int32_t, int64_t) {
    using Int = TestType;
    Environment scotch("CHOLMOD_NESDIS_SCOTCH_LEVELS", "0");
    Environment threads("CHOLMOD_NESDIS_NUM_THREADS", "2");
    Environment subtree("CHOLMOD_NESDIS_SERIAL_SUBTREE_SIZE", "16");
    Environment camd("CHOLMOD_NESDIS_CAMD_CUT_DEPTH", "4");
    std::vector<std::pair<Int, Int>> edges;
    for (Int v = 0; v < 64; ++v) {
        if (v % 8 < 7) edges.emplace_back(v, v + 1);
        if (v + 8 < 64) edges.emplace_back(v, v + 8);
    }
    Graph<Int> graph(64, edges);
    graph.view.stype = -1;
    Common common(std::is_same_v<Int, int32_t> ? CHOLMOD_INT : CHOLMOD_LONG);
    common.value.method[0].nd_small = 8;
    common.value.method[0].nd_compress = GENERATE(0, 1);
    std::vector<Int> perm(64), parent(64), member(64);
    const bool full = GENERATE(false, true);
    auto order = [&]() {
        if (full) {
            auto view = graph.view;
            view.stype = 0;
            return nested_dissection_from_graph<Int>(view, perm.data(), parent.data(), member.data(), &common.value);
        }
        if constexpr (std::is_same_v<Int, int32_t>)
            return cholmod_nested_dissection_parallel(&graph.view, nullptr, 0, perm.data(), parent.data(), member.data(), &common.value);
        else
            return cholmod_l_nested_dissection_parallel(&graph.view, nullptr, 0, perm.data(), parent.data(), member.data(), &common.value);
    };
    const auto count = order();
    REQUIRE(count > 0);
    const auto reference = perm;
    const auto memory_inuse = common.value.memory_inuse;
    for (const auto &[name, value] : std::initializer_list<std::pair<const char *, const char *>>{
            {"CHOLMOD_NESDIS_NUM_THREADS", "2junk"},
            {"CHOLMOD_NESDIS_NUM_THREADS", "-1"},
            {"CHOLMOD_NESDIS_NUM_THREADS", "999999999999999999999"},
            {"CHOLMOD_NESDIS_SERIAL_SUBTREE_SIZE", "0"},
            {"CHOLMOD_NESDIS_CAMD_CUT_DEPTH", "-2"},
            {"CHOLMOD_NESDIS_CAMD_TRACE_TREE", "yes"},
            {"CHOLMOD_NESDIS_SCOTCH_LEVELS", " "},
            {"CHOLMOD_NESDIS_SCOTCH_THREADS", "0"}}) {
        CAPTURE(name, value);
        {
            Environment invalid(name, value);
            REQUIRE(order() == -1);
            REQUIRE(common.value.status == CHOLMOD_INVALID);
            REQUIRE(common.value.memory_inuse == memory_inuse);
        }
        REQUIRE(order() == count);
        REQUIRE(perm == reference);
    }
    {
        // With Scotch this fails during workspace setup, after graph allocation;
        // without Scotch, requesting the unavailable backend fails at that point.
        scotch.set("1");
        Environment invalid("CHOLMOD_NESDIS_SCOTCH_STRATEGY", "not-a-strategy");
        REQUIRE(order() == -1);
        REQUIRE(common.value.status == CHOLMOD_INVALID);
        REQUIRE(common.value.memory_inuse == memory_inuse);
        const auto *flag = static_cast<const Int *>(common.value.Flag);
        for (size_t j = 0; j < common.value.nrow; ++j) REQUIRE(flag[j] == -1);
    }
    scotch.set("0");
    REQUIRE(order() == count);
    REQUIRE(perm == reference);
    {
        // A large valid threshold disables task splitting, including with int32.
        subtree.set("9223372036854775807");
        threads.set("0");
        Environment trace("CHOLMOD_NESDIS_CAMD_TRACE_TREE", "1");
        REQUIRE(order() == count);
        REQUIRE(perm == reference);
    }
}

TEMPLATE_TEST_CASE("CHOLMOD ordering compression updates cached and future contexts", "[hierarchical_camd]", int, SuiteSparse_long) {
    using Int = TestType;
    using Ordering = MeshFEM::CholmodOrdering;
    Environment scotch("CHOLMOD_NESDIS_SCOTCH_LEVELS", "0");
    Environment camd("CHOLMOD_NESDIS_CAMD_CUT_DEPTH", "-1");
    // A grid of three-vertex cliques gives compression a nontrivial effect.
    const Int side = 16, block = 3, n = side * side * block;
    std::vector<std::pair<Int, Int>> edges;
    for (Int v = 0; v < side * side; ++v) {
        for (Int a = 0; a < block; ++a) for (Int b = a + 1; b < block; ++b)
            edges.emplace_back(block * v + a, block * v + b);
        for (Int w : {v + 1, v + side}) {
            if (w >= side * side || (w == v + 1 && v % side == side - 1)) continue;
            for (Int a = 0; a < block; ++a) for (Int b = 0; b < block; ++b)
                edges.emplace_back(block * v + a, block * w + b);
        }
    }
    Graph<Int> graph(n, edges);
    graph.view.stype = -1;
    MeshFEM::TripletMatrix<> triplets(n, n);
    for (Int v = 0; v < n; ++v) triplets.addNZ(v, v, 1000);
    for (auto [a, b] : edges) { triplets.addNZ(a, b, -1); triplets.addNZ(b, a, -1); }
    MeshFEM::SuiteSparseMatrix full(triplets);
    auto matrix = full.toSymmetryMode(MeshFEM::SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE);
    auto reference = [&](bool enabled) {
        Common common(std::is_same_v<Int, int> ? CHOLMOD_INT : CHOLMOD_LONG);
        common.value.method[0].nd_compress = enabled;
        MeshFEM::VecX_T<Int> perm(n), parent(n), member(n);
        int64_t count;
        if constexpr (std::is_same_v<Int, int>)
            count = cholmod_nested_dissection(&graph.view, nullptr, 0, perm.data(), parent.data(), member.data(), &common.value);
        else
            count = cholmod_l_nested_dissection(&graph.view, nullptr, 0, perm.data(), parent.data(), member.data(), &common.value);
        REQUIRE(count > 0);
        return perm;
    };
    const auto compressed = reference(true), uncompressed = reference(false);
    REQUIRE((compressed.array() != uncompressed.array()).any());
    for (auto method : {Ordering::Method::NestedDissection, Ordering::Method::ParallelNestedDissection}) {
        Ordering ordering;
        REQUIRE((ordering.inversePermutation<Int>(matrix, method).array() == compressed.array()).all());
        for (bool enabled : {false, true, false}) {
            ordering.setNestedDissectionCompression(enabled);
            const auto &expected = enabled ? compressed : uncompressed;
            REQUIRE((ordering.inversePermutation<Int>(matrix, method).array() == expected.array()).all());
        }
        Ordering initially_disabled;
        initially_disabled.setNestedDissectionCompression(false);
        REQUIRE((initially_disabled.inversePermutation<Int>(matrix, method).array() == uncompressed.array()).all());
    }
}

TEMPLATE_TEST_CASE("CHOLMOD preliminary forest follows the final permutation", "[hierarchical_camd]", int, SuiteSparse_long) {
    using Int = TestType;
#ifdef MESHFEM_WITH_SCOTCH
    const char *levels = GENERATE("0", "1", "2");
#else
    const char *levels = "0";
#endif
    Environment scotch_levels("CHOLMOD_NESDIS_SCOTCH_LEVELS", levels);
    Environment scotch_threads("CHOLMOD_NESDIS_SCOTCH_THREADS", "1");
    using Ordering = MeshFEM::CholmodOrdering;
    Ordering ordering;
    typename Ordering::PreliminaryAssemblyForest<Int> forest;
    for (int side : {1, 9, 25}) {
        // Two isolated vertices also exercise multiple roots and CAMD's empty
        // variables, whose Pe/Nv output would require special treatment.
        int n = side * side + 2;
        MeshFEM::TripletMatrix<> triplets(n, n);
        for (int v = 0; v < n; ++v) triplets.addNZ(v, v, 5);
        for (int v = 0; v < side * side; ++v) {
            if (v % side + 1 < side) { triplets.addNZ(v, v + 1, -1); triplets.addNZ(v + 1, v, -1); }
            if (v + side < side * side) { triplets.addNZ(v, v + side, -1); triplets.addNZ(v + side, v, -1); }
        }
        MeshFEM::SuiteSparseMatrix full(triplets);
        auto matrix = full.toSymmetryMode(MeshFEM::SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE);
        auto full_pattern = matrix.toSymmetryModeImpl<SuiteSparse_long>(
            MeshFEM::SuiteSparseMatrix::SymmetryMode::NONE, [](size_t ii) { return ii; });
#if MESHFEM_WITH_CATAMARI && !MESHFEM_USE_LEGACY_CATAMARI
        catamari::CoordinateMatrix<double> coordinates;
        coordinates.Resize(n, n);
        for (const auto &entry : triplets.nz) coordinates.QueueEntryAddition(entry.i, entry.j, entry.v);
        coordinates.FlushEntryQueues();
#endif
        for (auto method : {Ordering::Method::NestedDissection, Ordering::Method::ParallelNestedDissection})
        for (const char *cut : std::initializer_list<const char *>{"-1", "2", nullptr})
        for (const char *skip : {"0", "1"}) {
            Environment skip_upper("CHOLMOD_NESDIS_CAMD_SKIP_UPPER", skip);
            Environment depth("CHOLMOD_NESDIS_CAMD_CUT_DEPTH", cut);
            auto perm = ordering.inversePermutation<Int>(matrix, method, &forest);
            auto plain_perm = ordering.inversePermutation<Int>(matrix, method);
            REQUIRE((perm.array() == plain_perm.array()).all());
            if (method == Ordering::Method::ParallelNestedDissection) {
                typename Ordering::PreliminaryAssemblyForest<Int> full_forest;
                auto full_perm = ordering.inversePermutation<Int>(matrix, method, &full_forest, &full_pattern);
                REQUIRE((perm.array() == full_perm.array()).all());
                REQUIRE(forest.sizes == full_forest.sizes);
                REQUIRE(forest.parents == full_forest.parents);
            }
            REQUIRE(std::accumulate(forest.sizes.begin(), forest.sizes.end(), Int(0)) == n);
            REQUIRE(forest.parents.size() == forest.sizes.size());
            std::vector<Int> group(n), position(n);
            Int k = 0;
            for (size_t g = 0; g < forest.sizes.size(); ++g) {
                REQUIRE(forest.sizes[g] > 0);
                REQUIRE((forest.parents[g] == -1 || forest.parents[g] > (Int)g));
                for (Int j = 0; j < forest.sizes[g]; ++j, ++k) {
                    group[perm[k]] = (Int)g;
                    position[perm[k]] = k;
                }
            }
            // Every original edge must stay inside a group or connect a group
            // to an ancestor, so unrelated groups may execute concurrently.
            for (const auto &entry : triplets.nz) {
                Int a = group[entry.i], b = group[entry.j];
                if (position[entry.i] > position[entry.j]) std::swap(a, b);
                while (a >= 0 && a != b) a = forest.parents[a];
                REQUIRE(a == b);
            }
#if MESHFEM_WITH_CATAMARI && !MESHFEM_USE_LEGACY_CATAMARI
            catamari::SymmetricOrdering cat_ordering;
            cat_ordering.permutation.Resize(n);
            cat_ordering.inverse_permutation.Resize(n);
            for (int j = 0; j < n; ++j) {
                cat_ordering.inverse_permutation[j] = perm[j];
                cat_ordering.permutation[perm[j]] = j;
            }
            cat_ordering.supernode_sizes.Resize(forest.sizes.size());
            std::copy(forest.sizes.begin(), forest.sizes.end(), cat_ordering.supernode_sizes.begin());
            catamari::OffsetScan(cat_ordering.supernode_sizes, &cat_ordering.supernode_offsets);
            cat_ordering.assembly_forest.parents.Resize(forest.parents.size());
            std::copy(forest.parents.begin(), forest.parents.end(), cat_ordering.assembly_forest.parents.begin());
            cat_ordering.assembly_forest.FillFromParents();
            catamari::Buffer<catamari::Int> serial_parent, serial_degree, parallel_parent, parallel_degree;
            catamari::scalar_ldl::EliminationForestAndDegrees(coordinates, cat_ordering, &serial_parent, &serial_degree);
            for (int threads : {1, 4}) {
                tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
                catamari::scalar_ldl::ParallelEliminationForestAndDegrees(coordinates, cat_ordering, &parallel_parent, &parallel_degree);
                for (int j = 0; j < n; ++j) {
                    REQUIRE(parallel_parent[j] == serial_parent[j]);
                    REQUIRE(parallel_degree[j] == serial_degree[j]);
                }
            }
#endif
        }
        ordering.inversePermutation<Int>(matrix, Ordering::Method::AMD, &forest);
        REQUIRE(forest.sizes.empty());
        REQUIRE(forest.parents.empty());
    }
}
