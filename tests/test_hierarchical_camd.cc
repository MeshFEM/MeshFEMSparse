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

namespace {
template<class Int>
void check_separator_forest(const Graph<Int> &graph, Int nc, const std::vector<Int> &parent,
                            const std::vector<Int> &member, const std::vector<Int> &perm) {
    REQUIRE(nc > 0);
    REQUIRE(size_t(nc) <= graph.view.nrow);
    auto sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    std::vector<Int> counts(nc, 0);
    for (size_t v = 0; v < sorted.size(); ++v) {
        REQUIRE(sorted[v] == Int(v));
        REQUIRE(member[v] >= 0);
        REQUIRE(member[v] < nc);
        ++counts[member[v]];
    }
    for (Int c = 0; c < nc; ++c) {
        REQUIRE(counts[c] > 0);
        REQUIRE((parent[c] == -1 || (parent[c] > c && parent[c] < nc)));
    }
    auto ancestor = [&](Int a, Int b) {
        for (; b >= 0; b = parent[b]) if (a == b) return true;
        return false;
    };
    for (size_t v = 0; v < graph.view.nrow; ++v)
        for (Int k = graph.p[v]; k < graph.p[v + 1]; ++k)
            REQUIRE((ancestor(member[v], member[graph.i[k]]) || ancestor(member[graph.i[k]], member[v])));
}

template<class Int>
PersistentSeparatorTree<Int> test_temporal_tree() {
    const Int parent[]{2, 2, 6, 5, 5, 6, 14, 9, 9, 13, 12, 12, 13, 14, -1};
    std::vector<Int> member(30);
    for (Int v = 0; v < 30; ++v) member[v] = v / 2;
    return PersistentSeparatorTree<Int>::from_cholmod(30, 15, parent, member.data());
}

template<class Int>
std::vector<std::pair<Int, Int>> test_temporal_edges() {
    auto tree = test_temporal_tree<Int>();
    std::vector<std::pair<Int, Int>> edges;
    for (Int c = 0; c < 15; ++c) {
        // Leave each node's two vertices unconnected so a later within-node
        // edge is an actual insertion. Diagonals exercise full-graph handling.
        edges.emplace_back(2 * c, 2 * c);
        Int p = tree.nodes[c].parent;
        if (p >= 0) {
            edges.emplace_back(2 * c, 2 * p);
            edges.emplace_back(2 * c + 1, 2 * p);
        }
    }
    return edges;
}
}

TEMPLATE_TEST_CASE("temporal ND detects minimal disjoint invalid subtrees", "[temporal_nd]", int32_t, int64_t) {
    using Int = TestType;
    auto tree = test_temporal_tree<Int>();
    auto edges = test_temporal_edges<Int>();
    std::vector<Int> expected;
    bool full = false;
    SECTION("unchanged") {}
    SECTION("deletion") { edges.clear(); }
    SECTION("within node") { edges.emplace_back(0, 1); }
    SECTION("separator to descendant") { edges.emplace_back(13, 0); }
    SECTION("siblings") { edges.emplace_back(0, 2); expected = {2}; }
    SECTION("duplicate violations") { edges.emplace_back(0, 2); edges.emplace_back(1, 3); expected = {2}; }
    SECTION("disjoint") { edges.emplace_back(0, 2); edges.emplace_back(14, 16); expected = {2, 9}; }
    SECTION("nested") { edges.emplace_back(0, 2); edges.emplace_back(0, 6); expected = {6}; }
    SECTION("root") { edges.emplace_back(0, 14); full = true; }
    Graph<Int> graph(30, edges);
    const auto dirty = tree.dirty_subtrees(graph.p.data(), graph.i.data());
    REQUIRE(dirty.full_rebuild == full);
    REQUIRE(dirty.roots == expected);
    std::vector<Int> parent(30), member(30), perm(30);
    auto nc = tree.flatten(parent.data(), member.data());
    auto roundtrip = PersistentSeparatorTree<Int>::from_cholmod(30, nc, parent.data(), member.data());
    REQUIRE(roundtrip.member == tree.member);
    REQUIRE(roundtrip.roots == tree.roots);
    for (size_t c = 0; c < tree.nodes.size(); ++c) {
        REQUIRE(roundtrip.nodes[c].parent == tree.nodes[c].parent);
        REQUIRE(roundtrip.nodes[c].children == tree.nodes[c].children);
    }
}

TEMPLATE_TEST_CASE("temporal ND parallel dirty scan preserves repair decisions", "[temporal_nd]", int32_t, int64_t) {
    using Int = TestType;
    const Int trees = GENERATE(1, 2), copies = 256, stride = 30 * trees, n = stride * copies;
    const bool unsorted = GENERATE(false, true);
    const auto base = test_temporal_tree<Int>();
    std::vector<Int> parent(15 * trees), member(n);
    for (Int c = 0; c < 15 * trees; ++c) {
        Int p = base.nodes[c % 15].parent;
        parent[c] = p < 0 ? -1 : p + 15 * (c / 15);
    }
    std::vector<std::pair<Int, Int>> edges;
    for (Int block = 0; block < copies; ++block) for (Int t = 0; t < trees; ++t) {
        const Int offset = block * stride + 30 * t;
        for (Int v = 0; v < 30; ++v) member[offset + v] = 15 * t + v / 2;
        for (auto [a, b] : test_temporal_edges<Int>()) edges.emplace_back(offset + a, offset + b);
    }
    const auto tree = PersistentSeparatorTree<Int>::from_cholmod(n, Int(parent.size()), parent.data(), member.data());
    std::vector<Int> expected;
    bool full = false;
    SECTION("valid edges") {}
    SECTION("concurrent marks of the same disjoint nodes") {
        for (Int block = 0; block < copies; ++block) {
            edges.emplace_back(block * stride, block * stride + 2);
            edges.emplace_back(block * stride + 14, block * stride + 16);
        }
        expected = {2, 9};
    }
    SECTION("dirty ancestors subsume concurrent descendant marks") {
        for (Int block = 0; block < copies; ++block) {
            edges.emplace_back(block * stride, block * stride + 2);
            edges.emplace_back(block * stride, block * stride + 6);
        }
        expected = {6};
    }
    SECTION("root violations respect the forest") {
        for (Int block = 0; block < copies; ++block)
            edges.emplace_back(block * stride, block * stride + 14);
        if (trees == 1) full = true;
        else expected = {14};
    }
    SECTION("late full rebuild overrides earlier dirty marks") {
        for (Int block = 0; block < copies; ++block)
            edges.emplace_back(block * stride, block * stride + 2);
        edges.emplace_back(0, trees == 1 ? n - stride + 14 : n - 30);
        full = true;
    }
    Graph<Int> graph(n, edges);
    // Matrix A*A' conversion can produce unsorted columns.
    if (unsorted)
        for (Int v = 0; v < n; ++v) std::reverse(graph.i.begin() + graph.p[v], graph.i.begin() + graph.p[v + 1]);
    for (size_t threads : {1, 2, 4}) {
        tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
        for (int repeat = 0; repeat < 8; ++repeat) {
            const auto dirty = tree.dirty_subtrees(graph.p.data(), graph.i.data());
            REQUIRE(dirty.full_rebuild == full);
            REQUIRE(dirty.roots == expected);
        }
    }
}

TEMPLATE_TEST_CASE("temporal ND collects vertices in tree order without duplicates", "[temporal_nd]", int32_t, int64_t) {
    using Int = TestType;
    const auto tree = test_temporal_tree<Int>();
    // In this subtree the separator's global IDs follow the leaf IDs. Raw
    // collection keeps this order; the batched repair gather orders by global ID.
    const auto vertices = tree.subtree_vertices(2);
    REQUIRE(vertices == std::vector<Int>{4, 5, 0, 1, 2, 3});
    auto all = tree.subtree_vertices(14);
    REQUIRE_FALSE(std::is_sorted(all.begin(), all.end()));
    std::sort(all.begin(), all.end());
    REQUIRE(all.size() == 30);
    for (Int v = 0; v < 30; ++v) REQUIRE(all[v] == v);
}

TEMPLATE_TEST_CASE("temporal ND batch gather matches sorted subtree regions", "[temporal_nd]", int32_t, int64_t) {
    using Int = TestType;
    auto tree = test_temporal_tree<Int>();
    std::vector<Int> dirty{2, 9};
    // A large unrelated root makes the selected regions small enough to use
    // merging. Without it, the same requests exercise the ordered scan.
    if (GENERATE(false, true)) {
        tree.nodes.emplace_back();
        tree.roots.push_back(15);
        for (Int v = 30; v < 1024; ++v) tree.nodes[15].separator_vertices.push_back(v);
        tree.member.resize(1024, 15);
        tree.update_depths();
    }
    SECTION("disjoint subtrees") {}
    SECTION("whole tree") { dirty = {14}; }
    SECTION("after replacement with nonmonotone vertex numbering") {
        const Int parent[]{-1, -1}, member[]{0, 0, 0, 1, 1, 1};
        auto replacement = PersistentSeparatorTree<Int>::from_cholmod(6, 2, parent, member);
        auto vertices = tree.subtree_vertices(2);
        tree.replace(2, replacement, vertices);
        tree.update_depths();
        dirty = {6, 9};
    }
    std::vector<std::vector<Int>> regions;
    auto labels = tree.label_subtree_regions(dirty, regions);
    std::vector<Int> local_index;
    tree.gather_subtree_regions(labels, regions, local_index);
    std::vector<Int> reference_index(tree.member.size(), -1);
    REQUIRE(regions.size() == dirty.size());
    for (size_t r = 0; r < dirty.size(); ++r) {
        auto reference = tree.subtree_vertices(dirty[r]);
        std::sort(reference.begin(), reference.end());
        REQUIRE(regions[r] == reference);
        for (size_t v = 0; v < reference.size(); ++v) reference_index[reference[v]] = Int(v);
    }
    REQUIRE(local_index == reference_index);
}

TEMPLATE_TEST_CASE("temporal ND repairs induced subgraphs and reruns CAMD", "[temporal_nd]", int32_t, int64_t) {
    using Int = TestType;
    Environment threads("CHOLMOD_NESDIS_NUM_THREADS", GENERATE("1", "4"));
    Environment serial_size("CHOLMOD_NESDIS_SERIAL_SUBTREE_SIZE", "4");
    Environment scotch("CHOLMOD_NESDIS_SCOTCH_LEVELS", "0");
    Environment camd("CHOLMOD_NESDIS_CAMD_CUT_DEPTH", GENERATE("-1", "0", "4"));
    Environment verify("CHOLMOD_NESDIS_CAMD_VERIFY", "1");
    Common common(sizeof(Int) == 4 ? CHOLMOD_INT : CHOLMOD_LONG);
    common.value.method[0].nd_small = 4;
    common.value.method[0].nd_compress = GENERATE(0, 1);
    common.value.method[0].prune_dense = -1;
    TemporalReuseState<Int> reuse;
    reuse.temporal_reuse_period = 8;
    auto edges = test_temporal_edges<Int>();
    Graph<Int> initial(30, edges);
    std::vector<Int> parent(30), member(30), perm(30);
    auto order = [&](const Graph<Int> &graph) {
        auto nc = nested_dissection_from_graph<Int>(graph.view, perm.data(), parent.data(), member.data(), &common.value, &reuse);
        REQUIRE(common.value.status == CHOLMOD_OK);
        check_separator_forest(graph, Int(nc), parent, member, perm);
        return nc;
    };
    order(initial);
    REQUIRE(reuse.statistics.full_rebuild);
    // Seed a known valid forest to make exact repair regions independent of
    // METIS version/partition choices. Subsequent repairs use the real ND path.
    reuse.tree = test_temporal_tree<Int>();
    auto before = reuse.tree;
    size_t subtrees = 0, vertices = 0;
    bool full = false;
    SECTION("unchanged") {}
    SECTION("deletion") { edges.clear(); }
    SECTION("within node") { edges.emplace_back(0, 1); }
    SECTION("ancestor") { edges.emplace_back(13, 0); }
    SECTION("siblings") { edges.emplace_back(0, 2); subtrees = 1; vertices = 6; }
    SECTION("duplicates") { edges.emplace_back(0, 2); edges.emplace_back(1, 3); subtrees = 1; vertices = 6; }
    SECTION("disjoint") { edges.emplace_back(0, 2); edges.emplace_back(14, 16); subtrees = 2; vertices = 12; }
    SECTION("nested") { edges.emplace_back(0, 2); edges.emplace_back(0, 6); subtrees = 1; vertices = 14; }
    SECTION("disconnected replacement") { edges = {{0, 2}}; subtrees = 1; vertices = 6; }
    SECTION("root") { edges.emplace_back(0, 14); full = true; vertices = 30; }
    auto dirty = [&] { Graph<Int> g(30, edges); return before.dirty_subtrees(g.p.data(), g.i.data()); }();
    Graph<Int> changed(30, edges);
    order(changed);
    REQUIRE(reuse.statistics.full_rebuild == full);
    REQUIRE(reuse.statistics.recomputed_subtrees == subtrees);
    REQUIRE(reuse.statistics.repartitioned_vertices == vertices);
    REQUIRE(reuse.incremental_analyses_since_rebuild == (full ? 0 : 1));
    REQUIRE(reuse.statistics.nd_seconds >= 0);
    if (!full) {
        std::set<Int> replaced;
        for (Int r : dirty.roots) for (Int c : before.subtree_nodes(r)) replaced.insert(c);
        for (size_t v = 0; v < before.member.size(); ++v)
            if (!replaced.count(before.member[v])) REQUIRE(reuse.tree.member[v] == before.member[v]);
        for (size_t c = 0; c < before.nodes.size(); ++c) {
            if (replaced.count(Int(c))) REQUIRE_FALSE(reuse.tree.nodes[c].alive);
            else {
                REQUIRE(reuse.tree.nodes[c].alive);
                REQUIRE(reuse.tree.nodes[c].separator_vertices == before.nodes[c].separator_vertices);
                REQUIRE(reuse.tree.nodes[c].parent == before.nodes[c].parent);
            }
        }
    }
    const auto last_parent = parent, last_member = member;
    order(changed);
    REQUIRE_FALSE(reuse.statistics.full_rebuild);
    REQUIRE(reuse.statistics.repartitioned_vertices == 0);
    REQUIRE(parent == last_parent);
    REQUIRE(member == last_member);
}

TEMPLATE_TEST_CASE("temporal ND rebuild period and cache lifetime", "[temporal_nd]", int32_t, int64_t) {
    using Int = TestType;
    Environment scotch("CHOLMOD_NESDIS_SCOTCH_LEVELS", "0");
    Common common(sizeof(Int) == 4 ? CHOLMOD_INT : CHOLMOD_LONG);
    common.value.method[0].nd_small = 4;
    common.value.method[0].nd_components = 1;
    TemporalReuseState<Int> reuse;
    reuse.temporal_reuse_period = 2;
    std::vector<Int> parent(30), member(30), perm(30);
    Graph<Int> graph(30, test_temporal_edges<Int>());
    auto order = [&](const Graph<Int> &g) {
        auto nc = nested_dissection_from_graph<Int>(g.view, perm.data(), parent.data(), member.data(), &common.value, &reuse);
        REQUIRE(nc > 0);
        REQUIRE(common.value.status == CHOLMOD_OK);
        return nc;
    };
    order(graph);
    auto initial_parent = parent, initial_member = member, initial_perm = perm;
    for (size_t call = 1; call <= 2; ++call) {
        order(graph);
        REQUIRE_FALSE(reuse.statistics.full_rebuild);
        REQUIRE(reuse.incremental_analyses_since_rebuild == call);
        REQUIRE(reuse.statistics.repartitioned_vertices == 0);
        REQUIRE(parent == initial_parent);
        REQUIRE(member == initial_member);
        REQUIRE(perm == initial_perm);
    }
    order(graph);
    REQUIRE(reuse.statistics.full_rebuild);
    REQUIRE(reuse.incremental_analyses_since_rebuild == 0);
    SECTION("dimension change") { Graph<Int> smaller(29, {}); order(smaller); REQUIRE(reuse.statistics.full_rebuild); }
    SECTION("empty graph clears history") {
        Graph<Int> empty(0, {});
        order(empty);
        REQUIRE(reuse.tree.nodes.empty());
        order(graph);
        REQUIRE(reuse.statistics.full_rebuild);
    }
    SECTION("all dense quick return seeds a reusable tree") {
        common.value.method[0].prune_dense = -1;
        Graph<Int> dense(4, {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}});
        REQUIRE(order(dense) == 1);
        REQUIRE(reuse.statistics.full_rebuild);
        REQUIRE(reuse.tree.nodes.size() == 1);
        REQUIRE(order(dense) == 1);
        REQUIRE_FALSE(reuse.statistics.full_rebuild);
        Graph<Int> deleted(4, {});
        REQUIRE(order(deleted) == 1);
        REQUIRE(reuse.statistics.repartitioned_vertices == 0);
    }
    SECTION("settings change") { common.value.method[0].nd_compress ^= 1; order(graph); REQUIRE(reuse.statistics.full_rebuild); }
    SECTION("explicit reset") { reuse.reset(); order(graph); REQUIRE(reuse.statistics.full_rebuild); }
    SECTION("disabled") {
        reuse.temporal_reuse_period = 0;
        order(graph);
        REQUIRE(reuse.tree.nodes.empty());
        REQUIRE(perm == initial_perm);
        reuse.temporal_reuse_period = 2;
        order(graph);
        REQUIRE(reuse.statistics.full_rebuild);
    }
    SECTION("failure retains previous successful tree") {
        { Environment invalid("CHOLMOD_NESDIS_NUM_THREADS", "invalid");
          REQUIRE(nested_dissection_from_graph<Int>(graph.view, perm.data(), parent.data(), member.data(), &common.value, &reuse) < 0); }
        REQUIRE(reuse.incremental_analyses_since_rebuild == 0);
        order(graph);
        REQUIRE_FALSE(reuse.statistics.full_rebuild);
    }
    SECTION("edge connects forest roots") {
        Graph<Int> disconnected(30, {});
        reuse.reset();
        order(disconnected);
        REQUIRE(reuse.tree.roots.size() > 1);
        Int a = reuse.tree.nodes[reuse.tree.roots[0]].separator_vertices.front();
        Int b = reuse.tree.nodes[reuse.tree.roots[1]].separator_vertices.front();
        Graph<Int> connected(30, {{a, b}});
        auto nc = order(connected);
        REQUIRE(reuse.statistics.full_rebuild);
        check_separator_forest(connected, Int(nc), parent, member, perm);
    }
}

TEMPLATE_TEST_CASE("temporal ND matrix API preserves symmetric and AAT semantics", "[temporal_nd]", int32_t, int64_t) {
    using Int = TestType;
    const bool aat = GENERATE(false, true);
    Environment scotch("CHOLMOD_NESDIS_SCOTCH_LEVELS", "0");
    Common common(sizeof(Int) == 4 ? CHOLMOD_INT : CHOLMOD_LONG);
    common.value.method[0].nd_small = 4;
    common.value.method[0].nd_camd = GENERATE(0, 1, 2);
    TemporalReuseState<Int> reuse;
    reuse.temporal_reuse_period = 2;
    auto edges = test_temporal_edges<Int>();
    std::vector<Int> parent(30), member(30), perm(30);
    for (int step = 0; step < 8; ++step) {
        if (step == 2) edges.emplace_back(0, 2);
        if (step == 4) edges.emplace_back(0, 14);
        if (step == 6) edges.clear();
        Graph<Int> matrix(30, edges);
        if (!aat) matrix.view.stype = 1;
        std::vector<std::pair<Int, Int>> product;
        if (aat) {
            for (Int col = 0; col < 30; ++col)
                for (Int a = matrix.p[col]; a < matrix.p[col + 1]; ++a)
                    for (Int b = a + 1; b < matrix.p[col + 1]; ++b)
                        product.emplace_back(matrix.i[a], matrix.i[b]);
        }
        Graph<Int> graph(30, aat ? product : edges);
        auto nc = nested_dissection<Int>(&matrix.view, nullptr, 0, perm.data(), parent.data(), member.data(), &common.value, reuse);
        REQUIRE(common.value.status == CHOLMOD_OK);
        check_separator_forest(graph, Int(nc), parent, member, perm);
        if (step == 0) REQUIRE(reuse.statistics.full_rebuild);
        if (step == 1) REQUIRE_FALSE(reuse.statistics.full_rebuild);
    }
}

#if MESHFEM_WITH_CATAMARI && !MESHFEM_USE_LEGACY_CATAMARI
TEST_CASE("Catamari temporal ND history survives symbolic analyses and resets for pin changes", "[temporal_nd][catamari]") {
    Environment threads("CHOLMOD_NESDIS_NUM_THREADS", "4");
    Environment scotch("CHOLMOD_NESDIS_SCOTCH_LEVELS", "0");
    CatamariFactorizer factorizer(false, GENERATE(false, true));
    factorizer.orderingMethod = CatamariFactorizer::OrderingMethod::CholmodNesdisParallel;
    factorizer.setTemporalReusePeriod(2);
    REQUIRE(factorizer.temporalReusePeriod() == 2);
    const int n = 256, side = 16;
    std::vector<std::pair<int, int>> edges;
    for (int v = 0; v < n; ++v) {
        if (v % side + 1 < side) edges.emplace_back(v, v + 1);
        if (v + side < n) edges.emplace_back(v, v + side);
    }
    for (int step = 0; step < 9; ++step) {
        CAPTURE(step);
        if (step == 4) edges.emplace_back(0, n - 1);
        MeshFEM::TripletMatrix<> triplets(n, n);
        for (int v = 0; v < n; ++v) triplets.addNZ(v, v, 10);
        for (auto [a, b] : edges) { triplets.addNZ(a, b, -1); triplets.addNZ(b, a, -1); }
        MeshFEM::SuiteSparseMatrix full(triplets);
        auto matrix = full.toSymmetryMode(MeshFEM::SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE);
        std::vector<size_t> pins;
        if (step >= 5) pins = {size_t(step <= 6 ? 0 : 1)};
        factorizer.factorizeSymbolic(matrix, pins);
        const auto &statistics = factorizer.temporalReuseStatistics();
        if (step == 0 || step == 3 || step == 5 || step == 7) REQUIRE(statistics.full_rebuild);
        if (step == 1 || step == 2 || step == 6 || step == 8) {
            REQUIRE_FALSE(statistics.full_rebuild);
            REQUIRE(statistics.repartitioned_vertices == 0);
        }
        factorizer.factorizeNumeric(matrix);
        Eigen::VectorXd expected = Eigen::VectorXd::LinSpaced(n, -1, 1);
        for (size_t v : pins) expected[v] = 0;
        Eigen::VectorXd rhs = matrix.apply(expected);
        REQUIRE((factorizer.solve(rhs) - expected).norm() < 1e-5 * expected.norm());
    }
}
#endif

TEMPLATE_TEST_CASE("temporal ND repeatedly splices real parallel separator trees", "[temporal_nd]", int32_t, int64_t) {
    using Int = TestType;
    Environment threads("CHOLMOD_NESDIS_NUM_THREADS", "4");
    Environment serial_size("CHOLMOD_NESDIS_SERIAL_SUBTREE_SIZE", "32");
#ifdef MESHFEM_WITH_SCOTCH
    Environment scotch("CHOLMOD_NESDIS_SCOTCH_LEVELS", GENERATE("0", "2"));
#else
    Environment scotch("CHOLMOD_NESDIS_SCOTCH_LEVELS", "0");
#endif
    Environment scotch_threads("CHOLMOD_NESDIS_SCOTCH_THREADS", "1");
    Environment verify("CHOLMOD_NESDIS_CAMD_VERIFY", "1");
    Common common(sizeof(Int) == 4 ? CHOLMOD_INT : CHOLMOD_LONG);
    common.value.method[0].nd_small = 8;
    common.value.method[0].nd_compress = GENERATE(0, 1);
    TemporalReuseState<Int> reuse;
    reuse.temporal_reuse_period = 8;
    const Int side = 32, n = side * side;
    std::vector<std::pair<Int, Int>> edges;
    for (Int v = 0; v < n; ++v) {
        if (v % side + 1 < side) edges.emplace_back(v, v + 1);
        if (v + side < n) edges.emplace_back(v, v + side);
    }
    std::vector<Int> perm(n), parent(n), member(n);
    size_t incremental_repairs = 0;
    for (int step = 0; step < 18; ++step) {
        CAPTURE(step);
        if (step % 3 == 1) {
            // Select two large, disjoint, non-root regions from the current
            // METIS/Scotch tree, then connect distinct children in each.
            std::vector<std::pair<size_t, Int>> candidates;
            for (size_t c = 0; c < reuse.tree.nodes.size(); ++c) {
                const auto &node = reuse.tree.nodes[c];
                if (node.alive && node.parent >= 0 && node.children.size() >= 2)
                    candidates.emplace_back(reuse.tree.subtree_vertices(Int(c)).size(), Int(c));
            }
            std::sort(candidates.rbegin(), candidates.rend());
            std::vector<Int> selected;
            for (auto [size, c] : candidates) {
                bool overlaps = false;
                for (Int r : selected) {
                    Int lca = reuse.tree.lca(c, r);
                    overlaps |= lca == c || lca == r;
                }
                if (overlaps) continue;
                selected.push_back(c);
                const auto &children = reuse.tree.nodes[c].children;
                edges.emplace_back(reuse.tree.subtree_vertices(children[0]).front(),
                                   reuse.tree.subtree_vertices(children[1]).front());
                if (selected.size() == 2) break;
            }
            REQUIRE(selected.size() == 2);
        }
        if (step % 3 == 2 && !edges.empty()) edges.pop_back();
        Graph<Int> graph(n, edges);
        const bool due = reuse.tree.roots.empty() || reuse.incremental_analyses_since_rebuild == reuse.temporal_reuse_period;
        auto dirty = due ? typename PersistentSeparatorTree<Int>::DirtySubtrees{}
                         : reuse.tree.dirty_subtrees(graph.p.data(), graph.i.data());
        const bool full = due || dirty.full_rebuild;
        size_t vertices = full ? size_t(n) : 0;
        if (!full) for (Int r : dirty.roots) vertices += reuse.tree.subtree_vertices(r).size();
        auto nc = nested_dissection_from_graph<Int>(graph.view, perm.data(), parent.data(), member.data(), &common.value, &reuse);
        REQUIRE(common.value.status == CHOLMOD_OK);
        check_separator_forest(graph, Int(nc), parent, member, perm);
        REQUIRE(reuse.statistics.full_rebuild == full);
        REQUIRE(reuse.statistics.repartitioned_vertices == vertices);
        if (!full) REQUIRE(reuse.statistics.recomputed_subtrees == dirty.roots.size());
        if (!full && !dirty.roots.empty()) ++incremental_repairs;
    }
    REQUIRE(incremental_repairs >= 4);
}

TEMPLATE_TEST_CASE("temporal ND repairs rectangular matrix products with changing column subsets", "[temporal_nd]", int32_t, int64_t) {
    using Int = TestType;
    Environment scotch("CHOLMOD_NESDIS_SCOTCH_LEVELS", "0");
    Common common(sizeof(Int) == 4 ? CHOLMOD_INT : CHOLMOD_LONG);
    common.value.method[0].nd_small = 4;
    TemporalReuseState<Int> reuse;
    reuse.temporal_reuse_period = 8;
    auto edges = test_temporal_edges<Int>();
    std::reverse(edges.begin(), edges.end());
    std::vector<Int> perm(30), parent(30), member(30);
    for (int step = 0; step < 3; ++step) {
        if (step == 1) { edges.emplace_back(0, 2); edges.emplace_back(14, 16); }
        // One incidence column per edge makes A*A' exactly the test graph.
        std::vector<Int> p{0}, i;
        for (auto [a, b] : edges) {
            i.push_back(std::min(a, b));
            if (a != b) i.push_back(std::max(a, b));
            p.push_back(Int(i.size()));
        }
        cholmod_sparse A{};
        A.nrow = 30; A.ncol = edges.size(); A.nzmax = i.size();
        A.p = p.data(); A.i = i.data();
        A.packed = A.sorted = 1;
        A.itype = common.value.itype; A.xtype = CHOLMOD_PATTERN;
        // Initially exclude the two new columns, then include them to trigger
        // two disjoint induced-graph repairs through the A*A' entry point.
        std::vector<Int> fset(edges.size() - (step == 1 ? 2 : 0));
        std::iota(fset.begin(), fset.end(), Int(0));
        auto nc = nested_dissection<Int>(&A, fset.data(), fset.size(), perm.data(), parent.data(), member.data(), &common.value, reuse);
        REQUIRE(common.value.status == CHOLMOD_OK);
        auto expected_edges = edges;
        if (step == 1) expected_edges.resize(expected_edges.size() - 2);
        Graph<Int> expected(30, expected_edges);
        check_separator_forest(expected, Int(nc), parent, member, perm);
        if (step == 0) reuse.tree = test_temporal_tree<Int>();
        if (step == 1) REQUIRE(reuse.statistics.repartitioned_vertices == 0);
        if (step == 2) {
            REQUIRE_FALSE(reuse.statistics.full_rebuild);
            REQUIRE(reuse.statistics.recomputed_subtrees == 2);
            REQUIRE(reuse.statistics.repartitioned_vertices == 12);
        }
    }
}

TEMPLATE_TEST_CASE("temporal ND repairs forest roots independently", "[temporal_nd]", int32_t, int64_t) {
    using Int = TestType;
    Environment threads("CHOLMOD_NESDIS_NUM_THREADS", GENERATE("1", "4"));
    Environment serial_size("CHOLMOD_NESDIS_SERIAL_SUBTREE_SIZE", "4");
    Environment scotch("CHOLMOD_NESDIS_SCOTCH_LEVELS", "0");
    Environment camd("CHOLMOD_NESDIS_CAMD_CUT_DEPTH", GENERATE("-1", "4"));
    Environment verify("CHOLMOD_NESDIS_CAMD_VERIFY", "1");
    Common common(sizeof(Int) == 4 ? CHOLMOD_INT : CHOLMOD_LONG);
    common.value.method[0].nd_small = 4;
    common.value.method[0].nd_components = 1;
    common.value.method[0].nd_compress = GENERATE(0, 1);
    TemporalReuseState<Int> reuse;
    reuse.temporal_reuse_period = 8;
    const auto tree = test_temporal_tree<Int>();
    std::vector<Int> parent(90), member(90), perm(90);
    std::vector<std::pair<Int, Int>> edges;
    for (Int t = 0; t < 3; ++t)
        for (auto [a, b] : test_temporal_edges<Int>()) edges.emplace_back(a + 30 * t, b + 30 * t);
    auto order = [&](const Graph<Int> &graph) {
        auto nc = nested_dissection_from_graph<Int>(graph.view, perm.data(), parent.data(), member.data(), &common.value, &reuse);
        REQUIRE(common.value.status == CHOLMOD_OK);
        check_separator_forest(graph, Int(nc), parent, member, perm);
        return nc;
    };
    Graph<Int> initial(90, edges);
    order(initial);
    // Three known trees make the intended dirty roots independent of the
    // bisector's choices; repair still runs through the real parallel ND path.
    for (Int c = 0; c < 45; ++c) {
        Int p = tree.nodes[c % 15].parent;
        parent[c] = p < 0 ? -1 : p + 15 * (c / 15);
    }
    for (Int v = 0; v < 90; ++v) member[v] = v / 2;
    reuse.tree = PersistentSeparatorTree<Int>::from_cholmod(90, 45, parent.data(), member.data());
    const auto before = reuse.tree;
    REQUIRE(before.roots == std::vector<Int>{14, 29, 44});
    std::vector<Int> expected;
    bool full = false, disconnected = false;
    SECTION("first root") { edges.emplace_back(0, 14); expected = {14}; }
    SECTION("middle root") { edges.emplace_back(30, 44); expected = {29}; }
    SECTION("last root") { edges.emplace_back(60, 74); expected = {44}; }
    SECTION("two roots") {
        edges.emplace_back(0, 14); edges.emplace_back(60, 74); expected = {14, 44};
    }
    SECTION("root and subtree in another tree") {
        edges.emplace_back(0, 14); edges.emplace_back(30, 32); expected = {14, 17};
    }
    SECTION("root subsumes descendant") {
        edges.emplace_back(0, 2); edges.emplace_back(0, 14); expected = {14};
    }
    SECTION("root splits into multiple roots") {
        edges.erase(std::remove_if(edges.begin(), edges.end(), [](auto e) { return e.first < 30; }), edges.end());
        edges.emplace_back(0, 14); expected = {14}; disconnected = true;
    }
    SECTION("cross-tree edge still forces full ND") {
        edges.emplace_back(0, 30); full = true;
    }
    Graph<Int> changed(90, edges);
    auto dirty = before.dirty_subtrees(changed.p.data(), changed.i.data());
    REQUIRE(dirty.full_rebuild == full);
    REQUIRE(dirty.roots == expected);
    size_t vertices = full ? 90 : 0;
    std::set<Int> replaced;
    for (Int r : expected) {
        vertices += before.subtree_vertices(r).size();
        for (Int c : before.subtree_nodes(r)) replaced.insert(c);
    }
    auto nc = order(changed);
    REQUIRE(reuse.statistics.full_rebuild == full);
    REQUIRE(reuse.statistics.recomputed_subtrees == expected.size());
    REQUIRE(reuse.statistics.repartitioned_vertices == vertices);
    REQUIRE(reuse.incremental_analyses_since_rebuild == (full ? 0 : 1));
    if (!full) {
        for (Int c = 0; c < 45; ++c) {
            if (replaced.count(c)) REQUIRE_FALSE(reuse.tree.nodes[c].alive);
            else {
                REQUIRE(reuse.tree.nodes[c].alive);
                REQUIRE(reuse.tree.nodes[c].parent == before.nodes[c].parent);
                REQUIRE(reuse.tree.nodes[c].separator_vertices == before.nodes[c].separator_vertices);
                REQUIRE(reuse.tree.depth[c] == before.depth[c]);
            }
        }
        for (Int v = 0; v < 90; ++v)
            if (!replaced.count(before.member[v])) REQUIRE(reuse.tree.member[v] == before.member[v]);
        for (Int r : before.roots)
            if (!replaced.count(r)) REQUIRE(std::count(reuse.tree.roots.begin(), reuse.tree.roots.end(), r) == 1);
        for (Int r : reuse.tree.roots) {
            REQUIRE(reuse.tree.nodes[r].alive);
            REQUIRE(reuse.tree.nodes[r].parent == -1);
            REQUIRE(reuse.tree.depth[r] == 0);
        }
        if (disconnected) REQUIRE(reuse.tree.roots.size() > before.roots.size());
    }
    const auto last_parent = parent, last_member = member;
    REQUIRE(order(changed) == nc);
    REQUIRE_FALSE(reuse.statistics.full_rebuild);
    REQUIRE(reuse.statistics.repartitioned_vertices == 0);
    REQUIRE(std::equal(parent.begin(), parent.begin() + nc, last_parent.begin()));
    REQUIRE(member == last_member);
}
