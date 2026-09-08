#include "hierarchical_camd.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <new>
#include <numeric>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <tbb/parallel_for.h>

namespace MeshFEM::CholmodParallelNesdis {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

template<class Int>
struct CAMDWorkspace {
    cholmod_common common;
    explicit CAMDWorkspace(const cholmod_common &settings) {
        if constexpr (std::is_same_v<Int, int32_t>) cholmod_start(&common);
        else                                       cholmod_l_start(&common);
        // Start with independent workspace and allocation accounting. Only copy
        // CAMD settings, never the caller's workspace pointers or error handler.
        common.current = settings.current;
        if (settings.current >= 0 && settings.current < CHOLMOD_MAXMETHODS)
            common.method[settings.current] = settings.method[settings.current];
    }
    CAMDWorkspace(const CAMDWorkspace &) = delete;
    CAMDWorkspace &operator=(const CAMDWorkspace &) = delete;
    ~CAMDWorkspace() {
        if constexpr (std::is_same_v<Int, int32_t>) cholmod_finish(&common);
        else                                       cholmod_l_finish(&common);
    }
};

template<class Int>
Int checked_index(size_t n) {
    if (n > (size_t)std::numeric_limits<Int>::max())
        throw std::length_error("hierarchical CAMD graph exceeds index range");
    return (Int)n;
}

template<class Int>
std::vector<Int> order_graph(std::vector<Int> &p, std::vector<Int> &i,
                            const std::vector<Int> &vertices, const Int *member,
                            const cholmod_common &settings, CAMDTaskStatistics &stats) {
    stats.vertices = vertices.size();
    stats.edges = i.size() / 2;
    std::vector<Int> constraints, labels, permutation(vertices.size());
    constraints.reserve(vertices.size());
    for (Int v : vertices) constraints.push_back(member[v]);
    labels = constraints;
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    stats.constraints = labels.size();
    for (Int &c : constraints)
        c = (Int)(std::lower_bound(labels.begin(), labels.end(), c) - labels.begin());
    if (vertices.empty()) return permutation;

    cholmod_sparse graph{};
    graph.nrow = graph.ncol = vertices.size();
    graph.nzmax = i.size();
    graph.p = p.data();
    // CHOLMOD expects a nonnull i pointer even for an edgeless graph.
    Int dummy = 0;
    graph.i = i.empty() ? &dummy : i.data();
    graph.stype = -1; // cholmod_camd must order this graph, not graph * graph'.
    graph.itype = std::is_same_v<Int, int32_t> ? CHOLMOD_INT : CHOLMOD_LONG;
    graph.xtype = CHOLMOD_PATTERN;
    graph.packed = 1;
    CAMDWorkspace<Int> workspace(settings);
    auto start = Clock::now();
    int ok;
    if constexpr (std::is_same_v<Int, int32_t>)
        ok = cholmod_camd(&graph, nullptr, 0, constraints.data(), permutation.data(), &workspace.common);
    else
        ok = cholmod_l_camd(&graph, nullptr, 0, constraints.data(), permutation.data(), &workspace.common);
    stats.camd_seconds = seconds(start);
    if (!ok) {
        if (workspace.common.status == CHOLMOD_OUT_OF_MEMORY) throw std::bad_alloc();
        throw std::runtime_error("hierarchical CAMD local ordering failed");
    }
    Int previous = -1;
    for (Int &v : permutation) {
        if (v < 0 || (size_t)v >= vertices.size() || constraints[v] < previous)
            throw std::runtime_error("hierarchical CAMD violated local constraints");
        previous = constraints[v];
        v = vertices[v];
    }
    return permutation;
}

template<class Int>
void verify_result(const cholmod_sparse &graph, const Int *parent, Int nc,
                   const Int *member, const HierarchicalCAMDResult<Int> &result, bool verify_upper) {
    size_t n = graph.nrow;
    if (result.permutation.size() != n) throw std::runtime_error("CAMD permutation size mismatch");
    std::vector<bool> seen(n, false);
    std::vector<size_t> first(nc, n), last(nc, 0);
    for (size_t k = 0; k < n; ++k) {
        Int v = result.permutation[k];
        if (v < 0 || (size_t)v >= n || seen[v]) throw std::runtime_error("invalid CAMD permutation");
        seen[v] = true;
        first[member[v]] = std::min(first[member[v]], k);
        last[member[v]] = std::max(last[member[v]], k);
    }
    // Siblings may commute, but every descendant must precede its ancestors.
    for (Int c = 0; c < nc; ++c)
        for (Int ancestor = parent[c]; ancestor >= 0; ancestor = parent[ancestor])
            if (first[c] < n && first[ancestor] < n && last[c] >= first[ancestor])
                throw std::runtime_error("CAMD violated separator ancestry");
    if (!verify_upper || n > 512) return;
    const Int *p = static_cast<const Int *>(graph.p), *i = static_cast<const Int *>(graph.i);
    std::vector<std::set<Int>> adjacency(n);
    for (size_t v = 0; v < n; ++v)
        for (Int k = p[v]; k < p[v + 1]; ++k) if (i[k] != (Int)v) adjacency[v].insert(i[k]);
    for (size_t k = 0; k < n - result.upper_vertices.size(); ++k) {
        Int v = result.permutation[k];
        const auto neighbors = adjacency[v];
        for (Int a : neighbors) {
            adjacency[a].erase(v);
            for (Int b : neighbors) if (a != b) adjacency[a].insert(b);
        }
        adjacency[v].clear();
    }
    std::vector<std::set<Int>> upper(result.upper_vertices.size() + result.quotient_elements);
    for (auto [a, b] : result.upper_edges) { upper[a].insert(b); upper[b].insert(a); }
    // Independently eliminate the actual synthetic-star input, then compare it
    // against elimination in the original graph, including disconnected tasks.
    for (size_t v = result.upper_vertices.size(); v < upper.size(); ++v) {
        const auto neighbors = upper[v];
        for (Int a : neighbors) {
            upper[a].erase((Int)v);
            for (Int b : neighbors) if (a != b) upper[a].insert(b);
        }
        upper[v].clear();
    }
    for (size_t a = 0; a < result.upper_vertices.size(); ++a)
        for (size_t b = a + 1; b < result.upper_vertices.size(); ++b)
            if (adjacency[result.upper_vertices[a]].count(result.upper_vertices[b]) != upper[a].count((Int)b))
                throw std::runtime_error("incorrect upper structural fill graph");
}
} // namespace

template<class Int>
void trace_camd_tree(size_t n, const Int *parent, Int nc, const Int *member) {
    std::vector<size_t> sizes(nc, 0), subtree;
    std::vector<int> depth(nc, 0);
    for (size_t v = 0; v < n; ++v) ++sizes[member[v]];
    subtree = sizes;
    for (Int c = 0; c < nc; ++c) if (parent[c] >= 0) subtree[parent[c]] += subtree[c];
    for (Int c = nc; c-- > 0;) if (parent[c] >= 0) depth[c] = depth[parent[c]] + 1;
    for (Int c = 0; c < nc; ++c)
        std::fprintf(stderr, "CAMD_TREE {\"component\":%lld,\"parent\":%lld,\"depth\":%d,\"vertices\":%zu,\"subtree_vertices\":%zu}\n",
                     (long long)c, (long long)parent[c], depth[c], sizes[c], subtree[c]);
}

template<class Int>
HierarchicalCAMDResult<Int> hierarchical_camd(
    const cholmod_sparse &graph, const Int *parent, Int nc,
    const Int *member, const cholmod_common &settings, const HierarchicalCAMDOptions &options) {
    auto start = Clock::now();
    HierarchicalCAMDResult<Int> result;
    size_t n = graph.nrow;
    checked_index<Int>(n);
    if (!graph.packed || graph.stype != 0 || graph.ncol != n || options.cut_depth < 0 || nc < 0)
        throw std::invalid_argument("invalid hierarchical CAMD input");
    const Int *p = static_cast<const Int *>(graph.p), *i = static_cast<const Int *>(graph.i);
    std::vector<int> depth(nc, 0);
    std::vector<Int> root(nc, -1), task_id(nc, -1);
    // Parents follow children in CHOLMOD's postorder. This recovers all task
    // membership in O(n + number of components), without full-graph scans/task.
    int max_depth = 0;
    for (Int c = nc; c-- > 0;) {
        if (parent[c] < -1 || parent[c] >= nc || (parent[c] >= 0 && parent[c] <= c))
            throw std::invalid_argument("separator forest is not postordered");
        if (parent[c] >= 0) depth[c] = depth[parent[c]] + 1;
        max_depth = std::max(max_depth, depth[c]);
    }
    int cut_depth = std::min(options.cut_depth, max_depth);
    for (Int c = nc; c-- > 0;) {
        if (depth[c] == cut_depth) root[c] = c;
        else if (depth[c] > cut_depth) root[c] = root[parent[c]];
    }
    size_t nt = 0;
    for (Int c = 0; c < nc; ++c) if (root[c] == c) task_id[c] = checked_index<Int>(nt++);
    std::vector<std::vector<Int>> vertices(nt);
    std::vector<Int> owner(n), local(n);
    for (size_t v = 0; v < n; ++v) {
        Int c = member[v];
        if (c < 0 || c >= nc) throw std::invalid_argument("invalid separator membership");
        owner[v] = root[c] < 0 ? -1 : task_id[root[c]];
        auto &list = owner[v] < 0 ? result.upper_vertices : vertices[owner[v]];
        local[v] = checked_index<Int>(list.size());
        list.push_back((Int)v);
    }
    result.decomposition_seconds = seconds(start);
    std::vector<std::vector<Int>> permutations(nt);
    std::vector<std::vector<std::vector<Int>>> boundaries(nt);
    result.tasks.resize(nt);
    start = Clock::now();
    tbb::parallel_for(size_t(0), nt, [&](size_t task) {
        const auto &owned = vertices[task];
        std::vector<Int> ap(owned.size() + 1, 0), ai;
        for (size_t v = 0; v < owned.size(); ++v) {
            Int g = owned[v];
            for (Int k = p[g]; k < p[g + 1]; ++k) {
                Int w = i[k];
                if (w == g) continue;
                if (owner[w] == (Int)task) ai.push_back(local[w]);
                else if (owner[w] >= 0) throw std::invalid_argument("edge crosses coarse ND subtrees");
            }
            ap[v + 1] = checked_index<Int>(ai.size());
        }
        auto &stats = result.tasks[task];
        permutations[task] = order_graph(ap, ai, owned, member, settings, stats);
        if (options.skip_upper) return;
        std::vector<bool> seen(owned.size(), false);
        std::vector<Int> stack, boundary;
        for (size_t seed = 0; seed < owned.size(); ++seed) {
            if (seen[seed]) continue;
            seen[seed] = true;
            stack.push_back((Int)seed);
            boundary.clear();
            while (!stack.empty()) {
                Int v = stack.back(); stack.pop_back();
                Int g = owned[v];
                for (Int k = p[g]; k < p[g + 1]; ++k) {
                    Int w = i[k];
                    if (owner[w] < 0) boundary.push_back(local[w]);
                    else if (!seen[local[w]]) {
                        seen[local[w]] = true;
                        stack.push_back(local[w]);
                    }
                }
            }
            std::sort(boundary.begin(), boundary.end());
            boundary.erase(std::unique(boundary.begin(), boundary.end()), boundary.end());
            ++stats.boundary_components;
            stats.max_boundary = std::max(stats.max_boundary, boundary.size());
            stats.sum_boundary += boundary.size();
            stats.sum_boundary_squared += double(boundary.size()) * double(boundary.size());
            if (boundary.size() >= 2) boundaries[task].push_back(boundary);
        }
    });
    result.local_seconds = seconds(start);
    auto &edges = result.upper_edges;
    std::vector<Int> upper_perm;
    if (options.skip_upper) {
        // Component IDs are postordered. Stable counting sort keeps every
        // separator contiguous, with original vertex order inside each group.
        std::vector<size_t> offsets((size_t)nc + 1, 0);
        for (Int v : result.upper_vertices) ++offsets[member[v] + 1];
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        upper_perm.resize(result.upper_vertices.size());
        for (Int v : result.upper_vertices) upper_perm[offsets[member[v]]++] = v;
    }
    else {
    start = Clock::now();
    for (size_t v = 0; v < result.upper_vertices.size(); ++v) {
        Int g = result.upper_vertices[v];
        for (Int k = p[g]; k < p[g + 1]; ++k)
            if (owner[i[k]] < 0 && (Int)v < local[i[k]]) edges.emplace_back((Int)v, local[i[k]]);
    }
    result.original_upper_edges = edges.size();
    size_t upper_n = result.upper_vertices.size();
    // A star eliminated before all live variables creates exactly clique(B),
    // but CAMD stores the result as an element, never as O(|B|^2) edges.
    // Deduplicated boundaries and distinct synthetic vertices introduce no
    // duplicate edges, so no global edge sort is needed.
    for (const auto &task : boundaries) for (const auto &b : task) {
        Int element = checked_index<Int>(upper_n++);
        for (Int v : b) edges.emplace_back(v, element);
        ++result.quotient_elements;
    }
    checked_index<Int>(upper_n);
    if (edges.size() > (size_t)std::numeric_limits<Int>::max() / 2)
        throw std::length_error("upper adjacency exceeds index range");
    std::vector<Int> up(upper_n + 1, 0), ui(2 * edges.size());
    for (auto [a, b] : edges) { ++up[a + 1]; ++up[b + 1]; }
    std::partial_sum(up.begin(), up.end(), up.begin());
    auto next = up;
    for (auto [a, b] : edges) { ui[next[a]++] = b; ui[next[b]++] = a; }
    result.upper_graph_seconds = seconds(start);
    CAMDTaskStatistics upper_stats;
    start = Clock::now();
    if (result.quotient_elements) {
        std::vector<Int> upper_member(upper_n, 0), identity(upper_n);
        std::iota(identity.begin(), identity.end(), Int(0));
        for (size_t v = 0; v < result.upper_vertices.size(); ++v)
            upper_member[v] = member[result.upper_vertices[v]] + 1;
        CAMDWorkspace<Int> upper_settings(settings);
        // Negative prune_dense still prunes completely dense vertices in CAMD.
        // alpha=sqrt(n) sets its threshold to n (or n-1 under rounding), so even
        // a single star touching every live vertex is eliminated normally.
        if (upper_settings.common.current < 0 || upper_settings.common.current >= CHOLMOD_MAXMETHODS)
            upper_settings.common.current = 0;
        upper_settings.common.method[upper_settings.common.current].prune_dense = std::sqrt(double(upper_n));
        auto augmented_perm = order_graph(up, ui, identity, upper_member.data(), upper_settings.common, upper_stats);
        for (size_t k = 0; k < augmented_perm.size(); ++k) {
            Int v = augmented_perm[k];
            if (k < result.quotient_elements) {
                if ((size_t)v < result.upper_vertices.size())
                    throw std::runtime_error("CAMD did not eliminate synthetic elements first");
            }
            else {
                if ((size_t)v >= result.upper_vertices.size())
                    throw std::runtime_error("CAMD returned a synthetic vertex in the live suffix");
                upper_perm.push_back(result.upper_vertices[v]);
            }
        }
    }
    else upper_perm = order_graph(up, ui, result.upper_vertices, member, settings, upper_stats);
    result.upper_camd_seconds = seconds(start);
    }
    result.permutation.reserve(n);
    for (const auto &perm : permutations) result.permutation.insert(result.permutation.end(), perm.begin(), perm.end());
    result.permutation.insert(result.permutation.end(), upper_perm.begin(), upper_perm.end());
    if (options.verify) verify_result(graph, parent, nc, member, result, !options.skip_upper);
    if (options.trace) {
        for (size_t t = 0; t < nt; ++t) {
            const auto &s = result.tasks[t];
            std::fprintf(stderr, "CAMD_TASK {\"task\":%zu,\"vertices\":%zu,\"edges\":%zu,\"constraints\":%zu,\"boundary_components\":%zu,\"max_boundary\":%zu,\"sum_boundary\":%zu,\"sum_boundary_squared\":%.0f,\"camd_seconds\":%.9g}\n",
                         t, s.vertices, s.edges, s.constraints, s.boundary_components, s.max_boundary, s.sum_boundary, s.sum_boundary_squared, s.camd_seconds);
        }
        std::fprintf(stderr, "CAMD_HIERARCHICAL {\"cut_depth\":%d,\"skip_upper\":%s,\"tasks\":%zu,\"upper_vertices\":%zu,\"original_upper_edges\":%zu,\"upper_input_edges\":%zu,\"quotient_elements\":%zu,\"decomposition_seconds\":%.9g,\"local_seconds\":%.9g,\"upper_graph_seconds\":%.9g,\"upper_camd_seconds\":%.9g}\n",
                     cut_depth, options.skip_upper ? "true" : "false", nt, result.upper_vertices.size(), result.original_upper_edges,
                     edges.size(), result.quotient_elements, result.decomposition_seconds,
                     result.local_seconds, result.upper_graph_seconds, result.upper_camd_seconds);
    }
    return result;
}

template HierarchicalCAMDResult<int32_t> hierarchical_camd(const cholmod_sparse &, const int32_t *, int32_t, const int32_t *, const cholmod_common &, const HierarchicalCAMDOptions &);
template HierarchicalCAMDResult<int64_t> hierarchical_camd(const cholmod_sparse &, const int64_t *, int64_t, const int64_t *, const cholmod_common &, const HierarchicalCAMDOptions &);
template void trace_camd_tree(size_t, const int32_t *, int32_t, const int32_t *);
template void trace_camd_tree(size_t, const int64_t *, int64_t, const int64_t *);
} // namespace MeshFEM::CholmodParallelNesdis
