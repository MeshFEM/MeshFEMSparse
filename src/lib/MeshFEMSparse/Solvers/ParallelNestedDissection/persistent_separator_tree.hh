#ifndef MESHFEMSPARSE_PERSISTENT_SEPARATOR_TREE_HH
#define MESHFEMSPARSE_PERSISTENT_SEPARATOR_TREE_HH

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>
#include <MeshFEMCore/GlobalBenchmark.hh>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace MeshFEM::CholmodParallelNesdis {

// Internal forest representation. IDs remain stable outside replaced subtrees;
// tombstones are discarded at the next full rebuild. No synthetic components
// are emitted for disconnected graphs or disconnected replacement subgraphs.
// Empty separators retain their place in the hierarchy internally but are
// omitted from CHOLMOD output.
template<class Int>
struct PersistentSeparatorTree {
    struct Node {
        Int parent = -1;
        std::vector<Int> children, separator_vertices;
        bool alive = true;
    };
    std::vector<Node> nodes;
    std::vector<Int> member, depth, roots;

    static PersistentSeparatorTree from_cholmod(size_t n, Int nc,
                                               const Int *parent, const Int *membership) {
        PersistentSeparatorTree result;
        result.nodes.resize(nc);
        result.member.assign(membership, membership + n);
        for (Int c = 0; c < nc; ++c) {
            if (parent[c] < -1 || parent[c] >= nc || (parent[c] >= 0 && parent[c] <= c))
                throw std::invalid_argument("separator forest is not postordered");
            result.nodes[c].parent = parent[c];
            if (parent[c] < 0) result.roots.push_back(c);
            else result.nodes[parent[c]].children.push_back(c);
        }
        for (size_t v = 0; v < n; ++v) {
            if (membership[v] < 0 || membership[v] >= nc)
                throw std::invalid_argument("invalid separator membership");
            result.nodes[membership[v]].separator_vertices.push_back(Int(v));
        }
        result.update_depths();
        return result;
    }

    void update_depths() {
        depth.assign(nodes.size(), -1);
        auto pending = roots;
        for (Int r : roots) depth[r] = 0;
        while (!pending.empty()) {
            Int c = pending.back(); pending.pop_back();
            for (Int child : nodes[c].children) {
                depth[child] = depth[c] + 1;
                pending.push_back(child);
            }
        }
    }

    Int lca(Int a, Int b) const {
        while (depth[a] > depth[b]) a = nodes[a].parent;
        while (depth[b] > depth[a]) b = nodes[b].parent;
        while (a != b && a >= 0 && b >= 0) {
            a = nodes[a].parent;
            b = nodes[b].parent;
        }
        return a == b ? a : Int(-1);
    }

    // Repair separator violations made by edges of the new graph
    // (represented in CSR/CSC format by arrays `p`, `i`) that occur
    // at separator tree nodes above a depth `cutoff`; the goal is to
    // avoid rerunning nested dissection on the violated subtree by
    // instead merging an edge endpoint into the subtree root (LCA).
    //
    // Root depth is zero, so a `cutoff` of 0 disables promotion, a
    // positive `cutoff` permits `depth(LCA) < cutoff`, and a special
    // value of `cutoff = -1` permits repair at all levels, effectively
    // disabling subtree ND calls.
    //
    // Return the number of distinct moved vertices.
    // On successful repair, update `this->member` and `this->nodes[*].separator_vertices`.
    //
    // If the separator tree is actually a forest
    // (e.g., arising from two disconnected simulation bodies),
    // and if a new edge connects vertices belonging to different trees,
    // create a new root over the two affected trees and promote an endpoint into it.
    // Eligibility uses depths at the start of this update, before any such joins.
    // If a separator node is emptied, keep it available as an LCA internally;
    // flatten() omits it from the exported forest.
    size_t promote_violations(const Int *p, const Int *i, int cutoff) {
        if (cutoff < -1) throw std::invalid_argument("invalid temporal reuse promotion depth");
        if (cutoff == 0) return 0;
        auto eligible = [&](Int r) { return cutoff < 0 || depth[r] < cutoff; };
        // Scan the full graph collecting edges that violate the separator tree
        // at a depth that makes them eligible for promotion.
        std::vector<std::pair<Int, Int>> edges;
        std::mutex edges_mutex;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, member.size(), 512), [&](const auto &range) {
            // Only violations need synchronization. Batch them on the stack;
            // no per-task heap allocation or full-graph temporary is needed.
            std::array<std::pair<Int, Int>, 64> pending;
            size_t count = 0;
            auto flush = [&] {
                std::lock_guard<std::mutex> lock(edges_mutex);
                edges.insert(edges.end(), pending.begin(), pending.begin() + count);
                count = 0;
            };
            for (size_t v = range.begin(); v < range.end(); ++v) {
                for (Int k = p[v]; k < p[v + 1]; ++k) {
                    Int u = i[k];
                    if (u >= Int(v)) continue;
                    Int a = member[v], b = member[u], r = lca(a, b);
                    if (r == a || r == b) continue;
                    if (r < 0 || eligible(r)) {
                        // Keep the adjacency offset to recover the original scan
                        // order even when the input columns are not sorted.
                        pending[count++] = {Int(v), k};
                        if (count == pending.size()) flush();
                    }
                }
            }
            if (count) flush();
        });
        if (edges.empty()) return 0;
        std::sort(edges.begin(), edges.end());
        std::vector<size_t> counts(member.size(), 0);
        for (auto [v, k] : edges) { ++counts[v]; ++counts[i[k]]; }
        auto proposed = member;
        for (auto [v, k] : edges) {
            Int u = i[k];
            // Confirm that, after previous promotions, the violation still exists.
            // We needn't re-check eligibility: promotion moves vertices upward
            // within the existing tree. Cross-root joins may increase depths,
            // but eligibility was determined by the initial scan, before those joins.
            Int a = proposed[v], b = proposed[u], r = lca(a, b);
            if (r == a || r == b) continue;
            if (r < 0) {
                while (nodes[a].parent >= 0) a = nodes[a].parent;
                while (nodes[b].parent >= 0) b = nodes[b].parent;
                if (nodes.size() >= size_t(std::numeric_limits<Int>::max()))
                    throw std::overflow_error("persistent separator tree exceeds index range");
                r = Int(nodes.size());
                auto first = std::find_if(roots.begin(), roots.end(), [&](Int c) { return c == a || c == b; });
                Int other = *first == a ? b : a;
                Node root;
                root.children = {*first, other};
                nodes.push_back(std::move(root));
                nodes[a].parent = nodes[b].parent = r;
                // Preserve the position of the earlier root and all unrelated roots.
                *first = r;
                roots.erase(std::find(first + 1, roots.end(), other));
                update_depths();
            }
            // Edges are visited with v > u; ties favor the smaller vertex ID `u`
            proposed[counts[v] > counts[u] ? v : u] = r;
        }

        // Determine whether each separator's vertex membership has changed and, if so, its new size.
        // This information is encoded in a single `size_t` that is 0 for
        // no change and `new_size + 1`; note that `new_size` can be the same as
        // the old size even when there are changes (if a vertex is added and removed),
        // and this is properly handled.
        size_t moved = 0;
        std::vector<size_t> sizes(nodes.size(), 0);
        for (size_t v = 0; v < member.size(); ++v) {
            Int from = member[v], to = proposed[v];
            if (from == to) continue;
            ++moved;
            if (!sizes[from]) sizes[from] = nodes[from].separator_vertices.size() + 1;
            if (!sizes[to])   sizes[  to] = nodes[  to].separator_vertices.size() + 1;
            --sizes[from];
            ++sizes[to];
        }
        // Update the (sorted) separator vertex lists without rebuilding from scratch.
        // Reserve all affected lists before clearing any, so allocation failure
        // preserves all memberships (exception safety).
        for (size_t c = 0; c < nodes.size(); ++c) if (sizes[c]) nodes[c].separator_vertices.reserve(sizes[c] - 1);
        for (size_t c = 0; c < nodes.size(); ++c) if (sizes[c]) nodes[c].separator_vertices.clear();
        for (size_t v = 0; v < member.size(); ++v) if (sizes[proposed[v]]) nodes[proposed[v]].separator_vertices.push_back(Int(v));
        member.swap(proposed);

        return moved;
    }

    // Scan one triangle of a symmetric, packed CSC graph. An edge between
    // distinct roots, or a violation at the sole root, requests a full rebuild.
    // A dirty root in a larger forest can be repaired independently.
    struct DirtySubtrees {
        bool full_rebuild = false;
        std::vector<Int> roots;
    };
    DirtySubtrees dirty_subtrees(const Int *p, const Int *i) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("DirtySubtrees.dirty_subtrees");
        // Multiple edges can mark the same node concurrently. Flags only move
        // from false to true, and parallel_for joins before we consume them.
        std::vector<std::atomic<bool>> dirty(nodes.size());
        for (auto &flag : dirty) flag.store(false, std::memory_order_relaxed);
        std::atomic<bool> full_rebuild{false};
        {
            // BENCHMARK_SCOPED_TIMER_SECTION marking_timer("mark dirty nodes");
            tbb::parallel_for(tbb::blocked_range<size_t>(0, member.size(), 512), [&](const auto &range) {
                for (size_t v = range.begin(); v < range.end(); ++v) {
                    for (Int k = p[v]; k < p[v + 1]; ++k) {
                        if (i[k] >= Int(v)) continue;
                        Int a = member[v], b = member[i[k]];
                        Int r = lca(a, b);
                        if (r == a || r == b) continue; // Separator `r` is not violated if one component is descendent of the other...
                        if (r < 0 || (nodes[r].parent < 0 && roots.size() == 1)) {
                            full_rebuild.store(true, std::memory_order_relaxed);
                            return;
                        }
                        // Avoid repeated writes to a shared cache line once a
                        // node is marked. Concurrent first stores are harmless.
                        if (!dirty[r].load(std::memory_order_relaxed))
                            dirty[r].store(true, std::memory_order_relaxed);
                    }
                }
            });
        }
        if (full_rebuild.load(std::memory_order_relaxed)) return {true, {}};
        // BENCHMARK_SCOPED_TIMER_SECTION pruning_timer("prune dirty descendants");
        DirtySubtrees result;
        for (size_t c = 0; c < nodes.size(); ++c) if (dirty[c].load(std::memory_order_relaxed)) {
            Int pnode = nodes[c].parent;
            while (pnode >= 0 && !dirty[pnode].load(std::memory_order_relaxed)) pnode = nodes[pnode].parent;
            if (pnode < 0) result.roots.push_back(Int(c));
        }
        return result;
    }

    std::vector<Int> subtree_nodes(Int root) const {
        std::vector<Int> result{root};
        for (size_t k = 0; k < result.size(); ++k) {
            const auto &children = nodes[result[k]].children;
            result.insert(result.end(), children.begin(), children.end());
        }
        return result;
    }

    std::vector<Int> subtree_vertices(Int root) const {
        const auto components = subtree_nodes(root);
        size_t count = 0;
        for (Int c : components) count += nodes[c].separator_vertices.size();
        std::vector<Int> result;
        result.reserve(count);
        for (Int c : components) {
            const auto &vertices = nodes[c].separator_vertices;
            result.insert(result.end(), vertices.begin(), vertices.end());
        }
        // Keep tree traversal order. The repair path uses the batched gather
        // below when it needs sorted global IDs and an inverse vertex map.
        return result;
    }

    struct SubtreeRegionLabels {
        std::vector<Int> region_for_node;
        std::vector<std::vector<Int>> components;
        size_t vertex_count = 0;
    };

    // Prepare disjoint dirty regions without visiting their vertices. Reserving
    // from component sizes avoids reallocations during the subsequent gather.
    SubtreeRegionLabels label_subtree_regions(const std::vector<Int> &dirty_roots,
                                              std::vector<std::vector<Int>> &regions) const {
        SubtreeRegionLabels labels;
        labels.region_for_node.assign(nodes.size(), -1);
        labels.components.resize(dirty_roots.size());
        regions.clear();
        regions.resize(dirty_roots.size());
        for (size_t r = 0; r < dirty_roots.size(); ++r) {
            size_t count = 0;
            labels.components[r] = subtree_nodes(dirty_roots[r]);
            for (Int c : labels.components[r]) {
                labels.region_for_node[c] = Int(r);
                count += nodes[c].separator_vertices.size();
            }
            labels.vertex_count += count;
            regions[r].reserve(count);
        }
        return labels;
    }

    // Preserve increasing global IDs without globally sorting collected lists.
    // For small repairs, merge sorted per-node lists; for large repairs, gather
    // with an ordered vertex scan. Both keep local numbering, induced-column
    // sortedness, and downstream graph traversal identical to the sorted path.
    void gather_subtree_regions(const SubtreeRegionLabels &labels,
                                std::vector<std::vector<Int>> &regions,
                                std::vector<Int> &local_index) const {
        // A full membership scan can dominate a tiny repair. The production
        // path maintains sorted per-node lists (from_cholmod and replacement
        // with sorted global IDs). Fall back to the scan if this invariant was
        // not preserved by an external/manual replacement.
        bool merge = labels.vertex_count <= member.size() / 16;
        if (merge) for (const auto &components : labels.components) {
            for (Int c : components) {
                const auto &vertices = nodes[c].separator_vertices;
                if (!std::is_sorted(vertices.begin(), vertices.end())) { merge = false; break; }
            }
            if (!merge) break;
        }
        if (merge) {
            struct Cursor { Int node; size_t offset; };
            std::vector<std::vector<Cursor>> heaps(regions.size());
            for (size_t r = 0; r < regions.size(); ++r) {
                heaps[r].reserve(labels.components[r].size());
                for (Int c : labels.components[r])
                    if (!nodes[c].separator_vertices.empty()) heaps[r].push_back({c, 0});
            }
            auto greater = [&](const Cursor &a, const Cursor &b) {
                return nodes[a.node].separator_vertices[a.offset] > nodes[b.node].separator_vertices[b.offset];
            };
            local_index.assign(member.size(), -1);
            for (size_t r = 0; r < regions.size(); ++r) {
                auto &heap = heaps[r];
                auto &region = regions[r];
                std::make_heap(heap.begin(), heap.end(), greater);
                while (!heap.empty()) {
                    std::pop_heap(heap.begin(), heap.end(), greater);
                    auto cursor = heap.back(); heap.pop_back();
                    const auto &vertices = nodes[cursor.node].separator_vertices;
                    Int v = vertices[cursor.offset++];
                    local_index[v] = Int(region.size());
                    region.push_back(v);
                    if (cursor.offset < vertices.size()) {
                        heap.push_back(cursor);
                        std::push_heap(heap.begin(), heap.end(), greater);
                    }
                }
            }
            return;
        }

        // Fuse collection with the dense inverse-map initialization so this
        // path needs no additional full-size fill of local_index.
        local_index.clear();
        local_index.reserve(member.size());
        for (size_t v = 0; v < member.size(); ++v) {
            Int r = labels.region_for_node[member[v]];
            if (r < 0) local_index.push_back(-1);
            else {
                auto &region = regions[r];
                local_index.push_back(Int(region.size()));
                region.push_back(Int(v));
            }
        }
    }

    // Replacement membership is indexed locally by vertices. The caller batches
    // disjoint replacements and updates depths once after all splices.
    void replace(Int root, const PersistentSeparatorTree &replacement,
                 const std::vector<Int> &vertices) {
        if (replacement.nodes.size() > size_t(std::numeric_limits<Int>::max()) - nodes.size())
            throw std::overflow_error("persistent separator tree exceeds index range");
        Int parent = nodes[root].parent, offset = Int(nodes.size());
        for (Int c : subtree_nodes(root)) {
            nodes[c].alive = false;
            nodes[c].children.clear();
            nodes[c].separator_vertices.clear();
        }
        for (const auto &source : replacement.nodes) {
            Node node;
            node.parent = source.parent < 0 ? parent : offset + source.parent;
            for (Int child : source.children) node.children.push_back(offset + child);
            for (Int v : source.separator_vertices) node.separator_vertices.push_back(vertices[v]);
            nodes.push_back(std::move(node));
        }
        auto &siblings = parent < 0 ? roots : nodes[parent].children;
        auto position = siblings.erase(std::find(siblings.begin(), siblings.end(), root));
        std::vector<Int> new_roots;
        for (Int r : replacement.roots) new_roots.push_back(offset + r);
        siblings.insert(position, new_roots.begin(), new_roots.end());
        for (size_t v = 0; v < vertices.size(); ++v)
            member[vertices[v]] = offset + replacement.member[v];
    }

    Int flatten(Int *parent, Int *membership) const {
        std::vector<Int> postorder, pending = roots, dense(nodes.size(), -1);
        while (!pending.empty()) {
            Int c = pending.back(); pending.pop_back();
            postorder.push_back(c);
            pending.insert(pending.end(), nodes[c].children.begin(), nodes[c].children.end());
        }
        std::reverse(postorder.begin(), postorder.end());
        Int nc = 0;
        for (Int c : postorder)
            if (!nodes[c].separator_vertices.empty()) dense[c] = nc++;
        // Map empty nodes to their nearest nonempty ancestor, visiting parents
        // first. This also keeps the exported component count bounded by n.
        for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
            Int c = *it, p = nodes[c].parent;
            if (nodes[c].separator_vertices.empty()) dense[c] = p < 0 ? -1 : dense[p];
        }
        for (Int c : postorder) if (!nodes[c].separator_vertices.empty()) {
            Int p = nodes[c].parent;
            parent[dense[c]] = p < 0 ? -1 : dense[p];
        }
        for (size_t v = 0; v < member.size(); ++v) membership[v] = dense[member[v]];
        return nc;
    }
};

} // namespace MeshFEM::CholmodParallelNesdis
#endif
