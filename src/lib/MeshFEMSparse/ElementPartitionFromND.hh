////////////////////////////////////////////////////////////////////////////////
// ElementPartitionFromND.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Reverse-engineer an element partitioning from a nested dissection separator
//  tree generated from the element set's block sparsity pattern.
//  This element partitioning can be used to accelerate spin-lock-based assembly
//  by parallelizing across partitions and only using locks for the smaller set
//  of separator vertices that receive contributions from multiple partitions.
//
//  Note that the ND tree used here *could* differ from the one used for matrix
//  factorization. The only requirement is that it respects the element stencil
//  set. It may, e.g., be beneficial to build an element partition for one
//  large static element set even in a larger problem featuring additional
//  stencils that are dynamically added and removed.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  09/17/2026 14:57:45
*///////////////////////////////////////////////////////////////////////////////
#ifndef ELEMENTPARTITIONFROMND_HH
#define ELEMENTPARTITIONFROMND_HH

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <type_traits>

namespace MeshFEM {

template<typename Index = int>
struct ElementPartitionFromND {
    static_assert(std::is_integral_v<Index> && std::is_signed_v<Index>,
                  "ElementPartitionFromND requires a signed integer index type (ND uses -1 sentinels)");
    using index_type = Index;
    // `partitionOffsets` is a packed-list representation of the element
    // partitions, where partition `p` contains the slice of elements
    // `partitionOffsets[p]:partitionOffsets[p + 1]`.
    //
    // However, the elements of a partition generally do not appear contiguously
    // like this in the original mesh. Their original, potentially scattered
    // indices are held in `elementOrder`. In other words, the original
    // indices of elements in partition `p` are actually
    // `elementOrder[partitionOffsets[p]:partitionOffsets[p + 1]]`.
    //
    // An empty `elementOrder` is shorthand for the identity permutation vector
    // `0..{numElements-1}`, and it is highly recommended to apply the
    // `elementOrder` permutation to the mesh and then empty it here.
    std::vector<Index> partitionOffsets, elementOrder;
    std::vector<uint8_t> variableNeedsLock;
    size_t numBlockVars() const { return variableNeedsLock.size(); }
    size_t numElements() const { return partitionOffsets.empty() ? 0 : size_t(partitionOffsets.back()); }
    Index elementIndex(Index i) const { return elementOrder.empty() ? i : elementOrder[i]; }
    Index numPartitions() const { return partitionOffsets.empty() ? 0 : Index(partitionOffsets.size() - 1); }

    // Construct partition from an `element(ei)` element stencil getter and a
    // nested dissection separator tree represented by `member` and `parent`.
    // The separator tree is compressed to depth `splitDepth`, with only tree
    // nodes above that depth retained as actual separators.
    //
    // Within each partition, `elementOrder` lists original element IDs in
    // lexicographic order of their sorted block-variable indices.
    template<class Element>
    ElementPartitionFromND(size_t ne, const Element &element,
                           const std::vector<Index> &parent, const std::vector<Index> &member,
                           int splitDepth = 5) {
        const size_t numBlockVars = member.size();
        if (splitDepth < 0) throw std::invalid_argument("Invalid splitDepth");
        if (numBlockVars > size_t(std::numeric_limits<Index>::max()) || ne > size_t(std::numeric_limits<Index>::max()) ||
            parent.size() > size_t(std::numeric_limits<Index>::max()))
            throw std::invalid_argument("ND partition exceeds range of the partition index type");
        const Index nc = Index(parent.size());
        std::vector<std::vector<Index>> children(nc);
        std::vector<Index> preorder;
        std::vector<Index> componentDepth(nc, -1);
        // Build up a pre-order (BFS) and determine node depths in the separator tree
        for (Index c = 0; c < nc; ++c) {
            Index p = parent[c];
            if (p < -1 || p >= nc || p == c) throw std::invalid_argument("Invalid CParent index");
            if (p < 0) { preorder.push_back(c); componentDepth[c] = 0; }
            else children[p].push_back(c);
        }
        for (size_t h = 0; h < preorder.size(); ++h) for (Index c : children[preorder[h]]) {
            componentDepth[c] = componentDepth[preorder[h]] + 1;
            preorder.push_back(c);
        }
        if (preorder.size() != parent.size()) throw std::invalid_argument("CParent contains a cycle");

        // Collapse the separator subtrees below `splitDepth` into coarser leaf
        // components ("regions"); the components corresponding to internal tree nodes
        // are the retained separators (tagged region -1).
        std::vector<Index> region(nc, -1);
        for (Index c : preorder) {
            Index p = parent[c];
            if (p >= 0 && region[p] >= 0) region[c] = region[p];
            else if (componentDepth[c] >= splitDepth || children[c].empty()) region[c] = c;
        }
        std::vector<Index> blockVarRegion(numBlockVars); // Transfer region markings from components onto the underlying variables.
        for (size_t v = 0; v < numBlockVars; ++v) {
            Index c = member[v];
            if (c < 0 || c >= nc) throw std::invalid_argument("Invalid CMember index");
            blockVarRegion[v] = region[c];
        }

        // Generate the element partition by assigning elements to a partition
        // generated for the unique non-separator region its nodes are marked
        // with. If more than one region label is encountered within an element,
        // an error is thrown (the separator tree is violated by the element
        // and must have been constructed from a different element set/the
        // element was not treated as a clique when generating the sparsity
        // pattern). If the element stencil instead references only
        // intra-separator variables (region -1), it is left unpartitioned in
        // this pass.
        Index np = 0;
        std::vector<Index> partitionForRegion(nc, -1);
        std::vector<Index> elementPartition(ne, -1);
        for (size_t e = 0; e < ne; ++e) {
            Index r = -1;
            const auto &nodes = element(e);
            for (decltype(nodes.size()) j = 0; j < nodes.size(); ++j) {
                auto v = nodes[j];
                if (size_t(v) >= size_t(numBlockVars)) throw std::invalid_argument("Invalid element block-variable index");
                Index label = blockVarRegion[v];
                if (label < 0) continue;
                if (r >= 0 && r != label)
                    throw std::invalid_argument("Element has multiple non-separator ND labels: ordering graph does not cover element stencil");
                r = label;
            }
            if (r < 0) continue;
            Index &p = partitionForRegion[r];
            if (p < 0) p = np++; // new partition!
            elementPartition[e] = p;
        }

        // Separator-only stencils can be placed arbitrarily.
        // Distribute them round-robin to the existing partitions,
        // creating a single new partition if none exists.
        if (ne && np == 0) np = 1;
        {
            Index nextPartition = 0;
            for (Index &p : elementPartition) if (p < 0) {
                p = nextPartition;
                nextPartition = (nextPartition + 1) % np;
            }
        }

        // Construct a packed list representation of the partitions.
        // in (partitionOffsets, elementOrder)
        {
            partitionOffsets.assign(size_t(np) + 1, 0);
            for (Index p : elementPartition) ++partitionOffsets[p + 1];
            std::partial_sum(partitionOffsets.begin(), partitionOffsets.end(), partitionOffsets.begin());
            elementOrder.resize(ne);
            auto bucketBack = partitionOffsets;
            for (size_t e = 0; e < ne; ++e) elementOrder[bucketBack[elementPartition[e]]++] = Index(e);
        }
        // Determine which variables need to be protected by locks.
        // We already know that this can be restricted to separator variables,
        // but not all separator variables necessarily need them.
        // Really only variables actually shared by multiple partitions need locks.
        variableNeedsLock.assign(numBlockVars, false);
        std::vector<Index> owningPartition(numBlockVars, -1);
        for (size_t e = 0; e < ne; ++e) {
            const auto &nodes = element(e);
            Index p = elementPartition[e];
            for (auto v : nodes) {
                if (owningPartition[v] < 0) owningPartition[v] = p;
                else if (owningPartition[v] != p) variableNeedsLock[v] = true;
            }
        }

        // Sort elements within each partition lexicographically by their
        // stencil indices.
        std::vector<size_t> offsets(ne + 1, 0);
        std::vector<Index> keys;
        for (size_t e = 0; e < ne; ++e) {
            const auto &nodes = element(e);
            for (decltype(nodes.size()) j = 0; j < nodes.size(); ++j) keys.push_back(nodes[j]);
            offsets[e + 1] = keys.size();
            std::sort(keys.begin() + offsets[e], keys.end());
        }
        auto less = [&](Index a, Index b) {
            auto ab = keys.begin() + offsets[a], ae = keys.begin() + offsets[a + 1];
            auto bb = keys.begin() + offsets[b], be = keys.begin() + offsets[b + 1];
            if (std::equal(ab, ae, bb, be)) return a < b; // Break ties by preserving existing order (stable sort)
            return std::lexicographical_compare(ab, ae, bb, be);
        };
        for (Index p = 0; p < np; ++p)
            std::sort(elementOrder.begin() + partitionOffsets[p], elementOrder.begin() + partitionOffsets[p + 1], less);
    }

    template<class Element>
    void validate(size_t ne, const Element &element) const {
        if ((!elementOrder.empty() && elementOrder.size() != ne) || ne != numElements() || partitionOffsets.empty() || partitionOffsets.front() != 0 ||
            size_t(partitionOffsets.back()) != ne)
            throw std::invalid_argument("Partition topology/array dimensions do not match");
        std::vector<Index> seen(ne, 0); // for verifying `elementOrder` is a valid permutation.
        std::vector<Index> owningPartition(numBlockVars(), -1);
        for (Index p = 0; p < numPartitions(); ++p) {
            if (partitionOffsets[p] < 0 || partitionOffsets[p] >= partitionOffsets[p+1] || size_t(partitionOffsets[p+1]) > ne) throw std::invalid_argument("Empty or invalid partition range");
            for (Index i = partitionOffsets[p]; i < partitionOffsets[p+1]; ++i) {
                Index e = elementIndex(i);
                if (size_t(e) >= ne || seen[e]++)
                    throw std::invalid_argument("Invalid element ownership/permutation");
                const auto &nodes = element(e);
                // Verify that no node has incident elements from distinct partitions
                // (unless it is protected by a lock).
                for (decltype(nodes.size()) j = 0; j < nodes.size(); ++j) {
                    auto v = nodes[j];
                    if (size_t(v) >= numBlockVars()) throw std::invalid_argument("Invalid element block-variable index");
                    if (variableNeedsLock[v]) continue;
                    if (owningPartition[v] >= 0 && owningPartition[v] != p) throw std::invalid_argument("Unlocked variable block has multiple writers");
                    owningPartition[v] = p;
                }
            }
        }
    }
};
} // namespace MeshFEM

#endif /* end of include guard: ELEMENTPARTITIONFROMND_HH */
