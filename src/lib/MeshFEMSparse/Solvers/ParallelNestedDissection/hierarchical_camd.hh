#ifndef MESHFEMSPARSE_HIERARCHICAL_CAMD_HH
#define MESHFEMSPARSE_HIERARCHICAL_CAMD_HH

#include <cholmod.h>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace MeshFEM::CholmodParallelNesdis {

// One coarse split of CHOLMOD's postordered ND forest. The input graph has both
// triangles and packed, duplicate-free columns; diagonal entries are ignored.
struct HierarchicalCAMDOptions {
    int cut_depth = 4; // Clamped to the deepest level present in the forest.
    bool verify = false; // Includes exact symbolic elimination for n <= 512.
    bool trace = false;
    bool skip_upper = false; // Keep separator postorder without upper CAMD.
};

struct CAMDTaskStatistics {
    size_t vertices = 0, edges = 0, constraints = 0;
    size_t boundary_components = 0, max_boundary = 0, sum_boundary = 0;
    double sum_boundary_squared = 0, camd_seconds = 0;
};

template<class Int>
struct HierarchicalCAMDResult {
    std::vector<Int> permutation, upper_vertices;
    // Unique undirected input edges. Synthetic vertices follow
    // upper_vertices; eliminating them produces the explicit upper fill graph.
    // Empty when skip_upper is enabled; no boundary/upper graph is constructed.
    std::vector<std::pair<Int, Int>> upper_edges;
    std::vector<CAMDTaskStatistics> tasks;
    size_t original_upper_edges = 0;
    size_t quotient_elements = 0;
    double decomposition_seconds = 0, local_seconds = 0;
    double upper_graph_seconds = 0, upper_camd_seconds = 0;
};

template<class Int>
HierarchicalCAMDResult<Int> hierarchical_camd(
    const cholmod_sparse &graph, const Int *parent, Int ncomponents,
    const Int *member, const cholmod_common &settings,
    const HierarchicalCAMDOptions &options);

// Optional baseline diagnostics: component sizes, depths and subtree sizes.
template<class Int>
void trace_camd_tree(size_t n, const Int *parent, Int ncomponents, const Int *member);

} // namespace MeshFEM::CholmodParallelNesdis
#endif
