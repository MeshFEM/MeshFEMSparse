#ifndef MESHFEMSPARSE_CHOLMOD_NESDIS_PARALLEL_HH
#define MESHFEMSPARSE_CHOLMOD_NESDIS_PARALLEL_HH

#include <cstdint>
#include <cstddef>
#include <string>
#include "persistent_separator_tree.hh"

extern "C" {
#include <cholmod.h>
}

extern "C" int64_t cholmod_nested_dissection_parallel(
    cholmod_sparse *A,
    int32_t *fset,
    size_t fsize,
    int32_t *Perm,
    int32_t *CParent,
    int32_t *Cmember,
    cholmod_common *Common);

extern "C" int64_t cholmod_l_nested_dissection_parallel(
    cholmod_sparse *A,
    int64_t *fset,
    size_t fsize,
    int64_t *Perm,
    int64_t *CParent,
    int64_t *Cmember,
    cholmod_common *Common);

namespace MeshFEM::CholmodParallelNesdis {

struct TemporalReuseStatistics {
    bool full_rebuild = false;
    size_t graph_vertices = 0;
    size_t recomputed_subtrees = 0;
    size_t repartitioned_vertices = 0;
    double nd_seconds = 0; // Graph preparation, validation and ND; excludes CAMD.
};

// Caller-owned history, independent for each ordering stream and index width.
// Vertex IDs must retain their meaning between calls. Reset on relabeling.
// A state/Common pair must not be used by concurrent calls. After a failed call,
// the previous successful history and statistics remain available.
template<class Int>
struct TemporalReuseState {
    size_t temporal_reuse_period = 0; // 0 disables; P permits P incremental calls.
    PersistentSeparatorTree<Int> tree;
    size_t incremental_analyses_since_rebuild = 0;
    TemporalReuseStatistics statistics;
    std::string partition_settings;
    void reset() {
        tree = {};
        incremental_analyses_since_rebuild = 0;
        statistics = {};
        partition_settings.clear();
    }
};

// Stateful C++ equivalent of the C entry points (including A*A' semantics).
template<class Int>
int64_t nested_dissection(cholmod_sparse *A, Int *fset, size_t fsize,
    Int *Perm, Int *CParent, Int *Cmember, cholmod_common *Common,
    TemporalReuseState<Int> &reuse);

// Order an already-expanded symmetric graph, not graph * graph'. Input must be
// valid, square, sorted, packed CSC with stype == 0 and matching Int indices.
// Both triangles must be present; diagonal entries are optional. The input is
// borrowed read-only throughout the call; discovery uses a private index copy
// with diagonals removed.
// Explicitly instantiated for int32_t and int64_t. Outputs follow the C API above.
template<class Int>
int64_t nested_dissection_from_graph(const cholmod_sparse &graph,
    Int *Perm, Int *CParent, Int *Cmember, cholmod_common *Common,
    TemporalReuseState<Int> *reuse = nullptr);

} // namespace MeshFEM::CholmodParallelNesdis

#endif /* MESHFEMSPARSE_CHOLMOD_NESDIS_PARALLEL_HH */
