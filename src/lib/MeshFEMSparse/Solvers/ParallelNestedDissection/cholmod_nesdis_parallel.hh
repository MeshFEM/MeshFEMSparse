#ifndef MESHFEMSPARSE_CHOLMOD_NESDIS_PARALLEL_HH
#define MESHFEMSPARSE_CHOLMOD_NESDIS_PARALLEL_HH

#include <cstdint>
#include <cstddef>

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

// Order an already-expanded symmetric graph, not graph * graph'. Input must be
// valid, square, sorted, packed CSC with stype == 0 and matching Int indices.
// Both triangles must be present; diagonal entries are optional. The input is
// borrowed read-only throughout the call; discovery uses a private index copy.
// Explicitly instantiated for int32_t and int64_t. Outputs follow the C API above.
template<class Int>
int64_t nested_dissection_from_graph(const cholmod_sparse &graph,
    Int *Perm, Int *CParent, Int *Cmember, cholmod_common *Common);

} // namespace MeshFEM::CholmodParallelNesdis

#endif /* MESHFEMSPARSE_CHOLMOD_NESDIS_PARALLEL_HH */
