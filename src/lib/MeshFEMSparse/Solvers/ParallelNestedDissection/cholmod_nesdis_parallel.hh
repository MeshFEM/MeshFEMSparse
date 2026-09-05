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

#endif /* MESHFEMSPARSE_CHOLMOD_NESDIS_PARALLEL_HH */
