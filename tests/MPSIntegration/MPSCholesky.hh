#ifndef MESHFEM_TESTS_MPSCHOLESKY_HH
#define MESHFEM_TESTS_MPSCHOLESKY_HH

#include <cstddef>

namespace MeshFEM {

bool mpsCholeskyAvailable();
double mpsLowerCholeskyFactorization(float *columnMajorMatrix, size_t n);

} // namespace MeshFEM

#endif /* end of include guard: MESHFEM_TESTS_MPSCHOLESKY_HH */
