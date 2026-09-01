#ifndef MESHFEM_TESTS_MPSGEMM_HH
#define MESHFEM_TESTS_MPSGEMM_HH

#include <cstddef>
#include <iosfwd>

namespace MeshFEM {

bool mpsGemmAvailable();
void printMPSGemmDeviceInfo(std::ostream &os);
double mpsGemm(float alpha,
               const float *columnMajorLeft,
               size_t height,
               size_t contractionSize,
               const float *columnMajorRight,
               size_t width,
               float beta,
               float *columnMajorOutput);

} // namespace MeshFEM

#endif /* end of include guard: MESHFEM_TESTS_MPSGEMM_HH */
