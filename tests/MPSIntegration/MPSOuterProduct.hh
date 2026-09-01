#ifndef MESHFEM_TESTS_MPSOUTERPRODUCT_HH
#define MESHFEM_TESTS_MPSOUTERPRODUCT_HH

#include <cstddef>
#include <iosfwd>

namespace MeshFEM {

bool mpsOuterProductAvailable();
void printMPSOuterProductDeviceInfo(std::ostream &os);
double mpsLowerNormalHermitianOuterProduct(float alpha,
                                           const float *columnMajorLeft,
                                           size_t height,
                                           size_t rank,
                                           float beta,
                                           float *columnMajorOutput);

} // namespace MeshFEM

#endif /* end of include guard: MESHFEM_TESTS_MPSOUTERPRODUCT_HH */
