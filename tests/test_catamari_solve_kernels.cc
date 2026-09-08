#if MESHFEM_WITH_CATAMARI && !defined(MESHFEM_USE_LEGACY_CATAMARI)
#include <catamari/sparse_ldl/supernodal/factorization/trs_kernels.hpp>
#include <array>
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("Catamari lower-block updates advance through RHS columns",
                   "[catamari][solve_kernels]", float, long double) {
    // Long double keeps testing the generic path if float gains a specialization.
    const catamari::Int indices[] = {2};
    const TestType lowerBlock[] = {2};
    // Three distinct RHS columns, with a padding entry after each column.
    std::array<TestType, 12> rhs{{9, 1, 5, 99, 8, 3, 11, 99, 7, -2, 0, 99}};
    const std::array<TestType, 12> expected{{9, 1, 3, 99, 8, 3, 5, 99, 7, -2, 4, 99}};

    catamari::supernodal_ldl::trs_kernels::MultiplyLowerBlock<TestType, 1>::run(
        indices, /* supernode_start = */ 1, /* supernode_size = */ 1,
        /* degree = */ 1, lowerBlock, /* A_leading_dim = */ 1,
        /* num_rhs = */ 3, rhs.data(), /* B_leading_dim = */ 4);

    for (size_t i = 0; i < rhs.size(); ++i) {
        CAPTURE(i);
        REQUIRE(rhs[i] == expected[i]);
    }
}

namespace {
template<class Field, int BS>
void checkEigenKernels() {
    using Matrix = Eigen::Matrix<Field, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix<Field, Eigen::Dynamic, 1>;
    using Map = Eigen::Map<Matrix, 0, Eigen::OuterStride<>>;
    for (int n : {BS, 2 * BS, 5 * BS}) {
        for (int degree : {BS, 7 * BS}) {
            for (int nrhs : {1, 3}) {
                CAPTURE(BS, n, degree, nrhs);
                const int start = BS, rows = start + n + 2 * degree;
                const int lda = degree + 2 * BS, ldb = rows + 2 * BS;
                // Exercise unaligned float addresses and padded column strides.
                const int offset = std::is_same<Field, float>::value ? 1 : 0;
                std::vector<Field> aStorage(lda * n + offset, Field(99));
                std::vector<Field> bStorage(ldb * nrhs + offset, Field(99));
                Map A(aStorage.data() + offset, degree, n, Eigen::OuterStride<>(lda));
                Map B(bStorage.data() + offset, rows, nrhs, Eigen::OuterStride<>(ldb));
                A.setRandom(); B.setRandom();
                Matrix original = B, update = Matrix::Zero(rows, rows);
                std::vector<catamari::Int> indices(degree);
                for (int i = 0; i < degree; ++i) {
                    indices[i] = start + n + 2 * BS * (i / BS) + i % BS;
                    update.block(indices[i], start, 1, n) = A.row(i);
                }
                Matrix expected = original - update * original;
                catamari::supernodal_ldl::trs_kernels::MultiplyLowerBlock<Field, BS>::run(
                    indices.data(), start, n, degree, A.data(), lda, nrhs, B.data(), ldb);
                const double tol = std::is_same<Field, float>::value ? 2e-5 : 1e-11;
                REQUIRE((B - expected).norm() < tol * (1 + expected.norm()));
                B = original;
                expected = original - update.transpose() * original;
                catamari::supernodal_ldl::trs_kernels::MultiplyLowerBlockAdjoint<Field, BS>::run(
                    true, indices.data(), start, n, degree, A.data(), lda, nrhs, B.data(), ldb);
                REQUIRE((B - expected).norm() < tol * (1 + expected.norm()));
                for (int j = 0; j < nrhs; ++j)
                    for (int i = rows; i < ldb; ++i)
                        REQUIRE(bStorage[offset + j * ldb + i] == Field(99));
            }
        }
        Matrix L = Matrix::Random(n, n);
        L.diagonal().array() += Field(n + 2);
        Vec x = Vec::Random(n);
        Vec b = L.template triangularView<Eigen::Lower>() * x;
        catamari::supernodal_ldl::trs_kernels::SolveLowerTri<Field, BS>::run(n, L.data(), n, b.data());
        const double tol = std::is_same<Field, float>::value ? 2e-5 : 1e-11;
        REQUIRE((b - x).norm() < tol * (1 + x.norm()));
        b = L.transpose().template triangularView<Eigen::Upper>() * x;
        catamari::supernodal_ldl::trs_kernels::SolveLowerTriAdjoint<Field, BS>::run(n, L.data(), n, b.data());
        REQUIRE((b - x).norm() < tol * (1 + x.norm()));
    }
}
}

TEMPLATE_TEST_CASE("Catamari real solve kernels match dense operations", "[catamari][solve_kernels]", float, double) {
    checkEigenKernels<TestType, 1>();
    checkEigenKernels<TestType, 2>();
    checkEigenKernels<TestType, 3>();
}
#endif
