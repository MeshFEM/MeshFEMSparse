#if MESHFEM_WITH_CATAMARI && !defined(MESHFEM_USE_LEGACY_CATAMARI)
#include <catamari/sparse_ldl/supernodal/factorization/trs_kernels.hpp>
#include <catamari/sparse_ldl/supernodal/factorization/solve_kernels.hpp>
#include <array>
#include <tbb/global_control.h>
#include <catch2/catch.hpp>

// These assertions also run in cross-target syntax checks: Apple Silicon tuning
// must not silently replace the established x86 kernel dispatch.
namespace {
template<int BS> constexpr bool checkSolveKernelPolicy() {
    using P = catamari::supernodal_ldl::solve_kernels::Policy<BS>;
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    return P::fused_forward_max_size == (BS == 2 ? 384 : 192) &&
           P::fused_backward_max_size == 48 && P::fused_backward_max_degree == 384 &&
           P::backward_size == (BS == 1 ? 64 : 96) &&
           P::backward_degree == (BS == 1 ? 100 : 384) &&
           P::parallel_min_entries == (BS == 3 ? 32768 : 0);
#else
    return P::fused_forward_max_size < 0 && P::fused_backward_max_size < 0 &&
           P::parallel_min_entries == 0 && P::backward_degree == 100 &&
#if defined(__APPLE__)
           P::backward_size == 64;
#else
           P::backward_size == 18;
#endif
#endif
}
static_assert(checkSolveKernelPolicy<1>() && checkSolveKernelPolicy<2>() &&
              checkSolveKernelPolicy<3>(), "Unexpected architecture-specific solve dispatch");
}

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
        // Include empty updates, chunk boundaries, and every block-sized tail.
        for (int degree : {0, BS, 2 * BS, 3 * BS, 4 * BS, 5 * BS, 7 * BS, 8 * BS, 9 * BS}) {
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
                std::vector<catamari::Int> indices(degree / BS);
                for (int i = 0; i < degree; ++i) {
                    const int row = start + n + 2 * BS * (i / BS) + i % BS;
                    if (i % BS == 0) indices[i / BS] = row;
                    update.block(row, start, 1, n) = A.row(i);
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

TEMPLATE_TEST_CASE("Catamari fused forward solve handles tails and initialization", "[catamari][solve_kernels]", float, double) {
    using Field = TestType;
    using Matrix = Eigen::Matrix<Field, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Field, Eigen::Dynamic, 1>;
    for (int n : {1, 2, 3, 15, 16, 23, 24, 47}) for (int d : {0, 1, 3, 16, 31})
    for (bool initialize : {false, true}) {
        CAPTURE(n, d, initialize);
        Matrix storage = Matrix::Random(n + d + 3, n);
        auto D = storage.topRows(n);
        auto E = storage.middleRows(n, d);
        D.diagonal().array() += Field(n + 2);
        Vector rhs = Vector::Random(n), schur = Vector::Random(d + 3);
        Vector expected = D.template triangularView<Eigen::Lower>().solve(rhs);
        Vector expected_schur = -E * expected;
        if (!initialize) expected_schur += schur.head(d);
        else schur.head(d).setConstant(std::numeric_limits<Field>::quiet_NaN());
        schur.tail(3).setConstant(Field(99));
        catamari::ConstBlasMatrixView<Field> dv, ev;
        dv.height = dv.width = n; dv.leading_dim = storage.rows(); dv.data = D.data();
        ev.height = d; ev.width = n; ev.leading_dim = storage.rows(); ev.data = E.data();
        catamari::supernodal_ldl::solve_kernels::fused_forward(dv, ev, rhs.data(), schur.data(), initialize);
        const double tolerance = std::is_same<Field, float>::value ? 2e-5 : 1e-11;
        REQUIRE((rhs - expected).norm() < tolerance * (1 + expected.norm()));
        REQUIRE((schur.head(d) - expected_schur).norm() < tolerance * (1 + expected_schur.norm()));
        REQUIRE((schur.tail(3).array() == Field(99)).all());
    }
}

namespace {
template<class Field, int BS>
void checkFusedScatter() {
    using Matrix = Eigen::Matrix<Field, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Field, Eigen::Dynamic, 1>;
    // Dense references exercise odd column counts, both destination sets,
    // noncontiguous block targets, padding, and the zero-degree root case.
    for (int n : {1, 2, 3, 17}) for (int blocks : {0, 1, 7})
    for (int owned = 0; owned <= blocks; ++owned) {
        CAPTURE(BS, n, blocks, owned);
        const int d = BS * blocks;
        Matrix storage = Matrix::Random(n + d + 3, n);
        auto D = storage.topRows(n); auto E = storage.middleRows(n, d);
        D.diagonal().array() += Field(n + 2);
        Vector global = Vector::Random(n + 2 * d + 5), boundary = Vector::Random(2 * d + 5);
        Vector expected_global = global, expected_boundary = boundary;
        Vector x = D.template triangularView<Eigen::Lower>().solve(global.head(n));
        Vector update = -E * x;
        expected_global.head(n) = x;
        std::vector<catamari::Int> indices(blocks), external(blocks - owned);
        for (int i = 0; i < owned; ++i) {
            indices[i] = n + 1 + 2 * BS * i;
            expected_global.segment(indices[i], BS) += update.segment(BS * i, BS);
        }
        for (int i = owned; i < blocks; ++i) {
            external[i - owned] = 1 + 2 * BS * (blocks - 1 - i);
            expected_boundary.segment(external[i - owned], BS) += update.segment(BS * i, BS);
        }
        catamari::ConstBlasMatrixView<Field> dv, ev;
        dv.height = dv.width = n; dv.leading_dim = storage.rows(); dv.data = D.data();
        ev.height = d; ev.width = n; ev.leading_dim = storage.rows(); ev.data = E.data();
        catamari::supernodal_ldl::solve_kernels::fused_forward_scatter<BS>(
            dv, ev, global.data(), global.data(), boundary.data(), indices.data(), owned, external.data());
        const double tol = std::is_same<Field, float>::value ? 2e-5 : 1e-11;
        REQUIRE((global - expected_global).norm() < tol * (1 + expected_global.norm()));
        REQUIRE((boundary - expected_boundary).norm() < tol * (1 + expected_boundary.norm()));
    }
}
}
TEMPLATE_TEST_CASE("Catamari fused scatter matches dense triangular solve and update", "[catamari][solve_kernels][subtree_accumulation]", float, double) {
    checkFusedScatter<TestType, 1>();
    checkFusedScatter<TestType, 2>();
    checkFusedScatter<TestType, 3>();
}

namespace {
template<class Field, int BS>
void checkFusedBackward() {
    using Matrix = Eigen::Matrix<Field, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Field, Eigen::Dynamic, 1>;
    for (int n : {1, 2, 3, 17, 24, 47, 96, 193}) for (int blocks : {0, 1, 7, 129}) {
        CAPTURE(BS, n, blocks);
        const int d = BS * blocks;
        Matrix storage = Matrix::Random(n + d + 3, n);
        auto D = storage.topRows(n); auto E = storage.middleRows(n, d);
        D.diagonal().array() += Field(n + 2);
        Vector global = Vector::Random(n + 2 * d + 5), expected = global;
        Vector ancestors(d);
        std::vector<catamari::Int> indices(blocks);
        for (int i = 0; i < blocks; ++i) {
            indices[i] = n + 3 + 2 * BS * i;
            ancestors.segment(BS * i, BS) = global.segment(indices[i], BS);
        }
        Vector updated = global.segment(1, n) - E.transpose() * ancestors;
        expected.segment(1, n) = D.transpose().template triangularView<Eigen::Upper>().solve(updated);
        catamari::ConstBlasMatrixView<Field> dv, ev;
        dv.height = dv.width = n; dv.leading_dim = storage.rows(); dv.data = D.data();
        ev.height = d; ev.width = n; ev.leading_dim = storage.rows(); ev.data = E.data();
        catamari::supernodal_ldl::solve_kernels::fused_backward<BS>(
            dv, ev, global.data() + 1, global.data(), indices.data());
        const double tol = std::is_same<Field, float>::value ? 2e-5 : 1e-11;
        REQUIRE((global - expected).norm() < tol * (1 + expected.norm()));
        REQUIRE(global[0] == expected[0]);
        REQUIRE((global.tail(global.size() - n - 1).array() == expected.tail(global.size() - n - 1).array()).all());
    }
}
}
TEMPLATE_TEST_CASE("Catamari fused backward matches dense transpose solve", "[catamari][solve_kernels][fused_backward]", float, double) {
    checkFusedBackward<TestType, 1>();
    checkFusedBackward<TestType, 2>();
    checkFusedBackward<TestType, 3>();
}


TEMPLATE_TEST_CASE("Catamari parallel dense updates match strided dense products", "[catamari][solve_kernels]", float, double) {
    using Field = TestType;
    using Matrix = Eigen::Matrix<Field, Eigen::Dynamic, Eigen::Dynamic>;
    tbb::global_control limit(tbb::global_control::max_allowed_parallelism, 4);
    for (bool transpose : {false, true}) for (int nrhs : {1, 3})
    for (Field beta : {Field(0), Field(1)}) {
        // Exceed the parallel threshold with uneven tiles and padded strides.
        constexpr int m = 389, n = 257;
        Matrix storage = Matrix::Random(m + 3, n);
        auto A = storage.topRows(m);
        const int rows = transpose ? n : m, inner = transpose ? m : n;
        Matrix X = Matrix::Random(inner + 2, nrhs), Y = Matrix::Random(rows + 3, nrhs);
        Matrix expected = beta * Y.topRows(rows);
        if (transpose) expected.noalias() -= A.transpose() * X.topRows(inner);
        else expected.noalias() -= A * X.topRows(inner);
        Y.bottomRows(3).setConstant(Field(99));
        if (beta == Field(0)) Y.topRows(rows).setConstant(std::numeric_limits<Field>::quiet_NaN());
        catamari::ConstBlasMatrixView<Field> a, x;
        catamari::BlasMatrixView<Field> y;
        a.height=m; a.width=n; a.leading_dim=storage.rows(); a.data=storage.data();
        x.height=inner; x.width=nrhs; x.leading_dim=X.rows(); x.data=X.data();
        y.height=rows; y.width=nrhs; y.leading_dim=Y.rows(); y.data=Y.data();
        catamari::supernodal_ldl::solve_kernels::multiply<3>(transpose, Field(-1), a, x, beta, &y);
        const double tol = std::is_same<Field, float>::value ? 2e-5 : 1e-11;
        REQUIRE((Y.topRows(rows) - expected).norm() < tol * (1 + expected.norm()));
        REQUIRE((Y.bottomRows(3).array() == Field(99)).all());
    }
}

#endif
