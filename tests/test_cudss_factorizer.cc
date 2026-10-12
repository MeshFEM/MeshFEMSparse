#if MESHFEM_WITH_CUDSS
#include <MeshFEMSparse/Solvers/CuDSSFactorizer.hh>
#include <MeshFEMSparse/Solvers/make_cholesky_factorizer.hh>
#include <MeshFEMSparse/SystemAssembler.hh>

// Include Catch after MeshFEM to avoid its BENCHMARK macro conflict.
#include <catch2/catch.hpp>

using namespace MeshFEM;

namespace {
// Upper triangle of the 2D finite-difference Laplacian on a k x k grid (SPD).
SuiteSparseMatrix laplacian2D(size_t k) {
    const size_t n = k * k;
    TripletMatrix<> triplets(n, n);
    for (size_t r = 0; r < k; ++r) {
        for (size_t c = 0; c < k; ++c) {
            const size_t i = r * k + c;
            triplets.addNZ(i, i, 4.0);
            if (c + 1 < k) triplets.addNZ(i, i + 1, -1.0);
            if (r + 1 < k) triplets.addNZ(i, i + k, -1.0);
        }
    }
    SuiteSparseMatrix A(triplets);
    A.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    return A;
}

// Strict diagonal dominance gives a well-conditioned SPD system with
// nontrivial off-diagonal blocks and coupling inside diagonal blocks.
SuiteSparseMatrix diagonallyDominant6x6() {
    TripletMatrix<> triplets(6, 6);
    for (size_t j = 0; j < 6; ++j)
        for (size_t i = 0; i <= j; ++i)
            triplets.addNZ(i, j, i == j ? 8.0 : 0.125 * (i + j + 1));
    SuiteSparseMatrix A(triplets);
    A.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    return A;
}

// Float has ~7 significant digits, and errors grow with the condition number
// (~180 for the 20x20 Laplacian), so single precision gets a looser tolerance.
double solveTolerance(const CuDSSFactorizer &factorizer) {
    return factorizer.isSinglePrecision() ? 1e-4 : 1e-12;
}

// Solve `(A + sigma I) x = b` for a known `x` and compare.
template<class Matrix>
void checkCuDSSSolves(CuDSSFactorizer &factorizer, const Matrix &A, const SuiteSparseMatrix &scalarA,
                      const std::vector<size_t> &pins) {
    Eigen::VectorXd expected = Eigen::VectorXd::LinSpaced(scalarA.n, -0.5, 1.0);
    for (size_t i : pins) expected[i] = 0.0;
    const double tolerance = solveTolerance(factorizer);
    const bool singlePrecision = factorizer.isSinglePrecision();

    // Exercise symbolic replacement, numeric reuse, and reuse after clearing.
    for (int repeat = 0; repeat < 3; ++repeat) {
        factorizer.factorizeSymbolic(A, pins);
        for (double sigma : {0.0, 0.75, 0.25, 0.0}) {
            CAPTURE(singlePrecision, pins, repeat, sigma);
            if (sigma == 0.0) factorizer.factorizeNumeric(A);
            else              factorizer.factorizeNumericWithShift(A, sigma);
            REQUIRE(factorizer.checkPosDef());
            Eigen::VectorXd b = scalarA.apply(expected) + sigma * expected;
            Eigen::VectorXd x = factorizer.solve(b);
            REQUIRE((x - expected).norm() < tolerance * expected.norm());
        }
        if (repeat == 1) factorizer.clearFactors();
    }
}
}

TEST_CASE("cuDSS solves scalar and block matrices", "[cudss]") {
    const bool singlePrecision = GENERATE(false, true);
    const auto pins = GENERATE(std::vector<size_t>{}, std::vector<size_t>{0, 1}, std::vector<size_t>{1});
    SuiteSparseMatrix scalarA = diagonallyDominant6x6();
    CuDSSFactorizer factorizer(singlePrecision);

    SECTION("scalar matrix") {
        checkCuDSSSolves(factorizer, scalarA, scalarA, pins);
    }
    SECTION("block matrix (converted to scalar by the base class)") {
        SystemAssembler<2> assembler(3);
        auto A = assembler.blockSparsityPattern(1, [](size_t) {
            return std::vector<size_t>{0, 1, 2};
        });
        A->setZero();
        for (const auto &entry : scalarA)
            A->addNZScalar(entry.i, entry.j, entry.value());
        checkCuDSSSolves(factorizer, *A, scalarA, pins);
    }
}

TEST_CASE("cuDSS handles a larger system and a change of sparsity pattern", "[cudss]") {
    const bool singlePrecision = GENERATE(false, true);
    const auto pins = GENERATE(std::vector<size_t>{}, std::vector<size_t>{0, 7, 100, 399});
    CuDSSFactorizer factorizer(singlePrecision);

    // The same factorizer (and its cuDSS data object) is reused for a second,
    // differently sized pattern.
    for (size_t k : {20, 13}) {
        CAPTURE(k);
        SuiteSparseMatrix A = laplacian2D(k);
        std::vector<size_t> validPins;
        for (size_t i : pins) if (i < size_t(A.n)) validPins.push_back(i);
        checkCuDSSSolves(factorizer, A, A, validPins);
    }
}

TEST_CASE("cuDSS rejects indefinite matrices and unsupported solves", "[cudss]") {
    const bool singlePrecision = GENERATE(false, true);
    CAPTURE(singlePrecision);
    SuiteSparseMatrix A = diagonallyDominant6x6();
    CuDSSFactorizer factorizer(singlePrecision);
    factorizer.factorizeSymbolic(A);

    SECTION("indefinite matrix throws, and the factorizer recovers") {
        // A's eigenvalues average 8 (trace 48 / 6), so A - 12 I has negative
        // eigenvalues and is not positive definite.
        REQUIRE_THROWS_AS(factorizer.factorizeNumericWithShift(A, -12.0), std::runtime_error);
        REQUIRE_FALSE(factorizer.checkPosDef());
        REQUIRE_THROWS(factorizer.solve(Eigen::VectorXd::Ones(6).eval()));

        // As in the Newton solver's retry loop: a later successful factorization works.
        factorizer.factorizeNumeric(A);
        REQUIRE(factorizer.checkPosDef());
        Eigen::VectorXd expected = Eigen::VectorXd::LinSpaced(6, -0.5, 1.0);
        Eigen::VectorXd x = factorizer.solve(A.apply(expected));
        REQUIRE((x - expected).norm() < solveTolerance(factorizer) * expected.norm());
    }

    SECTION("only plain, unpermuted solves with A are supported") {
        factorizer.factorizeNumeric(A);
        Eigen::VectorXd b = Eigen::VectorXd::Ones(6), x(6);
        REQUIRE_THROWS_AS(factorizer.solveRawReduced(b.data(), x.data(), CholeskySys::A, true), std::invalid_argument);
        REQUIRE_THROWS_AS(factorizer.solveRawReduced(b.data(), x.data(), CholeskySys::L),       std::invalid_argument);
    }

    SECTION("numeric factorization before symbolic throws") {
        CuDSSFactorizer fresh(singlePrecision);
        REQUIRE_THROWS(fresh.factorizeNumeric(A));
    }
}

TEST_CASE("cuDSS factory honors singlePrecision", "[cudss]") {
    const bool singlePrecision = GENERATE(false, true);
    auto f = make_cholesky_factorizer(CholeskyProvider::CUDSS, singlePrecision);
    REQUIRE(f->provider() == CholeskyProvider::CUDSS);
    REQUIRE(dynamic_cast<const CuDSSFactorizer &>(*f).isSinglePrecision() == singlePrecision);
}
#endif
