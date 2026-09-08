#ifdef __APPLE__
#include <MeshFEMSparse/Solvers/AccelerateFactorizer.hh>
#include <MeshFEMSparse/SystemAssembler.hh>

// Include Catch after MeshFEM to avoid its BENCHMARK macro conflict.
#include <catch2/catch.hpp>

using namespace MeshFEM;

namespace {
template<class Matrix>
void checkAccelerateSolves(const Matrix &A, const SuiteSparseMatrix &scalarA,
                          bool singlePrecision, const std::vector<size_t> &pins,
                          bool useBlockAccel = true) {
    AccelerateFactorizer factorizer(singlePrecision);
    factorizer.setUseBlockAccel(useBlockAccel);
    Eigen::VectorXd expected = Eigen::VectorXd::LinSpaced(scalarA.n, -0.5, 1.0);
    for (size_t i : pins) expected[i] = 0.0;
    const double tolerance = singlePrecision ? 1e-5 : 1e-12;

    // Exercise symbolic replacement, numeric reuse, and reuse after clearing.
    for (int repeat = 0; repeat < 3; ++repeat) {
        factorizer.factorizeSymbolic(A, pins);
        for (double sigma : {0.0, 0.75, 0.25, 0.0}) {
            CAPTURE(singlePrecision, pins, useBlockAccel, repeat, sigma);
            if (sigma == 0.0) factorizer.factorizeNumeric(A);
            else              factorizer.factorizeNumericWithShift(A, sigma);
            Eigen::VectorXd b = scalarA.apply(expected) + sigma * expected;
            Eigen::VectorXd x = factorizer.solve(b);
            REQUIRE((x - expected).norm() < tolerance * expected.norm());
        }
        if (repeat == 1) factorizer.clearFactors();
    }
    factorizer.clearFactors();
    factorizer.clearFactors();
}
}

TEST_CASE("Accelerate solves in double and single precision", "[accelerate]") {
    const bool singlePrecision = GENERATE(false, true);
    const auto pins = GENERATE(std::vector<size_t>{}, std::vector<size_t>{0, 1}, std::vector<size_t>{1});

    // Strict diagonal dominance gives a well-conditioned SPD system with
    // nontrivial off-diagonal blocks and coupling inside diagonal blocks.
    TripletMatrix<> triplets(6, 6);
    for (size_t j = 0; j < 6; ++j)
        for (size_t i = 0; i <= j; ++i)
            triplets.addNZ(i, j, i == j ? 8.0 : 0.125 * (i + j + 1));
    SuiteSparseMatrix scalarA(triplets);
    scalarA.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;

    SECTION("scalar matrix") {
        checkAccelerateSolves(scalarA, scalarA, singlePrecision, pins);
    }
    SECTION("block matrix") {
        SystemAssembler<2> assembler(3);
        auto A = assembler.blockSparsityPattern(1, [](size_t) {
            return std::vector<size_t>{0, 1, 2};
        });
        A->setZero();
        for (const auto &entry : scalarA)
            A->addNZScalar(entry.i, entry.j, entry.value());

        SECTION("native blocks with scalar fallback for partial pins") {
            checkAccelerateSolves(*A, scalarA, singlePrecision, pins);
        }
        SECTION("explicit scalar conversion") {
            checkAccelerateSolves(*A, scalarA, singlePrecision, pins, false);
        }
    }
}
#endif
