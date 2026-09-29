#include <MeshFEMSparse/Solvers/make_cholesky_factorizer.hh>
#include <MeshFEMSparse/Solvers/CholmodFactorizer.hh>
#include <MeshFEMSparse/Solvers/CatamariFactorizer.hh>
#include <MeshFEMSparse/SystemAssembler.hh>
#include <catch2/catch.hpp>
using namespace MeshFEM;

TEST_CASE("Native ordering preserves reduced input order", "[native_ordering]") {
    std::vector<CholeskyProvider> providers;
#if MESHFEM_WITH_CATAMARI && !defined(MESHFEM_USE_LEGACY_CATAMARI)
    providers.push_back(CholeskyProvider::CatamariNative);
#endif
#if MESHFEM_WITH_CHOLMOD
    providers.push_back(CholeskyProvider::CHOLMODNative);
#endif
    // A center-first star is deliberately not in elimination-tree postorder.
    // Dense or path matrices alone can hide an unwanted postordering.
    SystemAssembler<2> assembler(12);
    auto A = assembler.blockSparsityPattern(11, [](size_t e) { return std::array<size_t, 2>{{0, e + 1}}; });
    A->setZero();
    auto scalar = A->toScalar();
    for (auto e : scalar) A->addNZScalar(e.i, e.j, e.i == e.j ? 30.0 : -0.1 * (1 + (e.i + e.j) % 7));
    scalar = A->toScalar();
    for (auto provider : providers) for (bool single : {false, true}) {
        if (single && provider == CholeskyProvider::CHOLMODNative) continue;
        for (bool blocks : {false, true}) for (auto pins : {std::vector<size_t>{}, std::vector<size_t>{2,3}, std::vector<size_t>{3}}) {
            CAPTURE(provider, single, blocks, pins);
            auto f = make_cholesky_factorizer(provider, single);
            REQUIRE(f->provider() == provider);
            Eigen::VectorXd expected = Eigen::VectorXd::LinSpaced(24, -0.7, 1.3);
            for (auto i : pins) expected[i] = 0;
            if (blocks) f->factorizeSymbolic(*A, pins);
            else        f->factorizeSymbolic(scalar, pins);
            const double tol = single ? 1e-5 : 1e-12;
            for (double shift : {0.0, 0.25}) {
                if (shift == 0) {
                    if (blocks) f->factorizeNumeric(*A);
                    else        f->factorizeNumeric(scalar);
                }
                else if (blocks) f->factorizeNumericWithShift(*A, shift);
                else             f->factorizeNumericWithShift(scalar, shift);
                Eigen::VectorXd rhs = scalar.apply(expected) + shift * expected;
                REQUIRE((f->solve(rhs) - expected).norm() < tol * expected.norm());
                Eigen::VectorXd reduced, result(f->n_reduced()), full;
                f->removeFixedEntries(rhs, reduced, false);
                Eigen::VectorXd ordered = reduced;
                f->solveRawReduced(ordered.data(), result.data(), CholeskySys::A, true);
                f->extractFullSolution(result, full, false);
                REQUIRE((full - expected).norm() < tol * expected.norm());
#if MESHFEM_WITH_CATAMARI && !defined(MESHFEM_USE_LEGACY_CATAMARI)
                if (auto c = dynamic_cast<CatamariFactorizer *>(f.get())) {
                    Eigen::VectorXd permuted;
                    c->removeFixedEntries(rhs, permuted, true);
                    REQUIRE((permuted - reduced).norm() == 0);
                }
#endif
#if MESHFEM_WITH_CHOLMOD
                if (auto c = dynamic_cast<CholmodFactorizer *>(f.get())) {
                    // Query the actual factor permutation, not a requested setting.
                    c->solveRawReduced(reduced.data(), result.data(), CholeskySys::P);
                    REQUIRE((result - reduced).norm() == 0);
                }
#endif
            }
        }
    }
}

TEST_CASE("Native ordering changes take effect at symbolic analysis", "[native_ordering]") {
    std::vector<CholeskyProvider> providers;
#if MESHFEM_WITH_CATAMARI && !defined(MESHFEM_USE_LEGACY_CATAMARI)
    providers.push_back(CholeskyProvider::CatamariNative);
#endif
#if MESHFEM_WITH_CHOLMOD
    providers.push_back(CholeskyProvider::CHOLMODNative);
#endif
    TripletMatrix<> entries(12, 12);
    for (size_t j = 0; j < 12; ++j) {
        entries.addNZ(j, j, 20.0);
        if (j) entries.addNZ(0, j, -0.2 * j);
    }
    SuiteSparseMatrix A(entries);
    A.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    Eigen::VectorXd expected = Eigen::VectorXd::LinSpaced(12, -1, 2);
    const Eigen::VectorXd rhs = A.apply(expected);
    for (auto provider : providers) {
        CAPTURE(provider);
        auto f = make_cholesky_factorizer(provider);
        auto setNative = [&](bool native) {
#if MESHFEM_WITH_CATAMARI && !defined(MESHFEM_USE_LEGACY_CATAMARI)
            if (auto c = dynamic_cast<CatamariFactorizer *>(f.get()))
                c->orderingMethod = native ? CatamariFactorizer::OrderingMethod::Native : CatamariFactorizer::OrderingMethod::AMD;
#endif
#if MESHFEM_WITH_CHOLMOD
            if (auto c = dynamic_cast<CholmodFactorizer *>(f.get()))
                c->setOrderingMethod(native ? CholmodFactorizer::OrderingMethod::Native : CholmodFactorizer::OrderingMethod::AMD);
#endif
        };
        for (bool native : {true, false, true}) {
            setNative(native);
            f->factorize(A);
            REQUIRE((f->solve(rhs) - expected).norm() < 1e-12);
            setNative(!native); // Must not reinterpret the existing factor's order.
            REQUIRE((f->solve(rhs) - expected).norm() < 1e-12);
        }
    }
#if MESHFEM_WITH_CHOLMOD
    for (bool forceLL : {false, true}) {
        CholmodFactorizer c(false, forceLL);
        c.setOrderingMethod(CholmodFactorizer::OrderingMethod::Native);
        c.factorize(A);
        c.stashFactorization();
        c.factorizeNumericWithShift(A, 0.5);
        c.swapStashedFactorization();
        Eigen::VectorXd x(12);
        c.solveRawReduced(rhs.data(), x.data(), CholeskySys::A, true);
        REQUIRE((x - expected).norm() < 1e-12);
    }
#endif
}

#if MESHFEM_WITH_CATAMARI && !defined(MESHFEM_USE_LEGACY_CATAMARI)
TEST_CASE("Catamari native ordering amalgamates without permuting", "[catamari][native_ordering]") {
    // Interleaved branches: unrestricted relaxation merges all five blocks
    // and reorders them. Native relaxation can retain [0] and merge [1,2,3,4].
    const std::array<std::array<size_t, 2>, 4> edges{{{{0,3}}, {{1,2}}, {{2,4}}, {{3,4}}}};
    SystemAssembler<2> assembler(5);
    auto A = assembler.blockSparsityPattern(edges.size(), [&](size_t e) { return edges[e]; });
    A->setZero();
    auto scalar = A->toScalar();
    for (auto e : scalar) A->addNZScalar(e.i, e.j, e.i == e.j ? 20.0 + e.i : 0.2);
    scalar = A->toScalar();
    const Eigen::VectorXd expected = Eigen::VectorXd::LinSpaced(10, 1, 10);
    const Eigen::VectorXd rhs = scalar.apply(expected);
    for (bool single : {false, true}) {
        CAPTURE(single);
        auto f = make_cholesky_factorizer(CholeskyProvider::CatamariNative, single);
        f->factorizeSymbolic(*A, {});
        // 3 entries in the 2x2 triangle, 36 in the 8x8 triangle, and a 2x2 off-diagonal block.
        // Fundamental-only supernodes store 31 entries, so this also checks relaxation is enabled.
        REQUIRE(f->getFactorNNZ() == 43);
        f->factorizeNumeric(*A);
        const double tol = single ? 1e-5 : 1e-12;
        REQUIRE((f->solve(rhs) - expected).norm() < tol * expected.norm());
        Eigen::VectorXd result(10);
        f->solveRawReduced(rhs.data(), result.data(), CholeskySys::A, true);
        REQUIRE((result - expected).norm() < tol * expected.norm());
        Eigen::VectorXd permuted;
        f->removeFixedEntries(rhs, permuted, true);
        REQUIRE((permuted - rhs).norm() == 0);
    }
}
#endif

#if MESHFEM_WITH_CATAMARI && MESHFEM_WITH_CHOLMOD && !defined(MESHFEM_USE_LEGACY_CATAMARI)
TEST_CASE("Catamari final permutation indexes reduced scalar variables", "[catamari][nd_ordering]") {
    const std::array<std::array<size_t, 2>, 4> edges{{{{0,3}}, {{1,2}}, {{2,4}}, {{3,4}}}};
    SystemAssembler<2> assembler(5);
    auto A = assembler.blockSparsityPattern(edges.size(), [&](size_t e) { return edges[e]; });
    for (bool single : {false, true}) for (bool blocks : {false, true})
    for (const auto &pins : {std::vector<size_t>{}, std::vector<size_t>{2,3}, std::vector<size_t>{3}}) {
        CAPTURE(single, blocks, pins);
        CatamariFactorizer f(false, single);
        f.orderingMethod = CatamariFactorizer::OrderingMethod::CholmodNesdisParallel;
        REQUIRE_THROWS(f.getInversePermutation());
        if (blocks) f.factorizeSymbolic(*A, pins);
        else        f.factorizeSymbolic(A->toScalar(true), pins);
        const auto inverse = f.getInversePermutation();
        REQUIRE(inverse.size() == Eigen::Index(f.n_reduced()));
        const Eigen::VectorXd reduced = Eigen::VectorXd::LinSpaced(f.n_reduced(), 0, f.n_reduced() - 1);
        Eigen::VectorXd full, permuted;
        f.extractFullSolution(reduced, full, false);
        f.removeFixedEntries(full, permuted, true);
        REQUIRE((inverse.cast<double>() - permuted).norm() == 0);
        auto sorted = inverse;
        std::sort(sorted.data(), sorted.data() + sorted.size());
        REQUIRE((sorted.cast<double>() - reduced).norm() == 0);
        f.clearFactors();
        REQUIRE_THROWS(f.getInversePermutation());
        REQUIRE((inverse.cast<double>() - permuted).norm() == 0); // Snapshot owns its storage.
    }
}
#endif
