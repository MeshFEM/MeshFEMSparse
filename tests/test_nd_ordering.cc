#include <MeshFEMSparse/Solvers/CatamariFactorizer.hh>
#include <MeshFEMSparse/Solvers/AccelerateFactorizer.hh>
#include <MeshFEMSparse/Solvers/cholmod_ordering.hh>
#include <MeshFEMSparse/SystemAssembler.hh>
#include <catch2/catch.hpp>
using namespace MeshFEM;

#if MESHFEM_WITH_CHOLMOD
namespace {
auto gridElements() {
    constexpr size_t side = 24;
    std::vector<std::array<size_t, 3>> e;
    for (size_t y=0; y+1<side; ++y) for (size_t x=0; x+1<side; ++x) {
        size_t a=y*side+x;
        e.push_back({{a,a+1,a+side}}); e.push_back({{a+1,a+side+1,a+side}});
    }
    return e;
}
template<class Factorizer, class Method>
void checkNDLifecycle(Factorizer &f, Method serial, Method parallel, Method nonND) {
    auto elements=gridElements();
    auto element=[&](size_t e){return elements[e];};
    SystemAssembler<2> assembler(24*24);
    auto A=assembler.blockSparsityPattern(elements.size(),element);
    A->setZero();
    for (size_t i=0;i<assembler.numVars();++i) A->addNZScalar(i,i,2.0);
    REQUIRE_FALSE(f.ndOrdering());
    for (auto method:{serial,parallel}) {
        f.orderingMethod=method;
        INFO("ND method: " << int(method));
        INFO("f.factorizeSymbolic(*A,{});");
        f.factorizeSymbolic(*A,{});
        REQUIRE(f.ndOrdering());
        const auto snapshot=*f.ndOrdering();
        REQUIRE(snapshot.blockSize==2);
        REQUIRE(snapshot.CMember.size()==24*24);
        REQUIRE(snapshot.CParent.size()>1);
        ElementPartitionFromND<SuiteSparse_long> p(elements.size(),element,snapshot.CParent,snapshot.CMember,3);
        p.validate(elements.size(),element);
        REQUIRE(p.numPartitions()>1);
        INFO("f.factorizeNumeric(*A);");
        f.factorizeNumeric(*A);
        REQUIRE(f.ndOrdering()->CParent==snapshot.CParent);
        REQUIRE(f.ndOrdering()->CMember==snapshot.CMember);
        INFO("f.factorizeNumericWithShift(*A,0.5);");
        f.factorizeNumericWithShift(*A,0.5);
        REQUIRE(f.ndOrdering()->CMember==snapshot.CMember);
        // The published indices describe the reduced ordering graph, not the full mesh.
        INFO("f.factorizeSymbolic(*A,{0,1});");
        f.factorizeSymbolic(*A,{0,1});
        REQUIRE(f.ndOrdering());
        REQUIRE(f.ndOrdering()->CMember.size()==24*24-1);
        REQUIRE(f.ndOrdering()->blockSize==2);
        REQUIRE(f.hasFixedVars());
        // A partial pin falls back to scalar analysis.
        INFO("f.factorizeSymbolic(*A,{0});");
        f.factorizeSymbolic(*A,{0});
        REQUIRE(f.ndOrdering());
        REQUIRE(f.ndOrdering()->CMember.size()==2*24*24-1);
        REQUIRE(f.ndOrdering()->blockSize==1);
        f.orderingMethod=nonND;
        INFO("f.factorizeSymbolic(*A,{});");
        f.factorizeSymbolic(*A,{});
        REQUIRE_FALSE(f.ndOrdering());
        REQUIRE(snapshot.CMember.size()==24*24);
        f.orderingMethod=method;
        INFO("f.factorizeSymbolic(*A,{});");
        f.factorizeSymbolic(*A,{});
        f.clearFactors();
        REQUIRE_FALSE(f.ndOrdering());
    }
}
}

TEST_CASE("Ordering helper retains raw ND trees for both index widths", "[nd_ordering]") {
    auto elements=gridElements();
    auto element=[&](size_t e){return elements[e];};
    SystemAssembler<1> assembler(24*24);
    auto A=assembler.blockSparsityPattern(elements.size(),element)->toScalar();
    A.Ax.assign(A.nz,1.0);
    CholmodOrdering ordering;
    std::optional<CholeskyFactorizerBase::NDOrdering> nd;
    for(auto method:{CholmodOrdering::Method::NestedDissection,CholmodOrdering::Method::ParallelNestedDissection}) {
        ordering.inversePermutation<int>(A,method,nullptr,nullptr,&nd);
        REQUIRE(nd);
        ElementPartitionFromND<SuiteSparse_long>(elements.size(),element,nd->CParent,nd->CMember,3).validate(elements.size(),element);
        ordering.inversePermutation<SuiteSparse_long>(A,method,nullptr,nullptr,&nd);
        REQUIRE(nd);
        REQUIRE(nd->CMember.size()==24*24);
        ordering.inversePermutation<int>(A,CholmodOrdering::Method::AMD,nullptr,nullptr,&nd);
        REQUIRE_FALSE(nd);
    }
}
#if MESHFEM_WITH_CATAMARI && !defined(MESHFEM_USE_LEGACY_CATAMARI)
TEST_CASE("Catamari ND metadata lifecycle", "[nd_ordering]") {
    CatamariFactorizer f;
    using O=CatamariFactorizer::OrderingMethod;
    checkNDLifecycle(f,O::CholmodNesdis,O::CholmodNesdisParallel,O::AMD);
}
#endif
#ifdef __APPLE__
TEST_CASE("Accelerate ND metadata lifecycle", "[nd_ordering]") {
    AccelerateFactorizer f;
    using O=AccelerateFactorizer::OrderingMethod;
    checkNDLifecycle(f,O::Nesdis,O::CholmodNesdisParallel,O::AMD);
}
#endif
#endif
