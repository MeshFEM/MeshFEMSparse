#ifndef CHOLMOD_ORDERING_HH
#define CHOLMOD_ORDERING_HH

#include "CholeskyFactorizerBase.hh"
#include <MeshFEMCore/GlobalBenchmark.hh>

#include <memory>
#include <stdexcept>
#include <type_traits>

#if MESHFEM_WITH_CHOLMOD
extern "C" {
#include <cholmod.h>
}
#endif

namespace MeshFEM {

class CholmodOrdering {
public:
    enum class Method { AMD, NestedDissection, Metis };

    ~CholmodOrdering() {
#if MESHFEM_WITH_CHOLMOD
        if (m_c)     cholmod_l_finish(m_c.get());
        if (m_c_int) cholmod_finish  (m_c_int.get());
#endif
    }

    template<class Index>
    VecX_T<Index> inversePermutation(const SuiteSparseMatrix &A, Method method) {
        static_assert(std::is_same_v<Index, SuiteSparse_long> || std::is_same_v<Index, int>,
                      "CholmodOrdering supports only SuiteSparse_long and int indices.");
#if MESHFEM_WITH_CHOLMOD
        if constexpr (std::is_same_v<Index, SuiteSparse_long>) return inversePermutationLong(A, method);
        else                                                   return inversePermutationInt (A, method);
#else
        (void) A;
        (void) method;
        throw std::runtime_error("CHOLMOD ordering requested, but CHOLMOD support is not available in this build.");
#endif
    }

private:
#if MESHFEM_WITH_CHOLMOD
    std::unique_ptr<cholmod_common> m_c, m_c_int;

    cholmod_common *commonLong() {
        if (!m_c) {
            m_c = std::make_unique<cholmod_common>();
            cholmod_l_start(m_c.get());
        }
        return m_c.get();
    }

    cholmod_common *commonInt() {
        if (!m_c_int) {
            m_c_int = std::make_unique<cholmod_common>();
            cholmod_start(m_c_int.get());
        }
        return m_c_int.get();
    }

    template<class Index_, class Ptr_>
    static cholmod_sparse sparseView(SuiteSparse_long m, SuiteSparse_long n, SuiteSparse_long nz,
                                     Index_ *Ai, Ptr_ *Ap) {
        cholmod_sparse A;
        A.nrow   = m;
        A.ncol   = n;
        A.nzmax  = nz;
        A.p      = Ap;
        A.i      = Ai;
        A.nz     = nullptr; /* not needed because `result` is packed. */
        A.z      = nullptr; /* not needed because `result` is real. */
        A.stype  = 1; // upper triangle stored.
        A.itype  = sizeof(Index_) == sizeof(SuiteSparse_long) ? CHOLMOD_LONG : CHOLMOD_INT;
        A.xtype  = CHOLMOD_PATTERN;
        A.dtype  = CHOLMOD_DOUBLE;
        A.sorted = true;
        A.packed = true;
        return A;
    }

    [[noreturn]] static void throwPartitionUnavailable() {
        throw std::runtime_error("CHOLMOD Partition support is not available in this build.");
    }

    VecX_T<SuiteSparse_long> inversePermutationLong(const SuiteSparseMatrix &A, Method method) {
        auto cholmat = sparseView(A.m, A.n, A.nz,
                                  const_cast<SuiteSparse_long *>(A.Ai.data()),
                                  const_cast<SuiteSparse_long *>(A.Ap.data()));

        VecX_T<SuiteSparse_long> iperm(A.m);
        switch (method) {
            case Method::AMD: {
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_l_amd");
                cholmod_l_amd(&cholmat, /* fset = */ nullptr, /* fsize = */ 0, iperm.data(), commonLong());
                break;
            }
            case Method::NestedDissection: {
#ifdef NPARTITION
                throwPartitionUnavailable();
#else
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_l_nested_dissection");
                VecX_T<SuiteSparse_long> CParent(A.m), CMember(A.m);
                cholmod_l_nested_dissection(&cholmat, /* fset = */ nullptr, /* fsize = */ 0,
                                            iperm.data(), CParent.data(), CMember.data(), commonLong());
#endif
                break;
            }
            case Method::Metis: {
#ifdef NPARTITION
                throwPartitionUnavailable();
#else
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_l_metis");
                cholmod_l_metis(&cholmat, /* fset = */ nullptr, /* fsize = */ 0, /* postorder = */ true,
                                iperm.data(), commonLong());
#endif
                break;
            }
        }

        return iperm;
    }

    VecX_T<int> inversePermutationInt(const SuiteSparseMatrix &A, Method method) {
        VecX_T<int> Ai_downcast = Eigen::Map<const VecX_T<SuiteSparse_long>>(A.Ai.data(), A.Ai.size()).template cast<int>();
        VecX_T<int> Ap_downcast = Eigen::Map<const VecX_T<SuiteSparse_long>>(A.Ap.data(), A.Ap.size()).template cast<int>();
        auto cholmat = sparseView(A.m, A.n, A.nz, Ai_downcast.data(), Ap_downcast.data());

        VecX_T<int> iperm(A.m);
        switch (method) {
            case Method::AMD: {
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_amd");
                cholmod_amd(&cholmat, /* fset = */ nullptr, /* fsize = */ 0, iperm.data(), commonInt());
                break;
            }
            case Method::NestedDissection: {
#ifdef NPARTITION
                throwPartitionUnavailable();
#else
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_nested_dissection");
                VecX_T<int> CParent(A.m), CMember(A.m);
                cholmod_nested_dissection(&cholmat, /* fset = */ nullptr, /* fsize = */ 0,
                                          iperm.data(), CParent.data(), CMember.data(), commonInt());
#endif
                break;
            }
            case Method::Metis: {
#ifdef NPARTITION
                throwPartitionUnavailable();
#else
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_metis");
                cholmod_metis(&cholmat, /* fset = */ nullptr, /* fsize = */ 0, /* postorder = */ true,
                              iperm.data(), commonInt());
#endif
                break;
            }
        }

        return iperm;
    }
#endif
};

} // namespace MeshFEM

#endif /* end of include guard: CHOLMOD_ORDERING_HH */
