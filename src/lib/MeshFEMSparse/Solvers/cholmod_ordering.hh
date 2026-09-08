#ifndef CHOLMOD_ORDERING_HH
#define CHOLMOD_ORDERING_HH

#include "CholeskyFactorizerBase.hh"
#include <MeshFEMCore/GlobalBenchmark.hh>

#include <memory>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if MESHFEM_WITH_CHOLMOD
#include "ParallelNestedDissection/cholmod_nesdis_parallel.hh"
extern "C" {
#include <cholmod.h>
}
#endif

namespace MeshFEM {

class CholmodOrdering {
public:
    enum class Method { AMD, NestedDissection, ParallelNestedDissection, Metis };

    void setNestedDissectionCompression(bool enabled) {
        m_ndCompress = enabled;
#if MESHFEM_WITH_CHOLMOD
        if (m_c)     m_c->method[0].nd_compress = enabled;
        if (m_c_int) m_c_int->method[0].nd_compress = enabled;
#endif
    }

    // Contiguous column groups in permutation order, with child-before-parent
    // dependencies. These are scheduling groups, not fundamental supernodes.
    template<class Index>
    struct PreliminaryAssemblyForest {
        std::vector<Index> sizes, parents;
    };

    ~CholmodOrdering() {
#if MESHFEM_WITH_CHOLMOD
        if (m_c)     cholmod_l_finish(m_c.get());
        if (m_c_int) cholmod_finish  (m_c_int.get());
#endif
    }

    // fullPattern optionally supplies the same graph with both triangles for
    // ParallelNestedDissection; other methods continue to use A's upper triangle.
    template<class Index>
    VecX_T<Index> inversePermutation(const SuiteSparseMatrix &A, Method method,
                                    PreliminaryAssemblyForest<Index> *forest = nullptr,
                                    const CSCMatrix<SuiteSparse_long, SuiteSparse_long> *fullPattern = nullptr) {
        static_assert(std::is_same_v<Index, SuiteSparse_long> || std::is_same_v<Index, int>,
                      "CholmodOrdering supports only SuiteSparse_long and int indices.");
        if (forest) *forest = {};
#if MESHFEM_WITH_CHOLMOD
        if (fullPattern && method == Method::ParallelNestedDissection) {
            if (fullPattern->m != A.m || fullPattern->n != A.n)
                throw std::invalid_argument("Full ordering pattern dimensions do not match the matrix.");
            return inversePermutationFullPattern<Index>(*fullPattern, forest);
        }
        if constexpr (std::is_same_v<Index, SuiteSparse_long>) return inversePermutationLong(A, method, forest);
        else                                                   return inversePermutationInt (A, method, forest);
#else
        (void) A;
        (void) method;
        (void) fullPattern;
        throw std::runtime_error("CHOLMOD ordering requested, but CHOLMOD support is not available in this build.");
#endif
    }

private:
    bool m_ndCompress = true;
#if MESHFEM_WITH_CHOLMOD
    std::unique_ptr<cholmod_common> m_c, m_c_int;

    void configure_common(cholmod_common *c) {
        c->method[0].nd_compress = m_ndCompress;
        // c->method[0].nd_components = false;
        // c->method[0].nd_small = 50;
        // c->method[0].nd_camd = 0;
        // c->method[0].prune_dense = -1;
    }

    cholmod_common *commonLong() {
        if (!m_c) {
            m_c = std::make_unique<cholmod_common>();
            cholmod_l_start(m_c.get());
            configure_common(m_c.get());
        }
        return m_c.get();
    }

    cholmod_common *commonInt() {
        if (!m_c_int) {
            m_c_int = std::make_unique<cholmod_common>();
            cholmod_start(m_c_int.get());
            configure_common(m_c_int.get());
        }
        return m_c_int.get();
    }

    template<class Index_, class Ptr_>
    static cholmod_sparse sparseView(SuiteSparse_long m, SuiteSparse_long n, SuiteSparse_long nz,
                                     Index_ *Ai, Ptr_ *Ap) {
        cholmod_sparse A{};
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

    template<class Index>
    VecX_T<Index> inversePermutationFullPattern(
        const CSCMatrix<SuiteSparse_long, SuiteSparse_long> &A,
        PreliminaryAssemblyForest<Index> *forest) {
#ifdef NPARTITION
        (void) A;
        (void) forest;
        throwPartitionUnavailable();
#else
        BENCHMARK_SCOPED_TIMER_SECTION timer("parallel nesdis from full graph");
        using Pattern = CSCMatrix<SuiteSparse_long, SuiteSparse_long>;
        if (A.m != A.n || A.n < 0 || A.nz < 0 ||
            A.symmetry_mode != Pattern::SymmetryMode::NONE ||
            A.Ap.size() != (size_t)A.n + 1 || A.Ai.size() != A.nz ||
            A.Ap.front() != 0 || A.Ap.back() != A.nz)
            throw std::invalid_argument("Invalid full ordering pattern.");
        if (A.n > std::numeric_limits<Index>::max() || A.nz > std::numeric_limits<Index>::max())
            throw std::overflow_error("Full ordering pattern exceeds the ordering index range.");

        VecX_T<Index> Ai_downcast, Ap_downcast;
        cholmod_sparse graph;
        cholmod_common *common;
        if constexpr (std::is_same_v<Index, SuiteSparse_long>) {
            graph = sparseView(A.m, A.n, A.nz,
                const_cast<SuiteSparse_long *>(A.Ai.data()), const_cast<SuiteSparse_long *>(A.Ap.data()));
            common = commonLong();
        }
        else {
            Ai_downcast = A.Ai.template cast<Index>();
            Ap_downcast = Eigen::Map<const VecX_T<SuiteSparse_long>>(A.Ap.data(), A.Ap.size()).template cast<Index>();
            graph = sparseView(A.m, A.n, A.nz, Ai_downcast.data(), Ap_downcast.data());
            common = commonInt();
        }
        graph.stype = 0;
        VecX_T<Index> perm(A.n), parent(A.n), member(A.n);
        auto nc = CholmodParallelNesdis::nested_dissection_from_graph<Index>(
            graph, perm.data(), parent.data(), member.data(), common);
        if (nc < 0) throw std::runtime_error("Parallel CHOLMOD nested dissection from full graph failed.");
        if (forest) *forest = extractAssemblyForest(perm, parent, member, nc);
        return perm;
#endif
    }

    template<class Index>
    static PreliminaryAssemblyForest<Index> extractAssemblyForest(
        const VecX_T<Index> &perm, const VecX_T<Index> &parent,
        const VecX_T<Index> &member, int64_t ncomponents) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("ND assembly forest extraction");
        PreliminaryAssemblyForest<Index> result;
        if (perm.size() == 0) return result;
        if (ncomponents <= 0 || ncomponents > parent.size())
            throw std::runtime_error("Invalid ND component count.");
        std::vector<Index> components, group_for_component(ncomponents, -1);
        // Hierarchical CAMD may reorder independent branches relative to the
        // component numbering; construct groups in the actual permutation order.
        for (Index k = 0; k < perm.size(); ++k) {
            Index c = member[perm[k]];
            if (c < 0 || c >= ncomponents) throw std::runtime_error("Invalid ND component.");
            if (components.empty() || components.back() != c) {
                if (group_for_component[c] >= 0)
                    throw std::runtime_error("ND component is not contiguous in the permutation.");
                group_for_component[c] = Index(components.size());
                components.push_back(c);
                result.sizes.push_back(0);
            }
            ++result.sizes.back();
        }
        for (Index c = 0; c < ncomponents; ++c)
            if (parent[c] < -1 || parent[c] >= ncomponents || (parent[c] >= 0 && parent[c] <= c))
                throw std::runtime_error("ND separator forest is not postordered.");
        result.parents.resize(components.size());
        for (size_t g = 0; g < components.size(); ++g) {
            Index c = parent[components[g]];
            while (c >= 0 && group_for_component[c] < 0) c = parent[c];
            Index p = c < 0 ? -1 : group_for_component[c];
            if (p >= 0 && (size_t)p <= g)
                throw std::runtime_error("ND separator precedes its descendants.");
            result.parents[g] = p;
        }
        return result;
    }

    VecX_T<SuiteSparse_long> inversePermutationLong(const SuiteSparseMatrix &A, Method method,
                                                   PreliminaryAssemblyForest<SuiteSparse_long> *forest) {
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
                auto nc = cholmod_l_nested_dissection(&cholmat, /* fset = */ nullptr, /* fsize = */ 0,
                                            iperm.data(), CParent.data(), CMember.data(), commonLong());
                if (nc < 0) throw std::runtime_error("CHOLMOD nested dissection failed.");
                if (forest) *forest = extractAssemblyForest(iperm, CParent, CMember, nc);
#endif
                break;
            }
            case Method::ParallelNestedDissection: {
#ifdef NPARTITION
                throwPartitionUnavailable();
#else
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_l_nested_dissection_parallel");
                VecX_T<SuiteSparse_long> CParent(A.m), CMember(A.m);
                auto nc = cholmod_l_nested_dissection_parallel(&cholmat, /* fset = */ nullptr, /* fsize = */ 0,
                                                     iperm.data(), CParent.data(), CMember.data(), commonLong());
                if (nc < 0)
                    throw std::runtime_error("Parallel CHOLMOD nested dissection failed.");
                if (forest) *forest = extractAssemblyForest(iperm, CParent, CMember, nc);
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
            default:
                throw std::runtime_error("cholmod_ordering: unrecognized method");
        }

        return iperm;
    }

    VecX_T<int> inversePermutationInt(const SuiteSparseMatrix &A, Method method,
                                     PreliminaryAssemblyForest<int> *forest) {
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
                auto nc = cholmod_nested_dissection(&cholmat, /* fset = */ nullptr, /* fsize = */ 0,
                                          iperm.data(), CParent.data(), CMember.data(), commonInt());
                if (nc < 0) throw std::runtime_error("CHOLMOD nested dissection failed.");
                if (forest) *forest = extractAssemblyForest(iperm, CParent, CMember, nc);
#endif
                break;
            }
            case Method::ParallelNestedDissection: {
#ifdef NPARTITION
                throwPartitionUnavailable();
#else
                BENCHMARK_SCOPED_TIMER_SECTION t("cholmod_nested_dissection_parallel");
                VecX_T<int> CParent(A.m), CMember(A.m);
                auto nc = cholmod_nested_dissection_parallel(&cholmat, /* fset = */ nullptr, /* fsize = */ 0,
                                                  iperm.data(), CParent.data(), CMember.data(), commonInt());
                if (nc < 0)
                    throw std::runtime_error("Parallel CHOLMOD nested dissection failed.");
                if (forest) *forest = extractAssemblyForest(iperm, CParent, CMember, nc);
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
            default:
                throw std::runtime_error("cholmod_ordering: unrecognized method");
        }

        return iperm;
    }
#endif
};

} // namespace MeshFEM

#endif /* end of include guard: CHOLMOD_ORDERING_HH */
