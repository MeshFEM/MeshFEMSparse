////////////////////////////////////////////////////////////////////////////////
// ScalarCSCView.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Type-erased view of a scalar CSC sparse matrix.
*///////////////////////////////////////////////////////////////////////////////
#ifndef SCALARCSCVIEW_HH
#define SCALARCSCVIEW_HH

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <MeshFEMSparse/SparseMatrices.hh>
#include <Eigen/Sparse>

namespace MeshFEM {

struct ScalarCSCView {
    enum class IndexTypeEnum { Int32, SuiteSparseLong };

    const void *Ap = nullptr;
    const void *Ai = nullptr;
    const double *Ax = nullptr;
    size_t rows = 0;
    size_t cols = 0;
    size_t nnz = 0;
    IndexTypeEnum indexType = IndexTypeEnum::SuiteSparseLong;
	bool isSparsityOnly() const { return Ax == nullptr; }

    static ScalarCSCView from(const SuiteSparseMatrix &A) {
        return ScalarCSCView{ A.Ap.data(), A.Ai.data(), A.Ax.data(), size_t(A.m), size_t(A.n), A.nnz(), IndexTypeEnum::SuiteSparseLong };
    }

	template<typename _Index>
    static ScalarCSCView from(const Eigen::SparseMatrix<double, Eigen::ColMajor, _Index> &A) {
        return ScalarCSCView{ A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(), size_t(A.rows()), size_t(A.cols()), size_t(A.nonZeros()), std::is_same_v<_Index, int32_t> ? IndexTypeEnum::Int32 : IndexTypeEnum::SuiteSparseLong };
    }

    template<class F>
    void visit(F &&f) const {
        switch (indexType) {
            case IndexTypeEnum::Int32:
                std::forward<F>(f)(static_cast<const int32_t *>(Ap), static_cast<const int32_t *>(Ai), Ax);
                return;
            case IndexTypeEnum::SuiteSparseLong:
                std::forward<F>(f)(static_cast<const SuiteSparse_long *>(Ap), static_cast<const SuiteSparse_long *>(Ai), Ax);
                return;
        }
        throw std::runtime_error("ScalarCSCView: unsupported index type");
    }
};

} // namespace MeshFEM

#endif /* end of include guard: SCALARCSCVIEW_HH */
