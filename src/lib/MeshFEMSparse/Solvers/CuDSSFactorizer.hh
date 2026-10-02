////////////////////////////////////////////////////////////////////////////////
// CuDSSFactorizer.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Cholesky factorization on the GPU using NVIDIA's cuDSS library.
//  The phases map onto cuDSS's `cudssExecute` phases:
//      factorizeSymbolic  -> CUDSS_PHASE_ANALYSIS
//      factorizeNumeric   -> CUDSS_PHASE_FACTORIZATION
//      solveRawReduced    -> CUDSS_PHASE_SOLVE
*/
////////////////////////////////////////////////////////////////////////////////
#ifndef CUDSSFACTORIZER_HH
#define CUDSSFACTORIZER_HH

#include "CholeskyFactorizerBase.hh"
#include <cudss.h>
#include <stdexcept>
#include <vector>

namespace MeshFEM {

struct MESHFEM_EXPORT CuDSSFactorizer final : public CholeskyFactorizerBase {
    // Fill-reducing ordering computed by cuDSS during the analysis
    // (CUDSS_CONFIG_REORDERING_ALG); takes effect at the next factorizeSymbolic.
    enum class OrderingMethod { Default, AMD, NestedDissection };
    OrderingMethod orderingMethod = OrderingMethod::Default;

    CuDSSFactorizer();

    // *Scalar* size of the reduced (pin-removed) system.
    size_t m_reduced() const override { return m_reducedSize; }
    size_t n_reduced() const override { return m_reducedSize; }

    using CholeskyFactorizerBase::factorizeSymbolic; // Don't shadow
    using CholeskyFactorizerBase::factorizeNumeric;
    using CholeskyFactorizerBase::factorizeNumericWithShift;

    void factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) override;
    void factorizeNumeric(const SuiteSparseMatrix &fullMat, bool isInTryCatch=false) override;

    // Compute the numeric factorization of `A + sigma * B`, reusing the
    // symbolic factorization if it exists.
    void factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, const SuiteSparseMatrix &B, bool isInTryCatch=false) override;

    // Compute the numeric factorization of `A + sigma * I`, reusing the
    // symbolic factorization if it exists.
    void factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, bool isInTryCatch=false) override;

    // TODO: override the `BlockCSCHessianBase` versions (as in PardisoFactorizer)
    // if we want to exploit cuDSS's block support. Until then, the base class
    // converts block matrices to scalar ones.

    using CholeskyFactorizerBase::factorize; // Don't hide.
    void factorize(const SuiteSparseMatrix &mat, const std::vector<size_t> &fixedVars = std::vector<size_t>(), bool isInTryCatch = false) override {
        factorizeSymbolic(mat, fixedVars);
        factorizeNumeric(mat, isInTryCatch);
    }

    // Raw pointer version (Use with care! Caller must allocate/own both pointers)
    // `b` and `x` are host pointers to the reduced right-hand side and solution.
    void solveRawReduced(const Real *b, Real *x, CholeskySys sys = CholeskySys::A, bool alreadyPermuted = false) const override;

    bool preferInPlaceSolve() const override { return false; }
    bool supportsPrePermutation() const override { return false; }

    void        stashFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }
    bool   hasStashedFactorization() const override { throw std::runtime_error("Stashing unimplemented"); }
    void  swapStashedFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }
    void clearStashedFactorization()       override { throw std::runtime_error("Stashing unimplemented"); }

    void clearFactors() override;

    virtual CholeskyProvider provider() const override { return CholeskyProvider::CUDSS; }

    bool checkPosDef() const override { return m_factorizationType == FactorizationType::Numeric; }

    ~CuDSSFactorizer();
private:
    // cuDSS library objects (the counterpart of Pardiso's `pt`/`iparm`).
    // All four live as long as this object (created in the constructor,
    // destroyed in the destructor). The data object holds the ordering and
    // factors; each new symbolic factorization overwrites its contents.
    cudaStream_t  m_stream = nullptr; // All cuDSS work is queued on this stream.
    cudssHandle_t m_handle = nullptr;
    cudssConfig_t m_config = nullptr;
    cudssData_t   m_data   = nullptr;

    // The row/col-removed matrix handed to cuDSS; its arrays live on the GPU.
    // MeshFEM stores the upper triangle in CSC format. Read as CSR, the same
    // arrays describe the lower triangle, so we upload them unchanged and
    // tell cuDSS to use the lower-triangle view (no transpose needed).
    cudssMatrix_t A_cudss = nullptr;
    int  *m_rowStart_d   = nullptr; // CSR row offsets    (= CSC column pointers `Ap`)
    int  *m_colIndices_d = nullptr; // CSR column indices (= CSC row indices `Ai`)
    Real *m_values_d     = nullptr; // CSR values         (= CSC values `Ax`)

    // Right-hand side b and solution x (dense, one column) on the GPU, with
    // their wrappers. cudssExecute needs them in every phase, and the solve
    // copies into/out of them.
    cudssMatrix_t m_b = nullptr, m_x = nullptr;
    Real *m_b_values_d = nullptr, *m_x_values_d = nullptr;

    VecX_T<Real> m_shiftedValues; // Host staging buffer for the values of `A + sigma I`.

    size_t m_reducedSize = 0;
    SuiteSparse_long m_nnz = 0;

    void m_cudssRelease(); // Free the matrix wrapper and its device arrays (keeps `m_data`).
    void m_copyPatternToDevice(const SuiteSparseMatrix &A_reduced);            // Copy the sparsity pattern to the GPU and create `A_cudss`.
    void m_copyValuesToDevice (const SuiteSparseMatrix &A, Real sigma = 0.0); // Copy the values of `A + sigma I` to the GPU.
};

} // namespace MeshFEM

#endif /* end of include guard: CUDSSFACTORIZER_HH */
