// Always listed in `add_library`; compiles to nothing unless cuDSS is enabled.
#if MESHFEM_WITH_CUDSS
#include "CuDSSFactorizer.hh"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <string>
#include <limits>

// Adapted from NVIDIA's cuDSS examples: throw instead of returning an error code.
// (GPU memory is freed by `m_cudssRelease`, called from the destructor.)
#define CUDA_CALL_AND_CHECK(call, msg)                                                   \
    do {                                                                                 \
        cudaError_t cuda_error = call;                                                   \
        if (cuda_error != cudaSuccess)                                                   \
            throw std::runtime_error("CUDA API returned error = "                        \
                + std::to_string(int(cuda_error)) + ", details: " msg);                       \
    } while (0)

#define CUDSS_CALL_AND_CHECK(call, status, msg)                                          \
    do {                                                                                 \
        status = call;                                                                   \
        if (status != CUDSS_STATUS_SUCCESS)                                              \
            throw std::runtime_error("CUDSS call ended unsuccessfully with status = "    \
                + std::to_string(int(status)) + ", details: " msg);                           \
    } while (0)

namespace MeshFEM {

CuDSSFactorizer::CuDSSFactorizer() {
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    /* Create a CUDA stream */
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&m_stream), "cudaStreamCreate");

    /* Creating the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssCreate(&m_handle), status, "cudssCreate");

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(m_handle, m_stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&m_config), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(m_handle, &m_data), status, "cudssDataCreate");
}

void CuDSSFactorizer::factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("CuDSS Symbolic Factorization");
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    /* Free the previous matrix, b and x (this also resets m_factorizationType to None) */
    m_cudssRelease();

    /* Remove the rows/columns of the pinned variables (done by the base class) */
    const SuiteSparseMatrix *A_reduced = m_initRowColRemoval(mat, pinnedVars);

    /* Copy the sparsity pattern to the GPU and create A_cudss */
    m_copyPatternToDevice(*A_reduced);

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices). */
    int64_t nrows = m_reducedSize, ncols = m_reducedSize;
    int     nrhs = 1;
    int64_t ldb = nrows, ldx = ncols;
    CUDA_CALL_AND_CHECK(cudaMalloc(&m_b_values_d, nrhs * nrows * sizeof(Real)), "cudaMalloc for b_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&m_x_values_d, nrhs * ncols * sizeof(Real)), "cudaMalloc for x_values");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&m_b, nrows, nrhs, ldb, m_b_values_d,
                                             CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&m_x, ncols, nrhs, ldx, m_x_values_d,
                                             CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for x");

    /* Symbolic factorization (synchronous: returns once the analysis is done) */
    CUDSS_CALL_AND_CHECK(cudssExecute(m_handle, CUDSS_PHASE_ANALYSIS, m_config, m_data, A_cudss, m_x, m_b),
                         status, "cudssExecute for analysis");

    m_factorizationType = FactorizationType::Symbolic;
}

void CuDSSFactorizer::factorizeNumeric(const SuiteSparseMatrix &fullMat, bool isInTryCatch) {
    factorizeNumericWithShift(fullMat, /* sigma = */ 0.0, isInTryCatch);
}

void CuDSSFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix & /* A */, Real /* sigma */, const SuiteSparseMatrix & /* B */, bool /* isInTryCatch */) {
    // Like Pardiso and Accelerate, `A + sigma * B` is not supported yet.
    throw std::runtime_error("CuDSSFactorizer::factorizeNumericWithShift with B not yet implemented");
}

// `isInTryCatch` only suppresses CHOLMOD's own error printing; we print nothing, so it is unused.
void CuDSSFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, bool /* isInTryCatch */) {
    assertFactorization(FactorizationType::Symbolic);
    BENCHMARK_SCOPED_TIMER_SECTION timer("CuDSS Numeric Factorization");
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    // The previous numeric factorization is invalid from here on (until this one succeeds).
    invalidateNumericFactorization();

    /* Remove the pinned rows/columns from the values (done by the base class) */
    const SuiteSparseMatrix *A_reduced = m_rowColRemoval(A);

    /* Copy the values of A + sigma I to the GPU */
    m_copyValuesToDevice(*A_reduced, sigma);

    /* Clear a failure flag left by a previous (failed) factorization: cuDSS keeps
       CUDSS_DATA_INFO set (even across a new analysis) and then skips the
       factorization, which would break the Newton solver's shift-and-retry loop. */
    int info = 0;
    CUDSS_CALL_AND_CHECK(cudssDataSet(m_handle, m_data, CUDSS_DATA_INFO, &info, sizeof(info)),
                         status, "cudssDataSet for info");

    /* Factorization (asynchronous: only queued on m_stream) */
    CUDSS_CALL_AND_CHECK(cudssExecute(m_handle, CUDSS_PHASE_FACTORIZATION, m_config, m_data, A_cudss, m_x, m_b),
                         status, "cudssExecute for factor");

    /* Wait for the factorization to finish before checking its result */
    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(m_stream), "cudaStreamSynchronize after factorization");

    /* Check for a zero/negative pivot, i.e., a matrix that is not positive definite */
    size_t sizeWritten = 0;
    CUDSS_CALL_AND_CHECK(cudssDataGet(m_handle, m_data, CUDSS_DATA_INFO, &info, sizeof(info), &sizeWritten),
                         status, "cudssDataGet for info");
    if (info != 0)
        throw std::runtime_error("[cuDSS] Numeric factorization failed (matrix not positive definite?), info = " + std::to_string(info));

    m_factorizationType = FactorizationType::Numeric;
}

// `b` and `x` are host (CPU) arrays of size `m_reducedSize`.
void CuDSSFactorizer::solveRawReduced(const Real *b, Real *x, CholeskySys sys, bool alreadyPermuted) const {
    assertFactorization(sys);
    BENCHMARK_SCOPED_TIMER_SECTION timer("CuDSS Solve");
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    // cuDSS solves with the full A (not with L, L^T or P separately), and we
    // report supportsPrePermutation() == false, so we never get pre-permuted input.
    if (sys != CholeskySys::A) throw std::invalid_argument("[cuDSS] Only CholeskySys::A solves are supported");
    if (alreadyPermuted)       throw std::invalid_argument("[cuDSS] Pre-permuted solves are not supported");

    /* Check that b and x (created by factorizeSymbolic) still match the system size */
    int64_t nrows_b = 0, nrows_x = 0, ncols, ld;
    void *values;
    cudssDataType_t type;
    cudssLayout_t   layout;
    CUDSS_CALL_AND_CHECK(cudssMatrixGetDn(m_b, &nrows_b, &ncols, &ld, &values, &type, &layout), status, "cudssMatrixGetDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixGetDn(m_x, &nrows_x, &ncols, &ld, &values, &type, &layout), status, "cudssMatrixGetDn for x");
    if ((size_t(nrows_b) != m_reducedSize) || (size_t(nrows_x) != m_reducedSize))
        throw std::logic_error("[cuDSS] b/x size does not match the factorized system; was factorizeSymbolic skipped?");

    /* Copy host memory to device for b */
    CUDA_CALL_AND_CHECK(cudaMemcpy(m_b_values_d, b, m_reducedSize * sizeof(Real), cudaMemcpyHostToDevice),
                        "cudaMemcpy for b_values");

    /* Solving */
    CUDSS_CALL_AND_CHECK(cudssExecute(m_handle, CUDSS_PHASE_SOLVE, m_config, m_data, A_cudss, m_x, m_b),
                         status, "cudssExecute for solve");

    /* Wait for the solve to finish before reading x */
    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(m_stream), "cudaStreamSynchronize after solve");

    /* Copy the solution back to the host */
    CUDA_CALL_AND_CHECK(cudaMemcpy(x, m_x_values_d, m_reducedSize * sizeof(Real), cudaMemcpyDeviceToHost),
                        "cudaMemcpy for x_values");
}

void CuDSSFactorizer::clearFactors() {
}

// Free the matrix wrapper and its device arrays (the counterpart of CUDSS_EXAMPLE_FREE).
// `m_data` is kept: it lives as long as this object and is freed in the destructor.
// Also called from the destructor, so it must not throw: errors are ignored.
void CuDSSFactorizer::m_cudssRelease() {
    // Destroy the wrappers before the arrays they point to.
    if (A_cudss) cudssMatrixDestroy(A_cudss);
    if (m_b)     cudssMatrixDestroy(m_b);
    if (m_x)     cudssMatrixDestroy(m_x);
    cudaFree(m_rowStart_d);                   // cudaFree(nullptr) is a no-op.
    cudaFree(m_colIndices_d);
    cudaFree(m_values_d);
    cudaFree(m_b_values_d);
    cudaFree(m_x_values_d);

    // Reset so a second release (or the destructor after a release) doesn't double-free.
    A_cudss         = nullptr;
    m_b             = nullptr;
    m_x             = nullptr;
    m_b_values_d    = nullptr;
    m_x_values_d    = nullptr;
    m_rowStart_d    = nullptr;
    m_colIndices_d  = nullptr;
    m_values_d      = nullptr;
    m_nnz           = 0;
    m_reducedSize   = 0;

    m_factorizationType = FactorizationType::None;
}

// Copy the sparsity pattern of `A` to the GPU, allocate space for its values,
// and wrap both in `A_cudss`. The values are copied later by `m_copyValuesToDevice`.
// Expects `m_cudssRelease()` to have been called first (so nothing is leaked).
void CuDSSFactorizer::m_copyPatternToDevice(const SuiteSparseMatrix &A) {
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    if (A_cudss) throw std::logic_error("[cuDSS] m_copyPatternToDevice called without releasing the previous matrix");
    if (A.m != A.n) throw std::runtime_error("[cuDSS] Matrix must be square");
    // Upper triangle in CSC == lower triangle in CSR, which is what we tell cuDSS below.
    if (A.symmetry_mode != SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE)
        throw std::runtime_error("[cuDSS] Expected an upper-triangle matrix");
    if (A.nz > std::numeric_limits<int>::max()) throw std::runtime_error("[cuDSS] Too many nonzeros for 32-bit indices");

    m_reducedSize = A.m;
    m_nnz         = A.nz;
    const int64_t n = m_reducedSize, nnz = m_nnz;

    /* Convert the (64-bit) SuiteSparseMatrix indices to the 32-bit ones used in the example */
    std::vector<int> csr_offsets_h(A.Ap.begin(), A.Ap.end());            // CSC column pointers
    std::vector<int> csr_columns_h(A.Ai.data(), A.Ai.data() + A.nz);    // CSC row indices

    /* Allocate device memory for A */
    CUDA_CALL_AND_CHECK(cudaMalloc(&m_rowStart_d,   (n + 1) * sizeof(int)), "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&m_colIndices_d, nnz * sizeof(int)),     "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&m_values_d,     nnz * sizeof(Real)),    "cudaMalloc for csr_values");

    /* Copy host memory to device for A (the values are copied by m_copyValuesToDevice) */
    CUDA_CALL_AND_CHECK(cudaMemcpy(m_rowStart_d,   csr_offsets_h.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(m_colIndices_d, csr_columns_h.data(), nnz * sizeof(int),     cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrixType_t     mtype = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_LOWER;
    cudssIndexBase_t      base  = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A_cudss, n, n, nnz, m_rowStart_d, NULL,
                                              m_colIndices_d, m_values_d, CUDSS_R_32I,
                                              CUDSS_R_32I, CUDSS_R_64F, mtype, mview,
                                              base),
                         status, "cudssMatrixCreateCsr");
}

void CuDSSFactorizer::m_copyValuesToDevice(const SuiteSparseMatrix &  A , Real  sigma ) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("CuDSSFactorizer.m_copyValuesToDevice");
    // m_values_d was allocated for m_nnz values; copying more would overrun it.
    if (A.nz != m_nnz) throw std::runtime_error("[cuDSS] Sparsity pattern changed since the symbolic factorization");

    const Real *src = A.Ax.data();          // sigma == 0: upload A's values directly

    if (sigma != 0) {
        m_shiftedValues.resize(A.nz);
        // One pass over columns: copy each column's entries and shift its diagonal.
        // Each column's entries belong to exactly one iteration, so there are no races.
        tbb::parallel_for(tbb::blocked_range<SuiteSparse_long>(0, A.n),
            [&](const tbb::blocked_range<SuiteSparse_long> &r) {
                for (SuiteSparse_long j = r.begin(); j < r.end(); ++j) {
                    for (SuiteSparse_long ii = A.Ap[j]; ii < A.Ap[j + 1]; ++ii)
                        m_shiftedValues[ii] = A.Ax[ii];
                    m_shiftedValues[A.findDiagEntry(j)] += sigma;
                }
            });
        src = m_shiftedValues.data();
    }

    // CPU → GPU. `A_cudss` points at m_values_d, so it sees the new values.
    CUDA_CALL_AND_CHECK(cudaMemcpy(m_values_d, src, A.nz * sizeof(Real), cudaMemcpyHostToDevice), "cudaMemcpy for values");

}

CuDSSFactorizer::~CuDSSFactorizer() {
    m_cudssRelease();
    if (m_data)   cudssDataDestroy(m_handle, m_data); // Needs m_handle, so it must come before cudssDestroy.
    if (m_config) cudssConfigDestroy(m_config);
    if (m_handle) cudssDestroy(m_handle);
    if (m_stream) cudaStreamDestroy(m_stream); // After the handle that uses it.
}

} // namespace MeshFEM

#endif // MESHFEM_WITH_CUDSS
