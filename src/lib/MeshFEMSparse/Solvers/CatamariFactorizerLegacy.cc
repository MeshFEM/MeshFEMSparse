#ifndef MESHFEM_USE_LEGACY_CATAMARI
#error CatamariFactorizerLegacy.cc should be included only when MESHFEM_USE_LEGACY_CATAMARI is set!
#endif

#include "CatamariFactorizer.hh"
#include <MeshFEMCore/GlobalBenchmark.hh>

#include "CholmodFactorizer.hh"
#include "amd.h"

#include <catamari/apply_sparse.hpp>
#include <catamari/blas_matrix.hpp>
#include <catamari/norms.hpp>
#include <catamari/sparse_ldl.hpp>
#include <specify.hpp>

#if MESHFEM_WITH_SCOTCH
#include "ScotchOrdering.hh"
#endif

#include "accelerate_ordering.hh"
#include "pardiso_ordering.hh"

#if CATAMARI_FINEGRAINED_TIMERS
#include <filesystem>
#endif

#include "CatamariConverter.hh"

namespace MeshFEM {

CatamariFactorizer::CatamariFactorizer(bool legacy) : CatamariFactorizer(legacy, false) { }

CatamariFactorizer::CatamariFactorizer(bool legacy, bool singlePrecision) {
    if (singlePrecision) throw std::invalid_argument("Single precision is unavailable for legacy Catamari");
    m_legacy = true;
    m_ldl        = std::make_unique<catamari::SparseLDL<double>>();
    m_ldlControl = std::make_unique<catamari::SparseLDLControl<double>>();
    m_ldlControl->SetFactorizationType(catamari::kCholeskyFactorization);
    m_ldlControl->supernodal_strategy = catamari::kSupernodalFactorization;
    m_ldlControl->supernodal_control.algorithm = catamari::kRightLookingLDL; // catamari::kRightLookingLDL;
    m_ldlControl->supernodal_control.relaxation_control.relax_supernodes = true;
    // m_ldlControl->supernodal_control.factor_tile_size = std::numeric_limits<catamari::Int>::max(); // Effectively disable node-level parallelism
}

void CatamariFactorizer::setUseLeftLooking(bool use_left) { m_ldlControl->supernodal_control.algorithm = use_left ? catamari::kLeftLookingLDL : catamari::kRightLookingLDL; }
bool CatamariFactorizer::getUseLeftLooking() const { return m_ldlControl->supernodal_control.algorithm == catamari::kLeftLookingLDL; }

size_t CatamariFactorizer::m_reduced() const { assertFactorization(FactorizationType::Symbolic); return m_ldl->NumRows(); }
size_t CatamariFactorizer::n_reduced() const { assertFactorization(FactorizationType::Symbolic); return m_ldl->NumRows(); }

void CatamariFactorizer::factorizeSymbolic(const BlockCSCHessianBase &mat, const std::vector<size_t> &pinnedVars) {
    g_matrixRecorder.recordSymbolic(mat, pinnedVars);

    const bool blockFactorizationSupported = m_useBlockAccel && mat.uniformBlockSize() && (mat.maxBlockSize() <= MAX_INSTANTIATED_BLOCK_SIZE);
    m_blockSize = 1;
    if (mat.maxBlockSize() == 1) m_factorizeSymbolic((const SuiteSparseMatrix &) mat, pinnedVars); // Already scalar
    else {
        m_scalarHessian = mat.toScalar(/* sparsityOnly = */ true);
        m_dataOffsetForScalarHessianLoc = mat.dataOffsetsForScalarCSCDataOffsets(m_scalarHessian);
        m_blockSize = 1;
        m_factorizeSymbolic(m_scalarHessian, pinnedVars);
    }
}

void CatamariFactorizer::factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) {
    m_blockSize = 1;
    m_factorizeSymbolic(mat, pinnedVars);
}

// When `m_blockSize > 1` then `mat` holds the block sparsity pattern with
// uniform block size `m_blockSize`.
// `pinnedVars` always holds scalar variables.
void CatamariFactorizer::m_factorizeSymbolic(const SuiteSparseMatrix &mat, const std::vector<size_t> &pinnedVars) {
    m_prepareTemporalReuse(pinnedVars);
    const SuiteSparseMatrix *A_reduced;
    std::vector<SuiteSparse_long> reducedRowForRow_block;
    std::vector<SuiteSparse_long> blockEntryForReducedBlockEntry; // the original block nz corresponding to each nz in the block row-col-removed matrix

    A_reduced = m_initRowColRemoval(mat, pinnedVars);
    reducedRowForRow_block = m_reducedRowForRow;

    m_permutedReducedRowForRow.clear(); // The upcoming symbolic factorization will change any existing permutation...

    BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Symbolic Factorize");
    if (m_catamariConverter) {
        BENCHMARK_SCOPED_TIMER_SECTION t2("CatamariConverter_reset");
        m_catamariConverter.reset();
    }
    m_catamariConverter = std::make_unique<CatamariConverter>(*A_reduced, /* block_size = */ 1, /* legacy = */ true, m_entryForReducedEntry);
    using catamari::Int;

    if (orderingMethod == OrderingMethod::Catamari)
        m_ldl->Factor(m_catamariConverter->get(), *m_ldlControl);
    else if ((orderingMethod == OrderingMethod::CholmodNesdis) || (orderingMethod == OrderingMethod::CholmodNesdisParallel) || (orderingMethod == OrderingMethod::Metis)
          || (orderingMethod == OrderingMethod::AMD) || (orderingMethod == OrderingMethod::Adaptive)) {

        OrderingMethod actualOrderingMethod = orderingMethod;
        if (orderingMethod == OrderingMethod::Adaptive) {
            actualOrderingMethod = adaptiveOrdering.updateSelection();
            // std::cout << "Adaptive ordering method selected: "
            //           << ((actualOrderingMethod == OrderingMethod::CholmodNesdis) ? "Nesdis" : "AMD")
            //           << "  " << adaptiveOrdering.factorizationTimingDescription();
            // std::cout << std::endl;
        }

        // Time the ordering-dependent parts of the symbolic factorization
        auto sym_fact_start = std::chrono::steady_clock::now();

        catamari::SymmetricOrdering ordering;
        {
            ordering.inverse_permutation.Resize(A_reduced->m);
            if ((actualOrderingMethod == OrderingMethod::CholmodNesdis) ||
                (actualOrderingMethod == OrderingMethod::CholmodNesdisParallel) ||
                (actualOrderingMethod == OrderingMethod::Metis)) {
                auto method = actualOrderingMethod == OrderingMethod::Metis ? CholmodOrdering::Method::Metis :
                    actualOrderingMethod == OrderingMethod::CholmodNesdisParallel ? CholmodOrdering::Method::ParallelNestedDissection :
                    CholmodOrdering::Method::NestedDissection;
                auto iperm = m_cholmodOrdering.inversePermutation<catamari::Int>(*A_reduced, method);
                std::copy(iperm.data(), iperm.data() + iperm.size(), ordering.inverse_permutation.Data());
                quotient::InvertPermutation(ordering.inverse_permutation, &ordering.permutation);
            }
            else if (actualOrderingMethod == OrderingMethod::AMD) {
                BENCHMARK_SCOPED_TIMER_SECTION t("AMD ordering");
                using ordering_index_type = int32_t;
                const ordering_index_type n = A_reduced->m;

                const auto &A = m_catamariConverter->get();

#if 0 // Validation
                {
                    Eigen::VectorXi Ap(n + 1), Ai;

                    for (ordering_index_type j = 0; j <= n; ++j)
                        Ap[j] = A.RowEntryOffset(j);

                    catamari::Int nz = A.NumEntries();
                    Ai.resize(nz);
                    for (catamari::Int ii = 0; ii < nz; ++ii)
                        Ai[ii] = A.Entry(ii).column; // Catamari matrix is transposed!

                    std::cout << "amd_valid: " << amd_valid(n, n, Ap.data(), Ai.data()) << std::endl;
                }
#endif

                // TODO: verify symmetry? This probably isn't checked by amd_valid

                // AMD_2 is passed only the off-diagonal entries of the *full* matrix (i.e., both upper and lower triangles).
                // Furthermore, it needs some additional "elbow room" in the
                // the row index array (the `cholmod_amd` wrapper allocates around 50%).
                ordering_index_type padded_input_matrix_size = (A.NumEntries() - n) * 1.5;

                VecX_T<ordering_index_type> Pe(n + 1), Nv(n), workspace;
                workspace.resize(7 * n + padded_input_matrix_size);

                ordering_index_type *Degree = workspace.data(),
                                    *Wi     = workspace.data() + n,
                                    *Len    = workspace.data() + 2 * n,
                                    *Elen   = workspace.data() + 3 * n,
                                    *Head   = workspace.data() + 4 * n,
                                    * perm  = workspace.data() + 5 * n,
                                    *iperm  = workspace.data() + 6 * n,
                                    *Iw     = workspace.data() + 7 * n; // length `padded_input_matrix_size`

                tbb::parallel_for(tbb::blocked_range<ordering_index_type>(0, n), [&](const tbb::blocked_range<ordering_index_type> &r) {
                    for (ordering_index_type j = r.begin(); j < r.end(); ++j) {
                        auto col_start = A.RowEntryOffset(j);
                        auto col_end   = A.RowEntryOffset(j + 1);
                        Len[j] = ordering_index_type(col_end - col_start - 1); // diagonal is excluded
                        Pe[j] = col_start - j; // diagonal is excluded
                        ordering_index_type back = Pe[j];
                        for (ordering_index_type ii = col_start; ii < col_end; ++ii) {
                            ordering_index_type i = A.Entry(ii).column; // Catamari matrix is transposed!
                            if (i != j) Iw[back++] = i;
                        }
                    }
                });
                Pe[n] = Pe[n - 1] + Len[n - 1];

                // for (ordering_index_type j = 0; j < n; ++j) {
                //     if (Pe[j + 1] - Pe[j] != Len[j]) throw std::logic_error("column pointer error");
                //     if (Len[j] <= 0) throw std::logic_error("Empty column");
                // }
                // if (A.NumEntries() - n != Pe[n]) throw std::logic_error("inconsistent nnz count");


                {
                    double Control[AMD_CONTROL];
                    amd_defaults(Control);
                    Control[AMD_DENSE] = -1; // Disable dense-node pruning; this could slow down the ordering, but ensures we get a valid assembly tree (e.g., passing the sanity check below).
                    // TODO: support the sparse assembly forest + dense border produced if we let amd_2's dense row detection kick in?

                    double Info [AMD_INFO];
                    BENCHMARK_SCOPED_TIMER_SECTION t2("amd_2");
                    // amd_l2(n, Pe.data(), Iw, Len, padded_input_matrix_size, Pe[n], Nv.data(), iperm, perm, Head, Elen,
                    //        Degree, Wi, Control, Info);
                    amd_2(n, Pe.data(), Iw, Len, padded_input_matrix_size, Pe[n], Nv.data(), iperm, perm, Head, Elen,
                          Degree, Wi, Control, Info);

                    // for (int i = 0; i < AMD_INFO; ++i)
                    //     std::cout << "Info[" << i << "] = " << Info[i] << std::endl;
                }

                if (Nv.cwiseMax(0).sum() != n) throw std::logic_error("amd_2 failed sanity check: Nv doesn't sum to n (" + std::to_string(Nv.cwiseMax(0).sum()) + " vs " + std::to_string(n) + ")");

                ordering.permutation.Resize(n);
                tbb::parallel_for(tbb::blocked_range<ordering_index_type>(0, n), [&](const tbb::blocked_range<ordering_index_type> &r) {
                    for (ordering_index_type j = r.begin(); j < r.end(); ++j) {
                        // Note that SuiteSparse and Catamari disagree on which
                        // permutation they call the "inverse" one.
                        // In Catamari, `permutation[j_orig]` gives the
                        // column index in the permuted matrix where column
                        // `j_orig` of the original matrix ends up.
                        ordering.inverse_permutation[j] =  perm[j];
                        ordering.permutation[j]         = iperm[j];
                    }
                });

#if 1
                // Extract preliminary supernode information and assembly tree from
                // the AMD output; this is needed to parallelize symbolic
                // factorization in Catamari.
                // (Note that the assembly tree created by AMD
                // is generally different from the supernodal assembly tree
                // consisting of fundamental supernodes).
                {

                    // Record which supernode contains column `j` of L.
                    // Note that only the entries corresponding to the
                    // "representative column" of each supernode are populated.
                    // By this, we mean the root of the "subtree" within
                    // each node of the assembly tree.
                    // In terms of the AMD output, these are the indices for
                    // which `Nv` is nonzero, and are the *last* column indices
                    // of each supernode.
                    VecX_T<Int> supernode_index(n);

                    // Determine supernodes and sizes
                    Int num_supernodes = (Nv.array() > 0).count();
                    ordering.supernode_sizes.Resize(num_supernodes);
                    for (Int s = 0, j_perm = 0; j_perm < Int(n); ++j_perm) { // Loop through columns of L
                        Int size = Nv[ordering.inverse_permutation[j_perm]];
                        if (size > 0) {
                            supernode_index[j_perm] = s;
                            ordering.supernode_sizes[s++] = size;
                        }
                    }

                    OffsetScan(ordering.supernode_sizes, &ordering.supernode_offsets);
                    if (ordering.supernode_offsets[num_supernodes] != n)
                        throw std::logic_error("Bad amd_2 result: supernodes fail to span all columns");

                    // Convert the assembly tree from AMD's `Pe` array into `ordering.assembly_forest.parents`.
                    // Note that, when `j` is the start of a supernode, `Pe[j]`
                    // holds the parent index of column `j` where all indices
                    // here are in the *original* matrix.
                    ordering.assembly_forest.parents.Resize(num_supernodes);
                    for (Int s = 0; s < num_supernodes; ++s) {
                        // Note: the "representative column" is the *last* column of the supernode.
                        Int representative_col = ordering.inverse_permutation[ordering.supernode_offsets[s + 1] - 1];
                        assert(Nv[representative_col] > 0 && "Failed to find supernode's 'representative'/'root' column");
                        Int parent_repcol_orig = Pe[representative_col];
                        if (parent_repcol_orig < 0) { ordering.assembly_forest.parents[s] = -1; continue; }
                        Int parent_representative_col = ordering.permutation[parent_repcol_orig];
                        assert(Nv[ordering.inverse_permutation[parent_representative_col]] > 0 && "Pe did not return a representative col.");
                        ordering.assembly_forest.parents[s] = supernode_index[parent_representative_col];
                    }

                    ordering.assembly_forest.FillFromParents();
                }
#endif
#if 0
                {
                    std::cout << "Nv: " << Nv.head(20).transpose() << std::endl;
                    std::cout << "Pe: " << Pe.head(20).transpose() << std::endl;

                    // Pe[j] and Nv[j] hold data for the original column `j`.
                    // We need arrays that are permuted to correspond to the
                    // lower factor L.
                    VecX_T<ordering_index_type> Pe_perm(n + 1), Nv_perm(n);
                    for (ordering_index_type j = 0; j < n; ++j) {
                        ordering_index_type j_orig = ordering.inverse_permutation[j];
                        Pe_perm[j] = ordering.permutation[Pe[j_orig]];
                        Nv_perm[j] = Nv[j_orig];
                    }

                    std::cout << "n: " << n << std::endl;
                    int argmin;
                    std::cout << "Nv.min(): " << Nv_perm.minCoeff(&argmin) << std::endl;
                    std::cout << "Nv.max(): " << Nv_perm.maxCoeff() << std::endl;
                    std::cout << "Nv.sum(): " << Nv_perm.sum() << std::endl;
                    std::cout << "Permuted Nv: " << Nv_perm.segment(0, 40).transpose() << std::endl;
                    std::cout << "Permuted Pe: " << Pe_perm.segment(0, 40).transpose() << std::endl;

                    std::ofstream("Nv_perm.txt") << Nv_perm << std::endl;
                    std::ofstream("Pe_perm.txt") << Pe_perm << std::endl;
                }
#endif
            }
            else throw std::runtime_error("Unknown orderingMethod");

        }

        {
            BENCHMARK_SCOPED_TIMER_SECTION t2("Catamari Factor Call with Ordering");
            m_ldl->Factor(m_catamariConverter->get(), ordering, *m_ldlControl);
        }

        double sym_fact_duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - sym_fact_start).count();
        if (orderingMethod == OrderingMethod::Adaptive)
            adaptiveOrdering.recordSymbolic(sym_fact_duration);
    }
    else if (orderingMethod == OrderingMethod::AccelerateMetis) {
        auto perm = compute_accelerate_ordering(*A_reduced);
        catamari::SymmetricOrdering ordering;
        ordering.permutation        .Resize(A_reduced->m);
        ordering.inverse_permutation.Resize(A_reduced->m);
        for (int i = 0; i < A_reduced->m; ++i)
            ordering.permutation[i] = perm[i]; // Accelerate's permutation convention is the same as ours
        quotient::InvertPermutation(ordering.permutation, &ordering.inverse_permutation);
        m_ldl->Factor(m_catamariConverter->get(), ordering, *m_ldlControl);
    }
    else if ((orderingMethod == OrderingMethod::PardisoMetis) || (orderingMethod == OrderingMethod::PardisoParallelMetis)) {
        auto perm = compute_pardiso_ordering(*A_reduced,  (orderingMethod == OrderingMethod::PardisoMetis) ? PardisoSparseOrder::Metis : PardisoSparseOrder::ParallelMetis);
        catamari::SymmetricOrdering ordering;
        ordering.permutation        .Resize(A_reduced->m);
        ordering.inverse_permutation.Resize(A_reduced->m);
        for (int i = 0; i < A_reduced->m; ++i)
            ordering.inverse_permutation[i] = perm[i]; // Pardiso's permutation convention is the inverse of ours
        quotient::InvertPermutation(ordering.inverse_permutation, &ordering.permutation);
        m_ldl->Factor(m_catamariConverter->get(), ordering, *m_ldlControl);
    }
    else if (orderingMethod == OrderingMethod::Scotch) {
#if MESHFEM_WITH_SCOTCH
        catamari::SymmetricOrdering ordering;
        ordering.permutation        .Resize(A_reduced->m);
        ordering.inverse_permutation.Resize(A_reduced->m);

        Eigen::Map<VecX_T<catamari::Int>> perm(ordering.permutation.Data(), A_reduced->m);
        Eigen::Map<VecX_T<catamari::Int>> iperm(ordering.inverse_permutation.Data(), A_reduced->m);

        scotch_ordering(*A_reduced, perm, iperm, scotchSettings.stratFlag, scotchSettings.imbalanceRatio);

        m_ldl->Factor(m_catamariConverter->get(), ordering, *m_ldlControl);
#else
        throw std::runtime_error("Scotch support not compiled in");
#endif
    }
    else throw std::runtime_error("Unknown orderingMethod");

    m_factorizationType = FactorizationType::Symbolic;
}

void CatamariFactorizer::factorizeNumeric(const SuiteSparseMatrix &A, bool /* isInTryCatch */) {
    m_numericFactorizationImpl(A);
}

void CatamariFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, const SuiteSparseMatrix &B, bool /* isInTryCatch */) {
    m_numericFactorizationImpl(A, sigma, B.Ax.data());
}

void CatamariFactorizer::factorizeNumericWithShift(const SuiteSparseMatrix &A, Real sigma, bool /* isInTryCatch */) {
    m_numericFactorizationImpl(A, sigma, nullptr);
}

template<typename... Args>
void CatamariFactorizer::m_numericFactorizationImpl(const SuiteSparseMatrix &A, Args&&... args) {
    assertFactorization(FactorizationType::Symbolic);


    catamari::SparseLDLResult<double> result;

    auto num_fact_start = std::chrono::steady_clock::now();

    BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Numeric Factorize");
    auto &cm = m_catamariConverter->convert(A.Ax.data(), m_dataOffsetForScalarHessianLoc, std::forward<Args>(args)...);

    {
        BENCHMARK_SCOPED_TIMER_SECTION timer2("Catamari Numeric Call");
        result = m_ldl->RefactorWithFixedSparsityPattern(cm);
    }

    double num_fact_duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - num_fact_start).count();

    if (size_t(result.num_successful_pivots) != n_reduced()) {
        m_factorizationType = FactorizationType::Symbolic;
        throw std::runtime_error(std::to_string(result.num_successful_pivots) + "/" +
                                 std::to_string(n_reduced()) + "  pivots successful in Catamari numeric factorization (non-positive definite?)");
    }
    m_factorizationType = FactorizationType::Numeric;

    // Only record the time of successful numeric factorizations
    // (since the failures generally happen much more quickly and could
    // throw off averaging/bias the ordering selection heuristic)
    if (orderingMethod == OrderingMethod::Adaptive)
        adaptiveOrdering.recordNumeric(num_fact_duration);
}

size_t CatamariFactorizer::getFactorNNZ() const {
    throw std::runtime_error("CatamariFactorizer::getFactorNNZ not available in legacy mode");
}

double CatamariFactorizer::getFlopEstimate() const {
    assertFactorization(FactorizationType::Symbolic);
    throw std::runtime_error("CatamariFactorizer::getFlopEstimate not available in legacy mode");
}

void CatamariFactorizer::writeSolveTimers() const { }

// Raw pointer version (Use with care! Caller must allocate/own both pointers)
void CatamariFactorizer::solveRawReduced(const Real *b, Real *x, CholeskySys sys, bool alreadyPermuted) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("CatamariFactorizer.solveRawReduced");
    const size_t s = m_reduced();
    copyParallel(m_reduced(), b, x);
    solveRawReducedInPlace(x, sys, alreadyPermuted);
}

void CatamariFactorizer::solveRawReducedInPlace(Real *bx, CholeskySys sys, bool alreadyPermuted) const {
    assertFactorization(sys);
    if (sys != CholeskySys::A) {
        std::cout << "Alternative CholeskySys not yet wrapped for Catamari" << std::endl;
        throw std::runtime_error("Alternative CholeskySys not yet wrapped for Catamari");
    }

    if (alreadyPermuted) throw std::runtime_error("CatamariFactorizer::solveRawReducedInPlace does not support alreadyPermuted=true in legacy mode");

    catamari::BlasMatrixView<double> v;
    const size_t s = m_reduced();
    v.height = s;
    v.width = 1;
    v.leading_dim = s;
    v.data = bx;

    BENCHMARK_SCOPED_TIMER_SECTION timer("Catamari Solve");

    auto solve_start = std::chrono::steady_clock::now();
    m_ldl->Solve(&v);
    double solve_duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();

    if (orderingMethod == OrderingMethod::Adaptive)
        adaptiveOrdering.recordSolve(solve_duration);
}

void CatamariFactorizer::solveMultiRHS(const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &B, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &X) const {
    throw std::runtime_error("CatamariFactorizer::solveMultiRHS not yet implemented in legacy mode");
}

void CatamariFactorizer::m_populatePermutedReducedRowForRow() const {
    throw std::runtime_error("CatamariFactorizer::m_populatePermutedReducedRowForRow should never be called in legacy mode");
}

// Stashing support
void CatamariFactorizer::       stashFactorization()       { throw std::runtime_error("CatamariFactorizer::stashFactorization not supported in legacy mode"); }
bool CatamariFactorizer::  hasStashedFactorization() const { return bool(m_ldlStash); }
void CatamariFactorizer:: swapStashedFactorization()       { if (!hasStashedFactorization()) { throw std::runtime_error("No stashed factorization"); } std::swap(m_ldl, m_ldlStash); }
void CatamariFactorizer::clearStashedFactorization()       { m_ldlStash.reset(); }

void CatamariFactorizer::clearFactors() { m_factorizationType = FactorizationType::None; }

CatamariFactorizer::~CatamariFactorizer() = default;

} // namespace MeshFEM
