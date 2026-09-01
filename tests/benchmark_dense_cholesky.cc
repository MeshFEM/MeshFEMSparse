////////////////////////////////////////////////////////////////////////////////
// benchmark_dense_cholesky.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Benchmark dense Cholesky implementations for float and double matrices.
*///////////////////////////////////////////////////////////////////////////////
#include "MPSIntegration/MPSCholesky.hh"

#include <MeshFEMCore/GlobalBenchmark.hh>
#include <MeshFEMCore/Parallelism.hh>
#include <catamari/dense_factorizations.hpp>
#include <tbb/tbb.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace MeshFEM;

template<class Real>
using MatrixX = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

template<class Real>
catamari::BlasMatrixView<Real> blasView(MatrixX<Real> &A) {
    catamari::BlasMatrixView<Real> result;
    result.data = A.data();
    result.height = A.rows();
    result.width = A.cols();
    result.leading_dim = A.rows();
    return result;
}

template<class Real>
MatrixX<Real> makeSPD(int n) {
    // MatrixX<Real> A = MatrixX<Real>::Random(n, n);
    // A = (A.transpose() * A).eval();
    // A.diagonal().array() += Real(n);
    MatrixX<Real> A = MatrixX<Real>::Identity(n, n);
    return A;
}

template<class Real, class Factorizer>
void validate(const MatrixX<Real> &A, const MatrixX<Real> &L_ref, const std::string &method, Factorizer &&factorizer) {
    MatrixX<Real> L = A;
    factorizer(L);
    L.template triangularView<Eigen::StrictlyUpper>().setZero();
    const double relerr = double((L - L_ref).norm() / L_ref.norm());
    const double tol = std::is_same<Real, float>::value ? 5e-4 : 1e-10;
    if (relerr > tol) {
        std::cerr << method << " relative error: " << relerr << " at size " << A.rows() << std::endl;
        throw std::runtime_error(method + " validation failed");
    }
}

template<class Real, class Factorizer>
double timeFactorization(const MatrixX<Real> &A, size_t numTrials, Factorizer &&factorizer) {
    double time = 0.0;
    for (size_t i = 0; i < numTrials; ++i) {
        MatrixX<Real> L = A;
        auto start = std::chrono::high_resolution_clock::now();
        factorizer(L);
        auto end = std::chrono::high_resolution_clock::now();
        time += std::chrono::duration<double>(end - start).count();
    }
    return time / numTrials;
}

template<class Real, class Factorizer>
double timeRawFactorization(const MatrixX<Real> &A, size_t numTrials, Factorizer &&factorizer) {
    double time = 0.0;
    for (size_t i = 0; i < numTrials; ++i) {
        MatrixX<Real> L = A;
        time += factorizer(L);
    }
    return time / numTrials;
}

template<class Real>
void runBenchmarksForType(int n, int block_size, int tile_size, size_t numTrials) {
    tbb::task_group_context tgc;

    {
        MatrixX<Real> A = makeSPD<Real>(n);
        MatrixX<Real> L_ref = A;
        auto refView = blasView(L_ref);
        if (catamari::LowerCholeskyFactorization(block_size, &refView) < n)
            throw std::runtime_error("Reference LAPACK Cholesky failed");
        L_ref.template triangularView<Eigen::StrictlyUpper>().setZero();

        auto lapack = [&](MatrixX<Real> &L) {
            auto view = blasView(L);
            if (catamari::LowerCholeskyFactorization(block_size, &view) < n)
                throw std::runtime_error("LAPACK Cholesky failed");
        };

        auto flowgraph = [&](MatrixX<Real> &L) {
            auto view = blasView(L);
            if (catamari::CholeskyFlowgraph<Real>(tgc, view, tile_size, block_size).run(view) < n)
                throw std::runtime_error("CholeskyFlowgraph failed");
        };

        validate(A, L_ref, "CholeskyFlowgraph", flowgraph);
        std::cout << (std::is_same<Real, float>::value ? "float" : "double") << ','
                  << n << ",CholeskyFlowgraph_" << tile_size << ','
                  << std::setprecision(16) << timeFactorization(A, numTrials, flowgraph) << std::endl;

        std::cout << (std::is_same<Real, float>::value ? "float" : "double") << ','
                  << n << ",LAPACK_potrf,"
                  << std::setprecision(16) << timeFactorization(A, numTrials, lapack) << std::endl;

        // if constexpr (std::is_same<Real, float>::value) {
        //     static const bool mpsAvailable = mpsCholeskyAvailable();
        //     if (mpsAvailable) {
        //         auto mps = [&](MatrixX<Real> &L) { return mpsLowerCholeskyFactorization(L.data(), n); };
        //         validate(A, L_ref, "MPS", mps);
        //         std::cout << "float," << n << ",MPS,"
        //                   << std::setprecision(16) << timeRawFactorization(A, numTrials, mps) << std::endl;
        //     }
        //     else if (n == sizes.front())
        //         std::cerr << "MPS Cholesky unavailable; skipping MPS timings" << std::endl;
        // }
    }
}

int main(int argc, const char *argv[]) {
    if (argc < 2 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> [size=3000] [tile_size=128] [num_trials=5]" << std::endl;
        return 1;
    }

    const size_t num_threads = std::stoul(argv[1]);
    const int size = (argc > 2) ? std::stoi(argv[2]) : 3000;
    const int tile_size = (argc > 3) ? std::stoi(argv[3]) : 128;
    const size_t numTrials = (argc > 4) ? std::stoul(argv[4]) : 5;

    set_max_num_tbb_threads(num_threads);

#if __linux__
    PinningObserver thread_pinner;
#endif

    std::cout << "# scalar,size,method,time" << std::endl;
    runBenchmarksForType<float >(size, 64, tile_size, numTrials);
    runBenchmarksForType<double>(size, 64, tile_size, numTrials);
    return 0;
}
