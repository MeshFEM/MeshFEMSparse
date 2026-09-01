////////////////////////////////////////////////////////////////////////////////
// benchmark_dense_outer_product.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Benchmark dense lower Hermitian outer-product implementations.
*///////////////////////////////////////////////////////////////////////////////
#include "MPSIntegration/MPSOuterProduct.hh"

#include <MeshFEMCore/GlobalBenchmark.hh>
#include <MeshFEMCore/Parallelism.hh>
#include <catamari/dense_basic_linear_algebra.hpp>
#include <tbb/tbb.h>

#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

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
catamari::ConstBlasMatrixView<Real> constBlasView(const MatrixX<Real> &A) {
    catamari::ConstBlasMatrixView<Real> result;
    result.data = A.data();
    result.height = A.rows();
    result.width = A.cols();
    result.leading_dim = A.rows();
    return result;
}

template<class Real>
void zeroStrictUpper(MatrixX<Real> &C) {
    C.template triangularView<Eigen::StrictlyUpper>().setZero();
}

template<class Real, class Updater>
void validate(const MatrixX<Real> &A, const MatrixX<Real> &C0, const MatrixX<Real> &C_ref,
              const std::string &method, Updater &&updater) {
    MatrixX<Real> C = C0;
    updater(C);
    zeroStrictUpper(C);
    const double relerr = double((C - C_ref).norm() / C_ref.norm());
    const double tol = std::is_same<Real, float>::value ? 5e-4 : 1e-11;
    if (relerr > tol) {
        std::cerr << method << " relative error: " << relerr << std::endl;
        throw std::runtime_error(method + " validation failed");
    }
}

template<class Real, class Updater>
double timeUpdate(const MatrixX<Real> &C0, size_t numTrials, Updater &&updater) {
    double time = 0.0;
    for (size_t i = 0; i < numTrials; ++i) {
        MatrixX<Real> C = C0;
        auto start = std::chrono::high_resolution_clock::now();
        updater(C);
        auto end = std::chrono::high_resolution_clock::now();
        time += std::chrono::duration<double>(end - start).count();
    }
    return time / numTrials;
}

template<class Real, class Updater>
double timeRawUpdate(const MatrixX<Real> &C0, size_t numTrials, Updater &&updater) {
    double time = 0.0;
    for (size_t i = 0; i < numTrials; ++i) {
        MatrixX<Real> C = C0;
        time += updater(C);
    }
    return time / numTrials;
}

template<class Real>
void runBenchmarkForType(int height, int rank, int tileSize, size_t numTrials) {
    const Real alpha = Real(-1);
    const Real beta = Real(1);
    MatrixX<Real> A = MatrixX<Real>::Random(height, rank);
    MatrixX<Real> C0 = MatrixX<Real>::Random(height, height);
    zeroStrictUpper(C0);

    MatrixX<Real> C_ref = C0;
    auto AView = constBlasView(A);
    auto refView = blasView(C_ref);
    catamari::LowerNormalHermitianOuterProduct(alpha, AView, beta, &refView);
    zeroStrictUpper(C_ref);

    auto syrk = [&](MatrixX<Real> &C) {
        auto CView = blasView(C);
        catamari::LowerNormalHermitianOuterProduct(alpha, AView, beta, &CView);
    };

    tbb::task_group_context tgc;
    auto tbbOuterProduct = [&](MatrixX<Real> &C) {
        auto CView = blasView(C);
        catamari::TBBLowerNormalHermitianOuterProduct(tgc, tileSize, alpha, AView, beta, &CView);
    };

    validate(A, C0, C_ref, "TBBLowerNormalHermitianOuterProduct", tbbOuterProduct);
    std::cout << (std::is_same<Real, float>::value ? "float" : "double") << ','
              << height << ',' << rank << ",TBBLowerNormalHermitianOuterProduct,"
              << std::setprecision(16) << timeUpdate(C0, numTrials, tbbOuterProduct) << std::endl;

    std::cout << (std::is_same<Real, float>::value ? "float" : "double") << ','
              << height << ',' << rank << ",BLAS_syrk,"
              << std::setprecision(16) << timeUpdate(C0, numTrials, syrk) << std::endl;

    if constexpr (std::is_same<Real, float>::value) {
        static const bool mpsAvailable = mpsOuterProductAvailable();
        if (mpsAvailable) {
            auto mps = [&](MatrixX<Real> &C) {
                return mpsLowerNormalHermitianOuterProduct(alpha, A.data(), height, rank, beta, C.data());
            };
            validate(A, C0, C_ref, "MPS", mps);
            std::cout << "float," << height << ',' << rank << ",MPS,"
                      << std::setprecision(16) << timeRawUpdate(C0, numTrials, mps) << std::endl;
        }
        else {
            std::cerr << "MPS outer product unavailable; skipping MPS timings" << std::endl;
        }
    }
}

int main(int argc, const char *argv[]) {
    if (argc < 4 || argc > 6) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <height> <rank> [num_trials=20] [tile_size=128]" << std::endl;
        return 1;
    }

    const size_t numThreads = std::stoul(argv[1]);
    const int height = std::stoi(argv[2]);
    const int rank = std::stoi(argv[3]);
    const size_t numTrials = (argc > 4) ? std::stoul(argv[4]) : 20;
    const int tileSize = (argc > 5) ? std::stoi(argv[5]) : 128;

    if (height <= 0 || rank <= 0 || tileSize <= 0)
        throw std::runtime_error("height, rank, and tile_size must be positive");

    set_max_num_tbb_threads(numThreads);
    printMPSOuterProductDeviceInfo(std::cerr);

#if __linux__
    PinningObserver thread_pinner;
#endif

    std::cout << "scalar,height,rank,method,time" << std::endl;
    runBenchmarkForType<float >(height, rank, tileSize, numTrials);
    runBenchmarkForType<double>(height, rank, tileSize, numTrials);
    return 0;
}
