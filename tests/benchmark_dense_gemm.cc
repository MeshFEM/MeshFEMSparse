////////////////////////////////////////////////////////////////////////////////
// benchmark_dense_gemm.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Benchmark dense GEMM implementations.
*///////////////////////////////////////////////////////////////////////////////
#include "MPSIntegration/MPSGemm.hh"

#include <MeshFEMCore/GlobalBenchmark.hh>
#include <MeshFEMCore/Parallelism.hh>
#include <catamari/dense_basic_linear_algebra.hpp>

#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <iostream>
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

template<class Real, class Multiplier>
void validate(const MatrixX<Real> &A, const MatrixX<Real> &B, const MatrixX<Real> &C0,
              const MatrixX<Real> &C_ref, const std::string &method, Multiplier &&multiplier) {
    MatrixX<Real> C = C0;
    multiplier(C);
    const double relerr = double((C - C_ref).norm() / C_ref.norm());
    const double tol = std::is_same<Real, float>::value ? 5e-4 : 1e-11;
    if (relerr > tol) {
        std::cerr << method << " relative error: " << relerr << std::endl;
        throw std::runtime_error(method + " validation failed");
    }
}

template<class Real, class Multiplier>
double timeMultiply(const MatrixX<Real> &C0, size_t numTrials, Multiplier &&multiplier) {
    double time = 0.0;
    for (size_t i = 0; i < numTrials; ++i) {
        MatrixX<Real> C = C0;
        auto start = std::chrono::high_resolution_clock::now();
        multiplier(C);
        auto end = std::chrono::high_resolution_clock::now();
        time += std::chrono::duration<double>(end - start).count();
    }
    return time / numTrials;
}

template<class Real, class Multiplier>
double timeRawMultiply(const MatrixX<Real> &C0, size_t numTrials, Multiplier &&multiplier) {
    double time = 0.0;
    for (size_t i = 0; i < numTrials; ++i) {
        MatrixX<Real> C = C0;
        time += multiplier(C);
    }
    return time / numTrials;
}

template<class Real>
void runBenchmarkForType(int height, int width, int contractionSize, size_t numTrials) {
    const Real alpha = Real(1.25);
    const Real beta = Real(-0.75);
    MatrixX<Real> A = MatrixX<Real>::Ones(height, contractionSize);
    MatrixX<Real> B = MatrixX<Real>::Ones(contractionSize, width);
    MatrixX<Real> C0 = MatrixX<Real>::Ones(height, width);

    auto AView = constBlasView(A);
    auto BView = constBlasView(B);

    auto gemm = [&](MatrixX<Real> &C) {
        auto CView = blasView(C);
        catamari::MatrixMultiplyNormalNormal(alpha, AView, BView, beta, &CView);
    };

    MatrixX<Real> C_ref = C0;
    gemm(C_ref);

    std::cout << (std::is_same<Real, float>::value ? "float" : "double") << ','
              << height << ',' << width << ',' << contractionSize << ",BLAS_gemm,"
              << std::setprecision(16) << timeMultiply(C0, numTrials, gemm) << std::endl;

    if constexpr (std::is_same<Real, float>::value) {
        static const bool mpsAvailable = mpsGemmAvailable();
        if (mpsAvailable) {
            auto mps = [&](MatrixX<Real> &C) {
                return mpsGemm(alpha, A.data(), height, contractionSize, B.data(), width, beta, C.data());
            };
            validate(A, B, C0, C_ref, "MPS", mps);
            std::cout << "float," << height << ',' << width << ',' << contractionSize << ",MPS,"
                      << std::setprecision(16) << timeRawMultiply(C0, numTrials, mps) << std::endl;
        }
        else {
            std::cerr << "MPS GEMM unavailable; skipping MPS timings" << std::endl;
        }
    }
}

int main(int argc, const char *argv[]) {
    if (argc < 5 || argc > 6) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <height> <width> <contraction_size> [num_trials=20]" << std::endl;
        return 1;
    }

    const size_t numThreads = std::stoul(argv[1]);
    const int height = std::stoi(argv[2]);
    const int width = std::stoi(argv[3]);
    const int contractionSize = std::stoi(argv[4]);
    const size_t numTrials = (argc > 5) ? std::stoul(argv[5]) : 20;

    if (height <= 0 || width <= 0 || contractionSize <= 0)
        throw std::runtime_error("height, width, and contraction_size must be positive");

    set_max_num_tbb_threads(numThreads);
    printMPSGemmDeviceInfo(std::cerr);

    std::cout << "scalar,height,width,contraction_size,method,time" << std::endl;
    runBenchmarkForType<float >(height, width, contractionSize, numTrials);
    runBenchmarkForType<double>(height, width, contractionSize, numTrials);
    return 0;
}
