#include "MPSCholesky.hh"

#if defined(__APPLE__)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>
#include <stdexcept>
#include <string>

namespace MeshFEM {
namespace {

id<MTLDevice> mpsDevice() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

id<MTLCommandQueue> mpsCommandQueue() {
    static id<MTLCommandQueue> queue = [mpsDevice() newCommandQueue];
    return queue;
}

} // anonymous namespace

bool mpsCholeskyAvailable() {
    return mpsDevice() != nil && mpsCommandQueue() != nil;
}

double mpsLowerCholeskyFactorization(float *columnMajorMatrix, size_t n) {
    if (!mpsCholeskyAvailable())
        throw std::runtime_error("Metal Performance Shaders is not available");

    @autoreleasepool {
        id<MTLDevice> device = mpsDevice();
        id<MTLCommandQueue> queue = mpsCommandQueue();
        const NSUInteger order = static_cast<NSUInteger>(n);
        const NSUInteger rowBytes = [MPSMatrixDescriptor rowBytesForColumns:order dataType:MPSDataTypeFloat32];
        const NSUInteger dataBytes = rowBytes * order;

        id<MTLBuffer> srcBuffer = [device newBufferWithLength:dataBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> dstBuffer = [device newBufferWithLength:dataBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> statusBuffer = [device newBufferWithLength:sizeof(MPSMatrixDecompositionStatus) options:MTLResourceStorageModeShared];
        if (srcBuffer == nil || dstBuffer == nil || statusBuffer == nil)
            throw std::runtime_error("Failed to allocate MPS Cholesky buffers");

        auto *src = static_cast<float *>([srcBuffer contents]);
        for (size_t i = 0; i < n; ++i) {
            auto *row = reinterpret_cast<float *>(reinterpret_cast<char *>(src) + i * rowBytes);
            for (size_t j = 0; j < n; ++j)
                row[j] = columnMajorMatrix[i + j * n];
        }

        auto *status = static_cast<MPSMatrixDecompositionStatus *>([statusBuffer contents]);
        *status = MPSMatrixDecompositionStatusFailure;

        MPSMatrixDescriptor *desc = [MPSMatrixDescriptor matrixDescriptorWithRows:order columns:order rowBytes:rowBytes dataType:MPSDataTypeFloat32];
        MPSMatrix *srcMatrix = [[MPSMatrix alloc] initWithBuffer:srcBuffer descriptor:desc];
        MPSMatrix *dstMatrix = [[MPSMatrix alloc] initWithBuffer:dstBuffer descriptor:desc];
        MPSMatrixDecompositionCholesky *chol = [[MPSMatrixDecompositionCholesky alloc] initWithDevice:device lower:YES order:order];

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        auto start = std::chrono::high_resolution_clock::now();
        [chol encodeToCommandBuffer:commandBuffer sourceMatrix:srcMatrix resultMatrix:dstMatrix status:statusBuffer];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        auto end = std::chrono::high_resolution_clock::now();
        const double rawTime = std::chrono::duration<double>(end - start).count();

        if ([commandBuffer error] != nil)
            throw std::runtime_error(std::string("MPS Cholesky command failed: ") + [[[commandBuffer error] localizedDescription] UTF8String]);
        if (*status != MPSMatrixDecompositionStatusSuccess)
            throw std::runtime_error("MPS Cholesky factorization failed with status " + std::to_string(static_cast<int>(*status)));

        const auto *dst = static_cast<const float *>([dstBuffer contents]);
        for (size_t i = 0; i < n; ++i) {
            const auto *row = reinterpret_cast<const float *>(reinterpret_cast<const char *>(dst) + i * rowBytes);
            for (size_t j = 0; j < n; ++j)
                columnMajorMatrix[i + j * n] = (i >= j) ? row[j] : 0.0f;
        }
        return rawTime;
    }
}

} // namespace MeshFEM

#else // !defined(__APPLE__)

#include <stdexcept>

namespace MeshFEM {

bool mpsCholeskyAvailable() { return false; }
double mpsLowerCholeskyFactorization(float *, size_t) {
    throw std::runtime_error("Metal Performance Shaders is only available on Apple platforms");
}

} // namespace MeshFEM

#endif // defined(__APPLE__)
