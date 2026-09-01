#include "MPSOuterProduct.hh"

#if defined(__APPLE__)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <string>

namespace MeshFEM {
namespace {

id<MTLDevice> mpsOuterProductDevice() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

id<MTLCommandQueue> mpsOuterProductCommandQueue() {
    static id<MTLCommandQueue> queue = [mpsOuterProductDevice() newCommandQueue];
    return queue;
}

} // anonymous namespace

bool mpsOuterProductAvailable() {
    return mpsOuterProductDevice() != nil && mpsOuterProductCommandQueue() != nil;
}

void printMPSOuterProductDeviceInfo(std::ostream &os) {
    id<MTLDevice> device = mpsOuterProductDevice();
    if (device == nil) {
        os << "MPS device: unavailable" << std::endl;
        return;
    }

    os << "MPS device: " << [[device name] UTF8String] << std::endl;
    os << "  lowPower: " << ([device isLowPower] ? "true" : "false") << std::endl;
    os << "  removable: " << ([device isRemovable] ? "true" : "false") << std::endl;
    os << "  headless: " << ([device isHeadless] ? "true" : "false") << std::endl;
    os << "  recommendedMaxWorkingSetSize: "
       << static_cast<unsigned long long>([device recommendedMaxWorkingSetSize]) << std::endl;
    if (@available(macOS 10.15, *))
        os << "  unifiedMemory: " << ([device hasUnifiedMemory] ? "true" : "false") << std::endl;
}

double mpsLowerNormalHermitianOuterProduct(float alpha,
                                           const float *columnMajorLeft,
                                           size_t height,
                                           size_t rank,
                                           float beta,
                                           float *columnMajorOutput) {
    if (!mpsOuterProductAvailable())
        throw std::runtime_error("Metal Performance Shaders is not available");

    @autoreleasepool {
        id<MTLDevice> device = mpsOuterProductDevice();
        id<MTLCommandQueue> queue = mpsOuterProductCommandQueue();
        const NSUInteger m = static_cast<NSUInteger>(height);
        const NSUInteger k = static_cast<NSUInteger>(rank);
        const NSUInteger leftTransposeRowBytes = m * sizeof(float);
        const NSUInteger resultTransposeRowBytes = m * sizeof(float);
        const NSUInteger leftBytes = height * rank * sizeof(float);
        const NSUInteger resultBytes = height * height * sizeof(float);

        id<MTLBuffer> leftTransposeBuffer = [device newBufferWithLength:leftBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultTransposeBuffer = [device newBufferWithLength:resultBytes options:MTLResourceStorageModeShared];
        if (leftTransposeBuffer == nil || resultTransposeBuffer == nil)
            throw std::runtime_error("Failed to allocate MPS outer-product buffers");

        std::memcpy([leftTransposeBuffer contents], columnMajorLeft, leftBytes);
        std::memcpy([resultTransposeBuffer contents], columnMajorOutput, resultBytes);

        // Column-major A and C are row-major A^T and C^T in the same memory.
        // Compute C^T = alpha (A^T)^T A^T + beta C^T to avoid explicit transposes.
        MPSMatrixDescriptor *leftDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:m rowBytes:leftTransposeRowBytes dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *resultDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:m rowBytes:resultTransposeRowBytes dataType:MPSDataTypeFloat32];
        MPSMatrix *leftMatrix = [[MPSMatrix alloc] initWithBuffer:leftTransposeBuffer descriptor:leftDesc];
        MPSMatrix *resultMatrix = [[MPSMatrix alloc] initWithBuffer:resultTransposeBuffer descriptor:resultDesc];
        MPSMatrixMultiplication *multiply = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                              transposeLeft:YES
                                                                             transposeRight:NO
                                                                                 resultRows:m
                                                                              resultColumns:m
                                                                            interiorColumns:k
                                                                                      alpha:alpha
                                                                                       beta:beta];

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        auto start = std::chrono::high_resolution_clock::now();
        [multiply encodeToCommandBuffer:commandBuffer leftMatrix:leftMatrix rightMatrix:leftMatrix resultMatrix:resultMatrix];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        auto end = std::chrono::high_resolution_clock::now();
        const double rawTime = std::chrono::duration<double>(end - start).count();

        if ([commandBuffer error] != nil)
            throw std::runtime_error(std::string("MPS outer product command failed: ") + [[[commandBuffer error] localizedDescription] UTF8String]);

        std::memcpy(columnMajorOutput, [resultTransposeBuffer contents], resultBytes);
        return rawTime;
    }
}

} // namespace MeshFEM

#else // !defined(__APPLE__)

#include <ostream>
#include <stdexcept>

namespace MeshFEM {

bool mpsOuterProductAvailable() { return false; }
void printMPSOuterProductDeviceInfo(std::ostream &os) {
    os << "MPS device: unavailable; Metal Performance Shaders is only available on Apple platforms" << std::endl;
}
double mpsLowerNormalHermitianOuterProduct(float, const float *, size_t, size_t, float, float *) {
    throw std::runtime_error("Metal Performance Shaders is only available on Apple platforms");
}

} // namespace MeshFEM

#endif // defined(__APPLE__)
