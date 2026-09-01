#include "MPSGemm.hh"

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

id<MTLDevice> mpsGemmDevice() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

id<MTLCommandQueue> mpsGemmCommandQueue() {
    static id<MTLCommandQueue> queue = [mpsGemmDevice() newCommandQueue];
    return queue;
}

} // anonymous namespace

bool mpsGemmAvailable() {
    return mpsGemmDevice() != nil && mpsGemmCommandQueue() != nil;
}

void printMPSGemmDeviceInfo(std::ostream &os) {
    id<MTLDevice> device = mpsGemmDevice();
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

double mpsGemm(float alpha,
               const float *columnMajorLeft,
               size_t height,
               size_t contractionSize,
               const float *columnMajorRight,
               size_t width,
               float beta,
               float *columnMajorOutput) {
    if (!mpsGemmAvailable())
        throw std::runtime_error("Metal Performance Shaders is not available");

    @autoreleasepool {
        id<MTLDevice> device = mpsGemmDevice();
        id<MTLCommandQueue> queue = mpsGemmCommandQueue();
        const NSUInteger m = static_cast<NSUInteger>(height);
        const NSUInteger n = static_cast<NSUInteger>(width);
        const NSUInteger k = static_cast<NSUInteger>(contractionSize);
        const NSUInteger leftTransposeRowBytes = m * sizeof(float);
        const NSUInteger rightTransposeRowBytes = k * sizeof(float);
        const NSUInteger outputTransposeRowBytes = m * sizeof(float);

        id<MTLBuffer> leftTransposeBuffer = [device newBufferWithLength:leftTransposeRowBytes * k options:MTLResourceStorageModeShared];
        id<MTLBuffer> rightTransposeBuffer = [device newBufferWithLength:rightTransposeRowBytes * n options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputTransposeBuffer = [device newBufferWithLength:outputTransposeRowBytes * n options:MTLResourceStorageModeShared];
        if (leftTransposeBuffer == nil || rightTransposeBuffer == nil || outputTransposeBuffer == nil)
            throw std::runtime_error("Failed to allocate MPS GEMM buffers");

        std::memcpy([leftTransposeBuffer contents], columnMajorLeft, height * contractionSize * sizeof(float));
        std::memcpy([rightTransposeBuffer contents], columnMajorRight, contractionSize * width * sizeof(float));
        std::memcpy([outputTransposeBuffer contents], columnMajorOutput, height * width * sizeof(float));

        // Column-major A, B, C are row-major A^T, B^T, C^T in the same memory.
        // Compute C^T = alpha B^T A^T + beta C^T to avoid explicit transposes.
        MPSMatrixDescriptor *leftDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:rightTransposeRowBytes dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *rightDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:m rowBytes:leftTransposeRowBytes dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *outputDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:m rowBytes:outputTransposeRowBytes dataType:MPSDataTypeFloat32];
        MPSMatrix *leftMatrix = [[MPSMatrix alloc] initWithBuffer:rightTransposeBuffer descriptor:leftDesc];
        MPSMatrix *rightMatrix = [[MPSMatrix alloc] initWithBuffer:leftTransposeBuffer descriptor:rightDesc];
        MPSMatrix *outputMatrix = [[MPSMatrix alloc] initWithBuffer:outputTransposeBuffer descriptor:outputDesc];
        MPSMatrixMultiplication *multiply = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                              transposeLeft:NO
                                                                             transposeRight:NO
                                                                                 resultRows:n
                                                                              resultColumns:m
                                                                            interiorColumns:k
                                                                                      alpha:alpha
                                                                                       beta:beta];

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        auto start = std::chrono::high_resolution_clock::now();
        [multiply encodeToCommandBuffer:commandBuffer leftMatrix:leftMatrix rightMatrix:rightMatrix resultMatrix:outputMatrix];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        auto end = std::chrono::high_resolution_clock::now();
        const double rawTime = std::chrono::duration<double>(end - start).count();

        if ([commandBuffer error] != nil)
            throw std::runtime_error(std::string("MPS GEMM command failed: ") + [[[commandBuffer error] localizedDescription] UTF8String]);

        std::memcpy(columnMajorOutput, [outputTransposeBuffer contents], height * width * sizeof(float));
        return rawTime;
    }
}

} // namespace MeshFEM

#else // !defined(__APPLE__)

#include <ostream>
#include <stdexcept>

namespace MeshFEM {

bool mpsGemmAvailable() { return false; }
void printMPSGemmDeviceInfo(std::ostream &os) {
    os << "MPS device: unavailable; Metal Performance Shaders is only available on Apple platforms" << std::endl;
}
double mpsGemm(float, const float *, size_t, size_t, const float *, size_t, float, float *) {
    throw std::runtime_error("Metal Performance Shaders is only available on Apple platforms");
}

} // namespace MeshFEM

#endif // defined(__APPLE__)
