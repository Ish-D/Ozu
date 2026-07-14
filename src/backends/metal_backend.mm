#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstring>

#include "metal_backend.hpp"

struct MetalBackend::Impl {
    id<MTLDevice>        device       = nil;
    id<MTLCommandQueue>  queue        = nil;
    id<MTLCommandBuffer> currentCb    = nil;
    NSMutableArray*      buffers      = nil;   // retains all MTLBuffers for backend lifetime
    CpuBackend           cpu;                  // reference for not-yet-ported blocks

    explicit Impl(const ScratchDims& dims) : cpu(dims) {}
};

MetalBackend::MetalBackend(const ScratchDims& dims) : impl(std::make_unique<Impl>(dims)) {
    impl->device = MTLCreateSystemDefaultDevice();
    if (!impl->device) utils::error("MetalBackend: no Metal device available");
    impl->queue   = [impl->device newCommandQueue];
    impl->buffers = [NSMutableArray array];
}

MetalBackend::~MetalBackend() = default;

void MetalBackend::beginSequence() { impl->currentCb = [impl->queue commandBuffer]; }
void MetalBackend::endSequence()   { if (impl->currentCb) [impl->currentCb commit]; }
void MetalBackend::synchronize() {
    if (impl->currentCb) { [impl->currentCb waitUntilCompleted]; impl->currentCb = nil; }
}

std::shared_ptr<Storage> MetalBackend::allocateStorage(size_t bytes, const std::string& name) {
    id<MTLBuffer> buf = [impl->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buf) utils::error("MetalBackend: failed to allocate {} bytes for {}", bytes, name);
    [impl->buffers addObject:buf];

    auto s = std::make_shared<Storage>();
    s->host     = buf.contents;
    s->device   = (__bridge void*)buf;
    s->bytes    = bytes;
    s->where    = Device::GPU;
    s->ownsHost = false;   // MTLBuffer owns the memory; backend retains the buffer
    return s;
}

std::shared_ptr<Storage> MetalBackend::adoptStorage(void* host, size_t bytes, const std::string& name) {
    // No-copy wrap needs page-aligned base and length; mmap gives an aligned base
    // and maps whole pages, so rounding the length up stays within the mapping.
    constexpr size_t pageSize = 16384;
    const size_t rounded = ((bytes + pageSize - 1) / pageSize) * pageSize;

    id<MTLBuffer> buf = [impl->device newBufferWithBytesNoCopy:host
                                                        length:rounded
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
    if (!buf) {
        buf = [impl->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf) utils::error("MetalBackend: failed to adopt {} bytes for {}", bytes, name);
        std::memcpy(buf.contents, host, bytes);
    }
    [impl->buffers addObject:buf];

    auto s = std::make_shared<Storage>();
    s->host     = buf.contents;
    s->device   = (__bridge void*)buf;
    s->bytes    = bytes;
    s->where    = Device::GPU;
    s->ownsHost = false;
    return s;
}

// Blocks delegate to the CPU reference; unified shared memory keeps this correct.
void MetalBackend::embed(Tensor& x, const Tensor& table, const Tensor& tokenIds, EmbedParams p) {
    impl->cpu.embed(x, table, tokenIds, p);
}
void MetalBackend::attentionBlock(Tensor& x, const AttnWeights& w, KVCache& cache, const AttnBlockParams& p) {
    impl->cpu.attentionBlock(x, w, cache, p);
}
void MetalBackend::ffnBlock(Tensor& x, const FFNWeights& w, const FFNBlockParams& p) {
    impl->cpu.ffnBlock(x, w, p);
}
void MetalBackend::finalLogits(Tensor& logits, const Tensor& x, const Tensor& finalNorm,
                               const Tensor& lmHead, FinalParams p) {
    impl->cpu.finalLogits(logits, x, finalNorm, lmHead, p);
}
void MetalBackend::sample(Tensor& nextTokenId, const Tensor& logits, SampleParams p) {
    impl->cpu.sample(nextTokenId, logits, p);
}
