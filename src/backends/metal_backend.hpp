#pragma once

#include <memory>

#include "../backend.hpp"
#include "cpu_backend.hpp"

// Metal GPU backend. Owns device memory as MTLBuffers (shared storage, so host
// pointers stay valid for unified-memory access). Blocks not yet ported to Metal
// kernels delegate to an embedded CpuBackend reference implementation.
class MetalBackend : public Backend {
    struct Impl;
    std::unique_ptr<Impl> impl;

public:
    explicit MetalBackend(const ScratchDims& dims);
    ~MetalBackend() override;

    void beginSequence() override;
    void endSequence()   override;
    void synchronize()   override;

    std::shared_ptr<Storage> allocateStorage(size_t bytes, const std::string& name) override;
    std::shared_ptr<Storage> adoptStorage(void* host, size_t bytes, const std::string& name) override;

    void embed(Tensor& x, const Tensor& table, const Tensor& tokenIds, EmbedParams) override;
    void attentionBlock(Tensor& x, const AttnWeights&, KVCache&, const AttnBlockParams&) override;
    void ffnBlock(Tensor& x, const FFNWeights&, const FFNBlockParams&) override;
    void finalLogits(Tensor& logits, const Tensor& x, const Tensor& finalNorm,
                     const Tensor& lmHead, FinalParams) override;
    void sample(Tensor& nextTokenId, const Tensor& logits, SampleParams) override;
};
