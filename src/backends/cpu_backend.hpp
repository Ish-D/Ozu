#pragma once

#include "../backend.hpp"

// Scratch dimensions the backend preallocates once (max over the model's layers).
struct ScratchDims {
    int maxSeqLen;
    int hiddenSize;
    int qDim;            // max numQHeads * headDim
    int kvDim;           // max numKVHeads * headDim
    int intermediate;    // max dense FFN width
};

// Reference CPU backend. Blocks are thin orchestration over the ops:: kernels;
// the backend owns all transient scratch. Only the Llama path is implemented;
// Gemma-only features error out explicitly (see docs/backend-design.md §13).
class CpuBackend : public Backend {
private:
    Tensor xNorm;
    Tensor q, k, v;
    Tensor attnOut;
    Tensor oProjOut;
    Tensor ffnGate, ffnUp, ffnDown;
    Tensor attnScores;

public:
    explicit CpuBackend(const ScratchDims& dims);

    void beginSequence() override {}
    void endSequence()   override {}
    void synchronize()   override {}

    Tensor allocate(std::vector<int> shape, DataType dtype, std::string name) override;
    Tensor adopt(void* host, std::vector<int> shape, DataType dtype, std::string name) override;

    void embed(Tensor& x, const Tensor& table, const Tensor& tokenIds, EmbedParams) override;
    void attentionBlock(Tensor& x, const AttnWeights&, KVCache&, const AttnBlockParams&) override;
    void ffnBlock(Tensor& x, const FFNWeights&, const FFNBlockParams&) override;
    void finalLogits(Tensor& logits, const Tensor& x, const Tensor& finalNorm,
                     const Tensor& lmHead, FinalParams) override;
    void sample(Tensor& nextTokenId, const Tensor& logits, SampleParams) override;
};
