#pragma once

#include <vector>
#include <string>

#include "tensor.hpp"

// Compute-backend seam. The model layer calls block ops on a Backend and never
// touches a device API. Ops record work that writes into `out` tensors; the host
// may read a result only after synchronize(). See docs/backend-design.md.

struct NormSpec { float eps; float affineOffset; };   // Llama {eps,0}; Gemma-family {eps,1}
enum class ActFn    { Silu, GeluTanh };
enum class RopeType { Default, Proportional };

struct EmbedParams  { float scale = 1.0f; };                             // Gemma: sqrt(hidden)
struct FinalParams  { NormSpec norm; float logitSoftcap = 0.0f; };       // Gemma: 30
struct SampleParams { float temperature = 0.0f; int topK = 0; float topP = 0.0f; }; // temp 0 => greedy

struct AttnWeights {
    Tensor* inputNorm    = nullptr;   // pre-attention norm
    Tensor* postAttnNorm = nullptr;   // sandwich post-attention norm (null on Llama)
    Tensor* qProj        = nullptr;
    Tensor* kProj        = nullptr;
    Tensor* vProj        = nullptr;   // null on layers where k_eq_v (V reuses K)
    Tensor* oProj        = nullptr;
    Tensor* qNorm        = nullptr;   // per-head QK-norm (null on Llama)
    Tensor* kNorm        = nullptr;
    Tensor* layerScalar  = nullptr;   // role TBD
};

struct AttnBlockParams {
    int numQHeads   = 0;
    int numKVHeads  = 0;
    int headDim     = 0;
    int startPos    = 0;
    NormSpec norm{};
    float    ropeTheta          = 0.0f;
    RopeType ropeType           = RopeType::Default;
    float    partialRotaryFactor = 1.0f;   // 1.0 => full rotary
    float    attnScale          = 0.0f;    // 0 => 1/sqrt(headDim)
    int      windowSize         = 0;       // 0 => global (full causal)
    bool     kEqV               = false;
};

struct MoEExperts {                        // null on dense models
    Tensor* gateUpProj    = nullptr;       // [E, 2*moeInter, hidden] (fused)
    Tensor* downProj      = nullptr;       // [E, hidden, moeInter]
    Tensor* routerProj    = nullptr;       // [E, hidden]
    Tensor* routerScale   = nullptr;       // [hidden]
    Tensor* perExpertScale = nullptr;      // [E]
};

struct FFNWeights {
    Tensor* preNorm  = nullptr;            // pre-FFN norm (Llama: post_attention_layernorm)
    Tensor* postNorm = nullptr;            // sandwich post-FFN norm (null on Llama)
    Tensor* gateProj = nullptr;
    Tensor* upProj   = nullptr;
    Tensor* downProj = nullptr;
    MoEExperts* experts = nullptr;         // routed path (MoE only)
    Tensor* preNorm2  = nullptr;           // MoE-path norms (wiring TBD)
    Tensor* postNorm1 = nullptr;
    Tensor* postNorm2 = nullptr;
};

struct FFNBlockParams {
    NormSpec norm{};
    ActFn act = ActFn::Silu;
    int denseIntermediate = 0;
    int numExperts        = 0;             // 0 => dense-only
    int topK              = 0;
    int moeIntermediate   = 0;
};

struct KVCache {
    Tensor k;
    Tensor v;                              // aliases k when kEqV
    int    capacity = 0;                   // window size (ring) or max context
    bool   ring     = false;
};

class Backend {
public:
    virtual ~Backend() = default;

    // Sequence lifecycle. On CPU these are no-ops; on Metal they bracket one
    // command buffer. Driven by generate() around forward()+sample().
    virtual void beginSequence() = 0;
    virtual void endSequence()   = 0;
    virtual void synchronize()   = 0;

    // Storage
    virtual Tensor allocate(std::vector<int> shape, DataType dtype, std::string name) = 0;
    virtual Tensor adopt(void* host, std::vector<int> shape, DataType dtype, std::string name) = 0;

    // Blocks
    virtual void embed(Tensor& x, const Tensor& table, const Tensor& tokenIds, EmbedParams) = 0;
    virtual void attentionBlock(Tensor& x, const AttnWeights&, KVCache&, const AttnBlockParams&) = 0;
    virtual void ffnBlock(Tensor& x, const FFNWeights&, const FFNBlockParams&) = 0;
    virtual void finalLogits(Tensor& logits, const Tensor& x, const Tensor& finalNorm,
                             const Tensor& lmHead, FinalParams) = 0;
    virtual void sample(Tensor& nextTokenId, const Tensor& logits, SampleParams) = 0;
};
