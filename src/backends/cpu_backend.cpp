#include "cpu_backend.hpp"

#include <cmath>

#include "../ops.hpp"

CpuBackend::CpuBackend(const ScratchDims& d)
    : xNorm     ({d.maxSeqLen, d.hiddenSize},   DataType::fp32, "xNorm"),
      q         ({d.maxSeqLen, d.qDim},         DataType::fp32, "q"),
      k         ({d.maxSeqLen, d.kvDim},        DataType::fp32, "k"),
      v         ({d.maxSeqLen, d.kvDim},        DataType::fp32, "v"),
      attnOut   ({d.maxSeqLen, d.qDim},         DataType::fp32, "attnOut"),
      oProjOut  ({d.maxSeqLen, d.hiddenSize},   DataType::fp32, "oProjOut"),
      ffnGate   ({d.maxSeqLen, d.intermediate}, DataType::fp32, "ffnGate"),
      ffnUp     ({d.maxSeqLen, d.intermediate}, DataType::fp32, "ffnUp"),
      ffnDown   ({d.maxSeqLen, d.hiddenSize},   DataType::fp32, "ffnDown"),
      attnScores({d.maxSeqLen},                 DataType::fp32, "attnScores") {}

Tensor CpuBackend::allocate(std::vector<int> shape, DataType dtype, std::string name) {
    return Tensor(std::move(shape), dtype, std::move(name), Device::CPU);
}

Tensor CpuBackend::adopt(void* host, std::vector<int> shape, DataType dtype, std::string name) {
    return Tensor(std::move(name), std::move(shape), dtype, host, Device::CPU);
}

void CpuBackend::embed(Tensor& x, const Tensor& table, const Tensor& tokenIds, EmbedParams params) {
    const int seqLen = tokenIds.shape[0];
    const int hidden = table.shape[1];
    x.reshape({seqLen, hidden});

    const auto* ids  = static_cast<const int32_t*>(tokenIds.data);
    auto*       xData = static_cast<float*>(x.data);

    for (int s = 0; s < seqLen; s++) {
        const int token = ids[s];
        for (int i = 0; i < hidden; i++) {
            const size_t idx = static_cast<size_t>(token) * hidden + i;
            float val = (table.dataType == DataType::bf16)
                            ? bf16Tof32(static_cast<const uint16_t*>(table.data)[idx])
                            : static_cast<const float*>(table.data)[idx];
            xData[s * hidden + i] = val * params.scale;
        }
    }
}

void CpuBackend::attentionBlock(Tensor& x, const AttnWeights& w, KVCache& cache, const AttnBlockParams& p) {
    // Gemma-only features: explicit until implemented against the HF reference.
    if (w.qNorm || w.kNorm)              utils::error("CpuBackend: QK-norm not implemented");
    if (w.postAttnNorm)                  utils::error("CpuBackend: sandwich post-attn norm not implemented");
    if (w.layerScalar)                   utils::error("CpuBackend: layer_scalar not implemented");
    if (p.ropeType != RopeType::Default) utils::error("CpuBackend: proportional RoPE not implemented");
    if (p.partialRotaryFactor != 1.0f)   utils::error("CpuBackend: partial RoPE not implemented");
    if (p.kEqV || !w.vProj)              utils::error("CpuBackend: k_eq_v not implemented");
    if (p.windowSize != 0)               utils::error("CpuBackend: sliding-window attention not implemented");
    if (p.attnScale != 0.0f)             utils::error("CpuBackend: custom attention scale not implemented");
    if (p.norm.affineOffset != 0.0f)     utils::error("CpuBackend: (1+w) RMSNorm not implemented");

    const int seqLen = x.shape[0];
    const int hidden = x.shape[1];
    const int qDim   = w.qProj->shape[0];
    const int kvDim  = w.kProj->shape[0];

    xNorm.reshape({seqLen, hidden});
    q.reshape({seqLen, qDim});
    k.reshape({seqLen, kvDim});
    v.reshape({seqLen, kvDim});
    attnOut.reshape({seqLen, qDim});
    oProjOut.reshape({seqLen, hidden});

    ops::rmsNorm(xNorm, x, *w.inputNorm, p.norm.eps);
    ops::matmul(q, xNorm, *w.qProj);
    ops::matmul(k, xNorm, *w.kProj);
    ops::matmul(v, xNorm, *w.vProj);
    ops::applyRope(q, k, p.startPos, p.ropeTheta, p.headDim);
    ops::updateCache(cache.k, k, p.startPos);
    ops::updateCache(cache.v, v, p.startPos);
    ops::attention(attnOut, q, cache.k, cache.v, attnScores,
                   p.numQHeads, p.numKVHeads, p.headDim, p.startPos);
    ops::matmul(oProjOut, attnOut, *w.oProj);
    ops::add(x, x, oProjOut);
}

void CpuBackend::ffnBlock(Tensor& x, const FFNWeights& w, const FFNBlockParams& p) {
    if (w.experts || p.numExperts)   utils::error("CpuBackend: MoE experts not implemented");
    if (w.postNorm)                  utils::error("CpuBackend: sandwich post-FFN norm not implemented");
    if (p.act != ActFn::Silu)        utils::error("CpuBackend: GeLU activation not implemented");
    if (p.norm.affineOffset != 0.0f) utils::error("CpuBackend: (1+w) RMSNorm not implemented");

    const int seqLen = x.shape[0];
    const int hidden = x.shape[1];
    const int inter  = w.gateProj->shape[0];

    xNorm.reshape({seqLen, hidden});
    ffnGate.reshape({seqLen, inter});
    ffnUp.reshape({seqLen, inter});
    ffnDown.reshape({seqLen, hidden});

    ops::rmsNorm(xNorm, x, *w.preNorm, p.norm.eps);
    ops::matmul(ffnGate, xNorm, *w.gateProj);
    ops::matmul(ffnUp, xNorm, *w.upProj);
    ops::silu(ffnGate, ffnGate);
    ops::mul(ffnGate, ffnGate, ffnUp);
    ops::matmul(ffnDown, ffnGate, *w.downProj);
    ops::add(x, x, ffnDown);
}

void CpuBackend::finalLogits(Tensor& logits, const Tensor& x, const Tensor& finalNorm,
                             const Tensor& lmHead, FinalParams params) {
    if (params.norm.affineOffset != 0.0f) utils::error("CpuBackend: (1+w) RMSNorm not implemented");

    const int seqLen = x.shape[0];
    const int hidden = x.shape[1];
    xNorm.reshape({seqLen, hidden});

    ops::rmsNorm(xNorm, x, finalNorm, params.norm.eps);
    Tensor lastRow = Tensor::rowView(xNorm, seqLen - 1);   // [hidden]
    ops::matmul(logits, lastRow, lmHead);

    if (params.logitSoftcap != 0.0f) {
        const float cap = params.logitSoftcap;
        auto* data = static_cast<float*>(logits.data);
        const size_t n = logits.numElements();
        for (size_t i = 0; i < n; i++) data[i] = cap * std::tanh(data[i] / cap);
    }
}

void CpuBackend::sample(Tensor& nextTokenId, const Tensor& logits, SampleParams) {
    // Greedy argmax (temperature-based sampling is a later addition).
    const auto* data = static_cast<const float*>(logits.data);
    const int vocab  = logits.shape[0];

    int   bestToken = 0;
    float maxLogit  = data[0];
    for (int i = 1; i < vocab; i++) {
        if (data[i] > maxLogit) { maxLogit = data[i]; bestToken = i; }
    }

    static_cast<int32_t*>(nextTokenId.data)[0] = bestToken;
}
