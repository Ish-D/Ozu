#include "llama.hpp"

#include <chrono>

Llama::Llama(SafetensorsLoader &loader, const LlamaConfig &config) : config(config) {
    for (int i = 0; i < this->config.numLayers; i++) { // Note: changed from numKVHeads to numLayers
        kvCache.emplace_back(this->config.maxSeqLen, this->config.numKVHeads, this->config.headDim);
    }

    // --- NEW ROBUST GETTER ---
    auto getTensor = [&loader](const std::string& key) -> Tensor* {
        auto it = loader.tensors.find(key);
        if (it == loader.tensors.end()) {
            std::print("\n[FATAL ERROR] Safetensors missing key {}", key);
            std::abort();
        }
        return &it->second;
    };
    // -------------------------

    weights.tokenEmbeddings = getTensor("model.embed_tokens.weight");
    weights.finalLayerNormWeight = getTensor("model.norm.weight");
    auto it = loader.tensors.find("lm_head.weight");
    if (it != loader.tensors.end()) {
        weights.lmHead = &it->second;
    } else {
        weights.lmHead = weights.tokenEmbeddings;
    }

    weights.layers.resize(this->config.numLayers);
    for (int i = 0; i < this->config.numLayers; i++) {
        std::string prefix = "model.layers." + std::to_string(i) + ".";

        weights.layers[i].inputLayerNormWeight = getTensor(prefix + "input_layernorm.weight");

        weights.layers[i].qProjWeight = getTensor(prefix + "self_attn.q_proj.weight");
        weights.layers[i].kProjWeight = getTensor(prefix + "self_attn.k_proj.weight");
        weights.layers[i].vProjWeight = getTensor(prefix + "self_attn.v_proj.weight");
        weights.layers[i].oProjWeight = getTensor(prefix + "self_attn.o_proj.weight");

        weights.layers[i].postAttentionLayerNormWeight = getTensor(prefix + "post_attention_layernorm.weight");

        weights.layers[i].gateProjWeight = getTensor(prefix + "mlp.gate_proj.weight");
        weights.layers[i].upProjWeight = getTensor(prefix + "mlp.up_proj.weight");
        weights.layers[i].downProjWeight = getTensor(prefix + "mlp.down_proj.weight");
    }
}


Tensor Llama::forward(const std::vector<int>& tokens, int startPos) {
    int token = tokens[0];

    int hiddenDim = config.hiddenSize;
    Tensor x({hiddenDim}, DataType::fp32, "x");
    Tensor xNorm({hiddenDim}, DataType::fp32, "xNorm");

    Tensor q({config.numHeads * config.headDim}, DataType::fp32, "q");
    Tensor k({config.numKVHeads * config.headDim}, DataType::fp32, "k");
    Tensor v({config.numKVHeads * config.headDim}, DataType::fp32, "v");
    Tensor attentionOut({hiddenDim}, DataType::fp32, "attentionOut");

    Tensor ffnGate({config.intermediateSize}, DataType::fp32, "ffnGate");
    Tensor ffnUp({config.intermediateSize}, DataType::fp32, "ffnUp");
    Tensor ffnDown({hiddenDim}, DataType::fp32, "ffnDown");

    auto* embedData = static_cast<const uint16_t*>(weights.tokenEmbeddings->data);
    auto* xData = static_cast<float*>(x.data);
    for (int i = 0; i < hiddenDim; i++) {
        xData[i] = bf16Tof32(embedData[token * hiddenDim + i]);
    }

    for (int l = 0; l < config.numLayers; l++) {
        auto& layer = weights.layers[l];
        auto& cache = kvCache[l];

        ops::rmsNorm(xNorm, x, *layer.inputLayerNormWeight, config.rmsNormEps);

        ops::matmul(q, xNorm, *layer.qProjWeight);
        ops::matmul(k, xNorm, *layer.kProjWeight);
        ops::matmul(v, xNorm, *layer.vProjWeight);

        ops::applyRope(q, k, startPos, config.ropeTheta);

        ops::updateCache(cache.kCache, k, startPos);
        ops::updateCache(cache.vCache, v, startPos);

        ops::attention(attentionOut, q, cache.kCache, cache.vCache, config.numHeads, config.numKVHeads, config.headDim, startPos);

        Tensor oProjOut({hiddenDim}, DataType::fp32, "oProjOut");
        ops::matmul(oProjOut, attentionOut, *layer.oProjWeight);
        ops::add(x, x, oProjOut);

        ops::rmsNorm(xNorm, x, *layer.postAttentionLayerNormWeight, config.rmsNormEps);

        ops::matmul(ffnGate, xNorm, *layer.gateProjWeight);
        ops::matmul(ffnUp, xNorm, *layer.upProjWeight);

        ops::silu(ffnGate, ffnGate);
        ops::mul(ffnGate, ffnGate, ffnUp);

        ops::matmul(ffnDown, ffnGate, *layer.downProjWeight);
        ops::add(x, x, ffnDown);
    }

    ops::rmsNorm(xNorm, x, *weights.finalLayerNormWeight, config.rmsNormEps);

    Tensor logits({config.vocabSize}, DataType::fp32, "logits");
    ops::matmul(logits, xNorm, *weights.lmHead);

    return logits;
}

timing::TimingMetrics Llama::generate(const std::vector<int>& promptTokens,
                                                 int maxTokens,
                                                 const std::function<int(const Tensor&)>& sampler,
                                                 const std::function<bool(int)>& onTokenGenerated) {
    timing::TimingMetrics metrics;
    metrics.prefillTokens = promptTokens.size();

    int currentPos = 0;
    Tensor logits({config.vocabSize}, DataType::fp32, "logits");
    const auto prefillStart = std::chrono::high_resolution_clock::now();

    for (int promptToken : promptTokens) {
        logits = forward({promptToken}, currentPos);
        currentPos++;
    }

    const auto prefillEnd = std::chrono::high_resolution_clock::now();
    metrics.prefillTime = std::chrono::duration<double, std::milli>(prefillEnd - prefillStart).count();

    auto decodeStart = std::chrono::high_resolution_clock::now();
    int nextToken = sampler(logits);

    for (int i = 0; i < maxTokens; i++) {
        metrics.decodeTime ++;

        if (!onTokenGenerated(nextToken)) {
            break;
        }

        logits = forward({nextToken}, currentPos);
        nextToken = sampler(logits);
        currentPos++;
    }

    auto decodeEnd = std::chrono::high_resolution_clock::now();
    metrics.decodeTime = std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();

    return metrics;
}