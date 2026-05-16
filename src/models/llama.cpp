#include "llama.hpp"

#include <chrono>

Llama::Llama(SafetensorsLoader &loader, const LlamaConfig &config) : config(config), activations(config) {
    for (int i = 0; i < this->config.numLayers; i++) { // Note: changed from numKVHeads to numLayers
        kvCache.emplace_back(this->config.maxSeqLen, this->config.numKVHeads, this->config.headDim);
    }

    auto getTensor = [&loader](const std::string& key) -> Tensor* {
        auto it = loader.tensors.find(key);
        if (it == loader.tensors.end()) {
            utils::error("Safetensors missing key {}", key);
        }
        return &it->second;
    };

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


auto Llama::forward(const std::vector<int>& tokens, int startPos) -> const Tensor& {
    const int seqLen  = tokens.size();
    const int hiddenDim = config.hiddenSize;

    assert(seqLen >= 1);
    assert(startPos + seqLen <= config.maxSeqLen);

    activations.setSeqLen(seqLen, config);

    // TODO: deal with correct data type more seamlessly
    auto* embedData = static_cast<const uint16_t*>(weights.tokenEmbeddings->data);
    auto* xData = static_cast<float*>(activations.x.data);
    for (int s = 0; s < seqLen; s++) {
        const int token = tokens[s];
        for (int i = 0; i < hiddenDim; i++) {
            xData[s * hiddenDim + i] = bf16Tof32(embedData[token * hiddenDim + i]);
        }
    }

    for (int l = 0; l < config.numLayers; l++) {
        auto& layer = weights.layers[l];
        auto& cache = kvCache[l];

        ops::rmsNorm(activations.xNorm, activations.x, *layer.inputLayerNormWeight, config.rmsNormEps);

        ops::matmul(activations.q, activations.xNorm, *layer.qProjWeight);
        ops::matmul(activations.k, activations.xNorm, *layer.kProjWeight);
        ops::matmul(activations.v, activations.xNorm, *layer.vProjWeight);

        ops::applyRope(activations.q, activations.k, startPos, config.ropeTheta, config.headDim);

        ops::updateCache(cache.kCache, activations.k, startPos);
        ops::updateCache(cache.vCache, activations.v, startPos);

        ops::attention(activations.attentionOut, activations.q, cache.kCache, cache.vCache, activations.attentionScores, config.numHeads, config.numKVHeads, config.headDim, startPos);

        ops::matmul(activations.oProjOut, activations.attentionOut, *layer.oProjWeight);
        ops::add(activations.x, activations.x, activations.oProjOut);

        ops::rmsNorm(activations.xNorm, activations.x, *layer.postAttentionLayerNormWeight, config.rmsNormEps);

        ops::matmul(activations.ffnGate, activations.xNorm, *layer.gateProjWeight);
        ops::matmul(activations.ffnUp, activations.xNorm, *layer.upProjWeight);

        ops::silu(activations.ffnGate, activations.ffnGate);
        ops::mul(activations.ffnGate, activations.ffnGate, activations.ffnUp);

        ops::matmul(activations.ffnDown, activations.ffnGate, *layer.downProjWeight);
        ops::add(activations.x, activations.x, activations.ffnDown);
    }

    ops::rmsNorm(activations.xNorm, activations.x, *weights.finalLayerNormWeight, config.rmsNormEps);
    ops::matmul(activations.logits, activations.xNormLastRow, *weights.lmHead);

    return activations.logits;
}

timing::TimingMetrics Llama::generate(const std::vector<int>& promptTokens,
                                                 int maxTokens,
                                                 const std::function<int(const Tensor&)>& sampler,
                                                 const std::function<bool(int)>& onTokenGenerated) {
    timing::TimingMetrics metrics;
    metrics.prefillTokens = promptTokens.size();

    int currentPos = 0;
    const auto prefillStart = std::chrono::high_resolution_clock::now();
    forward(promptTokens, currentPos);
    const auto prefillEnd = std::chrono::high_resolution_clock::now();
    metrics.prefillTime = std::chrono::duration<double, std::milli>(prefillEnd - prefillStart).count();

    currentPos = promptTokens.size();
    auto decodeStart = std::chrono::high_resolution_clock::now();
    int nextToken = sampler(activations.logits);

    for (int i = 0; i < maxTokens; i++) {
        metrics.decodeTokens++;

        if (!onTokenGenerated(nextToken)) {
            break;
        }

        forward({nextToken}, currentPos);
        nextToken = sampler(activations.logits);
        currentPos++;
    }

    auto decodeEnd = std::chrono::high_resolution_clock::now();
    metrics.decodeTime = std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();

    return metrics;
}