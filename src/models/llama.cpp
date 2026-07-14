#include "llama.hpp"

#include <cassert>
#include <chrono>

Llama::Llama(SafetensorsLoader& loader, const LlamaConfig& config, Backend& backend)
    : backend(backend),
      config(config),
      x         (backend.allocate({config.maxSeqLen, config.hiddenSize}, DataType::fp32, "x")),
      logits    (backend.allocate({config.vocabSize}, DataType::fp32, "logits")),
      idsScratch(backend.allocate({config.maxSeqLen}, DataType::i32, "idsScratch")),
      nextTokenId(backend.allocate({1}, DataType::i32, "nextTokenId")) {

    const int kvDim = config.numKVHeads * config.headDim;
    for (int i = 0; i < config.numLayers; i++) {
        kvCache.push_back(KVCache{
            backend.allocate({config.maxSeqLen, kvDim}, DataType::fp32, "kCache"),
            backend.allocate({config.maxSeqLen, kvDim}, DataType::fp32, "vCache"),
            config.maxSeqLen,
            false});
    }

    auto getTensor = [&loader](const std::string& key) -> Tensor* {
        auto it = loader.tensors.find(key);
        if (it == loader.tensors.end()) {
            utils::error("Safetensors missing key {}", key);
        }
        return &it->second;
    };

    tokenEmbeddings = getTensor("model.embed_tokens.weight");
    finalNormWeight = getTensor("model.norm.weight");
    auto it = loader.tensors.find("lm_head.weight");
    lmHead = (it != loader.tensors.end()) ? &it->second : tokenEmbeddings;

    attnWeights.resize(config.numLayers);
    ffnWeights.resize(config.numLayers);
    for (int i = 0; i < config.numLayers; i++) {
        std::string prefix = "model.layers." + std::to_string(i) + ".";

        attnWeights[i].inputNorm = getTensor(prefix + "input_layernorm.weight");
        attnWeights[i].qProj     = getTensor(prefix + "self_attn.q_proj.weight");
        attnWeights[i].kProj     = getTensor(prefix + "self_attn.k_proj.weight");
        attnWeights[i].vProj     = getTensor(prefix + "self_attn.v_proj.weight");
        attnWeights[i].oProj     = getTensor(prefix + "self_attn.o_proj.weight");

        // Llama applies post_attention_layernorm as the pre-FFN norm (not a sandwich).
        ffnWeights[i].preNorm  = getTensor(prefix + "post_attention_layernorm.weight");
        ffnWeights[i].gateProj = getTensor(prefix + "mlp.gate_proj.weight");
        ffnWeights[i].upProj   = getTensor(prefix + "mlp.up_proj.weight");
        ffnWeights[i].downProj = getTensor(prefix + "mlp.down_proj.weight");
    }

    attnParams  = {config.numHeads, config.numKVHeads, config.headDim, 0,
                   {config.rmsNormEps, 0.0f}, config.ropeTheta, RopeType::Default, 1.0f, 0.0f, 0, false};
    ffnParams   = {{config.rmsNormEps, 0.0f}, ActFn::Silu, config.intermediateSize, 0, 0, 0};
    finalParams = {{config.rmsNormEps, 0.0f}, 0.0f};
    embedParams = {1.0f};
}

auto Llama::forward(const std::vector<int>& tokens, int startPos) -> const Tensor& {
    const int seqLen = tokens.size();
    assert(seqLen >= 1);
    assert(startPos + seqLen <= config.maxSeqLen);

    idsScratch.reshape({seqLen});
    auto* ids = static_cast<int32_t*>(idsScratch.data);
    for (int s = 0; s < seqLen; s++) ids[s] = tokens[s];

    backend.embed(x, *tokenEmbeddings, idsScratch, embedParams);

    for (int l = 0; l < config.numLayers; l++) {
        attnParams.startPos = startPos;
        backend.attentionBlock(x, attnWeights[l], kvCache[l], attnParams);
        backend.ffnBlock(x, ffnWeights[l], ffnParams);
    }

    backend.finalLogits(logits, x, *finalNormWeight, *lmHead, finalParams);
    return logits;
}

timing::TimingMetrics Llama::generate(const std::vector<int>& promptTokens,
                                      int maxTokens,
                                      SampleParams sampleParams,
                                      const std::function<bool(int)>& onTokenGenerated) {
    timing::TimingMetrics metrics;
    metrics.prefillTokens = promptTokens.size();

    const auto readNext = [this]() -> int {
        return static_cast<const int32_t*>(nextTokenId.data)[0];
    };

    int currentPos = 0;
    const auto prefillStart = std::chrono::high_resolution_clock::now();
    backend.beginSequence();
    forward(promptTokens, currentPos);
    backend.sample(nextTokenId, logits, sampleParams);
    backend.endSequence();
    backend.synchronize();
    const auto prefillEnd = std::chrono::high_resolution_clock::now();
    metrics.prefillTime = std::chrono::duration<double, std::milli>(prefillEnd - prefillStart).count();

    currentPos = promptTokens.size();
    int nextToken = readNext();

    const auto decodeStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < maxTokens; i++) {
        metrics.decodeTokens++;

        if (!onTokenGenerated(nextToken)) break;

        backend.beginSequence();
        forward({nextToken}, currentPos);
        backend.sample(nextTokenId, logits, sampleParams);
        backend.endSequence();
        backend.synchronize();

        nextToken = readNext();
        currentPos++;
    }
    const auto decodeEnd = std::chrono::high_resolution_clock::now();
    metrics.decodeTime = std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();

    return metrics;
}
