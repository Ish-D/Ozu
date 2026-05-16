#pragma once

#include <vector>
#include <fstream>

#include "../model.hpp"
#include "../safetensors.hpp"
#include "../ops.hpp"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

struct LlamaConfig {
    int vocabSize;
    int hiddenSize;
    int intermediateSize;
    int numLayers;
    int numHeads;
    int numKVHeads;
    int headDim;
    float rmsNormEps;
    float ropeTheta;
    int maxSeqLen;

    static LlamaConfig load(const std::string& configPath) {
        std::ifstream f(configPath);
        if (!f.is_open()) {
            std::print("Failed to open config file {}.", configPath);
            std::abort();
        }

        json data = json::parse(f);
        LlamaConfig config;

        config.vocabSize = data["vocab_size"];
        config.hiddenSize = data["hidden_size"];
        config.intermediateSize = data["intermediate_size"];
        config.numLayers = data["num_hidden_layers"];
        config.numHeads = data["num_attention_heads"];

        config.numKVHeads = data.value("num_key_value_heads", config.numHeads);
        config.headDim = config.hiddenSize / config.numHeads;
        config.rmsNormEps = data["rms_norm_eps"];
        config.ropeTheta = data.value("rope_theta", 10000.0f);

        config.maxSeqLen = 2048;

        return config;
    }
};

struct LayerWeights {
    Tensor* inputLayerNormWeight;
    Tensor* postAttentionLayerNormWeight;
    Tensor* qProjWeight;
    Tensor* kProjWeight;
    Tensor* vProjWeight;
    Tensor* oProjWeight;
    Tensor* gateProjWeight;
    Tensor* upProjWeight;
    Tensor* downProjWeight;
};

struct LlamaWeights {
    Tensor* tokenEmbeddings;
    std::vector<LayerWeights> layers;
    Tensor* finalLayerNormWeight;
    Tensor* lmHead;
};

struct LlamaActivations {
    Tensor x;
    Tensor xNorm;
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor attentionOut;
    Tensor oProjOut;
    Tensor ffnGate;
    Tensor ffnUp;
    Tensor ffnDown;
    Tensor logits;
    Tensor xNormLastRow;
    Tensor attentionScores;

    explicit LlamaActivations(const LlamaConfig& config) : x            ({config.maxSeqLen, config.hiddenSize},                  DataType::fp32, "x"),
                                                           xNorm        ({config.maxSeqLen, config.hiddenSize},                  DataType::fp32, "xNorm"),
                                                           q            ({config.maxSeqLen, config.numHeads * config.headDim},   DataType::fp32, "q"),
                                                           k            ({config.maxSeqLen, config.numKVHeads * config.headDim}, DataType::fp32, "k"),
                                                           v            ({config.maxSeqLen, config.numKVHeads * config.headDim}, DataType::fp32, "v"),
                                                           attentionOut ({config.maxSeqLen, config.hiddenSize},                  DataType::fp32, "attentionOut"),
                                                           oProjOut     ({config.maxSeqLen, config.hiddenSize},                  DataType::fp32, "oProjOut"),
                                                           ffnGate      ({config.maxSeqLen, config.intermediateSize},            DataType::fp32, "ffnGate"),
                                                           ffnUp        ({config.maxSeqLen, config.intermediateSize},            DataType::fp32, "ffnUp"),
                                                           ffnDown      ({config.maxSeqLen, config.hiddenSize},                  DataType::fp32, "ffnDown"),
                                                           logits       ({config.vocabSize},                   DataType::fp32, "logits"),
                                                           xNormLastRow ("xNormLastRow", {config.hiddenSize}, DataType::fp32, xNorm.data),
                                                           attentionScores({config.maxSeqLen}, DataType::fp32, "attentionScores") {}


    void setSeqLen(int seqLen, const LlamaConfig& config) {
        x.reshape(           {seqLen, config.hiddenSize});
        xNorm.reshape(       {seqLen, config.hiddenSize});
        q.reshape(           {seqLen, config.numHeads * config.headDim});
        k.reshape(           {seqLen, config.numKVHeads * config.headDim});
        v.reshape(           {seqLen, config.numKVHeads * config.headDim});
        attentionOut.reshape({seqLen, config.hiddenSize});
        oProjOut.reshape(    {seqLen, config.hiddenSize});
        ffnGate.reshape(     {seqLen, config.intermediateSize});
        ffnUp.reshape(       {seqLen, config.intermediateSize});
        ffnDown.reshape(     {seqLen, config.hiddenSize});

        size_t rowBytes = static_cast<size_t>(config.hiddenSize) * xNorm.elementSize();
        xNormLastRow.data = static_cast<char*>(xNorm.data) + (seqLen - 1) * rowBytes;
    }
};

struct KVCacheLayer {
    Tensor kCache;
    Tensor vCache;

    KVCacheLayer(int maxSeqLen, int numKVHeads, int headDim) : kCache({maxSeqLen, numKVHeads * headDim}, DataType::fp32, "k_cache"),
                                                               vCache({maxSeqLen, numKVHeads * headDim}, DataType::fp32, "v_cache") {}
};

class Llama : public LanguageModel {
private:
    LlamaConfig config;
    LlamaWeights weights;
    LlamaActivations activations;
    std::vector<KVCacheLayer> kvCache;

public:
    Llama(SafetensorsLoader& loader, const LlamaConfig &config);

    auto forward(const std::vector<int>& tokens, int startPos) -> const Tensor& override;
    timing::TimingMetrics generate(const std::vector<int>& promptTokens,
                                                 int maxTokens,
                                                 const std::function<int(const Tensor&)>& sampler,
                                                 const std::function<bool(int)>& onTokenGenerated) override;
};