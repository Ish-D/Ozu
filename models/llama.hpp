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

struct LLamaWeights {
    Tensor* tokenEmbeddings;
    std::vector<LayerWeights> layers;
    Tensor* finalLayerNormWeight;
    Tensor* lmHead;
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
    LLamaWeights weights;
    std::vector<KVCacheLayer> kvCache;

public:
    Llama(SafetensorsLoader& loader, const LlamaConfig &config);

    Tensor forward(const std::vector<int>& tokens, int startPos) override;
    timing::TimingMetrics generate(const std::vector<int>& promptTokens,
                                                 int maxTokens,
                                                 const std::function<int(const Tensor&)>& sampler,
                                                 const std::function<bool(int)>& onTokenGenerated) override;
};