#pragma once

#include <vector>
#include <fstream>

#include "../model.hpp"
#include "../backend.hpp"
#include "../safetensors.hpp"

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

class Llama : public LanguageModel {
private:
    Backend& backend;
    LlamaConfig config;

    Tensor* tokenEmbeddings = nullptr;
    Tensor* finalNormWeight = nullptr;
    Tensor* lmHead          = nullptr;
    std::vector<AttnWeights> attnWeights;
    std::vector<FFNWeights>  ffnWeights;
    std::vector<KVCache>     kvCache;

    // Model-owned state (backend owns transient scratch).
    Tensor x;
    Tensor logits;
    Tensor idsScratch;   // i32 token ids handed to embed()
    Tensor nextTokenId;  // i32 [1], written by sample()

    AttnBlockParams attnParams;
    FFNBlockParams  ffnParams;
    FinalParams     finalParams;
    EmbedParams     embedParams;

public:
    Llama(SafetensorsLoader& loader, const LlamaConfig& config, Backend& backend);

    auto forward(const std::vector<int>& tokens, int startPos) -> const Tensor& override;
    timing::TimingMetrics generate(const std::vector<int>& promptTokens,
                                   int maxTokens,
                                   SampleParams sampleParams,
                                   const std::function<bool(int)>& onTokenGenerated) override;
};
