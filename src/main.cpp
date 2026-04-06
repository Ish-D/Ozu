#include <iostream>

#include "tokenizer.hpp"
#include "models/llama.hpp"
#include "utils.hpp"

int greedySample(const Tensor& logits) {
    auto* data = static_cast<float*>(logits.data);
    int vocabSize = logits.shape[0];

    int bestToken = 0;
    float maxLogit = data[0];

    for (int i = 1; i < vocabSize; ++i) {
        if (data[i] > maxLogit) {
            maxLogit = data[i];
            bestToken = i;
        }
    }

    return bestToken;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::print("Error: Missing model directory.\n");
        std::print("Usage: {}/path/to/huggingface/model_folder\n", argv[0]);
        return 1;
    }

    std::filesystem::path modelDir = argv[1];
    std::string configPath    = (modelDir / "config.json").string();
    std::string modelPath     = (modelDir / "model.safetensors").string();
    std::string tokenizerPath = (modelDir / "tokenizer.json").string();

    LlamaConfig config = LlamaConfig::load(configPath);
    SafetensorsLoader loader;

    loader.load(modelPath);

    Llama model(loader, config);

    Tokenizer tokenizer(tokenizerPath);

    auto streamOutput = [&tokenizer](const int token) -> bool {
        if (token == 128009) return false;

        std::print("{}", tokenizer.decode(token));
        return true;
    };

    while (true) {
        std::print("\nUser: ");

        std::string prompt;
        std::getline(std::cin, prompt);

        if (prompt.empty()) continue;
        std::vector<int> promptTokens = tokenizer.encode(prompt);
        std::print("Ozu: ");
        timing::TimingMetrics metrics = model.generate(promptTokens, 20, greedySample, streamOutput);
        metrics.print();
    }

    return 0;
}