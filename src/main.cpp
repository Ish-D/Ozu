#include <iostream>

#include "tokenizer.hpp"
#include "models/llama.hpp"
#include "backends/cpu_backend.hpp"
#include "utils.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::print("Error: Missing model directory.\n");
        std::print("Usage: {} /path/to/huggingface/model_folder [prompt]\n", argv[0]);
        return 1;
    }

    std::filesystem::path modelDir = argv[1];
    std::string configPath    = (modelDir / "config.json").string();
    std::string modelPath     = (modelDir / "model.safetensors").string();
    std::string tokenizerPath = (modelDir / "tokenizer.json").string();

    LlamaConfig config = LlamaConfig::load(configPath);
    SafetensorsLoader loader;
    loader.load(modelPath);

    ScratchDims dims{
        config.maxSeqLen,
        config.hiddenSize,
        config.numHeads * config.headDim,
        config.numKVHeads * config.headDim,
        config.intermediateSize};
    CpuBackend backend(dims);

    Llama model(loader, config, backend);
    Tokenizer tokenizer(tokenizerPath);

    // Non-interactive mode:
    // `Ozu <modelDir> <prompt>` generates a fixed number of greedy tokens and prints their ids, then exits.
    if (argc >= 3) {
        std::vector<int> ids;
        auto collect = [&ids](const int token) -> bool {
            if (token == 128009) return false;
            ids.push_back(token);
            return true;
        };
        std::vector<int> promptTokens = tokenizer.encode(argv[2]);
        model.generate(promptTokens, 16, SampleParams{}, collect);
        for (const int id : ids) std::print("{} ", id);
        std::print("\n");
        return 0;
    }

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
        timing::TimingMetrics metrics = model.generate(promptTokens, 20, SampleParams{}, streamOutput);
        metrics.print();
    }

    return 0;
}
