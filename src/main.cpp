#include <iostream>
#include <memory>
#include <string>

#include "tokenizer.hpp"
#include "models/llama.hpp"
#include "backends/cpu_backend.hpp"
#include "backends/metal_backend.hpp"
#include "utils.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::print("Usage: {} /path/to/huggingface/model_folder <cpu|gpu> [prompt]\n", argv[0]);
        return 1;
    }

    std::filesystem::path modelDir = argv[1];
    std::string configPath    = (modelDir / "config.json").string();
    std::string modelPath     = (modelDir / "model.safetensors").string();
    std::string tokenizerPath = (modelDir / "tokenizer.json").string();

    std::string deviceArg = argv[2];
    if (deviceArg != "cpu" && deviceArg != "gpu") {
        std::print("Unknown device '{}'. Expected 'cpu' or 'gpu'.\n", deviceArg);
        return 1;
    }

    LlamaConfig config = LlamaConfig::load(configPath);

    ScratchDims dims{
        config.maxSeqLen,
        config.hiddenSize,
        config.numHeads * config.headDim,
        config.numKVHeads * config.headDim,
        config.intermediateSize};

    std::unique_ptr<Backend> backend;
    if (deviceArg == "gpu") backend = std::make_unique<MetalBackend>(dims);
    else                    backend = std::make_unique<CpuBackend>(dims);

    SafetensorsLoader loader;
    loader.load(modelPath, *backend);

    Llama model(loader, config, *backend);
    Tokenizer tokenizer(tokenizerPath);

    // Non-interactive: `Ozu <modelDir> <device> <prompt>` generates a fixed number
    // of greedy tokens and prints their ids, then exits.
    if (argc >= 4) {
        std::vector<int> ids;
        auto collect = [&ids](const int token) -> bool {
            if (token == 128009) return false;
            ids.push_back(token);
            return true;
        };
        std::vector<int> promptTokens = tokenizer.encode(argv[3]);
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
