#pragma once

#include "tensor.hpp"
#include "utils.hpp"
#include "backend.hpp"
#include <functional>

class LanguageModel {
public:
    virtual ~LanguageModel() = default;

    virtual auto forward(const std::vector<int>& tokens, int startPos) -> const Tensor& = 0;
    virtual timing::TimingMetrics generate(const std::vector<int>& promptTokens,
                                           int maxTokens,
                                           SampleParams sampleParams,
                                           const std::function<bool(int)>& onTokenGenerated) = 0;
};
