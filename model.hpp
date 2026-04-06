#pragma once

#include "tensor.hpp"
#include "utils.hpp"
#include <functional>

class LanguageModel {
public:
    virtual ~LanguageModel() = default;

    virtual Tensor forward(const std::vector<int>& tokens, int startPos) = 0;
    virtual timing::TimingMetrics generate(const std::vector<int>& promptTokens,
                                                 int maxTokens,
                                                 const std::function<int(const Tensor&)>& sampler,
                                                 const std::function<bool(int)>& onTokenGenerated) = 0;
};