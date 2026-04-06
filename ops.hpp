#pragma once

#include <bit>

#include "tensor.hpp"


inline float bf16Tof32(uint16_t bf16) {
    const uint32_t f32Bits = static_cast<uint32_t>(bf16) << 16;
    return std::bit_cast<float>(f32Bits);
}

namespace ops {
    inline float getFloat(const Tensor& tensor, size_t index);
    void matmul(Tensor& out, const Tensor& a, const Tensor& b);
    void rmsNorm(Tensor& out, const Tensor& a, const Tensor& weight, float eps);
    void add(Tensor& out, const Tensor& a, const Tensor& b);
    void applyRope(Tensor& q, Tensor& k, int position, float ropeTheta);
    void silu(Tensor& out, const Tensor& a);
    void mul(Tensor& out, const Tensor& a, const Tensor& b);
    void softmax(Tensor& out, const Tensor& a);
    void updateCache(Tensor& cache, const Tensor& val, int pos);
    void attention(Tensor& out, const Tensor& q, const Tensor& kCache, const Tensor& vCache, int numQHeads, int numKVHeads, int headDim, int seqPos);
}