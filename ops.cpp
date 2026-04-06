#include <cassert>
#include <cmath>

#include "ops.hpp"

#include <algorithm>

namespace ops {
    inline float getFloat(const Tensor& tensor, size_t index) {
        if (tensor.dataType == DataType::fp32) return static_cast<const float*>(tensor.data)[index];
        else if (tensor.dataType == DataType::bf16) return bf16Tof32(static_cast<const uint16_t*>(tensor.data)[index]);
    }

    void matmul(Tensor& out, const Tensor& a, const Tensor& b) {
        assert(b.shape.size() == 2 && "Weight must be 2d, [out_features, in_features]");

        int inFeatures = b.shape[1];
        int outFeatures = b.shape[0];

        assert(a.shape.back() == inFeatures && "Input features must match  weight in_features");

        auto* outData = static_cast<float*>(out.data);
        for (int i = 0; i < outFeatures; i++) {
            float sum = 0.0f;
            for (int j = 0; j < inFeatures; j++) {
                float aVal = getFloat(a, j);
                float bVal = getFloat(b, i * inFeatures + j);

                sum += aVal * bVal;
            }
            outData[i] = sum;
        }
    }

    void rmsNorm(Tensor& out, const Tensor& a, const Tensor& weight, float eps) {
        const size_t size = a.numElements();
        auto* outData = static_cast<float*>(out.data);

        float sumSquares = 0.0f;
        for (int i = 0; i < size; i++) {
            float val = getFloat(a, i);
            sumSquares += val * val;
        }

        float invRMS = 1.0f / std::sqrt((sumSquares / size) + eps);

        for (int i = 0; i < size; i++) {
            float normalizedVal = getFloat(a, i) * invRMS;
            outData[i] = normalizedVal * getFloat(weight, i);
        }
    }

    void add(Tensor& out, const Tensor& a, const Tensor& b) {
        assert(a.numElements() == b.numElements() && "Tensors must have same number of elements");

        size_t size = a.numElements();
        auto* outData = static_cast<float*>(out.data);

        for (int i = 0; i < size; i++) {
            outData[i] = getFloat(a, i) + getFloat(b, i);
        }
    }

    void applyRope(Tensor &q, Tensor &k, int position, float ropeTheta) {
        assert(q.dataType == DataType::fp32 && k.dataType == DataType::fp32 && "Q/K datatypes must match");

        auto* qData = static_cast<float*>(q.data);
        auto* kData = static_cast<float*>(k.data);

        const int headDim = 64;
        const size_t qElements = q.numElements();
        const size_t kElements = k.numElements();

        // Process Q (Split Halves)
        for (int h = 0; h < qElements; h += headDim) {
            for (int d = 0; d < headDim / 2; d++) {
                int i1 = h + d;
                int i2 = h + d + headDim / 2; // Pair with the element half-way across the head

                // The frequency formula for split halves uses d * 2
                float freq = 1.0f / std::pow(ropeTheta, static_cast<float>(d * 2) / headDim);
                float val = static_cast<float>(position) * freq;
                float cos_val = std::cos(val);
                float sin_val = std::sin(val);

                float q1 = qData[i1];
                float q2 = qData[i2];
                qData[i1] = q1 * cos_val - q2 * sin_val;
                qData[i2] = q1 * sin_val + q2 * cos_val;
            }
        }

        // Process K (Split Halves)
        for (int h = 0; h < kElements; h += headDim) {
            for (int d = 0; d < headDim / 2; d++) {
                int i1 = h + d;
                int i2 = h + d + headDim / 2;

                float freq = 1.0f / std::pow(ropeTheta, static_cast<float>(d * 2) / headDim);
                float val = static_cast<float>(position) * freq;
                float cos_val = std::cos(val);
                float sin_val = std::sin(val);

                float k1 = kData[i1];
                float k2 = kData[i2];
                kData[i1] = k1 * cos_val - k2 * sin_val;
                kData[i2] = k1 * sin_val + k2 * cos_val;
            }
        }
    }

    void silu(Tensor &out, const Tensor &a) {
        assert(out.numElements() == a.numElements() && "Tensor sizes must match in SiLU");

        size_t size = a.numElements();
        auto* outData = static_cast<float*>(out.data);

        for (int i = 0; i < size; i++) {
            float val = getFloat(a, i);
            outData[i] = val / (1.0f + std::exp(-val));
        }
    }

    void mul(Tensor& out, const Tensor& a, const Tensor& b) {
        assert(a.numElements() == b.numElements() && "Tensor sizes must match in elementwise mul");
        assert(out.numElements() == a.numElements() && "Output tensor size must match input tensor size in mul");

        size_t size = a.numElements();
        auto* outData = static_cast<float*>(out.data);

        for (int i = 0; i < size; i++) {
            outData[i] = getFloat(a, i) * getFloat(b, i);
        }
    }

    void softmax(Tensor& out, const Tensor& a) {
        assert(out.shape == a.shape && "Tensor sizes must match in softmax");

        const int lastDim = a.shape.back();
        const int numRows = a.numElements() / lastDim;

        auto* outData = static_cast<float*>(out.data);

        for (int row = 0; row < numRows; row++) {
            int rowOffset = row * lastDim;

            float maxVal = getFloat(a, rowOffset);
            for (int i = 1; i < lastDim; i++) {
                float val = getFloat(a, rowOffset + i);
                maxVal = std::max(val, maxVal);
            }

            float sumExp = 0.0f;
            for (int i = 0; i < lastDim; i++) {
                float val = std::exp(getFloat(a, rowOffset + i) - maxVal);
                outData[rowOffset + i] = val;
                sumExp += val;
            }

            for (int i = 0; i < lastDim; i++) {
                outData[rowOffset + i] /= sumExp;
            }
        }
    }

    void updateCache(Tensor& cache, const Tensor& val, int pos) {
        assert(val.numElements() == cache.shape[1] && "[updateCache] Value size must match cache row size");
        assert(pos < cache.shape[0] && "[updateCache] Position exceeds maximum cache sequence length");

        auto* cacheData = static_cast<float*>(cache.data);

        int rowSize = val.numElements();
        int offset = pos * rowSize;

        for (int i = 0; i < rowSize; i++) {
            cacheData[offset + i] = getFloat(val, i);
        }
    }

    void attention(Tensor &out, const Tensor &q, const Tensor &kCache, const Tensor &vCache, int numQHeads, int numKVHeads, int headDim, int seqPos) {
        int kvMul = numQHeads / numKVHeads;
        float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

        auto* outData = static_cast<float*>(out.data);
        auto* kCacheData = static_cast<const float*>(kCache.data);
        auto* vCacheData = static_cast<const float*>(vCache.data);

        int numTokens = seqPos + 1;
        std::vector<float> scores(numTokens);

        for (int h = 0; h < numQHeads; h++) {
            int kvHead = h / kvMul;

            for (int t = 0; t < numTokens; t++) {
                float score = 0.0f;

                for (int d = 0; d < headDim; d++) {
                    float qVal = getFloat(q, h * headDim + d);
                    float kVal = kCacheData[t * (numKVHeads * headDim) + (kvHead * headDim) + d];

                    score += qVal * kVal;
                }

                scores[t] = score * scale;
            }

            float maxScore = scores[0];
            for (int t = 1; t < numTokens; t++) {
                maxScore = std::max(scores[t], maxScore);
            }

            float sumExp = 0.0f;
            for (int t = 0; t < numTokens; t++) {
                scores[t] = std::exp(scores[t] - maxScore);
                sumExp += scores[t];
            }

            for (int t = 0; t < numTokens; t++) {
                scores[t] /= sumExp;
            }

            for (int d = 0; d < headDim; d++) {
                float outVal = 0.0f;
                for (int t = 0; t < numTokens; t++) {
                    float vVal = vCacheData[t * (numKVHeads * headDim) + (kvHead * headDim) + d];
                    outVal += scores[t] * vVal;
                }

                outData[h * headDim + d] = outVal;
            }
        }
    }
}
