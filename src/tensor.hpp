#pragma once

#include <vector>
#include <string>
#include <print>

#include "utils.hpp"

enum class DataType {
    fp32,
    bf16
};

struct Tensor {
    std::string name;
    std::vector<int> shape;
    DataType dataType;
    void* data;

private:
    bool ownsMemory;

public:
    Tensor(std::string name, std::vector<int> shape, DataType dataType, void* dataPtr): name(std::move(name)),
                                                                                        shape(std::move(shape)),
                                                                                        dataType(dataType),
                                                                                        data(dataPtr),
                                                                                        ownsMemory(false) {}

    Tensor(std::vector<int> shape, DataType dataType, std::string name = "buffer"): name(std::move(name)),
                                                                                    shape(std::move(shape)),
                                                                                    dataType(dataType),
                                                                                    ownsMemory(true) {
        size_t bytes = sizeInBytes();
        size_t pageSize = 16384; // Apple silicon page size

        if (posix_memalign(&data, pageSize, bytes) != 0) {
            utils::error("Tensor::Tensor() Failed to allocate memory for {}", name);
        }
    }

    ~Tensor() {
        if (ownsMemory && data != nullptr) {
            free(data);
            data = nullptr;
        }
    }

    // Prevent copying of tensors
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Allow moving of tensors
    Tensor(Tensor&& other) noexcept : name(std::move(other.name)),
                                      shape(std::move(other.shape)),
                                      dataType(other.dataType),
                                      data(other.data),
                                      ownsMemory(other.ownsMemory) {
        other.data = nullptr;
        other.ownsMemory = false;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (ownsMemory && data != nullptr) {
                free(data);
            }

            name = std::move(other.name);
            shape = std::move(other.shape);
            dataType = other.dataType;
            data = other.data;
            ownsMemory = other.ownsMemory;
            other.data = nullptr;
            other.ownsMemory = false;
        }

        return *this;
    }

    [[nodiscard]] size_t numElements() const {
        if (shape.empty()) {
            return 0;
        }

        size_t count = 1;
        for (int dim : shape) {
            count *= dim;
        }

        return count;
    }

    [[nodiscard]] size_t elementSize() const {
        switch (dataType) {
            case DataType::fp32: return 4;
            case DataType::bf16: return 2;
            default: {
                utils::error("Tensor::elementSize(): unsupported dataType for tensor {}", name);
            }
        }
    }

    [[nodiscard]] size_t sizeInBytes() const {
        return numElements() * elementSize();
    }
};