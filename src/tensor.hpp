#pragma once

#include <vector>
#include <string>
#include <print>

#include "utils.hpp"

enum class Device {
    CPU,
    GPU
};

enum class DataType {
    fp32,
    bf16,
    i32
};

struct Tensor {
    std::string      name;
    std::vector<int> shape;
    DataType         dataType;
    Device           device;
    void*            data;

private:
    bool   ownsMemory;
    size_t allocatedBytes = 0;

public:
    Tensor(std::string name, std::vector<int> shape, const DataType dataType, void* dataPtr, const Device device = Device::CPU): name(std::move(name)),
                                                                                                                                 shape(std::move(shape)),
                                                                                                                                 dataType(dataType),
                                                                                                                                 device(device),
                                                                                                                                 data(dataPtr),
                                                                                                                                 ownsMemory(false) {
        allocatedBytes = sizeInBytes();
    }

    Tensor(std::vector<int> shape, const DataType dataType, std::string name = "buffer", const Device device = Device::CPU): name(std::move(name)),
                                                                                                                             shape(std::move(shape)),
                                                                                                                             dataType(dataType),
                                                                                                                             device(device),
                                                                                                                             ownsMemory(true) {
        allocatedBytes = sizeInBytes();
        size_t pageSize = 16384; // Apple silicon page size

        if (posix_memalign(&data, pageSize, allocatedBytes) != 0) {
            utils::error("Tensor::Tensor() Failed to allocate memory for {}", this->name);
        }
    }

    void reshape(std::vector<int> newShape) {
        shape = std::move(newShape);

        if (sizeInBytes() > allocatedBytes) {
            utils::error("Tensor()::reshape exceeds allocated capacity for {}", this->name);
        }
    }

    static Tensor rowView(const Tensor& src, int rowIdx) {
        if (src.shape.size() < 2) {
            utils::error("Tensor::rowView(): src tensor ({}) must be multidimensional", src.name);
        }

        int rowElems = 1;
        for (size_t i = 1; i < src.shape.size(); ++i) rowElems *= src.shape[i];

        size_t offsetBytes = static_cast<size_t>(rowIdx) * rowElems * src.elementSize();
        std::vector<int> newShape(src.shape.begin() + 1, src.shape.end());

        return Tensor(src.name + "_row",
                      std::move(newShape),
                      src.dataType, static_cast<char*>(src.data) + offsetBytes,
                      src.device);
    }

    void releaseOwnedMemory() {
        if (!ownsMemory || data == nullptr) return;

        switch (device) {
            case Device::CPU: free(data); break;
            case Device::GPU: break; // TODO
        }

        data = nullptr;
    }

    ~Tensor() {
        releaseOwnedMemory();
    }

    // Prevent copying of tensors
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Allow moving of tensors
    Tensor(Tensor&& other) noexcept : name(std::move(other.name)),
                                      shape(std::move(other.shape)),
                                      dataType(other.dataType),
                                      device(other.device),
                                      data(other.data),
                                      ownsMemory(other.ownsMemory),
                                      allocatedBytes(other.allocatedBytes) {
        other.data = nullptr;
        other.ownsMemory = false;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            releaseOwnedMemory();

            name           = std::move(other.name);
            shape          = std::move(other.shape);
            dataType       = other.dataType;
            device         = other.device;
            data           = other.data;
            ownsMemory     = other.ownsMemory;
            allocatedBytes = other.allocatedBytes;

            other.data           = nullptr;
            other.ownsMemory     = false;
            other.allocatedBytes = 0;
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
            case DataType::i32:  return 4;
            default: {
                utils::error("Tensor::elementSize(): unsupported dataType for tensor {}", this->name);
            }
        }
    }

    [[nodiscard]] size_t sizeInBytes() const {
        return numElements() * elementSize();
    }
};