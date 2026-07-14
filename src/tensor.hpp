#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdlib>

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

inline size_t dataTypeSize(const DataType dt) {
    switch (dt) {
        case DataType::fp32: return 4;
        case DataType::bf16: return 2;
        case DataType::i32:  return 4;
    }
    utils::error("dataTypeSize(): unsupported dataType");
}

// Backing allocation, shared by a tensor and all views into it. Holds the
// host-visible pointer and (later) the device buffer handle used for GPU binding.
// One Storage backs one CPU allocation or one mmap'd safetensors shard.
struct Storage {
    void*  host     = nullptr;
    void*  device   = nullptr;   // e.g. id<MTLBuffer>; null on CPU
    size_t bytes    = 0;
    Device where    = Device::CPU;
    bool   ownsHost = false;     // free host on destruction (CPU allocations only)

    Storage() = default;
    Storage(const Storage&)            = delete;
    Storage& operator=(const Storage&) = delete;

    ~Storage() {
        if (ownsHost && host) std::free(host);
        // device buffer release is the GPU backend's responsibility (TODO)
    }
};

// Typed, shaped view into a Storage at a byte offset. Views share the Storage;
// GPU binding is (storage->device, offsetBytes). `data` caches the host pointer.
struct Tensor {
    std::shared_ptr<Storage> storage;
    size_t           offsetBytes = 0;
    void*            data = nullptr;   // cached: storage->host + offsetBytes
    std::vector<int> shape;
    DataType         dataType;
    Device           device = Device::CPU;
    std::string      name;

private:
    size_t capacityBytes = 0;          // bytes available in storage from offsetBytes

    void refreshData() {
        data = (storage && storage->host)
                   ? static_cast<char*>(storage->host) + offsetBytes
                   : nullptr;
    }

public:
    // Owning allocation, page-aligned for zero-copy GPU buffer wrapping later.
    Tensor(std::vector<int> shape, const DataType dataType, std::string name = "buffer", const Device device = Device::CPU)
        : shape(std::move(shape)), dataType(dataType), device(device), name(std::move(name)) {
        storage           = std::make_shared<Storage>();
        storage->where    = device;
        storage->bytes    = sizeInBytes();
        storage->ownsHost = true;

        constexpr size_t pageSize = 16384; // Apple silicon page size
        if (posix_memalign(&storage->host, pageSize, storage->bytes) != 0) {
            utils::error("Tensor::Tensor() failed to allocate memory for {}", this->name);
        }

        capacityBytes = storage->bytes;
        refreshData();
    }

    // Adopt an external host pointer as its own storage (backend.adopt).
    Tensor(std::string name, std::vector<int> shape, const DataType dataType, void* dataPtr, const Device device = Device::CPU)
        : shape(std::move(shape)), dataType(dataType), device(device), name(std::move(name)) {
        storage           = std::make_shared<Storage>();
        storage->host     = dataPtr;
        storage->where    = device;
        storage->bytes    = sizeInBytes();
        storage->ownsHost = false;

        capacityBytes = storage->bytes;
        refreshData();
    }

    // View into an existing shared storage at a byte offset (safetensors shard, rowView).
    Tensor(std::shared_ptr<Storage> storage, const size_t offsetBytes, std::vector<int> shape, const DataType dataType, std::string name)
        : storage(std::move(storage)), offsetBytes(offsetBytes), shape(std::move(shape)), dataType(dataType), name(std::move(name)) {
        device        = this->storage->where;
        capacityBytes = this->storage->bytes - offsetBytes;
        refreshData();
    }

    void reshape(std::vector<int> newShape) {
        shape = std::move(newShape);

        if (sizeInBytes() > capacityBytes) {
            utils::error("Tensor()::reshape exceeds allocated capacity for {}", this->name);
        }
    }

    static Tensor rowView(const Tensor& src, int rowIdx) {
        if (src.shape.size() < 2) {
            utils::error("Tensor::rowView(): src tensor ({}) must be multidimensional", src.name);
        }

        int rowElems = 1;
        for (size_t i = 1; i < src.shape.size(); ++i) rowElems *= src.shape[i];

        size_t offset = src.offsetBytes + static_cast<size_t>(rowIdx) * rowElems * src.elementSize();
        std::vector<int> newShape(src.shape.begin() + 1, src.shape.end());

        return Tensor(src.storage, offset, std::move(newShape), src.dataType, src.name + "_row");
    }

    // Move-only: storage ownership transfers, views release via shared_ptr refcount.
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept            = default;
    Tensor& operator=(Tensor&&) noexcept = default;

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
        return dataTypeSize(dataType);
    }

    [[nodiscard]] size_t sizeInBytes() const {
        return numElements() * elementSize();
    }
};
