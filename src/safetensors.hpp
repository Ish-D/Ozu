#pragma once

#include <vector>
#include <string>
#include <unordered_map>

// POSIX headers for memory mapping
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "tensor.hpp"
#include "utils.hpp"

using json = nlohmann::json;

class SafetensorsLoader {
private:
    int fd = -1;
    size_t fileSize = 0;
    char* mappedData = nullptr;

public:
    std::unordered_map<std::string, Tensor> tensors;

    bool load(const std::string& modelPath) {
        fd = open(modelPath.c_str(), O_RDONLY);
        if (fd == -1) {
            utils::error("Safetensorsloader::load failed to open model file {}", modelPath);
        }

        struct stat statInfo;
        if (fstat(fd, &statInfo) == -1) {
            close(fd);
            utils::error("Failed to get file stats for {}", modelPath);
        }

        fileSize = statInfo.st_size;

        mappedData = static_cast<char*>(mmap(nullptr, fileSize, PROT_READ, MAP_SHARED, fd, 0));
        if (mappedData == MAP_FAILED) {
            close(fd);
            utils::error("Failed to map model file {}", modelPath);
        }

        uint64_t headerSize = *reinterpret_cast<uint64_t*>(mappedData);
        if (8 + headerSize > fileSize) {
            utils::error("Invalid Safetensors file: header size exceeds file size {}.", modelPath);
        }

        std::string jsonStr(mappedData + 8, headerSize);
        json header = json::parse(jsonStr);

        char* rawDataStart = mappedData + 8 + headerSize;
        for (auto& element : header.items()) {
            std::string tensorName = element.key();

            if (tensorName == "__metadata__") continue;

            auto tensorInfo = element.value();

            // Parse datatype
            DataType dataType;
            std::string dataTypeStr = tensorInfo["dtype"];
            if      (dataTypeStr == "F32")  dataType = DataType::fp32;
            else if (dataTypeStr == "BF16") dataType = DataType::bf16;
            else utils::error("Unsupported data type in safetensors: {}", dataTypeStr);

            std::vector<int> shape = tensorInfo["shape"].get<std::vector<int>>();

            size_t startOffset = tensorInfo["data_offsets"][0];
            size_t endOffset = tensorInfo["data_offsets"][1];
            size_t fileSizeInBytes = endOffset - startOffset;

            void* dataPtr = static_cast<void*>(rawDataStart + startOffset);

            Tensor tensor(tensorName, std::move(shape), dataType, dataPtr);
            if (tensor.sizeInBytes() != fileSizeInBytes) {
                utils::error("Shape/byte mismatch for tensor {}. File expects {}, Computed {}.", tensorName, fileSizeInBytes, tensor.sizeInBytes());
            }

            tensors.emplace(tensorName, std::move(tensor));
        }

        return true;
    }
};