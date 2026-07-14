cmake_minimum_required(VERSION 3.20)
project(Ozu CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

add_executable(Ozu src/main.cpp
                   src/tensor.cpp
                   src/ops.cpp
                   src/model.cpp
                   src/models/llama.cpp
                   src/backends/cpu_backend.cpp
                   src/tokenizer.cpp
                   src/utils.cpp)

target_link_libraries(Ozu PRIVATE nlohmann_json::nlohmann_json)

enable_language(OBJCXX)
set(CMAKE_OBJCXX_STANDARD 23)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
target_sources(Ozu PRIVATE src/backends/metal_backend.mm)
set_source_files_properties(src/backends/metal_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries(Ozu PRIVATE "-framework Metal" "-framework Foundation")