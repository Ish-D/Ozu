cmake_minimum_required(VERSION 3.20)
project(Ozu CXX)

# Enforce C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent module
include(FetchContent)

# Fetch nlohmann/json directly from GitHub
FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# Create the executable
add_executable(Ozu main.cpp
                   tensor.cpp
                   ops.cpp
                   model.cpp
                   models/llama.cpp
                   tokenizer.cpp
                   utils.cpp)

# Link the JSON library to the executable
target_link_libraries(Ozu PRIVATE nlohmann_json::nlohmann_json)