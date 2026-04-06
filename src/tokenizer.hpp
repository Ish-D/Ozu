#pragma once

#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Tokenizer {
private:
    std::unordered_map<int, std::string> idToToken;
    std::unordered_map<std::string, int> tokenToId;
public:
    explicit Tokenizer(const std::string& tokenizerPath);
    void encodeWord(std::string word, std::vector<int>& tokens);
    std::string decode(int id);
    std::vector<int> encode(const std::string& text);
};