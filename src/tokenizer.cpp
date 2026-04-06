#include "tokenizer.hpp"
#include <fstream>
#include "utils.hpp"
Tokenizer::Tokenizer(const std::string &tokenizerPath) {
    std::ifstream f(tokenizerPath);
    if (!f.is_open()) {
        utils::error("Failed to open tokenizer {}", tokenizerPath);
    }

    json data = json::parse(f);

    auto vocab = data["model"]["vocab"];
    for (auto& [text, id] : vocab.items()) {
        idToToken[id] = text;
        tokenToId[text] = id;
    }
}

std::string Tokenizer::decode(const int id) {
    if (idToToken.contains(id)) {
        std::string token = idToToken[id];

        size_t pos = 0;
        while ((pos = token.find("Ġ", pos)) != std::string::npos) {
            token.replace(pos, 2, " ");
        }

        pos = 0;
        while ((pos = token.find("Ċ", pos)) != std::string::npos) {
            token.replace(pos, 2, "\n");
        }

        return token;
    }

    return "";
}

void Tokenizer::encodeWord(std::string word, std::vector<int>& tokens) {
    while (!word.empty()) {
        bool found = false;

        // Greedily take longest acceptable prefix
        for (size_t len = word.length(); len > 0; --len) {
            std::string substr = word.substr(0, len);
            if (tokenToId.contains(substr)) {
                tokens.push_back(tokenToId[substr]);
                word = word.substr(len);
                found = true;
                break;
            }
        }

        if (!found) {
            word = word.substr(1);
        }
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> tokens;

    tokens.push_back(128000);

    std::string word = "";
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == ' ') {
            if (!word.empty()) encodeWord(word, tokens);
            word = "Ġ";
        }
        else if (text[i] == '\n') {
            if (!word.empty()) encodeWord(word, tokens);
            word = "Ċ";
        }
        else {
            word += text[i];
        }
    }

    if (!word.empty()) encodeWord(word, tokens);

    return tokens;
}
