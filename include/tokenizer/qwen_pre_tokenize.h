#pragma once

#include <string>
#include <vector>

struct QwenPreTokenizer {
    static constexpr const char* pattern =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

    static void expect_pattern(const std::string& from_file);
    std::vector<std::string> split(const std::string& text) const;
};
