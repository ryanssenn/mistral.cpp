#pragma once

#include "loader/binary_reader.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct Bpe {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, uint32_t> token_to_id;
    std::unordered_map<uint64_t, uint32_t> merge_to_rank;
    std::unordered_map<uint64_t, uint32_t> merge_to_id;

    void load(BinaryReader& reader);

    uint64_t pack(uint32_t left, uint32_t right) const;
    std::vector<uint32_t> get_id(const std::string& token) const;
    uint64_t get_lowest_pair(const std::vector<uint32_t>& tokens) const;
    std::vector<uint32_t> merge(const std::vector<uint32_t>& tokens, uint32_t left, uint32_t right, uint32_t merged) const;
    std::vector<uint32_t> apply_merges(std::vector<uint32_t> tokens) const;
};
