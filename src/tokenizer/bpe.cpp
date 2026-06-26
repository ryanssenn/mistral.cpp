#include "tokenizer/bpe.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

// Loads the BPE tables from the MOG header.
// This gives us the vocabulary and the list of "these two tokens can become
// this bigger token" rules. Qwen-specific code decides what the starting tokens
// are; this file only knows how to merge them.
void Bpe::load(BinaryReader& reader) {
    uint32_t vocab_count = reader.read_u32();
    uint32_t max_id = 0;

    std::vector<std::pair<std::string, uint32_t>> vocab_entries;
    vocab_entries.reserve(vocab_count);

    for (uint32_t i = 0; i < vocab_count; i++){
        std::string token = reader.read_string();
        uint32_t id = reader.read_u32();
        vocab_entries.emplace_back(token, id);
        max_id = std::max(max_id, id);
    }

    id_to_token.assign(max_id + 1, "");
    token_to_id.clear();

    for (auto& [token, id] : vocab_entries){
        id_to_token[id] = token;
        token_to_id[token] = id;
    }

    uint32_t merge_count = reader.read_u32();
    for (uint32_t i = 0; i < merge_count; i++){
        std::string merge = reader.read_string();

        size_t s = merge.find(" ");
        std::string token1 = merge.substr(0, s);
        std::string token2 = merge.substr(s + 1);

        uint32_t id1 = token_to_id[token1];
        uint32_t id2 = token_to_id[token2];
        uint64_t packed = pack(id1, id2);

        merge_to_rank[packed] = i;
        merge_to_id[packed] = token_to_id[token1 + token2];
    }
}

// Makes one lookup key out of two token ids.
// BPE checks neighboring pairs constantly, so this avoids building strings or
// small objects just to ask "can left + right merge?"
uint64_t Bpe::pack(uint32_t left, uint32_t right) const {
    uint64_t packed = left;
    packed = packed << 32;
    packed = packed | right;
    return packed;
}

// Converts a token string to its id.
// By the time we call this, tokenizer-specific code should have produced a
// string that exists in the vocab.
std::vector<uint32_t> Bpe::get_id(const std::string& token) const {
    auto it = token_to_id.find(token);
    if (it == token_to_id.end()) {
        std::cerr << "token missing from vocabulary: " << token << std::endl;
        std::exit(1);
    }
    return {it->second};
}

// Finds the best pair to merge next.
// "Best" means lowest rank. During tokenizer training, common pairs were added
// earlier, so lower rank is higher priority. Encoding replays that order.
uint64_t Bpe::get_lowest_pair(const std::vector<uint32_t>& tokens) const {
    uint32_t lowest_rank = UINT32_MAX;
    uint64_t result = UINT64_MAX;

    for (size_t i = 0; i + 1 < tokens.size(); i++){
        uint64_t packed = pack(tokens[i], tokens[i+1]);
        auto it = merge_to_rank.find(packed);
        if (it != merge_to_rank.end() && it->second < lowest_rank){
            lowest_rank = it->second;
            result = packed;
        }
    }

    return result;
}

// Does one merge pass.
// If we decided A + B should become C, this walks the token list and replaces
// every non-overlapping A, B pair with C.
std::vector<uint32_t> Bpe::merge(const std::vector<uint32_t>& tokens, uint32_t left, uint32_t right, uint32_t merged) const {
    std::vector<uint32_t> merged_tokens;
    merged_tokens.reserve(tokens.size());

    size_t i = 0;
    while (i < tokens.size()){
        if (tokens[i] == left && i + 1 < tokens.size() && tokens[i+1] == right){
            merged_tokens.push_back(merged);
            i += 2;
            continue;
        }

        merged_tokens.push_back(tokens[i]);
        i++;
    }

    return merged_tokens;
}

// Runs BPE until nothing else can merge.
// Start with tiny ids, repeatedly merge the highest-priority neighboring pair,
// and return the final token ids the model expects.
std::vector<uint32_t> Bpe::apply_merges(std::vector<uint32_t> tokens) const {
    uint64_t packed = get_lowest_pair(tokens);
    while (packed != UINT64_MAX){
        uint32_t left  = static_cast<uint32_t>(packed >> 32);
        uint32_t right = static_cast<uint32_t>(packed & 0xFFFFFFFFu);
        tokens = merge(tokens, left, right, merge_to_id.at(packed));
        packed = get_lowest_pair(tokens);
    }

    return tokens;
}
