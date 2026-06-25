#pragma once

#include <unordered_map>
#include <string>
#include "tensor.h"
#include "fp16.h"
#include "tokenizer.h"
#include <variant>

struct Config {
    std::string architecture;
    size_t hidden_size;
    size_t intermediate_size;
    size_t n_layers;
    size_t n_heads;
    size_t n_kv_heads;
    size_t vocab_size;
    size_t head_dim;
    size_t sliding_window;
    size_t max_position_embeddings;
    float rope_theta;
    float norm_eps;
    std::string quant;
    bool tie_word_embeddings = false;
    uint32_t bos_token_id = 0;
    uint32_t eos_token_id = 0;
};

class BinaryReader;

using WeightTensor = std::variant<Tensor<float>, Tensor<int8_t>, Tensor<fp16_t>>;

struct Parameters {
    uint32_t format_version = model_format::FORMAT_VERSION;
    Config config;
    Tokenizer tokenizer;

    std::unordered_map<std::string, WeightTensor> global_weights;
    std::vector<std::unordered_map<std::string, WeightTensor>> layer_weights;

    static void* map_file(int fd, size_t& size);
    void load_tensor(std::unordered_map<std::string, WeightTensor>& m, char* p, const std::string& key, uint8_t dtype, const std::vector<size_t>& shape, uint64_t offset, uint64_t scale_offset, uint32_t scale_size);

    void load_config(BinaryReader& reader);
    void load_weights(char* p, BinaryReader& reader);
    void load_parameters(const std::string& path);

    bool uses_f16_aux_weights() const;

    template<typename T>
    Tensor<T> get_tensor(int layer, const std::string& name);
};

inline bool is_q8f16(const std::string& quant) {
    return quant == "Q8F16" || quant == "int8";  // "int8" = legacy .mog files
}

inline bool is_f16(const std::string& quant) {
    return quant == "f16";
}
