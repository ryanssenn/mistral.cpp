#include "setup/context.h"
#include "model_format.h"
#include "fp16.h"
#include "parameters.h"

#include <cstring>
#include <fstream>
#include <variant>
#include <vector>

namespace {

using TensorVariant = std::variant<Tensor<float>, Tensor<int8_t>, Tensor<fp16_t>>;

std::string resolve_model_path() {
    std::string model_path = "qwen3-0.6B.mog";
    if (!std::ifstream(model_path).good()) {
        model_path = "../qwen3-0.6B.mog";
    }
    return model_path;
}

template<typename T>
bool expect_eq(const char* field, T got, T want) {
    if (got != want) {
        std::cerr << field << " mismatch: got " << got << ", want " << want << "\n";
        return false;
    }
    return true;
}

bool expect_near(const char* field, float got, float want, float atol) {
    if (std::fabs(got - want) >= atol) {
        std::cerr << field << " mismatch: got " << got << ", want " << want << "\n";
        return false;
    }
    return true;
}

std::vector<size_t> tensor_shape(const TensorVariant& v) {
    if (std::holds_alternative<Tensor<float>>(v)) {
        return std::get<Tensor<float>>(v).shape;
    }
    if (std::holds_alternative<Tensor<fp16_t>>(v)) {
        return std::get<Tensor<fp16_t>>(v).shape;
    }
    return std::get<Tensor<int8_t>>(v).shape;
}

size_t tensor_numel(const TensorVariant& v) {
    if (std::holds_alternative<Tensor<float>>(v)) {
        return std::get<Tensor<float>>(v).get_numel();
    }
    if (std::holds_alternative<Tensor<fp16_t>>(v)) {
        return std::get<Tensor<fp16_t>>(v).get_numel();
    }
    return std::get<Tensor<int8_t>>(v).get_numel();
}

bool is_fp16_tensor(const TensorVariant& v) {
    return std::holds_alternative<Tensor<fp16_t>>(v);
}

float tensor_get(const TensorVariant& v, size_t i) {
    if (std::holds_alternative<Tensor<float>>(v)) {
        return std::get<Tensor<float>>(v).get(i);
    }
    if (std::holds_alternative<Tensor<fp16_t>>(v)) {
        return std::get<Tensor<fp16_t>>(v).get(i);
    }
    return std::get<Tensor<int8_t>>(v).get(i);
}

size_t shape_product(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (size_t d : shape) {
        n *= d;
    }
    return n;
}

bool check_shape(const std::string& name, const std::vector<size_t>& got,
                 const std::vector<size_t>& want) {
    if (got != want) {
        std::cerr << name << " shape mismatch: got [";
        for (size_t i = 0; i < got.size(); ++i) {
            if (i) std::cerr << ", ";
            std::cerr << got[i];
        }
        std::cerr << "], want [";
        for (size_t i = 0; i < want.size(); ++i) {
            if (i) std::cerr << ", ";
            std::cerr << want[i];
        }
        std::cerr << "]\n";
        return false;
    }
    return true;
}

TensorVariant& resolve_tensor(Parameters& params, const std::string& key, int layer) {
    if (layer == -1) {
        return params.global_weights.at(key);
    }
    return params.layer_weights[static_cast<size_t>(layer)].at(key);
}

const std::vector<const char*> kLayerTensorNames = {
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
};

const std::vector<const char*> kGlobalTensorNames = {
    "model.embed_tokens.weight",
    "model.norm.weight",
};

const std::vector<std::pair<const char*, std::vector<size_t>>> kLayerShapeSpecs = {
    {"input_layernorm.weight", {1024}},
    {"post_attention_layernorm.weight", {1024}},
    {"self_attn.q_norm.weight", {128}},
    {"self_attn.k_norm.weight", {128}},
    {"self_attn.q_proj.weight", {2048, 1024}},
    {"self_attn.k_proj.weight", {1024, 1024}},
    {"self_attn.v_proj.weight", {1024, 1024}},
    {"self_attn.o_proj.weight", {1024, 2048}},
    {"mlp.gate_proj.weight", {3072, 1024}},
    {"mlp.up_proj.weight", {3072, 1024}},
    {"mlp.down_proj.weight", {1024, 3072}},
};

const std::vector<std::pair<const char*, std::vector<size_t>>> kGlobalShapeSpecs = {
    {"model.embed_tokens.weight", {151936, 1024}},
    {"model.norm.weight", {1024}},
};

struct WeightSpotCheck {
    const char* key;
    int layer;
    float first;
    float last;
};

const std::vector<WeightSpotCheck> kWeightSpotChecks = {
    {"model.embed_tokens.weight", -1, -0.00927734375f, 0.00567626953125f},
    {"model.norm.weight", -1, 3.9375f, 3.640625f},
    {"input_layernorm.weight", 0, 0.1357421875f, 0.1552734375f},
    {"post_attention_layernorm.weight", 0, 0.474609375f, 0.5703125f},
    {"self_attn.q_proj.weight", 0, 0.0034027099609375f, 0.07763671875f},
    {"self_attn.k_proj.weight", 0, -0.0166015625f, -0.006072998046875f},
    {"self_attn.v_proj.weight", 0, 0.01214599609375f, -0.0283203125f},
    {"self_attn.o_proj.weight", 0, 0.01446533203125f, -0.0228271484375f},
    {"self_attn.q_norm.weight", 0, 4.53125f, 2.390625f},
    {"self_attn.k_norm.weight", 0, 1.296875f, 2.015625f},
    {"mlp.gate_proj.weight", 0, -0.0186767578125f, 0.01348876953125f},
    {"mlp.up_proj.weight", 0, -0.04248046875f, 0.0162353515625f},
    {"mlp.down_proj.weight", 0, 0.02099609375f, 0.00677490234375f},
    {"input_layernorm.weight", 27, 17.625f, 13.3125f},
    {"self_attn.q_proj.weight", 27, -0.034423828125f, 0.01019287109375f},
    {"mlp.down_proj.weight", 27, -0.01434326171875f, 0.01361083984375f},
};

} // namespace

// Validates the raw file prefix: MOG magic, format version 2, and a sane header_size
// that fits within the file. Catches truncated or non-MOG inputs before load.
int test_mog_header() {
    const std::string path = resolve_model_path();
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        std::cerr << "model file not found: " << path << "\n";
        return 1;
    }

    f.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    if (file_size < model_format::FILE_PREFIX_SIZE) {
        std::cerr << "file too small for MOG prefix: " << file_size << " bytes\n";
        return 1;
    }

    char prefix[model_format::FILE_PREFIX_SIZE];
    f.read(prefix, sizeof(prefix));
    if (!f) {
        std::cerr << "failed to read MOG prefix\n";
        return 1;
    }

    if (std::memcmp(prefix, model_format::MAGIC, 4) != 0) {
        std::cerr << "magic mismatch: expected MOG\\0\n";
        return 1;
    }

    uint32_t version;
    std::memcpy(&version, prefix + 4, sizeof(version));
    if (version != model_format::FORMAT_VERSION_V2) {
        std::cerr << "format version mismatch: got " << version
                  << ", want " << model_format::FORMAT_VERSION_V2 << "\n";
        return 1;
    }

    uint64_t header_size;
    std::memcpy(&header_size, prefix + 8, sizeof(header_size));
    if (header_size == 0) {
        std::cerr << "header_size must be non-zero\n";
        return 1;
    }
    if (header_size >= file_size) {
        std::cerr << "header_size out of bounds: " << header_size
                  << " >= file_size " << file_size << "\n";
        return 1;
    }

    return 0;
}

// Checks that Parameters parsed the config KV for Qwen3-0.6B: architecture, dims,
// rope/norm settings, f16 quant, tied embeddings, and special token ids.
int test_mog_config() {
    const Config& c = get_params()->config;

    if (c.architecture != "qwen3") {
        std::cerr << "architecture mismatch: got " << c.architecture << ", want qwen3\n";
        return 1;
    }
    if (!expect_eq("hidden_size", c.hidden_size, size_t{1024})) return 1;
    if (!expect_eq("intermediate_size", c.intermediate_size, size_t{3072})) return 1;
    if (!expect_eq("n_layers", c.n_layers, size_t{28})) return 1;
    if (!expect_eq("n_heads", c.n_heads, size_t{16})) return 1;
    if (!expect_eq("n_kv_heads", c.n_kv_heads, size_t{8})) return 1;
    if (!expect_eq("head_dim", c.head_dim, size_t{128})) return 1;
    if (!expect_eq("vocab_size", c.vocab_size, size_t{151936})) return 1;
    if (!expect_eq("sliding_window", c.sliding_window, size_t{0})) return 1;
    if (!expect_eq("max_position_embeddings", c.max_position_embeddings, size_t{40960})) return 1;
    if (!expect_near("rope_theta", c.rope_theta, 1000000.0f, 1.0f)) return 1;
    if (!expect_near("norm_eps", c.norm_eps, 1e-6f, 1e-9f)) return 1;

    if (c.quant != "f16") {
        std::cerr << "quant mismatch: got " << c.quant << ", want f16\n";
        return 1;
    }
    if (!c.tie_word_embeddings) {
        std::cerr << "tie_word_embeddings should be true\n";
        return 1;
    }
    if (!expect_eq("bos_token_id", c.bos_token_id, uint32_t{151643})) return 1;
    if (!expect_eq("eos_token_id", c.eos_token_id, uint32_t{151645})) return 1;

    return 0;
}

// MOG v2 stores a tiktoken-style pre-tokenize regex after BPE merges. This test
// ensures the loader reads it so Qwen tokenization can use the exported pattern.
int test_mog_tokenizer_metadata() {
    const std::shared_ptr<Parameters> params = get_params();

    if (params->tokenizer.pre_tokenize_regex.empty()) {
        std::cerr << "missing MOG v2 pre_tokenize_regex\n";
        return 1;
    }

    if (params->tokenizer.pre_tokenize_regex.find("(?i:'s|'t|'re|'ve|'m|'ll|'d)") == std::string::npos) {
        std::cerr << "unexpected Qwen pre_tokenize_regex\n";
        return 1;
    }

    return 0;
}

// Verifies the tensor table: 28 layers x 11 weights (incl. q_norm/k_norm), two
// global f16 tensors (embed + norm, no lm_head when tied), and expected shapes/dtypes.
int test_mog_tensor_inventory() {
    const std::shared_ptr<Parameters> params = get_params();

    if (!expect_eq("global tensor count", params->global_weights.size(), size_t{2})) {
        return 1;
    }
    if (!expect_eq("layer count", params->layer_weights.size(), params->config.n_layers)) {
        return 1;
    }

    for (size_t layer = 0; layer < params->layer_weights.size(); ++layer) {
        const auto& weights = params->layer_weights[layer];
        if (!expect_eq("layer tensor count", weights.size(), size_t{11})) {
            std::cerr << "  at layer " << layer << "\n";
            return 1;
        }

        for (const char* name : kLayerTensorNames) {
            if (weights.find(name) == weights.end()) {
                std::cerr << "missing layer tensor: " << name << " at layer " << layer << "\n";
                return 1;
            }
        }

        for (const auto& spec : kLayerShapeSpecs) {
            const TensorVariant& v = weights.at(spec.first);
            if (!check_shape(spec.first, tensor_shape(v), spec.second)) {
                std::cerr << "  at layer " << layer << "\n";
                return 1;
            }
            if (tensor_numel(v) != shape_product(spec.second)) {
                std::cerr << spec.first << " numel mismatch at layer " << layer << "\n";
                return 1;
            }
            if (!is_fp16_tensor(v)) {
                std::cerr << spec.first << " should be f16 at layer " << layer << "\n";
                return 1;
            }
        }
    }

    for (const char* name : kGlobalTensorNames) {
        if (params->global_weights.find(name) == params->global_weights.end()) {
            std::cerr << "missing global tensor: " << name << "\n";
            return 1;
        }
    }

    for (const auto& spec : kGlobalShapeSpecs) {
        const TensorVariant& v = params->global_weights.at(spec.first);
        if (!check_shape(spec.first, tensor_shape(v), spec.second)) {
            return 1;
        }
        if (tensor_numel(v) != shape_product(spec.second)) {
            std::cerr << spec.first << " numel mismatch\n";
            return 1;
        }
        if (!is_fp16_tensor(v)) {
            std::cerr << spec.first << " should be f16\n";
            return 1;
        }
    }

    if (params->global_weights.find("lm_head.weight") != params->global_weights.end()) {
        std::cerr << "lm_head.weight should be absent when tie_word_embeddings is true\n";
        return 1;
    }

    (void)params->get_tensor<fp16_t>(0, "self_attn.q_proj.weight");
    (void)params->get_tensor<fp16_t>(-1, "model.embed_tokens.weight");

    return 0;
}

// Spot-checks first/last f16 values at known payload offsets on a few tensors
// (layers 0 and 27). Confirms mmap offsets and f16 decode, not just header layout.
int test_mog_weight_spotcheck() {
    const std::shared_ptr<Parameters> params = get_params();
    constexpr float atol = 1e-3f;

    for (const auto& entry : kWeightSpotChecks) {
        TensorVariant& v = resolve_tensor(*params, entry.key, entry.layer);
        const size_t n = tensor_numel(v);

        const float first_val = tensor_get(v, 0);
        const float last_val = tensor_get(v, n - 1);

        if (!expect_near("first val", first_val, entry.first, atol)) {
            std::cerr << entry.key;
            if (entry.layer >= 0) std::cerr << " (layer " << entry.layer << ")";
            std::cerr << "\n";
            return 1;
        }
        if (!expect_near("last val", last_val, entry.last, atol)) {
            std::cerr << entry.key;
            if (entry.layer >= 0) std::cerr << " (layer " << entry.layer << ")";
            std::cerr << "\n";
            return 1;
        }
    }

    return 0;
}

static RegisterTest mog_header_f16("header", "f16", &test_mog_header);
static RegisterTest mog_config_f16("config", "f16", &test_mog_config);
static RegisterTest mog_tokenizer_f16("tokenizer metadata", "f16", &test_mog_tokenizer_metadata);
static RegisterTest mog_inventory_f16("tensor inventory", "f16", &test_mog_tensor_inventory);
static RegisterTest mog_spotcheck_f16("weight spotcheck", "f16", &test_mog_weight_spotcheck);
