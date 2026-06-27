#include "setup/context.h"
#include "model/model.h"
#include "model/inference_state.h"

#include "goldens/L0.inc"
#include "goldens/stack.inc"

#include <cassert>
#include <cstring>
#include <iostream>

namespace {

using namespace module_golden_L0_pos0;
namespace stack = stack_golden_pos0;

constexpr float kAtol = 1e-3f;
constexpr int kLayer = 0;
constexpr int kMiddleLayer = static_cast<int>(stack::kStackNumLayers[0]) / 2;
constexpr int kLastLayer = static_cast<int>(stack::kStackNumLayers[0]) - 1;

void copy_golden(Tensor& dst, const float* data, size_t n) {
    assert(n == dst.numel);
    dst.copy_from(golden_tensor(data, n));
}

Model make_model() {
    return Model(get_model());
}

InferenceState make_state() {
    InferenceState state(get_model()->config);
    state.pos = 0;
    return state;
}

void run_through_layers(Model& model, InferenceState& infer, size_t token_id, int through_layer) {
    model.embedding.forward(infer, token_id);
    for (int i = 0; i <= through_layer; i++) {
        model.layers[i].forward(infer);
    }
}

} // namespace

// Embedding::forward
int test_module_embedding() {
    Model model = make_model();
    InferenceState infer = make_state();

    model.embedding.forward(infer, static_cast<size_t>(kEmbeddingToken[0]));

    if (!expect_tensor(infer.hidden_state, kEmbeddingOut, kEmbeddingOutSize, kAtol, "embedding")) {
        return 1;
    }
    return 0;
}

// RMSNorm::forward (input_layernorm)
int test_module_rmsnorm_input() {
    Model model = make_model();
    InferenceState infer = make_state();
    copy_golden(infer.hidden_state, kRmsnormL0InputIn, kRmsnormL0InputInSize);

    model.layers[kLayer].input_norm.forward(infer);

    if (!expect_tensor(infer.hidden_state, kRmsnormL0InputOut, kRmsnormL0InputOutSize, kAtol, "rmsnorm input")) {
        return 1;
    }
    return 0;
}

// RMSNorm::forward (post_attention_layernorm)
int test_module_rmsnorm_post() {
    Model model = make_model();
    InferenceState infer = make_state();
    copy_golden(infer.hidden_state, kRmsnormL0PostIn, kRmsnormL0PostInSize);

    model.layers[kLayer].output_norm.forward(infer);

    if (!expect_tensor(infer.hidden_state, kRmsnormL0PostOut, kRmsnormL0PostOutSize, kAtol, "rmsnorm post")) {
        return 1;
    }
    return 0;
}

// RotaryEmbedding::forward
int test_module_rotary_emb() {
    InferenceState infer = make_state();

    RotaryEmbedding::forward(infer);

    if (!expect_tensor(infer.cos, kRopeL0Cos, kRopeL0CosSize, kAtol, "rotary cos")) {
        return 1;
    }
    if (!expect_tensor(infer.sin, kRopeL0Sin, kRopeL0SinSize, kAtol, "rotary sin")) {
        return 1;
    }
    return 0;
}

// rope() kernel
int test_module_rope() {
    InferenceState infer = make_state();
    copy_golden(infer.cos, kRopeL0Cos, kRopeL0CosSize);
    copy_golden(infer.sin, kRopeL0Sin, kRopeL0SinSize);

    copy_golden(infer.q_state, kRopeL0QIn, kRopeL0QInSize);
    rope(infer.q_state, infer.q_state, infer.cos, infer.sin);
    if (!expect_tensor(infer.q_state, kRopeL0QOut, kRopeL0QOutSize, kAtol, "rope q")) {
        return 1;
    }

    copy_golden(infer.k_state, kRopeL0KIn, kRopeL0KInSize);
    rope(infer.k_state, infer.k_state, infer.cos, infer.sin);
    if (!expect_tensor(infer.k_state, kRopeL0KOut, kRopeL0KOutSize, kAtol, "rope k")) {
        return 1;
    }
    return 0;
}

// Attention::forward
int test_module_attention() {
    Model model = make_model();
    InferenceState infer = make_state();
    copy_golden(infer.hidden_state, kAttnL0In, kAttnL0InSize);

    model.layers[kLayer].attn.forward(infer);

    if (!expect_tensor(infer.hidden_state, kAttnL0Out, kAttnL0OutSize, kAtol, "attention")) {
        return 1;
    }
    return 0;
}

// MLP::forward
int test_module_mlp() {
    Model model = make_model();
    InferenceState infer = make_state();
    copy_golden(infer.hidden_state, kMlpL0In, kMlpL0InSize);

    model.layers[kLayer].mlp.forward(infer);

    if (!expect_tensor(infer.hidden_state, kMlpL0Out, kMlpL0OutSize, kAtol, "mlp")) {
        return 1;
    }
    return 0;
}

// Layer::forward
int test_module_layer() {
    Model model = make_model();
    InferenceState infer = make_state();
    copy_golden(infer.hidden_state, kLayerL0In, kLayerL0InSize);

    model.layers[kLayer].forward(infer);

    if (!expect_tensor(infer.hidden_state, kLayerL0Out, kLayerL0OutSize, kAtol, "layer")) {
        return 1;
    }
    return 0;
}

// Hidden state after first decoder layer (stack golden)
int test_stack_hidden_first() {
    Model model = make_model();
    InferenceState infer = make_state();
    const size_t token = static_cast<size_t>(stack::kStackToken[0]);

    run_through_layers(model, infer, token, 0);

    if (!expect_tensor(infer.hidden_state, stack::kHiddenL0Out, stack::kHiddenL0OutSize, kAtol, "hidden L0")) {
        return 1;
    }
    return 0;
}

// Hidden state after middle decoder layer (stack golden)
int test_stack_hidden_middle() {
    Model model = make_model();
    InferenceState infer = make_state();
    const size_t token = static_cast<size_t>(stack::kStackToken[0]);

    run_through_layers(model, infer, token, kMiddleLayer);

    if (!expect_tensor(infer.hidden_state, stack::kHiddenL14Out, stack::kHiddenL14OutSize, kAtol, "hidden L14")) {
        return 1;
    }
    return 0;
}

// Hidden state after last decoder layer (stack golden)
int test_stack_hidden_last() {
    Model model = make_model();
    InferenceState infer = make_state();
    const size_t token = static_cast<size_t>(stack::kStackToken[0]);

    run_through_layers(model, infer, token, kLastLayer);

    if (!expect_tensor(infer.hidden_state, stack::kHiddenL27Out, stack::kHiddenL27OutSize, kAtol, "hidden L27")) {
        return 1;
    }
    return 0;
}

static RegisterTest module_embedding("module embedding", "f16", &test_module_embedding);
static RegisterTest module_rmsnorm_input("module rmsnorm input", "f16", &test_module_rmsnorm_input);
static RegisterTest module_rmsnorm_post("module rmsnorm post", "f16", &test_module_rmsnorm_post);
static RegisterTest module_rotary_emb("module rotary emb", "f16", &test_module_rotary_emb);
static RegisterTest module_rope("module rope", "f16", &test_module_rope);
static RegisterTest module_attention("module attention", "f16", &test_module_attention);
static RegisterTest module_mlp("module mlp", "f16", &test_module_mlp);
static RegisterTest module_layer("module layer", "f16", &test_module_layer);
static RegisterTest stack_hidden_first("stack hidden first", "f16", &test_stack_hidden_first);
static RegisterTest stack_hidden_middle("stack hidden middle", "f16", &test_stack_hidden_middle);
static RegisterTest stack_hidden_last("stack hidden last", "f16", &test_stack_hidden_last);
