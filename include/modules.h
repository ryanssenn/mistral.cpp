#include "kernels.h"
#include "parameters.h"
#include "inference_state.h"


// https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
template <typename TAux>
struct Embedding {
    Tensor<TAux> table;
    size_t num_embeddings;
    size_t embedding_dim;
    Embedding(const Tensor<TAux>& table) : table(table), num_embeddings(table.shape[0]), embedding_dim(table.shape[1]) {}
    void forward(InferenceState& infer, size_t token_id);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L268
struct RotaryEmbedding {
    static void init_freq(InferenceState& infer, Config& config);
    static void forward(InferenceState& infer);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L58
template <typename TAux>
struct RMSNorm {
    Tensor<TAux> g;
    float e = 1e-5f;

    RMSNorm(const Tensor<TAux>& g) : g(g) {}
    void forward(InferenceState& infer);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L123
template <typename TMatmul>
struct Attention {
    Tensor<TMatmul> q_proj;
    Tensor<TMatmul> k_proj;
    Tensor<TMatmul> v_proj;
    Tensor<TMatmul> o_proj;

    size_t layer;

    Attention(const Tensor<TMatmul>& q_proj,
              const Tensor<TMatmul>& k_proj,
              const Tensor<TMatmul>& v_proj,
              const Tensor<TMatmul>& o_proj,
              size_t layer)
            : q_proj(q_proj),
              k_proj(k_proj),
              v_proj(v_proj),
              o_proj(o_proj),
              layer(layer){}

    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L35
template <typename TMatmul>
struct MLP {
    Tensor<TMatmul> down_proj;
    Tensor<TMatmul> gate_proj;
    Tensor<TMatmul> up_proj;

    MLP(const Tensor<TMatmul>& down_proj, const Tensor<TMatmul>& gate_proj, const Tensor<TMatmul>& up_proj)
        : down_proj(down_proj), gate_proj(gate_proj), up_proj(up_proj) {}

    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L206
template <typename TMatmul, typename TAux>
struct Layer {
    int i;
    RMSNorm<TAux> input_norm;
    RMSNorm<TAux> output_norm;
    Attention<TMatmul> attn;
    MLP<TMatmul> mlp;

    Layer(int i, std::shared_ptr<Parameters> p) :
                                i(i),

                                input_norm(p->get_tensor<TAux>(i, "input_layernorm.weight")),
                                output_norm(p->get_tensor<TAux>(i, "post_attention_layernorm.weight")),

                                attn(p->get_tensor<TMatmul>(i, "self_attn.q_proj.weight"),
                                     p->get_tensor<TMatmul>(i, "self_attn.k_proj.weight"),
                                     p->get_tensor<TMatmul>(i, "self_attn.v_proj.weight"),
                                     p->get_tensor<TMatmul>(i, "self_attn.o_proj.weight"), i),

                                mlp(p->get_tensor<TMatmul>(i, "mlp.down_proj.weight"),
                                    p->get_tensor<TMatmul>(i, "mlp.gate_proj.weight"),
                                    p->get_tensor<TMatmul>(i, "mlp.up_proj.weight"))
                                {}


    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L414
template <typename TAux>
struct LMHead {
    Tensor<TAux> lm_head;

    LMHead(std::shared_ptr<Parameters> params) : lm_head(params->get_tensor<TAux>(-1, "lm_head.weight")) {}

    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L334
template <typename TMatmul, typename TAux>
struct Model {
    Embedding<TAux> embedding;
    RMSNorm<TAux> norm;
    LMHead<TAux> lmHead;
    std::vector<Layer<TMatmul, TAux>> layers;

    Model(std::shared_ptr<Parameters> params) : embedding(params->get_tensor<TAux>(-1, "model.embed_tokens.weight")), norm(params->get_tensor<TAux>(-1, "model.norm.weight")), lmHead(params){
        for (int i=0;i<params->config.n_layers; i++){
            layers.emplace_back(i, params);
        }
    }

    void forward(InferenceState& infer, size_t token_id);
};



