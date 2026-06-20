#include "setup/context.h"


int test_rope(){
    std::shared_ptr<Parameters> params = get_params();
    RotaryEmbedding::init_freq(infer, params->config);

    for (size_t i = 0; i<4; i++){
        Tensor<float> q(arena, {1,128});

        for (int i=0;i<q.numel;i++){
            q.data[i] = (i%128) / 256.0f;
        }

        infer.pos = i;
        RotaryEmbedding::forward(infer);

        rope(q, q, infer.cos, infer.sin);

        if (!equals(q, expected.at("rope" + std::to_string(i)))){
            std::cout << "Mismatch RoPE at pos " + std::to_string(i) << std::endl;
            return 1;
        }
    }

    return 0;
}

RegisterTest rope_reg("test rope", "any", &test_rope);


int test_matmul(){
    Tensor<float> w(arena, {1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor<float> x(arena, {10, 20, 30}, {3});
    Tensor<float> xout(arena, {0, 0}, {2});
    Tensor<float> expected(arena, {140, 320}, {2});

    matmul(xout, w, x);

    if (!equals(xout, expected)){
        std::cout << "matmul mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest matmul_reg("test matmul", "any", &test_matmul);

int test_row_matmul(){
    Tensor<float> w(arena, {1, 2,
                     3, 4,
                     5, 6}, {3, 2});
    Tensor<float> x(arena, {10, 20, 30}, {3});
    Tensor<float> xout(arena, {0, 0}, {2});
    Tensor<float> expected(arena, {220, 280}, {2});

    row_matmul(xout, x, w);

    if (!equals(xout, expected)){
        std::cout << "row_matmul mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest row_matmul_reg("test row matmul", "any", &test_row_matmul);


int test_softmax(){
    Tensor<float> x(arena, {1.0f, 2.0f, 3.0f}, {3});
    Tensor<float> xout(arena, {0.0f, 0.0f, 0.0f}, {3});
    Tensor<float> expected(arena, {0.09003057f, 0.24472848f, 0.66524094f}, {3});

    softmax(xout, x);

    if (!equals(xout, expected)){
        std::cout << "softmax mismatch" << std::endl;
        return 1;
    }
    return 0;
}

RegisterTest softmaxl_reg("test softmax", "any", &test_softmax);


int test_silu(){
    Tensor<float> x(arena, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});

    Tensor<float> expected(arena, {-0.2384, -0.2689,  0.0000,  0.7311,  1.7616}, {5});

    Tensor<float> xout(arena, {5});

    silu(xout, x);

    if (!equals(xout, expected)){
        std::cout << "Tensor<float> silu mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest silu_reg("test silu", "any", &test_silu);






