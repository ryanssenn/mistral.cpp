#include "context.h"
#include <cmath>
#include <fstream>

std::vector<TestCase> tests;

std::shared_ptr<ModelLoad> get_model(){
    static bool init = false;

    static std::shared_ptr<ModelLoad> params = std::make_shared<ModelLoad>();

    if (init){
        return params;
    }

    std::string model_path = "qwen3-0.6B.mog";
    if (!std::ifstream(model_path).good()) {
        model_path = "../qwen3-0.6B.mog";
    }

    params->load(model_path);
    init = true;

    return params;
}

bool equals(const Tensor& x, const Tensor& y, float atol){
    if (x.numel != y.numel){
        return false;
    }

    const float* x_data = x.f32();
    const float* y_data = y.f32();
    for (size_t i=0; i<x.numel; i++){
        if (std::fabs(x_data[i] - y_data[i]) >= atol){
            std::cout << "Mismatch at pos " << i
                      << ": expected " << y_data[i]
                      << ", got " << x_data[i] << std::endl;
            return false;
        }
    }

    return true;
}

bool expect_tensor(const Tensor& got, const float* golden, size_t n, float atol, const char* name) {
    if (equals(got, golden_tensor(golden, n), atol)) {
        return true;
    }
    std::cerr << name << " mismatch\n";
    return false;
}
