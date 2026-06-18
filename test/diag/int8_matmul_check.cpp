// Diagnostic D1: does the int8 matmul kernel reproduce a plain f32 matmul of the
// dequantized weights? If yes, the kernel is faithful and any error is pure
// quantization. If no, there is a kernel bug (scale indexing, accumulation...).
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "../../include/parameters.h"
#include "../../include/kernels.h"

int main(int argc, char** argv) {
    std::string model_path = argc > 1 ? argv[1] : "mistral.bin";

    Parameters params;
    params.load_parameters(model_path);

    if (params.config.quant != "int8") {
        std::cerr << "Model is not int8 (" << params.config.quant << ")\n";
        return 1;
    }

    // gate_proj of layer 0: [intermediate_size, hidden_size] int8 + per-group scales.
    Tensor<int8_t> w = params.get_tensor<int8_t>(0, "mlp.gate_proj.weight");
    size_t n = w.shape[0];
    size_t d = w.shape[1];
    std::cout << "gate_proj shape [" << n << ", " << d << "], scales " << w.scales.size() << "\n";

    // Dequantize to f32 using the same get() the engine relies on.
    std::vector<float> deq(w.numel);
    for (size_t i = 0; i < w.numel; i++) deq[i] = w.get(i);
    Tensor<float> w_f32(deq.data(), {n, d});

    // Random activation vector.
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> xbuf(d);
    for (size_t i = 0; i < d; i++) xbuf[i] = dist(rng);
    Tensor<float> x(xbuf.data(), {d});

    std::vector<float> out_i8(n), out_f32(n);
    Tensor<float> ti8(out_i8.data(), {n});
    Tensor<float> tf32(out_f32.data(), {n});

    matmul(ti8, w, x);        // int8 kernel path
    matmul(tf32, w_f32, x);   // plain f32 matmul of dequantized weights

    double max_abs = 0.0, sum_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < n; i++) {
        double e = std::fabs((double)out_i8[i] - (double)out_f32[i]);
        max_abs = std::max(max_abs, e);
        sum_abs += e;
        double denom = std::fabs((double)out_f32[i]) + 1e-6;
        max_rel = std::max(max_rel, e / denom);
    }

    std::cout << "int8 vs dequant-f32 matmul:\n";
    std::cout << "  max abs error  = " << max_abs << "\n";
    std::cout << "  mean abs error = " << (sum_abs / n) << "\n";
    std::cout << "  max rel error  = " << max_rel << "\n";
    return 0;
}
