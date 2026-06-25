#include <string>
#include <iostream>
#include "parameters.h"
#include "kernels.h"
#include "modules.h"

struct TestCase {
    std::string name;
    std::string quant;
    int (*func)();
};

extern std::vector<TestCase> tests;

struct RegisterTest {
    RegisterTest(std::string name, std::string quant, int (*func)()){
        tests.push_back({name, quant, func});
    }
};

std::shared_ptr<Parameters> get_params();
inline InferenceState infer(get_params()->config);
inline Arena arena(4*1024*1024); // 4 MB

inline std::unordered_map<std::string, Tensor<float>> expected;
void load_expected_values();

struct TopK {
    std::vector<uint32_t> ids;
    std::vector<float> vals;
};

TopK get_topk(const Tensor<float>& logits, size_t k);
bool has_logits_golden();

bool equals(float x, float y);
bool equals(const Tensor<float>& x, const Tensor<float>& y);
bool equals(const Tensor<float>& x, const Tensor<float>& y, float atol);
