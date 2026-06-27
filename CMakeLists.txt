cmake_minimum_required(VERSION 3.20)
project(qmog.cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(QMOG_COMMON_SRC
    src/common/shape.cpp
    src/common/tensor.cpp
)

set(QMOG_LOADER_SRC
    src/loader/model_load.cpp
)

set(QMOG_TOKENIZER_SRC
    src/tokenizer/bpe.cpp
    src/tokenizer/qwen_pre_tokenize.cpp
    src/tokenizer/qwen_tokenizer.cpp
)

set(QMOG_MODEL_SRC
    src/model/inference_state.cpp
    src/model/model.cpp
)

set(QMOG_BACKEND_SRC
    src/backend/cpu/kernels.cpp
)

set(QMOG_DECODE_SRC
    src/decode/sampling.cpp
)

set(QMOG_EVAL_SRC
    src/eval/perplexity.cpp
)

set(TEST_QWEN_SRC
    test/qwen/main.cpp
    test/qwen/setup/context.cpp
    test/qwen/test_model_load.cpp
    test/qwen/test_pre_tokenize.cpp
    test/qwen/test_tokenizer.cpp
)

add_executable(qmog-cli
    src/main.cpp
    ${QMOG_COMMON_SRC}
    ${QMOG_LOADER_SRC}
    ${QMOG_TOKENIZER_SRC}
    ${QMOG_MODEL_SRC}
    ${QMOG_BACKEND_SRC}
    ${QMOG_DECODE_SRC}
    ${QMOG_EVAL_SRC}
)

add_executable(test_exec
    ${QMOG_COMMON_SRC}
    ${QMOG_LOADER_SRC}
    ${QMOG_TOKENIZER_SRC}
    ${QMOG_MODEL_SRC}
    ${QMOG_BACKEND_SRC}
    ${TEST_QWEN_SRC}
)

target_include_directories(qmog-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(test_exec PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test/qwen
)
