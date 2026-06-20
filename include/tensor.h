#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <initializer_list>
#include <cassert>
#include <memory>

struct Arena {
    size_t BUFFER_SIZE;
    char* buffer;
    size_t offset = 0;

    Arena(size_t BUFFER_SIZE) : BUFFER_SIZE(BUFFER_SIZE), buffer(new char[BUFFER_SIZE]) {}

    void* allocate(size_t size){
        assert(offset + size < BUFFER_SIZE && "Tensor allocator out of memory");
        char* result = buffer + offset;
        offset += size;

        return result;
    }

    ~Arena(){
        delete[] buffer;
    }
};


template <typename T>
struct Tensor {
    std::vector<size_t> shape;

    size_t numel;
    size_t type_size = sizeof(T);

    T* data;
    std::vector<size_t> strides;
    std::vector<float> scales;


    size_t get_numel() const;
    void init_strides();

    Tensor() {}
    Tensor(T* data, const std::vector<size_t>& shape);
    Tensor(Arena& arena, const std::vector<size_t>& shape);
    Tensor(Arena& arena, const std::vector<float>& arr, const std::vector<size_t>& shape);

    // Quantized
    Tensor(T* data, const std::vector<float>& scales, const std::vector<size_t>& shape);

    void copy_from(const Tensor& tensor);

    Tensor at(std::initializer_list<size_t> idx);
    float max();

    Tensor reshape(std::vector<size_t> new_shape);

    float get(size_t i) const;
};