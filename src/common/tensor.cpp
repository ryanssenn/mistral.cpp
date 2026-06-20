#include "tensor.h"
#include <cassert>
#include <cstring>

template <typename T>
size_t Tensor<T>::get_numel() const {
    size_t s = 1;
    for (size_t d : shape) {
        assert(d > 0 && "dim=0 not supported");
        s *= d;
    }
    return s;
}

template <typename T>
void Tensor<T>::init_strides() {
    strides.assign(shape.size(), 1);
    if (shape.size() < 2) return;
    size_t stride = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        stride *= shape[i+1];
        strides[i] = stride;
    }
}

template <typename T>
Tensor<T>::Tensor(T* data, const std::vector<size_t>& shape) : shape(shape), numel(get_numel()), data(data){
    init_strides();
}

template <typename T>
Tensor<T>::Tensor(Arena& arena, const std::vector<size_t>& shape) : shape(shape), numel(get_numel()), data(static_cast<T*>(arena.allocate(numel*type_size))){
    init_strides();
}

template <typename T>
Tensor<T>::Tensor(Arena& arena, const std::vector<float>& arr, const std::vector<size_t>& shape) : shape(shape), numel(get_numel()), data(static_cast<T*>(arena.allocate(numel*type_size))){
    init_strides();
    std::copy(arr.begin(), arr.end(), data);
}


template <typename T>
Tensor<T>::Tensor(T* data, const std::vector<float>& scales, const std::vector<size_t>& shape) : Tensor(data, shape) {
    this->scales = scales;
}


template <typename T>
void Tensor<T>::copy_from(const Tensor& tensor) {
    std::memcpy(data, tensor.data, tensor.numel*type_size);
}

template <typename T>
Tensor<T> Tensor<T>::at(std::initializer_list<size_t> idx) {
    assert(idx.size() <= shape.size() && "Too many indices for tensor");
    T* new_data = data;

    int i = 0;
    for (auto v : idx) {
        assert(v < shape[i] && "Index out of range");
        new_data += strides[i] * v;
        i++;
    }

    std::vector<size_t> new_shape(shape.begin() + i, shape.end());
    return Tensor(new_data, new_shape);
}

template<>
float Tensor<float>::max(){
    float result = data[0];
    for (int i=0; i<numel; i++){
        result = std::max(result, data[i]);
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::reshape(std::vector<size_t> new_shape) {
    size_t new_numel = 1;
    for (auto d : new_shape) new_numel *= d;
    assert(new_numel <= numel && "Reshape size mismatch");
    return Tensor(data, new_shape);
}

template<>
float Tensor<float>::get(size_t i){
    return data[i];
}

template<>
float Tensor<int8_t>::get(size_t i){
    return data[i] / scales[scales.size() * i / numel];
}


template class Tensor<float>;
template class Tensor<signed char>;