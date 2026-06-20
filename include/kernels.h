#pragma once

#include "tensor.h"
#include "fp16.h"
#include <cmath>

void matmul(Tensor<float>& xout, Tensor<float>& w, Tensor<float>& x);
void matmul(Tensor<float>& xout, Tensor<int8_t>& w, Tensor<float>& x);
void matmul(Tensor<float>& xout, Tensor<fp16_t>& w, Tensor<float>& x);
void row_matmul(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& w);

void softmax(Tensor<float>& xout, Tensor<float>& x);
void rope(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& cos, Tensor<float>& sin);

void silu(Tensor<float>& xout, Tensor<float>& x);

void add(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& y);

void mul(Tensor<float>& xout, Tensor<float>& x, Tensor<float>& y);
void mul(Tensor<float>& xout, Tensor<float>& x, float c);
