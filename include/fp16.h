#pragma once

#include <cstdint>

using fp16_t = uint16_t;

float fp16_to_f32(fp16_t h);
fp16_t f32_to_fp16(float f);
