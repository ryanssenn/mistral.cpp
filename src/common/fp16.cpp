#include "fp16.h"

#include <cstring>

float fp16_to_f32(fp16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;

    if (exp == 0) {
        if (mant == 0) {
            float result;
            uint32_t bits = sign;
            std::memcpy(&result, &bits, sizeof(result));
            return result;
        }

        while ((mant & 0x400u) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3ffu;
    } else if (exp == 31) {
        float result;
        uint32_t bits = sign | 0x7f800000u | (mant << 13);
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    float result;
    uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

fp16_t f32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x007fffffu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<fp16_t>(sign);
        }
        mant |= 0x00800000u;
        mant >>= static_cast<uint32_t>(1 - exp);
        return static_cast<fp16_t>(sign | (mant >> 13));
    }

    if (exp >= 31) {
        return static_cast<fp16_t>(sign | 0x7c00u);
    }

    return static_cast<fp16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}
