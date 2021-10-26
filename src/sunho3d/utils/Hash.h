#pragma once

// taken from rpcs3

constexpr uint64_t fnvSeed = 14695981039346656037ull;
constexpr uint64_t fnvPrime = 1099511628211ull;

template <typename T>
static uint64_t hashBase(T value) {
    return static_cast<uint64_t>(value);
}

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
static inline uint64_t hash64(uint64_t hash_value, T data) {
    hash_value ^= data;
    hash_value *= fnvPrime;
    return hash_value;
}

template <typename T, typename U>
static uint64_t hashStructBase(const T& value) {
    uint64_t result = fnvSeed;
    const uint8_t* bits = reinterpret_cast<const uint8_t*>(&value);

    for (uint64_t n = 0; n < (sizeof(T) / sizeof(U)); ++n) {
        U val{};
        std::memcpy(&val, bits + (n * sizeof(U)), sizeof(U));
        result = hash64(result, val);
    }

    return result;
}

template <typename T>
inline uint64_t hashStruct(const T& value) {
    static constexpr auto block_sz = sizeof(T);

    if constexpr ((block_sz & 0x7) == 0) {
        return hashStructBase<T, uint64_t>(value);
    }

    if constexpr ((block_sz & 0x3) == 0) {
        return hashStructBase<T, uint32_t>(value);
    }

    if constexpr ((block_sz & 0x1) == 0) {
        return hashStructBase<T, uint16_t>(value);
    }

    return hashStructBase<T, uint8_t>(value);
}
