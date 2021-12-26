#pragma once

#include <vector>
#include "../kernels/VectorMath.cuh"

struct Image {
    Image() = default;
    explicit Image(size_t width, size_t height)
        : width(width), height(height), buffer(width* height) {
        std::fill(buffer.begin(), buffer.end(), make_float3(1.0, 1.0, 1.0));
    }

    float3* operator[](size_t y) {
        return &buffer[y * width];
    }

    const float3* operator[](size_t y) const {
        return &buffer[y * width];
    }

    std::vector<char> pack() const {
        std::vector<char> outBuffer(4 * width * height);
        for (size_t i = 0; i < width * height; ++i) {
            auto& pixel = buffer[i];
            outBuffer[4 * i + 0] = static_cast<char>(clamp(pixel.x * 255.0f, 0.0f, 255.0f));
            outBuffer[4 * i + 1] = static_cast<char>(clamp(pixel.y * 255.0f, 0.0f, 255.0f));
            outBuffer[4 * i + 2] = static_cast<char>(clamp(pixel.z * 255.0f, 0.0f, 255.0f));
            outBuffer[4 * i + 3] = static_cast<char>(255.0f);
        }
        return outBuffer;
    }

    void clear() {
        std::fill(buffer.begin(), buffer.end(), make_float3(1.0, 1.0, 1.0));
    }

    int getWidth() const {
        return width;
    }

    int getHeight() const {
        return height;
    }

private:
    size_t width{ 0 };
    size_t height{ 0 };
    std::vector<float3> buffer;
};
