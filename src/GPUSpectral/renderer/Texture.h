#pragma once
#include <span>
#include <cuda.h>
#include <cuda_runtime.h>

enum class TextureFormat : uint16_t {
    RGBA16F,
    RGBA8,
    SRGB8_A8,
};

class Renderer;
class Texture {
public:
    Texture() {}
    Texture(Renderer& renderer, TextureFormat format, uint32_t width, uint32_t height);
    Texture(Texture&& other);
    Texture& operator=(Texture&& other);
    Texture(const Texture& other) = delete;
    Texture& operator=(const Texture& other) = delete;
    ~Texture();

    cudaTextureObject_t getTextureObject() const;
    void upload(const void* data);

private:
    TextureFormat format;
    uint32_t width;
    uint32_t height;
    cudaArray_t deviceArray;
    cudaTextureObject_t deviceTextureObject;
};