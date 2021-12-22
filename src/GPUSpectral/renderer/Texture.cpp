#include "Texture.h"
#include "Renderer.h"


static cudaChannelFormatDesc translateTextureFormat(TextureFormat format) {
    switch (format) {
        case TextureFormat::RGBA16F:
            return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
        case TextureFormat::RGBA8:
            return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        case TextureFormat::SRGB8_A8:
            return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    }
}

static uint32_t textureFormatSize(TextureFormat format) {
    switch (format) {
        case TextureFormat::RGBA16F:
            return sizeof(float) * 4;
        case TextureFormat::RGBA8:
            return sizeof(uint8_t) * 4;
        case TextureFormat::SRGB8_A8:
            return sizeof(uint8_t) * 4;
    }
}

Texture::Texture(Renderer& renderer, TextureFormat format, uint32_t width, uint32_t height) 
    : width(width), height(height), format(format) {

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = translateTextureFormat(format);
    CUDA_CHECK(cudaMallocArray(&deviceArray, &channelDesc, width, height));

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = deviceArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.sRGB = format == TextureFormat::SRGB8_A8 ? 1 : 0;
    texDesc.normalizedCoords = 1;

    // Create texture object
    CUDA_CHECK(cudaCreateTextureObject(&deviceTextureObject, &resDesc, &texDesc, NULL));
}

void Texture::upload(const void* data) {
    // size of each row in byes including padding
    const size_t spitch = width * textureFormatSize(format);
    CUDA_CHECK(cudaMemcpy2DToArray(deviceArray, 0, 0, data, spitch, width * textureFormatSize(format),
        height, cudaMemcpyHostToDevice));
}

Texture::Texture(Texture&& other) : deviceArray(other.deviceArray), deviceTextureObject(other.deviceTextureObject), width(other.width), height(other.height), format(other.format) {
    other.deviceArray = nullptr;
    other.deviceTextureObject = 0;
}

Texture& Texture::operator=(Texture&& other) {
    std::swap(width, other.width);
    std::swap(height, other.height);
    std::swap(format, other.format);
    std::swap(deviceArray, other.deviceArray);
    std::swap(deviceTextureObject, other.deviceTextureObject);
    return *this;
}

Texture::~Texture() {
    cudaDestroyTextureObject(deviceTextureObject);
    cudaFreeArray(deviceArray);
}

cudaTextureObject_t Texture::getTextureObject() const {
    return deviceTextureObject;
}
