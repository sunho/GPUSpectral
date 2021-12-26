#include "Procedural.h"

Image Procedural::createCheckerborad(uint32_t uSize, uint32_t vSize, uint32_t width, uint32_t height, float3 colorOn, float3 colorOff) {
    Image outImage(width, height);
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            uint32_t u = j * uSize / width;
            uint32_t v = i * vSize / height;
            if ((u & 0x1) ^ (v & 0x1)) {
                outImage[i][j] = colorOn;
            }
            else {
                outImage[i][j] = colorOff;
            }
        }
    }
    return outImage;
}
