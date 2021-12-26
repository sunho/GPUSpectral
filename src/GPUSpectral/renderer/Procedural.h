#pragma once

#include <vector>
#include "Image.h"
#include "../kernels/VectorMath.cuh"

namespace Procedural {
Image createCheckerborad(uint32_t uSize, uint32_t vSize, uint32_t width, uint32_t height, float3 colorOn, float3 colorOff);
};