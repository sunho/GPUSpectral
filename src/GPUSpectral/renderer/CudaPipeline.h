#pragma once

#include <optix.h>
class Renderer;

struct CudaPipeline {
    CudaPipeline(Renderer& renderer, OptixDeviceContext context);

    OptixProgramGroup              raygenProgGroup;
    OptixProgramGroup              radianceMissGroup;
    OptixProgramGroup              radianceHitGroup;
    OptixProgramGroup              shadowHitGroup;
    OptixProgramGroup              shadowMissGroup;
    
    OptixModule                    ptxModule;
    OptixPipeline                  pipeline;
private:
    void initModule(Renderer& renderer, OptixDeviceContext context);
    void initProgramGroups(Renderer& renderer, OptixDeviceContext context);
    void initPipeline(Renderer& renderer, OptixDeviceContext context);
};
