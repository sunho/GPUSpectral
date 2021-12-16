set(SOURCE
  loader/SceneLoader.h
  loader/SceneLoader.cpp
  renderer/Renderer.h
  renderer/Renderer.cpp
  renderer/Scene.h
  renderer/Scene.cpp
  utils/CudaUtil.h
  main.cpp
)

set(SHADERS_HEADERS
  kernels/VectorMath.cuh
  kernels/PathTracer.cuh
  kernels/Common.cuh
  kernels/LightSampler.cuh
  kernels/BSDFSampler.cuh
)

set(SHADERS
  kernels/PathTracer.cu
)

set(SOURCE_LIST
  ${SOURCE}
  ${SHADERS}
  ${SHADERS_HEADERS}
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_LIST})

NVCUDA_COMPILE_PTX( SOURCES ${SHADERS}
                    DEPENDENCIES ${SHADERS_HEADERS}
                    TARGET_PATH "${CMAKE_CURRENT_BINARY_DIR}/kernels"
                    GENERATED_FILES PTX_SOURCES
                    NVCC_OPTIONS "--gpu-architecture=compute_50" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-Wno-deprecated-gpu-targets" "-I${OPTIX_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/kernels"
                  )

add_executable(GPUSpectral ${SOURCE} ${SHADERS} ${SHADERS_HEADERS} ${PTX_SOURCES}) 
target_link_libraries(
  GPUSpectral
  stb_image
  optix
  cuda
  tinyparser_mitsuba
  tinyobjloader
)

add_custom_command(
    TARGET GPUSpectral
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/kernels "$<TARGET_FILE_DIR:GPUSpectral>/kernels"
)

add_custom_command(
    TARGET GPUSpectral
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets "$<TARGET_FILE_DIR:GPUSpectral>/assets"
)