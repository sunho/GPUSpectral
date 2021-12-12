set(SOURCE
  main.cpp
)

set(SHADERS_HEADERS
  kernels/vector_math.h
  kernels/path_tracer.h
)

set(SHADERS
  kernels/path_tracer.cu
)

set(SOURCE_LIST
  ${SOURCE}
  ${SHADERS}
  ${SHADERS_HEADERS}
)

find_package(CUDA 10.0 REQUIRED)
if (OptiX74_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX74_INCLUDE_DIR}")
elseif (OptiX73_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX73_INCLUDE_DIR}")
elseif (OptiX72_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX72_INCLUDE_DIR}")
elseif(OptiX71_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX71_INCLUDE_DIR}")
elseif(OptiX70_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX70_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "No OptiX SDK 7.x found.")
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_LIST})

include_directories(${OPTIX_INCLUDE_DIR})

NVCUDA_COMPILE_PTX( SOURCES ${SHADERS}
                    DEPENDENCIES ${SHADERS_HEADERS}
                    TARGET_PATH "${CMAKE_CURRENT_BINARY_DIR}/kernels"
                    GENERATED_FILES PTX_SOURCES
                    NVCC_OPTIONS "--gpu-architecture=compute_50" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-Wno-deprecated-gpu-targets" "-I${OPTIX_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/kernels"
                  )

add_executable(GPUSpectral ${SOURCE} ${SHADERS_HEADERS} ${PTX_SOURCES}) 
target_link_libraries(
  GPUSpectral
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
)

add_custom_command(
    TARGET GPUSpectral
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/kernels "$<TARGET_FILE_DIR:GPUSpectral>/kernels"
)
