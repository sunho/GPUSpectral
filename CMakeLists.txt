cmake_minimum_required(VERSION 3.17)
project(GPUSpectral LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20) 

include("nvcuda_compile_ptx")
set(LOCAL_ARCH "amd64")

find_package(OptiX70)
find_package(OptiX71)
find_package(OptiX72)
find_package(OptiX73)
find_package(OptiX74)

add_subdirectory(src/GPUSpectral)