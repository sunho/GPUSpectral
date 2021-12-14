cmake_minimum_required(VERSION 3.17)
project(GPUSpectral LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20) 

include("nvcuda_compile_ptx")
set(LOCAL_ARCH "amd64")

add_subdirectory(external)
add_subdirectory(src/GPUSpectral)