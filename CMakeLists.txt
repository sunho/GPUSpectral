cmake_minimum_required(VERSION 3.17)
project(VKGIRenderer)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")


add_subdirectory(external)
add_subdirectory(src/VKGIRenderer)
