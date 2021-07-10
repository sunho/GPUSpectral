cmake_minimum_required(VERSION 3.17)
project(sunho3d)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 17) 

add_subdirectory(external)
add_subdirectory(src/sunho3d)
add_subdirectory(demo)