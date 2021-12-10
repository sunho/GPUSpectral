set(SOURCE_LIST
  kernels/add.h
  kernels/add.cu
  main.cpp
)
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

include_directories( ${OPTIX_INCLUDE_DIR})
add_executable(particle_test ${SOURCE_LIST})