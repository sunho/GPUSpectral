
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw)

add_subdirectory(glm)

add_subdirectory(fmt)

add_subdirectory(vk-bootstrap)
add_subdirectory(VulkanMemoryAllocator)

set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
add_subdirectory(spdlog)

add_subdirectory(TinyParser-Mitsuba)

find_package(Vulkan)
add_library(Vulkan INTERFACE)
target_compile_definitions(Vulkan INTERFACE -DHAVE_VULKAN)
target_link_libraries(Vulkan INTERFACE Vulkan::Vulkan)

add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tinygltf)

add_subdirectory(tinyobjloader)

add_subdirectory(SPIRV-Cross)

set(TRACY_CALLSTACK ON CACHE BOOL "" FORCE)
set(TRACY_ENABLE OFF CACHE BOOL "" FORCE)
add_subdirectory(tracy)

function(add_vulkan_library LIBNAME DEBUGLIBPATH RELEASELIBPATH)
    add_library(${LIBNAME} UNKNOWN IMPORTED GLOBAL)
    set_target_properties(${LIBNAME} PROPERTIES IMPORTED_LOCATION_DEBUG $ENV{VULKAN_SDK}/Lib/${DEBUGLIBPATH} IMPORTED_LOCATION_RELEASE $ENV{VULKAN_SDK}/Lib/${RELEASELIBPATH})
    set_property(TARGET ${LIBNAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES $ENV{VULKAN_SDK}/Include)
endfunction(add_vulkan_library)

# add_vulkan_library(GenericCodeGen GenericCodeGend.lib GenericCodeGen.lib)
# add_vulkan_library(OGLCompiler OGLCompilerd.lib OGLCompiler.lib)
# add_vulkan_library(OSDependent OSDependentd.lib OSDependent.lib)
# add_vulkan_library(glslang glslangd.lib glslang.lib)
# add_vulkan_library(SPIRV-Tools SPIRV-Toolsd.lib SPIRV-Tools.lib)
# add_vulkan_library(SPIRV-Tools-opt SPIRV-Tools-optd.lib SPIRV-Tools-opt.lib)
# add_vulkan_library(spirv-cross-glsl spirv-cross-glsld.lib spirv-cross-glsl.lib)
# add_vulkan_library(spirv-cross-core spirv-cross-cored.lib spirv-cross-core.lib)
# add_vulkan_library(spirv-cross-cpp spirv-cross-cppd.lib spirv-cross-cpp.lib)
# add_vulkan_library(shaderc shadercd.lib shaderc.lib)
# add_vulkan_library(MachineIndependent MachineIndependentd.lib MachineIndependent.lib)
# add_vulkan_library(shaderc-util shaderc_utild.lib shaderc_util.lib)
# add_vulkan_library(SPIRV SPIRVd.lib SPIRV.lib)
# target_link_libraries(shaderc INTERFACE shaderc-util SPIRV glslang SPIRV-Tools OSDependent OGLCompiler SPIRV-Tools-opt GenericCodeGen MachineIndependent)
# target_link_libraries(spirv-cross-core INTERFACE spirv-cross-glsl)