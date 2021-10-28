
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw)

add_subdirectory(glm)


add_subdirectory(vk-bootstrap)
add_subdirectory(VulkanMemoryAllocator)

add_subdirectory(spdlog EXCLUDE_FROM_ALL)
set(spdlog_DIR ${CMAKE_CURRENT_SOURCE_DIR}/spdlog/include)
set(ENABLE_VULKAN on)
set(ENABLE_TESTING off)
add_subdirectory(RadeonRays)

find_package(Vulkan)
add_library(Vulkan INTERFACE)
target_compile_definitions(Vulkan INTERFACE -DHAVE_VULKAN)
target_link_libraries(Vulkan INTERFACE Vulkan::Vulkan)

add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tinygltf)

add_library(tinyobjloader INTERFACE)
target_include_directories(tinyobjloader INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader)