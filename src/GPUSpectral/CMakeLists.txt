set(SOURCE_LIST
    backend/Handles.h
    backend/Handles.cpp
    backend/Command.inc
    backend/Program.h
    backend/Driver.h
    backend/PipelineState.h
    backend/DriverBase.h
    backend/DriverTypes.h
    backend/vulkan/VulkanTypes.h
    backend/vulkan/VulkanTypes.cpp
    backend/vulkan/VulkanHandles.h
    backend/vulkan/VulkanHandles.cpp
    backend/vulkan/VulkanDevice.h
    backend/vulkan/VulkanDevice.cpp
    backend/vulkan/VulkanWSI.h
    backend/vulkan/VulkanWSI.cpp
    backend/vulkan/VulkanDriver.h
    backend/vulkan/VulkanDriver.cpp
    backend/vulkan/VulkanBuffer.h
    backend/vulkan/VulkanBuffer.cpp
    backend/vulkan/VulkanPipelineCache.h
    backend/vulkan/VulkanPipelineCache.cpp
    backend/vulkan/VulkanTexture.h
    backend/vulkan/VulkanTexture.cpp
    backend/vulkan/VulkanRays.h
    backend/vulkan/VulkanRays.cpp
    renderer/framegraph/FrameGraph.h
    renderer/framegraph/FrameGraph.cpp
    renderer/framegraph/Resource.h
    renderer/Renderer.h
    renderer/Renderer.cpp
    renderer/PathTracer.h
    renderer/PathTracer.cpp
    renderer/Transform.h
    renderer/Camera.h
    renderer/Camera.cpp
    renderer/Scene.h
    renderer/Scene.cpp
    renderer/Mesh.h
    renderer/Mesh.cpp
    utils/Log.h
    utils/ResourceList.h
    utils/Util.h
    utils/FixedVector.h
    utils/Hash.h
    utils/GCPool.h
    utils/HalfFloat/umHalf.h
    utils/HalfFloat/umHalf.inl
    utils/HalfFloat/stdint.h
    engine/Engine.h
    engine/Engine.cpp
    engine/Window.h
    engine/Window.cpp    
    engine/Loader.h
    engine/Loader.cpp
    BSDF.inc
    main.cpp
)


source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_LIST})
add_executable(GPUSpectral ${SOURCE_LIST}) 
target_link_libraries(GPUSpectral PUBLIC fmt spirv-cross-core TracyClient tinyparser_mitsuba VulkanMemoryAllocator vk-bootstrap::vk-bootstrap glfw glm Vulkan tinygltf tinyobjloader)
set_property(TARGET GPUSpectral PROPERTY INCLUDE_DIRECTORIES  ${CMAKE_SOURCE_DIR}/external/SPIRV-Cross)
target_include_directories(GPUSpectral PUBLIC .. ${CMAKE_CURRENT_BINARY_DIR}/..)

add_executable(embedfile utils/EmbedFile.c)
set(SHADER_BINARIES "")
function(add_shader SHADER)
    find_program(GLSLC glslc)

	set(current-shader-path ${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/${SHADER})
    set(current-output-path ${CMAKE_BINARY_DIR}/shaders/${SHADER}.spv)

	# Add a custom command to compile GLSL to SPIR-V.
	get_filename_component(current-output-dir ${current-output-path} DIRECTORY)
	file(MAKE_DIRECTORY ${current-output-dir})
	add_custom_command(
		OUTPUT ${current-output-path}
		COMMAND ${GLSLC} --target-env=vulkan1.2 -o ${current-output-path} ${current-shader-path}
        COMMAND ${CMAKE_COMMAND} -E echo "COMIPLE SHADER"
		DEPENDS ${current-shader-path}
		IMPLICIT_DEPENDS CXX ${current-shader-path}
		VERBATIM)

	# Make sure our native build depends on this output.
	set_source_files_properties(${current-output-path} PROPERTIES GENERATED TRUE)
    set(SHADER_BINARIES ${SHADER_BINARIES} ${current-output-path} PARENT_SCOPE)
endfunction(add_shader)
add_shader(raygen.rgen)
add_shader(rayhit.rchit)
add_shader(miss.rmiss)
add_shader(shadowmiss.rmiss)
add_shader(DrawTexture.vert)
add_shader(DrawTexture.frag)

add_custom_target(shader_binary DEPENDS ${SHADER_BINARIES})
message("shader: ${SHADER_BINARIES}")

add_custom_command(
    TARGET GPUSpectral
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets "$<TARGET_FILE_DIR:GPUSpectral>/assets"
)

add_custom_target(
    copy_shaders
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/shaders "$<TARGET_FILE_DIR:GPUSpectral>/assets/shaders"
    COMMENT "Copying shaders"
    DEPENDS shader_binary
)

add_dependencies(GPUSpectral copy_shaders)