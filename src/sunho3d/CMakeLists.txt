set(SOURCE_LIST
    backend/Handles.h
    backend/Handles.cpp
    backend/Command.inc
    backend/Program.h
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
    renderer/shaders/ForwardPhong.frag
    renderer/shaders/ForwardPhong.vert
    renderer/shaders/DisplayTexture.frag
    renderer/shaders/DisplayTexture.vert
    renderer/shaders/ForwardRT.frag
    renderer/shaders/ForwardRT.vert
    renderer/shaders/common.glsl
    renderer/shaders/DDGIProbeRayGen.comp
    renderer/shaders/DDGIProbeRayShade.comp
    renderer/shaders/DDGIProbeUpdate.comp
    renderer/shaders/DDGIShade.vert
    renderer/shaders/DDGIShade.frag
    renderer/framegraph/FrameGraph.h
    renderer/framegraph/FrameGraph.cpp
    renderer/framegraph/Resource.h
    renderer/Renderer.h
    renderer/Renderer.cpp
    utils/ResourceList.h
    utils/FixedVector.h
    utils/Hash.h
    utils/GCPool.h
    utils/HalfFloat/umHalf.h
    utils/HalfFloat/umHalf.inl
    utils/HalfFloat/stdint.h
    Engine.h
    Engine.cpp
    Window.h
    Window.cpp
    Entity.h
    Entity.cpp
    Transform.h
    Light.h
    Light.cpp
    Camera.h
    Camera.cpp
    Scene.h
    Scene.cpp
    Loader.h
    Loader.cpp
    Material.h
    Material.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_LIST})
add_library(sunho3d ${SOURCE_LIST}) 
target_link_libraries(sunho3d PUBLIC VulkanMemoryAllocator vk-bootstrap::vk-bootstrap glfw glm Vulkan tinygltf tinyobjloader radeonrays)
target_include_directories(sunho3d PUBLIC .. ${CMAKE_CURRENT_BINARY_DIR}/..)

add_executable(embedfile utils/EmbedFile.c)

function(add_shader TARGET SHADENAME SHADER)
    find_program(GLSLC glslc)

	set(current-shader-path ${CMAKE_CURRENT_SOURCE_DIR}/renderer/shaders/${SHADER})
    set(current-output-path ${CMAKE_BINARY_DIR}/renderer/shaders/${SHADER}.spv)

	# Add a custom command to compile GLSL to SPIR-V.
	get_filename_component(current-output-dir ${current-output-path} DIRECTORY)
	file(MAKE_DIRECTORY ${current-output-dir})
	add_custom_command(
		OUTPUT ${current-output-path}
		COMMAND ${GLSLC} --target-env=vulkan1.2 -o ${current-output-path} ${current-shader-path}
		DEPENDS ${current-shader-path}
		IMPLICIT_DEPENDS CXX ${current-shader-path}
		VERBATIM)

	# Make sure our native build depends on this output.
	set_source_files_properties(${current-output-path} PROPERTIES GENERATED TRUE)

    set(COUT ${CMAKE_CURRENT_BINARY_DIR}/${SHADENAME}.c)
    set(CHEAD ${CMAKE_CURRENT_BINARY_DIR}/renderer/shaders/${SHADENAME}.h)
    add_custom_command(
        OUTPUT ${COUT}
        COMMAND embedfile ${SHADENAME} ${current-output-path}
        DEPENDS ${current-output-path})
    target_sources(${TARGET} PRIVATE ${COUT})

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/renderer/shaders)
    file(WRITE ${CHEAD} "#pragma once\nextern \"C\" const char ${SHADENAME}[]; extern \"C\" const size_t ${SHADENAME}Size;")

endfunction(add_shader)

add_shader(sunho3d ForwardPhongVert ForwardPhong.vert)
add_shader(sunho3d ForwardPhongFrag ForwardPhong.frag)

add_shader(sunho3d DisplayTextureVert DisplayTexture.vert)
add_shader(sunho3d DisplayTextureFrag DisplayTexture.frag)

add_shader(sunho3d ForwardRTVert ForwardRT.vert)
add_shader(sunho3d ForwardRTFrag ForwardRT.frag)

add_shader(sunho3d DDGIProbeRayGen DDGIProbeRayGen.comp)
add_shader(sunho3d DDGIProbeRayShade DDGIProbeRayShade.comp)
add_shader(sunho3d DDGIProbeUpdate DDGIProbeUpdate.comp)
add_shader(sunho3d DDGIShadeVert DDGIShade.vert)
add_shader(sunho3d DDGIShadeFrag DDGIShade.frag)
add_shader(sunho3d GBufferGenVert GBufferGen.vert)
add_shader(sunho3d GBufferGenFrag GBufferGen.frag)
add_shader(sunho3d DeferredRenderVert DeferredRender.vert)
add_shader(sunho3d DeferredRenderFrag DeferredRender.frag)