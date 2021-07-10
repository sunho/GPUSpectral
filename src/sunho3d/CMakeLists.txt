set(SOURCE_LIST
    backend/Command.inc
    backend/vulkan/Handles.h
    backend/vulkan/Handles.cpp
    backend/vulkan/Context.h
    backend/vulkan/Context.cpp
    backend/vulkan/Driver.h
    backend/vulkan/Driver.cpp
    shaders/triangle.frag
    shaders/triangle.vert
    Engine.h
    Engine.cpp
    Window.h
    Window.cpp
    Entity.h
    Entity.cpp
    Renderer.h
    Renderer.cpp
    Scene.h
    Scene.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_LIST})
add_library(sunho3d ${SOURCE_LIST})
target_link_libraries(sunho3d PUBLIC glfw glm Vulkan)
target_include_directories(sunho3d PUBLIC .. ${CMAKE_CURRENT_BINARY_DIR}/..)

add_executable(embedfile EmbedFile.c)

function(add_shader TARGET SHADENAME SHADER)
    find_program(GLSLC glslc)

	set(current-shader-path ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER})
    set(current-output-path ${CMAKE_BINARY_DIR}/shaders/${SHADER}.spv)

	# Add a custom command to compile GLSL to SPIR-V.
	get_filename_component(current-output-dir ${current-output-path} DIRECTORY)
	file(MAKE_DIRECTORY ${current-output-dir})
	add_custom_command(
		OUTPUT ${current-output-path}
		COMMAND ${GLSLC} -o ${current-output-path} ${current-shader-path}
		DEPENDS ${current-shader-path}
		IMPLICIT_DEPENDS CXX ${current-shader-path}
		VERBATIM)

	# Make sure our native build depends on this output.
	set_source_files_properties(${current-output-path} PROPERTIES GENERATED TRUE)

    set(COUT ${CMAKE_CURRENT_BINARY_DIR}/${SHADENAME}.c)
    set(CHEAD ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADENAME}.h)
    add_custom_command(
        OUTPUT ${COUT}
        COMMAND embedfile ${SHADENAME} ${current-output-path}
        DEPENDS ${current-output-path})
    target_sources(${TARGET} PRIVATE ${COUT})

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)
    file(WRITE ${CHEAD} "#pragma once\nextern \"C\" const char ${SHADENAME}[]; extern \"C\" const size_t ${SHADENAME}_len;")

endfunction(add_shader)

add_shader(sunho3d triangle_vert triangle.vert)
add_shader(sunho3d triangle_frag triangle.frag)