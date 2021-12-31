add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE VKGIRenderer)

add_custom_command(
    TARGET demo
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets "$<TARGET_FILE_DIR:demo>/assets"
)

add_custom_command(
    TARGET demo
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_FILE_DIR:VKGIRenderer>/assets" "$<TARGET_FILE_DIR:demo>/assets"
    COMMAND ${CMAKE_COMMAND} -E echo "Copying assets from VKGIRenderer"
)

add_dependencies(demo shader_binary)