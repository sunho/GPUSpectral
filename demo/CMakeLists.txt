add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE sunho3d)

add_custom_command(
    TARGET demo
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets "$<TARGET_FILE_DIR:demo>/assets"
)

add_custom_command(
    TARGET demo
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets "$<TARGET_FILE_DIR:demo>/assets"
)

add_custom_command(
    TARGET demo
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../src/sunho3d/assets "$<TARGET_FILE_DIR:demo>/assets"
)