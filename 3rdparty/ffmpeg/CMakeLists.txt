project(opencv_ffmpeg)

if(MSVC64 OR MINGW64)
   set(FFMPEG_SUFFIX _64)
endif()

set(module_bare_name "opencv_ffmpeg${FFMPEG_SUFFIX}.dll")
set(module_name "${CMAKE_CURRENT_SOURCE_DIR}/${module_bare_name}")

message(STATUS "ffmpeg output dir: ${EXECUTABLE_OUTPUT_PATH}")

add_custom_target(opencv_ffmpeg ALL
                  COMMAND ${CMAKE_COMMAND} -E copy "${module_name}" "${EXECUTABLE_OUTPUT_PATH}/${module_bare_name}"
                  COMMENT "Copying ${module_name} to the output directory")

install(FILES ${module_name} DESTINATION bin COMPONENT main)
