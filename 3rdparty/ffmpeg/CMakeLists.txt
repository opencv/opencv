project(opencv_ffmpeg)

if(MSVC64 OR MINGW64)
   set(FFMPEG_SUFFIX _64)
endif()

set(module_bare_name "opencv_ffmpeg${FFMPEG_SUFFIX}.dll")
set(module_name "${CMAKE_CURRENT_SOURCE_DIR}/${module_bare_name}")

get_target_property(bin_dir opencv_traincascade LOCATION)
get_filename_component(bin_dir ${bin_dir} PATH) 

message(STATUS "ffmpeg output dir: ${bin_dir}")

add_custom_target(opencv_ffmpeg ALL
                  COMMAND ${CMAKE_COMMAND} -E copy
                    ${module_name} ${bin_dir}/${module_bare_name}
                  COMMENT "Copying ${module_name} to the output directory")

install(FILES ${module_name} DESTINATION bin COMPONENT main)
