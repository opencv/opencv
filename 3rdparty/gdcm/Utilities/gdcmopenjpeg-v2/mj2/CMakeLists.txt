# Makefile for the MJ2 codecs of the OpenJPEG library: frames_to_mj2, mj2_to_frames, extract_j2k_from_mj2 and wrap_j2k_in_mj2

# Headers file are located here:
include_directories(
  ${OPENJPEG_SOURCE_DIR}/libopenjpeg
  )

add_executable(frames_to_mj2
  frames_to_mj2.c
  compat/getopt.c
  mj2_convert.c mj2.c )
target_link_libraries(frames_to_mj2 ${OPJ_PREFIX}openjpeg)
if(UNIX)
  target_link_libraries(frames_to_mj2 m)
endif()

add_executable(mj2_to_frames
    mj2_to_frames.c
    compat/getopt.c mj2_convert.c mj2.c )
target_link_libraries(mj2_to_frames ${OPJ_PREFIX}openjpeg)
if(UNIX)
  target_link_libraries(mj2_to_frames m)
endif()

add_executable(extract_j2k_from_mj2
    extract_j2k_from_mj2.c
    mj2.c )
target_link_libraries(extract_j2k_from_mj2 ${OPJ_PREFIX}openjpeg)
if(UNIX)
  target_link_libraries(extract_j2k_from_mj2 m)
endif()

add_executable(wrap_j2k_in_mj2
    wrap_j2k_in_mj2.c
    mj2.c )
target_link_libraries(wrap_j2k_in_mj2 ${OPJ_PREFIX}openjpeg)
if(UNIX)
  target_link_libraries(wrap_j2k_in_mj2 m)
endif()

install(TARGETS frames_to_mj2 mj2_to_frames extract_j2k_from_mj2 wrap_j2k_in_mj2
	DESTINATION bin)
