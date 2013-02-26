# ----------------------------------------------------------------------------
#  CMake file for libwebp. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(${WEBP_LIBRARY})

ocv_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB lib_srcs dec/*.c dsp/*.c enc/*.c mux/*.c utils/*.c webp/*.c)
file(GLOB lib_hdrs dec/*.h dsp/*.h enc/*.h mux/*.h utils/*.h webp/*.h)

if(NOT ANDROID)
	ocv_list_filterout(lib_srcs cpu-features.c)
endif()
 
# ----------------------------------------------------------------------------------
#         Define the library target:
# ----------------------------------------------------------------------------------

add_definitions(-DWEBP_USE_THREAD)

add_library(${WEBP_LIBRARY} STATIC ${lib_srcs} ${lib_hdrs})

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()

ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-variable -Wshadow)

set_target_properties(${WEBP_LIBRARY}
  PROPERTIES OUTPUT_NAME ${WEBP_LIBRARY}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${WEBP_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  install(TARGETS ${WEBP_LIBRARY} ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT main)
endif()
