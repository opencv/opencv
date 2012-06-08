# ----------------------------------------------------------------------------
#  CMake file for libjpeg. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(${JPEG_LIBRARY})

ocv_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB lib_srcs *.c)
file(GLOB lib_hdrs *.h)

# ----------------------------------------------------------------------------------
#         Define the library target:
# ----------------------------------------------------------------------------------

add_library(${JPEG_LIBRARY} STATIC ${lib_srcs} ${lib_hdrs})

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
  set_source_files_properties(jcdctmgr.c PROPERTIES COMPILE_FLAGS "-O1")
endif()

ocv_warnings_disable(CMAKE_C_FLAGS -Wcast-align)

set_target_properties(${JPEG_LIBRARY}
  PROPERTIES OUTPUT_NAME ${JPEG_LIBRARY}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/3rdparty/${OPENCV_LIB_INSTALL_PATH}
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${JPEG_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  install(TARGETS ${JPEG_LIBRARY} ARCHIVE DESTINATION share/OpenCV/3rdparty/${OPENCV_LIB_INSTALL_PATH} COMPONENT main)
endif()
