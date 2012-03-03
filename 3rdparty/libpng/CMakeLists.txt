# ----------------------------------------------------------------------------
#  CMake file for libpng. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(${PNG_LIBRARY})

# List of C++ files:

ocv_include_directories("${CMAKE_CURRENT_SOURCE_DIR}" ${ZLIB_INCLUDE_DIR})

file(GLOB lib_srcs *.c)
file(GLOB lib_hdrs *.h)

# ----------------------------------------------------------------------------------
# 				Define the library target:
# ----------------------------------------------------------------------------------

add_library(${PNG_LIBRARY} STATIC ${lib_srcs} ${lib_hdrs})

if(MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
endif()

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()

set_target_properties(${PNG_LIBRARY}
	PROPERTIES OUTPUT_NAME ${PNG_LIBRARY}
	DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/3rdparty/${OPENCV_LIB_INSTALL_PATH}"
	)
	
if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${PNG_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()
    
if(NOT BUILD_SHARED_LIBS)
  install(TARGETS ${PNG_LIBRARY} ARCHIVE DESTINATION share/OpenCV/3rdparty/${OPENCV_LIB_INSTALL_PATH} COMPONENT main)
endif()
