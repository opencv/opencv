if(ANDROID)
define_3rdparty_module(zlib)
else()
# ----------------------------------------------------------------------------
#  CMake file for zlib. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
PROJECT(zlib)

# List of C++ files:

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    )

file(GLOB lib_srcs *.c)
file(GLOB lib_hdrs *.h)
set(lib_ext_hdrs "../include/zlib.h" "../include/zconf.h")

set(the_target "zlib")

add_library(${the_target} STATIC ${lib_srcs} ${lib_hdrs} ${lib_ext_hdrs})

if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
endif()

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()

set_target_properties(${the_target}
    PROPERTIES
        OUTPUT_NAME "${the_target}"
        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/3rdparty/lib"
    )

endif(ANDROID)
