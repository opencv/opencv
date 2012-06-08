# ----------------------------------------------------------------------------
#  CMake file for zlib. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

project(${ZLIB_LIBRARY} C)

include(CheckFunctionExists)
include(CheckIncludeFile)
include(CheckCSourceCompiles)

#
# Check for fseeko
#
check_function_exists(fseeko HAVE_FSEEKO)
if(NOT HAVE_FSEEKO)
  add_definitions(-DNO_FSEEKO)
endif()

#
# Check for unistd.h
#
check_include_file(unistd.h Z_HAVE_UNISTD_H)

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
  add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/zconf.h.cmakein"
               "${CMAKE_CURRENT_BINARY_DIR}/zconf.h" @ONLY)
ocv_include_directories("${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")

set(ZLIB_PUBLIC_HDRS
    "${CMAKE_CURRENT_BINARY_DIR}/zconf.h"
    zlib.h
)
set(ZLIB_PRIVATE_HDRS
    crc32.h
    deflate.h
    gzguts.h
    inffast.h
    inffixed.h
    inflate.h
    inftrees.h
    trees.h
    zutil.h
)
set(ZLIB_SRCS
    adler32.c
    compress.c
    crc32.c
    deflate.c
    gzclose.c
    gzlib.c
    gzread.c
    gzwrite.c
    inflate.c
    infback.c
    inftrees.c
    inffast.c
    trees.c
    uncompr.c
    zutil.c
)

add_library(${ZLIB_LIBRARY} STATIC ${ZLIB_SRCS} ${ZLIB_PUBLIC_HDRS} ${ZLIB_PRIVATE_HDRS})
set_target_properties(${ZLIB_LIBRARY} PROPERTIES DEFINE_SYMBOL ZLIB_DLL)

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()

ocv_warnings_disable(CMAKE_C_FLAGS /wd4013 -Wattributes -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations)

set_target_properties(${ZLIB_LIBRARY} PROPERTIES
        OUTPUT_NAME ${ZLIB_LIBRARY}
        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/3rdparty/${OPENCV_LIB_INSTALL_PATH}"
    )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${ZLIB_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  install(TARGETS ${ZLIB_LIBRARY} ARCHIVE DESTINATION share/OpenCV/3rdparty/${OPENCV_LIB_INSTALL_PATH} COMPONENT main)
endif()

ocv_parse_header2(ZLIB "${CMAKE_CURRENT_SOURCE_DIR}/zlib.h" ZLIB_VERSION PARENT_SCOPE)

