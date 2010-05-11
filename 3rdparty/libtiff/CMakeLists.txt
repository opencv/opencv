# ----------------------------------------------------------------------------
#  CMake file for libtiff. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(libtiff)

# List of C++ files:

include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}/../include"
	)

add_definitions(-DHAVE_STRING_H=1)

# The .cpp files:

set(lib_srcs
	tif_stream.cxx
	tif_zip.c
	tif_write.c
	tif_warning.c
	tif_version.c
	tif_tile.c
	tif_thunder.c
	tif_swab.c
	tif_strip.c
	tif_read.c
	tif_print.c
	tif_predict.c
	tif_pixarlog.c
	tif_packbits.c
	tif_open.c
	tif_ojpeg.c
	tif_next.c
	tif_lzw.c
	tif_luv.c
	tif_jpeg.c
	tif_getimage.c
	tif_flush.c
	tif_fax3sm.c
	tif_fax3.c
	tif_extension.c
	tif_error.c
	tif_dumpmode.c
	tif_dirwrite.c
	tif_dirread.c
	tif_dirinfo.c
	tif_dir.c
	tif_compress.c
	tif_color.c
	tif_codec.c
	tif_close.c
	tif_aux.c
	)

if(UNIX)
    set(lib_srcs ${lib_srcs} tif_unix.c)
endif()

if(WIN32)
    set(lib_srcs ${lib_srcs} tif_win32.c)
endif(WIN32)

#if(APPLE)
#    set(lib_srcs ${lib_srcs} tif_apple.c)
#endif(APPLE)

file(GLOB lib_hdrs *.h*)

set(lib_ext_hdrs "../include/tiff.h" "../include/tiffio.h" "../include/tiffvers.h")

set(the_target "libtiff")

if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
endif()

if(UNIX)
    if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    endif()
endif()

add_library(${the_target} STATIC ${lib_srcs} ${lib_hdrs} ${lib_ext_hdrs})

set_target_properties(${the_target}
	PROPERTIES
	OUTPUT_NAME "${the_target}"
	DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/3rdparty/lib/"
	)
