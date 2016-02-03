#
# Try to find the FreeImage library and include path.
# Once done this will define
#
# FREEIMAGE_FOUND
# FREEIMAGE_INCLUDE_PATH
# FREEIMAGE_LIBRARY
#

if (WIN32)
	find_path( FREEIMAGE_INCLUDE_PATH FreeImage.h
		${OPENJPEG_SOURCE_DIR}/libs/FreeImage
		DOC "The directory where FreeImage.h resides")
	find_library( FREEIMAGE_LIBRARY
		NAMES FreeImage freeimage freeimage.s
		PATHS
		${OPENJPEG_SOURCE_DIR}/libs/FreeImage
		DOC "The FreeImage library")
else ()
	find_path( FREEIMAGE_INCLUDE_PATH FreeImage.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		DOC "The directory where FreeImage.h resides")
	find_library( FREEIMAGE_LIBRARY
		NAMES FreeImage freeimage
		PATHS
		/usr/lib64
		/usr/lib
		/usr/local/lib64
		/usr/local/lib
		/sw/lib
		/opt/local/lib
		DOC "The FreeImage library")
endif ()

set(FREEIMAGE_LIBRARIES ${FREEIMAGE_LIBRARY})

if (FREEIMAGE_INCLUDE_PATH AND FREEIMAGE_LIBRARY)
	set( FREEIMAGE_FOUND TRUE CACHE BOOL "Set to TRUE if GLEW is found, FALSE otherwise")
else ()
	set( FREEIMAGE_FOUND FALSE CACHE BOOL "Set to TRUE if GLEW is found, FALSE otherwise")
endif ()

mark_as_advanced(
	FREEIMAGE_FOUND
	FREEIMAGE_LIBRARY
	FREEIMAGE_LIBRARIES
	FREEIMAGE_INCLUDE_PATH)
