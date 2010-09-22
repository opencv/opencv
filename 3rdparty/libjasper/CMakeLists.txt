if(ANDROID)
set(android_defs "-DEXCLUDE_MIF_SUPPORT -DEXCLUDE_PNM_SUPPORT -DEXCLUDE_BMP_SUPPORT -DEXCLUDE_RAS_SUPPORT  -DEXCLUDE_JPG_SUPPORT -DEXCLUDE_PGX_SUPPORT")
define_3rdparty_module(jasper)
set(android_defs "")
else(ANDROID)


# ----------------------------------------------------------------------------
#  CMake file for libjasper. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(libjasper)


add_definitions(-DEXCLUDE_MIF_SUPPORT -DEXCLUDE_PNM_SUPPORT -DEXCLUDE_BMP_SUPPORT -DEXCLUDE_RAS_SUPPORT  -DEXCLUDE_JPG_SUPPORT -DEXCLUDE_PGX_SUPPORT)

# List of C++ files:
include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	"${CMAKE_CURRENT_SOURCE_DIR}/../include"
	)

# The .cpp files:
file(GLOB lib_srcs *.c)
file(GLOB lib_hdrs *.h)
file(GLOB lib_ext_hdrs ../include/jasper/*.h)

# ----------------------------------------------------------------------------------
# 				Define the library target:
# ----------------------------------------------------------------------------------

set(the_target "libjasper")

add_library(${the_target} STATIC ${lib_srcs} ${lib_hdrs} ${lib_ext_hdrs})

if(MSVC)
    string(REPLACE "/W3" "/W0" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "/W4" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    add_definitions(-DJAS_WIN_MSVC_BUILD)
endif()

if(UNIX)
    if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-implicit-function-declaration -Wno-unused")
endif()



set_target_properties(${the_target}
	PROPERTIES
	OUTPUT_NAME "${the_target}"
	DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/3rdparty/lib"
	)
endif(ANDROID)#!android
