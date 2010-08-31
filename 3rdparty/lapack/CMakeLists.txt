# ----------------------------------------------------------------------------
#  CMake file for opencv_lapack. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(opencv_lapack)

# List of C++ files:

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	"${CMAKE_CURRENT_SOURCE_DIR}/../include"
    ${CMAKE_CURRENT_BINARY_DIR}
	)

# The .cpp files:
file(GLOB lib_srcs *.c)
file(GLOB lib_hdrs *.h)
set(lib_ext_hdrs "../include/f2c.h" "../include/cblas.h" "../include/clapack.h")

# ----------------------------------------------------------------------------------
# 				Define the library target:
# ----------------------------------------------------------------------------------

set(the_target "opencv_lapack")

add_library(${the_target} STATIC ${lib_srcs} ${lib_hdrs} ${lib_ext_hdrs})

if(PCHSupport_FOUND)
    set(pch_header ${CMAKE_CURRENT_SOURCE_DIR}/../include/clapack.h)
    if(${CMAKE_GENERATOR} MATCHES "Visual*" OR ${CMAKE_GENERATOR} MATCHES "Xcode*")
        if(${CMAKE_GENERATOR} MATCHES "Visual*")
            set(${the_target}_pch "precomp.c")
        endif()            
        add_native_precompiled_header(${the_target} ${pch_header})
    #elseif(CMAKE_COMPILER_IS_GNUCXX AND ${CMAKE_GENERATOR} MATCHES ".*Makefiles")
    #    add_precompiled_header(${the_target} ${pch_header})
    endif()
endif()

if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
endif()

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-parentheses -Wno-uninitialized -Wno-implicit-function-declaration -Wno-unused")
endif()

set_target_properties(${the_target}
	PROPERTIES OUTPUT_NAME "${the_target}"
	DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
	ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/3rdparty/lib
	)
