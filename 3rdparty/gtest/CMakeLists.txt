project(opencv_gtest)

# List of C++ files:

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

# The .cpp files:
file(GLOB lib_srcs *.cpp)
file(GLOB lib_hdrs *.h)

# ----------------------------------------------------------------------------------
# 				Define the library target:
# ----------------------------------------------------------------------------------

set(the_target "opencv_gtest")

add_library(${the_target} STATIC ${lib_srcs} ${lib_hdrs})

if(UNIX)
     if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
     endif()
endif()

set_target_properties(${the_target}
	PROPERTIES OUTPUT_NAME "${the_target}"
	DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
	ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/3rdparty/lib
	)
