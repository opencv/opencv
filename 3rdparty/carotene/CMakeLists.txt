cmake_minimum_required(VERSION ${MIN_VER_CMAKE} FATAL_ERROR)

project(Carotene)

set(CAROTENE_NS "carotene" CACHE STRING "Namespace for Carotene definitions")

set(CAROTENE_INCLUDE_DIR include)
set(CAROTENE_SOURCE_DIR src)

file(GLOB_RECURSE carotene_headers RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "${CAROTENE_INCLUDE_DIR}/*.hpp")
file(GLOB_RECURSE carotene_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "${CAROTENE_SOURCE_DIR}/*.cpp"
                                                                        "${CAROTENE_SOURCE_DIR}/*.hpp")

include_directories(${CAROTENE_INCLUDE_DIR})

if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden ${CMAKE_CXX_FLAGS}")

    # allow more inlines - these parameters improve performance for:
    # - matchTemplate about 5-10%
    # - goodFeaturesToTrack 10-20%
    # - cornerHarris 30% for some cases
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0.0")
        set_source_files_properties(${carotene_sources} COMPILE_FLAGS "--param ipcp-unit-growth=100000 --param inline-unit-growth=100000 --param large-stack-frame-growth=5000")
    else()
        set_source_files_properties(${carotene_sources} COMPILE_FLAGS "--param ipa-cp-unit-growth=100000 --param inline-unit-growth=100000 --param large-stack-frame-growth=5000")
    endif()
endif()

if(APPLE AND CV_CLANG AND WITH_NEON)
    ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-unused-function)
endif()

add_library(carotene_objs OBJECT EXCLUDE_FROM_ALL
  ${carotene_headers}
  ${carotene_sources}
)

if(NOT CAROTENE_NS STREQUAL "carotene")
    target_compile_definitions(carotene_objs PUBLIC "-DCAROTENE_NS=${CAROTENE_NS}")
endif()

if(WITH_NEON)
    target_compile_definitions(carotene_objs PRIVATE "-DWITH_NEON")
    if(NOT DEFINED CAROTENE_NEON_ARCH )
    elseif(CAROTENE_NEON_ARCH EQUAL 8)
	    target_compile_definitions(carotene_objs PRIVATE "-DCAROTENE_NEON_ARCH=8")
    elseif(CAROTENE_NEON_ARCH EQUAL 7)
	    target_compile_definitions(carotene_objs PRIVATE "-DCAROTENE_NEON_ARCH=7")
    else()
	    target_compile_definitions(carotene_objs PRIVATE "-DCAROTENE_NEON_ARCH=0")
    endif()
endif()

 if(MINGW) 
    target_compile_definitions(carotene_objs PRIVATE "-D_USE_MATH_DEFINES=1")
 endif()

# we add dummy file to fix XCode build
add_library(carotene STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} "$<TARGET_OBJECTS:carotene_objs>" "${CAROTENE_SOURCE_DIR}/dummy.cpp")
