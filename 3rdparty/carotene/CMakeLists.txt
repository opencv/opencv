cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)

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

    set_source_files_properties(${carotene_sources} COMPILE_FLAGS "--param ipcp-unit-growth=100000 --param inline-unit-growth=100000 --param large-stack-frame-growth=5000")
endif()

add_library(carotene_objs OBJECT
  ${carotene_headers}
  ${carotene_sources}
)

if(NOT CAROTENE_NS STREQUAL "carotene")
    target_compile_definitions(carotene_objs PUBLIC "-DCAROTENE_NS=${CAROTENE_NS}")
endif()

if(WITH_NEON)
    target_compile_definitions(carotene_objs PRIVATE "-DWITH_NEON")
endif()

set_target_properties(carotene_objs PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(carotene STATIC EXCLUDE_FROM_ALL "$<TARGET_OBJECTS:carotene_objs>")
