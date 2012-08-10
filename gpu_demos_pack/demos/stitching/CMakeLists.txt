cmake_minimum_required(VERSION 2.8.3)

set(target demo_stitching)

include_directories(${OpenCV_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})
file(GLOB sources "*.h*" "*.cpp")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(TBB REQUIRED)
if(TBB_FOUND AND NOT TBB_INTERFACE_VERSION LESS 5000)
    set(HAVE_TBB 1)
    include_directories(${TBB_INCLUDE_DIRS})
    link_directories(${TBB_LIBRARY_DIRS})
endif()

add_executable(${target} ${sources})
add_dependencies(${target} utility_lib)
target_link_libraries(${target} ${OpenCV_LIBS} utility_lib tbb)
set_target_properties(${target} PROPERTIES DEBUG_POSTFIX d)

set_target_properties(
		${target} PROPERTIES 
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
		ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
	set_target_properties(
			${target} PROPERTIES 
			RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin
			LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin
			ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)
