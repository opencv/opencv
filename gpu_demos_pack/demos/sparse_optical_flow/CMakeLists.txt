cmake_minimum_required(VERSION 2.8.3)

set(target demo_sparse_optical_flow)

include_directories(${OpenCV_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})
file(GLOB sources "*.h*" "*.cpp")

add_executable(${target} ${sources})
add_dependencies(${target} utility)
target_link_libraries(${target} ${OpenCV_LIBS} utility)
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
