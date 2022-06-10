# ----------------------------------------------------------------------------
#  CMake file for libspng. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

project(${SPNG_LIBRARY})

set(CURR_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set_property(GLOBAL PROPERTY SPNG_INCLUDE_DIR ${CURR_INCLUDE_DIR})
ocv_include_directories(${ZLIB_INCLUDE_DIRS})

file(GLOB_RECURSE spng_headers RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "*.h")
file(GLOB_RECURSE spng_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "*.c")

message(STATUS "libspng will be used as PNG codec")

# ----------------------------------------------------------------------------------
#         Define the library target:
# ----------------------------------------------------------------------------------

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
endif(MSVC)

add_library(${SPNG_LIBRARY} STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${spng_headers} ${spng_sources})
ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-variable)
target_link_libraries(${SPNG_LIBRARY} ${ZLIB_LIBRARIES})

set_target_properties(${SPNG_LIBRARY}
        PROPERTIES OUTPUT_NAME ${SPNG_LIBRARY}
        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
        COMPILE_PDB_NAME ${SPNG_LIBRARY}
        COMPILE_PDB_NAME_DEBUG "${SPNG_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
        ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
        )

target_compile_definitions(${SPNG_LIBRARY} PUBLIC SPNG_STATIC)

if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${SPNG_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
    ocv_install_target(${SPNG_LIBRARY} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev OPTIONAL)
endif()

ocv_install_3rdparty_licenses(${SPNG_LIBRARY} LICENSE)
