# LISTIFY
# Given a string of space-delimited tokens, reparse as a string of
# semi-colon delimited tokens, which in CMake land is exactly equivalent
# to a list
macro(listify OUT_LIST IN_STRING)
    string(REPLACE " " ";" ${OUT_LIST} ${IN_STRING})
endmacro()

# listify multiple-argument inputs
listify(MEX_INCLUDE_DIRS_LIST ${MEX_INCLUDE_DIRS})
if (${CONFIGURATION} MATCHES "Debug")
    listify(MEX_LIBS_LIST ${MEX_DEBUG_LIBS})
else()
    listify(MEX_LIBS_LIST ${MEX_LIBS})
endif()

# if it's MSVC building a Debug configuration, don't build bindings
if ("${CONFIGURATION}" MATCHES "Debug")
    message(STATUS "Matlab bindings are only available in Release configurations. Skipping...")
    return()
endif()

# -----------------------------------------------------------------------------
# Compile
# -----------------------------------------------------------------------------
# for each generated source file:
# 1. check if the file has already been compiled
# 2. attempt compile if required
# 3. if the compile fails, throw an error and cancel compilation
file(GLOB SOURCE_FILES "${CMAKE_CURRENT_BINARY_DIR}/src/*.cpp")
foreach(SOURCE_FILE ${SOURCE_FILES})
    # strip out the filename
    get_filename_component(FILENAME ${SOURCE_FILE} NAME_WE)
    # compile the source file using mex
    if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/+cv/${FILENAME}.${MATLAB_MEXEXT})
        execute_process(
            COMMAND ${MATLAB_MEX_SCRIPT} ${MEX_OPTS} "CXXFLAGS=\$CXXFLAGS ${MEX_CXXFLAGS}" ${MEX_INCLUDE_DIRS_LIST}
                    ${MEX_LIB_DIR} ${MEX_LIBS_LIST} ${SOURCE_FILE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/+cv
            OUTPUT_QUIET
            ERROR_VARIABLE FAILED
        )
    endif()
    # TODO: If a mex file fails to compile, should we error out?
    # TODO: Warnings are currently treated as errors...
    if (FAILED)
        message(FATAL_ERROR "Failed to compile ${FILENAME}: ${FAILED}")
    endif()
endforeach()
