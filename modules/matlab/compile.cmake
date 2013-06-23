macro(listify OUT_LIST IN_STRING)
    string(REPLACE " " ";" ${OUT_LIST} ${IN_STRING})
endmacro()

listify(MEX_INCLUDE_DIRS_LIST ${MEX_INCLUDE_DIRS})
file(GLOB SOURCE_FILES "${CMAKE_CURRENT_BINARY_DIR}/src/*.cpp")
foreach(SOURCE_FILE ${SOURCE_FILES})
    # strip out the filename
    get_filename_component(FILENAME ${SOURCE_FILE} NAME_WE)
    # compile the source file using mex
    execute_process(
        COMMAND ${MATLAB_MEX_SCRIPT} ${MEX_OPTS} ${MEX_INCLUDE_DIRS_LIST} 
                ${MEX_LIB_DIR} ${MEX_LIBS} ${SOURCE_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src
        ERROR_VARIABLE FAILED
    )
    # TODO: If a mex file fails to cmpile, should we error out?
    if (FAILED)
        message(FATAL_ERROR "Failed to compile ${FILENAME}: ${FAILED}")
    endif()
endforeach()
