macro(listify OUT_LIST IN_STRING)
    string(REPLACE " " ";" ${OUT_LIST} ${IN_STRING})
endmacro()

listify(MEX_INCLUDE_DIRS_LIST ${MEX_INCLUDE_DIRS})
set(MEX_CXXFLAGS "CXXFLAGS=\$CXXFLAGS -msse -msse2 -msse3 -msse4.1 -msse4.2 -pedantic -Wall -Wextra -Weffc++ -Wno-unused-parameter -Wold-style-cast -Wshadow -Wmissing-declarations -Wmissing-include-dirs -Wnon-virtual-dtor -Wno-newline-eof")
file(GLOB SOURCE_FILES "${CMAKE_CURRENT_BINARY_DIR}/src/*.cpp")
foreach(SOURCE_FILE ${SOURCE_FILES})
    # strip out the filename
    get_filename_component(FILENAME ${SOURCE_FILE} NAME_WE)
    # compie the source file using mex
    if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/+cv/${FILENAME}.${MATLAB_MEXEXT})
        execute_process(
            COMMAND ${MATLAB_MEX_SCRIPT} "${MEX_CXXFLAGS}" ${MEX_INCLUDE_DIRS_LIST} 
                    ${MEX_LIB_DIR} ${MEX_LIBS} ${SOURCE_FILE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/+cv
            ERROR_VARIABLE FAILED
        )
    endif()
    # TODO: If a mex file fails to cmpile, should we error out?
    if (FAILED)
        message(FATAL_ERROR "Failed to compile ${FILENAME}: ${FAILED}")
    endif()
endforeach()
