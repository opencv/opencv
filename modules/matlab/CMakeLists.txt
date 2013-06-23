# ----------------------------------------------------------------------------
#  CMake file for Matlab/Octave support
#
#  Matlab code generation and compilation is broken down into two distinct
#  stages: configure time and build time. The idea is that we want to give
#  the user reasonable guarantees that once they type 'make', wrapper
#  generation is unlikely to fail. Therefore we run a series of tests at
#  configure time to check the working status of the core components.
#
#  Configure Time
#  During configure time, the script attempts to ascertain whether the
#  generator and mex compiler are working for a given architecture.
#  Currently this involves:
#   1) Generating a simple CV_EXPORTS_W symbol and checking whether a file
#      of the symbol name is generated
#   2) Compiling a simple mex gateway to check that Bridge.hpp and mex.h
#      can be found, and that a file with the mexext is produced
#
#  Build Time
#  If the configure time tests pass, then we assume Matlab wrapper generation
#  will not fail during build time. We simply glob all of the symbols in
#  the OpenCV module headers, generate intermediate .cpp files, then compile
#  them with mex.
# ----------------------------------------------------------------------------

# PREPEND
# Given a list of strings IN and a TOKEN, prepend the token to each string
# and append to OUT. This is used for passing command line "-I", "-L" and "-l"
# arguments to mex. e.g.
# prepend("-I" OUT /path/to/include/dir) --> -I/path/to/include/dir
macro(PREPEND TOKEN OUT IN)
    foreach(VAR ${IN})
        list(APPEND ${OUT} "${TOKEN}${VAR}")
    endforeach()
endmacro()

# make sure we're on a supported architecture with Matlab and python installed
if (IOS OR ANDROID OR NOT MATLAB_FOUND OR NOT PYTHONLIBS_FOUND)
    ocv_module_disable(matlab)
    return()
endif()

# TODO: matlab bindings should depend on python (maybe)
set(the_description "The Matlab/Octave bindings")
ocv_add_module(matlab   BINDINGS opencv_core #TODO: does it actually NEED to depend on core?
                        OPTIONAL opencv_objdetect opencv_features2d opencv_video
                                 opencv_highgui opencv_ml opencv_calib3d opencv_photo
                                 opencv_nonfree opencv_calib opencv_imgproc)

# TODO: Undo this when building all modules to find python properly
#set(HDR_PARSER_PATH ${OPENCV_MODULE_opencv_python_LOCATION}/src2)
set(HDR_PARSER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../python/src2)

# set mex compiler options
prepend("-I" MEX_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
prepend("-L" MEX_LIB_DIR  ${CMAKE_BINARY_DIR}/lib)
set(MEX_OPTS "-largeArrayDims")

if (ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${the_module} PROPERTIES FOLDER "Matlab bindings")
endif()

if (BUILD_TESTS)
    add_subdirectory(test)
endif()

# ----------------------------------------------------------------------------
#  Configure time components
# ----------------------------------------------------------------------------
set(MATLAB_DEPS ${OPENCV_MODULE_${the_module}_REQ_DEPS} ${OPENCV_MODULE_${the_module}_OPT_DEPS})
foreach(opencv_module ${MATLAB_DEPS})
    if (HAVE_${opencv_module})
        string(REPLACE "opencv_" "" module ${opencv_module})
        list(APPEND opencv_hdrs "${OPENCV_MODULE_${opencv_module}_LOCATION}/include/opencv2/${module}.hpp")
        list(APPEND ${the_module}_ACTUAL_DEPS ${opencv_module})
        prepend("-I" MEX_INCLUDE_DIRS "${OPENCV_MODULE_${opencv_module}_LOCATION}/include")
        prepend("-l" MEX_LIBS ${opencv_module})
    endif()
endforeach()

# Configure checks
# Check to see whether the generator and the mex compiler are working.
# The checks currently test:
#   - whether the python generator can be found
#   - whether the python generator correctly outputs a file for a definition
#   - whether the mex compiler can find the required headers
#   - whether the mex compiler can compile a trivial definition
if (NOT MEX_WORKS)
    # attempt to generate a gateway for a function
    message(STATUS "Trying to generate Matlab code")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_SOURCE_DIR}/test/trigger.cpp
        COMMAND ${PYTHON_EXECUTABLE} 
                ${CMAKE_CURRENT_SOURCE_DIR}/generator/gen_matlab_caller.py ${HDR_PARSER_PATH}
                ${CMAKE_CURRENT_SOURCE_DIR}/test/test_generator.hpp ${CMAKE_BINARY_DIR}/junk
        ERROR_VARIABLE GEN_ERROR
        OUTPUT_QUIET 
    )

    if (GEN_ERROR)
        message(${GEN_ERROR})
        message(STATUS "Error generating Matlab code. Disabling Matlab bindings...")
        return()
    else()
        message(STATUS "Trying to generate Matlab code - OK")
    endif()

    # attempt to compile a gateway using mex
    message(STATUS "Trying to compile mex file")
    execute_process(
        COMMAND ${MATLAB_MEX_SCRIPT} ${MEX_OPTS} ${MEX_INCLUDE_DIRS} 
                ${CMAKE_CURRENT_SOURCE_DIR}/test/test_compiler.cpp
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/junk
        ERROR_VARIABLE MEX_ERROR
        OUTPUT_QUIET
    )

    if (MEX_ERROR)
        message(${MEX_ERROR})
        message(STATUS "Error compiling mex file. Disabling Matlab bindings...")
        return()
    else()
        message(STATUS "Trying to compile mex file - OK")
    endif()
endif()

# if we make it here, mex works!
set_property(GLOBAL PROPERTY MEX_WORKS TRUE)
set(MEX_WORKS True CACHE BOOL ADVANCED)

# ----------------------------------------------------------------------------
#  Build time components
# ----------------------------------------------------------------------------

# proxies
# these proxies are used to trigger the add_custom_commands
# (which do the real work) only when they're outdated
set(GENERATE_PROXY ${CMAKE_CURRENT_BINARY_DIR}/generate.proxy)
set(COMPILE_PROXY ${CMAKE_CURRENT_BINARY_DIR}/compile.proxy)

# generate
# call the python executable to generate the Matlab gateways
add_custom_command(
    OUTPUT ${GENERATE_PROXY}
    COMMAND ${PYTHON_EXECUTABLE} 
            ${CMAKE_CURRENT_SOURCE_DIR}/generator/gen_matlab_caller.py ${HDR_PARSER_PATH}
            ${opencv_hdrs} ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${GENERATE_PROXY}
    COMMENT "Generating matlab source files"
)

# compile
# call the mex compiler to compile the gateways
# because we don't know the source files at configure-time, this
# has to be executed in a separate script in cmake's script processing mode
add_custom_command(
    OUTPUT ${COMPILE_PROXY}
    COMMAND ${CMAKE_COMMAND} -DMATLAB_MEX_SCRIPT=${MATLAB_MEX_SCRIPT}
                             -DMEX_OPTS=${MEX_OPTS}
                             -DMEX_INCLUDE_DIRS="${MEX_INCLUDE_DIRS}"
                             -DMEX_LIB_DIR=${MEX_LIB_DIR}
                             -DMEX_LIBS="${MEX_LIBS}"
                             -P ${CMAKE_CURRENT_SOURCE_DIR}/compile.cmake
    COMMAND ${CMAKE_COMMAND} -E touch ${COMPILE_PROXY}
    COMMENT "Compiling Matlab source files. This could take a while..."
)

# targets
# opencv_matlab_sources --> opencv_matlab_compile
add_custom_target(${the_module}_sources ALL DEPENDS ${GENERATE_PROXY})
add_custom_target(${the_module} ALL DEPENDS ${COMPILE_PROXY})
add_dependencies(${the_module} ${the_module}_sources ${${the_module}_ACTUAL_DEPS})

# ----------------------------------------------------------------------------
#  Install time components
# ----------------------------------------------------------------------------
file(GLOB MATLAB_FUNCTIONS "${CMAKE_CURRENT_BINARY_DIR}/src/*.mex*")
file(GLOB MATLAB_CLASSES   "${CMAKE_CURRENT_BINARY_DIR}/src/private/*.mex*" "${CMAKE_CURRENT_BINARY_DIR}/+cv/*.m")
install(FILES ${MATLAB_FUNCTIONS} ${MATLAB_CLASSES}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/matlab/+cv 
)
