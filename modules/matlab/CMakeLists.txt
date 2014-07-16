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
    foreach(VAR ${IN} ${ARGN})
        list(APPEND ${OUT} "${TOKEN}${VAR}")
    endforeach()
endmacro()


# WARN_MIXED_PRECISION
# Formats a warning message if the compiler and Matlab bitness is different
macro(WARN_MIXED_PRECISION COMPILER_BITNESS MATLAB_BITNESS)
    set(MSG "Your compiler is ${COMPILER_BITNESS}-bit")
    set(MSG "${MSG} but your version of Matlab is ${MATLAB_BITNESS}-bit.")
    set(MSG "${MSG} To build Matlab bindings, please switch to a ${MATLAB_BITNESS}-bit compiler.")
    message(WARNING ${MSG})
endmacro()

# ----------------------------------------------------------------------------
#  Architecture checks
# ----------------------------------------------------------------------------
# make sure we're on a supported architecture with Matlab and python installed
if (IOS OR ANDROID OR NOT MATLAB_FOUND)
    ocv_module_disable(matlab)
    return()
elseif (NOT PYTHONLIBS_FOUND)
    message(WARNING "A required dependency of the matlab module (PythonLibs) was not found. Disabling Matlab bindings...")
    ocv_module_disable(matlab)
    return()
endif()


# If the user built OpenCV as X-bit, but they have a Y-bit version of Matlab,
# attempting to link to OpenCV during binding generation will fail, since
# mixed precision pointers are not allowed. Disable the bindings.
math(EXPR ARCH "${CMAKE_SIZEOF_VOID_P} * 8")
if (${ARCH} EQUAL 32 AND ${MATLAB_ARCH} MATCHES "64")
    warn_mixed_precision("32" "64")
    ocv_module_disable(matlab)
    return()
elseif (${ARCH} EQUAL 64 AND NOT ${MATLAB_ARCH} MATCHES "64")
    warn_mixed_precision("64" "32")
    ocv_module_disable(matlab)
    return()
endif()

# If it's MSVC, warn the user that bindings will only be built in Release mode.
# Debug mode seems to cause issues...
if (MSVC)
    message(STATUS "Warning: Matlab bindings will only be built in Release configurations")
endif()


# ----------------------------------------------------------------------------
#  Configure time components
# ----------------------------------------------------------------------------
set(the_description "The Matlab/Octave bindings")
ocv_add_module(matlab   BINDINGS
                        OPTIONAL opencv_core
                                 opencv_imgproc opencv_ml
                                 opencv_imgcodecs opencv_videoio opencv_highgui
                                 opencv_objdetect opencv_flann opencv_features2d
                                 opencv_photo opencv_video opencv_videostab
                                 opencv_calib opencv_calib3d
                                 opencv_stitching opencv_superres
                                 opencv_nonfree
)

# get the commit information
execute_process(COMMAND git log -1 --pretty=%H OUTPUT_VARIABLE GIT_COMMIT ERROR_QUIET)
string(REGEX REPLACE "(\r?\n)+$" "" GIT_COMMIT "${GIT_COMMIT}")

# set the path to the C++ header and doc parser, and template engine
set(JINJA2_PATH ${CMAKE_SOURCE_DIR}/3rdparty)
set(HDR_PARSER_PATH ${CMAKE_SOURCE_DIR}/modules/python/src2)
set(RST_PARSER_PATH ${CMAKE_SOURCE_DIR}/modules/java/generator)

# set mex compiler options
prepend("-I" MEX_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
if (MSVC)
    prepend("-L" MEX_LIB_DIR  ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})
else()
    prepend("-L" MEX_LIB_DIR  ${LIBRARY_OUTPUT_PATH})
endif()
set(MEX_OPTS "-largeArrayDims")

if (BUILD_TESTS)
    add_subdirectory(test)
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


# intersection of available modules and optional dependencies
# 1. populate the command-line include directories (-I/path/to/module/header, ...)
# 2. populate the command-line link libraries (-lopencv_core, ...) for Debug and Release
set(MATLAB_DEPS ${OPENCV_MODULE_${the_module}_REQ_DEPS} ${OPENCV_MODULE_${the_module}_OPT_DEPS})
foreach(opencv_module ${MATLAB_DEPS})
    if (HAVE_${opencv_module})
        string(REPLACE "opencv_" "" module ${opencv_module})
        list(APPEND opencv_modules ${module})
        list(APPEND ${the_module}_ACTUAL_DEPS ${opencv_module})
        prepend("-I" MEX_INCLUDE_DIRS "${OPENCV_MODULE_${opencv_module}_LOCATION}/include")
        prepend("-l" MEX_LIBS ${opencv_module}${OPENCV_DLLVERSION})
        prepend("-l" MEX_DEBUG_LIBS ${opencv_module}${OPENCV_DLLVERSION}${OPENCV_DEBUG_POSTFIX})
    endif()
endforeach()

# add extra headers by hand
list(APPEND opencv_extra_hdrs "core=${OPENCV_MODULE_opencv_core_LOCATION}/include/opencv2/core/base.hpp")
list(APPEND opencv_extra_hdrs "video=${OPENCV_MODULE_opencv_video_LOCATION}/include/opencv2/video/tracking.hpp")

# pass the OPENCV_CXX_EXTRA_FLAGS through to the mex compiler
# remove the visibility modifiers, so the mex gateway is visible
# TODO: get mex working without warnings
string(REGEX REPLACE "[^\ ]*visibility[^\ ]*" "" MEX_CXXFLAGS "${OPENCV_EXTRA_FLAGS} ${OPENCV_EXTRA_CXX_FLAGS}")

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
        COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/generator/gen_matlab.py
                --jinja2    ${JINJA2_PATH}
                --hdrparser ${HDR_PARSER_PATH}
                --rstparser ${RST_PARSER_PATH}
                --extra     "test=${CMAKE_CURRENT_SOURCE_DIR}/test/test_generator.hpp"
                --outdir    ${CMAKE_BINARY_DIR}/junk
        ERROR_VARIABLE GEN_ERROR
        OUTPUT_QUIET
    )

    if (GEN_ERROR)
        message(${GEN_ERROR})
        message(STATUS "Error generating Matlab code. Disabling Matlab bindings...")
        ocv_module_disable(matlab)
        return()
    else()
        message(STATUS "Trying to generate Matlab code - OK")
    endif()

    # attempt to compile a gateway using mex
    message(STATUS "Trying to compile mex file")
    execute_process(
        COMMAND ${MATLAB_MEX_SCRIPT} ${MEX_OPTS} "CXXFLAGS=\$CXXFLAGS ${MEX_CXX_FLAGS}"
                ${MEX_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_compiler.cpp
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/junk
        ERROR_VARIABLE MEX_ERROR
        OUTPUT_QUIET
    )

    if (MEX_ERROR)
        message(${MEX_ERROR})
        message(STATUS "Error compiling mex file. Disabling Matlab bindings...")
        ocv_module_disable(matlab)
        return()
    else()
        message(STATUS "Trying to compile mex file - OK")
    endif()
endif()

# if we make it here, mex works!
set(MEX_WORKS True CACHE BOOL ADVANCED)


# ----------------------------------------------------------------------------
#  Build time components
# ----------------------------------------------------------------------------

# proxies
# these proxies are used to trigger the add_custom_commands
# (which do the real work) only when they're outdated
set(GENERATE_PROXY ${CMAKE_CURRENT_BINARY_DIR}/generate.proxy)
set(COMPILE_PROXY ${CMAKE_CURRENT_BINARY_DIR}/compile.proxy)
# TODO: Remove following line before merging with master
file(REMOVE ${GENERATE_PROXY} ${COMPILE_PROXY})

# generate
# call the python executable to generate the Matlab gateways
add_custom_command(
    OUTPUT ${GENERATE_PROXY}
    COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/generator/gen_matlab.py
            --jinja2     ${JINJA2_PATH}
            --hdrparser  ${HDR_PARSER_PATH}
            --rstparser  ${RST_PARSER_PATH}
            --moduleroot ${CMAKE_SOURCE_DIR}/modules
            --modules    ${opencv_modules}
            --extra      ${opencv_extra_hdrs}
            --outdir     ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/generator/build_info.py
            --jinja2         ${JINJA2_PATH}
            --os             ${CMAKE_SYSTEM}
            --arch           ${ARCH} ${CMAKE_SYSTEM_PROCESSOR}
            --compiler       ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}
            --mex_arch       ${MATLAB_ARCH}
            --mex_script     ${MATLAB_MEX_SCRIPT}
            --cxx_flags      ${MEX_CXXFLAGS}
            --opencv_version ${OPENCV_VERSION}
            --commit         ${GIT_COMMIT}
            --modules        ${opencv_modules}
            --configuration  $<CONFIGURATION>
            --outdir         ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/generator/cvmex.py
            --jinja2 ${JINJA2_PATH}
            --opts="${MEX_OPTS}"
            --include_dirs="${MEX_INCLUDE_DIRS}"
            --lib_dir="${MEX_LIB_DIR}"
            --libs="${MEX_LIBS}"
            --flags  ${MEX_CXXFLAGS}
            --outdir ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/test/help.m ${CMAKE_CURRENT_BINARY_DIR}/+cv
    COMMAND ${CMAKE_COMMAND} -E touch ${GENERATE_PROXY}
    COMMENT "Generating Matlab source files"
)

# compile
# call the mex compiler to compile the gateways
# because we don't know the source files at configure-time, this
# has to be executed in a separate script in cmake's script processing mode
add_custom_command(
    OUTPUT ${COMPILE_PROXY}
    COMMAND ${CMAKE_COMMAND} -DMATLAB_MEX_SCRIPT=${MATLAB_MEX_SCRIPT}
                             -DMATLAB_MEXEXT=${MATLAB_MEXEXT}
                             -DMEX_OPTS=${MEX_OPTS}
                             -DMEX_CXXFLAGS=${MEX_CXX_FLAGS}
                             -DMEX_INCLUDE_DIRS="${MEX_INCLUDE_DIRS}"
                             -DMEX_LIB_DIR="${MEX_LIB_DIR}"
                             -DCONFIGURATION="$<CONFIGURATION>"
                             -DMEX_LIBS="${MEX_LIBS}"
                             -DMEX_DEBUG_LIBS="${MEX_DEBUG_LIBS}"
                             -P ${CMAKE_CURRENT_SOURCE_DIR}/compile.cmake
    COMMAND ${CMAKE_COMMAND} -E touch ${COMPILE_PROXY}
    COMMENT "Compiling Matlab source files. This could take a while..."
)

# targets
# opencv_matlab_sources --> opencv_matlab
add_custom_target(${the_module}_sources ALL DEPENDS ${GENERATE_PROXY})
add_custom_target(${the_module} ALL DEPENDS ${COMPILE_PROXY})
add_dependencies(${the_module} ${the_module}_sources ${${the_module}_ACTUAL_DEPS})

if (ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${the_module} PROPERTIES FOLDER "modules")
endif()


# ----------------------------------------------------------------------------
#  Install time components
# ----------------------------------------------------------------------------
# NOTE: Trailing slashes on the DIRECTORY paths are important!
# TODO: What needs to be done with rpath????

# install the +cv directory verbatim
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${OPENCV_INCLUDE_INSTALL_PATH})
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/+cv/     DESTINATION matlab/+cv)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cv.m         DESTINATION matlab)

# update the custom mex compiler to point to the install locations
string(REPLACE ";" "\\ " MEX_OPTS "${MEX_OPTS}")
string(REPLACE ";" "\\ " MEX_LIBS "${MEX_LIBS}")
string(REPLACE " " "\\ " MEX_CXXFLAGS ${MEX_CXXFLAGS})
string(REPLACE ";" "\\ " MEX_INCLUDE_DIRS "${MEX_INCLUDE_DIRS}")
install(CODE
    "execute_process(
    COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/generator/cvmex.py
            --jinja2 ${JINJA2_PATH}
            --opts=${MEX_OPTS}
            --include_dirs=-I${CMAKE_INSTALL_PREFIX}/${OPENCV_INCLUDE_INSTALL_PATH}
            --lib_dir=-L${CMAKE_INSTALL_PREFIX}/${OPENCV_LIB_INSTALL_PATH}
            --libs=${MEX_LIBS}
            --flags=${MEX_CXXFLAGS}
            --outdir ${CMAKE_INSTALL_PREFIX}/matlab
    )"
)
