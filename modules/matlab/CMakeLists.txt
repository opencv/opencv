# ----------------------------------------------------------------------------
#  CMake file for Matlab/Octave support
# ----------------------------------------------------------------------------
macro(PREPEND TOKEN OUT IN)
    foreach(VAR ${IN})
        string(REGEX REPLACE "^/" "${TOKEN}/" TMP ${VAR})
        list(APPEND ${OUT} ${TMP})
    endforeach()
endmacro()

# make sure we're on a supported architecture with Matlab and python installed
if (IOS OR ANDROID OR NOT MATLAB_FOUND OR NOT PYTHONLIBS_FOUND)
    ocv_module_disable(matlab)
endif()

set(the_description "The Matlab/Octave bindings")
ocv_add_module(matlab   BINDINGS opencv_core opencv_imgproc 
                        OPTIONAL opencv_objdetect opencv_features2d opencv_video
                                 opencv_highgui opencv_ml opencv_calib3d opencv_photo
                                 opencv_nonfree opencv_calib)

# add the python generator to the python path
set(PYPATH_CACHE $ENV{PYTHONPATH})
set(ENV{PYTHONPATH} ${OPENCV_MODULE_opencv_python_LOCATION}/src2:$ENV{PYTHONPATH})

# get the include path of the Bridge
prepend("-I" MEX_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ----------------------------------------------------------------------------
#  Configure time components
# ----------------------------------------------------------------------------
message("-- Trying to generate Matlab code")
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generator/gen_matlab_caller.py 
            ${CMAKE_CURRENT_SOURCE_DIR}/test/rand.hpp ${CMAKE_CURRENT_BINARY_DIR}
    ERROR_VARIABLE GEN_ERROR
    OUTPUT_QUIET 
)

if (GEN_ERROR)
    message(${GEN_ERROR})
    message("-- Generating Matlab code failed. Disabling Matlab bindings...")
    # restore the pythonpath
    set(ENV{PYTHONPATH} ${PYPATH_CACHE})
    return()
else()
    message("-- Trying to generate Matlab code - OK")
endif()

# attempt to compile the file using mex
message("-- Trying to compile mex file")
execute_process(
    COMMAND ${MATLAB_MEX_SCRIPT} ${MEX_INCLUDES} ${CMAKE_CURRENT_BINARY_DIR}/src/rand.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src
    ERROR_VARIABLE MEX_ERROR
    OUTPUT_QUIET
)

if (MEX_ERROR)
    message(${MEX_ERROR})
    message("-- Error compiling mex file. Disabling Matlab bindings...")
    # restore the pythonpath
    set(ENV{PYTHONPATH} ${PYPATH_CACHE})
    return()
else()
    message("-- Trying to compile mex file - OK")
endif()

# if we make it here, mex works!
set_property(GLOBAL PROPERTY MEX_WORKS TRUE)

# ----------------------------------------------------------------------------
#  Build time components
# ----------------------------------------------------------------------------
#string(REPLACE "opencv_" "" OPENCV_MATLAB_MODULES "${OPENCV_MODULE_${the_module}_REQ_DEPS};
#                                                   ${OPENCV_MODULE_${the_module}_OPT_DEPS}")
foreach(module ${OPENCV_MATLAB_MODULES})
    if (HAVE_opencv_${module})
        list(APPEND opencv_hdrs "${OPENCV_MODULE_opencv_${module}_LOCATION}/include/opencv2/${module}.hpp")
    endif()
endforeach()

# synthesise the matlab sources
# TODO:These should be build-time (ie add_custom_command)
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generator/gen_matlab_caller.py 
            ${opencv_hdrs} ${CMAKE_CURRENT_BINARY_DIR})

# compile the matlab sources
file(GLOB SOURCE_FILES ${CMAKE_CURRENT_BINARY_DIR}/src)
foreach(SOURCE_FILE ${SOURCE_FILES})
  # compile the source file using mex
endforeach()

# restore the pythonpath
set(ENV{PYTHONPATH} ${PYPATH_CACHE})
