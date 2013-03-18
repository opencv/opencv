# ----------------------------------------------------------------------------
#  CMake file for Matlab/Octave support
# ----------------------------------------------------------------------------

# make sure we're on a supported architecture with Matlab and python installed
if (IOS OR ANDROID OR NOT MATLAB_FOUND OR NOT PYTHONLIBS_FOUND)
    ocv_module_disable(matlab)
endif()

set(the_description "The Matlab/Octave bindings")
ocv_add_module(matlab   BINDINGS opencv_core opencv_imgproc 
                        OPTIONAL opencv_objdetect opencv_features2d opencv_video
                                 opencv_highgui opencv_ml opencv_calib3d opencv_photo
                                 opencv_nonfree opencv_calib)

# Add all of the headers we wish to parse
string(REPLACE "opencv_" "" OPENCV_MATLAB_MODULES "${OPENCV_MODULE_${the_module}_REQ_DEPS};
                                                   ${OPENCV_MODULE_${the_module}_OPT_DEPS}")
foreach(module ${OPENCV_MATLAB_MODULES})
    if (HAVE_opencv_${module})
        list(APPEND opencv_hdrs "${OPENCV_MODULE_opencv_${module}_LOCATION}/include/opencv2/${module}/${module}.hpp")
    endif()
endforeach()

# add the python generator to the python path
set(PYPATH_CACHE $ENV{PYTHONPATH})
set(ENV{PYTHONPATH} ${OPENCV_MODULE_opencv_python_LOCATION}/src2:$ENV{PYTHONPATH})

# synthesise the matlab sources
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
