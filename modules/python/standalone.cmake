if(NOT DEFINED OpenCV_BINARY_DIR)
  message(FATAL_ERROR "Define OpenCV_BINARY_DIR")
endif()
include("${OpenCV_BINARY_DIR}/opencv_python_config.cmake")
if(NOT DEFINED OpenCV_SOURCE_DIR)
  message(FATAL_ERROR "Missing OpenCV_SOURCE_DIR")
endif()
if(DEFINED OPENCV_PYTHON_STANDALONE_INSTALL_PATH)
  set(OPENCV_Python_INSTALL_PATH "${OPENCV_PYTHON_STANDALONE_INSTALL_PATH}")
elseif(NOT OPENCV_Python_INSTALL_PATH)
  message(FATAL_ERROR "Missing OPENCV_PYTHON_STANDALONE_INSTALL_PATH / OPENCV_Python_INSTALL_PATH")
endif()

include("${OpenCV_SOURCE_DIR}/cmake/OpenCVUtils.cmake")

set(OPENCV_PYTHON_SKIP_DETECTION ON)
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectPython.cmake")
find_package(Python COMPONENTS Interpreter Development NumPy)
if(NOT Python_EXECUTABLE OR NOT Python_INCLUDE_DIR)
  message(FATAL_ERROR "Can't find Python development files")
endif()
if(NOT Python_NUMPY_INCLUDE_DIRS)
  message(FATAL_ERROR "Can't find Python 'numpy' development files")
endif()

include("${OpenCV_SOURCE_DIR}/cmake/OpenCVInstallLayout.cmake")
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectDLPack.cmake")

status("-----------------------------------------------------------------")
status("  Python:")
status("    Interpreter:"   "${Python_EXECUTABLE} (ver ${Python3_VERSION})")
status("    Libraries:"     "${Python_LIBRARIES} (ver ${Python3_VERSION})")
status("    numpy:"         "${Python3_NumPy_INCLUDE_DIRS} (ver ${Python3_NumPy_VERSION})")
status("")
status("  Install to:" "${CMAKE_INSTALL_PREFIX}")
status("-----------------------------------------------------------------")

set(OpenCV_DIR "${OpenCV_BINARY_DIR}")
find_package(OpenCV REQUIRED)

set(PYTHON Python)

macro(ocv_add_module module_name)
  set(the_module opencv_${module_name})
  project(${the_module} CXX)
endmacro()

macro(ocv_module_include_directories module)
  include_directories(${ARGN})
endmacro()

set(MODULE_NAME python)
set(MODULE_INSTALL_SUBDIR "")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(deps ${OpenCV_LIBRARIES})
include("${CMAKE_CURRENT_LIST_DIR}/common.cmake")  # generate python target

# done, cleanup
unset(OPENCV_BUILD_INFO_STR CACHE)  # remove from cache
