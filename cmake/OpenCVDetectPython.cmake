find_host_package(PythonInterp)
find_host_package(PythonLibs)

# cmake 2.4 (at least on Ubuntu 8.04 (hardy)) don't define PYTHONLIBS_FOUND
if(NOT PYTHONLIBS_FOUND AND PYTHON_INCLUDE_PATH)
  set(PYTHONLIBS_FOUND ON)
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} --version
  ERROR_VARIABLE PYTHON_VERSION_FULL
  OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REGEX MATCH "[0-9]+.[0-9]+" PYTHON_VERSION_MAJOR_MINOR "${PYTHON_VERSION_FULL}")
if(CMAKE_HOST_UNIX)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import *; print get_python_lib()"
                  RESULT_VARIABLE PYTHON_CVPY_PROCESS
                  OUTPUT_VARIABLE PYTHON_STD_PACKAGES_PATH
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if("${PYTHON_STD_PACKAGES_PATH}" MATCHES "site-packages")
    set(PYTHON_PACKAGES_PATH lib/python${PYTHON_VERSION_MAJOR_MINOR}/site-packages CACHE PATH "Where to install the python packages.")
  else() #debian based assumed, install to the dist-packages.
    set(PYTHON_PACKAGES_PATH lib/python${PYTHON_VERSION_MAJOR_MINOR}/dist-packages CACHE PATH "Where to install the python packages.")
  endif()
endif()

if(CMAKE_HOST_WIN32)
  get_filename_component(PYTHON_PATH "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${PYTHON_VERSION_MAJOR_MINOR}\\InstallPath]" ABSOLUTE CACHE)
  set(PYTHON_PACKAGES_PATH "${PYTHON_PATH}/Lib/site-packages")
endif()

# Attempt to discover the NumPy include directory. If this succeeds, then build python API with NumPy
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import os; os.environ['DISTUTILS_USE_SDK']='1'; import numpy.distutils; print numpy.distutils.misc_util.get_numpy_include_dirs()[0]"
                RESULT_VARIABLE PYTHON_NUMPY_PROCESS
                OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIRS
                OUTPUT_STRIP_TRAILING_WHITESPACE)

if(PYTHON_NUMPY_PROCESS EQUAL 0)
  set(PYTHON_USE_NUMPY 1)
  add_definitions(-DPYTHON_USE_NUMPY=1)
  include_directories(AFTER ${PYTHON_NUMPY_INCLUDE_DIRS})
  message(STATUS "    Use NumPy headers from: ${PYTHON_NUMPY_INCLUDE_DIRS}")
else()
  set(PYTHON_USE_NUMPY 0)
endif()

# look for Sphinx
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sphinx; print sphinx.__version__"
                RESULT_VARIABLE SPHINX_PROCESS
                OUTPUT_VARIABLE SPHINX_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set(HAVE_SPHINX 0)
if(SPHINX_PROCESS EQUAL 0)
  find_host_program(SPHINX_BUILD sphinx-build)
  if(SPHINX_BUILD)
    set(HAVE_SPHINX 1)
    message(STATUS "    Found Sphinx ${SPHINX_VERSION}: ${SPHINX_BUILD}")
  endif()
endif()
