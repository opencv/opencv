if(MSVC AND NOT PYTHON_EXECUTABLE)
  # search for executable with the same bitness as resulting binaries
  # standard FindPythonInterp always prefers executable from system path
  foreach(_CURRENT_VERSION ${Python_ADDITIONAL_VERSIONS} 2.7 2.6 2.5 2.4 2.3 2.2 2.1 2.0 1.6 1.5)
    find_host_program(PYTHON_EXECUTABLE
      NAMES python${_CURRENT_VERSION} python
      PATHS [HKEY_LOCAL_MACHINE\\\\SOFTWARE\\\\Python\\\\PythonCore\\\\${_CURRENT_VERSION}\\\\InstallPath]
      NO_SYSTEM_ENVIRONMENT_PATH
    )
  endforeach()
endif()
find_host_package(PythonInterp)

unset(PYTHON_USE_NUMPY CACHE)
unset(HAVE_SPHINX CACHE)

if(PYTHON_EXECUTABLE)
  if(NOT ANDROID AND NOT IOS)
    find_host_package(PythonLibs)
    # cmake 2.4 (at least on Ubuntu 8.04 (hardy)) don't define PYTHONLIBS_FOUND
    if(NOT PYTHONLIBS_FOUND AND PYTHON_INCLUDE_PATH)
      set(PYTHONLIBS_FOUND ON)
    endif()
  endif()

  execute_process(COMMAND ${PYTHON_EXECUTABLE} --version
    ERROR_VARIABLE PYTHON_VERSION_FULL
    ERROR_STRIP_TRAILING_WHITESPACE)

  string(REGEX MATCH "[0-9]+.[0-9]+" PYTHON_VERSION_MAJOR_MINOR "${PYTHON_VERSION_FULL}")
  string(REGEX MATCH "[0-9]+.[0-9]+.[0-9]+" PYTHON_VERSION_FULL "${PYTHON_VERSION_FULL}")

  if(NOT ANDROID AND NOT IOS)
    if(CMAKE_HOST_UNIX)
      execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import *; print get_python_lib()"
                      RESULT_VARIABLE PYTHON_CVPY_PROCESS
                      OUTPUT_VARIABLE PYTHON_STD_PACKAGES_PATH
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      if("${PYTHON_STD_PACKAGES_PATH}" MATCHES "site-packages")
        set(PYTHON_PACKAGES_PATH "python${PYTHON_VERSION_MAJOR_MINOR}/site-packages")
      else() #debian based assumed, install to the dist-packages.
        set(PYTHON_PACKAGES_PATH "python${PYTHON_VERSION_MAJOR_MINOR}/dist-packages")
      endif()
      if(EXISTS "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}/${PYTHON_PACKAGES_PATH}")
        set(PYTHON_PACKAGES_PATH "lib${LIB_SUFFIX}/${PYTHON_PACKAGES_PATH}")
      else()
        set(PYTHON_PACKAGES_PATH "lib/${PYTHON_PACKAGES_PATH}")
      endif()
      set(PYTHON_PACKAGES_PATH "${PYTHON_PACKAGES_PATH}" CACHE PATH "Where to install the python packages.")
    elseif(CMAKE_HOST_WIN32)
      get_filename_component(PYTHON_PATH "${PYTHON_EXECUTABLE}" PATH)
      file(TO_CMAKE_PATH "${PYTHON_PATH}" PYTHON_PATH)
      if(NOT EXISTS "${PYTHON_PATH}/Lib/site-packages")
        unset(PYTHON_PATH)
        get_filename_component(PYTHON_PATH "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${PYTHON_VERSION_MAJOR_MINOR}\\InstallPath]" ABSOLUTE)
        file(TO_CMAKE_PATH "${PYTHON_PATH}" PYTHON_PATH)
      endif()
      set(PYTHON_PACKAGES_PATH "${PYTHON_PATH}/Lib/site-packages")
    endif()

    if(NOT PYTHON_NUMPY_INCLUDE_DIR)
      # Attempt to discover the NumPy include directory. If this succeeds, then build python API with NumPy
      execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import os; os.environ['DISTUTILS_USE_SDK']='1'; import numpy.distutils; print numpy.distutils.misc_util.get_numpy_include_dirs()[0]"
                      RESULT_VARIABLE PYTHON_NUMPY_PROCESS
                      OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
                      
      if(PYTHON_NUMPY_PROCESS EQUAL 0)
        file(TO_CMAKE_PATH "${PYTHON_NUMPY_INCLUDE_DIR}" PYTHON_NUMPY_INCLUDE_DIR)
        set(PYTHON_NUMPY_INCLUDE_DIR ${PYTHON_NUMPY_INCLUDE_DIR} CACHE PATH "Path to numpy headers")
      endif()
    endif()
    
    if(PYTHON_NUMPY_INCLUDE_DIR)
      set(PYTHON_USE_NUMPY TRUE)
      execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print numpy.version.version"
                        RESULT_VARIABLE PYTHON_NUMPY_PROCESS
                        OUTPUT_VARIABLE PYTHON_NUMPY_VERSION
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
  endif(NOT ANDROID AND NOT IOS)

  if(BUILD_DOCS)
    # look for Sphinx
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sphinx; print sphinx.__version__"
                    RESULT_VARIABLE SPHINX_PROCESS
                    OUTPUT_VARIABLE SPHINX_VERSION
                    OUTPUT_STRIP_TRAILING_WHITESPACE)


    if(SPHINX_PROCESS EQUAL 0)
      find_host_program(SPHINX_BUILD sphinx-build)
      if(SPHINX_BUILD)
        set(HAVE_SPHINX 1)
        message(STATUS "  Found Sphinx ${SPHINX_VERSION}: ${SPHINX_BUILD}")
      endif()
    endif()
  endif(BUILD_DOCS)
endif(PYTHON_EXECUTABLE)
