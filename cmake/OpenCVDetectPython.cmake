if(WIN32 AND NOT PYTHON_EXECUTABLE)
  # search for executable with the same bitness as resulting binaries
  # standard FindPythonInterp always prefers executable from system path
  # this is really important because we are using the interpreter for numpy search and for choosing the install location
  foreach(_CURRENT_VERSION ${Python_ADDITIONAL_VERSIONS} 2.7 2.6 2.5 2.4 2.3 2.2 2.1 2.0)
    find_host_program(PYTHON_EXECUTABLE
      NAMES python${_CURRENT_VERSION} python
      PATHS
        [HKEY_LOCAL_MACHINE\\\\SOFTWARE\\\\Python\\\\PythonCore\\\\${_CURRENT_VERSION}\\\\InstallPath]
        [HKEY_CURRENT_USER\\\\SOFTWARE\\\\Python\\\\PythonCore\\\\${_CURRENT_VERSION}\\\\InstallPath]
      NO_SYSTEM_ENVIRONMENT_PATH
    )
  endforeach()
endif()
find_host_package(PythonInterp 2.0)

unset(PYTHON_USE_NUMPY CACHE)
unset(HAVE_SPHINX CACHE)
if(PYTHON_EXECUTABLE)
  if(PYTHON_VERSION_STRING)
    set(PYTHON_VERSION_MAJOR_MINOR "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
    set(PYTHON_VERSION_FULL "${PYTHON_VERSION_STRING}")
  else()
    execute_process(COMMAND ${PYTHON_EXECUTABLE} --version
      ERROR_VARIABLE PYTHON_VERSION_FULL
      ERROR_STRIP_TRAILING_WHITESPACE)

    string(REGEX MATCH "[0-9]+.[0-9]+" PYTHON_VERSION_MAJOR_MINOR "${PYTHON_VERSION_FULL}")
  endif()

  if("${PYTHON_VERSION_FULL}" MATCHES "[0-9]+.[0-9]+.[0-9]+")
    set(PYTHON_VERSION_FULL "${CMAKE_MATCH_0}")
  elseif("${PYTHON_VERSION_FULL}" MATCHES "[0-9]+.[0-9]+")
    set(PYTHON_VERSION_FULL "${CMAKE_MATCH_0}")
  else()
    unset(PYTHON_VERSION_FULL)
  endif()

  if(NOT ANDROID AND NOT IOS)
    ocv_check_environment_variables(PYTHON_LIBRARY PYTHON_INCLUDE_DIR)
    if(CMAKE_VERSION VERSION_GREATER 2.8.8 AND PYTHON_VERSION_FULL)
      find_host_package(PythonLibs ${PYTHON_VERSION_FULL} EXACT)
    else()
      find_host_package(PythonLibs ${PYTHON_VERSION_FULL})
    endif()
    # cmake 2.4 (at least on Ubuntu 8.04 (hardy)) don't define PYTHONLIBS_FOUND
    if(NOT PYTHONLIBS_FOUND AND PYTHON_INCLUDE_PATH)
      set(PYTHONLIBS_FOUND ON)
    endif()
  endif()

  if(NOT ANDROID AND NOT IOS)
    if(CMAKE_HOST_UNIX)
      execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import *; print get_python_lib()"
                      RESULT_VARIABLE PYTHON_CVPY_PROCESS
                      OUTPUT_VARIABLE PYTHON_STD_PACKAGES_PATH
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      if("${PYTHON_STD_PACKAGES_PATH}" MATCHES "site-packages")
        set(_PYTHON_PACKAGES_PATH "python${PYTHON_VERSION_MAJOR_MINOR}/site-packages")
      else() #debian based assumed, install to the dist-packages.
        set(_PYTHON_PACKAGES_PATH "python${PYTHON_VERSION_MAJOR_MINOR}/dist-packages")
      endif()
      if(EXISTS "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}/${PYTHON_PACKAGES_PATH}")
        set(_PYTHON_PACKAGES_PATH "lib${LIB_SUFFIX}/${_PYTHON_PACKAGES_PATH}")
      else()
        set(_PYTHON_PACKAGES_PATH "lib/${_PYTHON_PACKAGES_PATH}")
      endif()
    elseif(CMAKE_HOST_WIN32)
      get_filename_component(PYTHON_PATH "${PYTHON_EXECUTABLE}" PATH)
      file(TO_CMAKE_PATH "${PYTHON_PATH}" PYTHON_PATH)
      if(NOT EXISTS "${PYTHON_PATH}/Lib/site-packages")
        unset(PYTHON_PATH)
        get_filename_component(PYTHON_PATH "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${PYTHON_VERSION_MAJOR_MINOR}\\InstallPath]" ABSOLUTE)
        if(NOT PYTHON_PATH)
           get_filename_component(PYTHON_PATH "[HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${PYTHON_VERSION_MAJOR_MINOR}\\InstallPath]" ABSOLUTE)
        endif()
        file(TO_CMAKE_PATH "${PYTHON_PATH}" PYTHON_PATH)
      endif()
      set(_PYTHON_PACKAGES_PATH "${PYTHON_PATH}/Lib/site-packages")
    endif()
    SET(PYTHON_PACKAGES_PATH "${_PYTHON_PACKAGES_PATH}" CACHE PATH "Where to install the python packages.")

    if(NOT PYTHON_NUMPY_INCLUDE_DIR)
      # Attempt to discover the NumPy include directory. If this succeeds, then build python API with NumPy
      execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import os; os.environ['DISTUTILS_USE_SDK']='1'; import numpy.distutils; print numpy.distutils.misc_util.get_numpy_include_dirs()[0]"
                      RESULT_VARIABLE PYTHON_NUMPY_PROCESS
                      OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR
                      OUTPUT_STRIP_TRAILING_WHITESPACE)

      if(PYTHON_NUMPY_PROCESS EQUAL 0)
        file(TO_CMAKE_PATH "${PYTHON_NUMPY_INCLUDE_DIR}" _PYTHON_NUMPY_INCLUDE_DIR)
        set(PYTHON_NUMPY_INCLUDE_DIR ${_PYTHON_NUMPY_INCLUDE_DIR} CACHE PATH "Path to numpy headers")
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
    find_host_program(SPHINX_BUILD sphinx-build)
    if(SPHINX_BUILD)
        execute_process(COMMAND "${SPHINX_BUILD}"
                        OUTPUT_QUIET
                        ERROR_VARIABLE SPHINX_OUTPUT
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(SPHINX_OUTPUT MATCHES "Sphinx v([0-9][^ \n]*)")
          set(SPHINX_VERSION "${CMAKE_MATCH_1}")
          set(HAVE_SPHINX 1)
          message(STATUS "Found Sphinx ${SPHINX_VERSION}: ${SPHINX_BUILD}")
        endif()
    endif()
  endif(BUILD_DOCS)
endif(PYTHON_EXECUTABLE)
