if(WIN32 AND NOT PYTHON_EXECUTABLE)
  # search for executable with the same bitness as resulting binaries
  # standard FindPythonInterp always prefers executable from system path
  # this is really important because we are using the interpreter for numpy search and for choosing the install location
  foreach(_CURRENT_VERSION ${Python_ADDITIONAL_VERSIONS} 2.7 "${MIN_VER_PYTHON}")
    find_host_program(PYTHON_EXECUTABLE
      NAMES python${_CURRENT_VERSION} python
      PATHS
        [HKEY_LOCAL_MACHINE\\\\SOFTWARE\\\\Python\\\\PythonCore\\\\${_CURRENT_VERSION}\\\\InstallPath]
        [HKEY_CURRENT_USER\\\\SOFTWARE\\\\Python\\\\PythonCore\\\\${_CURRENT_VERSION}\\\\InstallPath]
      NO_SYSTEM_ENVIRONMENT_PATH
    )
  endforeach()
endif()
find_host_package(PythonInterp 2.7)
if(NOT PYTHONINTERP_FOUND)
find_host_package(PythonInterp "${MIN_VER_PYTHON}")
endif()

unset(HAVE_SPHINX CACHE)

if(PYTHONINTERP_FOUND)
  set(PYTHON_VERSION_MAJOR_MINOR "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

  if(NOT ANDROID AND NOT IOS)
    ocv_check_environment_variables(PYTHON_LIBRARY PYTHON_INCLUDE_DIR)
    # not using PYTHON_VERSION_STRING here, because it might not conform to the CMake version format
    find_host_package(PythonLibs "${PYTHON_VERSION_MAJOR_MINOR}.${PYTHON_VERSION_PATCH}" EXACT)
  endif()

  if(NOT ANDROID AND NOT IOS)
    if(CMAKE_HOST_UNIX)
      execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import *; print(get_python_lib())"
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

    if(NOT PYTHON_NUMPY_INCLUDE_DIRS)
      # Attempt to discover the NumPy include directory. If this succeeds, then build python API with NumPy
      execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
                        "import os; os.environ['DISTUTILS_USE_SDK']='1'; import numpy.distutils; print(os.pathsep.join(numpy.distutils.misc_util.get_numpy_include_dirs()))"
                      RESULT_VARIABLE PYTHON_NUMPY_PROCESS
                      OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIRS
                      OUTPUT_STRIP_TRAILING_WHITESPACE)

      if(PYTHON_NUMPY_PROCESS EQUAL 0)
        file(TO_CMAKE_PATH "${PYTHON_NUMPY_INCLUDE_DIRS}" _PYTHON_NUMPY_INCLUDE_DIRS)
        set(PYTHON_NUMPY_INCLUDE_DIRS "${_PYTHON_NUMPY_INCLUDE_DIRS}" CACHE PATH "Path to numpy headers")
      endif()
    endif()

    if(PYTHON_NUMPY_INCLUDE_DIRS)
      execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import numpy; print(numpy.version.version)"
                      OUTPUT_VARIABLE PYTHON_NUMPY_VERSION
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
  endif(NOT ANDROID AND NOT IOS)
endif()

if(BUILD_DOCS)
  find_host_program(SPHINX_BUILD sphinx-build)
  find_host_program(PLANTUML plantuml)
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
