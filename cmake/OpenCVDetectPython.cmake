# Find specified Python version
# Arguments:
#   preferred_version (value): Version to check for first
#   min_version (value): Minimum supported version
#   library_env (value): Name of Python library ENV variable to check
#   include_dir_env (value): Name of Python include directory ENV variable to check
#   found (variable): Set if interpreter found
#   executable (variable): Output of executable found
#   version_string (variable): Output of found version
#   version_major (variable): Output of found major version
#   version_minor (variable): Output of found minor version
#   libs_found (variable): Set if libs found
#   libs_version_string (variable): Output of found libs version
#   libraries (variable): Output of found Python libraries
#   library (variable): Output of found Python library
#   debug_libraries (variable): Output of found Python debug libraries
#   debug_library (variable): Output of found Python debug library
#   include_path (variable): Output of found Python include path
#   include_dir (variable): Output of found Python include dir
#   include_dir2 (variable): Output of found Python include dir2
#   packages_path (variable): Output of found Python packages path
#   numpy_include_dirs (variable): Output of found Python Numpy include dirs
#   numpy_version (variable): Output of found Python Numpy version
function(find_python preferred_version min_version library_env include_dir_env
         found executable version_string version_major version_minor
         libs_found libs_version_string libraries library debug_libraries
         debug_library include_path include_dir include_dir2 packages_path
         numpy_include_dirs numpy_version)

  ocv_check_environment_variables(${executable})
  if(${executable})
    set(PYTHON_EXECUTABLE "${${executable}}")
  endif()

  if(WIN32 AND NOT ${executable})
    # search for executable with the same bitness as resulting binaries
    # standard FindPythonInterp always prefers executable from system path
    # this is really important because we are using the interpreter for numpy search and for choosing the install location
    foreach(_CURRENT_VERSION ${Python_ADDITIONAL_VERSIONS} "${preferred_version}" "${min_version}")
      find_host_program(PYTHON_EXECUTABLE
        NAMES python${_CURRENT_VERSION} python
        PATHS
          [HKEY_LOCAL_MACHINE\\\\SOFTWARE\\\\Python\\\\PythonCore\\\\${_CURRENT_VERSION}\\\\InstallPath]
          [HKEY_CURRENT_USER\\\\SOFTWARE\\\\Python\\\\PythonCore\\\\${_CURRENT_VERSION}\\\\InstallPath]
        NO_SYSTEM_ENVIRONMENT_PATH
      )
    endforeach()
  endif()

  find_host_package(PythonInterp "${preferred_version}")
  if(NOT PYTHONINTERP_FOUND)
    find_host_package(PythonInterp "${min_version}")
  endif()

  if(PYTHONINTERP_FOUND)
    # Copy outputs
    set(_found ${PYTHONINTERP_FOUND})
    set(_executable ${PYTHON_EXECUTABLE})
    set(_version_string ${PYTHON_VERSION_STRING})
    set(_version_major ${PYTHON_VERSION_MAJOR})
    set(_version_minor ${PYTHON_VERSION_MINOR})
    set(_version_patch ${PYTHON_VERSION_PATCH})

    # Clear find_host_package side effects
    unset(PYTHONINTERP_FOUND)
    unset(PYTHON_EXECUTABLE CACHE)
    unset(PYTHON_VERSION_STRING)
    unset(PYTHON_VERSION_MAJOR)
    unset(PYTHON_VERSION_MINOR)
    unset(PYTHON_VERSION_PATCH)
  endif()

  if(_found)
    set(_version_major_minor "${_version_major}.${_version_minor}")

    if(NOT ANDROID AND NOT IOS)
      ocv_check_environment_variables(${library_env} ${include_dir_env})
      if(NOT ${${library_env}} EQUAL "")
          set(PYTHON_LIBRARY "${${library_env}}")
      endif()
      if(NOT ${${include_dir_env}} EQUAL "")
          set(PYTHON_INCLUDE_DIR "${${include_dir_env}}")
      endif()

      # not using _version_string here, because it might not conform to the CMake version format
      if(CMAKE_CROSSCOMPILING)
        # builder version can differ from target, matching base version (e.g. 2.7)
        find_host_package(PythonLibs "${_version_major_minor}")
      else()
        find_host_package(PythonLibs "${_version_major_minor}.${_version_patch}" EXACT)
      endif()

      if(PYTHONLIBS_FOUND)
        # Copy outputs
        set(_libs_found ${PYTHONLIBS_FOUND})
        set(_libraries ${PYTHON_LIBRARIES})
        set(_include_path ${PYTHON_INCLUDE_PATH})
        set(_include_dirs ${PYTHON_INCLUDE_DIRS})
        set(_debug_libraries ${PYTHON_DEBUG_LIBRARIES})
        set(_libs_version_string ${PYTHONLIBS_VERSION_STRING})
        set(_debug_library ${PYTHON_DEBUG_LIBRARY})
        set(_library ${PYTHON_LIBRARY})
        set(_library_debug ${PYTHON_LIBRARY_DEBUG})
        set(_library_release ${PYTHON_LIBRARY_RELEASE})
        set(_include_dir ${PYTHON_INCLUDE_DIR})
        set(_include_dir2 ${PYTHON_INCLUDE_DIR2})

        # Clear find_host_package side effects
        unset(PYTHONLIBS_FOUND)
        unset(PYTHON_LIBRARIES)
        unset(PYTHON_INCLUDE_PATH)
        unset(PYTHON_INCLUDE_DIRS)
        unset(PYTHON_DEBUG_LIBRARIES)
        unset(PYTHONLIBS_VERSION_STRING)
        unset(PYTHON_DEBUG_LIBRARY CACHE)
        unset(PYTHON_LIBRARY)
        unset(PYTHON_LIBRARY_DEBUG)
        unset(PYTHON_LIBRARY_RELEASE)
        unset(PYTHON_LIBRARY CACHE)
        unset(PYTHON_LIBRARY_DEBUG CACHE)
        unset(PYTHON_LIBRARY_RELEASE CACHE)
        unset(PYTHON_INCLUDE_DIR CACHE)
        unset(PYTHON_INCLUDE_DIR2 CACHE)
      endif()
    endif()

    if(NOT ANDROID AND NOT IOS)
      if(CMAKE_HOST_UNIX)
        execute_process(COMMAND ${_executable} -c "from distutils.sysconfig import *; print(get_python_lib())"
                        RESULT_VARIABLE _cvpy_process
                        OUTPUT_VARIABLE _std_packages_path
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        if("${_std_packages_path}" MATCHES "site-packages")
          set(_packages_path "python${_version_major_minor}/site-packages")
        else() #debian based assumed, install to the dist-packages.
          set(_packages_path "python${_version_major_minor}/dist-packages")
        endif()
        if(EXISTS "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}/${${packages_path}}")
          set(_packages_path "lib${LIB_SUFFIX}/${_packages_path}")
        else()
          set(_packages_path "lib/${_packages_path}")
        endif()
      elseif(CMAKE_HOST_WIN32)
        get_filename_component(_path "${_executable}" PATH)
        file(TO_CMAKE_PATH "${_path}" _path)
        if(NOT EXISTS "${_path}/Lib/site-packages")
          unset(_path)
          get_filename_component(_path "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_version_major_minor}\\InstallPath]" ABSOLUTE)
          if(NOT _path)
             get_filename_component(_path "[HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_version_major_minor}\\InstallPath]" ABSOLUTE)
          endif()
          file(TO_CMAKE_PATH "${_path}" _path)
        endif()
        set(_packages_path "${_path}/Lib/site-packages")
        unset(_path)
      endif()

      set(_numpy_include_dirs ${${numpy_include_dirs}})

      if(NOT _numpy_include_dirs)
        if(CMAKE_CROSSCOMPILING)
          message(STATUS "Cannot probe for Python/Numpy support (because we are cross-compiling OpenCV)")
          message(STATUS "If you want to enable Python/Numpy support, set the following variables:")
          message(STATUS "  PYTHON2_INCLUDE_PATH")
          message(STATUS "  PYTHON2_LIBRARIES")
          message(STATUS "  PYTHON2_NUMPY_INCLUDE_DIRS")
          message(STATUS "  PYTHON3_INCLUDE_PATH")
          message(STATUS "  PYTHON3_LIBRARIES")
          message(STATUS "  PYTHON3_NUMPY_INCLUDE_DIRS")
        else()
          # Attempt to discover the NumPy include directory. If this succeeds, then build python API with NumPy
          execute_process(COMMAND "${_executable}" -c "import os; os.environ['DISTUTILS_USE_SDK']='1'; import numpy.distutils; print(os.pathsep.join(numpy.distutils.misc_util.get_numpy_include_dirs()))"
                          RESULT_VARIABLE _numpy_process
                          OUTPUT_VARIABLE _numpy_include_dirs
                          OUTPUT_STRIP_TRAILING_WHITESPACE)

          if(NOT _numpy_process EQUAL 0)
              unset(_numpy_include_dirs)
          endif()
        endif()
      endif()

      if(_numpy_include_dirs)
        file(TO_CMAKE_PATH "${_numpy_include_dirs}" _numpy_include_dirs)
        if(CMAKE_CROSSCOMPILING)
          if(NOT _numpy_version)
            set(_numpy_version "undefined - cannot be probed because of the cross-compilation")
          endif()
        else()
          execute_process(COMMAND "${_executable}" -c "import numpy; print(numpy.version.version)"
                          RESULT_VARIABLE _numpy_process
                          OUTPUT_VARIABLE _numpy_version
                          OUTPUT_STRIP_TRAILING_WHITESPACE)
        endif()
      endif()
    endif(NOT ANDROID AND NOT IOS)
  endif()

  # Export return values
  set(${found} "${_found}" PARENT_SCOPE)
  set(${executable} "${_executable}" CACHE FILEPATH "Path to Python interpretor")
  set(${version_string} "${_version_string}" PARENT_SCOPE)
  set(${version_major} "${_version_major}" PARENT_SCOPE)
  set(${version_minor} "${_version_minor}" PARENT_SCOPE)
  set(${libs_found} "${_libs_found}" PARENT_SCOPE)
  set(${libs_version_string} "${_libs_version_string}" PARENT_SCOPE)
  set(${libraries} "${_libraries}" PARENT_SCOPE)
  set(${library} "${_library}" CACHE FILEPATH "Path to Python library")
  set(${debug_libraries} "${_debug_libraries}" PARENT_SCOPE)
  set(${debug_library} "${_debug_library}" CACHE FILEPATH "Path to Python debug")
  set(${include_path} "${_include_path}" PARENT_SCOPE)
  set(${include_dir} "${_include_dir}" CACHE PATH "Python include dir")
  set(${include_dir2} "${_include_dir2}" CACHE PATH "Python include dir 2")
  set(${packages_path} "${_packages_path}" CACHE PATH "Where to install the python packages.")
  set(${numpy_include_dirs} ${_numpy_include_dirs} CACHE PATH "Path to numpy headers")
  set(${numpy_version} "${_numpy_version}" PARENT_SCOPE)
endfunction(find_python)

find_python(2.7 "${MIN_VER_PYTHON2}" PYTHON2_LIBRARY PYTHON2_INCLUDE_DIR
    PYTHON2INTERP_FOUND PYTHON2_EXECUTABLE PYTHON2_VERSION_STRING
    PYTHON2_VERSION_MAJOR PYTHON2_VERSION_MINOR PYTHON2LIBS_FOUND
    PYTHON2LIBS_VERSION_STRING PYTHON2_LIBRARIES PYTHON2_LIBRARY
    PYTHON2_DEBUG_LIBRARIES PYTHON2_LIBRARY_DEBUG PYTHON2_INCLUDE_PATH
    PYTHON2_INCLUDE_DIR PYTHON2_INCLUDE_DIR2 PYTHON2_PACKAGES_PATH
    PYTHON2_NUMPY_INCLUDE_DIRS PYTHON2_NUMPY_VERSION)

find_python(3.4 "${MIN_VER_PYTHON3}" PYTHON3_LIBRARY PYTHON3_INCLUDE_DIR
    PYTHON3INTERP_FOUND PYTHON3_EXECUTABLE PYTHON3_VERSION_STRING
    PYTHON3_VERSION_MAJOR PYTHON3_VERSION_MINOR PYTHON3LIBS_FOUND
    PYTHON3LIBS_VERSION_STRING PYTHON3_LIBRARIES PYTHON3_LIBRARY
    PYTHON3_DEBUG_LIBRARIES PYTHON3_LIBRARY_DEBUG PYTHON3_INCLUDE_PATH
    PYTHON3_INCLUDE_DIR PYTHON3_INCLUDE_DIR2 PYTHON3_PACKAGES_PATH
    PYTHON3_NUMPY_INCLUDE_DIRS PYTHON3_NUMPY_VERSION)

# Use Python 2 as default Python interpreter
if(PYTHON2INTERP_FOUND)
    set(PYTHON_DEFAULT_AVAILABLE "TRUE")
    set(PYTHON_DEFAULT_EXECUTABLE "${PYTHON2_EXECUTABLE}")
endif()
