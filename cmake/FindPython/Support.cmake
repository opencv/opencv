# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#
# This file is a "template" file used by various FindPython modules.
#

#
# Initial configuration
#
if (NOT DEFINED _PYTHON_PREFIX)
  message (FATAL_ERROR "FindPython: INTERNAL ERROR")
endif()
if (NOT DEFINED _${_PYTHON_PREFIX}_REQUIRED_VERSION_MAJOR)
  message (FATAL_ERROR "FindPython: INTERNAL ERROR")
endif()
if (_${_PYTHON_PREFIX}_REQUIRED_VERSION_MAJOR EQUAL 3)
  set(_${_PYTHON_PREFIX}_VERSIONS 3.8 3.7 3.6 3.5 3.4 3.3 3.2 3.1 3.0)
elseif (_${_PYTHON_PREFIX}_REQUIRED_VERSION_MAJOR EQUAL 2)
  set(_${_PYTHON_PREFIX}_VERSIONS 2.7 2.6 2.5 2.4 2.3 2.2 2.1 2.0)
else()
  message (FATAL_ERROR "FindPython: INTERNAL ERROR")
endif()


#
# helper commands
#
macro (_PYTHON_DISPLAY_FAILURE _PYTHON_MSG)
  if (${_PYTHON_PREFIX}_FIND_REQUIRED)
    message (FATAL_ERROR "${_PYTHON_MSG}")
  else()
    if (NOT ${_PYTHON_PREFIX}_FIND_QUIETLY)
      message(STATUS "${_PYTHON_MSG}")
    endif ()
  endif()

  set (${_PYTHON_PREFIX}_FOUND FALSE)
  string (TOUPPER "${_PYTHON_PREFIX}" _${_PYTHON_PREFIX}_UPPER_PREFIX)
  set (${_PYTHON_UPPER_PREFIX}_FOUND FALSE)
  return()
endmacro()


function (_PYTHON_GET_FRAMEWORKS _PYTHON_PGF_FRAMEWORK_PATHS _PYTHON_VERSION)
  set (_PYTHON_FRAMEWORK_PATHS)
  foreach (_PYTHON_FRAMEWORK IN LISTS Python_FRAMEWORKS)
    list (APPEND _PYTHON_FRAMEWORK_PATHS
          "${_PYTHON_FRAMEWORK}/Versions/${_PYTHON_VERSION}")
  endforeach()
  set (${_PYTHON_PGF_FRAMEWORK_PATHS} ${_PYTHON_FRAMEWORK_PATHS} PARENT_SCOPE)
endfunction()


function (_PYTHON_VALIDATE_INTERPRETER)
  if (NOT ${_PYTHON_PREFIX}_EXECUTABLE)
    return()
  endif()

  if (${_PYTHON_PREFIX}_EXECUTABLE MATCHES "python${CMAKE_EXECUTABLE_SUFFIX}$")
    # executable found do not have version in name
    # ensure major version is OK
    execute_process (COMMAND "${${_PYTHON_PREFIX}_EXECUTABLE}" -c
                             "import sys; sys.stdout.write(str(sys.version_info[0]))"
                     RESULT_VARIABLE result
                     OUTPUT_VARIABLE version
                     ERROR_QUIET
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (result OR NOT version EQUAL _${_PYTHON_PREFIX}_REQUIRED_VERSION_MAJOR)
      # interpreter not usable or has wrong major version
      set (${_PYTHON_PREFIX}_EXECUTABLE ${_PYTHON_PREFIX}_EXECUTABLE-NOTFOUND CACHE INTERNAL "" FORCE)
      return()
    endif()
  endif()

  if (CMAKE_SIZEOF_VOID_P AND "Development" IN_LIST ${_PYTHON_PREFIX}_FIND_COMPONENTS
      AND NOT CMAKE_CROSSCOMPILING)
    # In this case, interpreter must have same architecture as environment
    execute_process (COMMAND "${${_PYTHON_PREFIX}_EXECUTABLE}" -c
                             "import sys, struct; sys.stdout.write(str(struct.calcsize(\"P\")))"
                     RESULT_VARIABLE result
                     OUTPUT_VARIABLE size
                     ERROR_QUIET
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (result OR NOT size EQUAL CMAKE_SIZEOF_VOID_P)
      # interpreter not usable or has wrong architecture
      set (${_PYTHON_PREFIX}_EXECUTABLE ${_PYTHON_PREFIX}_EXECUTABLE-NOTFOUND CACHE INTERNAL "" FORCE)
      return()
    endif()
  endif()
endfunction()


function (_PYTHON_FIND_RUNTIME_LIBRARY _PYTHON_LIB)
  string (REPLACE "_RUNTIME" "" _PYTHON_LIB "${_PYTHON_LIB}")
  # look at runtime part on systems supporting it
  if (CMAKE_SYSTEM_NAME STREQUAL "Windows" OR
      (CMAKE_SYSTEM_NAME MATCHES "MSYS|CYGWIN"
        AND ${_PYTHON_LIB} MATCHES "${CMAKE_IMPORT_LIBRARY_SUFFIX}$"))
    set (CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
    # MSYS has a special syntax for runtime libraries
    if (CMAKE_SYSTEM_NAME MATCHES "MSYS")
      list (APPEND CMAKE_FIND_LIBRARY_PREFIXES "msys-")
    endif()
    find_library (${ARGV})
  endif()
endfunction()


function (_PYTHON_SET_LIBRARY_DIRS _PYTHON_SLD_RESULT)
  unset (_PYTHON_DIRS)
  set (_PYTHON_LIBS ${ARGV})
  list (REMOVE_AT _PYTHON_LIBS 0)
  foreach (_PYTHON_LIB IN LISTS _PYTHON_LIBS)
    if (${_PYTHON_LIB})
      get_filename_component (_PYTHON_DIR "${${_PYTHON_LIB}}" DIRECTORY)
      list (APPEND _PYTHON_DIRS "${_PYTHON_DIR}")
    endif()
  endforeach()
  if (_PYTHON_DIRS)
    list (REMOVE_DUPLICATES _PYTHON_DIRS)
  endif()
  set (${_PYTHON_SLD_RESULT} ${_PYTHON_DIRS} PARENT_SCOPE)
endfunction()


# If major version is specified, it must be the same as internal major version
if (DEFINED ${_PYTHON_PREFIX}_FIND_VERSION_MAJOR
    AND NOT ${_PYTHON_PREFIX}_FIND_VERSION_MAJOR VERSION_EQUAL _${_PYTHON_PREFIX}_REQUIRED_VERSION_MAJOR)
  _python_display_failure ("Could NOT find ${_PYTHON_PREFIX}: Wrong major version specified is \"${${_PYTHON_PREFIX}_FIND_VERSION_MAJOR}\", but expected major version is \"${_${_PYTHON_PREFIX}_REQUIRED_VERSION_MAJOR}\"")
endif()


# handle components
if (NOT ${_PYTHON_PREFIX}_FIND_COMPONENTS)
  set (${_PYTHON_PREFIX}_FIND_COMPONENTS Interpreter)
  set (${_PYTHON_PREFIX}_FIND_REQUIRED_Interpreter TRUE)
endif()
foreach (_${_PYTHON_PREFIX}_COMPONENT IN LISTS ${_PYTHON_PREFIX}_FIND_COMPONENTS)
  set (${_PYTHON_PREFIX}_${_${_PYTHON_PREFIX}_COMPONENT}_FOUND FALSE)
endforeach()
unset (_${_PYTHON_PREFIX}_FIND_VERSIONS)

# Set versions to search
## default: search any version
set (_${_PYTHON_PREFIX}_FIND_VERSIONS ${_${_PYTHON_PREFIX}_VERSIONS})

if (${_PYTHON_PREFIX}_FIND_VERSION_COUNT GREATER 1)
  if (${_PYTHON_PREFIX}_FIND_VERSION_EXACT)
    set (_${_PYTHON_PREFIX}_FIND_VERSIONS ${${_PYTHON_PREFIX}_FIND_VERSION_MAJOR}.${${_PYTHON_PREFIX}_FIND_VERSION_MINOR})
  else()
    unset (_${_PYTHON_PREFIX}_FIND_VERSIONS)
    # add all compatible versions
    foreach (_${_PYTHON_PREFIX}_VERSION IN LISTS _${_PYTHON_PREFIX}_VERSIONS)
      if ((_${_PYTHON_PREFIX}_VERSION VERSION_GREATER ${_PYTHON_PREFIX}_FIND_VERSION) OR
          (_${_PYTHON_PREFIX}_VERSION VERSION_EQUAL ${_PYTHON_PREFIX}_FIND_VERSION))
        list (APPEND _${_PYTHON_PREFIX}_FIND_VERSIONS ${_${_PYTHON_PREFIX}_VERSION})
      endif()
    endforeach()
  endif()
endif()

# Anaconda distribution: define which architectures can be used
if (CMAKE_SIZEOF_VOID_P)
  # In this case, search only for 64bit or 32bit
  math (EXPR _${_PYTHON_PREFIX}_ARCH "${CMAKE_SIZEOF_VOID_P} * 8")
  set (_${_PYTHON_PREFIX}_ARCH2 ${_${_PYTHON_PREFIX}_ARCH})
else()
  # architecture unknown, search for both 64bit and 32bit
  set (_${_PYTHON_PREFIX}_ARCH 64)
  set (_${_PYTHON_PREFIX}_ARCH2 32)
endif()

# IronPython support
if (CMAKE_SIZEOF_VOID_P)
  # In this case, search only for 64bit or 32bit
  math (EXPR _${_PYTHON_PREFIX}_ARCH "${CMAKE_SIZEOF_VOID_P} * 8")
  set (_${_PYTHON_PREFIX}_IRON_PYTHON_NAMES ipy${_${_PYTHON_PREFIX}_ARCH} ipy)
else()
  # architecture unknown, search for natural interpreter
  set (_${_PYTHON_PREFIX}_IRON_PYTHON_NAMES ipy)
endif()

# Apple frameworks handling
include (CMakeFindFrameworks)
cmake_find_frameworks (Python)

# Save CMAKE_FIND_FRAMEWORK
if (DEFINED CMAKE_FIND_FRAMEWORK)
  set (_${_PYTHON_PREFIX}_CMAKE_FIND_FRAMEWORK ${CMAKE_FIND_FRAMEWORK})
else()
  unset (_${_PYTHON_PREFIX}_CMAKE_FIND_FRAMEWORK)
endif()
# To avoid picking up the system elements pre-maturely.
set (CMAKE_FIND_FRAMEWORK LAST)


unset (_${_PYTHON_PREFIX}_REQUIRED_VARS)
unset (_${_PYTHON_PREFIX}_CACHED_VARS)


# first step, search for the interpreter
if ("Interpreter" IN_LIST ${_PYTHON_PREFIX}_FIND_COMPONENTS)
  if (${_PYTHON_PREFIX}_FIND_REQUIRED_Interpreter)
    list (APPEND _${_PYTHON_PREFIX}_REQUIRED_VARS ${_PYTHON_PREFIX}_EXECUTABLE)
    list (APPEND _${_PYTHON_PREFIX}_CACHED_VARS ${_PYTHON_PREFIX}_EXECUTABLE)
  endif()

  set (_${_PYTHON_PREFIX}_HINTS "${${_PYTHON_PREFIX}_ROOT_DIR}" ENV ${_PYTHON_PREFIX}_ROOT_DIR)

  # look-up for various versions and locations
  foreach (_${_PYTHON_PREFIX}_VERSION IN LISTS _${_PYTHON_PREFIX}_FIND_VERSIONS)
    string (REPLACE "." "" _${_PYTHON_PREFIX}_VERSION_NO_DOTS ${_${_PYTHON_PREFIX}_VERSION})

    _python_get_frameworks (_${_PYTHON_PREFIX}_FRAMEWORK_PATHS ${_${_PYTHON_PREFIX}_VERSION})

    # try using HINTS
    find_program (${_PYTHON_PREFIX}_EXECUTABLE
                  NAMES python${_${_PYTHON_PREFIX}_VERSION}
                  NAMES_PER_DIR
                  HINTS ${_${_PYTHON_PREFIX}_HINTS}
                  PATHS ${_${_PYTHON_PREFIX}_FRAMEWORK_PATHS}
                  PATH_SUFFIXES bin
                  NO_SYSTEM_ENVIRONMENT_PATH
                  NO_CMAKE_SYSTEM_PATH)
    # try using registry
    if (WIN32)
      find_program (${_PYTHON_PREFIX}_EXECUTABLE
                    NAMES python${_${_PYTHON_PREFIX}_VERSION} python
                          ${_${_PYTHON_PREFIX}_IRON_PYTHON_NAMES}
                    NAMES_PER_DIR
                    HINTS ${_${_PYTHON_PREFIX}_HINTS}
                    PATHS [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_${_PYTHON_PREFIX}_VERSION}\\InstallPath]
                          [HKEY_CURRENT_USER\\SOFTWARE\\Python\\ContinuumAnalytics\\Anaconda${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}-${_${_PYTHON_PREFIX}_ARCH}\\InstallPath]
                          [HKEY_CURRENT_USER\\SOFTWARE\\Python\\ContinuumAnalytics\\Anaconda${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}-${_${_PYTHON_PREFIX}_ARCH2}\\InstallPath]
                          [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_${_PYTHON_PREFIX}_VERSION}\\InstallPath]
                          [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\ContinuumAnalytics\\Anaconda${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}-${_${_PYTHON_PREFIX}_ARCH}\\InstallPath]
                          [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\ContinuumAnalytics\\Anaconda${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}-${_${_PYTHON_PREFIX}_ARCH2}\\InstallPath]
                          [HKEY_LOCAL_MACHINE\\SOFTWARE\\IronPython\\${_${_PYTHON_PREFIX}_VERSION}\\InstallPath]
                    PATH_SUFFIXES bin
                    NO_SYSTEM_ENVIRONMENT_PATH
                    NO_CMAKE_SYSTEM_PATH)
    endif()
    # try in standard paths
    find_program (${_PYTHON_PREFIX}_EXECUTABLE
                  NAMES python${_${_PYTHON_PREFIX}_VERSION})

    _python_validate_interpreter ()
    if (${_PYTHON_PREFIX}_EXECUTABLE)
      break()
    endif()
  endforeach()

  # try more generic names
  if (NOT ${_PYTHON_PREFIX}_EXECUTABLE)
    find_program (${_PYTHON_PREFIX}_EXECUTABLE
                  NAMES python${_${_PYTHON_PREFIX}_REQUIRED_VERSION_MAJOR} python
                        ${_${_PYTHON_PREFIX}_IRON_PYTHON_NAMES}
                  HINTS ${_${_PYTHON_PREFIX}_HINTS}
                  PATH_SUFFIXES bin)

    _python_validate_interpreter ()
  endif()

  # retrieve exact version of executable found
  if (${_PYTHON_PREFIX}_EXECUTABLE)
    execute_process (COMMAND "${${_PYTHON_PREFIX}_EXECUTABLE}" -c
                             "import sys; sys.stdout.write('.'.join([str(x) for x in sys.version_info[:3]]))"
                     RESULT_VARIABLE _${_PYTHON_PREFIX}_RESULT
                     OUTPUT_VARIABLE ${_PYTHON_PREFIX}_VERSION
                     ERROR_QUIET
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT _${_PYTHON_PREFIX}_RESULT)
      string (REGEX MATCHALL "[0-9]+" _${_PYTHON_PREFIX}_VERSIONS "${${_PYTHON_PREFIX}_VERSION}")
      list (GET _${_PYTHON_PREFIX}_VERSIONS 0 ${_PYTHON_PREFIX}_VERSION_MAJOR)
      list (GET _${_PYTHON_PREFIX}_VERSIONS 1 ${_PYTHON_PREFIX}_VERSION_MINOR)
      list (GET _${_PYTHON_PREFIX}_VERSIONS 2 ${_PYTHON_PREFIX}_VERSION_PATCH)
    else()
      # Interpreter is not usable
      set (${_PYTHON_PREFIX}_EXECUTABLE ${_PYTHON_PREFIX}_EXECUTABLE-NOTFOUND CACHE INTERNAL "" FORCE)
      unset (${_PYTHON_PREFIX}_VERSION)
    endif()
  endif()

  if (${_PYTHON_PREFIX}_EXECUTABLE
      AND ${_PYTHON_PREFIX}_VERSION_MAJOR VERSION_EQUAL _${_PYTHON_PREFIX}_REQUIRED_VERSION_MAJOR)
    set (${_PYTHON_PREFIX}_Interpreter_FOUND TRUE)
    # Use interpreter version for future searches to ensure consistency
    set (_${_PYTHON_PREFIX}_FIND_VERSIONS ${${_PYTHON_PREFIX}_VERSION_MAJOR}.${${_PYTHON_PREFIX}_VERSION_MINOR})
  endif()

  if (${_PYTHON_PREFIX}_Interpreter_FOUND)
    # retrieve interpreter identity
    execute_process (COMMAND "${${_PYTHON_PREFIX}_EXECUTABLE}" -V
                     RESULT_VARIABLE _${_PYTHON_PREFIX}_RESULT
                     OUTPUT_VARIABLE ${_PYTHON_PREFIX}_INTERPRETER_ID
                     ERROR_VARIABLE ${_PYTHON_PREFIX}_INTERPRETER_ID)
    if (NOT _${_PYTHON_PREFIX}_RESULT)
      if (${_PYTHON_PREFIX}_INTERPRETER_ID MATCHES "Anaconda")
        set (${_PYTHON_PREFIX}_INTERPRETER_ID "Anaconda")
      elseif (${_PYTHON_PREFIX}_INTERPRETER_ID MATCHES "Enthought")
        set (${_PYTHON_PREFIX}_INTERPRETER_ID "Canopy")
      else()
        string (REGEX REPLACE "^([^ ]+).*" "\\1" ${_PYTHON_PREFIX}_INTERPRETER_ID "${${_PYTHON_PREFIX}_INTERPRETER_ID}")
        if (${_PYTHON_PREFIX}_INTERPRETER_ID STREQUAL "Python")
          # try to get a more precise ID
          execute_process (COMMAND "${${_PYTHON_PREFIX}_EXECUTABLE}" -c "import sys; print(sys.copyright)"
                           RESULT_VARIABLE _${_PYTHON_PREFIX}_RESULT
                           OUTPUT_VARIABLE ${_PYTHON_PREFIX}_COPYRIGHT
                           ERROR_QUIET)
          if (${_PYTHON_PREFIX}_COPYRIGHT MATCHES "ActiveState")
            set (${_PYTHON_PREFIX}_INTERPRETER_ID "ActivePython")
          endif()
        endif()
      endif()
    else()
      set (${_PYTHON_PREFIX}_INTERPRETER_ID Python)
    endif()
  else()
    unset (${_PYTHON_PREFIX}_INTERPRETER_ID)
  endif()

  # retrieve various package installation directories
  execute_process (COMMAND "${${_PYTHON_PREFIX}_EXECUTABLE}" -c "import sys; from distutils import sysconfig;sys.stdout.write(';'.join([sysconfig.get_python_lib(plat_specific=False,standard_lib=True),sysconfig.get_python_lib(plat_specific=True,standard_lib=True),sysconfig.get_python_lib(plat_specific=False,standard_lib=False),sysconfig.get_python_lib(plat_specific=True,standard_lib=False)]))"

                   RESULT_VARIABLE _${_PYTHON_PREFIX}_RESULT
                   OUTPUT_VARIABLE _${_PYTHON_PREFIX}_LIBPATHS
                   ERROR_QUIET)
  if (NOT _${_PYTHON_PREFIX}_RESULT)
    list (GET _${_PYTHON_PREFIX}_LIBPATHS 0 ${_PYTHON_PREFIX}_STDLIB)
    list (GET _${_PYTHON_PREFIX}_LIBPATHS 1 ${_PYTHON_PREFIX}_STDARCH)
    list (GET _${_PYTHON_PREFIX}_LIBPATHS 2 ${_PYTHON_PREFIX}_SITELIB)
    list (GET _${_PYTHON_PREFIX}_LIBPATHS 3 ${_PYTHON_PREFIX}_SITEARCH)
  else()
    unset (${_PYTHON_PREFIX}_STDLIB)
    unset (${_PYTHON_PREFIX}_STDARCH)
    unset (${_PYTHON_PREFIX}_SITELIB)
    unset (${_PYTHON_PREFIX}_SITEARCH)
  endif()

  mark_as_advanced (${_PYTHON_PREFIX}_EXECUTABLE)
endif()


# second step, search for compiler (IronPython)
if ("Compiler" IN_LIST ${_PYTHON_PREFIX}_FIND_COMPONENTS)
  if (${_PYTHON_PREFIX}_FIND_REQUIRED_Compiler)
    list (APPEND _${_PYTHON_PREFIX}_REQUIRED_VARS ${_PYTHON_PREFIX}_COMPILER)
    list (APPEND _${_PYTHON_PREFIX}_CACHED_VARS ${_PYTHON_PREFIX}_COMPILER)
  endif()

  # IronPython specific artifacts
  # If IronPython interpreter is found, use its path
  unset (_${_PYTHON_PREFIX}_IRON_ROOT)
  if (${_PYTHON_PREFIX}_Interpreter_FOUND AND ${_PYTHON_PREFIX}_INTERPRETER_ID STREQUAL "IronPython")
    get_filename_component (_${_PYTHON_PREFIX}_IRON_ROOT "${${_PYTHON_PREFIX}_EXECUTABLE}" DIRECTORY)
  endif()

  # try using root dir and registry
  foreach (_${_PYTHON_PREFIX}_VERSION IN LISTS _${_PYTHON_PREFIX}_FIND_VERSIONS)
    find_program (${_PYTHON_PREFIX}_COMPILER
                  NAMES ipyc
                  HINTS ${_${_PYTHON_PREFIX}_IRON_ROOT} ${_${_PYTHON_PREFIX}_HINTS}
                  PATHS [HKEY_LOCAL_MACHINE\\SOFTWARE\\IronPython\\${_${_PYTHON_PREFIX}_VERSION}\\InstallPath]
                  NO_SYSTEM_ENVIRONMENT_PATH
                  NO_CMAKE_SYSTEM_PATH)
    if (${_PYTHON_PREFIX}_COMPILER)
      break()
    endif()
  endforeach()
  # try in standard paths
  find_program (${_PYTHON_PREFIX}_COMPILER
                NAMES ipyc)

  if (${_PYTHON_PREFIX}_COMPILER)
    # retrieve python environment version from compiler
    set (_${_PYTHON_PREFIX}_VERSION_DIR "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PythonCompilerVersion.dir")
    file (WRITE "${_${_PYTHON_PREFIX}_VERSION_DIR}/version.py" "import sys; sys.stdout.write('.'.join([str(x) for x in sys.version_info[:3]]))\n")
    execute_process (COMMAND "${${_PYTHON_PREFIX}_COMPILER}" /target:exe /embed "${_${_PYTHON_PREFIX}_VERSION_DIR}/version.py"
                     WORKING_DIRECTORY "${_${_PYTHON_PREFIX}_VERSION_DIR}"
                     OUTPUT_QUIET
                     ERROR_QUIET)
    execute_process (COMMAND "${_${_PYTHON_PREFIX}_VERSION_DIR}/version"
                     WORKING_DIRECTORY "${_${_PYTHON_PREFIX}_VERSION_DIR}"
                     RESULT_VARIABLE _${_PYTHON_PREFIX}_RESULT
                     OUTPUT_VARIABLE _${_PYTHON_PREFIX}_VERSION
                     ERROR_QUIET)
    if (NOT _${_PYTHON_PREFIX}_RESULT)
      string (REGEX MATCHALL "[0-9]+" _${_PYTHON_PREFIX}_VERSIONS "${_${_PYTHON_PREFIX}_VERSION}")
      list (GET _${_PYTHON_PREFIX}_VERSIONS 0 _${_PYTHON_PREFIX}_VERSION_MAJOR)
      list (GET _${_PYTHON_PREFIX}_VERSIONS 1 _${_PYTHON_PREFIX}_VERSION_MINOR)
      list (GET _${_PYTHON_PREFIX}_VERSIONS 2 _${_PYTHON_PREFIX}_VERSION_PATCH)

      if (NOT ${_PYTHON_PREFIX}_Interpreter_FOUND)
        # set public version information
        set (${_PYTHON_PREFIX}_VERSION ${_${_PYTHON_PREFIX}_VERSION})
        set (${_PYTHON_PREFIX}_VERSION_MAJOR ${_${_PYTHON_PREFIX}_VERSION_MAJOR})
        set (${_PYTHON_PREFIX}_VERSION_MINOR ${_${_PYTHON_PREFIX}_VERSION_MINOR})
        set (${_PYTHON_PREFIX}_VERSION_PATCH ${_${_PYTHON_PREFIX}_VERSION_PATCH})
      endif()
    else()
      # compiler not usable
      set (${_PYTHON_PREFIX}_COMPILER ${_PYTHON_PREFIX}_COMPILER-NOTFOUND CACHE INTERNAL "" FORCE)
    endif()
    file (REMOVE_RECURSE "${_${_PYTHON_PREFIX}_VERSION_DIR}")
  endif()

  if (${_PYTHON_PREFIX}_COMPILER)
    if (${_PYTHON_PREFIX}_Interpreter_FOUND)
      # Compiler must be compatible with interpreter
      if (${_${_PYTHON_PREFIX}_VERSION_MAJOR}.${_${_PYTHON_PREFIX}_VERSION_MINOR} VERSION_EQUAL ${${_PYTHON_PREFIX}_VERSION_MAJOR}.${${_PYTHON_PREFIX}_VERSION_MINOR})
        set (${_PYTHON_PREFIX}_Compiler_FOUND TRUE)
      endif()
    elseif (${_PYTHON_PREFIX}_VERSION_MAJOR VERSION_EQUAL _${_PYTHON_PREFIX}_REQUIRED_VERSION_MAJOR)
      set (${_PYTHON_PREFIX}_Compiler_FOUND TRUE)
    # Use compiler version for future searches to ensure consistency
    set (_${_PYTHON_PREFIX}_FIND_VERSIONS ${${_PYTHON_PREFIX}_VERSION_MAJOR}.${${_PYTHON_PREFIX}_VERSION_MINOR})
    endif()
  endif()

  if (${_PYTHON_PREFIX}_Compiler_FOUND)
    set (${_PYTHON_PREFIX}_COMPILER_ID IronPython)
  else()
    unset (${_PYTHON_PREFIX}_COMPILER_ID)
  endif()

  mark_as_advanced (${_PYTHON_PREFIX}_COMPILER)
endif()


# third step, search for the development artifacts
## Development environment is not compatible with IronPython interpreter
if ("Development" IN_LIST ${_PYTHON_PREFIX}_FIND_COMPONENTS
    AND NOT ${_PYTHON_PREFIX}_INTERPRETER_ID STREQUAL "IronPython")
  if (${_PYTHON_PREFIX}_FIND_REQUIRED_Development)
    list (APPEND _${_PYTHON_PREFIX}_REQUIRED_VARS ${_PYTHON_PREFIX}_LIBRARY
                                                  ${_PYTHON_PREFIX}_INCLUDE_DIR)
    list (APPEND _${_PYTHON_PREFIX}_CACHED_VARS ${_PYTHON_PREFIX}_LIBRARY
                                                ${_PYTHON_PREFIX}_LIBRARY_RELEASE
                                                ${_PYTHON_PREFIX}_RUNTIME_LIBRARY_RELEASE
                                                ${_PYTHON_PREFIX}_LIBRARY_DEBUG
                                                ${_PYTHON_PREFIX}_RUNTIME_LIBRARY_DEBUG
                                                ${_PYTHON_PREFIX}_INCLUDE_DIR)
  endif()

  # Support preference of static libs by adjusting CMAKE_FIND_LIBRARY_SUFFIXES
  unset (_${_PYTHON_PREFIX}_CMAKE_FIND_LIBRARY_SUFFIXES)
  if (DEFINED ${_PYTHON_PREFIX}_USE_STATIC_LIBS AND NOT WIN32)
    set(_${_PYTHON_PREFIX}_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    if(${_PYTHON_PREFIX}_USE_STATIC_LIBS)
      set (CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
    else()
      list (REMOVE_ITEM CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
  else()
  endif()

  # if python interpreter is found, use its location and version to ensure consistency
  # between interpreter and development environment
  unset (_${_PYTHON_PREFIX}_PREFIX)
  if (${_PYTHON_PREFIX}_Interpreter_FOUND)
    execute_process (COMMAND "${${_PYTHON_PREFIX}_EXECUTABLE}" -c
                             "import sys; from distutils import sysconfig; sys.stdout.write(sysconfig.PREFIX)"
                     RESULT_VARIABLE _${_PYTHON_PREFIX}_RESULT
                     OUTPUT_VARIABLE _${_PYTHON_PREFIX}_PREFIX
                     ERROR_QUIET
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (_${_PYTHON_PREFIX}_RESULT)
      unset (_${_PYTHON_PREFIX}_PREFIX)
    endif()
  endif()
  set (_${_PYTHON_PREFIX}_HINTS "${_${_PYTHON_PREFIX}_PREFIX}" "${${_PYTHON_PREFIX}_ROOT_DIR}" ENV ${_PYTHON_PREFIX}_ROOT_DIR)

  foreach (_${_PYTHON_PREFIX}_VERSION IN LISTS _${_PYTHON_PREFIX}_FIND_VERSIONS)
    string (REPLACE "." "" _${_PYTHON_PREFIX}_VERSION_NO_DOTS ${_${_PYTHON_PREFIX}_VERSION})

    # try to use pythonX.Y-config tool
    set (_${_PYTHON_PREFIX}_CONFIG_NAMES)
    if (DEFINED CMAKE_LIBRARY_ARCHITECTURE)
      set (_${_PYTHON_PREFIX}_CONFIG_NAMES "${CMAKE_LIBRARY_ARCHITECTURE}-python${_${_PYTHON_PREFIX}_VERSION}-config")
    endif()
    list (APPEND _${_PYTHON_PREFIX}_CONFIG_NAMES "python${_${_PYTHON_PREFIX}_VERSION}-config")
    find_program (_${_PYTHON_PREFIX}_CONFIG
                  NAMES ${_${_PYTHON_PREFIX}_CONFIG_NAMES}
                  HINTS ${_${_PYTHON_PREFIX}_HINTS}
                  PATH_SUFFIXES bin)
    unset (_${_PYTHON_PREFIX}_CONFIG_NAMES)

    if (NOT _${_PYTHON_PREFIX}_CONFIG)
      continue()
    endif()

    # retrieve root install directory
    execute_process (COMMAND "${_${_PYTHON_PREFIX}_CONFIG}" --prefix
                     RESULT_VARIABLE _${_PYTHON_PREFIX}_RESULT
                     OUTPUT_VARIABLE _${_PYTHON_PREFIX}_PREFIX
                     ERROR_QUIET
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (_${_PYTHON_PREFIX}_RESULT)
      # python-config is not usable
      unset (_${_PYTHON_PREFIX}_CONFIG CACHE)
      continue()
    endif()
    set (_${_PYTHON_PREFIX}_HINTS "${_${_PYTHON_PREFIX}_PREFIX}" "${${_PYTHON_PREFIX}_ROOT_DIR}" ENV ${_PYTHON_PREFIX}_ROOT_DIR)

    # retrieve library
    execute_process (COMMAND "${_${_PYTHON_PREFIX}_CONFIG}" --ldflags
                     RESULT_VARIABLE _${_PYTHON_PREFIX}_RESULT
                     OUTPUT_VARIABLE _${_PYTHON_PREFIX}_FLAGS
                     ERROR_QUIET
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT _${_PYTHON_PREFIX}_RESULT)
      # retrieve library directory
      string (REGEX MATCHALL "-L[^ ]+" _${_PYTHON_PREFIX}_LIB_DIRS "${_${_PYTHON_PREFIX}_FLAGS}")
      string (REPLACE "-L" "" _${_PYTHON_PREFIX}_LIB_DIRS "${_${_PYTHON_PREFIX}_LIB_DIRS}")
      list (REMOVE_DUPLICATES _${_PYTHON_PREFIX}_LIB_DIRS)
      # retrieve library name
      string (REGEX MATCHALL "-lpython[^ ]+" _${_PYTHON_PREFIX}_LIB_NAMES "${_${_PYTHON_PREFIX}_FLAGS}")
      string (REPLACE "-l" "" _${_PYTHON_PREFIX}_LIB_NAMES "${_${_PYTHON_PREFIX}_LIB_NAMES}")
      list (REMOVE_DUPLICATES _${_PYTHON_PREFIX}_LIB_NAMES)

      find_library (${_PYTHON_PREFIX}_LIBRARY_RELEASE
                    NAMES ${_${_PYTHON_PREFIX}_LIB_NAMES}
                    NAMES_PER_DIR
                    HINTS ${_${_PYTHON_PREFIX}_HINTS} ${_${_PYTHON_PREFIX}_LIB_DIRS}
                    PATH_SUFFIXES lib
                    NO_SYSTEM_ENVIRONMENT_PATH
                    NO_CMAKE_SYSTEM_PATH)
      # retrieve runtime library
      if (${_PYTHON_PREFIX}_LIBRARY_RELEASE)
        get_filename_component (_${_PYTHON_PREFIX}_PATH "${${_PYTHON_PREFIX}_LIBRARY_RELEASE}" DIRECTORY)
        _python_find_runtime_library (${_PYTHON_PREFIX}_RUNTIME_LIBRARY_RELEASE
                                      NAMES ${_${_PYTHON_PREFIX}_LIB_NAMES}
                                      NAMES_PER_DIR
                                      HINTS ${_${_PYTHON_PREFIX}_PATH} ${_${_PYTHON_PREFIX}_HINTS}
                                      PATH_SUFFIXES bin
                                      NO_SYSTEM_ENVIRONMENT_PATH
                                      NO_CMAKE_SYSTEM_PATH)
      endif()
    endif()

    # retrieve include directory
    execute_process (COMMAND "${_${_PYTHON_PREFIX}_CONFIG}" --includes
                     RESULT_VARIABLE _${_PYTHON_PREFIX}_RESULT
                     OUTPUT_VARIABLE _${_PYTHON_PREFIX}_FLAGS
                     ERROR_QUIET
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT _${_PYTHON_PREFIX}_RESULT)
      # retrieve include directory
      string (REGEX MATCHALL "-I[^ ]+" _${_PYTHON_PREFIX}_INCLUDE_DIRS "${_${_PYTHON_PREFIX}_FLAGS}")
      string (REPLACE "-I" "" _${_PYTHON_PREFIX}_INCLUDE_DIRS "${_${_PYTHON_PREFIX}_INCLUDE_DIRS}")
      list (REMOVE_DUPLICATES _${_PYTHON_PREFIX}_INCLUDE_DIRS)

      find_path (${_PYTHON_PREFIX}_INCLUDE_DIR
                 NAMES Python.h
                 HINTS ${_${_PYTHON_PREFIX}_INCLUDE_DIRS}
                 NO_SYSTEM_ENVIRONMENT_PATH
                 NO_CMAKE_SYSTEM_PATH)
    endif()

    if (${_PYTHON_PREFIX}_LIBRARY_RELEASE AND ${_PYTHON_PREFIX}_INCLUDE_DIR)
      break()
    endif()
  endforeach()

  # Rely on HINTS and standard paths if config tool failed to locate artifacts
  if (NOT (${_PYTHON_PREFIX}_LIBRARY_RELEASE OR ${_PYTHON_PREFIX}_LIBRARY_DEBUG) OR NOT ${_PYTHON_PREFIX}_INCLUDE_DIR)
    foreach (_${_PYTHON_PREFIX}_VERSION IN LISTS _${_PYTHON_PREFIX}_FIND_VERSIONS)
      string (REPLACE "." "" _${_PYTHON_PREFIX}_VERSION_NO_DOTS ${_${_PYTHON_PREFIX}_VERSION})

      _python_get_frameworks (_${_PYTHON_PREFIX}_FRAMEWORK_PATHS ${_${_PYTHON_PREFIX}_VERSION})

      # search first in known locations
      find_library (${_PYTHON_PREFIX}_LIBRARY_RELEASE
                    NAMES python${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}
                          python${_${_PYTHON_PREFIX}_VERSION}mu
                          python${_${_PYTHON_PREFIX}_VERSION}m
                          python${_${_PYTHON_PREFIX}_VERSION}u
                          python${_${_PYTHON_PREFIX}_VERSION}
                    NAMES_PER_DIR
                    HINTS ${_${_PYTHON_PREFIX}_HINTS}
                    PATHS ${_${_PYTHON_PREFIX}_FRAMEWORK_PATHS}
                          [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_${_PYTHON_PREFIX}_VERSION}\\InstallPath]
                          [HKEY_CURRENT_USER\\SOFTWARE\\Python\\ContinuumAnalytics\\Anaconda${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}-${_${_PYTHON_PREFIX}_ARCH}\\InstallPath]
                          [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_${_PYTHON_PREFIX}_VERSION}\\InstallPath]
                          [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\ContinuumAnalytics\\Anaconda${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}-${_${_PYTHON_PREFIX}_ARCH}\\InstallPath]
                    PATH_SUFFIXES lib/${CMAKE_LIBRARY_ARCHITECTURE} lib libs
                                  lib/python${_${_PYTHON_PREFIX}_VERSION}/config-${_${_PYTHON_PREFIX}_VERSION}mu
                                  lib/python${_${_PYTHON_PREFIX}_VERSION}/config-${_${_PYTHON_PREFIX}_VERSION}m
                                  lib/python${_${_PYTHON_PREFIX}_VERSION}/config-${_${_PYTHON_PREFIX}_VERSION}u
                                  lib/python${_${_PYTHON_PREFIX}_VERSION}/config-${_${_PYTHON_PREFIX}_VERSION}
                                  lib/python${_${_PYTHON_PREFIX}_VERSION}/config
                    NO_SYSTEM_ENVIRONMENT_PATH
                    NO_CMAKE_SYSTEM_PATH)
      # search in all default paths
      find_library (${_PYTHON_PREFIX}_LIBRARY_RELEASE
                    NAMES python${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}
                          python${_${_PYTHON_PREFIX}_VERSION}mu
                          python${_${_PYTHON_PREFIX}_VERSION}m
                          python${_${_PYTHON_PREFIX}_VERSION}u
                          python${_${_PYTHON_PREFIX}_VERSION}
                    NAMES_PER_DIR
                    PATH_SUFFIXES lib/${CMAKE_LIBRARY_ARCHITECTURE} lib libs
                                  lib/python${_${_PYTHON_PREFIX}_VERSION}/config-${_${_PYTHON_PREFIX}_VERSION}mu
                                  lib/python${_${_PYTHON_PREFIX}_VERSION}/config-${_${_PYTHON_PREFIX}_VERSION}m
                                  lib/python${_${_PYTHON_PREFIX}_VERSION}/config-${_${_PYTHON_PREFIX}_VERSION}u
                                  lib/python${_${_PYTHON_PREFIX}_VERSION}/config-${_${_PYTHON_PREFIX}_VERSION}
                                  lib/python${_${_PYTHON_PREFIX}_VERSION}/config)
      # retrieve runtime library
      if (${_PYTHON_PREFIX}_LIBRARY_RELEASE)
        get_filename_component (_${_PYTHON_PREFIX}_PATH "${${_PYTHON_PREFIX}_LIBRARY_RELEASE}" DIRECTORY)
        _python_find_runtime_library (${_PYTHON_PREFIX}_RUNTIME_LIBRARY_RELEASE
                                      NAMES python${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}
                                            python${_${_PYTHON_PREFIX}_VERSION}mu
                                            python${_${_PYTHON_PREFIX}_VERSION}m
                                            python${_${_PYTHON_PREFIX}_VERSION}u
                                            python${_${_PYTHON_PREFIX}_VERSION}
                                      NAMES_PER_DIR
                                      HINTS "${_${_PYTHON_PREFIX}_PATH}" ${_${_PYTHON_PREFIX}_HINTS}
                                      PATHS ${_${_PYTHON_PREFIX}_FRAMEWORK_PATHS}
                                            [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_${_PYTHON_PREFIX}_VERSION}\\InstallPath]
                                            [HKEY_CURRENT_USER\\SOFTWARE\\Python\\ContinuumAnalytics\\Anaconda${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}-${_${_PYTHON_PREFIX}_ARCH}\\InstallPath]
                                            [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_${_PYTHON_PREFIX}_VERSION}\\InstallPath]
                                            [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\ContinuumAnalytics\\Anaconda${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}-${_${_PYTHON_PREFIX}_ARCH}\\InstallPath]
                                      PATH_SUFFIXES bin)
      endif()

      if (WIN32)
        # search for debug library
        if (${_PYTHON_PREFIX}_LIBRARY_RELEASE)
          # use library location as a hint
          get_filename_component (_${_PYTHON_PREFIX}_PATH "${${_PYTHON_PREFIX}_LIBRARY_RELEASE}" DIRECTORY)
          find_library (${_PYTHON_PREFIX}_LIBRARY_DEBUG
                      NAMES python${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}_d
                      NAMES_PER_DIR
                      HINTS "${_${_PYTHON_PREFIX}_PATH}" ${_${_PYTHON_PREFIX}_HINTS}
                      NO_DEFAULT_PATH)
        else()
          # search first in known locations
          find_library (${_PYTHON_PREFIX}_LIBRARY_DEBUG
                        NAMES python${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}_d
                        NAMES_PER_DIR
                        HINTS ${_${_PYTHON_PREFIX}_HINTS}
                        PATHS [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_${_PYTHON_PREFIX}_VERSION}\\InstallPath]
                              [HKEY_CURRENT_USER\\SOFTWARE\\Python\\ContinuumAnalytics\\Anaconda${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}-${_${_PYTHON_PREFIX}_ARCH}\\InstallPath]
                              [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_${_PYTHON_PREFIX}_VERSION}\\InstallPath]
                              [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\ContinuumAnalytics\\Anaconda${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}-${_${_PYTHON_PREFIX}_ARCH}\\InstallPath]
                        PATH_SUFFIXES lib libs
                        NO_SYSTEM_ENVIRONMENT_PATH
                        NO_CMAKE_SYSTEM_PATH)
          # search in all default paths
          find_library (${_PYTHON_PREFIX}_LIBRARY_DEBUG
                        NAMES python${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}_d
                        NAMES_PER_DIR
                        PATH_SUFFIXES lib libs)
        endif()
        if (${_PYTHON_PREFIX}_LIBRARY_DEBUG)
          get_filename_component (_${_PYTHON_PREFIX}_PATH "${${_PYTHON_PREFIX}_LIBRARY_DEBUG}" DIRECTORY)
          _python_find_runtime_library (${_PYTHON_PREFIX}_RUNTIME_LIBRARY_DEBUG
                                        NAMES python${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}_d
                                        NAMES_PER_DIR
                                        HINTS "${_${_PYTHON_PREFIX}_PATH}" ${_${_PYTHON_PREFIX}_HINTS}
                                        PATHS [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_${_PYTHON_PREFIX}_VERSION}\\InstallPath]
                                              [HKEY_CURRENT_USER\\SOFTWARE\\Python\\ContinuumAnalytics\\Anaconda${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}-${_${_PYTHON_PREFIX}_ARCH}\\InstallPath]
                                              [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_${_PYTHON_PREFIX}_VERSION}\\InstallPath]
                                              [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\ContinuumAnalytics\\Anaconda${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}-${_${_PYTHON_PREFIX}_ARCH}\\InstallPath]
                                        PATH_SUFFIXES bin)
        endif()
      endif()

      # Don't search for include dir until library location is known
      if (${_PYTHON_PREFIX}_LIBRARY_RELEASE OR ${_PYTHON_PREFIX}_LIBRARY_DEBUG)
        unset (_${_PYTHON_PREFIX}_INCLUDE_HINTS)
        foreach (_${_PYTHON_PREFIX}_LIB IN ITEMS ${_PYTHON_PREFIX}_LIBRARY_RELEASE ${_PYTHON_PREFIX}_LIBRARY_DEBUG)
          if (${_${_PYTHON_PREFIX}_LIB})
            # Use the library's install prefix as a hint
            if (${_${_PYTHON_PREFIX}_LIB} MATCHES "^(.+/Frameworks/Python.framework/Versions/[0-9.]+)")
              list (APPEND _${_PYTHON_PREFIX}_INCLUDE_HINTS "${CMAKE_MATCH_1}")
            elseif (${_${_PYTHON_PREFIX}_LIB} MATCHES "^(.+)/lib(64|32)?/python[0-9.]+/config")
              list (APPEND _${_PYTHON_PREFIX}_INCLUDE_HINTS "${CMAKE_MATCH_1}")
            elseif (DEFINED CMAKE_LIBRARY_ARCHITECTURE AND ${_${_PYTHON_PREFIX}_LIB} MATCHES "^(.+)/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
              list (APPEND _${_PYTHON_PREFIX}_INCLUDE_HINTS "${CMAKE_MATCH_1}")
            else()
              # assume library is in a directory under root
              get_filename_component (_${_PYTHON_PREFIX}_PREFIX "${${_${_PYTHON_PREFIX}_LIB}}" DIRECTORY)
              get_filename_component (_${_PYTHON_PREFIX}_PREFIX "${_${_PYTHON_PREFIX}_PREFIX}" DIRECTORY)
              list (APPEND _${_PYTHON_PREFIX}_INCLUDE_HINTS "${_${_PYTHON_PREFIX}_PREFIX}")
            endif()
          endif()
        endforeach()
        list (REMOVE_DUPLICATES _${_PYTHON_PREFIX}_INCLUDE_HINTS)

        find_path (${_PYTHON_PREFIX}_INCLUDE_DIR
                   NAMES Python.h
                   HINTS ${_${_PYTHON_PREFIX}_INCLUDE_HINTS} ${_${_PYTHON_PREFIX}_HINTS}
                   PATHS ${_${_PYTHON_PREFIX}_FRAMEWORK_PATHS}
                         [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_${_PYTHON_PREFIX}_VERSION}\\InstallPath]
                         [HKEY_CURRENT_USER\\SOFTWARE\\Python\\ContinuumAnalytics\\Anaconda${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}-${_${_PYTHON_PREFIX}_ARCH}\\InstallPath]
                         [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_${_PYTHON_PREFIX}_VERSION}\\InstallPath]
                         [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\ContinuumAnalytics\\Anaconda${_${_PYTHON_PREFIX}_VERSION_NO_DOTS}-${_${_PYTHON_PREFIX}_ARCH}\\InstallPath]
                   PATH_SUFFIXES include/python${_${_PYTHON_PREFIX}_VERSION}mu
                                 include/python${_${_PYTHON_PREFIX}_VERSION}m
                                 include/python${_${_PYTHON_PREFIX}_VERSION}u
                                 include/python${_${_PYTHON_PREFIX}_VERSION}
                                 include
                   NO_SYSTEM_ENVIRONMENT_PATH
                   NO_CMAKE_SYSTEM_PATH)
      endif()

      if ((${_PYTHON_PREFIX}_LIBRARY_RELEASE OR ${_PYTHON_PREFIX}_LIBRARY_DEBUG) AND ${_PYTHON_PREFIX}_INCLUDE_DIR)
        break()
      endif()
    endforeach()

    # search header file in standard locations
    find_path (${_PYTHON_PREFIX}_INCLUDE_DIR
               NAMES Python.h)
  endif()

  if (${_PYTHON_PREFIX}_INCLUDE_DIR)
    # retrieve version from header file
    file (STRINGS "${${_PYTHON_PREFIX}_INCLUDE_DIR}/patchlevel.h" _${_PYTHON_PREFIX}_VERSION
          REGEX "^#define[ \t]+PY_VERSION[ \t]+\"[^\"]+\"")
    string (REGEX REPLACE "^#define[ \t]+PY_VERSION[ \t]+\"([^\"]+)\".*" "\\1"
                          _${_PYTHON_PREFIX}_VERSION "${_${_PYTHON_PREFIX}_VERSION}")
    string (REGEX MATCHALL "[0-9]+" _${_PYTHON_PREFIX}_VERSIONS "${_${_PYTHON_PREFIX}_VERSION}")
    list (GET _${_PYTHON_PREFIX}_VERSIONS 0 _${_PYTHON_PREFIX}_VERSION_MAJOR)
    list (GET _${_PYTHON_PREFIX}_VERSIONS 1 _${_PYTHON_PREFIX}_VERSION_MINOR)
    list (GET _${_PYTHON_PREFIX}_VERSIONS 2 _${_PYTHON_PREFIX}_VERSION_PATCH)

    if (NOT ${_PYTHON_PREFIX}_Interpreter_FOUND AND NOT ${_PYTHON_PREFIX}_Compiler_FOUND)
      # set public version information
      set (${_PYTHON_PREFIX}_VERSION ${_${_PYTHON_PREFIX}_VERSION})
      set (${_PYTHON_PREFIX}_VERSION_MAJOR ${_${_PYTHON_PREFIX}_VERSION_MAJOR})
      set (${_PYTHON_PREFIX}_VERSION_MINOR ${_${_PYTHON_PREFIX}_VERSION_MINOR})
      set (${_PYTHON_PREFIX}_VERSION_PATCH ${_${_PYTHON_PREFIX}_VERSION_PATCH})
    endif()
  endif()

  # define public variables
  include (SelectLibraryConfigurations)
  select_library_configurations (${_PYTHON_PREFIX})
  if (${_PYTHON_PREFIX}_RUNTIME_LIBRARY_RELEASE)
    set (${_PYTHON_PREFIX}_RUNTIME_LIBRARY "${${_PYTHON_PREFIX}_RUNTIME_LIBRARY_RELEASE}")
  elseif (${_PYTHON_PREFIX}_RUNTIME_LIBRARY_DEBUG)
    set (${_PYTHON_PREFIX}_RUNTIME_LIBRARY "${${_PYTHON_PREFIX}_RUNTIME_LIBRARY_DEBUG}")
  else()
    set (${_PYTHON_PREFIX}_RUNTIME_LIBRARY "$${_PYTHON_PREFIX}_RUNTIME_LIBRARY-NOTFOUND")
  endif()

  _python_set_library_dirs (${_PYTHON_PREFIX}_LIBRARY_DIRS
                            ${_PYTHON_PREFIX}_LIBRARY_RELEASE ${_PYTHON_PREFIX}_LIBRARY_DEBUG)
  if (UNIX)
    if (${_PYTHON_PREFIX}_LIBRARY_RELEASE MATCHES "${CMAKE_SHARED_LIBRARY_SUFFIX}$"
        OR ${_PYTHON_PREFIX}_LIBRARY_RELEASE MATCHES "${CMAKE_SHARED_LIBRARY_SUFFIX}$")
      set (${_PYTHON_PREFIX}_RUNTIME_LIBRARY_DIRS ${${_PYTHON_PREFIX}_LIBRARY_DIRS})
    endif()
  else()
      _python_set_library_dirs (${_PYTHON_PREFIX}_RUNTIME_LIBRARY_DIRS
                                ${_PYTHON_PREFIX}_RUNTIME_LIBRARY_RELEASE ${_PYTHON_PREFIX}_RUNTIME_LIBRARY_DEBUG)
  endif()

  set (${_PYTHON_PREFIX}_INCLUDE_DIRS "${${_PYTHON_PREFIX}_INCLUDE_DIR}")

  mark_as_advanced (${_PYTHON_PREFIX}_RUNTIME_LIBRARY_RELEASE
                    ${_PYTHON_PREFIX}_RUNTIME_LIBRARY_DEBUG
                    ${_PYTHON_PREFIX}_INCLUDE_DIR)

  if ((${_PYTHON_PREFIX}_LIBRARY_RELEASE OR ${_PYTHON_PREFIX}_LIBRARY_DEBUG)
      AND ${_PYTHON_PREFIX}_INCLUDE_DIR)
    if (${_PYTHON_PREFIX}_Interpreter_FOUND OR ${_PYTHON_PREFIX}_Compiler_FOUND)
      # development environment must be compatible with interpreter/compiler
      if (${_${_PYTHON_PREFIX}_VERSION_MAJOR}.${_${_PYTHON_PREFIX}_VERSION_MINOR} VERSION_EQUAL ${${_PYTHON_PREFIX}_VERSION_MAJOR}.${${_PYTHON_PREFIX}_VERSION_MINOR})
        set (${_PYTHON_PREFIX}_Development_FOUND TRUE)
      endif()
    elseif (${_PYTHON_PREFIX}_VERSION_MAJOR VERSION_EQUAL _${_PYTHON_PREFIX}_REQUIRED_VERSION_MAJOR)
      set (${_PYTHON_PREFIX}_Development_FOUND TRUE)
    endif()
  endif()

  # Restore the original find library ordering
  if (DEFINED _${_PYTHON_PREFIX}_CMAKE_FIND_LIBRARY_SUFFIXES)
    set (CMAKE_FIND_LIBRARY_SUFFIXES ${_${_PYTHON_PREFIX}_CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
endif()

# final validation
if (${_PYTHON_PREFIX}_VERSION_MAJOR AND
    NOT ${_PYTHON_PREFIX}_VERSION_MAJOR VERSION_EQUAL _${_PYTHON_PREFIX}_REQUIRED_VERSION_MAJOR)
  _python_display_failure ("Could NOT find ${_PYTHON_PREFIX}: Found unsuitable major version \"${${_PYTHON_PREFIX}_VERSION_MAJOR}\", but required major version is exact version \"${_${_PYTHON_PREFIX}_REQUIRED_VERSION_MAJOR}\"")
endif()

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (${_PYTHON_PREFIX}
                                   REQUIRED_VARS ${_${_PYTHON_PREFIX}_REQUIRED_VARS}
                                   VERSION_VAR ${_PYTHON_PREFIX}_VERSION
                                   HANDLE_COMPONENTS)

# Create imported targets and helper functions
if ("Interpreter" IN_LIST ${_PYTHON_PREFIX}_FIND_COMPONENTS
    AND ${_PYTHON_PREFIX}_Interpreter_FOUND
    AND NOT TARGET ${_PYTHON_PREFIX}::Interpreter)
  add_executable (${_PYTHON_PREFIX}::Interpreter IMPORTED)
  set_property (TARGET ${_PYTHON_PREFIX}::Interpreter
                PROPERTY IMPORTED_LOCATION "${${_PYTHON_PREFIX}_EXECUTABLE}")
endif()

if ("Compiler" IN_LIST ${_PYTHON_PREFIX}_FIND_COMPONENTS
    AND ${_PYTHON_PREFIX}_Compiler_FOUND
    AND NOT TARGET ${_PYTHON_PREFIX}::Compiler)
  add_executable (${_PYTHON_PREFIX}::Compiler IMPORTED)
  set_property (TARGET ${_PYTHON_PREFIX}::Compiler
                PROPERTY IMPORTED_LOCATION "${${_PYTHON_PREFIX}_COMPILER}")
endif()

if ("Development" IN_LIST ${_PYTHON_PREFIX}_FIND_COMPONENTS
    AND ${_PYTHON_PREFIX}_Development_FOUND AND NOT TARGET ${_PYTHON_PREFIX}::Python)

  if (${_PYTHON_PREFIX}_LIBRARY_RELEASE MATCHES "${CMAKE_SHARED_LIBRARY_SUFFIX}$"
      OR ${_PYTHON_PREFIX}_LIBRARY_DEBUG MATCHES "${CMAKE_SHARED_LIBRARY_SUFFIX}$"
      OR ${_PYTHON_PREFIX}_RUNTIME_LIBRARY_RELEASE OR ${_PYTHON_PREFIX}_RUNTIME_LIBRARY_DEBUG)
    set (_${_PYTHON_PREFIX}_LIBRARY_TYPE SHARED)
  else()
    set (_${_PYTHON_PREFIX}_LIBRARY_TYPE STATIC)
  endif()

  add_library (${_PYTHON_PREFIX}::Python ${_${_PYTHON_PREFIX}_LIBRARY_TYPE} IMPORTED)

  set_property (TARGET ${_PYTHON_PREFIX}::Python
                PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${${_PYTHON_PREFIX}_INCLUDE_DIR}")

  if ((${_PYTHON_PREFIX}_LIBRARY_RELEASE AND ${_PYTHON_PREFIX}_RUNTIME_LIBRARY_RELEASE)
      OR (${_PYTHON_PREFIX}_LIBRARY_DEBUG AND ${_PYTHON_PREFIX}_RUNTIME_LIBRARY_DEBUG))
    # System manage shared libraries in two parts: import and runtime
    if (${_PYTHON_PREFIX}_LIBRARY_RELEASE AND ${_PYTHON_PREFIX}_LIBRARY_DEBUG)
      set_property (TARGET ${_PYTHON_PREFIX}::Python PROPERTY IMPORTED_CONFIGURATIONS RELEASE DEBUG)
      set_target_properties (${_PYTHON_PREFIX}::Python
                             PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
                                        IMPORTED_IMPLIB_RELEASE "${${_PYTHON_PREFIX}_LIBRARY_RELEASE}"
                                        IMPORTED_LOCATION_RELEASE "${${_PYTHON_PREFIX}_RUNTIME_LIBRARY_RELEASE}")
      set_target_properties (${_PYTHON_PREFIX}::Python
                             PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
                                        IMPORTED_IMPLIB_DEBUG "${${_PYTHON_PREFIX}_LIBRARY_DEBUG}"
                                        IMPORTED_LOCATION_DEBUG "${${_PYTHON_PREFIX}_RUNTIME_LIBRARY_DEBUG}")
    else()
      set_target_properties (${_PYTHON_PREFIX}::Python
                             PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                                        IMPORTED_IMPLIB "${${_PYTHON_PREFIX}_LIBRARY}"
                                        IMPORTED_LOCATION "${${_PYTHON_PREFIX}_RUNTIME_LIBRARY}")
    endif()
  else()
    if (${_PYTHON_PREFIX}_LIBRARY_RELEASE AND ${_PYTHON_PREFIX}_LIBRARY_DEBUG)
      set_property (TARGET ${_PYTHON_PREFIX}::Python PROPERTY IMPORTED_CONFIGURATIONS RELEASE DEBUG)
      set_target_properties (${_PYTHON_PREFIX}::Python
                             PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
                                        IMPORTED_LOCATION_RELEASE "${${_PYTHON_PREFIX}_LIBRARY_RELEASE}")
      set_target_properties (${_PYTHON_PREFIX}::Python
                             PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
                                        IMPORTED_LOCATION_DEBUG "${${_PYTHON_PREFIX}_LIBRARY_DEBUG}")
    else()
      set_target_properties (${_PYTHON_PREFIX}::Python
                             PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                                        IMPORTED_LOCATION "${${_PYTHON_PREFIX}_LIBRARY}")
    endif()
  endif()

  if (_${_PYTHON_PREFIX}_CONFIG AND _${_PYTHON_PREFIX}_LIBRARY_TYPE STREQUAL "STATIC")
    # extend link information with dependent libraries
    execute_process (COMMAND "${_${_PYTHON_PREFIX}_CONFIG}" --ldflags
                     RESULT_VARIABLE _${_PYTHON_PREFIX}_RESULT
                     OUTPUT_VARIABLE _${_PYTHON_PREFIX}_FLAGS
                     ERROR_QUIET
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT _${_PYTHON_PREFIX}_RESULT)
      string (REGEX MATCHALL "-[Ll][^ ]+" _${_PYTHON_PREFIX}_LINK_LIBRARIES "${_${_PYTHON_PREFIX}_FLAGS}")
      # remove elements relative to python library itself
      list (FILTER _${_PYTHON_PREFIX}_LINK_LIBRARIES EXCLUDE REGEX "-lpython")
      foreach (_${_PYTHON_PREFIX}_DIR IN LISTS ${_PYTHON_PREFIX}_LIBRARY_DIRS)
        list (FILTER _${_PYTHON_PREFIX}_LINK_LIBRARIES EXCLUDE REGEX "-L${${_PYTHON_PREFIX}_DIR}")
      endforeach()
      set_property (TARGET ${_PYTHON_PREFIX}::Python
                    PROPERTY INTERFACE_LINK_LIBRARIES ${_${_PYTHON_PREFIX}_LINK_LIBRARIES})
    endif()
  endif()

  #
  # PYTHON_ADD_LIBRARY (<name> [STATIC|SHARED|MODULE] src1 src2 ... srcN)
  # It is used to build modules for python.
  #
  function (__${_PYTHON_PREFIX}_ADD_LIBRARY prefix name)
    cmake_parse_arguments (PARSE_ARGV 2 PYTHON_ADD_LIBRARY
                           "STATIC;SHARED;MODULE" "" "")

    unset (type)
    if (NOT (PYTHON_ADD_LIBRARY_STATIC
          OR PYTHON_ADD_LIBRARY_SHARED
          OR PYTHON_ADD_LIBRARY_MODULE))
      set (type MODULE)
    endif()
    add_library (${name} ${type} ${ARGN})
    target_link_libraries (${name} PRIVATE ${prefix}::Python)

    # customize library name to follow module name rules
    get_property (type TARGET ${name} PROPERTY TYPE)
    if (type STREQUAL "MODULE_LIBRARY")
      set_property (TARGET ${name} PROPERTY PREFIX "")
      if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set_property (TARGET ${name} PROPERTY SUFFIX ".pyd")
      endif()
    endif()
  endfunction()
endif()

# final clean-up

# Restore CMAKE_FIND_FRAMEWORK
if (DEFINED _${_PYTHON_PREFIX}_CMAKE_FIND_FRAMEWORK)
  set (CMAKE_FIND_FRAMEWORK ${_${_PYTHON_PREFIX}_CMAKE_FIND_FRAMEWORK})
  unset (_${_PYTHON_PREFIX}_CMAKE_FIND_FRAMEWORK)
else()
  unset (CMAKE_FIND_FRAMEWORK)
endif()

unset (_${_PYTHON_PREFIX}_CONFIG CACHE)
