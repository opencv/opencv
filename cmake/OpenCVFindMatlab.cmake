# ----- Find Matlab/Octave -----
#
# OpenCVFindMatlab.cmake attempts to locate the install path of Matlab in order
# to extract the mex headers, libraries and shell scripts. If found
# successfully, the following variables will be defined
#
#   MATLAB_FOUND:       true/false
#   MATLAB_ROOT_DIR:    Root of Matlab installation
#   MATLAB_BIN:         The main Matlab "executable" (shell script)
#   MATLAB_MEX_SCRIPT:  The mex script used to compile mex files
#   MATLAB_INCLUDE_DIRS:Path to "mex.h"
#   MATLAB_LIBRARY_DIRS:Path to mex and matrix libraries
#   MATLAB_LIBRARIES:   The Matlab libs, usually mx, mex, mat
#   MATLAB_MEXEXT:      The mex library extension. It will be one of:
#                         mexwin32, mexwin64,  mexglx, mexa64, mexmac,
#                         mexmaci,  mexmaci64, mexsol, mexs64
#   MATLAB_ARCH:        The installation architecture. It is **usually**
#                       the MEXEXT with the preceding "mex" removed,
#                       though it's different for linux distros.
#
# There doesn't appear to be an elegant way to detect all versions of Matlab
# across different platforms. If you know the matlab path and want to avoid
# the search, you can define the path to the Matlab root when invoking cmake:
#
#   cmake -DMATLAB_ROOT_DIR='/PATH/TO/ROOT_DIR' ..



# ----- set_library_presuffix -----
#
# Matlab tends to use some non-standard prefixes and suffixes on its libraries.
# For example, libmx.dll on Windows (Windows does not add prefixes) and
# mkl.dylib on OS X (OS X uses "lib" prefixes).
# On some versions of Windows the .dll suffix also appears to not be checked.
#
# This function modifies the library prefixes and suffixes used by
# find_library when finding Matlab libraries. It does not affect scopes
# outside of this file.
function(set_libarch_prefix_suffix)
  if (UNIX AND NOT APPLE)
    set(CMAKE_FIND_LIBRARY_PREFIXES "lib" PARENT_SCOPE)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a" PARENT_SCOPE)
  elseif (APPLE)
    set(CMAKE_FIND_LIBRARY_PREFIXES "lib" PARENT_SCOPE)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib" ".a" PARENT_SCOPE)
  elseif (WIN32)
    set(CMAKE_FIND_LIBRARY_PREFIXES "lib" PARENT_SCOPE)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll" PARENT_SCOPE)
  endif()
endfunction()



# ----- locate_matlab_root -----
#
# Attempt to find the path to the Matlab installation. If successful, sets
# the absolute path in the variable MATLAB_ROOT_DIR
function(locate_matlab_root)

  # --- UNIX/APPLE ---
  if (UNIX)
    # possible root locations, in order of likelihood
    set(SEARCH_DIRS_ /Applications /usr/local /opt/local /usr /opt)
    foreach (DIR_ ${SEARCH_DIRS_})
      file(GLOB MATLAB_ROOT_DIR_ ${DIR_}/*matlab*)
      if (MATLAB_ROOT_DIR_)
        # sort in order from highest to lowest
        # normally it's in the format MATLAB_R[20XX][A/B]
        # TODO: numerical rather than lexicographic sort. However,
        # CMake does not support floating-point MATH(EXPR ...) at this time.
        list(SORT MATLAB_ROOT_DIR_)
        list(REVERSE MATLAB_ROOT_DIR_)
        list(GET MATLAB_ROOT_DIR_ 0 MATLAB_ROOT_DIR_)
        set(MATLAB_ROOT_DIR ${MATLAB_ROOT_DIR_} PARENT_SCOPE)
        return()
      endif()
    endforeach()

  # --- WINDOWS ---
  elseif (WIN32)
    # 1. search the path environment variable
    find_program(MATLAB_ROOT_DIR_ matlab PATHS ENV PATH)
    if (MATLAB_ROOT_DIR_)
      # get the root directory from the full path
      # /path/to/matlab/rootdir/bin/matlab.exe
      get_filename_component(MATLAB_ROOT_DIR_ ${MATLAB_ROOT_DIR_} PATH)
      get_filename_component(MATLAB_ROOT_DIR_ ${MATLAB_ROOT_DIR_} PATH)
      set(MATLAB_ROOT_DIR ${MATLAB_ROOT_DIR_} PARENT_SCOPE)
      return()
    endif()

    # 2. search the registry
    # determine the available Matlab versions
    set(REG_EXTENSION_ "SOFTWARE\\Mathworks\\MATLAB")
    set(REG_ROOTS_ "HKEY_LOCAL_MACHINE" "HKEY_CURRENT_USER")
    foreach(REG_ROOT_ ${REG_ROOTS_})
      execute_process(COMMAND reg query "${REG_ROOT_}\\${REG_EXTENSION_}" OUTPUT_VARIABLE QUERY_RESPONSE_)
      if (QUERY_RESPONSE_)
        string(REGEX MATCHALL "[0-9]\\.[0-9]" VERSION_STRINGS_ ${QUERY_RESPONSE_})
        list(APPEND VERSIONS_ ${VERSION_STRINGS_})
      endif()
    endforeach()

    # select the highest version
    list(APPEND VERSIONS_ "0.0")
    list(SORT VERSIONS_)
    list(REVERSE VERSIONS_)
    list(GET VERSIONS_ 0 VERSION_)

    # request the MATLABROOT from the registry
    foreach(REG_ROOT_ ${REG_ROOTS_})
      get_filename_component(QUERY_RESPONSE_ [${REG_ROOT_}\\${REG_EXTENSION_}\\${VERSION_};MATLABROOT] ABSOLUTE)
      if (NOT ${QUERY_RESPONSE_} MATCHES "registry$")
        set(MATLAB_ROOT_DIR ${QUERY_RESPONSE_} PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endif()
endfunction()



# ----- locate_matlab_components -----
#
# Given a directory MATLAB_ROOT_DIR, attempt to find the Matlab components
# (include directory and libraries) under the root. If everything is found,
# sets the variable MATLAB_FOUND to TRUE
function(locate_matlab_components MATLAB_ROOT_DIR)
  # get the mex extension
  find_file(MATLAB_MEXEXT_SCRIPT_ NAMES mexext mexext.bat PATHS ${MATLAB_ROOT_DIR}/bin NO_DEFAULT_PATH)
  execute_process(COMMAND ${MATLAB_MEXEXT_SCRIPT_}
                  OUTPUT_VARIABLE MATLAB_MEXEXT_
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (NOT MATLAB_MEXEXT_)
    return()
  endif()

  # map the mexext to an architecture extension
  set(ARCHITECTURES_ "maci64" "maci" "glnxa64" "glnx64" "sol64" "sola64" "win32" "win64" )
  foreach(ARCHITECTURE_ ${ARCHITECTURES_})
    if(EXISTS ${MATLAB_ROOT_DIR}/bin/${ARCHITECTURE_})
      set(MATLAB_ARCH_ ${ARCHITECTURE_})
      break()
    endif()
  endforeach()

  # get the path to the libraries
  set(MATLAB_LIBRARY_DIRS_ ${MATLAB_ROOT_DIR}/bin/${MATLAB_ARCH_})

  # get the libraries
  set_libarch_prefix_suffix()
  find_library(MATLAB_LIB_MX_  mx  PATHS ${MATLAB_LIBRARY_DIRS_} NO_DEFAULT_PATH)
  find_library(MATLAB_LIB_MEX_ mex PATHS ${MATLAB_LIBRARY_DIRS_} NO_DEFAULT_PATH)
  find_library(MATLAB_LIB_MAT_ mat PATHS ${MATLAB_LIBRARY_DIRS_} NO_DEFAULT_PATH)
  set(MATLAB_LIBRARIES_ ${MATLAB_LIB_MX_} ${MATLAB_LIB_MEX_} ${MATLAB_LIB_MAT_})

  # get the include path
  find_path(MATLAB_INCLUDE_DIRS_ mex.h ${MATLAB_ROOT_DIR}/extern/include)

  # get the mex shell script
  find_program(MATLAB_MEX_SCRIPT_ NAMES mex mex.bat PATHS ${MATLAB_ROOT_DIR}/bin NO_DEFAULT_PATH)

  # get the Matlab executable
  find_program(MATLAB_BIN_ NAMES matlab PATHS ${MATLAB_ROOT_DIR}/bin NO_DEFAULT_PATH)

  # export into parent scope
  if (MATLAB_MEX_SCRIPT_ AND MATLAB_LIBRARIES_ AND MATLAB_INCLUDE_DIRS_)
    set(MATLAB_BIN          ${MATLAB_BIN_}          PARENT_SCOPE)
    set(MATLAB_MEX_SCRIPT   ${MATLAB_MEX_SCRIPT_}   PARENT_SCOPE)
    set(MATLAB_INCLUDE_DIRS ${MATLAB_INCLUDE_DIRS_} PARENT_SCOPE)
    set(MATLAB_LIBRARIES    ${MATLAB_LIBRARIES_}    PARENT_SCOPE)
    set(MATLAB_LIBRARY_DIRS ${MATLAB_LIBRARY_DIRS_} PARENT_SCOPE)
    set(MATLAB_MEXEXT       ${MATLAB_MEXEXT_}       PARENT_SCOPE)
    set(MATLAB_ARCH         ${MATLAB_ARCH_}         PARENT_SCOPE)
  endif()
endfunction()



# ----------------------------------------------------------------------------
# FIND MATLAB COMPONENTS
# ----------------------------------------------------------------------------
if (NOT MATLAB_FOUND)

  # attempt to find the Matlab root folder
  if (NOT MATLAB_ROOT_DIR)
    locate_matlab_root()
  endif()

  # given the matlab root folder, find the library locations
  if (MATLAB_ROOT_DIR)
    locate_matlab_components(${MATLAB_ROOT_DIR})
  endif()
  find_package_handle_standard_args(Matlab DEFAULT_MSG
                                           MATLAB_MEX_SCRIPT   MATLAB_INCLUDE_DIRS
                                           MATLAB_ROOT_DIR     MATLAB_LIBRARIES
                                           MATLAB_LIBRARY_DIRS MATLAB_MEXEXT
                                           MATLAB_ARCH         MATLAB_BIN)
endif()
