# ----- Find Matlab/Octave -----
#
# OpenCVFindMatlab.cmake attempts to locate the install path of Matlab in order
# to extract the mex headers, libraries and shell scripts. If found 
# successfully, the following variables will be defined
#
#   MATLAB_FOUND:       true/false
#   MATLAB_ROOT_DIR:    Root of Matlab installation
#   MATLAB_MEX_SCRIPT   The mex script used to compile mex files
#   MATLAB_INCLUDE_DIRS Path to "mex.h"
#   MATLAB_LIBRARY_DIRS Path to mex and matrix libraries
#   MATLAB_LIBS         The Matlab libs, usually mx, mex, mat
#   MATLAB_MEXEXT       The mex library extension. It will be one of:
#                         mexwin32, mexwin64,  mexglx, mexa64, mexmac, 
#                         mexmaci,  mexmaci64, mexsol, mexs64
#   MATLAB_ARCH         The installation architecture. It is simply
#                       the MEXEXT with the preceding "mex" removed
#
# There doesn't appear to be an elegant way to detect all versions of Matlab
# across different platforms. If you know the matlab path and want to avoid
# the search, you can define the path to the Matlab root when invoking cmake:
#
#   cmake -DMATLAB_ROOT_DIR='/PATH/TO/ROOT_DIR' ..



# ----- locate_matlab_root -----
#
# Attempt to find the path to the Matlab installation. If successful, sets
# the absolute path in the variable MATLAB_ROOT_DIR
MACRO(locate_matlab_root)
  # --- LINUX ---
  if (UNIX AND NOT APPLE)
    # possible root locations, in order of likelihood
    set(SEARCH_DIRS_ /usr/local /opt/local /usr /opt)
    foreach (DIR_ ${SEARCH_DIRS_})
      file(GLOB MATLAB_ROOT_DIR ${DIR_}/*matlab*)
      if (MATLAB_ROOT_DIR)
        break()
      endif()
    endforeach()

  # --- APPLE ---
  elseif (APPLE)
    # possible root locations, in order of likelihood
    set(SEARCH_DIRS_ /Applications /usr/local /opt/local /usr /opt)
    foreach (DIR_ ${SEARCH_DIRS_})
      file(GLOB MATLAB_ROOT_DIR ${DIR_}/*matlab*)
      if (MATLAB_ROOT_DIR)
        break()
      endif()
    endforeach()

  # --- WINDOWS ---
  elseif (WIN32)
  endif()
  # unset local variables
  unset(SEARCH_DIRS_)
  unset(DIR_)
endMACRO()



# ----- locate_matlab_components -----
#
# Given a directory MATLAB_ROOT_DIR, attempt to find the Matlab components
# (include directory and libraries) under the root. If everything is found,
# sets the variable MATLAB_FOUND to TRUE
MACRO(locate_matlab_components MATLAB_ROOT_DIR)
  if (UNIX)
    # get the mex extension
    execute_process(COMMAND ${MATLAB_ROOT_DIR}/bin/mexext OUTPUT_VARIABLE MATLAB_MEXEXT)
    string(STRIP ${MATLAB_MEXEXT} MATLAB_MEXEXT)
    string(REPLACE "mex" "" MATLAB_ARCH ${MATLAB_MEXEXT})

    # get the path to the libraries
    set(MATLAB_LIBRARY_DIRS ${MATLAB_ROOT_DIR}/bin/${MATLAB_ARCH})

    # get the libraries
    find_library(MATLAB_LIB_MX  mx  PATHS ${MATLAB_LIBRARY_DIRS} NO_DEFAULT_PATH)
    find_library(MATLAB_LIB_MEX mex PATHS ${MATLAB_LIBRARY_DIRS} NO_DEFAULT_PATH)
    find_library(MATLAB_LIB_MAT mat PATHS ${MATLAB_LIBRARY_DIRS} NO_DEFAULT_PATH)
    set(MATLAB_LIBS ${MATLAB_LIB_MX} ${MATLAB_LIB_MEX} ${MATLAB_LIB_MAT})

    # get the include path
    find_path(MATLAB_INCLUDE_DIRS mex.h ${MATLAB_ROOT_DIR}/extern/include) 

    # get the mex shell script
    find_file(MATLAB_MEX_SCRIPT NAMES mex mex.bat PATHS ${MATLAB_ROOT_DIR}/bin NO_DEFAULT_PATH)
    
  elseif (WIN32)
  endif()

  # verify that everything has been found successfully
  if (MATLAB_LIB_MX AND MATLAB_LIB_MEX AND MATLAB_LIB_MAT AND MATLAB_INCLUDE_DIRS AND MATLAB_MEX_SCRIPT)
    set(MATLAB_FOUND TRUE)
  else()
    unset(MATLAB_LIB_MX)
    unset(MATLAB_LIB_MEX)
    unset(MATLAB_LIB_MAT)
    unset(MATLAB_LIBS)
    unset(MATLAB_LIBRARY_DIRS)
    unset(MATLAB_INCLUDE_DIRS)
    unset(MATLAB_ROOT_DIR)
    unset(MATLAB_MEXEXT)
    unset(MATLAB_MEX_SCRIPT)
    unset(MATLAB_ARCH)
  endif()
endMACRO()


# ----------------------------------------------------------------------------
# FIND MATLAB COMPONENTS
# ----------------------------------------------------------------------------
if (NOT MATLAB_FOUND)
 
  # guilty until proven innocent
  set(MATLAB_FOUND FALSE)

  # attempt to find the Matlab root folder
  if (NOT MATLAB_ROOT_DIR)
    locate_matlab_root()
  endif()

  # given the matlab root folder, find the library locations 
  if (MATLAB_ROOT_DIR)
    locate_matlab_components(${MATLAB_ROOT_DIR})
  endif()

  # if Matlab was not found, unset the local variables
  if (NOT MATLAB_FOUND)
    unset (MATLAB_ROOT_DIR)
  endif()
endif()
