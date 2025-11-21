# Search for OpenBLAS library
#
# OpenBLAS is packaged differently across Linux distributions:
# - Fedora/RHEL: Headers in /usr/include/openblas/
#                Libraries: libopenblas.so (serial), libopenblasp.so (threaded)
# - Debian/Ubuntu: Headers in /usr/include/
#                  Library: libopenblas.so (typically threaded via symlink)
#
# Detection strategy:
# 1. Try find_package(OpenBLAS) if available
# 2. Check environment variables (OpenBLAS, OpenBLAS_HOME)
# 3. Search system-wide standard locations
#
# On return:
#   OpenBLAS_FOUND
#   OpenBLAS_LIBRARIES
#   OpenBLAS_INCLUDE_DIRS

if(NOT OpenBLAS_FOUND AND NOT SKIP_OPENBLAS_PACKAGE)
  find_package(OpenBLAS QUIET)
  if(OpenBLAS_FOUND)
    message(STATUS "Found OpenBLAS package")
  endif()
endif()

if(NOT OpenBLAS_FOUND)
  # Search using environment hints (OpenBLAS or OpenBLAS_HOME)
  # Prioritize threaded library for performance on RHEL systems
  find_library(OpenBLAS_LIBRARIES 
    NAMES openblasp openblas 
    PATHS ENV "OpenBLAS" ENV "OpenBLAS_HOME" 
    PATH_SUFFIXES "lib" 
    NO_DEFAULT_PATH)
  
  # Support both namespaced (RHEL) and standard (Debian) header layouts
  find_path(OpenBLAS_INCLUDE_DIRS 
    NAMES cblas.h 
    PATHS ENV "OpenBLAS" ENV "OpenBLAS_HOME" 
    PATH_SUFFIXES "include/openblas" "include" 
    NO_DEFAULT_PATH)
  
  find_path(OpenBLAS_LAPACKE_DIR 
    NAMES lapacke.h 
    PATHS "${OpenBLAS_INCLUDE_DIRS}" ENV "OpenBLAS" ENV "OpenBLAS_HOME" 
    PATH_SUFFIXES "include" 
    NO_DEFAULT_PATH)
  
  if(OpenBLAS_LIBRARIES AND OpenBLAS_INCLUDE_DIRS)
    message(STATUS "Found OpenBLAS using environment hint")
    set(OpenBLAS_FOUND TRUE)
  else()
    ocv_clear_vars(OpenBLAS_LIBRARIES OpenBLAS_INCLUDE_DIRS)
  endif()
endif()

if(NOT OpenBLAS_FOUND)
  # System-wide search
  # Prefer openblasp (threaded on RHEL) over openblas (serial on RHEL, threaded symlink on Debian)
  find_library(OpenBLAS_LIBRARIES NAMES openblasp openblas)

  # Try namespaced location first (RHEL: /usr/include/openblas/)
  find_path(OpenBLAS_INCLUDE_DIRS NAMES cblas.h PATH_SUFFIXES openblas)
  
  # Fall back to standard location (Debian: /usr/include/)
  if(NOT OpenBLAS_INCLUDE_DIRS)
    find_path(OpenBLAS_INCLUDE_DIRS NAMES cblas.h)
  endif()

  find_path(OpenBLAS_LAPACKE_DIR NAMES lapacke.h PATHS "${OpenBLAS_INCLUDE_DIRS}")
  
  if(OpenBLAS_LIBRARIES AND OpenBLAS_INCLUDE_DIRS)
    message(STATUS "Found OpenBLAS in the system")
    set(OpenBLAS_FOUND TRUE)
  else()
    ocv_clear_vars(OpenBLAS_LIBRARIES OpenBLAS_INCLUDE_DIRS)
  endif()
endif()

if(OpenBLAS_FOUND)
  if(OpenBLAS_LAPACKE_DIR)
    set(OpenBLAS_INCLUDE_DIRS "${OpenBLAS_INCLUDE_DIRS};${OpenBLAS_LAPACKE_DIR}")
  endif()
  message(STATUS "OpenBLAS_LIBRARIES=${OpenBLAS_LIBRARIES}")
  message(STATUS "OpenBLAS_INCLUDE_DIRS=${OpenBLAS_INCLUDE_DIRS}")
endif()

mark_as_advanced(OpenBLAS_LIBRARIES OpenBLAS_INCLUDE_DIRS OpenBLAS_LAPACKE_DIR)
