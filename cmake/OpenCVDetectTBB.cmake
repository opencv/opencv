# Search TBB library (4.1 - 4.4, 2017)
#
# Own TBB (3rdparty/tbb):
# - set cmake option BUILD_TBB to ON
#
# External TBB (from system):
# - Fedora: install 'tbb-devel' package
# - Ubuntu: install 'libtbb-dev' package
#
# External TBB (from official site):
# - Linux/OSX:
#   - in tbbvars.sh replace 'SUBSTITUTE_INSTALL_DIR_HERE' with absolute path to TBB dir
#   - in terminal run 'source tbbvars.sh intel64 linux' ('source tbbvars.sh' in OSX)
# - Windows:
#   - in terminal run 'tbbvars.bat intel64 vs2015'
#
# External TBB (intel oneApi):
# - Linux/OSX:
#
# - Windows:
#   - in terminal run 'setvars.bat intel64 vs2015'
#
# Return:
#   - HAVE_TBB set to TRUE
#   - "tbb" target exists and added to OPENCV_LINKER_LIBS

function(ocv_tbb_cmake_guess _found)
    find_package(TBB QUIET COMPONENTS tbb PATHS "$ENV{TBBROOT}/cmake" "$ENV{TBBROOT}/lib/cmake/tbb")
    if(TBB_FOUND)
      if(NOT TARGET TBB::tbb)
        message(WARNING "No TBB::tbb target found!")
        return()
      endif()
      get_target_property(_lib TBB::tbb IMPORTED_LOCATION_RELEASE)
      message(STATUS "Found TBB (cmake): ${_lib}")
      get_target_property(_inc TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
      add_library(tbb INTERFACE IMPORTED)
      set_target_properties(tbb PROPERTIES
        INTERFACE_LINK_LIBRARIES TBB::tbb
      )
      ocv_tbb_read_version("${_inc}" tbb)
      set(${_found} TRUE PARENT_SCOPE)
    endif()
endfunction()

function(ocv_tbb_env_verify)
  if (NOT "$ENV{TBBROOT}" STREQUAL "")
    # check that library and include dir are inside TBBROOT location
    get_filename_component(_root "$ENV{TBBROOT}" ABSOLUTE)
    get_filename_component(_lib "${TBB_ENV_LIB}" ABSOLUTE)
    get_filename_component(_inc "${TBB_ENV_INCLUDE}" ABSOLUTE)
    string(FIND "${_lib}" "${_root}" _lib_pos)
    string(FIND "${_inc}" "${_root}" _inc_pos)
    if (NOT (_lib_pos EQUAL 0 AND _inc_pos EQUAL 0))
      message(SEND_ERROR
        "Possible issue with TBB detection - TBBROOT is set, "
        "but library/include path is not inside it:\n "
        "TBBROOT: $ENV{TBBROOT}\n "
        "(absolute): ${_root}\n "
        "INCLUDE: ${_inc}\n "
        "LIBRARY: ${_lib}\n")
    endif()
  endif()
endfunction()

function(ocv_tbb_env_one_api_guess _found)
  if(NOT TBB_DIR)
    file(TO_CMAKE_PATH "$ENV{ONEAPI_ROOT}/tbb/latest" path)
    find_path(TBB_DIR include/tbb/tbb.h PATHS ${path})
  endif()
  if(TBB_DIR AND EXISTS "${TBB_DIR}/include/tbb/tbb.h")
    find_path(TBB_ENV_INCLUDE tbb/tbb.h PATHS ${TBB_DIR}/include)
    #determine arch
    if(CMAKE_CXX_SIZEOF_DATA_PTR EQUAL 8)
      set(TBB_ARCH_LIST "intel64")
    else()
      set(TBB_ARCH_LIST "ia32")
    endif()

    set(tbb_lib_find_paths ${TBB_LIB_FIND_PATHS} ${TBB_DIR}/lib)
    foreach(TBB_ARCH ${TBB_ARCH_LIST})
      list(APPEND tbb_lib_find_paths
        ${TBB_DIR}/lib/${TBB_ARCH}/vc_mt
      )
    endforeach()

    file(GLOB tbb_lib_list LIST_DIRECTORIES false ${tbb_lib_find_paths}/*.lib)
	file(GLOB tbb_debug_lib_list LIST_DIRECTORIES false ${tbb_lib_find_paths}/*_debug.lib)
	list(REMOVE_ITEM tbb_lib_list ${tbb_debug_lib_list})
	list(REMOVE_DUPLICATES tbb_lib_list)
    set(SUB_LIB_LIST "")
    foreach(lib ${tbb_lib_list})
      set(lib_var_name TBB_LIBRARY_${lib})
      find_library(${lib_var_name} NAMES ${lib} HINTS ${tbb_lib_find_paths})
      mark_as_advanced(${lib_var_name})
      set(lib_var_name_debug TBB_LIBRARY_${lib}_DEBUG)
      find_library(${lib_var_name_debug} NAMES ${lib}_debug HINTS ${tbb_lib_find_paths})
      mark_as_advanced(${lib_var_name_debug})
      if(NOT ${lib_var_name})
        return()
      endif()
      list(APPEND SUB_LIB_LIST tbb::${lib})
      add_library(tbb::${lib} UNKNOWN IMPORTED)
      set_target_properties(tbb::${lib} PROPERTIES
        IMPORTED_LOCATION "${${lib_var_name}}"
      )
      if (${lib_var_name_debug})
        set_target_properties(tbb::${lib} PROPERTIES
          IMPORTED_LOCATION_DEBUG "${${lib_var_name_debug}}"
        )
      endif()
    endforeach()
    ocv_tbb_read_version_one_api("${TBB_ENV_INCLUDE}")
    add_library(tbb INTERFACE IMPORTED)
    set_target_properties(tbb PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${TBB_ENV_INCLUDE}"
    )
    set_property(TARGET tbb
      PROPERTY
      INTERFACE_LINK_LIBRARIES ${SUB_LIB_LIST}
    )
    # workaround: system TBB library is used for linking instead of provided
    if(CV_GCC)
      set_target_properties(tbb PROPERTIES INTERFACE_LINK_LIBRARIES "-L${tbb_lib_find_paths}")
    endif()
    message(STATUS "Found TBB (oneApi): ${TBB_DIR}")
    set(${_found} TRUE PARENT_SCOPE)
  endif()
endfunction()

function(ocv_tbb_env_guess _found)
  find_path(TBB_ENV_INCLUDE NAMES "tbb/tbb.h" PATHS ENV CPATH NO_DEFAULT_PATH)
  find_path(TBB_ENV_INCLUDE NAMES "tbb/tbb.h")
  find_library(TBB_ENV_LIB NAMES "tbb" PATHS ENV LIBRARY_PATH NO_DEFAULT_PATH)
  find_library(TBB_ENV_LIB NAMES "tbb")
  find_library(TBB_ENV_LIB_DEBUG NAMES "tbb_debug" PATHS ENV LIBRARY_PATH NO_DEFAULT_PATH)
  find_library(TBB_ENV_LIB_DEBUG NAMES "tbb_debug")
  if (TBB_ENV_INCLUDE AND (TBB_ENV_LIB OR TBB_ENV_LIB_DEBUG))
    ocv_tbb_env_verify()
    add_library(tbb UNKNOWN IMPORTED)
    set_target_properties(tbb PROPERTIES
      IMPORTED_LOCATION "${TBB_ENV_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${TBB_ENV_INCLUDE}"
    )
    if (TBB_ENV_LIB_DEBUG)
      set_target_properties(tbb PROPERTIES
        IMPORTED_LOCATION_DEBUG "${TBB_ENV_LIB_DEBUG}"
      )
    endif()
    # workaround: system TBB library is used for linking instead of provided
    if(CV_GCC)
      get_filename_component(_dir "${TBB_ENV_LIB}" DIRECTORY)
      set_target_properties(tbb PROPERTIES INTERFACE_LINK_LIBRARIES "-L${_dir}")
    endif()
    ocv_tbb_read_version("${TBB_ENV_INCLUDE}" tbb)
    message(STATUS "Found TBB (env): ${TBB_ENV_LIB}")
    set(${_found} TRUE PARENT_SCOPE)
  endif()
endfunction()

function(ocv_tbb_read_version _path _tgt)
  find_file(TBB_VER_FILE oneapi/tbb/version.h "${_path}" NO_DEFAULT_PATH CMAKE_FIND_ROOT_PATH_BOTH)
  find_file(TBB_VER_FILE tbb/tbb_stddef.h "${_path}" NO_DEFAULT_PATH CMAKE_FIND_ROOT_PATH_BOTH)
  ocv_parse_header("${TBB_VER_FILE}" TBB_VERSION_LINES TBB_VERSION_MAJOR TBB_VERSION_MINOR TBB_INTERFACE_VERSION CACHE)
endfunction()

function(ocv_tbb_read_version_one_api _path)
  find_file(TBB_VER_FILE oneapi/tbb/version.h "${_path}" NO_DEFAULT_PATH CMAKE_FIND_ROOT_PATH_BOTH)
  ocv_parse_header("${TBB_VER_FILE}" TBB_VERSION_LINES TBB_VERSION_MAJOR TBB_VERSION_MINOR TBB_INTERFACE_VERSION CACHE)
endfunction()

#=====================================================================

if(BUILD_TBB)
  add_subdirectory("${OpenCV_SOURCE_DIR}/3rdparty/tbb")
  if(NOT TARGET tbb)
    return()
  endif()
  set(HAVE_TBB TRUE)
endif()

if(NOT HAVE_TBB)
  ocv_tbb_cmake_guess(HAVE_TBB)
endif()

if(NOT HAVE_TBB)
  ocv_tbb_env_guess(HAVE_TBB)
endif()

if (NOT HAVE_TBB)
  ocv_tbb_env_one_api_guess(HAVE_TBB)
endif()

if(TBB_INTERFACE_VERSION LESS 6000) # drop support of versions < 4.0
  set(HAVE_TBB FALSE)
endif()
