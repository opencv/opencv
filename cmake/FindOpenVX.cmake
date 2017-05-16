ocv_clear_vars(HAVE_OPENVX)

set(OPENVX_ROOT "" CACHE PATH "OpenVX install directory")
set(OPENVX_LIB_CANDIDATES "openvx;vxu" CACHE STRING "OpenVX library candidates list")

function(find_openvx_libs _found)
  foreach(one ${OPENVX_LIB_CANDIDATES})
    find_library(OPENVX_${one}_LIBRARY ${one} PATHS "${OPENVX_ROOT}/lib" "${OPENVX_ROOT}/bin")
    if(OPENVX_${one}_LIBRARY)
      list(APPEND _list ${OPENVX_${one}_LIBRARY})
    endif()
  endforeach()
  set(${_found} ${_list} PARENT_SCOPE)
endfunction()

if(OPENVX_ROOT)
  find_path(OPENVX_INCLUDE_DIR "VX/vx.h" PATHS "${OPENVX_ROOT}/include" DOC "OpenVX include path")
  if(NOT DEFINED OPENVX_LIBRARIES)
    find_openvx_libs(found)
    if(found)
      set(OPENVX_LIBRARIES "${found}" CACHE STRING "OpenVX libraries")
    endif()
  endif()
endif()

if(OPENVX_INCLUDE_DIR AND OPENVX_LIBRARIES)
  set(HAVE_OPENVX TRUE)

  try_compile(OPENVX_RENAMED_REF
      "${OpenCV_BINARY_DIR}"
      "${OpenCV_SOURCE_DIR}/cmake/checks/openvx_refenum_test.cpp"
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${OPENVX_INCLUDE_DIR}"
      LINK_LIBRARIES ${OPENVX_LIBRARIES}
      OUTPUT_VARIABLE OUTPUT
  )
  if(OPENVX_RENAMED_REF)
      add_definitions(-DIVX_RENAMED_REFS=1)
      message(STATUS "OpenVX: Checking reference attribute name convention... New")
  else()
      message(STATUS "OpenVX: Checking reference attribute name convention... Old")
  endif()
endif()

if(NOT HAVE_OPENVX)
  ocv_clear_vars(HAVE_OPENVX OPENVX_LIBRARIES OPENVX_INCLUDE_DIR)
endif()
