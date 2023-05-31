if(WIN32)
  try_compile(__VALID_DIRECTX
    "${OpenCV_BINARY_DIR}"
    "${OpenCV_SOURCE_DIR}/cmake/checks/directx.cpp"
    LINK_LIBRARIES d3d11
    OUTPUT_VARIABLE TRY_OUT
  )
  if(NOT __VALID_DIRECTX)
    message(STATUS "No support for DirectX (install Windows 8 SDK)")
    return()
  endif()
  try_compile(__VALID_DIRECTX_NV12
    "${OpenCV_BINARY_DIR}"
    "${OpenCV_SOURCE_DIR}/cmake/checks/directx.cpp"
    COMPILE_DEFINITIONS "-DCHECK_NV12"
    LINK_LIBRARIES d3d11
    OUTPUT_VARIABLE TRY_OUT
  )
  if(__VALID_DIRECTX_NV12)
    set(HAVE_DIRECTX_NV12 ON)
  else()
    message(STATUS "No support for DirectX NV12 format (install Windows 8 SDK)")
  endif()
  set(HAVE_DIRECTX ON)
  set(HAVE_D3D11 ON)
  set(HAVE_D3D10 ON)
  set(HAVE_D3D9 ON)

  if(HAVE_OPENCL AND WITH_OPENCL_D3D11_NV AND EXISTS "${OPENCL_INCLUDE_DIR}/CL/cl_d3d11_ext.h")
    set(HAVE_OPENCL_D3D11_NV ON)
  endif()

endif()
