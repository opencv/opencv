if(WIN32)
  try_compile(__VALID_DIRECTX
    "${OpenCV_BINARY_DIR}"
    "${OpenCV_SOURCE_DIR}/cmake/checks/directx.cpp"
    OUTPUT_VARIABLE TRY_OUT
  )
  if(NOT __VALID_DIRECTX)
    return()
  endif()
  set(HAVE_DIRECTX ON)
  set(HAVE_D3D11 ON)
  set(HAVE_D3D10 ON)
  set(HAVE_D3D9 ON)
endif()
