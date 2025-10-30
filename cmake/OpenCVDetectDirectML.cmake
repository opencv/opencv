if(WIN32)
  try_compile(__VALID_DIRECTML
    "${OpenCV_BINARY_DIR}"
    "${OpenCV_SOURCE_DIR}/cmake/checks/directml.cpp"
    LINK_LIBRARIES d3d12 dxcore directml
    OUTPUT_VARIABLE TRY_OUT
  )
  if(NOT __VALID_DIRECTML)
    message(STATUS "No support for DirectML. d3d12, dxcore, directml libs are required, first bundled with Windows SDK 10.0.19041.0.")
    return()
  endif()
  set(HAVE_DIRECTML ON)
endif()
