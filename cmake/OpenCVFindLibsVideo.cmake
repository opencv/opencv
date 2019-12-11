# --- Extra HighGUI and VideoIO libs on Windows ---
if(WIN32)
  list(APPEND HIGHGUI_LIBRARIES comctl32 gdi32 ole32 setupapi ws2_32)
endif(WIN32)

# --- VA & VA_INTEL ---
if(WITH_VA_INTEL)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindVA_INTEL.cmake")
  if(VA_INTEL_IOCL_INCLUDE_DIR)
    ocv_include_directories(${VA_INTEL_IOCL_INCLUDE_DIR})
  endif()
  set(WITH_VA YES)
endif(WITH_VA_INTEL)

if(WITH_VA)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindVA.cmake")
  if(VA_INCLUDE_DIR)
    ocv_include_directories(${VA_INCLUDE_DIR})
  endif()
endif(WITH_VA)
