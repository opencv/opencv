include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectTBB.cmake")

if(HAVE_TBB)
  ocv_add_external_target(tbb "" "tbb" "HAVE_TBB=1")
endif()
