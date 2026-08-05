# Detect and enable g2o (3rdparty/g2o).
# Sets HAVE_G2O, G2O_LIBRARIES, G2O_INCLUDE_DIRS.

if(NOT WITH_G2O)
  return()
endif()

if(NOT HAVE_EIGEN)
  message(STATUS "G2O: skipped — Eigen3 is required but not found")
  return()
endif()

add_subdirectory("${OpenCV_SOURCE_DIR}/3rdparty/g2o")

if(HAVE_G2O)
  message(STATUS "G2O: static library ready")
else()
  message(STATUS "G2O: build did not produce a library")
endif()
