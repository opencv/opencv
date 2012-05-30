# ----------------------------------------------------------------------------
#  Detect other 3rd-party performance and math libraries
# ----------------------------------------------------------------------------

# --- TBB ---
if(WITH_TBB)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectTBB.cmake" REQUIRED)
endif(WITH_TBB)

# --- IPP ---
ocv_clear_vars(IPP_FOUND)
if(WITH_IPP)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindIPP.cmake")
  if(IPP_FOUND)
    add_definitions(-DHAVE_IPP)
    ocv_include_directories(${IPP_INCLUDE_DIRS})
    link_directories(${IPP_LIBRARY_DIRS})
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${IPP_LIBRARIES})
  endif()
endif(WITH_IPP)

# --- CUDA ---
if(WITH_CUDA)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectCUDA.cmake" REQUIRED)
endif(WITH_CUDA)

# --- Eigen ---
if(WITH_EIGEN)
  find_path(EIGEN_INCLUDE_PATH "Eigen/Core"
            PATHS /usr/local /opt /usr ENV ProgramFiles ENV ProgramW6432
            PATH_SUFFIXES include/eigen3 include/eigen2 Eigen/include/eigen3 Eigen/include/eigen2
            DOC "The path to Eigen3/Eigen2 headers")

  if(EIGEN_INCLUDE_PATH)
    ocv_include_directories(${EIGEN_INCLUDE_PATH})
    ocv_parse_header("${EIGEN_INCLUDE_PATH}/Eigen/src/Core/util/Macros.h" EIGEN_VERSION_LINES EIGEN_WORLD_VERSION EIGEN_MAJOR_VERSION EIGEN_MINOR_VERSION)
    set(HAVE_EIGEN 1)
  endif()
endif(WITH_EIGEN)
