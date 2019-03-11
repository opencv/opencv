# ----------------------------------------------------------------------------
#  Detect other 3rd-party performance and math libraries
# ----------------------------------------------------------------------------

# --- TBB ---
if(WITH_TBB)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectTBB.cmake")
endif(WITH_TBB)

# --- IPP ---
if(WITH_IPP)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindIPP.cmake")
  if(HAVE_IPP)
    include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindIPPIW.cmake")
    if(HAVE_IPP_IW)
      ocv_include_directories(${IPP_IW_INCLUDES})
      list(APPEND OPENCV_LINKER_LIBS ${IPP_IW_LIBRARIES})
    endif()
    ocv_include_directories(${IPP_INCLUDE_DIRS})
    list(APPEND OPENCV_LINKER_LIBS ${IPP_LIBRARIES})

    # Details: #10229
    if(ANDROID AND NOT OPENCV_SKIP_ANDROID_IPP_FIX_1)
      set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--exclude-libs,libippicv.a -Wl,--exclude-libs,libippiw.a ${CMAKE_SHARED_LINKER_FLAGS}")
    elseif(ANDROID AND NOT OPENCV_SKIP_ANDROID_IPP_FIX_2)
      set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-Bsymbolic ${CMAKE_SHARED_LINKER_FLAGS}")
    endif()
  endif()
endif()

# --- CUDA ---
if(WITH_CUDA)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectCUDA.cmake")
  if(NOT HAVE_CUDA)
    message(WARNING "OpenCV is not able to find/configure CUDA SDK (required by WITH_CUDA).
CUDA support will be disabled in OpenCV build.
To eliminate this warning remove WITH_CUDA=ON CMake configuration option.
")
  endif()
endif(WITH_CUDA)

# --- Eigen ---
if(WITH_EIGEN)
  find_path(EIGEN_INCLUDE_PATH "Eigen/Core"
            PATHS /usr/local /opt /usr $ENV{EIGEN_ROOT}/include ENV ProgramFiles ENV ProgramW6432
            PATH_SUFFIXES include/eigen3 include/eigen2 Eigen/include/eigen3 Eigen/include/eigen2
            DOC "The path to Eigen3/Eigen2 headers"
            CMAKE_FIND_ROOT_PATH_BOTH)

  if(EIGEN_INCLUDE_PATH)
    ocv_include_directories(${EIGEN_INCLUDE_PATH})
    ocv_parse_header("${EIGEN_INCLUDE_PATH}/Eigen/src/Core/util/Macros.h" EIGEN_VERSION_LINES EIGEN_WORLD_VERSION EIGEN_MAJOR_VERSION EIGEN_MINOR_VERSION)
    set(HAVE_EIGEN 1)
  endif()
endif(WITH_EIGEN)

# --- Clp ---
# Ubuntu: sudo apt-get install coinor-libclp-dev coinor-libcoinutils-dev
ocv_clear_vars(HAVE_CLP)
if(WITH_CLP)
  if(UNIX)
    PKG_CHECK_MODULES(CLP clp)
    if(CLP_FOUND)
      set(HAVE_CLP TRUE)
      if(NOT ${CLP_INCLUDE_DIRS} STREQUAL "")
        ocv_include_directories(${CLP_INCLUDE_DIRS})
      endif()
      link_directories(${CLP_LIBRARY_DIRS})
      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${CLP_LIBRARIES})
    endif()
  endif()

  if(NOT CLP_FOUND)
    find_path(CLP_INCLUDE_PATH "coin"
              PATHS "/usr/local/include" "/usr/include" "/opt/include"
              DOC "The path to Clp headers")
    if(CLP_INCLUDE_PATH)
      ocv_include_directories(${CLP_INCLUDE_PATH} "${CLP_INCLUDE_PATH}/coin")
      get_filename_component(_CLP_LIBRARY_DIR "${CLP_INCLUDE_PATH}/../lib" ABSOLUTE)
      set(CLP_LIBRARY_DIR "${_CLP_LIBRARY_DIR}" CACHE PATH "Full path of Clp library directory")
      link_directories(${CLP_LIBRARY_DIR})
      if(UNIX)
        set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} Clp CoinUtils m)
      else()
        if(MINGW)
            set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} Clp CoinUtils)
        else()
            set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} libClp libCoinUtils)
        endif()
      endif()
      set(HAVE_CLP TRUE)
    endif()
  endif()
endif(WITH_CLP)
