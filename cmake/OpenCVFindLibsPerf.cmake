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

    if(OPENCV_FORCE_IPP_EXCLUDE_LIBS
        OR (HAVE_IPP_ICV
            AND UNIX AND NOT ANDROID AND NOT APPLE
            AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        )
        AND NOT OPENCV_SKIP_IPP_EXCLUDE_LIBS
    )
      set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--exclude-libs,libippicv.a -Wl,--exclude-libs,libippiw.a ${CMAKE_SHARED_LINKER_FLAGS}")
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
if(WITH_EIGEN AND NOT HAVE_EIGEN)
  find_package(Eigen3 QUIET)

  if(Eigen3_FOUND)
    if(TARGET Eigen3::Eigen)
      # Use Eigen3 imported target if possible
      list(APPEND OPENCV_LINKER_LIBS Eigen3::Eigen)
      set(HAVE_EIGEN 1)
    else()
      if(DEFINED EIGEN3_INCLUDE_DIRS)
        set(EIGEN_INCLUDE_PATH ${EIGEN3_INCLUDE_DIRS})
        set(HAVE_EIGEN 1)
      elseif(DEFINED EIGEN3_INCLUDE_DIR)
        set(EIGEN_INCLUDE_PATH ${EIGEN3_INCLUDE_DIR})
        set(HAVE_EIGEN 1)
      endif()
    endif()
    if(HAVE_EIGEN)
      if(DEFINED EIGEN3_WORLD_VERSION)  # CMake module
        set(EIGEN_WORLD_VERSION ${EIGEN3_WORLD_VERSION})
        set(EIGEN_MAJOR_VERSION ${EIGEN3_MAJOR_VERSION})
        set(EIGEN_MINOR_VERSION ${EIGEN3_MINOR_VERSION})
      else()  # Eigen config file
        set(EIGEN_WORLD_VERSION ${EIGEN3_VERSION_MAJOR})
        set(EIGEN_MAJOR_VERSION ${EIGEN3_VERSION_MINOR})
        set(EIGEN_MINOR_VERSION ${EIGEN3_VERSION_PATCH})
      endif()
    endif()
  endif()

  if(NOT HAVE_EIGEN)
    if(NOT EIGEN_INCLUDE_PATH OR NOT EXISTS "${EIGEN_INCLUDE_PATH}")
      set(__find_paths "")
      set(__find_path_extra_options "")
      if(NOT CMAKE_CROSSCOMPILING)
        list(APPEND __find_paths /opt)
      endif()
      if(DEFINED ENV{EIGEN_ROOT})
        set(__find_paths "$ENV{EIGEN_ROOT}/include")
        list(APPEND __find_path_extra_options NO_DEFAULT_PATH)
      else()
        set(__find_paths ENV ProgramFiles ENV ProgramW6432)
      endif()
      find_path(EIGEN_INCLUDE_PATH "Eigen/Core"
                PATHS ${__find_paths}
                PATH_SUFFIXES include/eigen3 include/eigen2 Eigen/include/eigen3 Eigen/include/eigen2
                DOC "The path to Eigen3/Eigen2 headers"
                ${__find_path_extra_options}
      )
    endif()
    if(EIGEN_INCLUDE_PATH AND EXISTS "${EIGEN_INCLUDE_PATH}")
      ocv_parse_header("${EIGEN_INCLUDE_PATH}/Eigen/src/Core/util/Macros.h" EIGEN_VERSION_LINES EIGEN_WORLD_VERSION EIGEN_MAJOR_VERSION EIGEN_MINOR_VERSION)
      set(HAVE_EIGEN 1)
    endif()
  endif()
endif()
if(HAVE_EIGEN)
  if(EIGEN_INCLUDE_PATH AND EXISTS "${EIGEN_INCLUDE_PATH}")
    ocv_include_directories(SYSTEM ${EIGEN_INCLUDE_PATH})
  endif()
endif()

# --- Clp ---
# Ubuntu: sudo apt-get install coinor-libclp-dev coinor-libcoinutils-dev
ocv_clear_vars(HAVE_CLP)
if(WITH_CLP)
  if(UNIX)
    ocv_check_modules(CLP clp)
    if(CLP_FOUND)
      set(HAVE_CLP TRUE)
      if(NOT ${CLP_INCLUDE_DIRS} STREQUAL "")
        ocv_include_directories(${CLP_INCLUDE_DIRS})
      endif()
      list(APPEND OPENCV_LINKER_LIBS ${CLP_LIBRARIES})
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
