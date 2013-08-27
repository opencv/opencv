# ----------------------------------------------------------------------------
#  Detect other 3rd-party performance and math libraries
# ----------------------------------------------------------------------------

# --- TBB ---
if(WITH_TBB)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectTBB.cmake")
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
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectCUDA.cmake")
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

# --- C= ---
if(WITH_CSTRIPES AND NOT HAVE_TBB)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectCStripes.cmake")
else()
  set(HAVE_CSTRIPES 0)
endif()

# --- OpenMP ---
if(NOT HAVE_TBB AND NOT HAVE_CSTRIPES)
  set(_fname "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/omptest.cpp")
  file(WRITE "${_fname}" "#ifndef _OPENMP\n#error\n#endif\nint main() { return 0; }\n")
  try_compile(HAVE_OPENMP "${CMAKE_BINARY_DIR}" "${_fname}")
  file(REMOVE "${_fname}")
else()
  set(HAVE_OPENMP 0)
endif()

# --- GCD ---
if(APPLE AND NOT HAVE_TBB AND NOT HAVE_CSTRIPES AND NOT HAVE_OPENMP)
  set(HAVE_GCD 1)
else()
  set(HAVE_GCD 0)
endif()

# --- Concurrency ---
if(MSVC AND NOT HAVE_TBB AND NOT HAVE_CSTRIPES AND NOT HAVE_OPENMP)
  set(_fname "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/concurrencytest.cpp")
  file(WRITE "${_fname}" "#if _MSC_VER < 1600\n#error\n#endif\nint main() { return 0; }\n")
  try_compile(HAVE_CONCURRENCY "${CMAKE_BINARY_DIR}" "${_fname}")
  file(REMOVE "${_fname}")
else()
  set(HAVE_CONCURRENCY 0)
endif()
