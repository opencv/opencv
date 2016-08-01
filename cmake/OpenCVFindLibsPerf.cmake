# ----------------------------------------------------------------------------
#  Detect other 3rd-party performance and math libraries
# ----------------------------------------------------------------------------

# --- Lapack ---
if(WITH_LAPACK)
  find_package(LAPACK)
  if(LAPACK_FOUND)
    find_path(LAPACKE_INCLUDE_DIR "lapacke.h")
    if(LAPACKE_INCLUDE_DIR)
      find_path(CBLAS_INCLUDE_DIR "cblas.h")
      if(CBLAS_INCLUDE_DIR)
        set(HAVE_LAPACK 1)
        ocv_include_directories(${LAPACKE_INCLUDE_DIR} ${CBLAS_INCLUDE_DIR})
        list(APPEND OPENCV_LINKER_LIBS ${LAPACK_LIBRARIES})
      endif()
    endif()
  endif()
endif()

# --- TBB ---
if(WITH_TBB)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectTBB.cmake")
endif(WITH_TBB)

# --- IPP ---
if(WITH_IPP)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindIPP.cmake")
  if(HAVE_IPP)
    ocv_include_directories(${IPP_INCLUDE_DIRS})
    list(APPEND OPENCV_LINKER_LIBS ${IPP_LIBRARIES})
  endif()
endif()

# --- IPP Async ---

if(WITH_IPP_A)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindIPPAsync.cmake")
  if(IPP_A_INCLUDE_DIR AND IPP_A_LIBRARIES)
    ocv_include_directories(${IPP_A_INCLUDE_DIR})
    link_directories(${IPP_A_LIBRARIES})
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} ${IPP_A_LIBRARIES})
   endif()
endif(WITH_IPP_A)

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

# --- GCD ---
if(APPLE AND NOT HAVE_TBB AND NOT HAVE_CSTRIPES)
  set(HAVE_GCD 1)
else()
  set(HAVE_GCD 0)
endif()

# --- Concurrency ---
if(MSVC AND NOT HAVE_TBB AND NOT HAVE_CSTRIPES)
  set(_fname "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/concurrencytest.cpp")
  file(WRITE "${_fname}" "#if _MSC_VER < 1600\n#error\n#endif\nint main() { return 0; }\n")
  try_compile(HAVE_CONCURRENCY "${CMAKE_BINARY_DIR}" "${_fname}")
  file(REMOVE "${_fname}")
else()
  set(HAVE_CONCURRENCY 0)
endif()

# --- OpenMP ---
if(WITH_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
  set(HAVE_OPENMP "${OPENMP_FOUND}")
endif()

if(NOT MSVC AND NOT DEFINED HAVE_PTHREADS)
  set(_fname "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/pthread_test.cpp")
  file(WRITE "${_fname}" "#include <pthread.h>\nint main() { (void)pthread_self(); return 0; }\n")
  try_compile(HAVE_PTHREADS "${CMAKE_BINARY_DIR}" "${_fname}")
  file(REMOVE "${_fname}")
endif()

ocv_clear_vars(HAVE_PTHREADS_PF)
if(WITH_PTHREADS_PF)
  set(HAVE_PTHREADS_PF ${HAVE_PTHREADS})
else()
  set(HAVE_PTHREADS_PF 0)
endif()
