# ----------------------------------------------------------------------------
#  Detect frameworks that may be used by 3rd-party libraries as well as OpenCV
# ----------------------------------------------------------------------------

# --- HPX ---
if(WITH_HPX)
  find_package(HPX REQUIRED)
  ocv_include_directories(${HPX_INCLUDE_DIRS})
  set(HAVE_HPX TRUE)
endif(WITH_HPX)

# --- GCD ---
if(APPLE AND NOT HAVE_TBB)
  set(HAVE_GCD 1)
else()
  set(HAVE_GCD 0)
endif()

# --- Concurrency ---
if(MSVC AND NOT HAVE_TBB AND NOT OPENCV_DISABLE_THREAD_SUPPORT)
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
    # NOTES:
    # 1. OpenMP_CXX_INCLUDE_DIRS is defined since CMake >= 3.16.0.
    # 2. OpenMP_CXX_LIBRARIES is defined since CMake >= 3.9.0.
    # 2. For gnu openmp (libgomp, e.g. on Linux), OpenMP_CXX_INCLUDE_DIRS is null
    #      regardless CMake version. Passing flag `-fopenmp` is enough.
    # 3. For clang openmp (libomp, e.g. on macOS), OpenMP_CXX_INCLUDE_DIRS is not 
    #      null and need to include header and link libomp in addition to passing flag.
    if(DEFINED OpenMP_CXX_INCLUDE_DIRS AND OpenMP_CXX_INCLUDE_DIRS)
      ocv_include_directories(${OpenMP_CXX_INCLUDE_DIRS})
    endif()
  endif()
  set(HAVE_OPENMP "${OPENMP_FOUND}")
endif()

ocv_clear_vars(HAVE_PTHREADS_PF)
if(WITH_PTHREADS_PF AND HAVE_PTHREAD)
  set(HAVE_PTHREADS_PF 1)
else()
  set(HAVE_PTHREADS_PF 0)
endif()
