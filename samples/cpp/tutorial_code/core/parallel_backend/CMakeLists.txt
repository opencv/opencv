cmake_minimum_required(VERSION 3.9)

find_package(OpenCV REQUIRED COMPONENTS opencv_core)

if(NOT OPENCV_EXAMPLES_SKIP_PARALLEL_BACKEND_OPENMP
    AND NOT OPENCV_EXAMPLES_SKIP_OPENMP
)
  project(opencv_example_openmp_backend)
  find_package(OpenMP)
  if(OpenMP_FOUND)
    add_executable(opencv_example_openmp_backend example-openmp.cpp)
    target_link_libraries(opencv_example_openmp_backend PRIVATE
        opencv_core
        OpenMP::OpenMP_CXX
    )
  endif()
endif()

if(NOT OPENCV_EXAMPLES_SKIP_PARALLEL_BACKEND_TBB
    AND NOT OPENCV_EXAMPLES_SKIP_TBB
    AND NOT OPENCV_EXAMPLE_SKIP_TBB  # deprecated (to be removed in OpenCV 5.0)
)
  project(opencv_example_tbb_backend)
  find_package(TBB QUIET)
  if(NOT TBB_FOUND)
    find_path(TBB_INCLUDE_DIR NAMES "tbb/tbb.h")
    find_library(TBB_LIBRARY NAMES "tbb")
  endif()
  if(TBB_INCLUDE_DIR AND TBB_LIBRARY)
    add_executable(opencv_example_tbb_backend example-tbb.cpp)
    target_include_directories(opencv_example_tbb_backend SYSTEM PRIVATE ${TBB_INCLUDE_DIR})
    target_link_libraries(opencv_example_tbb_backend PRIVATE
        opencv_core
        ${TBB_LIBRARY}
    )
  endif()
endif()
