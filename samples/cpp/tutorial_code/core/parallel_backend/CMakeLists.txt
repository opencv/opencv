cmake_minimum_required(VERSION 3.9)

find_package(OpenCV REQUIRED COMPONENTS opencv_core)

find_package(OpenMP)
if(OpenMP_FOUND)
  project(opencv_example_openmp_backend)
  add_executable(opencv_example_openmp_backend example-openmp.cpp)
  target_link_libraries(opencv_example_openmp_backend PRIVATE
      opencv_core
      OpenMP::OpenMP_CXX
  )
endif()

# TODO: find_package(TBB)
find_path(TBB_INCLUDE_DIR NAMES "tbb/tbb.h")
find_library(TBB_LIBRARY NAMES "tbb")
if(TBB_INCLUDE_DIR AND TBB_LIBRARY AND NOT OPENCV_EXAMPLE_SKIP_TBB)
  project(opencv_example_tbb_backend)
  add_executable(opencv_example_tbb_backend example-tbb.cpp)
  target_include_directories(opencv_example_tbb_backend SYSTEM PRIVATE ${TBB_INCLUDE_DIR})
  target_link_libraries(opencv_example_tbb_backend PRIVATE
      opencv_core
      ${TBB_LIBRARY}
  )
endif()
