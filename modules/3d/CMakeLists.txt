set(the_description "Camera Calibration and 3D Reconstruction")

ocv_add_dispatched_file(undistort SSE2 AVX2)

set(debug_modules "")
if(DEBUG_opencv_3d)
  list(APPEND debug_modules opencv_highgui)
endif()
ocv_define_module(3d opencv_imgproc opencv_features2d opencv_flann ${debug_modules}
    WRAP java objc python js
)
ocv_target_link_libraries(${the_module} ${LAPACK_LIBRARIES})


#TODO: remove this dependency
find_package(Ceres QUIET)

if(Ceres_FOUND)
  ocv_target_compile_definitions(${the_module} PUBLIC CERES_FOUND)
  ocv_target_link_libraries(${the_module} ${CERES_LIBRARIES})
  if(Ceres_VERSION VERSION_LESS 2.0.0)
    ocv_include_directories("${CERES_INCLUDE_DIRS}")
  endif()
  add_definitions(/DGLOG_NO_ABBREVIATED_SEVERITIES)  # avoid ERROR macro conflict in glog (ceres dependency)
else()
  message(STATUS "3d/rgbd: CERES support is disabled. Ceres Solver is Required for Posegraph optimization")
endif()
