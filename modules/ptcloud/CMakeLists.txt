set(the_description "High level point cloud and mesh operations")

set(debug_modules "")
if(DEBUG_opencv_ptcloud)
  list(APPEND debug_modules opencv_highgui)
endif()
ocv_define_module(ptcloud opencv_geometry opencv_imgproc opencv_features opencv_video ${debug_modules}
  WRAP java objc python js
)
ocv_target_link_libraries(${the_module} ${LAPACK_LIBRARIES})

if(NOT HAVE_EIGEN)
  message(STATUS "Geometry: Eigen support is disabled. Eigen is Required for Posegraph optimization")
endif()

if(HAVE_G2O)
  ocv_target_include_directories(${the_module} SYSTEM PRIVATE ${G2O_INCLUDE_DIRS})
  ocv_target_link_libraries(${the_module} ${G2O_LIBRARIES})
  ocv_target_compile_definitions(${the_module} PRIVATE HAVE_G2O)
  message(STATUS "ptcloud: g2o backend enabled (real pose BA / local BA / global BA / loop closure)")
else()
  message(STATUS "ptcloud: g2o not available — SLAM BA/loop-closure compile to no-ops")
endif()
