set(the_description "Camera Calibration and 3D Reconstruction")

ocv_add_dispatched_file(undistort SSE2 AVX2)

set(debug_modules "")
if(DEBUG_opencv_calib3d)
  list(APPEND debug_modules opencv_highgui)
endif()
ocv_add_module(calib3d opencv_imgproc opencv_features2d opencv_flann ${debug_modules}
    WRAP java objc python js
)
ocv_glob_module_sources()
ocv_module_include_directories()

add_library(opencv_calib3d_epnp OBJECT "${CMAKE_CURRENT_LIST_DIR}/src/epnp.cpp")
target_compile_definitions(opencv_calib3d_epnp PRIVATE __OPENCV_BUILD=1)
set_property(TARGET opencv_calib3d_epnp PROPERTY CXX_STANDARD 14)
set_property(TARGET opencv_calib3d_epnp PROPERTY CXX_STANDARD_REQUIRED ON)
set_target_properties(opencv_calib3d_epnp PROPERTIES POSITION_INDEPENDENT_CODE ON)
list(REMOVE_ITEM OPENCV_MODULE_${the_module}_SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/epnp.cpp")
list(APPEND OPENCV_MODULE_${the_module}_SOURCES "$<TARGET_OBJECTS:opencv_calib3d_epnp>")
set(OPENCV_MODULE_${the_module}_SOURCES ${OPENCV_MODULE_${the_module}_SOURCES}
    CACHE INTERNAL "List of source files for ${the_module}" FORCE)

ocv_create_module()
target_include_directories(opencv_calib3d_epnp PRIVATE
    "$<TARGET_PROPERTY:${the_module_target},INCLUDE_DIRECTORIES>")

ocv_add_accuracy_tests()
if(TARGET opencv_test_calib3d)
  target_sources(opencv_test_calib3d PRIVATE "$<TARGET_OBJECTS:opencv_calib3d_epnp>")
endif()
ocv_add_perf_tests()
ocv_add_samples()

ocv_target_link_libraries(${the_module} ${LAPACK_LIBRARIES})
