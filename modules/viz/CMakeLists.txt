if(NOT WITH_VTK OR NOT DEFINED HAVE_VTK OR NOT HAVE_VTK)
  ocv_module_disable(viz)
endif()

include(${VTK_USE_FILE})
set(the_description "Viz")
ocv_define_module(viz opencv_core ${VTK_LIBRARIES})

if(APPLE AND BUILD_opencv_viz)
  ocv_target_link_libraries(opencv_viz "-framework Cocoa")
endif()

if(TARGET opencv_test_viz)
  set_target_properties(opencv_test_viz PROPERTIES MACOSX_BUNDLE TRUE)
endif()
