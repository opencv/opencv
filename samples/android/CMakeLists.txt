# ----------------------------------------------------------------------------
#  CMake file for Android samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
SET(OPENCV_ANDROID_SAMPLES_REQUIRED_DEPS opencv_contrib opencv_legacy opencv_objdetect opencv_calib3d opencv_features2d opencv_video opencv_highgui opencv_ml opencv_imgproc opencv_flann opencv_core)

ocv_check_dependencies(${OPENCV_ANDROID_SAMPLES_REQUIRED_DEPS} opencv_java)

if(BUILD_ANDROID_EXAMPLES AND OCV_DEPENDENCIES_FOUND)
  project(android_samples)

  ocv_include_modules(${OPENCV_ANDROID_SAMPLES_REQUIRED_DEPS})
  
  file(GLOB android_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
  list(REMOVE_ITEM android_samples hello-android)
  list(SORT android_samples)
    
  foreach(sample ${android_samples})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${sample}/AndroidManifest.xml")
      add_android_project("${sample}" "${CMAKE_CURRENT_SOURCE_DIR}/${sample}" INSTALL)
    endif()
  endforeach()

  #hello-android sample
  ADD_EXECUTABLE(hello-android hello-android/main.cpp)
  TARGET_LINK_LIBRARIES(hello-android ${OPENCV_LINKER_LIBS} ${OPENCV_ANDROID_SAMPLES_REQUIRED_DEPS})
  set_target_properties(hello-android PROPERTIES OUTPUT_NAME hello-android RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")
  if(INSTALL_ANDROID_EXAMPLES)
    install(TARGETS hello-android DESTINATION bin COMPONENT main)
  endif()
endif()
