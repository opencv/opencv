find_package(Boost COMPONENTS
  python
  )
find_package(PythonLibs)

include_directories(
    "${CMAKE_SOURCE_DIR}/modules/core/include"
    "${CMAKE_SOURCE_DIR}/modules/imgproc/include"
    "${CMAKE_SOURCE_DIR}/modules/video/include"
    "${CMAKE_SOURCE_DIR}/modules/highgui/include"
    "${CMAKE_SOURCE_DIR}/modules/ml/include"
    "${CMAKE_SOURCE_DIR}/modules/features2d/include"
    "${CMAKE_SOURCE_DIR}/modules/flann/include"
    "${CMAKE_SOURCE_DIR}/modules/calib3d/include"
    "${CMAKE_SOURCE_DIR}/modules/objdetect/include"
    "${CMAKE_SOURCE_DIR}/modules/legacy/include"
    "${CMAKE_SOURCE_DIR}/modules/contrib/include"
    )

include_directories(
  ${PYTHON_INCLUDE_PATH}
  ${Boost_INCLUDE_DIRS}
  )

set(bp_opencv_target opencv_python_boost)

add_library(${bp_opencv_target} SHARED
  opencv.cpp
  cv_core.cpp
  cv_mat.cpp
  cv_points.cpp
  cv_highgui.cpp
  highgui_defines.cpp
  )

set_target_properties(${bp_opencv_target}
  PROPERTIES
  OUTPUT_NAME opencv
  PREFIX ""
  )
  
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_video opencv_ml opencv_features2d opencv_highgui opencv_calib3d opencv_objdetect opencv_legacy opencv_contrib)

target_link_libraries(${bp_opencv_target}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${OpenCV_LIBS}
  )
if(WIN32)
  install(TARGETS ${bp_opencv_target}
          RUNTIME DESTINATION "Python${PYTHON_VERSION_MAJOR_MINOR}/Lib/site-packages" COMPONENT main
          LIBRARY DESTINATION "Python${PYTHON_VERSION_MAJOR_MINOR}/Lib/site-packages" COMPONENT main
          ARCHIVE DESTINATION "Python${PYTHON_VERSION_MAJOR_MINOR}/Lib/site-packages" COMPONENT main
          )
else()
  #install(FILES ${LIBRARY_OUTPUT_PATH}/cv${CVPY_SUFFIX} DESTINATION ${PYTHON_DIST_PACKAGES_PATH})
  install(TARGETS ${bp_opencv_target}
          RUNTIME DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT main
          LIBRARY DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT main
          ARCHIVE DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT main)
endif()

