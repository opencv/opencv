file(GLOB HAAR_CASCADES haarcascades/*.xml)
file(GLOB LBP_CASCADES lbpcascades/*.xml)

if(ANDROID)
  install(FILES ${HAAR_CASCADES} DESTINATION sdk/etc/haarcascades COMPONENT libs)
  install(FILES ${LBP_CASCADES}  DESTINATION sdk/etc/lbpcascades  COMPONENT libs)
elseif(NOT WIN32)
  install(FILES ${HAAR_CASCADES} DESTINATION share/OpenCV/haarcascades COMPONENT libs)
  install(FILES ${LBP_CASCADES}  DESTINATION share/OpenCV/lbpcascades  COMPONENT libs)
endif()

if(INSTALL_TESTS AND OPENCV_TEST_DATA_PATH)
  install(DIRECTORY "${OPENCV_TEST_DATA_PATH}/" DESTINATION "${OPENCV_TEST_DATA_INSTALL_PATH}" COMPONENT "tests")
endif()