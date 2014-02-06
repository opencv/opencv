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
  if(ANDROID)
    install(DIRECTORY ${OPENCV_TEST_DATA_PATH} DESTINATION sdk/etc/testdata COMPONENT tests)
  elseif(NOT WIN32)
    # CPack does not set correct permissions by default, so we do it explicitly.
    install(DIRECTORY ${OPENCV_TEST_DATA_PATH}
            DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
              GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            DESTINATION share/OpenCV/testdata COMPONENT tests)
  endif()
endif()