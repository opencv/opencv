file(GLOB HAAR_CASCADES haarcascades/*.xml)
file(GLOB LBP_CASCADES lbpcascades/*.xml)

if(ANDROID)
  install(FILES ${HAAR_CASCADES} DESTINATION sdk/etc/haarcascades COMPONENT libs)
  install(FILES ${LBP_CASCADES}  DESTINATION sdk/etc/lbpcascades  COMPONENT libs)
else()
  install(FILES ${HAAR_CASCADES} DESTINATION share/OpenCV/haarcascades COMPONENT libs)
  install(FILES ${LBP_CASCADES}  DESTINATION share/OpenCV/lbpcascades  COMPONENT libs)
endif()

if(INSTALL_TESTS AND OPENCV_TEST_DATA_PATH)
  if(ANDROID)
    install(DIRECTORY ${OPENCV_TEST_DATA_PATH} DESTINATION sdk/etc/testdata COMPONENT tests)
  else()
    # CPack does not set correct permissions by default, so we do it explicitly.
    install(DIRECTORY ${OPENCV_TEST_DATA_PATH}
            DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
              GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            DESTINATION share/OpenCV/testdata COMPONENT tests)
    if(BUILD_opencv_python)
	  file(GLOB DATAFILES_CPP ../samples/cpp/left*.jpg)
      list(APPEND DATAFILES_CPP
          "../samples/cpp/board.jpg"
          "../samples/cpp/pic1.png"
          "../samples/cpp/pic6.png"
          "../samples/cpp/right01.jpg"
          "../samples/cpp/right02.jpg"
          "../samples/cpp/building.jpg"
          "../samples/cpp/tsukuba_l.png"
          "../samples/cpp/tsukuba_r.png"
          "../samples/cpp/letter-recognition.data")
      install(FILES ${DATAFILES_CPP} DESTINATION share/OpenCV/testdata/samples/cpp COMPONENT tests)
      set(DATAFILES_C
          "../samples/c/lena.jpg"
          "../samples/c/box.png")
      install(FILES ${DATAFILES_C} DESTINATION share/OpenCV/testdata/samples/c COMPONENT tests)
      set(DATAFILES_GPU
          "../samples/gpu/basketball1.png"
          "../samples/gpu/basketball2.png"
          "../samples/gpu/rubberwhale1.png")
      install(FILES ${DATAFILES_GPU} DESTINATION share/OpenCV/testdata/samples/gpu COMPONENT tests)
      set(DATAFILES_PYTHON
          "../samples/python2/data/graf1.png"
          "../samples/python2/data/pca_test1.jpg"
          "../samples/python2/data/digits.png")
      install(FILES ${DATAFILES_PYTHON} DESTINATION share/OpenCV/testdata/samples/python2/data COMPONENT tests)
      set(DATAFILES_CASCADES
          "haarcascades/haarcascade_frontalface_alt.xml"
          "haarcascades/haarcascade_eye.xml")
      install(FILES ${DATAFILES_CASCADES} DESTINATION share/OpenCV/testdata/data/haarcascades COMPONENT tests)
    endif()
  endif()
endif()