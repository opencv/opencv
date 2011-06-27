file(GLOB HAAR_CASCADES haarcascades/*.xml)
file(GLOB LBP_CASCADES lbpcascades/*.xml)

if(NOT WIN32)
install(FILES ${HAAR_CASCADES} DESTINATION share/OpenCV/haarcascades COMPONENT main)
install(FILES ${LBP_CASCADES} DESTINATION share/OpenCV/lbpcascades COMPONENT main)
endif()
