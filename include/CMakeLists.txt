file(GLOB old_hdrs "opencv/*.h*")
install(FILES ${old_hdrs}
    DESTINATION ${OPENCV_INCLUDE_PREFIX}/opencv
    COMPONENT main)
install(FILES "opencv2/opencv.hpp"
    DESTINATION ${OPENCV_INCLUDE_PREFIX}/opencv2
    COMPONENT main)


