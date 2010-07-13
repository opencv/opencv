file(GLOB old_hdrs "opencv/*.h*")
install(FILES ${old_hdrs}
    DESTINATION include/opencv
    COMPONENT main)

install(FILES "opencv2/opencv.hpp"
    DESTINATION include/opencv2
    COMPONENT main)
