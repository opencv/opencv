file(GLOB old_hdrs "opencv/*.h*")
install(FILES ${old_hdrs}
    DESTINATION include/opencv
    COMPONENT main)
