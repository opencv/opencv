file(GLOB old_hdrs "include/*.h*")
install(FILES ${lib_hdrs}
    DESTINATION include/opencv
    COMPONENT main)
