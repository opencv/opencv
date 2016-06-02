configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${ANDROID_MANIFEST_FILE}" "${OpenCV_BINARY_DIR}/platforms/android/service/engine/.build/${ANDROID_MANIFEST_FILE}"  @ONLY)
unset(__android_project_chain CACHE)
add_android_project(opencv_engine "${CMAKE_CURRENT_SOURCE_DIR}" SDK_TARGET 9 ${ANDROID_SDK_TARGET} IGNORE_JAVA ON IGNORE_MANIFEST ON COPY_LIBS ON)
