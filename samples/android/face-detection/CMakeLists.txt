set(sample example-face-detection)

ocv_download(FILENAME "face_detection_yunet_2023mar.onnx"
             HASH "4ae92eeb150c82ce15ac80738b3b8167"
             URL
               "${OPENCV_FACE_DETECT_YN_URL}"
               "$ENV{OPENCV_FACE_DETECT_YN_URL}"
               "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
             DESTINATION_DIR "${CMAKE_CURRENT_LIST_DIR}/res/raw"
             ID OPENCV_FACE_DETECT_YN
             STATUS res)

add_android_project(${sample} "${CMAKE_CURRENT_SOURCE_DIR}" LIBRARY_DEPS "${OPENCV_ANDROID_LIB_DIR}" SDK_TARGET 11 "${ANDROID_SDK_TARGET}")
if(TARGET ${sample})
  add_dependencies(opencv_android_examples ${sample})
endif()
