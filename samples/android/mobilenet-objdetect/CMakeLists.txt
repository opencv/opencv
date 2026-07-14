set(sample example-mobilenet-objdetect)

ocv_download(FILENAME "mobilenet_iter_73000.caffemodel"
             HASH "bbcb3b6a0afe1ec89e1288096b5b8c66"
             URL
               "${OPENCV_MOBILENET_SSD_WEIGHTS_URL}"
               "$ENV{OPENCV_MOBILENET_SSD_WEIGHTS_URL}"
               "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/97406996b1eee2d40eb0a00ae567cf41e23369f9/mobilenet_iter_73000.caffemodel"
             DESTINATION_DIR "${CMAKE_CURRENT_LIST_DIR}/res/raw"
             ID OPENCV_MOBILENET_SSD_WEIGHTS
             STATUS res)

ocv_download(FILENAME "deploy.prototxt"
             HASH "f1978dc4fe20c680e850ce99830c5945"
             URL
               "${OPENCV_MOBILENET_SSD_CONFIG_URL}"
               "$ENV{OPENCV_MOBILENET_SSD_CONFIG_URL}"
               "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/97406996b1eee2d40eb0a00ae567cf41e23369f9/deploy.prototxt"
             DESTINATION_DIR "${CMAKE_CURRENT_LIST_DIR}/res/raw"
             ID OPENCV_MOBILENET_SSD_CONFIG
             STATUS res)

add_android_project(${sample} "${CMAKE_CURRENT_SOURCE_DIR}" LIBRARY_DEPS "${OPENCV_ANDROID_LIB_DIR}" SDK_TARGET 11 "${ANDROID_SDK_TARGET}")
if(TARGET ${sample})
  add_dependencies(opencv_android_examples ${sample})
endif()
