set(sample example-mobilenet-objdetect)

# The Caffe MobileNet-SSD (mobilenet_iter_73000.caffemodel + deploy.prototxt) has been
# replaced by a single-file ONNX model. The sample now loads "res/raw/mobilenet.onnx".
# Provide it through the OPENCV_MOBILENET_SSD_ONNX_URL / OPENCV_MOBILENET_SSD_ONNX_HASH
# variables (env or CMake), or drop the file into res/raw manually. No Caffe assets are
# downloaded anymore.
if((DEFINED ENV{OPENCV_MOBILENET_SSD_ONNX_URL} OR DEFINED OPENCV_MOBILENET_SSD_ONNX_URL) AND OPENCV_MOBILENET_SSD_ONNX_HASH)
  ocv_download(FILENAME "mobilenet.onnx"
               HASH "${OPENCV_MOBILENET_SSD_ONNX_HASH}"
               URL
                 "${OPENCV_MOBILENET_SSD_ONNX_URL}"
                 "$ENV{OPENCV_MOBILENET_SSD_ONNX_URL}"
               DESTINATION_DIR "${CMAKE_CURRENT_LIST_DIR}/res/raw"
               ID OPENCV_MOBILENET_SSD_ONNX
               STATUS res)
endif()

add_android_project(${sample} "${CMAKE_CURRENT_SOURCE_DIR}" LIBRARY_DEPS "${OPENCV_ANDROID_LIB_DIR}" SDK_TARGET 11 "${ANDROID_SDK_TARGET}")
if(TARGET ${sample})
  add_dependencies(opencv_android_examples ${sample})
endif()
