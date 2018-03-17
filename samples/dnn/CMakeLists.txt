ocv_install_example_src(dnn *.cpp *.hpp CMakeLists.txt)

set(OPENCV_DNN_SAMPLES_REQUIRED_DEPS
  opencv_core
  opencv_imgproc
  opencv_dnn
  opencv_imgcodecs
  opencv_videoio
  opencv_highgui)
ocv_check_dependencies(${OPENCV_DNN_SAMPLES_REQUIRED_DEPS})

if(NOT BUILD_EXAMPLES OR NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

# Model branch name: dnn_samples_face_detector_20170830
set(DNN_FACE_DETECTOR_MODEL_COMMIT "b2bfc75f6aea5b1f834ff0f0b865a7c18ff1459f")
set(DNN_FACE_DETECTOR_MODEL_HASH "afbb6037fd180e8d2acb3b58ca737b9e")
set(DNN_FACE_DETECTOR_MODEL_NAME "res10_300x300_ssd_iter_140000.caffemodel")
set(DNN_FACE_DETECTOR_MODEL_DOWNLOAD_DIR "${CMAKE_CURRENT_LIST_DIR}/face_detector")
if(COMMAND ocv_download)
  ocv_download(FILENAME ${DNN_FACE_DETECTOR_MODEL_NAME}
             HASH ${DNN_FACE_DETECTOR_MODEL_HASH}
             URL
               "$ENV{OPENCV_DNN_MODELS_URL}"
               "${OPENCV_DNN_MODELS_URL}"
               "https://raw.githubusercontent.com/opencv/opencv_3rdparty/${DNN_FACE_DETECTOR_MODEL_COMMIT}/"
             DESTINATION_DIR ${DNN_FACE_DETECTOR_MODEL_DOWNLOAD_DIR}
             ID DNN_FACE_DETECTOR
             RELATIVE_URL
             STATUS res)
endif()
project(dnn_samples)
ocv_include_modules_recurse(${OPENCV_DNN_SAMPLES_REQUIRED_DEPS})
file(GLOB_RECURSE dnn_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(sample_filename ${dnn_samples})
  ocv_define_sample(tgt ${sample_filename} dnn)
  ocv_target_link_libraries(${tgt} ${OPENCV_LINKER_LIBS} ${OPENCV_DNN_SAMPLES_REQUIRED_DEPS})
endforeach()
