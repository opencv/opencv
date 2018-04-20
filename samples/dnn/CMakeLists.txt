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

function(download_net name commit hash)
  set(DNN_FACE_DETECTOR_MODEL_DOWNLOAD_DIR "${CMAKE_CURRENT_LIST_DIR}/face_detector")
  if(COMMAND ocv_download)
    ocv_download(FILENAME ${name}
               HASH ${hash}
               URL
                 "$ENV{OPENCV_DNN_MODELS_URL}"
                 "${OPENCV_DNN_MODELS_URL}"
                 "https://raw.githubusercontent.com/opencv/opencv_3rdparty/${commit}/"
               DESTINATION_DIR ${DNN_FACE_DETECTOR_MODEL_DOWNLOAD_DIR}
               ID DNN_FACE_DETECTOR
               RELATIVE_URL
               STATUS res)
  endif()
endfunction()

# Model branch name: dnn_samples_face_detector_20180205_fp16
download_net("res10_300x300_ssd_iter_140000_fp16.caffemodel"
             "19512576c112aa2c7b6328cb0e8d589a4a90a26d"
             "f737f886e33835410c69e3ccfe0720a1")

# Model branch name: dnn_samples_face_detector_20180220_uint8
download_net("opencv_face_detector_uint8.pb"
             "7b425df276ba2161b8edaab0f0756f4a735d61b9"
             "56acf81f55d9b9e96c3347bc65409b9e")

project(dnn_samples)
ocv_include_modules_recurse(${OPENCV_DNN_SAMPLES_REQUIRED_DEPS})
file(GLOB_RECURSE dnn_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(sample_filename ${dnn_samples})
  ocv_define_sample(tgt ${sample_filename} dnn)
  ocv_target_link_libraries(${tgt} ${OPENCV_LINKER_LIBS} ${OPENCV_DNN_SAMPLES_REQUIRED_DEPS})
endforeach()
