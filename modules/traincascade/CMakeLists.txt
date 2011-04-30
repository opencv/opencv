project(traincascade)

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/modules/core/include"
    "${CMAKE_SOURCE_DIR}/modules/imgproc/include"
    "${CMAKE_SOURCE_DIR}/modules/objdetect/include"
    "${CMAKE_SOURCE_DIR}/modules/ml/include"
    "${CMAKE_SOURCE_DIR}/modules/highgui/include"
    "${CMAKE_SOURCE_DIR}/modules/video/include"
    "${CMAKE_SOURCE_DIR}/modules/features2d/include"
    "${CMAKE_SOURCE_DIR}/modules/flann/include"
    "${CMAKE_SOURCE_DIR}/modules/calib3d/include"
    "${CMAKE_SOURCE_DIR}/modules/legacy/include"
    )

set(traincascade_libs opencv_core opencv_ml opencv_imgproc
    opencv_objdetect opencv_highgui opencv_haartraining_engine)

set(traincascade_files traincascade.cpp
    cascadeclassifier.cpp cascadeclassifier.h
    boost.cpp boost.h features.cpp traincascade_features.h
    haarfeatures.cpp haarfeatures.h
    lbpfeatures.cpp lbpfeatures.h
    imagestorage.cpp imagestorage.h)

set(the_target opencv_traincascade)    
add_executable(${the_target} ${traincascade_files})

add_dependencies(${the_target} ${traincascade_libs})
set_target_properties(${the_target} PROPERTIES
                      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
                      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
                      INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
                      OUTPUT_NAME "opencv_traincascade")

target_link_libraries(${the_target} ${traincascade_libs})

install(TARGETS ${the_target} RUNTIME DESTINATION bin COMPONENT main)
