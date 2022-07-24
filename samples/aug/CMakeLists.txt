ocv_install_example_src(aug *.cpp *.hpp CMakeLists.txt)

set(OPENCV_AUG_SAMPLES_REQUIRED_DEPS opencv_aug opencv_imgproc opencv_core opencv_imgcodecs opencv_highgui)

ocv_check_dependencies(${OPENCV_AUG_SAMPLES_REQUIRED_DEPS})

if(NOT BUILD_EXAMPLES OR NOT OCV_DEPENDENCIES_FOUND)
    message(${BUILD_EXAMPLES})
    message(${OCV_DEPENDENCIES_FOUND})
    return()
endif()

project(aug_samples)

ocv_include_modules_recurse(${OPENCV_AUG_SAMPLES_REQUIRED_DEPS})
file(GLOB all_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(sample_filename ${all_samples})
    ocv_define_sample(tgt ${sample_filename} aug)
    ocv_target_link_libraries(${tgt} PRIVATE ${OPENCV_LINKER_LIBS} ${OPENCV_AUG_SAMPLES_REQUIRED_DEPS})
endforeach()