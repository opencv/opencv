# ----------------------------------------------------------------------------
#  CMake file for Android samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

if (BUILD_ANDROID_EXAMPLES)
    project(android_samples)

    include_directories(
        "${CMAKE_SOURCE_DIR}/modules/core/include"
        "${CMAKE_SOURCE_DIR}/modules/flann/include"
        "${CMAKE_SOURCE_DIR}/modules/imgproc/include"
        "${CMAKE_SOURCE_DIR}/modules/video/include"
        "${CMAKE_SOURCE_DIR}/modules/highgui/include"
        "${CMAKE_SOURCE_DIR}/modules/ml/include"
        "${CMAKE_SOURCE_DIR}/modules/calib3d/include"
        "${CMAKE_SOURCE_DIR}/modules/features2d/include"
        "${CMAKE_SOURCE_DIR}/modules/objdetect/include"
        "${CMAKE_SOURCE_DIR}/modules/legacy/include"
        "${CMAKE_SOURCE_DIR}/modules/contrib/include"
    )

    SET (sample_dependencies opencv_contrib opencv_legacy opencv_objdetect opencv_calib3d opencv_features2d opencv_video opencv_highgui opencv_ml opencv_imgproc opencv_flann opencv_core)
    if(NOT BUILD_SHARED_LIBS)
        LIST(APPEND sample_dependencies opencv_androidcamera)
    endif()

    file(GLOB android_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
    list(REMOVE_ITEM android_samples hello-android)
    list(SORT android_samples)
    
    foreach(sample ${android_samples})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${sample}/AndroidManifest.xml")
            add_android_project("${sample}" "${CMAKE_CURRENT_SOURCE_DIR}/${sample}")
        endif()
    endforeach()

    #hello-android sample
    ADD_EXECUTABLE(hello-android hello-android/main.cpp)
    ADD_DEPENDENCIES(hello-android ${sample_dependencies})
    TARGET_LINK_LIBRARIES(hello-android ${OPENCV_LINKER_LIBS} ${sample_dependencies})
    set_target_properties(hello-android PROPERTIES OUTPUT_NAME hello-android RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")
    if(INSTALL_ANDROID_EXAMPLES)
        install(TARGETS hello-android DESTINATION bin COMPONENT main)
    endif()

endif()
