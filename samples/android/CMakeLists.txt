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

    SET(additional_clean_files "")

    macro(ADD_ANDROID_SAMPLE sample_name)
        #message(STATUS "Build android sample: '${sample_name}'")
        #SET(sample_dir "${CMAKE_CURRENT_SOURCE_DIR}/${sample}")
        SET(sample_dir "${CMAKE_CURRENT_BINARY_DIR}/${sample}")
        
        add_custom_target(${sample} ALL)
        add_dependencies(${sample} opencv_java)

        set_target_properties(${sample} PROPERTIES
            PROJECT_LABEL "(ANDROID EXAMPLE) ${sample}")

        file(RELATIVE_PATH OPENCV_REFERENCE_PATH "${sample_dir}" "${CMAKE_BINARY_DIR}")

        if(NOT ("${sample}" MATCHES "tutorial-[03].*"))
            SET(opencv_reference --library \"${OPENCV_REFERENCE_PATH}\")
        else()
            SET(opencv_reference)
        endif()

        add_custom_command(
            TARGET ${sample}
            WORKING_DIRECTORY ${sample_dir}
            COMMAND ${CMAKE_COMMAND} -E remove -f "${sample_dir}/default.properties"
            COMMAND ${CMAKE_COMMAND} -E touch "${sample_dir}/default.properties"
            COMMAND ${ANDROID_EXECUTABLE} update project --name "${sample}" --target "${ANDROID_SDK_TARGET}" ${opencv_reference} --path .
            COMMAND ${ANT_EXECUTABLE} debug
            COMMAND ${CMAKE_COMMAND} -E copy "${sample_dir}/bin/${sample}-debug.apk" "${CMAKE_BINARY_DIR}/bin/${sample}.apk"
        )

        LIST(APPEND additional_clean_files "${CMAKE_BINARY_DIR}/bin/${sample}.apk" "${sample_dir}/build.xml" "${sample_dir}/local.properties" "${sample_dir}/proguard.cfg")
        
        if(IS_DIRECTORY "${sample_dir}/jni")
            INCLUDE_DIRECTORIES("${sample_dir}/jni")
            FILE( GLOB srcs "${sample_dir}/jni/*.cpp" )

            FILE(STRINGS "${sample_dir}/jni/Android.mk" JNI_LIB_NAME REGEX "LOCAL_MODULE[ ]*:=[ ]*.*" )
            string(REGEX REPLACE "LOCAL_MODULE[ ]*:=[ ]*([a-zA-Z_][a-zA-Z_0-9]*)[ ]*" "\\1" JNI_LIB_NAME "${JNI_LIB_NAME}")

            ADD_LIBRARY( ${JNI_LIB_NAME} MODULE ${srcs} )

            ADD_DEPENDENCIES(${JNI_LIB_NAME} ${sample_dependencies})
            TARGET_LINK_LIBRARIES(${JNI_LIB_NAME} ${OPENCV_LINKER_LIBS} ${sample_dependencies})

            set_target_properties(${JNI_LIB_NAME} PROPERTIES
                OUTPUT_NAME "${JNI_LIB_NAME}"
                LIBRARY_OUTPUT_DIRECTORY "${sample_dir}/libs/${ARMEABI_NDK_NAME}"
            )

            ADD_CUSTOM_COMMAND(
                TARGET ${JNI_LIB_NAME}
                POST_BUILD
                COMMAND ${CMAKE_STRIP} "${sample_dir}/libs/${ARMEABI_NDK_NAME}/*.so"
                )

            add_dependencies(${sample} ${JNI_LIB_NAME})
        endif()

        if(INSTALL_ANDROID_EXAMPLES)
            install(FILES "${CMAKE_BINARY_DIR}/bin/${sample}.apk" DESTINATION bin COMPONENT main)
        endif()
    endmacro()

    file(GLOB android_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
    list(REMOVE_ITEM android_samples hello-android)
    list(SORT android_samples)
    
    file(COPY ${android_samples} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}"
        PATTERN ".svn" EXCLUDE
        PATTERN "gen" EXCLUDE
        PATTERN "bin" EXCLUDE
        )

    foreach(sample ${android_samples})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${sample}/AndroidManifest.xml)
            ADD_ANDROID_SAMPLE(${sample})
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

    set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${additional_clean_files}")
endif()
