# creates target "${_target}_android_project" for building standard Android project
macro(add_android_project _target _path)
    SET (android_dependencies opencv_contrib opencv_legacy opencv_objdetect opencv_calib3d opencv_features2d opencv_video opencv_highgui opencv_ml opencv_imgproc opencv_flann opencv_core)
    if(NOT BUILD_SHARED_LIBS)
        LIST(APPEND android_dependencies opencv_androidcamera)
    endif()

    if (ANDROID AND CAN_BUILD_ANDROID_PROJECTS)
        file(GLOB_RECURSE res_files_all RELATIVE "${_path}" "${_path}/res/*")
        file(GLOB_RECURSE jni_files_all RELATIVE "${_path}" "${_path}/jni/*.c*" "${_path}/jni/*.h*")
        file(GLOB_RECURSE src_files_all RELATIVE "${_path}" "${_path}/src/*.java")

        #remove .svn
        set(res_files)
        foreach(f ${res_files_all})
            if(NOT f MATCHES "\\.svn")
                list(APPEND res_files "${f}")
            endif()
        endforeach()
        set(jni_files)
        foreach(f ${jni_files_all})
            if(NOT f MATCHES "\\.svn")
                list(APPEND jni_files "${f}")
            endif()
        endforeach()
        set(src_files)
        foreach(f ${src_files_all})
            if(NOT f MATCHES "\\.svn")
                list(APPEND src_files "${f}")
            endif()
        endforeach()

        # get temporary location for the project
        file(RELATIVE_PATH build_path "${OpenCV_SOURCE_DIR}" "${_path}")
        SET(build_path "${CMAKE_BINARY_DIR}/${build_path}")

        # copy project to temporary location
        SET(${_target}_project_files)
        foreach(f ${res_files} ${src_files} "AndroidManifest.xml")
            if(NOT "${build_path}" STREQUAL "${_path}")
                #this is not needed in case of in-source build
                add_custom_command(
                    OUTPUT "${build_path}/${f}"
                    COMMAND ${CMAKE_COMMAND} -E copy "${_path}/${f}" "${build_path}/${f}"
                    DEPENDS "${_path}/${f}"
                    COMMENT ""
                    )
            endif()
            list(APPEND ${_target}_project_files "${build_path}/${f}")
        endforeach()

        # process default.properties
        file(STRINGS "${_path}/default.properties" default_properties REGEX "^android\\.library\\.reference\\.1=.+$")
        if (default_properties)
            # has opencv dependency
            file(RELATIVE_PATH OPENCV_REFERENCE_PATH "${build_path}" "${CMAKE_BINARY_DIR}")
            add_custom_command(
                OUTPUT "${build_path}/default.properties"
                OUTPUT "${build_path}/build.xml"
                OUTPUT "${build_path}/local.properties"
                OUTPUT "${build_path}/proguard.cfg"
                COMMAND ${CMAKE_COMMAND} -E echo "" > "default.properties"
                COMMAND ${ANDROID_EXECUTABLE} update project --name "${_target}" --target "${ANDROID_SDK_TARGET}" --library "${OPENCV_REFERENCE_PATH}" --path .
                WORKING_DIRECTORY ${build_path}
                DEPENDS ${${_target}_project_files}
                DEPENDS "${CMAKE_BINARY_DIR}/default.properties"
                DEPENDS "${CMAKE_BINARY_DIR}/AndroidManifest.xml"
                COMMENT "Updating android project - ${_target}"
                )
        else()
            # has no opencv dependency
            add_custom_command(
                OUTPUT "${build_path}/default.properties"
                OUTPUT "${build_path}/build.xml"
                OUTPUT "${build_path}/local.properties"
                OUTPUT "${build_path}/proguard.cfg"
                COMMAND ${CMAKE_COMMAND} -E echo "" > "default.properties"
                COMMAND ${ANDROID_EXECUTABLE} update project --name "${_target}" --target "${ANDROID_SDK_TARGET}" --path .
                WORKING_DIRECTORY ${build_path}
                DEPENDS ${${_target}_project_files}
                COMMENT "Updating android project - ${_target}"
                )
        endif()

        if("${build_path}" STREQUAL "${_path}")
            #in case of in-source build default.properties file is not generated (it is just overwritten :)
            SET_SOURCE_FILES_PROPERTIES("${build_path}/default.properties" PROPERTIES GENERATED FALSE)
        endif()

        list(APPEND ${_target}_project_files "${build_path}/default.properties" "${build_path}/build.xml" "${build_path}/local.properties" "${build_path}/proguard.cfg")

        # build native part of android project
        if(jni_files)
            INCLUDE_DIRECTORIES("${_path}/jni")

            FILE(STRINGS "${_path}/jni/Android.mk" JNI_LIB_NAME REGEX "LOCAL_MODULE[ ]*:=[ ]*.*" )
            string(REGEX REPLACE "LOCAL_MODULE[ ]*:=[ ]*([a-zA-Z_][a-zA-Z_0-9]*)[ ]*" "\\1" JNI_LIB_NAME "${JNI_LIB_NAME}")

            SET(jni_sources)
            foreach(src ${jni_files})
                list(APPEND jni_sources "${_path}/${src}")
            endforeach()

            ADD_LIBRARY(${JNI_LIB_NAME} MODULE ${jni_sources})
            TARGET_LINK_LIBRARIES(${JNI_LIB_NAME} ${OPENCV_LINKER_LIBS} ${android_dependencies})

            set_target_properties(${JNI_LIB_NAME} PROPERTIES
                OUTPUT_NAME "${JNI_LIB_NAME}"
                LIBRARY_OUTPUT_DIRECTORY "${build_path}/libs/${ANDROID_NDK_ABI_NAME}"
            )

            ADD_CUSTOM_COMMAND(
                TARGET ${JNI_LIB_NAME}
                POST_BUILD
                COMMAND ${CMAKE_STRIP} "${build_path}/libs/${ANDROID_NDK_ABI_NAME}/*.so"
                )
        else()
            SET(JNI_LIB_NAME)
        endif()

        add_custom_command(
            OUTPUT "${build_path}/bin/${_target}-debug.apk"
            OUTPUT "${CMAKE_BINARY_DIR}/bin/${_target}.apk"
            COMMAND ${ANT_EXECUTABLE} -q -noinput -k debug
            COMMAND ${CMAKE_COMMAND} -E copy "${build_path}/bin/${_target}-debug.apk" "${CMAKE_BINARY_DIR}/bin/${_target}.apk"
            WORKING_DIRECTORY ${build_path}
            DEPENDS ${${_target}_project_files}
            DEPENDS "${LIBRARY_OUTPUT_PATH}/libopencv_java.so"
            COMMENT "Generating bin/${_target}.apk"
        )

        ADD_CUSTOM_TARGET(${_target}_android_project ALL
            DEPENDS "${build_path}/bin/${_target}-debug.apk"
            DEPENDS "${CMAKE_BINARY_DIR}/bin/${_target}.apk"
            )

        add_dependencies(${_target}_android_project opencv_java ${JNI_LIB_NAME})

        if("${ARGN}" STREQUAL "INSTALL" AND INSTALL_ANDROID_EXAMPLES)
            install(FILES "${CMAKE_BINARY_DIR}/bin/${_target}.apk" DESTINATION "bin" COMPONENT main)
        endif()
    endif()
endmacro()
