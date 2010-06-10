# ----------------------------------------------------------------------------
#  CMake file for C samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

if (BUILD_EXAMPLES)
    project(cpp_samples)

    include_directories(
        "${CMAKE_SOURCE_DIR}/modules/core/include"
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

    if(CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
    endif()
    
    # ---------------------------------------------
    #      Define executable targets
    # ---------------------------------------------
    MACRO(MY_DEFINE_EXAMPLE name srcs)
        add_executable(${name} ${srcs})
        set_target_properties(${name} PROPERTIES PROJECT_LABEL "(EXAMPLE) ${name}")
        add_dependencies(${name} opencv_core opencv_imgproc
            opencv_highgui opencv_ml opencv_video opencv_objdetect
            opencv_features2d opencv_calib3d opencv_contrib)
        target_link_libraries(${name} ${OPENCV_LINKER_LIBS} opencv_core
            opencv_imgproc opencv_highgui opencv_ml opencv_video
            opencv_objdetect opencv_features2d opencv_calib3d opencv_contrib)

        if(WIN32)
            install(TARGETS ${name}
                RUNTIME DESTINATION "samples/cpp" COMPONENT main)
        endif()
    ENDMACRO(MY_DEFINE_EXAMPLE)
    
    MY_DEFINE_EXAMPLE(select3dobj  select3dobj.cpp)
    MY_DEFINE_EXAMPLE(connected_components  connected_components.cpp)
    MY_DEFINE_EXAMPLE(contours2  contours2.cpp)
    MY_DEFINE_EXAMPLE(morphology2  morphology2.cpp)
    MY_DEFINE_EXAMPLE(segment_objects  segment_objects.cpp)
endif(BUILD_EXAMPLES)

if (INSTALL_C_EXAMPLES AND NOT WIN32)
    file(GLOB C_SAMPLES *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
    install(FILES ${C_SAMPLES}
            DESTINATION share/opencv/samples/cpp
            PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
endif ()

