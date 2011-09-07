# ----------------------------------------------------------------------------
#  CMake file for C samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

if (BUILD_EXAMPLES)
    project(cpp_samples)

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
        "${CMAKE_SOURCE_DIR}/modules/stitching/include"
    )
    if (NOT ANDROID)
       include_directories("${CMAKE_SOURCE_DIR}/modules/gpu/include")
    endif()

    if(CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
    endif()
    
    # ---------------------------------------------
    #      Define executable targets
    # ---------------------------------------------
    MACRO(MY_DEFINE_EXAMPLE name srcs)
        set(the_target "example_${name}")
        add_executable(${the_target} ${srcs})
        set_target_properties(${the_target} PROPERTIES
            OUTPUT_NAME "${name}"
            PROJECT_LABEL "(EXAMPLE) ${name}")
        add_dependencies(${the_target} opencv_core opencv_flann opencv_imgproc opencv_highgui
            opencv_ml opencv_video opencv_objdetect opencv_features2d
            opencv_calib3d opencv_legacy opencv_contrib opencv_stitching)
        target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} opencv_core
            opencv_flann opencv_imgproc opencv_highgui opencv_ml opencv_video opencv_objdetect
            opencv_features2d opencv_calib3d opencv_legacy opencv_contrib opencv_stitching)

       if (NOT ANDROID)
            target_link_libraries(${the_target} opencv_gpu)
            add_dependencies(${the_target} opencv_gpu)
       endif()
            
        if(ENABLE_SOLUTION_FOLDERS)
            set_target_properties(${the_target} PROPERTIES FOLDER "samples//cpp")
        endif()

        if(WIN32)
        if (MSVC AND NOT BUILD_SHARED_LIBS)
            set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
        endif()
            install(TARGETS ${the_target}
                RUNTIME DESTINATION "samples/cpp" COMPONENT main)
        endif()
    ENDMACRO(MY_DEFINE_EXAMPLE)
    
    file(GLOB cpp_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
    
    foreach(sample_filename ${cpp_samples})
        get_filename_component(sample ${sample_filename} NAME_WE)
        MY_DEFINE_EXAMPLE(${sample}  ${sample_filename})
    endforeach()
endif(BUILD_EXAMPLES)

if (INSTALL_C_EXAMPLES AND NOT WIN32)
    file(GLOB C_SAMPLES *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
    install(FILES ${C_SAMPLES}
            DESTINATION share/opencv/samples/cpp
            PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
endif ()

