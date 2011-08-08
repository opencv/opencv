if (BUILD_EXAMPLES)
    set(project "gpu")
    string(TOUPPER "${project}" project_upper)
    
    project("${project}_samples")

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
        "${CMAKE_SOURCE_DIR}/modules/gpu/include"
        "${CMAKE_SOURCE_DIR}/modules/gpu/src/nvidia"
        "${CMAKE_SOURCE_DIR}/modules/gpu/src/nvidia/core"
    )

    if(HAVE_CUDA)
        include_directories(${CUDA_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/modules/gpu/src/nvidia ${CMAKE_SOURCE_DIR}/modules/gpu/src/nvidia/core)
    endif()

    if(CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
    endif()

    # ---------------------------------------------
    #      Define executable targets
    # ---------------------------------------------
    MACRO(MY_DEFINE_EXAMPLE name srcs)
        set(the_target "example_${project}_${name}")
        add_executable(${the_target} ${srcs})
        set_target_properties(${the_target} PROPERTIES
                OUTPUT_NAME "${name}_${project}"                                       
                PROJECT_LABEL "(EXAMPLE_${project_upper}) ${name}")
        add_dependencies(${the_target} opencv_core opencv_flann opencv_imgproc opencv_highgui
            opencv_ml opencv_video opencv_objdetect opencv_features2d
            opencv_calib3d opencv_legacy opencv_contrib opencv_gpu)
        target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} opencv_core
            opencv_flann opencv_imgproc opencv_highgui opencv_ml opencv_video opencv_objdetect
            opencv_features2d opencv_calib3d opencv_legacy opencv_contrib opencv_gpu)

        if(ENABLE_SOLUTION_FOLDERS)
            set_target_properties(${the_target} PROPERTIES FOLDER "samples//${project}")
        endif()

        if(WIN32)
            if (MSVC AND NOT BUILD_SHARED_LIBS)
                set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
            endif()
            install(TARGETS ${the_target} RUNTIME DESTINATION "samples/${project}" COMPONENT main)
        endif()
    ENDMACRO(MY_DEFINE_EXAMPLE)

    file(GLOB all_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

    foreach(sample_filename ${all_samples})
        get_filename_component(sample ${sample_filename} NAME_WE)
        file(GLOB sample_srcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${sample}.*)
        MY_DEFINE_EXAMPLE(${sample} ${sample_srcs})
    endforeach()

    include("performance/CMakeLists.txt")
endif(BUILD_EXAMPLES)

if (NOT WIN32)
    file(GLOB install_list *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
    install(FILES ${install_list}
            DESTINATION share/opencv/samples/${project}
            PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
endif ()

