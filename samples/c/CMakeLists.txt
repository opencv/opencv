# ----------------------------------------------------------------------------
#  CMake file for C samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

if (BUILD_EXAMPLES)
    project(c_samples)

    if(CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
    endif()
    
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

    # ---------------------------------------------
    #      Define executable targets
    # ---------------------------------------------
    MACRO(MY_DEFINE_EXAMPLE name srcs)
        set(the_target "example_${name}")
    	add_executable(${the_target} ${srcs})
		set_target_properties(${the_target} PROPERTIES
		    OUTPUT_NAME "${name}"
		    PROJECT_LABEL "(EXAMPLE) ${name}")
    	add_dependencies(${the_target} opencv_core opencv_imgproc opencv_highgui
    	    opencv_ml opencv_video opencv_objdetect opencv_features2d
    	    opencv_calib3d opencv_legacy opencv_contrib)
    	target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} opencv_core
    	    opencv_imgproc opencv_highgui opencv_ml opencv_video opencv_objdetect
    	    opencv_features2d opencv_calib3d opencv_legacy opencv_contrib)

        if(WIN32)
            install(TARGETS ${the_target}
                RUNTIME DESTINATION "samples/c" COMPONENT main)
        endif()
    ENDMACRO(MY_DEFINE_EXAMPLE)
    
    file(GLOB cpp_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.c)
    
    foreach(sample_filename ${cpp_samples})
        get_filename_component(sample ${sample_filename} NAME_WE)
        MY_DEFINE_EXAMPLE(${sample}  ${sample_filename})
    endforeach()
endif(BUILD_EXAMPLES)

if (INSTALL_C_EXAMPLES AND NOT WIN32)
    file(GLOB C_SAMPLES *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
    install(FILES ${C_SAMPLES}
            DESTINATION share/opencv/samples/c
            PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
endif ()

