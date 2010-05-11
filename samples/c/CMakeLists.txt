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
    	add_executable(${name} ${srcs})
		set_target_properties(${name} PROPERTIES PROJECT_LABEL "(EXAMPLE) ${name}")
    	add_dependencies(${name} opencv_core opencv_imgproc opencv_highgui opencv_ml opencv_video opencv_objdetect opencv_features2d opencv_calib3d opencv_legacy opencv_contrib)
    	target_link_libraries(${name} ${OPENCV_LINKER_LIBS} opencv_core opencv_imgproc opencv_highgui opencv_ml opencv_video opencv_objdetect opencv_features2d opencv_calib3d opencv_legacy opencv_contrib)

        if(WIN32)
            install(TARGETS ${name}
                RUNTIME DESTINATION "samples/c" COMPONENT main)
        endif()
    ENDMACRO(MY_DEFINE_EXAMPLE)
    
    MY_DEFINE_EXAMPLE(adaptiveskindetector 	adaptiveskindetector.cpp)
    MY_DEFINE_EXAMPLE(bgfg_codebook 	bgfg_codebook.cpp)
    MY_DEFINE_EXAMPLE(bgfg_segm			bgfg_segm.cpp)
    MY_DEFINE_EXAMPLE(blobtrack_sample			blobtrack_sample.cpp)
    MY_DEFINE_EXAMPLE(camshiftdemo		camshiftdemo.c)
    MY_DEFINE_EXAMPLE(contours			contours.c)
    MY_DEFINE_EXAMPLE(convert_cascade	convert_cascade.c)
    MY_DEFINE_EXAMPLE(convexhull		convexhull.c)
    MY_DEFINE_EXAMPLE(delaunay			delaunay.c)
    MY_DEFINE_EXAMPLE(demhist			demhist.c)
    MY_DEFINE_EXAMPLE(detectors_sample  	detectors_sample.cpp)
    MY_DEFINE_EXAMPLE(dft				dft.c)
    MY_DEFINE_EXAMPLE(distrans			distrans.c)
    MY_DEFINE_EXAMPLE(drawing			drawing.c)
    MY_DEFINE_EXAMPLE(edge				edge.c)
    MY_DEFINE_EXAMPLE(facedetect		facedetect.cpp)
    MY_DEFINE_EXAMPLE(ffilldemo			ffilldemo.c)
    MY_DEFINE_EXAMPLE(fback			    fback.cpp)
    MY_DEFINE_EXAMPLE(fback_c		    fback_c.c)
    MY_DEFINE_EXAMPLE(find_obj			find_obj.cpp)
    MY_DEFINE_EXAMPLE(fitellipse		fitellipse.cpp)
    MY_DEFINE_EXAMPLE(houghlines		houghlines.c)
    MY_DEFINE_EXAMPLE(image				image.cpp)
    MY_DEFINE_EXAMPLE(inpaint			inpaint.cpp)
    MY_DEFINE_EXAMPLE(kalman			kalman.c)
    MY_DEFINE_EXAMPLE(kmeans			kmeans.c)
    MY_DEFINE_EXAMPLE(laplace			laplace.c)
    MY_DEFINE_EXAMPLE(letter_recog		letter_recog.cpp)
    MY_DEFINE_EXAMPLE(lkdemo			lkdemo.c)
    MY_DEFINE_EXAMPLE(minarea			minarea.c)
    MY_DEFINE_EXAMPLE(morphology		morphology.c)
    MY_DEFINE_EXAMPLE(motempl			motempl.c)
    MY_DEFINE_EXAMPLE(mushroom			mushroom.cpp)
	MY_DEFINE_EXAMPLE(tree_engine		tree_engine.cpp)
	MY_DEFINE_EXAMPLE(one_way_sample	one_way_sample.cpp)
    MY_DEFINE_EXAMPLE(peopledetect      peopledetect.cpp)
    MY_DEFINE_EXAMPLE(pyramid_segmentation	pyramid_segmentation.c)
    MY_DEFINE_EXAMPLE(squares			squares.c)
    MY_DEFINE_EXAMPLE(stereo_calib		stereo_calib.cpp)
    MY_DEFINE_EXAMPLE(stereo_match		stereo_match.cpp)
    MY_DEFINE_EXAMPLE(watershed			watershed.cpp)
	MY_DEFINE_EXAMPLE(grabcut			grabcut.cpp)
    MY_DEFINE_EXAMPLE(polar_transforms	polar_transforms.c)
    MY_DEFINE_EXAMPLE(calibration	calibration.cpp)
    MY_DEFINE_EXAMPLE(calibration_artificial calibration_artificial.cpp)
    MY_DEFINE_EXAMPLE(mser mser_sample.cpp)
    MY_DEFINE_EXAMPLE(find_obj_calonder find_obj_calonder.cpp)
    MY_DEFINE_EXAMPLE(find_obj_ferns find_obj_ferns.cpp)
endif(BUILD_EXAMPLES)

if (INSTALL_C_EXAMPLES AND NOT WIN32)
    file(GLOB C_SAMPLES *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
    install(FILES ${C_SAMPLES}
            DESTINATION share/opencv/samples/c
            PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
endif ()

