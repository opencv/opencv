project(haartraining)

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

if(WIN32)
    link_directories(${CMAKE_SOURCE_DIR}/lib ${CMAKE_SOURCE_DIR}/bin ${CMAKE_CURRENT_BINARY_DIR})
endif()

set(haartraining_libs opencv_core opencv_imgproc opencv_highgui opencv_objdetect opencv_calib3d opencv_haartraining_engine)

link_libraries(${haartraining_libs})

# -----------------------------------------------------------
#  Library
# -----------------------------------------------------------
set(cvhaartraining_lib_src
    _cvcommon.h
    cvclassifier.h
    _cvhaartraining.h
    cvhaartraining.h
    cvboost.cpp
    cvcommon.cpp
    cvhaarclassifier.cpp
    cvhaartraining.cpp
    cvsamples.cpp
    )

add_library(opencv_haartraining_engine STATIC ${cvhaartraining_lib_src})
set_target_properties(opencv_haartraining_engine PROPERTIES
    DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
    INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
    )

if(NOT ANDROID)
# -----------------------------------------------------------
#  haartraining
# -----------------------------------------------------------

add_executable(opencv_haartraining cvhaartraining.h haartraining.cpp)
add_dependencies(opencv_haartraining ${haartraining_libs})
set_target_properties(opencv_haartraining PROPERTIES
                      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                      OUTPUT_NAME "opencv_haartraining")

# -----------------------------------------------------------
#  createsamples
# -----------------------------------------------------------

add_executable(opencv_createsamples cvhaartraining.h createsamples.cpp)
add_dependencies(opencv_createsamples ${haartraining_libs})
set_target_properties(opencv_createsamples PROPERTIES
                      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                      OUTPUT_NAME "opencv_createsamples")

# -----------------------------------------------------------
#  performance
# -----------------------------------------------------------
add_executable(opencv_performance performance.cpp)
add_dependencies(opencv_performance ${haartraining_libs})
set_target_properties(opencv_performance PROPERTIES
                      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                      OUTPUT_NAME "opencv_performance")

# -----------------------------------------------------------
#  Install part
# -----------------------------------------------------------

install(TARGETS opencv_haartraining RUNTIME DESTINATION bin COMPONENT main)
install(TARGETS opencv_createsamples RUNTIME DESTINATION bin COMPONENT main)
install(TARGETS opencv_performance RUNTIME DESTINATION bin COMPONENT main)
endif()

if(SOLUTION_FOLDERS_ENABLED)
	set_target_properties(opencv_performance PROPERTIES FOLDER "applications")
	set_target_properties(opencv_createsamples PROPERTIES FOLDER "applications")
	set_target_properties(opencv_haartraining PROPERTIES FOLDER "applications")
	set_target_properties(opencv_haartraining_engine PROPERTIES FOLDER "applications")
endif()	
