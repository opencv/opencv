# ----------------------------------------------------------------------------
#  CMake file for python support
# ----------------------------------------------------------------------------
project(opencv_python)

include_directories(${PYTHON_INCLUDE_PATH})
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${CMAKE_SOURCE_DIR}/modules/core/include"
    "${CMAKE_SOURCE_DIR}/modules/imgproc/include"
    "${CMAKE_SOURCE_DIR}/modules/video/include"
    "${CMAKE_SOURCE_DIR}/modules/highgui/include"
    "${CMAKE_SOURCE_DIR}/modules/ml/include"
    "${CMAKE_SOURCE_DIR}/modules/features2d/include"
    "${CMAKE_SOURCE_DIR}/modules/flann/include"
    "${CMAKE_SOURCE_DIR}/modules/calib3d/include"
    "${CMAKE_SOURCE_DIR}/modules/objdetect/include"
    "${CMAKE_SOURCE_DIR}/modules/legacy/include"
    "${CMAKE_SOURCE_DIR}/modules/contrib/include"
    )
    
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
endif()

file(GLOB lib_srcs "*.cpp")

add_custom_command(
   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated0.i
   COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen.py ${CMAKE_CURRENT_SOURCE_DIR}
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/api
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/defs
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen.py
   )

set(the_target "opencv_python")
add_library(${the_target} ${lib_srcs} ${lib_hdrs} ${lib_int_hdrs} ${CMAKE_CURRENT_BINARY_DIR}/generated0.i)
target_link_libraries(${the_target} ${PYTHON_LIBRARIES} opencv_core opencv_imgproc opencv_video opencv_ml opencv_features2d opencv_highgui opencv_calib3d opencv_objdetect opencv_legacy opencv_contrib)
    
set_target_properties(${the_target} PROPERTIES PREFIX "")
set_target_properties(${the_target} PROPERTIES OUTPUT_NAME "cv")

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print distutils.sysconfig.get_config_var('SO')"
                RESULT_VARIABLE PYTHON_CVPY_PROCESS
                OUTPUT_VARIABLE CVPY_SUFFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set_target_properties(${the_target} PROPERTIES SUFFIX ${CVPY_SUFFIX})

set(cvpy_files cv${CVPY_SUFFIX})

if(WIN32)
    install(TARGETS ${the_target}
            RUNTIME DESTINATION "Python${PYTHON_VERSION_MAJOR_MINOR}/Lib/site-packages" COMPONENT main
            LIBRARY DESTINATION "Python${PYTHON_VERSION_MAJOR_MINOR}/Lib/site-packages" COMPONENT main
            ARCHIVE DESTINATION "Python${PYTHON_VERSION_MAJOR_MINOR}/Lib/site-packages" COMPONENT main
            )
else()
    #install(FILES ${LIBRARY_OUTPUT_PATH}/cv${CVPY_SUFFIX} DESTINATION ${PYTHON_PACKAGES_PATH})
    install(TARGETS ${the_target} 
            RUNTIME DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT main
            LIBRARY DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT main
            ARCHIVE DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT main)
endif()
