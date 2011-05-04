project(stitching)

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
    "${CMAKE_SOURCE_DIR}/modules/imgproc/src" # for gcgraph.hpp
    "${CMAKE_SOURCE_DIR}/modules/gpu/include"
    )

set(stitching_libs opencv_core opencv_imgproc opencv_highgui opencv_features2d opencv_calib3d opencv_gpu)

set(stitching_files blenders.hpp blenders.cpp
                    focal_estimators.hpp focal_estimators.cpp
                    motion_estimators.hpp motion_estimators.cpp
                    seam_finders.hpp seam_finders.cpp
                    util.hpp util.cpp util_inl.hpp
                    warpers.hpp warpers.cpp warpers_inl.hpp
                    main.cpp)

set(the_target opencv_stitching)    
add_executable(${the_target} ${stitching_files})

add_dependencies(${the_target} ${stitching_libs})
set_target_properties(${the_target} PROPERTIES
                      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
                      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
                      INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
                      OUTPUT_NAME "opencv_stitching")

target_link_libraries(${the_target} ${stitching_libs})

install(TARGETS ${the_target} RUNTIME DESTINATION bin COMPONENT main)
