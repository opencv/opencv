set(sample_dir ${CMAKE_CURRENT_SOURCE_DIR}/tutorial_code/calib3d/real_time_pose_estimation/src/)
set(target cpp-tutorial-)

if(HAVE_opencv_nonfree)
  ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/nonfree/include")
endif()

set(sample_pnplib
        ${sample_dir}CsvReader.cpp
        ${sample_dir}CsvWriter.cpp
        ${sample_dir}ModelRegistration.cpp
        ${sample_dir}Mesh.cpp
        ${sample_dir}Model.cpp
        ${sample_dir}PnPProblem.cpp
        ${sample_dir}Utils.cpp
        ${sample_dir}RobustMatcher.cpp
)

add_executable( ${target}pnp_registration ${sample_dir}main_registration.cpp ${sample_pnplib} )
add_executable( ${target}pnp_verification ${sample_dir}main_verification.cpp ${sample_pnplib} )
add_executable( ${target}pnp_detection ${sample_dir}main_detection.cpp ${sample_pnplib} )
add_executable( ${target}pnp_test ${sample_dir}test_pnp.cpp )

ocv_target_link_libraries( ${target}pnp_registration ${OPENCV_LINKER_LIBS} ${OPENCV_CPP_SAMPLES_REQUIRED_DEPS} )
ocv_target_link_libraries( ${target}pnp_verification ${OPENCV_LINKER_LIBS} ${OPENCV_CPP_SAMPLES_REQUIRED_DEPS} )
ocv_target_link_libraries( ${target}pnp_detection ${OPENCV_LINKER_LIBS} ${OPENCV_CPP_SAMPLES_REQUIRED_DEPS} )
ocv_target_link_libraries(${target}pnp_test ${OPENCV_LINKER_LIBS} ${OPENCV_CPP_SAMPLES_REQUIRED_DEPS} )
