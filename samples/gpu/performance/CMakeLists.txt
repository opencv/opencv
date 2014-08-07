set(the_target "example_gpu_performance")

file(GLOB sources "performance/*.cpp")
file(GLOB headers "performance/*.h")

if(HAVE_opencv_nonfree)
  ocv_include_directories("${OpenCV_SOURCE_DIR}/modules/nonfree/include")
endif()

add_executable(${the_target} ${sources} ${headers})
ocv_target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${OPENCV_CUDA_SAMPLES_REQUIRED_DEPS})

if(HAVE_opencv_nonfree)
  ocv_target_link_libraries(${the_target} opencv_nonfree)
endif()

set_target_properties(${the_target} PROPERTIES
    OUTPUT_NAME "performance_gpu"
    PROJECT_LABEL "(EXAMPLE_CUDA) performance")

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_target} PROPERTIES FOLDER "samples//gpu")
endif()

if(WIN32)
  install(TARGETS ${the_target} RUNTIME DESTINATION "${OPENCV_SAMPLES_BIN_INSTALL_PATH}/gpu" COMPONENT samples)
endif()

if(INSTALL_C_EXAMPLES AND NOT WIN32)
  file(GLOB CUDA_FILES performance/*.cpp performance/*.h)
  install(FILES ${CUDA_FILES}
          DESTINATION ${OPENCV_SAMPLES_SRC_INSTALL_PATH}/gpu/performance
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
          COMPONENT samples)
endif()
