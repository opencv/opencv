set(the_target "example_gpu_performance")

file(GLOB sources "performance/*.cpp")
file(GLOB headers "performance/*.h")

if(HAVE_opencv_xfeatures2d)
  ocv_include_modules_recurse(opencv_xfeatures2d)
endif()

if(HAVE_opencv_bgsegm)
  ocv_include_modules_recurse(opencv_bgsegm)
endif()

add_executable(${the_target} ${sources} ${headers})
ocv_target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${OPENCV_CUDA_SAMPLES_REQUIRED_DEPS})

if(HAVE_opencv_xfeatures2d)
  ocv_target_link_libraries(${the_target} opencv_xfeatures2d)
endif()

if(HAVE_opencv_bgsegm)
  ocv_target_link_libraries(${the_target} opencv_bgsegm)
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

ocv_install_example_src("gpu/performance" performance/*.cpp performance/*.h)
