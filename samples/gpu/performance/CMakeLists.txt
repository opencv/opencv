set(the_target "example_gpu_performance")

file(GLOB sources "performance/*.cpp")
file(GLOB headers "performance/*.h")

add_executable(${the_target} ${sources} ${headers})
target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${OPENCV_GPU_SAMPLES_REQUIRED_DEPS})

set_target_properties(${the_target} PROPERTIES
    OUTPUT_NAME "performance_gpu"
    PROJECT_LABEL "(EXAMPLE_GPU) performance")
    
if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_target} PROPERTIES FOLDER "samples//gpu")
endif()	

if(WIN32)
  install(TARGETS ${the_target} RUNTIME DESTINATION "samples/gpu" COMPONENT main)
endif()

if(NOT WIN32)
  file(GLOB GPU_FILES performance/*.cpp performance/*.h)
  install(FILES ${GPU_FILES}
          DESTINATION share/opencv/samples/gpu/performance
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
endif()
