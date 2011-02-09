#ENABLE_TESTING()

#add_subdirectory(cv)
#add_subdirectory(cxcore)
#add_subdirectory(ml)
#add_subdirectory(cxts)

#if(WITH_CUDA)
#    set (BUILD_TESTS_GPU OFF CACHE BOOL "Build tests GPU")
	
#    if(BUILD_TESTS_GPU AND WITH_CUDA)
        add_subdirectory(gpu)
#	 endif()
#endif()
