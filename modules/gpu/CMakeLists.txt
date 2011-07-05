set(name "gpu")

set(the_target "opencv_${name}")
project(${the_target})

set(DEPS "opencv_core" "opencv_imgproc" "opencv_objdetect" "opencv_features2d" "opencv_flann" "opencv_calib3d") #"opencv_features2d" "opencv_flann" "opencv_objdetect" - only headers needed 
set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} opencv_gpu)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include"                    
					"${CMAKE_CURRENT_SOURCE_DIR}/src/cuda"                    
					"${CMAKE_CURRENT_SOURCE_DIR}/src"
					"${CMAKE_CURRENT_BINARY_DIR}")

file(GLOB lib_srcs "src/*.cpp")
file(GLOB lib_int_hdrs "src/*.h*")
file(GLOB lib_cuda "src/cuda/*.cu*")
file(GLOB lib_cuda_hdrs "src/cuda/*.h*")
source_group("Src\\Host" FILES ${lib_srcs} ${lib_int_hdrs})
source_group("Src\\Cuda" FILES ${lib_cuda} ${lib_cuda_hdrs})

file(GLOB lib_hdrs "include/opencv2/${name}/*.h*")
source_group("Include" FILES ${lib_hdrs})

#file(GLOB lib_device_hdrs "include/opencv2/${name}/device/*.h*")
file(GLOB lib_device_hdrs "src/opencv2/gpu/device/*.h*")
source_group("Device" FILES ${lib_device_hdrs})

if (HAVE_CUDA)
    file(GLOB_RECURSE ncv_srcs "src/nvidia/*.cpp")	
    file(GLOB_RECURSE ncv_cuda "src/nvidia/*.cu")
    file(GLOB_RECURSE ncv_hdrs "src/nvidia/*.hpp" "src/nvidia/*.h")
    
    source_group("Src\\NVidia" FILES ${ncv_srcs} ${ncv_hdrs} ${ncv_cuda})
    include_directories("src/nvidia/core" "src/nvidia/NPP_staging")
endif()

if (HAVE_CUDA)       
    include_directories(${CUDA_INCLUDE_DIRS})

    if (UNIX OR APPLE)
        set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" "-fPIC")
    endif()

    #set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-keep")
    #set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;/EHsc-;")
    
    if (APPLE)
        set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;-fno-finite-math-only;")
    endif()
    

    string(REPLACE "/W4" "/W3" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/W4" "/W3" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/W4" "/W3" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

    if(MSVC)
        #string(REPLACE "/W4" "/W3" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        #string(REPLACE "/W4" "/W3" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        #string(REPLACE "/W4" "/W3" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4211 /wd4201 /wd4100 /wd4505 /wd4408")

        string(REPLACE "/EHsc-" "/EHs" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        string(REPLACE "/EHsc-" "/EHs" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
        string(REPLACE "/EHsc-" "/EHs" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
        string(REPLACE "/EHsc-" "/EHs" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REPLACE "/EHsc-" "/EHs" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        string(REPLACE "/EHsc-" "/EHs" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    endif()
	
    if (BUILD_SHARED_LIBS)
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;-DCVAPI_EXPORTS")
	endif()
    
    CUDA_COMPILE(cuda_objs ${lib_cuda} ${ncv_cuda})
    #CUDA_BUILD_CLEAN_TARGET()
endif()

foreach(d ${DEPS})
	if(${d} MATCHES "opencv_")
		string(REPLACE "opencv_" "${CMAKE_CURRENT_SOURCE_DIR}/../" d_dir ${d})
		include_directories("${d_dir}/include")
	endif()
endforeach()

add_library(${the_target} ${lib_srcs} ${lib_hdrs} ${lib_int_hdrs} ${lib_cuda} ${lib_cuda_hdrs} ${lib_device_hdrs} ${ncv_srcs} ${ncv_hdrs} ${ncv_cuda} ${cuda_objs})

if(PCHSupport_FOUND)
	set(pch_header ${CMAKE_CURRENT_SOURCE_DIR}/src/precomp.hpp)
	if(${CMAKE_GENERATOR} MATCHES "Visual*" OR ${CMAKE_GENERATOR} MATCHES "Xcode*")
		if(${CMAKE_GENERATOR} MATCHES "Visual*")
			set(${the_target}_pch "src/precomp.cpp")
		endif()
		add_native_precompiled_header(${the_target} ${pch_header})
	elseif(CMAKE_COMPILER_IS_GNUCXX AND ${CMAKE_GENERATOR} MATCHES ".*Makefiles")
		add_precompiled_header(${the_target} ${pch_header})
	endif()
endif()

# For dynamic link numbering convenions
set_target_properties(${the_target} PROPERTIES
	VERSION ${OPENCV_VERSION}
	SOVERSION ${OPENCV_SOVERSION}
	OUTPUT_NAME "${the_target}${OPENCV_DLLVERSION}"
	)

if(ENABLE_SOLUTION_FOLDERS)
	set_target_properties(${the_target} PROPERTIES FOLDER "modules")
endif()	
	
if (BUILD_SHARED_LIBS)
	if (MSVC)
		set_target_properties(${the_target} PROPERTIES DEFINE_SYMBOL CVAPI_EXPORTS)
	else()
		add_definitions(-DCVAPI_EXPORTS)	   
	endif()	
endif()

# Additional target properties
set_target_properties(${the_target} PROPERTIES
	DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    INSTALL_NAME_DIR lib
	)

# Add the required libraries for linking:
target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${IPP_LIBS} ${DEPS} )

if (HAVE_CUDA)
	target_link_libraries(${the_target} ${CUDA_LIBRARIES})    
    CUDA_ADD_CUFFT_TO_TARGET(${the_target})
	
	unset(CUDA_npp_LIBRARY CACHE)
	find_cuda_helper_libs(npp)
	target_link_libraries(${the_target} ${CUDA_npp_LIBRARY})    		
endif()

if(MSVC)
	if(CMAKE_CROSSCOMPILING)
		set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:secchk")
	endif()
    
	set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:libc")
endif()

# Dependencies of this target:
add_dependencies(${the_target} ${DEPS})

install(TARGETS ${the_target}
	RUNTIME DESTINATION bin COMPONENT main
	LIBRARY DESTINATION lib COMPONENT main
	ARCHIVE DESTINATION lib COMPONENT main)

install(FILES ${lib_hdrs}
	DESTINATION include/opencv2/${name}
	COMPONENT main)

install(FILES src/nvidia/NPP_staging/NPP_staging.hpp  src/nvidia/core/NCV.hpp
	DESTINATION include/opencv2/${name}
	COMPONENT main)

#install(FILES ${lib_device_hdrs}
#	DESTINATION include/opencv2/${name}/device
#	COMPONENT main)


################################################################################################################
################################      GPU Module Tests     #####################################################
################################################################################################################

# Test files processing is in the separated directory to avoid 'Src' source 
# filter creation in Visual Studio 
if(BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test)
    set(the_test_target "opencv_test_${name}")   
    
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include"
                        "${CMAKE_CURRENT_SOURCE_DIR}/test"
                        "${CMAKE_CURRENT_BINARY_DIR}")                      
    
    set(test_deps opencv_${name} opencv_ts opencv_highgui opencv_calib3d ${DEPS})
    foreach(d ${test_deps})
        if(${d} MATCHES "opencv_")
            if(${d} MATCHES "opencv_lapack")
            else()
                string(REPLACE "opencv_" "${CMAKE_CURRENT_SOURCE_DIR}/../" d_dir ${d})
                include_directories("${d_dir}/include")
            endif()
        endif()
    endforeach()
    
    file(GLOB test_srcs "test/*.cpp")
    file(GLOB test_hdrs "test/*.h*")
	
	source_group("Src" FILES ${test_hdrs} ${test_srcs})
    if(HAVE_CUDA)
		include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_SOURCE_DIR}/modules/gpu/src/nvidia ${OpenCV_SOURCE_DIR}/modules/gpu/src/nvidia/core ${OpenCV_SOURCE_DIR}/modules/gpu/src/nvidia/NPP_staging)

		file(GLOB nvidia "test/nvidia/*.cpp" "test/nvidia/*.h*")            
		source_group("Src\\NVidia" FILES ${nvidia})
    endif()
    
    
    add_executable(${the_test_target} ${test_srcs} ${test_hdrs} ${nvidia})
    
    if(PCHSupport_FOUND)
        set(pch_header ${CMAKE_CURRENT_SOURCE_DIR}/test/test_precomp.hpp)
        if(${CMAKE_GENERATOR} MATCHES "Visual*" OR ${CMAKE_GENERATOR} MATCHES "Xcode*")
            if(${CMAKE_GENERATOR} MATCHES "Visual*")
                set(${the_test_target}_pch "test/test_precomp.cpp")
            endif()            
            add_native_precompiled_header(${the_test_target} ${pch_header})
        elseif(CMAKE_COMPILER_IS_GNUCXX AND ${CMAKE_GENERATOR} MATCHES ".*Makefiles")
            add_precompiled_header(${the_test_target} ${pch_header})
        endif()
    endif()
    
    # Additional target properties
    set_target_properties(${the_test_target} PROPERTIES
        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
        RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
        )
	
	if(ENABLE_SOLUTION_FOLDERS)
		set_target_properties(${the_test_target} PROPERTIES FOLDER "tests")
	endif()			
    
    add_dependencies(${the_test_target} ${test_deps})
    
    # Add the required libraries for linking:
    target_link_libraries(${the_test_target} ${OPENCV_LINKER_LIBS} ${test_deps})
    
    enable_testing()
    get_target_property(LOC ${the_test_target} LOCATION)
    add_test(${the_test_target} "${LOC}")
    
    #if(WIN32)
    #    install(TARGETS ${the_test_target} RUNTIME DESTINATION bin COMPONENT main)
    #endif()
endif()
