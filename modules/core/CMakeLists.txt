if(ZLIB_FOUND)
    include_directories(${ZLIB_INCLUDE_DIR})
else()
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/zlib")
endif()

#define_opencv_module(core ${ZLIB_LIBRARY})

set(name "core")

project(opencv_${name})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src"
                    "${CMAKE_CURRENT_BINARY_DIR}")

file(GLOB lib_srcs "src/*.cpp")
file(GLOB lib_int_hdrs "src/*.h*")
file(GLOB lib_hdrs "include/opencv2/${name}/*.h*")
file(GLOB lib_hdrs_detail "include/opencv2/${name}/detail/*.h*")

if(COMMAND get_module_external_sources)
   get_module_external_sources(${name})
endif()

source_group("Src" FILES ${lib_srcs} ${lib_int_hdrs})
source_group("Include" FILES ${lib_hdrs})
source_group("Include\\detail" FILES ${lib_hdrs_detail})
list(APPEND lib_hdrs ${lib_hdrs_detail})

if (HAVE_CUDA)
    file(GLOB lib_cuda "src/cuda/*.cu")
    source_group("Cuda" FILES "${lib_cuda}")
 
    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories("${OpenCV_SOURCE_DIR}/modules/gpu/src")
    include_directories("${OpenCV_SOURCE_DIR}/modules/gpu/src/cuda")
 
    set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_10,code=sm_10
                                            -gencode arch=compute_11,code=sm_11
                                            -gencode arch=compute_12,code=sm_12
                                            -gencode arch=compute_13,code=sm_13
                                            -gencode arch=compute_20,code=sm_20
                                            -gencode arch=compute_20,code=sm_21)

    if (UNIX OR APPLE)
        set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" "-fPIC")
    endif()

    #set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-keep")
    #set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;/EHsc-;")
    
    if (APPLE)
        set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;-fno-finite-math-only;")
    endif()

    CUDA_COMPILE(cuda_objs ${lib_cuda})
    #CUDA_BUILD_CLEAN_TARGET()
endif()

set(the_target "opencv_${name}")
add_library(${the_target} ${lib_srcs} ${lib_hdrs} ${lib_int_hdrs} ${lib_cuda} ${cuda_objs})

# For dynamic link numbering convenions
if(NOT ANDROID)
    # Android SDK build scripts can include only .so files into final .apk
    # As result we should not set version properties for Android
    set_target_properties(${the_target} PROPERTIES
        VERSION ${OPENCV_VERSION}
        SOVERSION ${OPENCV_SOVERSION}
        )
endif()

set_target_properties(${the_target} PROPERTIES OUTPUT_NAME "${the_target}${OPENCV_DLLVERSION}" )    

if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${the_target} PROPERTIES FOLDER "modules")
endif() 
        
if (BUILD_SHARED_LIBS)
    if(MSVC)
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
target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${IPP_LIBS} ${ZLIB_LIBRARY})

if (HAVE_CUDA)
    target_link_libraries(${the_target} ${CUDA_LIBRARIES})

    unset(CUDA_npp_LIBRARY CACHE)
    find_cuda_helper_libs(npp)
    target_link_libraries(${the_target} ${CUDA_npp_LIBRARY})
endif()

if(MSVC)
    if(CMAKE_CROSSCOMPILING)
        set_target_properties(${the_target} PROPERTIES
            LINK_FLAGS "/NODEFAULTLIB:secchk"
            )
    endif()
    set_target_properties(${the_target} PROPERTIES
        LINK_FLAGS "/NODEFAULTLIB:libc /DEBUG"
        )
endif()

# Dependencies of this target:
add_dependencies(${the_target} ${ZLIB_LIBRARY})

install(TARGETS ${the_target}
    RUNTIME DESTINATION bin COMPONENT main
    LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT main
    ARCHIVE DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT main)

install(FILES ${lib_hdrs}
    DESTINATION ${OPENCV_INCLUDE_PREFIX}/opencv2/${name}
    COMPONENT main)
    
add_opencv_precompiled_headers(${the_target})

define_opencv_test(${name})
define_opencv_perf_test(${name})
