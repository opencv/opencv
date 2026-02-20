# Module: hip
# OpenCV GPU acceleration using HIP Threads (ROCm)

set(the_description "GPU acceleration using HIP Threads")
ocv_define_module(hip OPTIONAL)

if(NOT HAVE_HIP)
  message(WARNING "HIP runtime not found. GPU acceleration will be disabled.")
  return()
endif()

# Find HIP
find_package(HIP REQUIRED)
find_package(hipthreads REQUIRED)

# Source files
set(HIP_SRCS
    src/hip_dispatcher.cpp
    src/hip_kernels.cpp
    src/hip_gaussian_blur.cpp
    src/hip_resize.cpp
    src/hip_color_convert.cpp
    src/hip_config.cpp
)

# Headers
set(HIP_HEADERS
    include/opencv2/hip/hip_dispatcher.hpp
    include/opencv2/hip/hip_kernels.hpp
    include/opencv2/hip/hip_config.hpp
)

# Create the module
ocv_add_module(hip
    SOURCES ${HIP_SRCS}
    HEADERS ${HIP_HEADERS}
    DEPENDS core imgproc
)

# Add include directory for opencv2 headers from this module
target_include_directories(${the_module} 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link HIP and HIP Threads
target_link_libraries(${the_module} 
    PUBLIC HIP::HIP hipthreads::hipthreads
    PRIVATE opencv_core opencv_imgproc
)

target_include_directories(${the_module} PRIVATE ${HIP_INCLUDE_DIR})

# Compiler flags for HIP
if(HIP_COMPILER MATCHES "hipcc")
    target_compile_options(${the_module} PRIVATE -fPIC)
    set_source_files_properties(${HIP_SRCS} PROPERTIES LANGUAGE CXX)
endif()

# Tests
if(BUILD_TESTS)
    ocv_add_accuracy_tests()
    ocv_add_perf_tests()
endif()
