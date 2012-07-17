if(APPLE)
    set(OPENCL_FOUND YES)
    set(OPENCL_LIBRARIES "-framework OpenCL")
else()
    find_package(OpenCL)

    # Try AMD/ATI Stream SDK
    if (NOT OPENCL_FOUND)
        set(ENV_AMDSTREAMSDKROOT $ENV{AMDAPPSDKROOT})
        set(ENV_OPENCLROOT $ENV{OPENCLROOT})
        set(ENV_CUDA_PATH $ENV{CUDA_PATH})
        if(ENV_AMDSTREAMSDKROOT)
            set(OPENCL_INCLUDE_SEARCH_PATH ${ENV_AMDSTREAMSDKROOT}/include)
            if(CMAKE_SIZEOF_VOID_P EQUAL 4)
                set(OPENCL_LIB_SEARCH_PATH ${OPENCL_LIB_SEARCH_PATH} ${ENV_AMDSTREAMSDKROOT}/lib/x86)
            else()
                set(OPENCL_LIB_SEARCH_PATH ${OPENCL_LIB_SEARCH_PATH} ${ENV_AMDSTREAMSDKROOT}/lib/x86_64)
            endif()
        elseif(ENV_CUDAPATH AND WIN32)
            set(OPENCL_INCLUDE_SEARCH_PATH ${ENV_CUDA_PATH}/include)
            if(CMAKE_SIZEOF_VOID_P EQUAL 4)
                set(OPENCL_LIB_SEARCH_PATH ${OPENCL_LIB_SEARCH_PATH} ${ENV_CUDA_PATH}/lib/Win32)
            else()
                set(OPENCL_LIB_SEARCH_PATH ${OPENCL_LIB_SEARCH_PATH} ${ENV_CUDA_PATH}/lib/x64)
            endif()
        elseif(ENV_OPENCLROOT AND UNIX)
            set(OPENCL_INCLUDE_SEARCH_PATH ${ENV_OPENCLROOT}/inc)
            if(CMAKE_SIZEOF_VOID_P EQUAL 4)
                set(OPENCL_LIB_SEARCH_PATH ${OPENCL_LIB_SEARCH_PATH} /usr/lib)
            else()
                set(OPENCL_LIB_SEARCH_PATH ${OPENCL_LIB_SEARCH_PATH} /usr/lib64)
            endif()
        endif()
    
        if(OPENCL_INCLUDE_SEARCH_PATH)
            find_path(OPENCL_INCLUDE_DIR
                NAMES CL/cl.h OpenCL/cl.h
                PATHS ${OPENCL_INCLUDE_SEARCH_PATH}
                NO_DEFAULT_PATH)
        else()
            find_path(OPENCL_INCLUDE_DIR
                NAMES CL/cl.h OpenCL/cl.h)
        endif()
    
        if(OPENCL_LIB_SEARCH_PATH)
            find_library(OPENCL_LIBRARY NAMES OpenCL PATHS ${OPENCL_LIB_SEARCH_PATH} NO_DEFAULT_PATH)
        else()
            find_library(OPENCL_LIBRARY NAMES OpenCL PATHS ${OPENCL_LIB_SEARCH_PATH} NO_DEFAULT_PATH)
        endif()

        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(
          OPENCL
          DEFAULT_MSG
          OPENCL_LIBRARY OPENCL_INCLUDE_DIR
          )

        if(OPENCL_FOUND)
            set(OPENCL_LIBRARIES ${OPENCL_LIBRARY})
			set(HAVE_OPENCL 1)
        else()
            set(OPENCL_LIBRARIES)
        endif()
	else()
		set(HAVE_OPENCL 1)
    endif()
endif()
