############### Project ################

cmake_minimum_required(VERSION 2.8.10)

project(gpu_demos_pack)

find_package(OpenCV REQUIRED)
include_directories("${OpenCV_INCLUDE_DIRS}")

include_directories("${CMAKE_SOURCE_DIR}/utility")

add_subdirectory(utility)

if(WIN32)
    set(CPACK_PACKAGE_EXECUTABLES "" CACHE INTERNAL "" FORCE)
endif()

include(cmake/OpenCVGpuDemo.cmake)

add_subdirectory(demos)

############### Install ################

set(README_FILE  "${CMAKE_SOURCE_DIR}/README.txt")
set(LICENSE_FILE "${CMAKE_SOURCE_DIR}/LICENSE.txt")
install(FILES ${README_FILE} ${LICENSE_FILE} DESTINATION ".")

if(WIN32)
    install(DIRECTORY "codecs" DESTINATION ".")
endif()

if(WIN32)
    option(INTALL_OPENCV_LIBS "Install OpenCV shared libraries" ON)
    option(INTALL_CUDA_LIBS "Install CUDA shared libraries" ON)
else()
    option(INTALL_OPENCV_LIBS "Install OpenCV shared libraries" OFF)
    option(INTALL_CUDA_LIBS "Install CUDA shared libraries" OFF)
endif()

if(INTALL_OPENCV_LIBS)
    if(WIN32)
        string(REPLACE "lib" "bin" OpenCV_BIN_DIR ${OpenCV_LIB_DIR_OPT})
        file(GLOB ALL_OpenCV_LIBS "${OpenCV_BIN_DIR}/*.dll")
    else()
        file(GLOB ALL_OpenCV_LIBS "${OpenCV_LIB_DIR_OPT}/*.so*")
    endif()

    install(FILES ${ALL_OpenCV_LIBS} DESTINATION ".")
endif()

if(INTALL_CUDA_LIBS)
    find_package(CUDA ${OpenCV_CUDA_VERSION} EXACT REQUIRED)

    if(WIN32)
        set(CUDA_BIN_DIR "${CUDA_TOOLKIT_ROOT_DIR}/bin")

        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            file(GLOB CUDART_LIB "${CUDA_BIN_DIR}/cudart64*.dll")
            file(GLOB NPP_LIB "${CUDA_BIN_DIR}/npp64*.dll")
            file(GLOB CUBLAS_LIB "${CUDA_BIN_DIR}/cublas64*.dll")
            file(GLOB CUFFT_LIB "${CUDA_BIN_DIR}/cufft64*.dll")
        else()
            file(GLOB CUDART_LIB "${CUDA_BIN_DIR}/cudart32*.dll")
            file(GLOB NPP_LIB "${CUDA_BIN_DIR}/npp32*.dll")
            file(GLOB CUBLAS_LIB "${CUDA_BIN_DIR}/cublas32*.dll")
            file(GLOB CUFFT_LIB "${CUDA_BIN_DIR}/cufft32*.dll")
        endif()
    else()
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
        else()
            set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib")
        endif()

        file(GLOB CUDART_LIB "${CUDA_LIB_DIR}/libcudart.so*")
        file(GLOB NPP_LIB "${CUDA_LIB_DIR}/libnpp.so*")
        file(GLOB CUBLAS_LIB "${CUDA_LIB_DIR}/libcublas.so*")
        file(GLOB CUFFT_LIB "${CUDA_LIB_DIR}/libcufft.so*")
    endif()

    set(ALL_CUDA_LIBS ${CUDART_LIB} ${NPP_LIB})
    if(OpenCV_USE_CUBLAS)
        list(APPEND ALL_CUDA_LIBS ${CUBLAS_LIB})
    endif()
    if(OpenCV_USE_CUFFT)
        list(APPEND ALL_CUDA_LIBS ${CUFFT_LIB})
    endif()

    install(FILES ${ALL_CUDA_DLLS} DESTINATION ".")
endif()

set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ".")
include(InstallRequiredSystemLibraries)

################ CPack #################

set(CPACK_PACKAGE_NAME "OpenCV GPU Demos Pack")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenCV ${OpenCV_VERSION} GPU Demos Pack")
set(CPACK_PACKAGE_VENDOR "Itseez")

if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(CPACK_PACKAGE_FILE_NAME "OpenCV-${OpenCV_VERSION}-GPU-demos-pack-win64")
    else()
        set(CPACK_PACKAGE_FILE_NAME "OpenCV-${OpenCV_VERSION}-GPU-demos-pack-win32")
    endif()
else()
    set(CPACK_PACKAGE_FILE_NAME "OpenCV-${OpenCV_VERSION}-GPU-demos-pack-${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_FILE_NAME})
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY ${CPACK_PACKAGE_FILE_NAME})

set(CPACK_PACKAGE_VERSION_MAJOR "${OpenCV_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${OpenCV_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${OpenCV_VERSION_PATCH}")

set(CPACK_PACKAGE_DESCRIPTION_FILE "${README_FILE}")
set(CPACK_RESOURCE_FILE_README "${README_FILE}")
set(CPACK_RESOURCE_FILE_LICENSE "${LICENSE_FILE}")

set(CPACK_MONOLITHIC_INSTALL TRUE)

if(WIN32)
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_SOURCE_GENERATOR "ZIP")

    set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}\\\\opencv.ico")
    set(CPACK_NSIS_MUI_ICON ${CPACK_PACKAGE_ICON})
    set(CPACK_NSIS_MUI_UNIICON ${CPACK_PACKAGE_ICON})

    set(CPACK_NSIS_EXECUTABLES_DIRECTORY ".")

    set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "ExecWait \\\"$INSTDIR\\\\codecs\\\\ffdshow_beta7_rev3154_20091209.exe\\\"")
else()
    set(CPACK_GENERATOR "TGZ")
    set(CPACK_SOURCE_GENERATOR "TGZ")
endif()

include(CPack)
