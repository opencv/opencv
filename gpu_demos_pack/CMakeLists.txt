cmake_minimum_required(VERSION 2.8.10)

project(gpu_demos_pack)

find_package(OpenCV REQUIRED)
include_directories("${OpenCV_INCLUDE_DIRS}")

include_directories("${CMAKE_SOURCE_DIR}/utility")

add_subdirectory(utility)

set(CPACK_PACKAGE_EXECUTABLES "" CACHE INTERNAL "" FORCE)
include(cmake/OpenCVGpuDemo.cmake)

add_subdirectory(demos)

############### Install ################

set(README_FILE "${CMAKE_SOURCE_DIR}/README.txt")
set(LICENSE_FILE "${CMAKE_SOURCE_DIR}/LICENSE.txt")

install(FILES ${README_FILE} ${LICENSE_FILE} DESTINATION ".")

if(WIN32)
    install(DIRECTORY "codecs" DESTINATION ".")

    string(REPLACE "lib" "bin" OpenCV_BIN_DIR ${OpenCV_LIB_DIR_OPT})
    file(GLOB ALL_OpenCV_DLLS "${OpenCV_BIN_DIR}/*.dll")
    install(FILES ${ALL_OpenCV_DLLS} DESTINATION ".")

    find_package(CUDA REQUIRED)
    set(CUDA_BIN_DIR "${CUDA_TOOLKIT_ROOT_DIR}/bin")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        file(GLOB CUBLAS_DLL "${CUDA_BIN_DIR}/cublas64*.dll")
        file(GLOB CUDART_DLL "${CUDA_BIN_DIR}/cudart64*.dll")
        file(GLOB CUFFT_DLL "${CUDA_BIN_DIR}/cufft64*.dll")
        file(GLOB NPP_DLL "${CUDA_BIN_DIR}/npp64*.dll")
    else()
        file(GLOB CUBLAS_DLL "${CUDA_BIN_DIR}/cublas32*.dll")
        file(GLOB CUDART_DLL "${CUDA_BIN_DIR}/cudart32*.dll")
        file(GLOB CUFFT_DLL "${CUDA_BIN_DIR}/cufft32*.dll")
        file(GLOB NPP_DLL "${CUDA_BIN_DIR}/npp32*.dll")
    endif()
    set(ALL_CUDA_DLLS ${CUBLAS_DLL} ${CUDART_DLL} ${CUFFT_DLL} ${NPP_DLL})
    install(FILES ${ALL_CUDA_DLLS} DESTINATION ".")
endif()

set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ".")
include(InstallRequiredSystemLibraries)

################ CPack #################

set(CPACK_PACKAGE_NAME "OpenCV GPU Demos Pack")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenCV ${OpenCV_VERSION} GPU Demos Pack")
set(CPACK_PACKAGE_VENDOR "Itseez")
set(CPACK_PACKAGE_FILE_NAME "OpenCV-${OpenCV_VERSION}-GPU-demos-pack-${CMAKE_SYSTEM_PROCESSOR}")
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
