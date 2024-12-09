set(TIMVX_COMMIT_HASH "1d9c7ab941b3d8d9c4d28d80058402725731e3d6")
set(OCV_TIMVX_DIR "${OpenCV_BINARY_DIR}/3rdparty/libtim-vx")
set(OCV_TIMVX_SOURCE_PATH "${OCV_TIMVX_DIR}/TIM-VX-${TIMVX_COMMIT_HASH}")

# Download TIM-VX source code
if(EXISTS "${OCV_TIMVX_SOURCE_PATH}")
    message(STATUS "TIM-VX: Use cache of TIM-VX source code at ${OCV_TIMVX_SOURCE_PATH}")
    set(TIMVX_FOUND ON)
else()
    set(OCV_TIMVX_FILENAME "${TIMVX_COMMIT_HASH}.zip")
    set(OCV_TIMVX_URL "https://github.com/VeriSilicon/TIM-VX/archive/")
    set(timvx_zip_md5sum 92619cc4498014ac7a09834d5e33ebd5)

    ocv_download(FILENAME ${OCV_TIMVX_FILENAME}
                 HASH ${timvx_zip_md5sum}
                 URL "${OCV_TIMVX_URL}"
                 DESTINATION_DIR "${OCV_TIMVX_DIR}"
                 ID "TIM-VX"
                 STATUS res
                 UNPACK RELATIVE_URL)
    if(res)
        set(TIMVX_FOUND ON)
        message(STATUS "TIM-VX: Source code downloaded at ${OCV_TIMVX_SOURCE_PATH}.") 
    else()
        set(TIMVX_FOUND OFF)
        message(STATUS "TIM-VX: Failed to download source code from github. Turning off TIMVX_FOUND")
        return()
    endif()
endif()

# set VIVANTE SDK especially for x86_64 which comes along with TIM-VX source code
if(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
    set(VIVANTE_SDK_DIR "${OCV_TIMVX_SOURCE_PATH}/prebuilt-sdk/x86_64_linux")
    message(STATUS "TIM-VX: Build from source using prebuilt x86_64 VIVANTE SDK.")
endif()

# Verify if requested VIVANTE SDK libraries are all found
find_vivante_sdk_libs(missing ${VIVANTE_SDK_DIR})
if(missing)
    message(STATUS "TIM-VX: Failed to find ${missing} in ${VIVANTE_SDK_DIR}/lib. Turning off TIMVX_VIV_FOUND")
    set(TIMVX_VIV_FOUND OFF)
else()
    message(STATUS "TIM-VX: dependent VIVANTE SDK libraries are found at ${VIVANTE_SDK_DIR}/lib.")
    set(TIMVX_VIV_FOUND ON)
endif()

if(TIMVX_VIV_FOUND)
    # vars used by TIM-VX CMake scripts
    set(EXTERNAL_VIV_SDK "${VIVANTE_SDK_DIR}" CACHE INTERNAL "" FORCE)
    set(VIV_SDK_DRIVER_PREFIX "lib" CACHE INTERNAL "" FORCE)
endif()

if(TIMVX_FOUND AND TIMVX_VIV_FOUND)
    set(BUILD_TIMVX ON)
else()
    return()
endif()

if(BUILD_TIMVX)
    set(HAVE_TIMVX 1)

    ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-parameter -Wstrict-prototypes -Wundef -Wsign-compare -Wmissing-prototypes -Wmissing-declarations -Wstrict-aliasing -Wunused-but-set-variable -Wmaybe-uninitialized -Wshadow -Wsuggest-override -Wswitch)
    ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-parameter -Wstrict-prototypes -Wundef -Wsign-compare -Wunused-but-set-variable -Wshadow -Wsuggest-override -Wmissing-declarations -Wswitch)

    set(TIMVX_INC_DIR "${OCV_TIMVX_SOURCE_PATH}/include" CACHE INTERNAL "TIM-VX include directory")
    if(EXISTS "${OCV_TIMVX_SOURCE_PATH}/CMakeLists.txt")
        add_subdirectory("${OCV_TIMVX_SOURCE_PATH}" "${OCV_TIMVX_DIR}/build")
    else()
        message(WARNING "TIM-VX: Missing 'CMakeLists.txt' in the source code: ${OCV_TIMVX_SOURCE_PATH}")
    endif()
    ocv_install_target(tim-vx EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
    set(TIMVX_LIB "tim-vx")
endif()
