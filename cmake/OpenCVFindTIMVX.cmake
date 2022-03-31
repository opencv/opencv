set(TIMVX_INSTALL_DIR "" CACHE PATH "Path to libtim-vx installation")
set(VIVANTE_SDK_DIR "" CACHE PATH "Path to VIVANTE SDK needed by TIM-VX.")
set(VIVANTE_SDK_LIB_CANDIDATES "OpenVX;VSC;GAL;ArchModelSw;NNArchPerf" CACHE STRING "VIVANTE SDK library candidates")

# Ensure VIVANTE SDK library candidates are present in given search path
function(find_vivante_sdk_libs _viv_notfound _viv_search_path)
    foreach(one ${VIVANTE_SDK_LIB_CANDIDATES})
        #NO_DEFAULT_PATH is used to ensure VIVANTE SDK libs are from one only source
        find_library(VIV_${one}_LIB ${one} PATHS "${_viv_search_path}/lib" NO_DEFAULT_PATH)
        if(NOT VIV_${one}_LIB)
            list(APPEND _viv_notfound_list ${one})
        endif()
    endforeach()
    set(${_viv_notfound} ${_viv_notfound_list} PARENT_SCOPE)
endfunction()
# Default value for VIVANTE_SDK_DIR: /usr
if(NOT VIVANTE_SDK_DIR)
    set(VIVANTE_SDK_DIR "/usr")
endif()
# Environment variable VIVANTE_SDK_DIR overrides the one in this script
if(DEFINED ENV{VIVANTE_SDK_DIR})
    set(VIVANTE_SDK_DIR $ENV{VIVANTE_SDK_DIR})
    message(STATUS "TIM-VX: Load VIVANTE_SDK_DIR from system environment: ${VIVANTE_SDK_DIR}")
endif()


# Compile with pre-installed TIM-VX; Or compile together with TIM-VX from source
if(TIMVX_INSTALL_DIR AND NOT BUILD_TIMVX)
    message(STATUS "TIM-VX: Use binaries at ${TIMVX_INSTALL_DIR}")
    set(BUILD_TIMVX OFF)

    set(TIMVX_INC_DIR "${TIMVX_INSTALL_DIR}/include" CACHE INTERNAL "TIM-VX include directory")
    find_library(TIMVX_LIB "tim-vx" PATHS "${TIMVX_INSTALL_DIR}/lib")
    if(TIMVX_LIB)
        set(TIMVX_FOUND ON)
    else()
        set(TIMVX_FOUND OFF)
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
else()
    message(STATUS "TIM-VX: Build from source")
    include("${OpenCV_SOURCE_DIR}/3rdparty/libtim-vx/tim-vx.cmake")
endif()

if(TIMVX_FOUND AND TIMVX_VIV_FOUND)
    set(HAVE_TIMVX 1)

    message(STATUS "TIM-VX: Found TIM-VX includes: ${TIMVX_INC_DIR}")
    message(STATUS "TIM-VX: Found TIM-VX library: ${TIMVX_LIB}")
    set(TIMVX_LIBRARY   ${TIMVX_LIB})
    set(TIMVX_INCLUDE_DIR   ${TIMVX_INC_DIR})

    message(STATUS "TIM-VX: Found VIVANTE SDK libraries: ${VIVANTE_SDK_DIR}/lib")
    link_directories(${VIVANTE_SDK_DIR}/lib)
endif()

MARK_AS_ADVANCED(
	TIMVX_INC_DIR
	TIMVX_LIB
)
