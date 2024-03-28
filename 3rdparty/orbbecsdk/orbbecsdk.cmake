function(download_orbbec_sdk root_var)
    set(ORBBECSDK_DOWNLOAD_DIR "${OpenCV_BINARY_DIR}/3rdparty/orbbecsdk")
    set(ORBBECSDK_FILE_HASH_CMAKE "2624c84837d3416fd8b3e95750e6e725")
    ocv_download(FILENAME "v1.9.4.tar.gz"
                HASH ${ORBBECSDK_FILE_HASH_CMAKE}
                URL "https://github.com/orbbec/OrbbecSDK/archive/refs/tags/v1.9.4/"
                DESTINATION_DIR ${ORBBECSDK_DOWNLOAD_DIR}
                ID OrbbecSDK
                STATUS res
                UNPACK RELATIVE_URL
                )
    if(${res})
        message(STATUS "orbbec sdk downloaded to: ${ORBBECSDK_DOWNLOAD_DIR}")
        set(${root_var} "${ORBBECSDK_DOWNLOAD_DIR}/OrbbecSDK-1.9.4" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Failed to download orbbec sdk")
    endif()
endfunction()