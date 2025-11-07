ocv_update(ORBBEC_SDK_VERSION "2")
ocv_update(ORBBEC_SDK_DOWNLOAD_DIR "${OpenCV_BINARY_DIR}/3rdparty/orbbecsdk")

function(download_orbbec_sdk root_var)
    if(ORBBEC_SDK_VERSION STREQUAL "1")
        set(ORBBEC_SDK_FILE_HASH_CMAKE "e7566fa915a1b0c02640df41891916fe")
        set(ORBBEC_SDK_GIT_TAG "1.9.4")
        add_definitions(-DORBBEC_SDK_VERSION_MAJOR=1)
    elseif(ORBBEC_SDK_VERSION STREQUAL "2")
        set(ORBBEC_SDK_FILE_HASH_CMAKE "d828ac15618a56b9ae325bada8676e28")
        set(ORBBEC_SDK_GIT_TAG "2.5.5")
        add_definitions(-DORBBEC_SDK_VERSION_MAJOR=2)
    else()
        message(STATUS "Unsupported OrbbecSDK version: ${ORBBEC_SDK_VERSION}, use default version 2")
        set(ORBBEC_SDK_FILE_HASH_CMAKE "d828ac15618a56b9ae325bada8676e28")
        set(ORBBEC_SDK_GIT_TAG "2.5.5")
        add_definitions(-DORBBEC_SDK_VERSION_MAJOR=2)
    endif()

    ocv_download(FILENAME "v${ORBBEC_SDK_GIT_TAG}.tar.gz"
                HASH ${ORBBEC_SDK_FILE_HASH_CMAKE}
                URL "https://github.com/orbbec/OrbbecSDK/archive/refs/tags/v${ORBBEC_SDK_GIT_TAG}/"
                DESTINATION_DIR ${ORBBEC_SDK_DOWNLOAD_DIR}
                ID OrbbecSDK
                STATUS res
                UNPACK RELATIVE_URL
                )
    if(${res})
        message(STATUS "OrbbecSDK downloaded to: ${ORBBEC_SDK_DOWNLOAD_DIR}")
        set(${root_var} "${ORBBEC_SDK_DOWNLOAD_DIR}/OrbbecSDK-${ORBBEC_SDK_GIT_TAG}" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Failed to download OrbbecSDK")
    endif()
endfunction()