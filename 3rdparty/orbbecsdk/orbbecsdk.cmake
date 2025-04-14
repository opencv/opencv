function(download_orbbec_sdk root_var)
    set(ORBBECSDK_DOWNLOAD_DIR "${OpenCV_BINARY_DIR}/3rdparty/orbbecsdk")
    
    if(EXISTS ${ORBBECSDK_DOWNLOAD_DIR})
        message(STATUS "Orbbec SDK already exists at: ${ORBBECSDK_DOWNLOAD_DIR}, skipping clone.")
        set(${root_var} "${ORBBECSDK_DOWNLOAD_DIR}" PARENT_SCOPE)
    else()

        execute_process(
            COMMAND git clone --branch feature/2.x https://github.com/orbbec/OrbbecSDK.git ${ORBBECSDK_DOWNLOAD_DIR}
            RESULT_VARIABLE res
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(NOT res)
            message(STATUS "Orbbec SDK cloned to: ${ORBBECSDK_DOWNLOAD_DIR}")
            set(${root_var} "${ORBBECSDK_DOWNLOAD_DIR}" PARENT_SCOPE)
        else()
            message(FATAL_ERROR "Failed to clone Orbbec SDK")
        endif()
    endif()
endfunction()