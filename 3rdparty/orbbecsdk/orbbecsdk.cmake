function(download_orbbec_sdk root_var)

    # detect git
    find_package(Git)
    if(NOT GIT_FOUND)
        message(STATUS "Git not found, cannot download orbbec sdk")
        return()
    endif()

    # download orbbec sdk
    set(THE_ROOT "${OpenCV_BINARY_DIR}/3rdparty/orbbecsdk")
    set(THE_URL "https://github.com/orbbec/OrbbecSDK.git")

    if(EXISTS "${THE_ROOT}/.git")
        # update repo
        message(STATUS "Updating orbbec sdk")
        execute_process(COMMAND ${GIT_EXECUTABLE} pull
                            WORKING_DIRECTORY "${THE_ROOT}"
                            RESULT_VARIABLE res)
        message(STATUS "Updating orbbec sdk - done (${res})")
    else()
        # clone repo
        message(STATUS "Cloning orbbec sdk")
        execute_process(COMMAND ${GIT_EXECUTABLE} clone "${THE_URL}" "${THE_ROOT}"
                            RESULT_VARIABLE res)
        message(STATUS "Cloning orbbec sdk - done (${res})")
    endif()
    if(${res})
        message(FATAL_ERROR "Failed to download orbbec sdk")
    else()
        message(STATUS "orbbec sdk downloaded to: ${THE_ROOT}")
        set(${root_var} "${THE_ROOT}" PARENT_SCOPE)
    endif()
endfunction()