if(WITH_GAPI_ONEVPL)
    if (NOT ONEVPL_DIR)
        message("Search oneVPL package by PATH")
        find_package(VPL QUIET)
        if (VPL_FOUND)
                message("oneVPL found")
                set(HAVE_ONEVPL TRUE)
        else()
            message("No oneVPL found, clone it from repository")
            set(ONEVPL_PROJECT_PREFIX "${OpenCV_BINARY_DIR}/3rdparty/oneVPL")
            configure_file("${CMAKE_CURRENT_LIST_DIR}/DownloadOneVPL.cmake.in" ${ONEVPL_PROJECT_PREFIX}/oneVPL-download/CMakeLists.txt)
            execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                            RESULT_VARIABLE result
                            WORKING_DIRECTORY ${ONEVPL_PROJECT_PREFIX}/oneVPL-download )
            if(result)
                message(FATAL_ERROR "CMake step for oneVPL failed: ${result}")
            endif()

            execute_process(COMMAND ${CMAKE_COMMAND} --build .
                            RESULT_VARIABLE result
                            WORKING_DIRECTORY ${ONEVPL_PROJECT_PREFIX}/oneVPL-download )
            if(result)
                message(FATAL_ERROR "Build step for oneVPL failed: ${result}")
            endif()

            find_package(VPL REQUIRED PATHS "${ONEVPL_PROJECT_PREFIX}" NO_DEFAULT_PATH)
            set(HAVE_ONEVPL TRUE)
        endif()
    else ()
        message("Search oneVPL package in: ${ONEVPL_DIR}")
        find_package(VPL REQUIRED PATHS "${ONEVPL_DIR}" NO_DEFAULT_PATH)
        set(HAVE_ONEVPL TRUE)
    endif()
endif()
