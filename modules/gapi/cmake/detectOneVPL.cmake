if(WITH_GAPI_ONEVPL)
  find_package(VPL QUIET HINTS "${ONEVPLROOT}")
  if (VPL_FOUND)
        message("oneVPL found")
        set(HAVE_GAPI_ONEVPL TRUE)
  else()
    message("No oneVPL found, clone it from repository")
    set(ONEVPL_PROJECT_PREFIX "${OpenCV_BINARY_DIR}/3rdparty/oneVPL")
    configure_file(gapi/cmake/DownloadOneVPL.cmake.in ${ONEVPL_PROJECT_PREFIX}/oneVPL-download/CMakeLists.txt)
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

    find_package(VPL REQUIRED HINTS "${ONEVPL_PROJECT_PREFIX}")
    set(HAVE_GAPI_ONEVPL TRUE)
  endif()
endif()
