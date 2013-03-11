function(ocv_disable_gpu_demo DEMO_NAME)
    set(${DEMO_NAME}_DISABLED TRUE CACHE INTERNAL "" FORCE)
endfunction()

function(ocv_define_gpu_demo DEMO_NAME)
    include(CMakeParseArguments)
    set(options "")
    set(oneValueArgs DESCRIPTION)
    set(multiValueArgs SOURCES INCLUDE_DIRS DEPENDENCIES LINK_LIBRARIES)
    cmake_parse_arguments(DEMO "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT DEMO_DESCRIPTION)
        set(DEMO_DESCRIPTION ${DEMO_NAME})
    endif()

    set(target "demo_${DEMO_NAME}")

    if(${DEMO_NAME}_DISABLED)
        option(BUILD_${target} "Build ${DEMO_DESCRIPTION}" OFF)
    else()
        option(BUILD_${target} "Build ${DEMO_DESCRIPTION}" ON)
    endif()

    if(BUILD_${target})
        message(STATUS "Configure ${DEMO_NAME} - ${DEMO_DESCRIPTION}")

        if(NOT DEMO_SOURCES)
            file(GLOB DEMO_SOURCES
                "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
                "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
        endif()

        if(DEMO_INCLUDE_DIRS)
            include_directories(${DEMO_INCLUDE_DIRS})
        endif()

        add_executable(${target} ${DEMO_SOURCES})

        add_dependencies(${target} utility ${DEMO_DEPENDENCIES})

        target_link_libraries(${target} ${OpenCV_LIBS} utility ${DEMO_LINK_LIBRARIES})

        install(TARGETS ${target} RUNTIME DESTINATION ".")
        install(DIRECTORY "data" DESTINATION ".")

        if(WIN32)
            set(CPACK_PACKAGE_EXECUTABLES ${CPACK_PACKAGE_EXECUTABLES} "${target};${DEMO_DESCRIPTION}" CACHE INTERNAL "")
        endif()
    endif()
endfunction()
