ocv_check_environment_variables(ASCENDCL_INSTALL_DIR)
ocv_check_environment_variables(ASCEND_DRIVER_DIR)

set(LIB_ASCEND_DRIVER_CANDIDATES "ascend_hal;c_sec;mmpa" CACHE STRING "Candidates of Ascend driver libraries")

function(find_ascend_driver_libs _not_found _search_path)
    foreach(one ${LIB_ASCEND_DRIVER_CANDIDATES})
        find_library(ASCEND_DRIVER_${one}_LIB ${one} PATHS ${_search_path} NO_DEFAULT_PATH)
        if(NOT ASCEND_DRIVER_${one}_LIB)
            list(APPEND _not_found_list ${one})
        endif()
    endforeach()
    set(${_not_found} ${_not_found_list} PARENT_SCOPE)
endfunction()

if(ASCENDCL_INSTALL_DIR AND ASCEND_DRIVER_DIR)
    # Supported platforms: x86-64, arm64
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
    else()
        message(FATAL_ERROR "AscendCL: AscendCL toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    # include
    set(inc_ascendcl "${ASCENDCL_INSTALL_DIR}/include")

    # libs
    #  * libascendcl.so
    set(lib_ascendcl "${ASCENDCL_INSTALL_DIR}/acllib/lib64")
    find_library(found_lib_ascendcl NAMES ascendcl PATHS ${lib_ascendcl} NO_DEFAULT_PATH)
    if(NOT found_lib_ascendcl)
        message(FATAL_ERROR "AscendCL: Missing libascendcl.so. Turning off HAVE_ASCENDCL")
        set(HAVE_ASCENDCL OFF)
        return()
    else()
        message(STATUS "AscendCL: libascendcl.so is found in ${lib_ascendcl}")
        set(lib_ascendcl ${found_lib_ascendcl})
    endif()
    #  * libacl_op_compiler
    set(lib_acl_op_compiler "${ASCENDCL_INSTALL_DIR}/compiler/lib64")
    find_library(found_lib_acl_op_compiler NAMES acl_op_compiler PATHS ${lib_acl_op_compiler} NO_DEFAULT_PATH)
    if(NOT found_lib_acl_op_compiler)
        message(FATAL_ERROR "AscendCL: Missing libacl_op_compiler.so. Turning off HAVE_ASCENDCL")
        set(HAVE_ASCENDCL OFF)
        return()
    else()
        message(STATUS "AscendCL: libacl_op_compiler.so is found in ${lib_acl_op_compiler}")
        set(lib_acl_op_compiler ${found_lib_acl_op_compiler})
    endif()
    #  * driver libs: libascend_hal.so, libc_sec.so, libmmpa.so
    set(lib_ascend_driver "${ASCEND_DRIVER_DIR}/lib64")
    find_ascend_driver_libs(not_found ${lib_ascend_driver})
    if(not_found)
        message(STATUS "AscendCL: Failed to find ${not_found} in ${lib_ascend_driver}. Turning off HAVE_ASCENDCL")
        set(HAVE_ASCENDCL OFF)
        return()
    else()
        message(STATUS "AscendCL: AscendCL driver libs are found in ${lib_ascend_driver}.")
    endif()

    set(libs_ascendcl "")
    list(APPEND libs_ascendcl ${lib_ascendcl})
    list(APPEND libs_ascendcl ${lib_acl_op_compiler})

    try_compile(VALID_ASCENDCL
        "${OpenCV_BINARY_DIR}"
        "${OpenCV_SOURCE_DIR}/cmake/checks/ascend.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${inc_ascendcl}"
                    "-DLINK_LIBRARIES:STRING=${libs_ascendcl}"
                    "-DLINK_DIRECTORIES:STRING=${lib_ascend_driver}"
        OUTPUT_VARIABLE ASCEND_TRY_OUT)

    if(NOT ${VALID_ASCENDCL})
        message(WARNING "Cannot use AscendCL")
        set(HAVE_ASCENDCL OFF)
        return()
    endif()

    set(HAVE_ASCENDCL ON)
endif()

if(HAVE_ASCENDCL)
    set(ASCENDCL_INCLUDE_DIRS ${inc_ascendcl})
    set(ASCENDCL_LIBRARIES ${libs_ascendcl})
    link_directories(${lib_ascend_driver})
endif()

MARK_AS_ADVANCED(
    inc_ascendcl
    libs_ascendcl
    lib_ascendcl
    lib_acl_op_compiler
    lib_ascend_driver
)
