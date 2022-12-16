ocv_check_environment_variables(CANN_INSTALL_DIR)

if("cann${CANN_INSTALL_DIR}" STREQUAL "cann" AND DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(CANN_INSTALL_DIR $ENV{ASCEND_TOOLKIT_HOME})
    message(STATUS "CANN: updated CANN_INSTALL_DIR from ASCEND_TOOLKIT_HOME=$ENV{ASCEND_TOOLKIT_HOME}")
endif()

if(CANN_INSTALL_DIR)
    # Supported platforms: x86-64, arm64
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
    else()
        set(HAVE_CANN OFF)
        message(STATUS "CANN: CANN toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}. Turning off HAVE_CANN")
        return()
    endif()

    # Supported OS: linux (because of we need fork() to build models in child process)
    # done via checks in cann.cpp
    # FIXME: remove the check if a better model building solution is found

    # include
    set(incs_cann "${CANN_INSTALL_DIR}/include")
    list(APPEND incs_cann "${CANN_INSTALL_DIR}/opp")

    # libs
    #  * libascendcl.so
    set(lib_ascendcl "${CANN_INSTALL_DIR}/acllib/lib64")
    find_library(found_lib_ascendcl NAMES ascendcl PATHS ${lib_ascendcl} NO_DEFAULT_PATH)
    if(found_lib_ascendcl)
        set(lib_ascendcl ${found_lib_ascendcl})
        message(STATUS "CANN: libascendcl.so is found at ${lib_ascendcl}")
    else()
        message(STATUS "CANN: Missing libascendcl.so. Turning off HAVE_CANN")
        set(HAVE_CANN OFF)
        return()
    endif()
    #  * libgraph.so
    set(lib_graph "${CANN_INSTALL_DIR}/compiler/lib64")
    find_library(found_lib_graph NAMES graph PATHS ${lib_graph} NO_DEFAULT_PATH)
    if(found_lib_graph)
        set(lib_graph ${found_lib_graph})
        message(STATUS "CANN: libgraph.so is found at ${lib_graph}")
    else()
        message(STATUS "CANN: Missing libgraph.so. Turning off HAVE_CANN")
        set(HAVE_CANN OFF)
        return()
    endif()
    #  * libge_compiler.so
    set(lib_ge_compiler "${CANN_INSTALL_DIR}/compiler/lib64")
    find_library(found_lib_ge_compiler NAMES ge_compiler PATHS ${lib_ge_compiler} NO_DEFAULT_PATH)
    if(found_lib_ge_compiler)
        set(lib_ge_compiler ${found_lib_ge_compiler})
        message(STATUS "CANN: libge_compiler.so is found at ${lib_ge_compiler}")
    else()
        message(STATUS "CANN: Missing libge_compiler.so. Turning off HAVE_CANN")
        set(HAVE_CANN OFF)
        return()
    endif()
    #  * libopsproto.so
    set(lib_opsproto "${CANN_INSTALL_DIR}/opp/op_proto/built-in")
    find_library(found_lib_opsproto NAMES opsproto PATHS ${lib_opsproto} NO_DEFAULT_PATH)
    if(found_lib_opsproto)
        set(lib_opsproto ${found_lib_opsproto})
        message(STATUS "CANN: libopsproto.so is found at ${lib_opsproto}")
    else()
        message(STATUS "CANN: Missing libopsproto.so. Turning off HAVE_CANN")
        set(HAVE_CANN OFF)
        return()
    endif()


    set(libs_cann "")
    list(APPEND libs_cann ${lib_ascendcl})
    list(APPEND libs_cann ${lib_opsproto})
    list(APPEND libs_cann ${lib_graph})
    list(APPEND libs_cann ${lib_ge_compiler})

    try_compile(VALID_ASCENDCL
        "${OpenCV_BINARY_DIR}"
        "${OpenCV_SOURCE_DIR}/cmake/checks/cann.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${incs_cann}"
                    "-DLINK_LIBRARIES:STRING=${libs_cann}"
        OUTPUT_VARIABLE ASCEND_TRY_OUT)

    if(NOT ${VALID_ASCENDCL})
        message(WARNING "Cannot use CANN")
        set(HAVE_CANN OFF)
        return()
    endif()

    set(HAVE_CANN ON)
endif()

if(HAVE_CANN)
    set(CANN_INCLUDE_DIRS ${incs_cann})
    set(CANN_LIBRARIES ${libs_cann})
    ocv_add_external_target(cann "${CANN_INCLUDE_DIRS}" "${CANN_LIBRARIES}" "HAVE_CANN")
    ocv_warnings_disable(CMAKE_C_FLAGS -Wignored-qualifiers)
    ocv_warnings_disable(CMAKE_CXX_FLAGS -Wignored-qualifiers)
endif()

MARK_AS_ADVANCED(
    incs_cann
    libs_cann
    lib_ascendcl
    lib_graph
    lib_ge_compiler
)
