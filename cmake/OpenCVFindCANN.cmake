ocv_check_environment_variables(CANN_INSTALL_DIR)

if("cann${CANN_INSTALL_DIR}" STREQUAL "cann" AND DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(CANN_INSTALL_DIR $ENV{ASCEND_TOOLKIT_HOME})
    message(STATUS "CANN: updated CANN_INSTALL_DIR from ASCEND_TOOLKIT_HOME=$ENV{ASCEND_TOOLKIT_HOME}")
endif()

if(EXISTS "${CANN_INSTALL_DIR}/opp/op_proto/built-in/inc")
    set(CANN_VERSION_BELOW_6_3_ALPHA002 "YES" )
    add_definitions(-DCANN_VERSION_BELOW_6_3_ALPHA002="YES")
endif()

if(CANN_INSTALL_DIR)
    # Supported system: UNIX
    if(NOT UNIX)
        set(HAVE_CANN OFF)
        message(WARNING "CANN: CANN toolkit supports unix but not ${CMAKE_SYSTEM_NAME}. Turning off HAVE_CANN")
        return()
    endif()
    # Supported platforms: x86-64, arm64
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
    else()
        set(HAVE_CANN OFF)
        message(WARNING "CANN: CANN toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}. Turning off HAVE_CANN")
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
        message(WARNING "CANN: Missing libascendcl.so. Turning off HAVE_CANN")
        set(HAVE_CANN OFF)
        return()
    endif()
    #  * libacl_op_compiler.so
    set(lib_acl_op_compiler "${CANN_INSTALL_DIR}/lib64")
    find_library(found_lib_acl_op_compiler NAMES acl_op_compiler PATHS ${lib_acl_op_compiler} NO_DEFAULT_PATH)
    if(found_lib_acl_op_compiler)
        set(lib_acl_op_compiler ${found_lib_acl_op_compiler})
        message(STATUS "CANN: libacl_op_compiler.so is found at ${lib_acl_op_compiler}")
    else()
        message(STATUS "CANN: Missing libacl_op_compiler.so. Turning off HAVE_CANN")
        set(HAVE_CANN OFF)
        return()
    endif()

    #  * libacl_dvpp_mpi.so
    set(libacl_dvpp_mpi "${CANN_INSTALL_DIR}/lib64")
    find_library(found_libacldvppmpi NAMES acl_dvpp_mpi PATHS ${libacl_dvpp_mpi} NO_DEFAULT_PATH)
    if(found_libacldvppmpi)
        set(libacl_dvpp_mpi ${found_libacldvppmpi})
        message(STATUS "CANN: libacl_dvpp_mpi.so is found at ${libacl_dvpp_mpi}")
    else()
        message(STATUS "CANN: Missing libacl_dvpp_mpi.so. Turning off HAVE_CANN")
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
        message(WARNING "CANN: Missing libgraph.so. Turning off HAVE_CANN")
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
        message(WARNING "CANN: Missing libge_compiler.so. Turning off HAVE_CANN")
        set(HAVE_CANN OFF)
        return()
    endif()
    #  * libopsproto.so
    if (CANN_VERSION_BELOW_6_3_ALPHA002)
        set(lib_opsproto "${CANN_INSTALL_DIR}/opp/op_proto/built-in/")
    else()
        if(EXISTS "${CANN_INSTALL_DIR}/opp/built-in/op_proto/lib/linux")
            set(lib_opsproto "${CANN_INSTALL_DIR}/opp/built-in/op_proto/lib/linux/${CMAKE_HOST_SYSTEM_PROCESSOR}")
        else()
            set(lib_opsproto "${CANN_INSTALL_DIR}/opp/built-in/op_proto")
        endif()
    endif()
    find_library(found_lib_opsproto NAMES opsproto PATHS ${lib_opsproto} NO_DEFAULT_PATH)
    if(found_lib_opsproto)
        set(lib_opsproto ${found_lib_opsproto})
        message(STATUS "CANN: libopsproto.so is found at ${lib_opsproto}")
    else()
        message(WARNING "CANN: Missing libopsproto.so can't found at ${lib_opsproto}. Turning off HAVE_CANN")
        set(HAVE_CANN OFF)
        return()
    endif()

    set(libs_cann "")
    list(APPEND libs_cann ${lib_ascendcl})
    list(APPEND libs_cann ${lib_acl_op_compiler})
    list(APPEND libs_cann ${lib_opsproto})
    list(APPEND libs_cann ${lib_graph})
    list(APPEND libs_cann ${lib_ge_compiler})
    list(APPEND libs_cann ${libacl_dvpp_mpi})

    #  * lib_graph_base.so
    if(NOT CANN_VERSION_BELOW_6_3_ALPHA002)
        set(lib_graph_base "${CANN_INSTALL_DIR}/compiler/lib64")
        find_library(found_libgraph_base NAMES graph_base PATHS ${lib_graph_base} NO_DEFAULT_PATH)
        if(found_libgraph_base)
            set(lib_graph_base ${found_libgraph_base})
            message(STATUS "CANN: lib_graph_base.so is found at ${lib_graph_base}")
            list(APPEND libs_cann ${lib_graph_base})
        else()
            message(STATUS "CANN: Missing lib_graph_base.so. It is only required after cann version 6.3.RC1.alpha002")
        endif()
    endif()

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
