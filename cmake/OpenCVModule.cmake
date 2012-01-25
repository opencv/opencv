set(opencv_public_modules "" CACHE INTERNAL "List of OpenCV modules included into the build")
# helper macro for modules management
macro(opencv_module_register name)
   set(opencv_public_modules ${opencv_public_modules} ${name} CACHE INTERNAL "List of OpenCV modules included into the build")
endmacro()

# Setup include path for OpenCV headers for specified modules
macro(opencv_module_includes)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src"
                        "${CMAKE_CURRENT_BINARY_DIR}")
    foreach(d ${ARGN})
        if(d MATCHES "opencv_")
            string(REPLACE "opencv_" "${OpenCV_SOURCE_DIR}/modules/" d_dir ${d})
            if (EXISTS "${d_dir}/include")
                include_directories("${d_dir}/include")
            endif()
        endif()
    endforeach()
endmacro()

# opencv precompiled headers macro (can add pch to modules and tests)
# this macro must be called after any "add_definitions" commands, otherwise precompiled headers will not work
macro(add_opencv_precompiled_headers the_target)
    if("${the_target}" MATCHES "opencv_test_.*")
        SET(pch_name "test/test_precomp")
    elseif("${the_target}" MATCHES "opencv_perf_.*")
        SET(pch_name "perf/perf_precomp")
    else()
        SET(pch_name "src/precomp")
    endif()
    set(pch_header "${CMAKE_CURRENT_SOURCE_DIR}/${pch_name}.hpp")
    if(PCHSupport_FOUND AND ENABLE_PRECOMPILED_HEADERS AND EXISTS "${pch_header}")
        if(CMAKE_GENERATOR MATCHES Visual)
            set(${the_target}_pch "${CMAKE_CURRENT_SOURCE_DIR}/${pch_name}.cpp")
            add_native_precompiled_header(${the_target} ${pch_header})
        elseif(CMAKE_GENERATOR MATCHES Xcode)
            add_native_precompiled_header(${the_target} ${pch_header})
        elseif(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_GENERATOR MATCHES Makefiles)
            add_precompiled_header(${the_target} ${pch_header})
        endif()
    endif()
endmacro()

# this is a template for a OpenCV performance tests
# define_opencv_perf_test(<module_name> <extra_dependencies>)
macro(define_opencv_perf_test name)
    set(perf_path "${CMAKE_CURRENT_SOURCE_DIR}/perf")
    if(BUILD_PERF_TESTS AND EXISTS "${perf_path}")

        # opencv_highgui is required for imread/imwrite
        set(perf_deps opencv_${name} ${ARGN} opencv_ts opencv_highgui ${EXTRA_OPENCV_${name}_DEPS})

        include_directories("${perf_path}")
        opencv_module_includes(${perf_deps})

        file(GLOB perf_srcs "${perf_path}/*.cpp")
        file(GLOB perf_hdrs "${perf_path}/*.h*")

        source_group("Src" FILES ${perf_srcs})
        source_group("Include" FILES ${perf_hdrs})

        set(the_target "opencv_perf_${name}")
        add_executable(${the_target} ${perf_srcs} ${perf_hdrs})

        # Additional target properties
        set_target_properties(${the_target} PROPERTIES
            DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
            RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
            )

        if(ENABLE_SOLUTION_FOLDERS)
            set_target_properties(${the_target} PROPERTIES FOLDER "performance tests")
        endif()

        target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${perf_deps})

        add_opencv_precompiled_headers(${the_target})

        if (PYTHON_EXECUTABLE)
            add_dependencies(perf ${the_target})
        endif()
    endif()
endmacro()

# this is a template for a OpenCV regression tests
# define_opencv_test(<module_name> <extra_dependencies>)
macro(define_opencv_test name)
    set(test_path "${CMAKE_CURRENT_SOURCE_DIR}/test")
    if(BUILD_TESTS AND EXISTS "${test_path}")

        # opencv_highgui is required for imread/imwrite
        set(test_deps opencv_${name} ${ARGN} opencv_ts opencv_highgui ${EXTRA_OPENCV_${name}_DEPS})

        include_directories("${test_path}")
        opencv_module_includes(${test_deps})

        file(GLOB test_srcs "${test_path}/*.cpp")
        file(GLOB test_hdrs "${test_path}/*.h*")

        source_group("Src" FILES ${test_srcs})
        source_group("Include" FILES ${test_hdrs})

        set(the_target "opencv_test_${name}")
        add_executable(${the_target} ${test_srcs} ${test_hdrs})

        # Additional target properties
        set_target_properties(${the_target} PROPERTIES
            DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
            RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
            )

        if(ENABLE_SOLUTION_FOLDERS)
            set_target_properties(${the_target} PROPERTIES FOLDER "tests")
        endif()

        target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${test_deps})

        enable_testing()
        get_target_property(LOC ${the_target} LOCATION)
        add_test(${the_target} "${LOC}")

        #if(WIN32)
        #    install(TARGETS ${the_target} RUNTIME DESTINATION bin COMPONENT main)
        #endif()
        add_opencv_precompiled_headers(${the_target})
    endif()
endmacro()

# Set standard properties, install rules and precompiled headers for OpenCV module
macro(opencv_module_setup name)
    set(the_target "opencv_${name}")
    
    # For dynamic link numbering convenions
    if(NOT ANDROID)
        # Android SDK build scripts can include only .so files into final .apk
        # As result we should not set version properties for Android
        set_target_properties(${the_target} PROPERTIES
            VERSION ${OPENCV_VERSION}
            SOVERSION ${OPENCV_SOVERSION}
            )
    endif()

    set_target_properties(${the_target} PROPERTIES OUTPUT_NAME "${the_target}${OPENCV_DLLVERSION}" )

    if(ENABLE_SOLUTION_FOLDERS)
        set_target_properties(${the_target} PROPERTIES FOLDER "modules")
    endif()

    if (BUILD_SHARED_LIBS)
        if(MSVC)
            set_target_properties(${the_target} PROPERTIES DEFINE_SYMBOL CVAPI_EXPORTS)
        else()
            add_definitions(-DCVAPI_EXPORTS)
        endif()
    endif()

    # Additional target properties
    set_target_properties(${the_target} PROPERTIES
        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
        ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
        RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
        INSTALL_NAME_DIR lib
        )

    if(MSVC)
        if(CMAKE_CROSSCOMPILING)
            set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:secchk")
        endif()
        set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:libc /DEBUG")
    endif()

    install(TARGETS ${the_target}
        RUNTIME DESTINATION bin COMPONENT main
        LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT main
        ARCHIVE DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT main)

    if(lib_hdrs)
        install(FILES ${lib_hdrs}
            DESTINATION ${OPENCV_INCLUDE_PREFIX}/opencv2/${name}
            COMPONENT main)
    endif()

    add_opencv_precompiled_headers(${the_target})
endmacro()

# this is a template for a OpenCV module declaration
# define_opencv_moduleEx(<module_name> [public|internal] <dependencies>)
macro(define_opencv_moduleEx _name _visibility)
    string(TOLOWER "${_name}" name)
    string(TOUPPER "${_name}" mname)
    string(TOLOWER "${_visibility}" visibility)

    option(OCVMODULE_${mname} "Include ${name} module into the OpenCV build" ON)
    if(OCVMODULE_${mname})
        set(the_target "opencv_${name}")
        project(${the_target})

        opencv_module_includes(${ARGN})

        file(GLOB lib_srcs "src/*.cpp")
        file(GLOB lib_int_hdrs "src/*.h*")
        file(GLOB lib_hdrs "include/opencv2/${name}/*.h*")
        file(GLOB lib_hdrs_detail "include/opencv2/${name}/detail/*.h*")

        if(COMMAND get_module_external_sources)
            get_module_external_sources(${name})
        endif()

        source_group("Src" FILES ${lib_srcs} ${lib_int_hdrs})
        source_group("Include" FILES ${lib_hdrs})
        source_group("Include\\detail" FILES ${lib_hdrs_detail})

        list(APPEND lib_hdrs ${lib_hdrs_detail})

        add_library(${the_target} ${OPENCV_${mname}_MODULE_TYPE} ${lib_srcs} ${lib_hdrs} ${lib_int_hdrs})
        target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${IPP_LIBS} ${ARGN})

        if(visibility STREQUAL "public")
            opencv_module_register(${the_target})
        endif()

        opencv_module_setup(${name})
        define_opencv_test(${name})
        define_opencv_perf_test(${name})
    endif()
endmacro()

# this is a shorthand for a public OpenCV module declaration
# define_opencv_module(<module_name> <dependencies>)
macro(define_opencv_module name)
    define_opencv_moduleEx(${name} PUBLIC ${ARGN})
endmacro()

