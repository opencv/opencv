# this is template for a OpenCV module 
macro(define_opencv_module name)
    
    project(opencv_${name})
    add_definitions(-DCVAPI_EXPORTS)

    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src"
                        "${CMAKE_CURRENT_BINARY_DIR}")
    
    foreach(d ${ARGN})
        if(${d} MATCHES "opencv_")
            if(${d} MATCHES "opencv_lapack")
            else()
                string(REPLACE "opencv_" "${CMAKE_CURRENT_SOURCE_DIR}/../" d_dir ${d})
                include_directories("${d_dir}/include")
            endif()
        endif()
    endforeach()

    file(GLOB lib_srcs "src/*.cpp")
    file(GLOB lib_int_hdrs "src/*.h*")
    source_group("Src" FILES ${lib_srcs} ${lib_int_hdrs})

    file(GLOB lib_hdrs "include/opencv2/${name}/*.h*")
    source_group("Include" FILES ${lib_hdrs})

    set(the_target "opencv_${name}")

    add_library(${the_target} ${lib_srcs} ${lib_hdrs} ${lib_int_hdrs})

    # For dynamic link numbering convenions
    set_target_properties(${the_target} PROPERTIES
        VERSION ${OPENCV_VERSION}
        SOVERSION ${OPENCV_SOVERSION}
        OUTPUT_NAME "${the_target}${OPENCV_DLLVERSION}"
        )

    # Additional target properties
    set_target_properties(${the_target} PROPERTIES
        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
        INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
        )

    if(PCHSupport_FOUND)
        set(pch_header ${CMAKE_CURRENT_SOURCE_DIR}/src/precomp.hpp)
        if(${CMAKE_GENERATOR} MATCHES "Visual*" OR ${CMAKE_GENERATOR} MATCHES "Xcode*")
            if(${CMAKE_GENERATOR} MATCHES "Visual*")
                set(${the_target}_pch "src/precomp.cpp")
            endif()            
            add_native_precompiled_header(${the_target} ${pch_header})
        elseif(CMAKE_COMPILER_IS_GNUCXX AND ${CMAKE_GENERATOR} MATCHES ".*Makefiles")
            add_precompiled_header(${the_target} ${pch_header})
        endif()
    endif()

    # Add the required libraries for linking:
    target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${IPP_LIBS} ${ARGN})

    if(MSVC)
        if(CMAKE_CROSSCOMPILING)
            set_target_properties(${the_target} PROPERTIES
                LINK_FLAGS "/NODEFAULTLIB:secchk"
                )
        endif()
        set_target_properties(${the_target} PROPERTIES
            LINK_FLAGS "/NODEFAULTLIB:libc /DEBUG"
            )
    endif()

    # Dependencies of this target:
    add_dependencies(${the_target} ${ARGN})

    install(TARGETS ${the_target}
        RUNTIME DESTINATION bin COMPONENT main
        LIBRARY DESTINATION lib COMPONENT main
        ARCHIVE DESTINATION lib COMPONENT main)

    install(FILES ${lib_hdrs}
        DESTINATION include/opencv2/${name}
        COMPONENT main)
        
    if(BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test)
        include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include"
                            "${CMAKE_CURRENT_SOURCE_DIR}/test"
                            "${CMAKE_CURRENT_BINARY_DIR}")

        set(test_deps opencv_${name} ${ARGN} opencv_ts opencv_highgui ${EXTRA_${the_target}_DEPS})
        foreach(d ${test_deps})
            if(${d} MATCHES "opencv_")
                if(${d} MATCHES "opencv_lapack")
                else()
                    string(REPLACE "opencv_" "${CMAKE_CURRENT_SOURCE_DIR}/../" d_dir ${d})
                    include_directories("${d_dir}/include")
                endif()
            endif()
        endforeach()

        file(GLOB test_srcs "test/*.cpp")
        file(GLOB test_hdrs "test/*.h*")
		
		source_group("Src" FILES ${test_srcs})
		source_group("Include" FILES ${test_hdrs})

        set(the_target "opencv_test_${name}")

        add_executable(${the_target} ${test_srcs} ${test_hdrs})

        if(PCHSupport_FOUND)
            set(pch_header ${CMAKE_CURRENT_SOURCE_DIR}/test/test_precomp.hpp)
            if(${CMAKE_GENERATOR} MATCHES "Visual*" OR ${CMAKE_GENERATOR} MATCHES "Xcode*")
                if(${CMAKE_GENERATOR} MATCHES "Visual*")
                    set(${the_target}_pch "test/test_precomp.cpp")
                endif()            
                add_native_precompiled_header(${the_target} ${pch_header})
            elseif(CMAKE_COMPILER_IS_GNUCXX AND ${CMAKE_GENERATOR} MATCHES ".*Makefiles")
                add_precompiled_header(${the_target} ${pch_header})
            endif()
        endif()

        # Additional target properties
        set_target_properties(${the_target} PROPERTIES
            DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
            )

        add_dependencies(${the_target} ${test_deps})

        # Add the required libraries for linking:
        target_link_libraries(${the_target} ${OPENCV_LINKER_LIBS} ${test_deps})

        enable_testing()
        get_target_property(LOC ${the_target} LOCATION)
        add_test(${the_target} "${LOC}")

        if(WIN32)
            install(TARGETS ${the_target} RUNTIME DESTINATION bin COMPONENT main)
        endif()
    endif()    
        
endmacro()
