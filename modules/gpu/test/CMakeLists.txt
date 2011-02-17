set(the_target "opencv_test_${name}")	
project(${the_target})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/.."
                    "${CMAKE_CURRENT_BINARY_DIR}")						

set(test_deps opencv_${name} opencv_ts opencv_highgui opencv_calib3d ${DEPS})
foreach(d ${test_deps})
    if(${d} MATCHES "opencv_")
        if(${d} MATCHES "opencv_lapack")
        else()
            string(REPLACE "opencv_" "${CMAKE_CURRENT_SOURCE_DIR}/../../" d_dir ${d})
            include_directories("${d_dir}/include")
        endif()
    endif()
endforeach()

file(GLOB test_srcs "*.cpp")
file(GLOB test_hdrs "*.h*")

add_executable(${the_target} ${test_srcs} ${test_hdrs})

if(PCHSupport_FOUND)
    set(pch_header ${CMAKE_CURRENT_SOURCE_DIR}/test_precomp.hpp)
    if(${CMAKE_GENERATOR} MATCHES "Visual*" OR ${CMAKE_GENERATOR} MATCHES "Xcode*")
        if(${CMAKE_GENERATOR} MATCHES "Visual*")
            set(${the_target}_pch "test_precomp.cpp")
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
