cmake_minimum_required(VERSION 2.8)
PROJECT( learningOpenCV2_proj )
FIND_PACKAGE( OpenCV REQUIRED )

MACRO(MY_DEFINE_EXAMPLE name srcs)
    set(the_target "${name}")
    add_executable(${the_target} ${srcs} ${_${the_target}_EXTRA_SRCS})
    target_link_libraries(${the_target} ${OpenCV_LIBS})

    if(MSVC)
        set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
    endif()
ENDMACRO(MY_DEFINE_EXAMPLE)

file(GLOB cpp_examples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ch*.cpp)

set(_ch9_backgroundAvg_EXTRA_SRCS backgroundAvg.cpp cv_yuv_codebook.cpp)

foreach(example_filename ${cpp_examples})
    get_filename_component(example ${example_filename} NAME_WE)
    MY_DEFINE_EXAMPLE(${example}  ${example_filename})
endforeach()
