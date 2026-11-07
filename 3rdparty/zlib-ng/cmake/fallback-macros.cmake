# fallback-macros.cmake -- CMake fallback macros
# Copyright (C) 2022 Nathan Moinvaziri
# Licensed under the Zlib license, see LICENSE.md for details

# CMake less than version 3.5.2
if(NOT COMMAND add_compile_options)
    macro(add_compile_options options)
        string(APPEND CMAKE_C_FLAGS ${options})
        string(APPEND CMAKE_CXX_FLAGS ${options})
    endmacro()
endif()

# CMake less than version 3.14
if(NOT COMMAND add_link_options)
    macro(add_link_options options)
        string(APPEND CMAKE_EXE_LINKER_FLAGS ${options})
        string(APPEND CMAKE_SHARED_LINKER_FLAGS ${options})
    endmacro()
endif()
