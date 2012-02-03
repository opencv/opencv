if (WIN32 AND CMAKE_GENERATOR MATCHES "(MinGW)|(MSYS)")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "")
endif()

set(OPENCV_EXTRA_C_FLAGS "")
set(OPENCV_EXTRA_C_FLAGS_RELEASE "")
set(OPENCV_EXTRA_C_FLAGS_DEBUG "")
set(OPENCV_EXTRA_EXE_LINKER_FLAGS "")
set(OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE "")
set(OPENCV_EXTRA_EXE_LINKER_FLAGS_DEBUG "")

if(MSVC)
    set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} /D _CRT_SECURE_NO_DEPRECATE /D _CRT_NONSTDC_NO_DEPRECATE /D _SCL_SECURE_NO_WARNINGS")
    # 64-bit portability warnings, in MSVC8
    if(MSVC80)
        set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} /Wp64")
    endif()
    #if(MSVC90)
    #    set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} /D _BIND_TO_CURRENT_CRT_VERSION=1 /D _BIND_TO_CURRENT_VCLIBS_VERSION=1")
    #endif()

    if(BUILD_WITH_DEBUG_INFO)
        set(OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE "${OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE} /debug")
    endif()

    # Remove unreferenced functions: function level linking
    set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} /Gy")
    if(BUILD_WITH_DEBUG_INFO)
       set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} /Zi")
    endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    # High level of warnings.
    set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} -Wall")

    # The -Wno-long-long is required in 64bit systems when including sytem headers.
    if(X86_64)
        set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} -Wno-long-long")
    endif()

    # We need pthread's
    if(UNIX AND NOT ANDROID)
        set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} -pthread")
    endif()

    if(OPENCV_WARNINGS_ARE_ERRORS)
        set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} -Werror")
    endif()

    if(X86 AND NOT MINGW64 AND NOT X86_64 AND NOT APPLE)
        set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} -march=i686")
    endif()

    # Other optimizations
    if(ENABLE_OMIT_FRAME_POINTER)
        set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} -fomit-frame-pointer")
    else()
        set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} -fno-omit-frame-pointer")
    endif()
    if(ENABLE_FAST_MATH)
        set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} -ffast-math")
    endif()
    if(ENABLE_POWERPC)
        set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} -mcpu=G3 -mtune=G5")
    endif()
    if(ENABLE_SSE)
        set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} -msse")
    endif()
    if(ENABLE_SSE2)
        set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} -msse2")
    endif()

    # SSE3 and further should be disabled under MingW because it generates compiler errors
    if(NOT MINGW)
        if(ENABLE_SSE3)
            set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} -msse3")
        endif()

        if(${CMAKE_OPENCV_GCC_VERSION_NUM} GREATER 402)
            set(HAVE_GCC43_OR_NEWER 1)
        endif()
        if(${CMAKE_OPENCV_GCC_VERSION_NUM} GREATER 401)
            set(HAVE_GCC42_OR_NEWER 1)
        endif()

        if(HAVE_GCC42_OR_NEWER OR APPLE)
            if(ENABLE_SSSE3)
                set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} -mssse3")
            endif()
            if(HAVE_GCC43_OR_NEWER)
                if(ENABLE_SSE41)
                    set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} -msse4.1")
                endif()
                if(ENABLE_SSE42)
                    set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} -msse4.2")
                endif()
            endif()
        endif()
    endif(NOT MINGW)

    if(X86 OR X86_64)
        if(NOT APPLE AND CMAKE_SIZEOF_VOID_P EQUAL 4)
            set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} -mfpmath=387")
        endif()
    endif()

    # Profiling?
    if(ENABLE_PROFILING)
        set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} -pg -g")
    elseif(NOT APPLE)
        # Remove unreferenced functions: function level linking
        set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS} -ffunction-sections")
    endif()

    set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} -DNDEBUG")
    set(OPENCV_EXTRA_C_FLAGS_DEBUG "${OPENCV_EXTRA_C_FLAGS_DEBUG} -O0 -DDEBUG -D_DEBUG")
    if(BUILD_WITH_DEBUG_INFO)
       set(OPENCV_EXTRA_C_FLAGS_DEBUG "${OPENCV_EXTRA_C_FLAGS_DEBUG} -ggdb3")
    endif()
endif()

if(MSVC)
    # 64-bit MSVC compiler uses SSE/SSE2 by default
    if(NOT MSVC64)
       if(ENABLE_SSE)
          set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} /arch:SSE")
       endif()
       if(ENABLE_SSE2)
          set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} /arch:SSE2")
       endif()
    endif()
    if(ENABLE_SSE3)
       set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} /arch:SSE3")
    endif()
    if(ENABLE_SSE4_1)
       set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} /arch:SSE4.1")
    endif()
    if (ENABLE_SSE OR ENABLE_SSE2 OR ENABLE_SSE3 OR ENABLE_SSE4_1)
       set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE} /Oi")
    endif()
endif()

# Extra link libs if the user selects building static libs:
if(NOT BUILD_SHARED_LIBS AND CMAKE_COMPILER_IS_GNUCXX AND NOT ANDROID)
    # Android does not need these settings because they are already set by toolchain file
    set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} stdc++)
    set(OPENCV_EXTRA_C_FLAGS "-fPIC ${OPENCV_EXTRA_C_FLAGS}")
endif()

# Add user supplied extra options (optimization, etc...)
# ==========================================================
set(OPENCV_EXTRA_C_FLAGS "${OPENCV_EXTRA_C_FLAGS}" CACHE INTERNAL "Extra compiler options")
set(OPENCV_EXTRA_C_FLAGS_RELEASE "${OPENCV_EXTRA_C_FLAGS_RELEASE}" CACHE INTERNAL "Extra compiler options for Release build")
set(OPENCV_EXTRA_C_FLAGS_DEBUG "${OPENCV_EXTRA_C_FLAGS_DEBUG}" CACHE INTERNAL "Extra compiler options for Debug build")
set(OPENCV_EXTRA_EXE_LINKER_FLAGS "${OPENCV_EXTRA_EXE_LINKER_FLAGS}" CACHE INTERNAL "Extra linker flags")
set(OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE "${OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE}" CACHE INTERNAL "Extra linker flags for Release build")
set(OPENCV_EXTRA_EXE_LINKER_FLAGS_DEBUG "${OPENCV_EXTRA_EXE_LINKER_FLAGS_DEBUG}" CACHE INTERNAL "Extra linker flags for Debug build")

#combine all "extra" options
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPENCV_EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENCV_EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  ${OPENCV_EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OPENCV_EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OPENCV_EXTRA_C_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${OPENCV_EXTRA_C_FLAGS_DEBUG}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OPENCV_EXTRA_EXE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${OPENCV_EXTRA_EXE_LINKER_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${OPENCV_EXTRA_EXE_LINKER_FLAGS_DEBUG}")

if (WIN32 AND MSVC)
    # avoid warnings from MSVC about overriding the /W* option
    # we replace /W3 with /W4 only for C++ files,
    # since all the 3rd-party libraries OpenCV uses are in C,
    # and we do not care about their warnings.
    string(REPLACE "/W3" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/W3" "/W4" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/W3" "/W4" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

    # allow extern "C" functions throw exceptions
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/EHsc" "/EHsc-" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

    string(REPLACE "/Zm1000" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/Zm1000" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()
