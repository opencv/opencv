#
# The script to detect Intel(R) Math Kernel Library (MKL)
# installation/package
#
# Parameters:
# MKL_WITH_TBB
#
# On return this will define:
#
# HAVE_MKL          - True if Intel IPP found
# MKL_ROOT_DIR      - root of IPP installation
# MKL_INCLUDE_DIRS  - IPP include folder
# MKL_LIBRARIES     - IPP libraries that are used by OpenCV
#

macro (mkl_find_lib VAR NAME DIRS)
    find_path(${VAR} ${NAME} ${DIRS} NO_DEFAULT_PATH)
    set(${VAR} ${${VAR}}/${NAME})
    unset(${VAR} CACHE)
endmacro()

macro(mkl_fail)
    set(HAVE_MKL OFF CACHE BOOL "True if MKL found")
    set(MKL_ROOT_DIR ${MKL_ROOT_DIR} CACHE PATH "Path to MKL directory")
    unset(MKL_INCLUDE_DIRS CACHE)
    unset(MKL_LIBRARIES CACHE)
    return()
endmacro()

macro(get_mkl_version VERSION_FILE)
    # read MKL version info from file
    file(STRINGS ${VERSION_FILE} STR1 REGEX "__INTEL_MKL__")
    file(STRINGS ${VERSION_FILE} STR2 REGEX "__INTEL_MKL_MINOR__")
    file(STRINGS ${VERSION_FILE} STR3 REGEX "__INTEL_MKL_UPDATE__")
    #file(STRINGS ${VERSION_FILE} STR4 REGEX "INTEL_MKL_VERSION")

    # extract info and assign to variables
    string(REGEX MATCHALL "[0-9]+" MKL_VERSION_MAJOR ${STR1})
    string(REGEX MATCHALL "[0-9]+" MKL_VERSION_MINOR ${STR2})
    string(REGEX MATCHALL "[0-9]+" MKL_VERSION_UPDATE ${STR3})
    set(MKL_VERSION_STR "${MKL_VERSION_MAJOR}.${MKL_VERSION_MINOR}.${MKL_VERSION_UPDATE}" CACHE STRING "MKL version" FORCE)
endmacro()


if(NOT DEFINED MKL_USE_MULTITHREAD)
    OCV_OPTION(MKL_WITH_TBB "Use MKL with TBB multithreading" OFF)#ON IF WITH_TBB)
    OCV_OPTION(MKL_WITH_OPENMP "Use MKL with OpenMP multithreading" OFF)#ON IF WITH_OPENMP)
endif()

#check current MKL_ROOT_DIR
if(NOT MKL_ROOT_DIR OR NOT EXISTS ${MKL_ROOT_DIR}/include/mkl.h)
    set(mkl_root_paths ${MKL_ROOT_DIR})
    if(DEFINED $ENV{MKLROOT})
        list(APPEND mkl_root_paths $ENV{MKLROOT})
    endif()
    if(WIN32)
        set(ProgramFilesx86 "ProgramFiles(x86)")
        list(APPEND mkl_root_paths $ENV{${ProgramFilesx86}}/IntelSWTools/compilers_and_libraries/windows/mkl)
    endif()
    if(UNIX)
        list(APPEND mkl_root_paths "/opt/intel/mkl")
    endif()

    find_path(MKL_ROOT_DIR include/mkl.h PATHS ${mkl_root_paths})
endif()

if(NOT MKL_ROOT_DIR)
    mkl_fail()
endif()

set(MKL_INCLUDE_DIRS ${MKL_ROOT_DIR}/include)
get_mkl_version(${MKL_INCLUDE_DIRS}/mkl_version.h)

#determine arch
if(CMAKE_CXX_SIZEOF_DATA_PTR EQUAL 8)
    set(MKL_X64 1)
    set(MKL_ARCH "intel64")

    include(CheckTypeSize)
    CHECK_TYPE_SIZE(int _sizeof_int)
    if (_sizeof_int EQUAL 4)
        set(MKL_LP64 "lp64")
    else()
        set(MKL_LP64 "ilp64")
    endif()
else()
    set(MKL_ARCH "ia32")
endif()

if(${MKL_VERSION_STR} VERSION_GREATER "11.3.0" OR ${MKL_VERSION_STR} VERSION_EQUAL "11.3.0")
    set(mkl_lib_find_paths
        ${MKL_ROOT_DIR}/lib
        ${MKL_ROOT_DIR}/lib/${MKL_ARCH} ${MKL_ROOT_DIR}/../tbb/lib/${MKL_ARCH})

    set(mkl_lib_list
        mkl_core
        mkl_intel_${MKL_LP64})

    if(MKL_WITH_TBB)
        list(APPEND mkl_lib_list mkl_tbb_thread tbb)
    elseif(MKL_WITH_OPENMP)
        if(MSVC)
            list(APPEND mkl_lib_list mkl_intel_thread libiomp5md)
        else()
            list(APPEND mkl_lib_list libmkl_gnu_thread)
        endif()
    else()
        list(APPEND mkl_lib_list mkl_sequential)
    endif()
else()
    message(STATUS "MKL version ${MKL_VERSION_STR} is not supported")
    mkl_fail()
endif()


set(MKL_LIBRARIES "")
foreach(lib ${mkl_lib_list})
    find_library(${lib} ${lib} ${mkl_lib_find_paths})
    mark_as_advanced(${lib})
    if(NOT ${lib})
        mkl_fail()
    endif()
    list(APPEND MKL_LIBRARIES ${${lib}})
endforeach()

message(STATUS "Found MKL ${MKL_VERSION_STR} at: ${MKL_ROOT_DIR}")
set(HAVE_MKL ON CACHE BOOL "True if MKL found")
set(MKL_ROOT_DIR ${MKL_ROOT_DIR} CACHE PATH "Path to MKL directory")
set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIRS} CACHE PATH "Path to MKL include directory")
if(NOT UNIX)
    set(MKL_LIBRARIES ${MKL_LIBRARIES} CACHE FILEPATH "MKL libarries")
else()
    #it's ugly but helps to avoid cyclic lib problem
    set(MKL_LIBRARIES ${MKL_LIBRARIES} ${MKL_LIBRARIES} ${MKL_LIBRARIES} "-lpthread" "-lm" "-ldl")
    set(MKL_LIBRARIES ${MKL_LIBRARIES} CACHE STRING "MKL libarries")
endif()