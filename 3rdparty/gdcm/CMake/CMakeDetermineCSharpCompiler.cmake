
# determine the compiler to use for CSharp programs
# NOTE, a generator may set CMAKE_CSharp_COMPILER before
# loading this file to force a compiler.
# use environment variable CSHARP first if defined by user, next use
# the cmake variable CMAKE_GENERATOR_CSHARP which can be defined by a generator
# as a default compiler
#
# Sets the following variables:
#   CMAKE_CSharp_COMPILER
#   CMAKE_AR
#   CMAKE_RANLIB
#   CMAKE_COMPILER_IS_GNUGNAT

if(NOT CMAKE_CSharp_COMPILER)
  # prefer the environment variable CSHARP
  if($ENV{CSHARP} MATCHES ".+")
    get_filename_component(CMAKE_CSharp_COMPILER_INIT $ENV{CSHARP} PROGRAM PROGRAM_ARGS CMAKE_CSharp_FLAGS_ENV_INIT)
    if(CMAKE_CSharp_FLAGS_ENV_INIT)
      set(CMAKE_CSharp_COMPILER_ARG1 "${CMAKE_CSharp_FLAGS_ENV_INIT}" CACHE STRING "First argument to CSharp compiler")
    endif()
    if(NOT EXISTS ${CMAKE_CSharp_COMPILER_INIT})
      message(FATAL_ERROR "Could not find compiler set in environment variable CSHARP:\n$ENV{CSHARP}.")
    endif()
  endif()

  # next prefer the generator-specified compiler
  if(CMAKE_GENERATOR_CSHARP)
    if(NOT CMAKE_CSharp_COMPILER_INIT)
      set(CMAKE_CSharp_COMPILER_INIT ${CMAKE_GENERATOR_CSHARP})
    endif()
  endif()

  # finally list compilers to try
  if(CMAKE_CSharp_COMPILER_INIT)
    set(CMAKE_CSharp_COMPILER_LIST ${CMAKE_CSharp_COMPILER_INIT})
  else()
    # Known compilers:
    # mcs/gmcs/smcs # mono
    # csc: DotNet
    set(CMAKE_CSharp_COMPILER_LIST csc mcs gmcs smcs)
  endif()

  # Find the compiler.
  find_program(CMAKE_CSharp_COMPILER NAMES ${CMAKE_CSharp_COMPILER_LIST} DOC "CSharp compiler")
  if(CMAKE_CSharp_COMPILER_INIT AND NOT CMAKE_CSharp_COMPILER)
    set(CMAKE_CSharp_COMPILER "${CMAKE_CSharp_COMPILER_INIT}" CACHE FILEPATH "CSharp compiler" FORCE)
  endif()
endif()
mark_as_advanced(CMAKE_CSharp_COMPILER)

get_filename_component(COMPILER_LOCATION "${CMAKE_CSharp_COMPILER}"
  PATH)


#include(CMakeFindBinUtils)

# configure variables set in this file for fast reload later on
configure_file(
  #${CMAKE_ROOT}/Modules/CMakeCSharpCompiler.cmake.in
  ${CMAKE_MODULE_PATH}/CMakeCSharpCompiler.cmake.in
  #  "${CMAKE_PLATFORM_ROOT_BIN}/CMakeCSharpCompiler.cmake"
  ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeCSharpCompiler.cmake
  @ONLY
  )

set(CMAKE_CSharp_COMPILER_ENV_VAR "CSC")
