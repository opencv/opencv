
# This file is used by EnableLanguage in cmGlobalGenerator to
# determine that that selected CShapr compiler can actually compile
# and link the most basic of programs.   If not, a fatal error
# is set and cmake stops processing commands and will not generate
# any makefiles or projects.
if(NOT CMAKE_CSharp_COMPILER_WORKS)
  message(STATUS "Check for working CSharp compiler: ${CMAKE_CSharp_COMPILER}")
  file(WRITE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testCSharpCompiler.cs
    "class Dummy {\n"
    "static void Main() {\n"
    "}\n}\n")
  try_compile(CMAKE_CSharp_COMPILER_WORKS ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testCSharpCompiler.cs
    OUTPUT_VARIABLE OUTPUT)
  set(C_TEST_WAS_RUN 1)
endif()

if(NOT CMAKE_CSharp_COMPILER_WORKS)
  message(STATUS "Check for working CSharp compiler: ${CMAKE_CSharp_COMPILER} -- broken")
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
    "Determining if the CSharp compiler works failed with "
    "the following output:\n${OUTPUT}\n\n")
  message(FATAL_ERROR "The CSharp compiler \"${CMAKE_CSharp_COMPILER}\" "
    "is not able to compile a simple test program.\nIt fails "
    "with the following output:\n ${OUTPUT}\n\n"
    "CMake will not be able to correctly generate this project.")
else()
  if(C_TEST_WAS_RUN)
    message(STATUS "Check for working CSharp compiler: ${CMAKE_CSharp_COMPILER} -- works")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
      "Determining if the CSharp compiler works passed with "
      "the following output:\n${OUTPUT}\n\n")
  endif()
  set(CMAKE_CSharp_COMPILER_WORKS 1 CACHE INTERNAL "")

  if(CMAKE_CSharp_COMPILER_FORCED)
    # The compiler configuration was forced by the user.
    # Assume the user has configured all compiler information.
  else()
    # Try to identify the ABI and configure it into CMakeCSharpCompiler.cmake
    include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerABI.cmake)
    CMAKE_DETERMINE_COMPILER_ABI(C ${CMAKE_ROOT}/Modules/CMakeCSharpCompilerABI.c)
    configure_file(
      #${CMAKE_ROOT}/Modules/CMakeCSharpCompiler.cmake.in
      ${CMAKE_MODULE_PATH}/CMakeCSharpCompiler.cmake.in
      ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeCSharpCompiler.cmake
      @ONLY
      )
  endif()
endif()
