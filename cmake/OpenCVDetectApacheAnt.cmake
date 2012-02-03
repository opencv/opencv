file(TO_CMAKE_PATH "$ENV{ANT_DIR}" ANT_DIR_ENV_PATH)
file(TO_CMAKE_PATH "$ENV{ProgramFiles}" ProgramFiles_ENV_PATH)

find_host_program(ANT_EXECUTABLE NAMES ant.bat ant
  PATHS "${ANT_DIR_ENV_PATH}/bin"
        "${ProgramFiles_ENV_PATH}/apache-ant/bin"
  )

if(ANT_EXECUTABLE)
  execute_process(COMMAND ${ANT_EXECUTABLE} -version
    OUTPUT_VARIABLE ANT_VERSION_FULL
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH "[0-9]+.[0-9]+.[0-9]+" ANT_VERSION "${ANT_VERSION_FULL}")

  message(STATUS "    Found apache ant ${ANT_VERSION}: ${ANT_EXECUTABLE}")
endif()
