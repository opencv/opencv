if(CMAKE_HOST_WIN32)
    set(GHC_NAME ghc.exe)
    set(CABAL_NAME cabal.exe)
else()
    set(GHC_NAME ghc)
    set(CABAL_NAME cabal)
endif()

find_host_program(GHC_EXECUTABLE NAMES ${GHC_NAME}
    PATHS "/usr/bin" #TODO: Figure out where ghc/cabal are installed on other systems.
    NO_DEFAULT_PATH
)

find_host_program(GHC_EXECUTABLE NAMES ${GHC_NAME})

if(GHC_EXECUTABLE)
    execute_process(COMMAND ${GHC_EXECUTABLE} --version
        RESULT_VARIABLE GHC_ERROR_LEVEL
        OUTPUT_VARIABLE GHC_VERSION_FULL
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(GHC_ERROR_LEVEL)
        unset(GHC_EXECUTABLE)
        unset(GHC_EXECUTABLE CACHE)
    else()
        string(REGEX MATCH "[0-9]+.[0-9]+.[0-9]+" GHC_VERSION "${GHC_VERSION_FULL}")
        set(GHC_VERSION "${GHC_VERSION}" CACHE INTERNAL "Detected ghc version")
    endif()
endif()

find_host_program(CABAL_EXECUTABLE NAMES ${CABAL_NAME}
    PATHS "/usr/bin" 
    NO_DEFAULT_PATH
)

find_host_program(CABAL_EXECUTABLE NAMES ${CABAL_NAME})

if(CABAL_EXECUTABLE)
    execute_process(COMMAND ${CABAL_EXECUTABLE} --version
        RESULT_VARIABLE CABAL_ERROR_LEVEL
        OUTPUT_VARIABLE CABAL_VERSION_FULL
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(CABAL_ERROR_LEVEL)
        unset(CABAL_EXECUTABLE)
        unset(CABAL_EXECUTABLE CACHE)
    else()
        string(REGEX MATCH "[0-9]+.[0-9]+.[0-9]+" CABAL_VERSION "${CABAL_VERSION_FULL}")
        set(CABAL_VERSION "${CABAL_VERSION}" CACHE INTERNAL "Detected cabal version")
    endif()
endif()
