########################
# Initial cache settings for opencv on android
# run cmake with:
# cmake -C 
########################

#Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" )

#no python available on Android
set(BUILD_NEW_PYTHON_SUPPORT OFF CACHE INTERNAL "" FORCE)

#Enable SSE instructions
SET( ENABLE_SSE OFF CACHE INTERNAL "" FORCE )

#Enable SSE2 instructions
SET( ENABLE_SSE2 OFF CACHE INTERNAL "" FORCE )

#Enable SSE3 instructions
SET( ENABLE_SSE3 OFF CACHE INTERNAL "" FORCE )

#Enable SSE4.1 instructions
SET( ENABLE_SSE41 OFF CACHE INTERNAL "" FORCE )

#Enable SSE4.2 instructions
SET( ENABLE_SSE42 OFF CACHE INTERNAL "" FORCE )

#Enable SSSE3 instructions
SET( ENABLE_SSSE3 OFF CACHE INTERNAL "" FORCE )

#Set output folder to ${CMAKE_BINARY_DIR}
set( LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to" )
