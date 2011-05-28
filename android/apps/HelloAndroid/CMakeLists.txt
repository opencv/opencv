CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

#########################################################
# Set project name
#########################################################

IF( NOT PROJECT_NAME )
  IF ( NOT "x$ENV{PROJECT_NAME}" STREQUAL "x" )
    SET( PROJECT_NAME $ENV{PROJECT_NAME} )
  ELSE()
    SET( PROJECT_NAME HelloAndroid )
  ENDIF()
ENDIF()
SET( PROJECT_NAME ${PROJECT_NAME} CACHE STRING "The name of your project")

PROJECT( ${PROJECT_NAME} )

#########################################################
# Find OpenCV
#########################################################

SET( OpenCV_DIR ${CMAKE_SOURCE_DIR}/../../build CACHE PATH "The path where you built opencv for android" )
FIND_PACKAGE( OpenCV REQUIRED )

#########################################################
# c/c++ flags, includes and lib dependencies
#########################################################

#notice the "recycling" of CMAKE_C_FLAGS
#this is necessary to pick up android flags
SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pedantic" )
SET( CMAKE_CPP_FLAGS "${CMAKE_CPP_FLAGS} -Wall -pedantic" )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET( LIBRARY_DEPS ${OpenCV_LIBS} )
IF( ANDROID )
  SET( LIBRARY_DEPS ${LIBRARY_DEPS} log dl )
ENDIF()

#########################################################
# source files
#########################################################

FILE( GLOB hdrs "*.h*" )
FILE( GLOB srcs "*.cpp" )

ADD_EXECUTABLE( ${PROJECT_NAME} ${srcs} )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${LIBRARY_DEPS} )

#########################################################
# Summary report
#########################################################
message( STATUS "")
message( STATUS "General configuration for ${PROJECT_NAME} =====================================")
message( STATUS "")
message( STATUS "    OpenCV path:                ${OpenCV_DIR}")
message( STATUS "    Compiler:                   ${CMAKE_COMPILER}")
message( STATUS "    C++ flags (Release):        ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message( STATUS "    C++ flags (Debug):          ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
if(WIN32)
message( STATUS "    Linker flags (Release):     ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
message( STATUS "    Linker flags (Debug):       ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
else()
message( STATUS "    Linker flags (Release):     ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
message( STATUS "    Linker flags (Debug):       ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
endif()
