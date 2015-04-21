cmake_minimum_required(VERSION 2.8.9)

if(NOT GETOPT_NAMESPACE)
  set(GETOPT_NAMESPACE "GETOPT")
  set(GETOPT_STANDALONE 1)
endif()
# In all cases:
string(TOLOWER ${GETOPT_NAMESPACE} GETOPT_LIBRARY_NAME)

project(${GETOPT_NAMESPACE} C)

#set(BUILD_SHARED_LIBS OFF)

set(GETOPT_SRCS
  getopt.c
  getopt_long.c
  )
if(WIN32)
  if (BUILD_SHARED_LIBS)
    add_definitions(-DGETOPT_DLL)
  endif ()
endif()

#include_directories(BEFORE
#  ${CMAKE_CURRENT_SOURCE_DIR}
#  )

set(GETOPT_MAJOR_VERSION 1)
set(GETOPT_MINOR_VERSION 0)
set(GETOPT_BUILD_VERSION 0)
set(GETOPT_VERSION
  "${GETOPT_MAJOR_VERSION}.${GETOPT_MINOR_VERSION}.${GETOPT_BUILD_VERSION}")

set(GETOPT_LIBRARY_PROPERTIES ${GETOPT_LIBRARY_PROPERTIES}
  VERSION "${GETOPT_VERSION}"
  SOVERSION "${GETOPT_MAJOR_VERSION}.${GETOPT_MINOR_VERSION}"
)

# Only build this lib static. On most unix system
# we will be using the standart getopt
# so this lib will be for the poor Win32 user (those
# guys really need help)
add_library(${GETOPT_LIBRARY_NAME} ${GETOPT_SRCS})
set_target_properties(${GETOPT_LIBRARY_NAME} PROPERTIES ${GETOPT_LIBRARY_PROPERTIES})
if(CMAKE_COMPILER_IS_GNUCXX AND MINGW)
  # For some reason mingw fails to export the optind which is clearly marked as exported (using extern symbol)
  # ... for now use this flag :
  set_target_properties(${GETOPT_LIBRARY_NAME} PROPERTIES LINK_FLAGS "-Wl,--export-all-symbols")
endif()

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${GETOPT_LIBRARY_NAME} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${GETOPT_LIBRARY_NAME} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

#if(BUILD_EXAMPLES)
#if(GDCM_BUILD_TESTING)
#add_executable(ex_getopt ex_getopt.c)
#target_link_libraries(ex_getopt ${GETOPT_LIBRARY_NAME})
#if(NOT GETOPT_INSTALL_NO_LIBRARIES)
#  install(TARGETS ex_getopt
#    RUNTIME DESTINATION ${GETOPT_INSTALL_BIN_DIR} COMPONENT Applications
#    LIBRARY DESTINATION ${GETOPT_INSTALL_LIB_DIR} COMPONENT Libraries
#    ARCHIVE DESTINATION ${GETOPT_INSTALL_LIB_DIR} COMPONENT DebugDevel
#  ${CPACK_NAMELINK_TYPE}
#  )
#endif()
#endif()
#endif()

if(NOT GETOPT_INSTALL_NO_LIBRARIES)
  install(TARGETS ${GETOPT_LIBRARY_NAME}
    EXPORT ${GDCM_TARGETS_NAME}
    RUNTIME DESTINATION ${GETOPT_INSTALL_BIN_DIR} COMPONENT Applications
    LIBRARY DESTINATION ${GETOPT_INSTALL_LIB_DIR} COMPONENT Libraries #${NAMELINK_SKIP}
    ARCHIVE DESTINATION ${GETOPT_INSTALL_LIB_DIR} COMPONENT DebugDevel
  )
  #if(NAMELINK_ONLY)
  #  install(TARGETS ${GETOPT_LIBRARY_NAME}
  #    LIBRARY DESTINATION ${GETOPT_INSTALL_LIB_DIR} COMPONENT DebugDevel ${NAMELINK_ONLY}
  #  )
  #endif()
endif()

