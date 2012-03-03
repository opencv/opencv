# ----------------------------------------------------------------------------
#   Uninstall target, for "make uninstall"
# ----------------------------------------------------------------------------
CONFIGURE_FILE(
  "${OpenCV_SOURCE_DIR}/cmake/templates/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

ADD_CUSTOM_TARGET(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")
if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(uninstall PROPERTIES FOLDER "CMakeTargets")
endif()
  

# ----------------------------------------------------------------------------
# Source package, for "make package_source"
# ----------------------------------------------------------------------------
if(BUILD_PACKAGE)
  set(TARBALL_NAME "${CMAKE_PROJECT_NAME}-${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH}")
  if (NOT WIN32)
    if(APPLE)
      set(TAR_CMD gnutar)
    else()
      set(TAR_CMD tar)
    endif()
    set(TAR_TRANSFORM "\"s,^,${TARBALL_NAME}/,\"")
    add_custom_target(package_source
      #TODO: maybe we should not remove dll's
      COMMAND ${TAR_CMD} --transform ${TAR_TRANSFORM} -cjpf ${CMAKE_CURRENT_BINARY_DIR}/${TARBALL_NAME}.tar.bz2 --exclude=".svn" --exclude="*.pyc" --exclude="*.vcproj" --exclude="*/lib/*" --exclude="*.dll" ./
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    add_custom_target(package_source
      COMMAND zip -9 -r ${CMAKE_CURRENT_BINARY_DIR}/${TARBALL_NAME}.zip . -x '*/.svn/*' '*.vcproj' '*.pyc'
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(package_source PROPERTIES FOLDER "extra")
  endif()
endif()


# ----------------------------------------------------------------------------
# performance tests, for "make perf"
# ----------------------------------------------------------------------------
if(BUILD_PERF_TESTS AND PYTHON_EXECUTABLE)
  if(CMAKE_VERSION VERSION_GREATER "2.8.2")
    add_custom_target(perf
      ${PYTHON_EXECUTABLE} "${OpenCV_SOURCE_DIR}/modules/ts/misc/run.py" --configuration $<CONFIGURATION> "${CMAKE_BINARY_DIR}"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      DEPENDS "${OpenCV_SOURCE_DIR}/modules/ts/misc/run.py"
     )
  else()
    add_custom_target(perf
      ${PYTHON_EXECUTABLE} "${OpenCV_SOURCE_DIR}/modules/ts/misc/run.py" "${CMAKE_BINARY_DIR}"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      DEPENDS "${OpenCV_SOURCE_DIR}/modules/ts/misc/run.py"
      )
  endif()
  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(perf PROPERTIES FOLDER "extra")
  endif()
endif()


# ----------------------------------------------------------------------------
# spefial targets to build all OpenCV modules
# ----------------------------------------------------------------------------
add_custom_target(opencv_modules)
if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(opencv_modules PROPERTIES FOLDER "extra")
endif()


# ----------------------------------------------------------------------------
# spefial targets to build all tests
# ----------------------------------------------------------------------------
if(BUILD_TESTS)
  add_custom_target(opencv_tests)
  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(opencv_tests PROPERTIES FOLDER "extra")
  endif()
endif()
if(BUILD_PERF_TESTS)
  add_custom_target(opencv_perf_tests)
  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(opencv_perf_tests PROPERTIES FOLDER "extra")
  endif()
endif()
