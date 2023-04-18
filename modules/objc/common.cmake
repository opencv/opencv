ocv_warnings_disable(CMAKE_CXX_FLAGS -Wdeprecated-declarations)

# get list of modules to wrap
# message(STATUS "Wrapped in Objective-C:")
set(OPENCV_OBJC_MODULES)
foreach(m ${OPENCV_MODULES_BUILD})
  if (";${OPENCV_MODULE_${m}_WRAPPERS};" MATCHES ";objc;" AND HAVE_${m})
    list(APPEND OPENCV_OBJC_MODULES ${m})
    #message(STATUS "\t${m}")
  endif()
endforeach()
