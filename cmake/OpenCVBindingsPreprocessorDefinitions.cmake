function(ocv_bindings_generator_populate_preprocessor_definitions
         opencv_modules
         output_variable)
  set(defs "\"CV_VERSION_MAJOR\": ${OPENCV_VERSION_MAJOR}")

  macro(ocv_add_definition name value)
    set(defs "${defs},\n\"${name}\": ${value}")
  endmacro()

  ocv_add_definition(CV_VERSION_MINOR ${OPENCV_VERSION_MINOR})
  ocv_add_definition(CV_VERSION_PATCH ${OPENCV_VERSION_PATCH})
  ocv_add_definition(OPENCV_ABI_COMPATIBILITY "${OPENCV_VERSION_MAJOR}00")

  foreach(module IN LISTS ${opencv_modules})
    if(HAVE_${module})
        string(TOUPPER "${module}" module)
        ocv_add_definition("HAVE_${module}" 1)
    endif()
  endforeach()
  if(HAVE_EIGEN)
    ocv_add_definition(HAVE_EIGEN 1)
    ocv_add_definition(EIGEN_WORLD_VERSION ${EIGEN_WORLD_VERSION})
    ocv_add_definition(EIGEN_MAJOR_VERSION ${EIGEN_MAJOR_VERSION})
    ocv_add_definition(EIGEN_MINOR_VERSION ${EIGEN_MINOR_VERSION})
  else()
    # Some checks in parsed headers might not be protected with HAVE_EIGEN check
    ocv_add_definition(EIGEN_WORLD_VERSION 0)
    ocv_add_definition(EIGEN_MAJOR_VERSION 0)
    ocv_add_definition(EIGEN_MINOR_VERSION 0)
  endif()
  if(HAVE_LAPACK)
    ocv_add_definition(HAVE_LAPACK 1)
  endif()

  if(OPENCV_DISABLE_FILESYSTEM_SUPPORT)
    ocv_add_definition(OPENCV_HAVE_FILESYSTEM_SUPPORT 0)
  else()
    ocv_add_definition(OPENCV_HAVE_FILESYSTEM_SUPPORT 1)
  endif()

  ocv_add_definition(OPENCV_BINDINGS_PARSER 1)

  # Implementation details definitions, having no impact on how bindings are
  # generated, so their real values can be safely ignored
  ocv_add_definition(CV_ENABLE_UNROLLED 0)
  ocv_add_definition(CV__EXCEPTION_PTR 0)
  ocv_add_definition(CV_NEON 0)
  ocv_add_definition(TBB_INTERFACE_VERSION 0)
  ocv_add_definition(CV_SSE2 0)
  ocv_add_definition(CV_VSX 0)
  ocv_add_definition(OPENCV_SUPPORTS_FP_DENORMALS_HINT 0)
  ocv_add_definition(CV_LOG_STRIP_LEVEL 0)
  ocv_add_definition(CV_LOG_LEVEL_SILENT 0)
  ocv_add_definition(CV_LOG_LEVEL_FATAL 1)
  ocv_add_definition(CV_LOG_LEVEL_ERROR 2)
  ocv_add_definition(CV_LOG_LEVEL_WARN 3)
  ocv_add_definition(CV_LOG_LEVEL_INFO 4)
  ocv_add_definition(CV_LOG_LEVEL_DEBUG 5)
  ocv_add_definition(CV_LOG_LEVEL_VERBOSE 6)
  ocv_add_definition(CERES_FOUND 0)

  set(${output_variable} ${defs} PARENT_SCOPE)
endfunction()
