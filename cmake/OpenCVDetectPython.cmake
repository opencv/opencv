# FindPython2/FindPython3 CMake wrapper
function(find_python MIN_VERSION)

  string(REGEX MATCH "^[0-9]+" __PYTHON_FAMILY "${MIN_VERSION}")
  set(__PYTHON_PREFIX "Python${__PYTHON_FAMILY}")

  if (NOT ${__PYTHON_PREFIX}_FOUND)
    find_package(${__PYTHON_PREFIX} "${MIN_VERSION}" COMPONENTS Interpreter Development)

    set(${__PYTHON_PREFIX}_Interpreter_FOUND "${${__PYTHON_PREFIX}_Interpreter_FOUND}" PARENT_SCOPE)
    if(${__PYTHON_PREFIX}_Interpreter_FOUND)
        get_filename_component(${__PYTHON_PREFIX}_STDLIB ${${__PYTHON_PREFIX}_STDLIB} ABSOLUTE)
        get_filename_component(${__PYTHON_PREFIX}_STDARCH ${${__PYTHON_PREFIX}_STDARCH} ABSOLUTE)
        get_filename_component(${__PYTHON_PREFIX}_SITELIB ${${__PYTHON_PREFIX}_SITELIB} ABSOLUTE)
        get_filename_component(${__PYTHON_PREFIX}_SITEARCH ${${__PYTHON_PREFIX}_SITEARCH} ABSOLUTE)

        set(${__PYTHON_PREFIX}_EXECUTABLE "${${__PYTHON_PREFIX}_EXECUTABLE}" PARENT_SCOPE)
        set(${__PYTHON_PREFIX}_VERSION "${${__PYTHON_PREFIX}_VERSION}" PARENT_SCOPE)
        set(${__PYTHON_PREFIX}_VERSION_MAJOR "${${__PYTHON_PREFIX}_VERSION_MAJOR}" PARENT_SCOPE)
        set(${__PYTHON_PREFIX}_VERSION_MINOR "${${__PYTHON_PREFIX}_VERSION_MINOR}" PARENT_SCOPE)
        set(${__PYTHON_PREFIX}_VERSION_PATCH "${${__PYTHON_PREFIX}_VERSION_PATCH}" PARENT_SCOPE)
        set(${__PYTHON_PREFIX}_STDLIB "${${__PYTHON_PREFIX}_STDLIB}" PARENT_SCOPE)
        set(${__PYTHON_PREFIX}_STDARCH "${${__PYTHON_PREFIX}_STDARCH}" PARENT_SCOPE)
        set(${__PYTHON_PREFIX}_SITELIB "${${__PYTHON_PREFIX}_SITELIB}" PARENT_SCOPE)
        set(${__PYTHON_PREFIX}_SITEARCH "${${__PYTHON_PREFIX}_SITEARCH}" PARENT_SCOPE)
    endif()

    set(${__PYTHON_PREFIX}_Development_FOUND "${${__PYTHON_PREFIX}_Development_FOUND}" PARENT_SCOPE)
    if(${__PYTHON_PREFIX}_Development_FOUND)
        set(${__PYTHON_PREFIX}_INCLUDE_DIRS "${${__PYTHON_PREFIX}_INCLUDE_DIRS}" PARENT_SCOPE)
        set(${__PYTHON_PREFIX}_LIBRARIES "${${__PYTHON_PREFIX}_LIBRARIES}" PARENT_SCOPE)
        set(${__PYTHON_PREFIX}_RUNTIME_LIBRARY_DIRS "${${__PYTHON_PREFIX}_RUNTIME_LIBRARY_DIRS}" PARENT_SCOPE)
    endif()

    if(${__PYTHON_PREFIX}_Interpreter_FOUND AND NOT ANDROID AND NOT IOS AND NOT CMAKE_CROSSCOMPILING)
        python_process(${__PYTHON_PREFIX}_NUMPY_INCLUDE_DIRS PATH "Path to numpy headers"
            "${${__PYTHON_PREFIX}_EXECUTABLE}"
            "import os; from numpy import distutils; print(os.pathsep.join(distutils.misc_util.get_numpy_include_dirs()))")
        python_process(${__PYTHON_PREFIX}_NUMPY_VERSION INTERNAL ""
            "${${__PYTHON_PREFIX}_EXECUTABLE}"
            "from numpy.version import version as np_ver; print(np_ver)")

        set(${__PYTHON_PREFIX}_NUMPY_INCLUDE_DIRS "${${__PYTHON_PREFIX}_NUMPY_INCLUDE_DIRS}" PARENT_SCOPE)
        set(${__PYTHON_PREFIX}_NUMPY_VERSION "${${__PYTHON_PREFIX}_NUMPY_VERSION}" PARENT_SCOPE)
    endif()

  endif() #NOT ${__PYTHON_PREFIX}_FOUND
endfunction()

function(python_process OUTPUT_VAR OUTPUT_VAR_TYPE OUTPUT_VAR_DESC PYTHON_EXECUTABLE PYTHON_CODE)
    execute_process (COMMAND "${PYTHON_EXECUTABLE}" -c "${PYTHON_CODE}"
                   RESULT_VARIABLE _RESULT
                   OUTPUT_VARIABLE _OUTPUT_VAR
                   OUTPUT_STRIP_TRAILING_WHITESPACE
                   ERROR_QUIET)
    if(NOT _RESULT)
        set(${OUTPUT_VAR} "${_OUTPUT_VAR}" CACHE ${OUTPUT_VAR_TYPE} "${OUTPUT_VAR_DESC}")
    endif()
endfunction()

if(OPENCV_PYTHON_SKIP_DETECTION)
  return()
endif()

if(CMAKE_VERSION VERSION_LESS "3.12")
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${OpenCV_SOURCE_DIR}/cmake/FindPython")
endif()

find_python("${MIN_VER_PYTHON2}")
find_python("${MIN_VER_PYTHON3}")


if(PYTHON_DEFAULT_EXECUTABLE)
    set(PYTHON_DEFAULT_AVAILABLE "TRUE")
elseif(Python2_Interpreter_FOUND)
    # Use Python 2 as default Python interpreter
    set(PYTHON_DEFAULT_AVAILABLE "TRUE")
    set(PYTHON_DEFAULT_EXECUTABLE "${Python2_EXECUTABLE}")
elseif(Python3_Interpreter_FOUND)
    # Use Python 3 as fallback Python interpreter (if there is no Python 2)
    set(PYTHON_DEFAULT_AVAILABLE "TRUE")
    set(PYTHON_DEFAULT_EXECUTABLE "${Python3_EXECUTABLE}")
endif()
