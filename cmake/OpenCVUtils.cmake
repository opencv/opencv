if(COMMAND ocv_cmake_dump_vars)  # include guard
  return()
endif()

include(CMakeParseArguments)

# Debugging function
function(ocv_cmake_dump_vars)
  set(OPENCV_SUPPRESS_DEPRECATIONS 1)  # suppress deprecation warnings from variable_watch() guards
  get_cmake_property(__variableNames VARIABLES)
  cmake_parse_arguments(DUMP "FORCE" "TOFILE" "" ${ARGN})

  # avoid generation of excessive logs with "--trace" or "--trace-expand" parameters
  # Note: `-DCMAKE_TRACE_MODE=1` should be passed to CMake through command line. It is not a CMake buildin variable for now (2020-12)
  #       Use `cmake . -UCMAKE_TRACE_MODE` to remove this variable from cache
  if(CMAKE_TRACE_MODE AND NOT DUMP_FORCE)
    if(DUMP_TOFILE)
      file(WRITE ${CMAKE_BINARY_DIR}/${DUMP_TOFILE} "Skipped due to enabled CMAKE_TRACE_MODE")
    else()
      message(AUTHOR_WARNING "ocv_cmake_dump_vars() is skipped due to enabled CMAKE_TRACE_MODE")
    endif()
    return()
  endif()

  set(regex "${DUMP_UNPARSED_ARGUMENTS}")
  string(TOLOWER "${regex}" regex_lower)
  set(__VARS "")
  foreach(__variableName ${__variableNames})
    string(TOLOWER "${__variableName}" __variableName_lower)
    if((__variableName MATCHES "${regex}" OR __variableName_lower MATCHES "${regex_lower}")
        AND NOT __variableName_lower MATCHES "^__")
      get_property(__value VARIABLE PROPERTY "${__variableName}")
      set(__VARS "${__VARS}${__variableName}=${__value}\n")
    endif()
  endforeach()
  if(DUMP_TOFILE)
    file(WRITE ${CMAKE_BINARY_DIR}/${DUMP_TOFILE} "${__VARS}")
  else()
    message(AUTHOR_WARNING "${__VARS}")
  endif()
endfunction()


#
# CMake script hooks support
#
option(OPENCV_DUMP_HOOKS_FLOW "Dump called OpenCV hooks" OFF)
macro(ocv_cmake_hook_append hook_name)
  set(__var_name "__OPENCV_CMAKE_HOOKS_${hook_name}")
  set(__value "${${__var_name}}")
  message(STATUS "Registering hook '${hook_name}': ${ARGN}")
  list(APPEND __value ${ARGN})
  set(${__var_name} "${__value}" CACHE INTERNAL "")
endmacro()
macro(ocv_cmake_hook hook_name)
  set(__var_name "__OPENCV_CMAKE_HOOKS_${hook_name}")
  if(OPENCV_DUMP_HOOKS_FLOW)
    message(STATUS "Hook ${hook_name} ...")
  endif()
  foreach(__hook ${${__var_name}})
    #message(STATUS "Hook ${hook_name}: calling '${__hook}' ...")
    if(COMMAND "${__hook}")
      message(FATAL_ERROR "Indirect calling of CMake commands is not supported yet")
    else()
      include("${__hook}")
    endif()
  endforeach()
endmacro()
macro(ocv_cmake_reset_hooks)
  get_cmake_property(__variableNames VARIABLES)
  foreach(__variableName ${__variableNames})
    if(__variableName MATCHES "^__OPENCV_CMAKE_HOOKS_")
      unset(${__variableName})
      unset(${__variableName} CACHE)
    endif()
  endforeach()
endmacro()
macro(ocv_cmake_hook_register_dir dir)
  file(GLOB hook_files RELATIVE "${dir}" "${dir}/*.cmake")
  foreach(f ${hook_files})
    if(f MATCHES "^(.+)\\.cmake$")
      set(hook_name "${CMAKE_MATCH_1}")
      ocv_cmake_hook_append(${hook_name} "${dir}/${f}")
    endif()
  endforeach()
endmacro()


function(ocv_cmake_eval var_name)
  if(DEFINED ${var_name})
    file(WRITE "${CMAKE_BINARY_DIR}/CMakeCommand-${var_name}.cmake" ${${var_name}})
    include("${CMAKE_BINARY_DIR}/CMakeCommand-${var_name}.cmake")
  endif()
  if(";${ARGN};" MATCHES ";ONCE;")
    unset(${var_name} CACHE)
  endif()
endfunction()

macro(ocv_cmake_configure file_name var_name)
  file(READ "${file_name}" __config)
  string(CONFIGURE "${__config}" ${var_name} ${ARGN})
endmacro()

macro(ocv_update VAR)
  if(NOT DEFINED ${VAR})
    if("x${ARGN}" STREQUAL "x")
      set(${VAR} "")
    else()
      set(${VAR} ${ARGN})
    endif()
  else()
    #ocv_debug_message("Preserve old value for ${VAR}: ${${VAR}}")
  endif()
endmacro()

function(_ocv_access_removed_variable VAR ACCESS)
  if(ACCESS STREQUAL "MODIFIED_ACCESS")
    set(OPENCV_SUPPRESS_MESSAGE_REMOVED_VARIABLE_${VAR} 1 PARENT_SCOPE)
    return()
  endif()
  if(ACCESS MATCHES "UNKNOWN_.*"
      AND NOT OPENCV_SUPPRESS_MESSAGE_REMOVED_VARIABLE
      AND NOT OPENCV_SUPPRESS_MESSAGE_REMOVED_VARIABLE_${VAR}
  )
    message(WARNING "OpenCV: Variable has been removed from CMake scripts: ${VAR}")
    set(OPENCV_SUPPRESS_MESSAGE_REMOVED_VARIABLE_${VAR} 1 PARENT_SCOPE)  # suppress similar messages
  endif()
endfunction()
macro(ocv_declare_removed_variable VAR)
  if(NOT DEFINED ${VAR})  # don't hit external variables
    variable_watch(${VAR} _ocv_access_removed_variable)
  endif()
endmacro()
macro(ocv_declare_removed_variables)
  foreach(_var ${ARGN})
    ocv_declare_removed_variable(${_var})
  endforeach()
endmacro()

# Search packages for the host system instead of packages for the target system
# in case of cross compilation these macros should be defined by the toolchain file
if(NOT COMMAND find_host_package)
  macro(find_host_package)
    find_package(${ARGN})
  endmacro()
endif()
if(NOT COMMAND find_host_program)
  macro(find_host_program)
    find_program(${ARGN})
  endmacro()
endif()

# assert macro
# Note: it doesn't support lists in arguments
# Usage samples:
#   ocv_assert(MyLib_FOUND)
#   ocv_assert(DEFINED MyLib_INCLUDE_DIRS)
macro(ocv_assert)
  if(NOT (${ARGN}))
    string(REPLACE ";" " " __assert_msg "${ARGN}")
    message(AUTHOR_WARNING "Assertion failed: ${__assert_msg}")
  endif()
endmacro()

macro(ocv_debug_message)
  if(OPENCV_CMAKE_DEBUG_MESSAGES)
    string(REPLACE ";" " " __msg "${ARGN}")
    message(STATUS "${__msg}")
  endif()
endmacro()

macro(ocv_check_environment_variables)
  foreach(_var ${ARGN})
    if(" ${${_var}}" STREQUAL " " AND DEFINED ENV{${_var}})
      set(__value "$ENV{${_var}}")
      file(TO_CMAKE_PATH "${__value}" __value) # Assume that we receive paths
      set(${_var} "${__value}")
      message(STATUS "Update variable ${_var} from environment: ${${_var}}")
    endif()
  endforeach()
endmacro()

macro(ocv_path_join result_var P1 P2_)
  string(REGEX REPLACE "^[/]+" "" P2 "${P2_}")
  if("${P1}" STREQUAL "" OR "${P1}" STREQUAL ".")
    set(${result_var} "${P2}")
  elseif("${P1}" STREQUAL "/")
    set(${result_var} "/${P2}")
  elseif("${P2}" STREQUAL "")
    set(${result_var} "${P1}")
  else()
    set(${result_var} "${P1}/${P2}")
  endif()
  string(REPLACE "\\\\" "\\" ${result_var} "${${result_var}}")
  string(REPLACE "//" "/" ${result_var} "${${result_var}}")
  string(REGEX REPLACE "(^|[/\\])[\\.][/\\]" "\\1" ${result_var} "${${result_var}}")
  if("${${result_var}}" STREQUAL "")
    set(${result_var} ".")
  endif()
  #message(STATUS "'${P1}' '${P2_}' => '${${result_var}}'")
endmacro()


# Used to parse Android SDK 'source.properties' files
# File lines format:
# - '<var_name>=<value>' (with possible 'space' symbols around '=')
# - '#<any comment>'
# Parsed values are saved into CMake variables:
# - '${var_prefix}_${var_name}'
# Flags:
# - 'CACHE_VAR <var1> <var2>' - put these properties into CMake internal cache
# - 'MSG_PREFIX <msg>' - prefix string for emitted messages
# - flag 'VALIDATE' - emit messages about missing values from required cached variables
# - flag 'WARNING' - emit CMake WARNING instead of STATUS messages
function(ocv_parse_properties_file file var_prefix)
  cmake_parse_arguments(PARSE_PROPERTIES_PARAM "VALIDATE;WARNING" "" "CACHE_VAR;MSG_PREFIX" ${ARGN})

  set(__msg_type STATUS)
  if(PARSE_PROPERTIES_PARAM_WARNING)
    set(__msg_type WARNING)
  endif()

  if(EXISTS "${file}")
    set(SOURCE_PROPERTIES_REGEX "^[ ]*([^=:\n\"' ]+)[ ]*=[ ]*(.*)$")
    file(STRINGS "${file}" SOURCE_PROPERTIES_LINES REGEX "^[ ]*[^#].*$")
    foreach(line ${SOURCE_PROPERTIES_LINES})
      if(line MATCHES "${SOURCE_PROPERTIES_REGEX}")
        set(__name "${CMAKE_MATCH_1}")
        set(__value "${CMAKE_MATCH_2}")
        string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" __name ${__name})
        if(";${PARSE_PROPERTIES_PARAM_CACHE_VAR};" MATCHES ";${__name};")
          set(${var_prefix}_${__name} "${__value}" CACHE INTERNAL "from ${file}")
        else()
          set(${var_prefix}_${__name} "${__value}" PARENT_SCOPE)
        endif()
      else()
        message(${__msg_type} "${PARSE_PROPERTIES_PARAM_MSG_PREFIX}Can't parse source property: '${line}' (from ${file})")
      endif()
    endforeach()
    if(PARSE_PROPERTIES_PARAM_VALIDATE)
      set(__missing "")
      foreach(__name ${PARSE_PROPERTIES_PARAM_CACHE_VAR})
        if(NOT DEFINED ${var_prefix}_${__name})
          list(APPEND __missing ${__name})
        endif()
      endforeach()
      if(__missing)
        message(${__msg_type} "${PARSE_PROPERTIES_PARAM_MSG_PREFIX}Can't read properties '${__missing}' from '${file}'")
      endif()
    endif()
  else()
    message(${__msg_type} "${PARSE_PROPERTIES_PARAM_MSG_PREFIX}Can't find file: ${file}")
  endif()
endfunction()



# rename modules target to world if needed
macro(_ocv_fix_target target_var)
  if(BUILD_opencv_world)
    if(OPENCV_MODULE_${${target_var}}_IS_PART_OF_WORLD)
      set(${target_var} opencv_world)
    endif()
  endif()
endmacro()


# check if "sub" (file or dir) is below "dir"
function(ocv_is_subdir res dir sub )
  get_filename_component(dir "${dir}" ABSOLUTE)
  get_filename_component(sub "${sub}" ABSOLUTE)
  file(TO_CMAKE_PATH "${dir}" dir)
  file(TO_CMAKE_PATH "${sub}" sub)
  set(dir "${dir}/")
  string(LENGTH "${dir}" len)
  string(LENGTH "${sub}" len_sub)
  if(NOT len GREATER len_sub)
    string(SUBSTRING "${sub}" 0 ${len} prefix)
  endif()
  if(prefix AND prefix STREQUAL dir)
    set(${res} TRUE PARENT_SCOPE)
  else()
    set(${res} FALSE PARENT_SCOPE)
  endif()
endfunction()


function(ocv_is_opencv_directory result_var dir)
  set(result FALSE)
  foreach(parent ${OpenCV_SOURCE_DIR} ${OpenCV_BINARY_DIR} ${OPENCV_EXTRA_MODULES_PATH})
    ocv_is_subdir(result "${parent}" "${dir}")
    if(result)
      break()
    endif()
  endforeach()
  set(${result_var} ${result} PARENT_SCOPE)
endfunction()


# adds include directories in such a way that directories from the OpenCV source tree go first
function(ocv_include_directories)
  ocv_debug_message("ocv_include_directories( ${ARGN} )")
  set(__add_before "")
  foreach(dir ${ARGN})
    ocv_is_opencv_directory(__is_opencv_dir "${dir}")
    if(__is_opencv_dir)
      list(APPEND __add_before "${dir}")
    elseif(((CV_GCC AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0") OR CV_CLANG) AND
           dir MATCHES "/usr/include$")
      # workaround for GCC 6.x bug
    else()
      include_directories(AFTER SYSTEM "${dir}")
    endif()
  endforeach()
  include_directories(BEFORE ${__add_before})
endfunction()

function(ocv_append_target_property target prop)
  get_target_property(val ${target} ${prop})
  if(val)
    set(val "${val} ${ARGN}")
    set_target_properties(${target} PROPERTIES ${prop} "${val}")
  else()
    set_target_properties(${target} PROPERTIES ${prop} "${ARGN}")
  endif()
endfunction()

if(DEFINED OPENCV_DEPENDANT_TARGETS_LIST)
  foreach(v ${OPENCV_DEPENDANT_TARGETS_LIST})
    unset(${v} CACHE)
  endforeach()
  unset(OPENCV_DEPENDANT_TARGETS_LIST CACHE)
endif()

function(ocv_append_dependant_targets target)
  #ocv_debug_message("ocv_append_dependant_targets(${target} ${ARGN})")
  _ocv_fix_target(target)
  list(FIND OPENCV_DEPENDANT_TARGETS_LIST "OPENCV_DEPENDANT_TARGETS_${target}" __id)
  if(__id EQUAL -1)
    list(APPEND OPENCV_DEPENDANT_TARGETS_LIST "OPENCV_DEPENDANT_TARGETS_${target}")
    list(SORT OPENCV_DEPENDANT_TARGETS_LIST)
    set(OPENCV_DEPENDANT_TARGETS_LIST "${OPENCV_DEPENDANT_TARGETS_LIST}" CACHE INTERNAL "")
  endif()
  set(OPENCV_DEPENDANT_TARGETS_${target} "${OPENCV_DEPENDANT_TARGETS_${target}};${ARGN}" CACHE INTERNAL "" FORCE)
endfunction()

# adds include directories in such a way that directories from the OpenCV source tree go first
function(ocv_target_include_directories target)
  #ocv_debug_message("ocv_target_include_directories(${target} ${ARGN})")
  _ocv_fix_target(target)
  set(__params "")
  if(CV_GCC AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0" AND
      ";${ARGN};" MATCHES "/usr/include;")
    return() # workaround for GCC 6.x bug
  endif()
  set(__params "")
  set(__system_params "")
  set(__var_name __params)
  foreach(dir ${ARGN})
    if("${dir}" STREQUAL "SYSTEM")
      set(__var_name __system_params)
    else()
      get_filename_component(__abs_dir "${dir}" ABSOLUTE)
      ocv_is_opencv_directory(__is_opencv_dir "${dir}")
      if(__is_opencv_dir)
        list(APPEND ${__var_name} "${__abs_dir}")
      else()
        list(APPEND ${__var_name} "${dir}")
      endif()
    endif()
  endforeach()
  if(HAVE_CUDA OR CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories(${__params})
    include_directories(SYSTEM ${__system_params})
  else()
    if(TARGET ${target})
      if(__params)
        target_include_directories(${target} PRIVATE ${__params})
        if(OPENCV_DEPENDANT_TARGETS_${target})
          foreach(t ${OPENCV_DEPENDANT_TARGETS_${target}})
            target_include_directories(${t} PRIVATE ${__params})
          endforeach()
        endif()
      endif()
      if(__system_params)
        target_include_directories(${target} SYSTEM PRIVATE ${__system_params})
        if(OPENCV_DEPENDANT_TARGETS_${target})
          foreach(t ${OPENCV_DEPENDANT_TARGETS_${target}})
            target_include_directories(${t} SYSTEM PRIVATE ${__system_params})
          endforeach()
        endif()
      endif()
    else()
      if(__params)
        set(__new_inc ${OCV_TARGET_INCLUDE_DIRS_${target}})
        list(APPEND __new_inc ${__params})
        set(OCV_TARGET_INCLUDE_DIRS_${target} "${__new_inc}" CACHE INTERNAL "")
      endif()
      if(__system_params)
        set(__new_inc ${OCV_TARGET_INCLUDE_SYSTEM_DIRS_${target}})
        list(APPEND __new_inc ${__system_params})
        set(OCV_TARGET_INCLUDE_SYSTEM_DIRS_${target} "${__new_inc}" CACHE INTERNAL "")
      endif()
    endif()
  endif()
endfunction()

# clears all passed variables
macro(ocv_clear_vars)
  foreach(_var ${ARGN})
    unset(${_var})
    unset(${_var} CACHE)
  endforeach()
endmacro()


# Clears passed variables with INTERNAL type from CMake cache
macro(ocv_clear_internal_cache_vars)
  foreach(_var ${ARGN})
    get_property(_propertySet CACHE ${_var} PROPERTY TYPE SET)
    if(_propertySet)
      get_property(_type CACHE ${_var} PROPERTY TYPE)
      if(_type STREQUAL "INTERNAL")
        message("Cleaning INTERNAL cached variable: ${_var}")
        unset(${_var} CACHE)
      endif()
    endif()
  endforeach()
  unset(_propertySet)
  unset(_type)
endmacro()


set(OCV_COMPILER_FAIL_REGEX
    "argument .* is not valid"                  # GCC 9+ (including support of unicode quotes)
    "command[- ]line option .* is valid for .* but not for C\\+\\+" # GNU
    "command[- ]line option .* is valid for .* but not for C" # GNU
    "unrecognized .*option"                     # GNU
    "unknown .*option"                          # Clang
    "ignoring unknown option"                   # MSVC
    "warning D9002"                             # MSVC, any lang
    "option .*not supported"                    # Intel
    "[Uu]nknown option"                         # HP
    "[Ww]arning: [Oo]ption"                     # SunPro
    "command option .* is not recognized"       # XL
    "not supported in this configuration, ignored"       # AIX (';' is replaced with ',')
    "File with unknown suffix passed to linker" # PGI
    "WARNING: unknown flag:"                    # Open64
  )

MACRO(ocv_check_compiler_flag LANG FLAG RESULT)
  set(_fname "${ARGN}")
  if(NOT DEFINED ${RESULT})
    if(_fname)
      # nothing
    elseif("_${LANG}_" MATCHES "_CXX_")
      set(_fname "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx")
      if("${CMAKE_CXX_FLAGS} ${FLAG} " MATCHES "-Werror " OR "${CMAKE_CXX_FLAGS} ${FLAG} " MATCHES "-Werror=unknown-pragmas ")
        FILE(WRITE "${_fname}" "int main() { return 0; }\n")
      else()
        FILE(WRITE "${_fname}" "#pragma\nint main() { return 0; }\n")
      endif()
    elseif("_${LANG}_" MATCHES "_C_")
      set(_fname "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.c")
      if("${CMAKE_C_FLAGS} ${FLAG} " MATCHES "-Werror " OR "${CMAKE_C_FLAGS} ${FLAG} " MATCHES "-Werror=unknown-pragmas ")
        FILE(WRITE "${_fname}" "int main(void) { return 0; }\n")
      else()
        FILE(WRITE "${_fname}" "#pragma\nint main(void) { return 0; }\n")
      endif()
    elseif("_${LANG}_" MATCHES "_OBJCXX_")
      set(_fname "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.mm")
      if("${CMAKE_CXX_FLAGS} ${FLAG} " MATCHES "-Werror " OR "${CMAKE_CXX_FLAGS} ${FLAG} " MATCHES "-Werror=unknown-pragmas ")
        FILE(WRITE "${_fname}" "int main() { return 0; }\n")
      else()
        FILE(WRITE "${_fname}" "#pragma\nint main() { return 0; }\n")
      endif()
    else()
      unset(_fname)
    endif()
    if(_fname)
      if(NOT "x${ARGN}" STREQUAL "x")
        file(RELATIVE_PATH __msg "${CMAKE_SOURCE_DIR}" "${ARGN}")
        set(__msg " (check file: ${__msg})")
      else()
        set(__msg "")
      endif()
      if(CMAKE_REQUIRED_LIBRARIES)
        set(__link_libs LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
      else()
        set(__link_libs)
      endif()
      set(__cmake_flags "")
      if(CMAKE_EXE_LINKER_FLAGS)  # CMP0056 do this on new CMake
        list(APPEND __cmake_flags "-DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}")
      endif()

      # CMP0067 do this on new CMake
      if(DEFINED CMAKE_CXX_STANDARD)
        list(APPEND __cmake_flags "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}")
      endif()
      if(DEFINED CMAKE_CXX_STANDARD_REQUIRED)
        list(APPEND __cmake_flags "-DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}")
      endif()
      if(DEFINED CMAKE_CXX_EXTENSIONS)
        list(APPEND __cmake_flags "-DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}")
      endif()

      MESSAGE(STATUS "Performing Test ${RESULT}${__msg}")
      TRY_COMPILE(${RESULT}
        "${CMAKE_BINARY_DIR}"
        "${_fname}"
        CMAKE_FLAGS ${__cmake_flags}
        COMPILE_DEFINITIONS "${FLAG}"
        ${__link_libs}
        OUTPUT_VARIABLE OUTPUT)

      if(${RESULT})
        string(REPLACE ";" "," OUTPUT_LINES "${OUTPUT}")
        string(REPLACE "\n" ";" OUTPUT_LINES "${OUTPUT_LINES}")
        foreach(_regex ${OCV_COMPILER_FAIL_REGEX})
          if(NOT ${RESULT})
            break()
          endif()
          foreach(_line ${OUTPUT_LINES})
            if("${_line}" MATCHES "${_regex}")
              file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
                  "Build output check failed:\n"
                  "    Regex: '${_regex}'\n"
                  "    Output line: '${_line}'\n")
              set(${RESULT} 0)
              break()
            endif()
          endforeach()
        endforeach()
      endif()

      IF(${RESULT})
        SET(${RESULT} 1 CACHE INTERNAL "Test ${RESULT}")
        MESSAGE(STATUS "Performing Test ${RESULT} - Success")
      ELSE(${RESULT})
        MESSAGE(STATUS "Performing Test ${RESULT} - Failed")
        SET(${RESULT} "" CACHE INTERNAL "Test ${RESULT}")
        file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
            "Compilation failed:\n"
            "    source file: '${_fname}'\n"
            "    check option: '${FLAG}'\n"
            "===== BUILD LOG =====\n"
            "${OUTPUT}\n"
            "===== END =====\n\n")
      ENDIF(${RESULT})
    else()
      SET(${RESULT} 0)
    endif()
  endif()
ENDMACRO()

macro(ocv_check_flag_support lang flag varname base_options)
  if(CMAKE_BUILD_TYPE)
    set(CMAKE_TRY_COMPILE_CONFIGURATION ${CMAKE_BUILD_TYPE})
  endif()

  if("_${lang}_" MATCHES "_CXX_")
    set(_lang CXX)
  elseif("_${lang}_" MATCHES "_C_")
    set(_lang C)
  elseif("_${lang}_" MATCHES "_OBJCXX_")
    if(DEFINED CMAKE_OBJCXX_COMPILER)  # CMake 3.16+ and enable_language(OBJCXX) call are required
      set(_lang OBJCXX)
    else()
      set(_lang CXX)
    endif()
  else()
    set(_lang ${lang})
  endif()

  string(TOUPPER "${flag}" ${varname})
  string(REGEX REPLACE "^(/|-)" "HAVE_${_lang}_" ${varname} "${${varname}}")
  string(REGEX REPLACE " -|-|=| |\\.|," "_" ${varname} "${${varname}}")

  if(DEFINED CMAKE_${_lang}_COMPILER)
    ocv_check_compiler_flag("${_lang}" "${base_options} ${flag}" ${${varname}} ${ARGN})
  endif()
endmacro()

macro(ocv_check_runtime_flag flag result)
  set(_fname "${ARGN}")
  if(NOT DEFINED ${result})
    file(RELATIVE_PATH _rname "${CMAKE_SOURCE_DIR}" "${_fname}")
    message(STATUS "Performing Runtime Test ${result} (check file: ${_rname})")
    try_run(exec_return compile_result
      "${CMAKE_BINARY_DIR}"
      "${_fname}"
      CMAKE_FLAGS "-DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}" # CMP0056 do this on new CMake
      COMPILE_DEFINITIONS "${flag}"
      OUTPUT_VARIABLE OUTPUT)

    if(${compile_result})
      if(exec_return EQUAL 0)
        set(${result} 1 CACHE INTERNAL "Runtime Test ${result}")
        message(STATUS "Performing Runtime Test ${result} - Success")
      else()
        message(STATUS "Performing Runtime Test ${result} - Failed(${exec_return})")
        set(${result} 0 CACHE INTERNAL "Runtime Test ${result}")
      endif()
    else()
      set(${result} 0 CACHE INTERNAL "Runtime Test ${result}")
      message(STATUS "Performing Runtime Test ${result} - Compiling Failed")
    endif()

    if(NOT ${result})
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
        "Runtime Test failed:\n"
        "    source file: '${_fname}'\n"
        "    check option: '${flag}'\n"
        "    exec return: ${exec_return}\n"
        "===== BUILD AND RUNTIME LOG =====\n"
        "${OUTPUT}\n"
        "===== END =====\n\n")
    endif()
  endif()
endmacro()

# turns off warnings
macro(ocv_warnings_disable)
  if(NOT ENABLE_NOISY_WARNINGS)
    set(_flag_vars "")
    set(_msvc_warnings "")
    set(_gxx_warnings "")
    set(_icc_warnings "")
    foreach(arg ${ARGN})
      if(arg MATCHES "^CMAKE_")
        list(APPEND _flag_vars ${arg})
      elseif(arg MATCHES "^/wd")
        list(APPEND _msvc_warnings ${arg})
      elseif(arg MATCHES "^-W")
        list(APPEND _gxx_warnings ${arg})
      elseif(arg MATCHES "^-wd" OR arg MATCHES "^-Qwd" OR arg MATCHES "^/Qwd")
        list(APPEND _icc_warnings ${arg})
      endif()
    endforeach()
    if(MSVC AND _msvc_warnings AND _flag_vars)
      foreach(var ${_flag_vars})
        foreach(warning ${_msvc_warnings})
          set(${var} "${${var}} ${warning}")
        endforeach()
      endforeach()
    elseif(((CV_GCC OR CV_CLANG) OR (UNIX AND CV_ICC)) AND _gxx_warnings AND _flag_vars)
      foreach(var ${_flag_vars})
        foreach(warning ${_gxx_warnings})
          if(NOT warning MATCHES "^-Wno-")
            string(REGEX REPLACE "(^|[ ]+)${warning}(=[^ ]*)?([ ]+|$)" " " ${var} "${${var}}")
            string(REPLACE "-W" "-Wno-" warning "${warning}")
          endif()
          ocv_check_flag_support(${var} "${warning}" _varname "")
          if(${_varname})
            set(${var} "${${var}} ${warning}")
          endif()
        endforeach()
      endforeach()
    endif()
    if(CV_ICC AND _icc_warnings AND _flag_vars)
      foreach(var ${_flag_vars})
        foreach(warning ${_icc_warnings})
          if(UNIX)
            string(REPLACE "-Qwd" "-wd" warning "${warning}")
          else()
            string(REPLACE "-wd" "-Qwd" warning "${warning}")
          endif()
          ocv_check_flag_support(${var} "${warning}" _varname "")
          if(${_varname})
            set(${var} "${${var}} ${warning}")
          endif()
        endforeach()
      endforeach()
    endif()
    unset(_flag_vars)
    unset(_msvc_warnings)
    unset(_gxx_warnings)
    unset(_icc_warnings)
  endif(NOT ENABLE_NOISY_WARNINGS)
endmacro()

macro(ocv_append_source_file_compile_definitions source)
  get_source_file_property(_value "${source}" COMPILE_DEFINITIONS)
  if(_value)
    set(_value ${_value} ${ARGN})
  else()
    set(_value ${ARGN})
  endif()
  set_source_files_properties("${source}" PROPERTIES COMPILE_DEFINITIONS "${_value}")
endmacro()

macro(add_apple_compiler_options the_module)
  ocv_check_flag_support(OBJCXX "-fobjc-exceptions" HAVE_OBJC_EXCEPTIONS "")
  if(HAVE_OBJC_EXCEPTIONS)
    foreach(source ${OPENCV_MODULE_${the_module}_SOURCES})
      if("${source}" MATCHES "\\.mm$")
        get_source_file_property(flags "${source}" COMPILE_FLAGS)
        if(flags)
          set(flags "${_flags} -fobjc-exceptions")
        else()
          set(flags "-fobjc-exceptions")
        endif()

        set_source_files_properties("${source}" PROPERTIES COMPILE_FLAGS "${flags}")
      endif()
    endforeach()
  endif()
endmacro()

# Provides an option that the user can optionally select.
# Can accept condition to control when option is available for user.
# Usage:
#   option(<option_variable>
#          "help string describing the option"
#          <initial value or boolean expression>
#          [VISIBLE_IF <condition>]
#          [VERIFY <condition>])
macro(OCV_OPTION variable description value)
  set(__value ${value})
  set(__condition "")
  set(__verification)
  set(__varname "__value")
  foreach(arg ${ARGN})
    if(arg STREQUAL "IF" OR arg STREQUAL "if" OR arg STREQUAL "VISIBLE_IF")
      set(__varname "__condition")
    elseif(arg STREQUAL "VERIFY")
      set(__varname "__verification")
    else()
      list(APPEND ${__varname} ${arg})
    endif()
  endforeach()
  unset(__varname)
  if(__condition STREQUAL "")
    set(__condition 2 GREATER 1)
  endif()

  if(${__condition})
    if(__value MATCHES ";")
      if(${__value})
        option(${variable} "${description}" ON)
      else()
        option(${variable} "${description}" OFF)
      endif()
    elseif(DEFINED ${__value})
      if(${__value})
        option(${variable} "${description}" ON)
      else()
        option(${variable} "${description}" OFF)
      endif()
    else()
      option(${variable} "${description}" ${__value})
    endif()
  else()
    if(DEFINED ${variable} AND "${${variable}}"  # emit warnings about turned ON options only.
        AND NOT (OPENCV_HIDE_WARNING_UNSUPPORTED_OPTION OR "$ENV{OPENCV_HIDE_WARNING_UNSUPPORTED_OPTION}")
    )
      message(WARNING "Unexpected option: ${variable} (=${${variable}})\nCondition: IF (${__condition})")
    endif()
    if(OPENCV_UNSET_UNSUPPORTED_OPTION)
      unset(${variable} CACHE)
    endif()
  endif()
  if(__verification)
    set(OPENCV_VERIFY_${variable} "${__verification}") # variable containing condition to verify
    list(APPEND OPENCV_VERIFICATIONS "${variable}") # list of variable names (WITH_XXX;WITH_YYY;...)
  endif()
  unset(__condition)
  unset(__value)
endmacro()


# Check that each variable stored in OPENCV_VERIFICATIONS list
# is consistent with actual detection result (stored as condition in OPENCV_VERIFY_...) variables
function(ocv_verify_config)
  set(broken_options)
  foreach(var ${OPENCV_VERIFICATIONS})
    set(evaluated FALSE)
    if(${OPENCV_VERIFY_${var}})
      set(evaluated TRUE)
    endif()
    status("Verifying ${var}=${${var}} => '${OPENCV_VERIFY_${var}}'=${evaluated}")
    if (${var} AND NOT evaluated)
      list(APPEND broken_options ${var})
      message(WARNING
        "Option ${var} is enabled but corresponding dependency "
        "have not been found: \"${OPENCV_VERIFY_${var}}\" is FALSE")
    elseif(NOT ${var} AND evaluated)
      list(APPEND broken_options ${var})
      message(WARNING
        "Option ${var} is disabled or unset but corresponding dependency "
        "have been explicitly turned on: \"${OPENCV_VERIFY_${var}}\" is TRUE")
    endif()
  endforeach()
  if(broken_options)
    string(REPLACE ";" "\n" broken_options "${broken_options}")
    message(FATAL_ERROR
      "Some dependencies have not been found or have been forced, "
      "unset ENABLE_CONFIG_VERIFICATION option to ignore these failures "
      "or change following options:\n${broken_options}")
  endif()
endfunction()

# Usage: ocv_append_build_options(HIGHGUI FFMPEG)
macro(ocv_append_build_options var_prefix pkg_prefix)
  foreach(suffix INCLUDE_DIRS LIBRARIES LIBRARY_DIRS LINK_LIBRARIES)
    if(${pkg_prefix}_${suffix})
      list(APPEND ${var_prefix}_${suffix} ${${pkg_prefix}_${suffix}})
      list(REMOVE_DUPLICATES ${var_prefix}_${suffix})
    endif()
  endforeach()
endmacro()

function(ocv_append_source_files_cxx_compiler_options files_var)
  set(__flags "${ARGN}")
  ocv_check_flag_support(CXX "${__flags}" __HAVE_COMPILER_OPTIONS_VAR "")
  if(${__HAVE_COMPILER_OPTIONS_VAR})
    foreach(source ${${files_var}})
      if("${source}" MATCHES "\\.(cpp|cc|cxx)$")
        get_source_file_property(flags "${source}" COMPILE_FLAGS)
        if(flags)
          set(flags "${flags} ${__flags}")
        else()
          set(flags "${__flags}")
        endif()
        set_source_files_properties("${source}" PROPERTIES COMPILE_FLAGS "${flags}")
      endif()
    endforeach()
  endif()
endfunction()

# Usage is similar to CMake 'pkg_check_modules' command
# It additionally controls HAVE_${define} and ${define}_${modname}_FOUND variables
macro(ocv_check_modules define)
  unset(HAVE_${define})
  foreach(m ${ARGN})
    if (m MATCHES "(.*[^><])(>=|=|<=)(.*)")
      set(__modname "${CMAKE_MATCH_1}")
    else()
      set(__modname "${m}")
    endif()
    unset(${define}_${__modname}_FOUND)
  endforeach()
  if(PKG_CONFIG_FOUND OR PkgConfig_FOUND)
    pkg_check_modules(${define} ${ARGN})
  endif()
  if(${define}_FOUND)
    set(HAVE_${define} 1)
  endif()
  foreach(m ${ARGN})
    if (m MATCHES "(.*[^><])(>=|=|<=)(.*)")
      set(__modname "${CMAKE_MATCH_1}")
    else()
      set(__modname "${m}")
    endif()
    if(NOT DEFINED ${define}_${__modname}_FOUND AND ${define}_FOUND)
      set(${define}_${__modname}_FOUND 1)
    endif()
  endforeach()
  if(${define}_FOUND AND ${define}_LIBRARIES)
    if(${define}_LINK_LIBRARIES_XXXXX)  # CMake 3.12+: https://gitlab.kitware.com/cmake/cmake/merge_requests/2068
      set(${define}_LIBRARIES "${${define}_LINK_LIBRARIES}" CACHE INTERNAL "")
    else()
      unset(_libs)          # absolute paths
      unset(_libs_paths)  # -L args
      foreach(flag ${${define}_LDFLAGS})
        if(flag MATCHES "^-L(.*)")
          list(APPEND _libs_paths ${CMAKE_MATCH_1})
        elseif(IS_ABSOLUTE "${flag}")
          list(APPEND _libs "${flag}")
        elseif(flag MATCHES "^-l(.*)")
          set(_lib "${CMAKE_MATCH_1}")
          if(_libs_paths)
            find_library(pkgcfg_lib_${define}_${_lib} NAMES ${_lib}
                         HINTS ${_libs_paths} NO_DEFAULT_PATH)
          endif()
          find_library(pkgcfg_lib_${define}_${_lib} NAMES ${_lib})
          mark_as_advanced(pkgcfg_lib_${define}_${_lib})
          if(pkgcfg_lib_${define}_${_lib})
            list(APPEND _libs "${pkgcfg_lib_${define}_${_lib}}")
          else()
            message(WARNING "ocv_check_modules(${define}): can't find library '${_lib}'. Specify 'pkgcfg_lib_${define}_${_lib}' manually")
            list(APPEND _libs "${_lib}")
          endif()
        else()
          # -pthread
          #message(WARNING "ocv_check_modules(${define}): unknown LDFLAG '${flag}'")
        endif()
      endforeach()
      set(${define}_LINK_LIBRARIES "${_libs}")
      set(${define}_LIBRARIES "${_libs}" CACHE INTERNAL "")
      unset(_lib)
      unset(_libs)
      unset(_libs_paths)
    endif()
  endif()
endmacro()



if(NOT DEFINED CMAKE_ARGC) # Guard CMake standalone invocations

# Use this option carefully, CMake's install() will install symlinks instead of real files
# It is fine for development, but should not be used by real installations
set(__symlink_default OFF)  # preprocessing is required for old CMake like 2.8.12
if(DEFINED ENV{BUILD_USE_SYMLINKS})
  set(__symlink_default $ENV{BUILD_USE_SYMLINKS})
endif()
OCV_OPTION(BUILD_USE_SYMLINKS "Use symlinks instead of files copying during build (and !!INSTALL!!)" (${__symlink_default}) IF (UNIX OR DEFINED __symlink_default))

if(CMAKE_VERSION VERSION_LESS "3.2")
  macro(ocv_cmake_byproducts var_name)
    set(${var_name}) # nothing
  endmacro()
else()
  macro(ocv_cmake_byproducts var_name)
    set(${var_name} BYPRODUCTS ${ARGN})
  endmacro()
endif()

set(OPENCV_DEPHELPER "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dephelper" CACHE INTERNAL "")
file(MAKE_DIRECTORY ${OPENCV_DEPHELPER})

if(BUILD_USE_SYMLINKS)
  set(__file0 "${CMAKE_CURRENT_LIST_FILE}")
  set(__file1 "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/symlink_test")
  if(NOT IS_SYMLINK "${__file1}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${__file0}" "${__file1}"
        RESULT_VARIABLE SYMLINK_RESULT)
    if(NOT SYMLINK_RESULT EQUAL 0)
      file(REMOVE "${__file1}")
    endif()
    if(NOT IS_SYMLINK "${__file1}")
      set(BUILD_USE_SYMLINKS 0 CACHE INTERNAL "")
    endif()
  endif()
  if(NOT BUILD_USE_SYMLINKS)
    message(STATUS "Build symlinks are not available (disabled)")
  endif()
endif()

set(OPENCV_BUILD_INFO_STR "" CACHE INTERNAL "")
function(ocv_output_status msg)
  message(STATUS "${msg}")
  string(REPLACE "\\" "\\\\" msg "${msg}")
  string(REPLACE "\"" "\\\"" msg "${msg}")
  string(REGEX REPLACE "^\n+|\n+$" "" msg "${msg}")
  if(msg MATCHES "\n")
    message(WARNING "String to be inserted to version_string.inc has an unexpected line break: '${msg}'")
    string(REPLACE "\n" "\\n" msg "${msg}")
  endif()
  set(OPENCV_BUILD_INFO_STR "${OPENCV_BUILD_INFO_STR}\"${msg}\\n\"\n" CACHE INTERNAL "")
endfunction()

macro(ocv_finalize_status)
  set(OPENCV_BUILD_INFO_FILE "${CMAKE_BINARY_DIR}/version_string.tmp")
  if(EXISTS "${OPENCV_BUILD_INFO_FILE}")
    file(READ "${OPENCV_BUILD_INFO_FILE}" __content)
  else()
    set(__content "")
  endif()
  if("${__content}" STREQUAL "${OPENCV_BUILD_INFO_STR}")
    #message(STATUS "${OPENCV_BUILD_INFO_FILE} contains the same content")
  else()
    file(WRITE "${OPENCV_BUILD_INFO_FILE}" "${OPENCV_BUILD_INFO_STR}")
  endif()
  unset(__content)
  unset(OPENCV_BUILD_INFO_STR CACHE)

  if(NOT OPENCV_SKIP_STATUS_FINALIZATION)
    if(DEFINED OPENCV_MODULE_opencv_core_BINARY_DIR)
      execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_BUILD_INFO_FILE}" "${OPENCV_MODULE_opencv_core_BINARY_DIR}/version_string.inc" OUTPUT_QUIET)
    endif()
  endif()

  if(UNIX)
    install(FILES "${OpenCV_SOURCE_DIR}/platforms/scripts/valgrind.supp"
                  "${OpenCV_SOURCE_DIR}/platforms/scripts/valgrind_3rdparty.supp"
            DESTINATION "${OPENCV_OTHER_INSTALL_PATH}" COMPONENT "dev")
  endif()
endmacro()


# Status report function.
# Automatically align right column and selects text based on condition.
# Usage:
#   status(<text>)
#   status(<heading> <value1> [<value2> ...])
#   status(<heading> <condition> THEN <text for TRUE> ELSE <text for FALSE> )
function(status text)
  set(status_cond)
  set(status_then)
  set(status_else)

  set(status_current_name "cond")
  foreach(arg ${ARGN})
    if(arg STREQUAL "THEN")
      set(status_current_name "then")
    elseif(arg STREQUAL "ELSE")
      set(status_current_name "else")
    else()
      list(APPEND status_${status_current_name} ${arg})
    endif()
  endforeach()

  if(DEFINED status_cond)
    set(status_placeholder_length 32)
    string(RANDOM LENGTH ${status_placeholder_length} ALPHABET " " status_placeholder)
    string(LENGTH "${text}" status_text_length)
    if(status_text_length LESS status_placeholder_length)
      string(SUBSTRING "${text}${status_placeholder}" 0 ${status_placeholder_length} status_text)
    elseif(DEFINED status_then OR DEFINED status_else)
      ocv_output_status("${text}")
      set(status_text "${status_placeholder}")
    else()
      set(status_text "${text}")
    endif()

    if(DEFINED status_then OR DEFINED status_else)
      if(${status_cond})
        string(REPLACE ";" " " status_then "${status_then}")
        string(REGEX REPLACE "^[ \t]+" "" status_then "${status_then}")
        ocv_output_status("${status_text} ${status_then}")
      else()
        string(REPLACE ";" " " status_else "${status_else}")
        string(REGEX REPLACE "^[ \t]+" "" status_else "${status_else}")
        ocv_output_status("${status_text} ${status_else}")
      endif()
    else()
      string(REPLACE ";" " " status_cond "${status_cond}")
      string(REGEX REPLACE "^[ \t]+" "" status_cond "${status_cond}")
      ocv_output_status("${status_text} ${status_cond}")
    endif()
  else()
    ocv_output_status("${text}")
  endif()
endfunction()

endif() # NOT DEFINED CMAKE_ARGC

#
# Generate a list of enabled features basing on conditions:
#   IF <cond> THEN <title>: check condition and append title to the result if it is true
#   ELSE <title>: return provided value instead of empty result
#   EXCLUSIVE: break after first successful condition
#
# Usage:
#   ocv_build_features_string(out [EXCLUSIVE] [IF feature THEN title] ... [ELSE title])
#
function(ocv_build_features_string out)
  set(result)
  list(REMOVE_AT ARGV 0)
  foreach(arg ${ARGV})
    if(arg STREQUAL "EXCLUSIVE")
      set(exclusive TRUE)
    elseif(arg STREQUAL "IF")
      set(then FALSE)
      set(cond)
    elseif(arg STREQUAL "THEN")
      set(then TRUE)
      set(title)
    elseif(arg STREQUAL "ELSE")
      set(then FALSE)
      set(else TRUE)
    else()
      if(then)
        if(${cond})
          list(APPEND result "${arg}")
          if(exclusive)
            break()
          endif()
        endif()
      elseif(else)
        if(NOT result)
          set(result "${arg}")
        endif()
      else()
        list(APPEND cond ${arg})
      endif()
    endif()
  endforeach()
  set(${out} ${result} PARENT_SCOPE)
endfunction()


# remove all matching elements from the list
macro(ocv_list_filterout lst regex)
  foreach(item ${${lst}})
    if(item MATCHES "${regex}")
      list(REMOVE_ITEM ${lst} "${item}")
    endif()
  endforeach()
endmacro()

# filter matching elements from the list
macro(ocv_list_filter lst regex)
  set(dst ${ARGN})
  if(NOT dst)
    set(dst ${lst})
  endif()
  set(__result ${${lst}})
  foreach(item ${__result})
    if(NOT item MATCHES "${regex}")
      list(REMOVE_ITEM __result "${item}")
    endif()
  endforeach()
  set(${dst} ${__result})
endmacro()


# stable & safe duplicates removal macro
macro(ocv_list_unique __lst)
  if(${__lst})
    list(REMOVE_DUPLICATES ${__lst})
  endif()
endmacro()


# safe list reversal macro
macro(ocv_list_reverse __lst)
  if(${__lst})
    list(REVERSE ${__lst})
  endif()
endmacro()


# safe list sorting macro
macro(ocv_list_sort __lst)
  if(${__lst})
    list(SORT ${__lst})
  endif()
endmacro()


# add prefix to each item in the list
macro(ocv_list_add_prefix LST PREFIX)
  set(__tmp "")
  foreach(item ${${LST}})
    list(APPEND __tmp "${PREFIX}${item}")
  endforeach()
  set(${LST} ${__tmp})
  unset(__tmp)
endmacro()


# add suffix to each item in the list
macro(ocv_list_add_suffix LST SUFFIX)
  set(__tmp "")
  foreach(item ${${LST}})
    list(APPEND __tmp "${item}${SUFFIX}")
  endforeach()
  set(${LST} ${__tmp})
  unset(__tmp)
endmacro()


# gets and removes the first element from the list
macro(ocv_list_pop_front LST VAR)
  if(${LST})
    list(GET ${LST} 0 ${VAR})
    list(REMOVE_AT ${LST} 0)
  else()
    set(${VAR} "")
  endif()
endmacro()

# Get list of duplicates in the list of input items.
# ocv_get_duplicates(<output list> <element> [<element> ...])
function(ocv_get_duplicates res)
  if(ARGC LESS 2)
    message(FATAL_ERROR "Invalid call to ocv_get_duplicates")
  endif()
  set(lst ${ARGN})
  list(SORT lst)
  set(prev_item)
  foreach(item ${lst})
    if(item STREQUAL prev_item)
      list(APPEND dups ${item})
    endif()
    set(prev_item ${item})
  endforeach()
  set(${res} ${dups} PARENT_SCOPE)
endfunction()

# simple regex escaping routine (does not cover all cases!!!)
macro(ocv_regex_escape var regex)
  string(REGEX REPLACE "([+.*^$])" "\\\\1" ${var} "${regex}")
endmacro()


# convert list of paths to full paths
macro(ocv_convert_to_full_paths VAR)
  if(${VAR})
    set(__tmp "")
    foreach(path ${${VAR}})
      get_filename_component(${VAR} "${path}" ABSOLUTE)
      list(APPEND __tmp "${${VAR}}")
    endforeach()
    set(${VAR} ${__tmp})
    unset(__tmp)
  endif()
endmacro()


# convert list of paths to libraries names without lib prefix
function(ocv_convert_to_lib_name var)
  set(tmp "")
  foreach(path ${ARGN})
    get_filename_component(tmp_name "${path}" NAME)
    ocv_get_libname(tmp_name "${tmp_name}")
    list(APPEND tmp "${tmp_name}")
  endforeach()
  set(${var} ${tmp} PARENT_SCOPE)
endfunction()


# add install command
function(ocv_install_target)
  if(APPLE_FRAMEWORK AND BUILD_SHARED_LIBS)
    install(TARGETS ${ARGN} FRAMEWORK DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH})
  else()
    install(TARGETS ${ARGN})
  endif()

  set(isPackage 0)
  unset(__package)
  unset(__target)
  foreach(e ${ARGN})
    if(NOT DEFINED __target)
      set(__target "${e}")
    endif()
    if(isPackage EQUAL 1)
      set(__package "${e}")
      break()
    endif()
    if(e STREQUAL "EXPORT")
      set(isPackage 1)
    endif()
  endforeach()

  if(DEFINED __package)
    list(APPEND ${__package}_TARGETS ${__target})
    set(${__package}_TARGETS "${${__package}_TARGETS}" CACHE INTERNAL "List of ${__package} targets")
  endif()

  if(MSVC)
    set(__target "${ARGV0}")

    # don't move this into global scope of this file: compiler settings (like MSVC variable) are not available during processing
    if(BUILD_SHARED_LIBS)  # no defaults for static libs (modern CMake is required)
      if(NOT CMAKE_VERSION VERSION_LESS 3.6.0)
        option(INSTALL_PDB_COMPONENT_EXCLUDE_FROM_ALL "Don't install PDB files by default" ON)
        option(INSTALL_PDB "Add install PDB rules" ON)
      elseif(NOT CMAKE_VERSION VERSION_LESS 3.1.0)
        option(INSTALL_PDB_COMPONENT_EXCLUDE_FROM_ALL "Don't install PDB files by default (not supported)" OFF)
        option(INSTALL_PDB "Add install PDB rules" OFF)
      endif()
    endif()

    if(INSTALL_PDB AND NOT INSTALL_IGNORE_PDB
        AND NOT OPENCV_${__target}_PDB_SKIP
    )
      set(__location_key "ARCHIVE")  # static libs
      get_target_property(__target_type ${__target} TYPE)
      if("${__target_type}" STREQUAL "SHARED_LIBRARY")
        set(__location_key "RUNTIME")  # shared libs (.DLL)
      endif()

      set(processDst 0)
      set(isDst 0)
      unset(__dst)
      foreach(e ${ARGN})
        if(isDst EQUAL 1)
          set(__dst "${e}")
          break()
        endif()
        if(processDst EQUAL 1 AND e STREQUAL "DESTINATION")
          set(isDst 1)
        endif()
        if(e STREQUAL "${__location_key}")
          set(processDst 1)
        else()
          set(processDst 0)
        endif()
      endforeach()

#      message(STATUS "Process ${__target} dst=${__dst}...")
      if(DEFINED __dst)
        if(NOT CMAKE_VERSION VERSION_LESS 3.1.0)
          set(__pdb_install_component "pdb")
          if(DEFINED INSTALL_PDB_COMPONENT AND INSTALL_PDB_COMPONENT)
            set(__pdb_install_component "${INSTALL_PDB_COMPONENT}")
          endif()
          set(__pdb_exclude_from_all "")
          if(INSTALL_PDB_COMPONENT_EXCLUDE_FROM_ALL)
            if(NOT CMAKE_VERSION VERSION_LESS 3.6.0)
              set(__pdb_exclude_from_all EXCLUDE_FROM_ALL)
            else()
              message(WARNING "INSTALL_PDB_COMPONENT_EXCLUDE_FROM_ALL requires CMake 3.6+")
            endif()
          endif()

#          message(STATUS "Adding PDB file installation rule: target=${__target} dst=${__dst} component=${__pdb_install_component}")
          if("${__target_type}" STREQUAL "SHARED_LIBRARY" OR "${__target_type}" STREQUAL "MODULE_LIBRARY")
            install(FILES "$<TARGET_PDB_FILE:${__target}>" DESTINATION "${__dst}"
                COMPONENT ${__pdb_install_component} OPTIONAL ${__pdb_exclude_from_all})
          else()
            # There is no generator expression similar to TARGET_PDB_FILE and TARGET_PDB_FILE can't be used: https://gitlab.kitware.com/cmake/cmake/issues/16932
            # However we still want .pdb files like: 'lib/Debug/opencv_core341d.pdb' or '3rdparty/lib/zlibd.pdb'
            install(FILES "$<TARGET_PROPERTY:${__target},ARCHIVE_OUTPUT_DIRECTORY>/$<CONFIG>/$<IF:$<BOOL:$<TARGET_PROPERTY:${__target},COMPILE_PDB_NAME_DEBUG>>,$<TARGET_PROPERTY:${__target},COMPILE_PDB_NAME_DEBUG>,$<TARGET_PROPERTY:${__target},COMPILE_PDB_NAME>>.pdb"
                DESTINATION "${__dst}" CONFIGURATIONS Debug
                COMPONENT ${__pdb_install_component} OPTIONAL ${__pdb_exclude_from_all})
            install(FILES "$<TARGET_PROPERTY:${__target},ARCHIVE_OUTPUT_DIRECTORY>/$<CONFIG>/$<IF:$<BOOL:$<TARGET_PROPERTY:${__target},COMPILE_PDB_NAME_RELEASE>>,$<TARGET_PROPERTY:${__target},COMPILE_PDB_NAME_RELEASE>,$<TARGET_PROPERTY:${__target},COMPILE_PDB_NAME>>.pdb"
                DESTINATION "${__dst}" CONFIGURATIONS Release
                COMPONENT ${__pdb_install_component} OPTIONAL ${__pdb_exclude_from_all})
          endif()
        else()
          message(WARNING "PDB files installation is not supported (need CMake >= 3.1.0)")
        endif()
      endif()
    endif()
  endif()
endfunction()

# ocv_install_3rdparty_licenses(<library-name> <filename1> [<filename2> ..])
function(ocv_install_3rdparty_licenses library)
  foreach(filename ${ARGN})
    set(filepath "${filename}")
    if(NOT IS_ABSOLUTE "${filepath}")
      set(filepath "${CMAKE_CURRENT_LIST_DIR}/${filepath}")
    endif()
    get_filename_component(name "${filename}" NAME)
    install(
      FILES "${filepath}"
      DESTINATION "${OPENCV_LICENSES_INSTALL_PATH}"
      COMPONENT licenses
      RENAME "${library}-${name}"
    )
  endforeach()
endfunction()

# read set of version defines from the header file
macro(ocv_parse_header FILENAME FILE_VAR)
  set(vars_regex "")
  set(__parnet_scope OFF)
  set(__add_cache OFF)
  foreach(name ${ARGN})
    if(${name} STREQUAL "PARENT_SCOPE")
      set(__parnet_scope ON)
    elseif(${name} STREQUAL "CACHE")
      set(__add_cache ON)
    elseif(vars_regex)
      set(vars_regex "${vars_regex}|${name}")
    else()
      set(vars_regex "${name}")
    endif()
  endforeach()
  if(EXISTS "${FILENAME}")
    file(STRINGS "${FILENAME}" ${FILE_VAR} REGEX "#define[ \t]+(${vars_regex})[ \t]+[0-9]+" )
  else()
    unset(${FILE_VAR})
  endif()
  foreach(name ${ARGN})
    if(NOT ${name} STREQUAL "PARENT_SCOPE" AND NOT ${name} STREQUAL "CACHE")
      if(${FILE_VAR})
        if(${FILE_VAR} MATCHES ".+[ \t]${name}[ \t]+([0-9]+).*")
          string(REGEX REPLACE ".+[ \t]${name}[ \t]+([0-9]+).*" "\\1" ${name} "${${FILE_VAR}}")
        else()
          set(${name} "")
        endif()
        if(__add_cache)
          set(${name} ${${name}} CACHE INTERNAL "${name} parsed from ${FILENAME}" FORCE)
        elseif(__parnet_scope)
          set(${name} "${${name}}" PARENT_SCOPE)
        endif()
      else()
        unset(${name} CACHE)
      endif()
    endif()
  endforeach()
endmacro()

# read single version define from the header file
macro(ocv_parse_header2 LIBNAME HDR_PATH VARNAME)
  ocv_clear_vars(${LIBNAME}_VERSION_MAJOR
                 ${LIBNAME}_VERSION_MAJOR
                 ${LIBNAME}_VERSION_MINOR
                 ${LIBNAME}_VERSION_PATCH
                 ${LIBNAME}_VERSION_TWEAK
                 ${LIBNAME}_VERSION_STRING)
  set(${LIBNAME}_H "")
  if(EXISTS "${HDR_PATH}")
    file(STRINGS "${HDR_PATH}" ${LIBNAME}_H REGEX "^#define[ \t]+${VARNAME}[ \t]+\"[^\"]*\".*$" LIMIT_COUNT 1)
  endif()

  if(${LIBNAME}_H)
    string(REGEX REPLACE "^.*[ \t]${VARNAME}[ \t]+\"([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MAJOR "${${LIBNAME}_H}")
    string(REGEX REPLACE "^.*[ \t]${VARNAME}[ \t]+\"[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MINOR  "${${LIBNAME}_H}")
    string(REGEX REPLACE "^.*[ \t]${VARNAME}[ \t]+\"[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_PATCH "${${LIBNAME}_H}")
    set(${LIBNAME}_VERSION_MAJOR ${${LIBNAME}_VERSION_MAJOR} ${ARGN})
    set(${LIBNAME}_VERSION_MINOR ${${LIBNAME}_VERSION_MINOR} ${ARGN})
    set(${LIBNAME}_VERSION_PATCH ${${LIBNAME}_VERSION_PATCH} ${ARGN})
    set(${LIBNAME}_VERSION_STRING "${${LIBNAME}_VERSION_MAJOR}.${${LIBNAME}_VERSION_MINOR}.${${LIBNAME}_VERSION_PATCH}")

    # append a TWEAK version if it exists:
    set(${LIBNAME}_VERSION_TWEAK "")
    if("${${LIBNAME}_H}" MATCHES "^.*[ \t]${VARNAME}[ \t]+\"[0-9]+\\.[0-9]+\\.[0-9]+\\.([0-9]+).*$")
      set(${LIBNAME}_VERSION_TWEAK "${CMAKE_MATCH_1}" ${ARGN})
    endif()
    if(${LIBNAME}_VERSION_TWEAK)
      set(${LIBNAME}_VERSION_STRING "${${LIBNAME}_VERSION_STRING}.${${LIBNAME}_VERSION_TWEAK}" ${ARGN})
    else()
      set(${LIBNAME}_VERSION_STRING "${${LIBNAME}_VERSION_STRING}" ${ARGN})
    endif()
  endif()
endmacro()

# TODO remove this
# read single version info from the pkg file
macro(ocv_parse_pkg ver_varname LIBNAME PKG_PATH)
  if(EXISTS "${PKG_PATH}/${LIBNAME}.pc")
    file(STRINGS "${PKG_PATH}/${LIBNAME}.pc" line_to_parse REGEX "^Version:[ \t]+[0-9.]*.*$" LIMIT_COUNT 1)
    STRING(REGEX REPLACE ".*Version: ([^ ]+).*" "\\1" ${ver_varname} "${line_to_parse}" )
  endif()
endmacro()

################################################################################################
# short command to setup source group
function(ocv_source_group group)
  if(BUILD_opencv_world AND OPENCV_MODULE_${the_module}_IS_PART_OF_WORLD)
    set(group "${the_module}\\${group}")
  endif()
  cmake_parse_arguments(SG "" "DIRBASE" "GLOB;GLOB_RECURSE;FILES" ${ARGN})
  set(files "")
  if(SG_FILES)
    list(APPEND files ${SG_FILES})
  endif()
  if(SG_GLOB)
    file(GLOB srcs ${SG_GLOB})
    list(APPEND files ${srcs})
  endif()
  if(SG_GLOB_RECURSE)
    file(GLOB_RECURSE srcs ${SG_GLOB_RECURSE})
    list(APPEND files ${srcs})
  endif()
  if(SG_DIRBASE)
    foreach(f ${files})
      file(RELATIVE_PATH fpart "${SG_DIRBASE}" "${f}")
      if(fpart MATCHES "^\\.\\.")
        message(AUTHOR_WARNING "Can't detect subpath for source_group command: Group=${group} FILE=${f} DIRBASE=${SG_DIRBASE}")
        set(fpart "")
      else()
        get_filename_component(fpart "${fpart}" PATH)
        if(fpart)
          set(fpart "/${fpart}") # add '/'
          string(REPLACE "/" "\\" fpart "${fpart}")
        endif()
      endif()
      source_group("${group}${fpart}" FILES ${f})
    endforeach()
  else()
    source_group(${group} FILES ${files})
  endif()
endfunction()

macro(__ocv_push_target_link_libraries)
  if(NOT TARGET ${target})
    if(NOT DEFINED OPENCV_MODULE_${target}_LOCATION)
      message(FATAL_ERROR "ocv_target_link_libraries: invalid target: '${target}'")
    endif()
    set(OPENCV_MODULE_${target}_LINK_DEPS ${OPENCV_MODULE_${target}_LINK_DEPS} ${ARGN} CACHE INTERNAL "" FORCE)
  else()
    target_link_libraries(${target} ${ARGN})
  endif()
endmacro()

function(ocv_target_link_libraries target)
  set(LINK_DEPS ${ARGN})
  _ocv_fix_target(target)
  set(LINK_MODE "PRIVATE")
  set(LINK_PENDING "")
  foreach(dep ${LINK_DEPS})
    if(" ${dep}" STREQUAL " ${target}")
      # prevent "link to itself" warning (world problem)
    elseif(" ${dep}" STREQUAL " LINK_PRIVATE" OR " ${dep}" STREQUAL " LINK_PUBLIC"  # deprecated
        OR " ${dep}" STREQUAL " PRIVATE" OR " ${dep}" STREQUAL " PUBLIC" OR " ${dep}" STREQUAL " INTERFACE"
    )
      if(NOT LINK_PENDING STREQUAL "")
        __ocv_push_target_link_libraries(${LINK_MODE} ${LINK_PENDING})
        set(LINK_PENDING "")
      endif()
      set(LINK_MODE "${dep}")
    else()
      if(BUILD_opencv_world)
        if(OPENCV_MODULE_${dep}_IS_PART_OF_WORLD)
          set(dep opencv_world)
        endif()
      endif()
      list(APPEND LINK_PENDING "${dep}")
    endif()
  endforeach()
  if(NOT LINK_PENDING STREQUAL "")
    __ocv_push_target_link_libraries(${LINK_MODE} ${LINK_PENDING})
  endif()
endfunction()

function(ocv_target_compile_definitions target)
  _ocv_fix_target(target)
  if(NOT TARGET ${target})
    if(NOT DEFINED OPENCV_MODULE_${target}_LOCATION)
      message(FATAL_ERROR "ocv_target_compile_definitions: invalid target: '${target}'")
    endif()
    set(OPENCV_MODULE_${target}_COMPILE_DEFINITIONS ${OPENCV_MODULE_${target}_COMPILE_DEFINITIONS} ${ARGN} CACHE INTERNAL "" FORCE)
  else()
    target_compile_definitions(${target} ${ARGN})
  endif()
endfunction()


function(_ocv_append_target_includes target)
  if(DEFINED OCV_TARGET_INCLUDE_DIRS_${target})
    target_include_directories(${target} PRIVATE ${OCV_TARGET_INCLUDE_DIRS_${target}})
    if(OPENCV_DEPENDANT_TARGETS_${target})
      foreach(t ${OPENCV_DEPENDANT_TARGETS_${target}})
        target_include_directories(${t} PRIVATE ${OCV_TARGET_INCLUDE_DIRS_${target}})
      endforeach()
    endif()
    unset(OCV_TARGET_INCLUDE_DIRS_${target} CACHE)
  endif()

  if(DEFINED OCV_TARGET_INCLUDE_SYSTEM_DIRS_${target})
    target_include_directories(${target} SYSTEM PRIVATE ${OCV_TARGET_INCLUDE_SYSTEM_DIRS_${target}})
    if(OPENCV_DEPENDANT_TARGETS_${target})
      foreach(t ${OPENCV_DEPENDANT_TARGETS_${target}})
        target_include_directories(${t} SYSTEM PRIVATE ${OCV_TARGET_INCLUDE_SYSTEM_DIRS_${target}})
      endforeach()
    endif()
    unset(OCV_TARGET_INCLUDE_SYSTEM_DIRS_${target} CACHE)
  endif()
endfunction()

function(ocv_add_executable target)
  add_executable(${target} ${ARGN})
  _ocv_append_target_includes(${target})
endfunction()

function(ocv_add_library target)
  if(HAVE_CUDA AND ARGN MATCHES "\\.cu")
    ocv_include_directories(${CUDA_INCLUDE_DIRS})
    ocv_cuda_compile(cuda_objs ${ARGN})
    set(OPENCV_MODULE_${target}_CUDA_OBJECTS ${cuda_objs} CACHE INTERNAL "Compiled CUDA object files")
  endif()

  add_library(${target} ${ARGN} ${cuda_objs})

  if(APPLE_FRAMEWORK AND BUILD_SHARED_LIBS)
    message(STATUS "Setting Apple target properties for ${target}")

    set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG 1)

    set_target_properties(${target} PROPERTIES
      FRAMEWORK TRUE
      MACOSX_FRAMEWORK_IDENTIFIER org.opencv
      MACOSX_FRAMEWORK_INFO_PLIST ${CMAKE_BINARY_DIR}/ios/Info.plist
      # "current version" in semantic format in Mach-O binary file
      VERSION ${OPENCV_LIBVERSION}
      # "compatibility version" in semantic format in Mach-O binary file
      SOVERSION ${OPENCV_LIBVERSION}
      INSTALL_RPATH ""
      INSTALL_NAME_DIR "@rpath"
      BUILD_WITH_INSTALL_RPATH 1
      LIBRARY_OUTPUT_NAME "opencv2"
      XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
      #PUBLIC_HEADER "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/cvconfig.h"
      #XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "iPhone Developer"
    )
  endif()

  _ocv_append_target_includes(${target})
endfunction()


macro(ocv_get_libname var_name)
  get_filename_component(__libname "${ARGN}" NAME)
  # libopencv_core.so.3.3 -> opencv_core
  string(REGEX REPLACE "^lib(.+)\\.(a|so|dll)(\\.[.0-9]+)?$" "\\1" __libname "${__libname}")
  # MacOSX: libopencv_core.3.3.1.dylib -> opencv_core
  string(REGEX REPLACE "^lib(.+[^.0-9])\\.([.0-9]+\\.)?dylib$" "\\1" __libname "${__libname}")
  set(${var_name} "${__libname}")
endmacro()

# build the list of opencv libs and dependencies for all modules
#  _modules - variable to hold list of all modules
#  _extra - variable to hold list of extra dependencies
#  _3rdparty - variable to hold list of prebuilt 3rdparty libraries
macro(ocv_get_all_libs _modules _extra _3rdparty)
  set(${_modules} "")
  set(${_extra} "")
  set(${_3rdparty} "")
  foreach(m ${OPENCV_MODULES_PUBLIC})
    if(TARGET ${m})
      get_target_property(deps ${m} INTERFACE_LINK_LIBRARIES)
      if(NOT deps)
        set(deps "")
      endif()
    else()
      set(deps "")
    endif()
    set(_rev_deps "${deps};${m}")
    ocv_list_reverse(_rev_deps)
    foreach (dep ${_rev_deps})
      if(DEFINED OPENCV_MODULE_${dep}_LOCATION)
        list(INSERT ${_modules} 0 ${dep})
      endif()
    endforeach()
    foreach (dep ${deps} ${OPENCV_LINKER_LIBS})
      if (NOT DEFINED OPENCV_MODULE_${dep}_LOCATION)
        if(dep MATCHES "^\\$<LINK_ONLY:([^>]+)>$")
          set(dep "${CMAKE_MATCH_1}")
        endif()
        if(dep MATCHES "^\\$<")
          message(WARNING "Unexpected CMake generator expression: ${dep}")
        endif()
        if (TARGET ${dep})
          get_target_property(_type ${dep} TYPE)
          if((_type STREQUAL "STATIC_LIBRARY" AND BUILD_SHARED_LIBS)
              OR _type STREQUAL "INTERFACE_LIBRARY"
              OR DEFINED OPENCV_MODULE_${dep}_LOCATION  # OpenCV modules
          )
            # nothing
          else()
            get_target_property(_output ${dep} IMPORTED_LOCATION)
            if(NOT _output)
              get_target_property(_output ${dep} ARCHIVE_OUTPUT_DIRECTORY)
              get_target_property(_output_name ${dep} OUTPUT_NAME)
              if(NOT _output_name)
                set(_output_name "${dep}")
              endif()
            else()
              get_filename_component(_output_name "${_output}" NAME)
            endif()
            string(FIND "${_output}" "${CMAKE_BINARY_DIR}" _POS)
            if (_POS EQUAL 0)
              ocv_get_libname(_libname "${_output_name}")
              list(INSERT ${_3rdparty} 0 ${dep})
            else()
              if(_output)
                list(INSERT ${_extra} 0 ${_output})
              else()
                list(INSERT ${_extra} 0 ${dep})
              endif()
            endif()
          endif()
        else()
          list(INSERT ${_extra} 0 ${dep})
        endif()
      endif()
    endforeach()
  endforeach()

  ocv_list_filterout(${_modules} "^[\$]<")
  ocv_list_filterout(${_3rdparty} "^[\$]<")
  ocv_list_filterout(${_extra} "^[\$]<")

  # convert CMake lists to makefile literals
  foreach(lst ${_modules} ${_3rdparty} ${_extra})
    ocv_list_unique(${lst})
    ocv_list_reverse(${lst})
  endforeach()
endmacro()


function(ocv_add_test_from_target test_name test_kind the_target)
  if(CMAKE_VERSION VERSION_GREATER "2.8" AND NOT CMAKE_CROSSCOMPILING)
    if(NOT "${test_kind}" MATCHES "^(Accuracy|Performance|Sanity)$")
      message(FATAL_ERROR "Unknown test kind : ${test_kind}")
    endif()
    if(NOT TARGET "${the_target}")
      message(FATAL_ERROR "${the_target} is not a CMake target")
    endif()

    string(TOLOWER "${test_kind}" test_kind_lower)
    set(test_report_dir "${CMAKE_BINARY_DIR}/test-reports/${test_kind_lower}")
    file(MAKE_DIRECTORY "${test_report_dir}")

    add_test(NAME "${test_name}"
      COMMAND "${the_target}"
              "--gtest_output=xml:${the_target}.xml"
              ${ARGN})

    set_tests_properties("${test_name}" PROPERTIES
      LABELS "${OPENCV_MODULE_${the_module}_LABEL};${test_kind}"
      WORKING_DIRECTORY "${test_report_dir}")

    if(OPENCV_TEST_DATA_PATH)
      set_tests_properties("${test_name}" PROPERTIES
        ENVIRONMENT "OPENCV_TEST_DATA_PATH=${OPENCV_TEST_DATA_PATH}")
    endif()
  endif()
endfunction()

macro(ocv_add_testdata basedir dest_subdir)
  if(BUILD_TESTS)
    if(NOT CMAKE_CROSSCOMPILING AND NOT INSTALL_TESTS)
      file(COPY ${basedir}/
           DESTINATION ${CMAKE_BINARY_DIR}/${OPENCV_TEST_DATA_INSTALL_PATH}/${dest_subdir}
           ${ARGN}
      )
    endif()
    if(INSTALL_TESTS)
      install(DIRECTORY ${basedir}/
              DESTINATION ${OPENCV_TEST_DATA_INSTALL_PATH}/${dest_subdir}
              COMPONENT "tests"
              ${ARGN}
      )
    endif()
  endif()
endmacro()

macro(ocv_generate_vs_version_file DESTINATION)
  cmake_parse_arguments(VS_VER "" "NAME;FILEDESCRIPTION;FILEVERSION;INTERNALNAME;COPYRIGHT;ORIGINALFILENAME;PRODUCTNAME;PRODUCTVERSION;COMMENTS;FILEVERSION_QUAD;PRODUCTVERSION_QUAD" "" ${ARGN})

  macro(__vs_ver_update_variable name)
    if(VS_VER_NAME AND DEFINED OPENCV_${VS_VER_NAME}_VS_VER_${name})
      set(OPENCV_VS_VER_${name} "${OPENCV_${VS_VER_NAME}_VS_VER_${name}}")
    elseif(VS_VER_${name})
      set(OPENCV_VS_VER_${name} "${VS_VER_${name}}")
    endif()
  endmacro()

  __vs_ver_update_variable(FILEVERSION_QUAD)
  __vs_ver_update_variable(PRODUCTVERSION_QUAD)

  macro(__vs_ver_update_str_variable name)
    if(VS_VER_NAME AND DEFINED OPENCV_${VS_VER_NAME}_VS_VER_${name})
      set(OPENCV_VS_VER_${name}_STR "${OPENCV_${VS_VER_NAME}_VS_VER_${name}}")
    elseif(VS_VER_${name})
      set(OPENCV_VS_VER_${name}_STR "${VS_VER_${name}}")
    endif()
  endmacro()

  __vs_ver_update_str_variable(FILEDESCRIPTION)
  __vs_ver_update_str_variable(FILEVERSION)
  __vs_ver_update_str_variable(INTERNALNAME)
  __vs_ver_update_str_variable(COPYRIGHT)
  __vs_ver_update_str_variable(ORIGINALFILENAME)
  __vs_ver_update_str_variable(PRODUCTNAME)
  __vs_ver_update_str_variable(PRODUCTVERSION)
  __vs_ver_update_str_variable(COMMENTS)

  if(OPENCV_VS_VER_COPYRIGHT_STR)
    set(OPENCV_VS_VER_HAVE_COPYRIGHT_STR 1)
  else()
    set(OPENCV_VS_VER_HAVE_COPYRIGHT_STR 0)
  endif()

  if(OPENCV_VS_VER_COMMENTS_STR)
    set(OPENCV_VS_VER_HAVE_COMMENTS_STR 1)
  else()
    set(OPENCV_VS_VER_HAVE_COMMENTS_STR 0)
  endif()

  configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/vs_version.rc.in" "${DESTINATION}" @ONLY)
endmacro()

macro(ocv_cmake_script_append_var content_var)
  foreach(var_name ${ARGN})
    set(${content_var} "${${content_var}}
set(${var_name} \"${${var_name}}\")
")
  endforeach()
endmacro()

macro(ocv_copyfiles_append_file list_var src dst)
  list(LENGTH ${list_var} __id)
  list(APPEND ${list_var} ${__id})
  set(${list_var}_SRC_${__id} "${src}")
  set(${list_var}_DST_${__id} "${dst}")
endmacro()

macro(ocv_copyfiles_append_dir list_var src dst)
  set(__glob ${ARGN})
  list(LENGTH ${list_var} __id)
  list(APPEND ${list_var} ${__id})
  set(${list_var}_SRC_${__id} "${src}")
  set(${list_var}_DST_${__id} "${dst}")
  set(${list_var}_MODE_${__id} "COPYDIR")
  if(__glob)
    set(${list_var}_GLOB_${__id} ${__glob})
  endif()
endmacro()

macro(ocv_copyfiles_make_config_string content_var list_var)
  set(var_name "${list_var}")
  set(${content_var} "${${content_var}}
set(${var_name} \"${${var_name}}\")
")
  foreach(__id ${${list_var}})
    set(${content_var} "${${content_var}}
set(${list_var}_SRC_${__id} \"${${list_var}_SRC_${__id}}\")
set(${list_var}_DST_${__id} \"${${list_var}_DST_${__id}}\")
")
    if(DEFINED ${list_var}_MODE_${__id})
      set(${content_var} "${${content_var}}set(${list_var}_MODE_${__id} \"${${list_var}_MODE_${__id}}\")\n")
    endif()
    if(DEFINED ${list_var}_GLOB_${__id})
      set(${content_var} "${${content_var}}set(${list_var}_GLOB_${__id} \"${${list_var}_GLOB_${__id}}\")\n")
    endif()
  endforeach()
endmacro()

macro(ocv_copyfiles_make_config_file filename_var list_var)
  ocv_copyfiles_make_config_string(${list_var}_CONFIG ${list_var})
  set(${filename_var} "${CMAKE_CURRENT_BINARY_DIR}/copyfiles-${list_var}.cmake")
  file(WRITE "${${filename_var}}" "${${list_var}_CONFIG}")
endmacro()

macro(ocv_copyfiles_add_forced_target target list_var comment_str)
  ocv_copyfiles_make_config_file(CONFIG_FILE ${list_var})
  ocv_cmake_byproducts(__byproducts BYPRODUCTS "${OPENCV_DEPHELPER}/${target}")
  add_custom_target(${target}
      ${__byproducts}  # required for add_custom_target() by ninja
      COMMAND ${CMAKE_COMMAND}
        "-DCONFIG_FILE:PATH=${CONFIG_FILE}"
        "-DCOPYLIST_VAR:STRING=${list_var}"
        "-DDEPHELPER=${OPENCV_DEPHELPER}/${target}"
        -P "${OpenCV_SOURCE_DIR}/cmake/copy_files.cmake"
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "${comment_str}"
      DEPENDS "${OpenCV_SOURCE_DIR}/cmake/copy_files.cmake"
              # ninja warn about file(WRITE): "${SRC_COPY_CONFIG_FILE}"
  )
endmacro()

macro(ocv_copyfiles_add_target target list_var comment_str)
  set(deps ${ARGN})
  ocv_copyfiles_make_config_file(CONFIG_FILE ${list_var})
  add_custom_command(OUTPUT "${OPENCV_DEPHELPER}/${target}"
      COMMAND ${CMAKE_COMMAND}
        "-DCONFIG_FILE:PATH=${CONFIG_FILE}"
        "-DCOPYLIST_VAR:STRING=${list_var}"
        "-DDEPHELPER=${OPENCV_DEPHELPER}/${target}"
        -P "${OpenCV_SOURCE_DIR}/cmake/copy_files.cmake"
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "${comment_str}"
      DEPENDS "${OpenCV_SOURCE_DIR}/cmake/copy_files.cmake" ${deps}
              # ninja warn about file(WRITE): "${SRC_COPY_CONFIG_FILE}"
  )
  add_custom_target(${target} DEPENDS "${OPENCV_DEPHELPER}/${target}")
endmacro()

macro(ocv_get_smart_file_name output_var fpath)
  ocv_is_subdir(__subir "${OpenCV_BINARY_DIR}" "${fpath}")
  if(__subir)
    file(RELATIVE_PATH ${output_var} "${OpenCV_BINARY_DIR}" "${fpath}")
    set(${output_var} "<BUILD>/${${output_var}}")
  else()
    ocv_is_subdir(__subir "${OpenCV_SOURCE_DIR}" "${fpath}")
    if(__subir)
      file(RELATIVE_PATH ${output_var} "${OpenCV_SOURCE_DIR}" "${fpath}")
    else()
      set(${output_var} "${fpath}")
    endif()
  endif()
  unset(__subir)
endmacro()

# Needed by install(DIRECTORY ...)
if(NOT CMAKE_VERSION VERSION_LESS 3.1)
  set(compatible_MESSAGE_NEVER MESSAGE_NEVER)
else()
  set(compatible_MESSAGE_NEVER "")
endif()


macro(ocv_git_describe var_name path)
  if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" describe --tags --exact-match --dirty
      WORKING_DIRECTORY "${path}"
      OUTPUT_VARIABLE ${var_name}
      RESULT_VARIABLE GIT_RESULT
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT GIT_RESULT EQUAL 0)
      execute_process(COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty --match "[0-9].[0-9].[0-9]*" --exclude "[^-]*-cvsdk"
        WORKING_DIRECTORY "${path}"
        OUTPUT_VARIABLE ${var_name}
        RESULT_VARIABLE GIT_RESULT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      if(NOT GIT_RESULT EQUAL 0)  # --exclude is not supported by 'git'
        # match only tags with complete OpenCV versions (ignores -alpha/-beta/-rc suffixes)
        execute_process(COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty --match "[0-9].[0-9]*[0-9]"
          WORKING_DIRECTORY "${path}"
          OUTPUT_VARIABLE ${var_name}
          RESULT_VARIABLE GIT_RESULT
          ERROR_QUIET
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT GIT_RESULT EQUAL 0)
          set(${var_name} "unknown")
        endif()
      endif()
    endif()
  else()
    set(${var_name} "unknown")
  endif()
endmacro()


# ocv_update_file(filepath content [VERBOSE])
# - write content to file
# - will not change modification time in case when file already exists and content has not changed
function(ocv_update_file filepath content)
  if(EXISTS "${filepath}")
    file(READ "${filepath}" actual_content)
  else()
    set(actual_content "")
  endif()
  if("${actual_content}" STREQUAL "${content}")
    if(";${ARGN};" MATCHES ";VERBOSE;")
      message(STATUS "${filepath} contains the same content")
    endif()
  else()
    file(WRITE "${filepath}" "${content}")
  endif()
endfunction()

if(NOT BUILD_SHARED_LIBS AND (CMAKE_VERSION VERSION_LESS "3.14.0"))
  ocv_update(OPENCV_3RDPARTY_EXCLUDE_FROM_ALL "")  # avoid CMake warnings: https://gitlab.kitware.com/cmake/cmake/-/issues/18938
else()
  ocv_update(OPENCV_3RDPARTY_EXCLUDE_FROM_ALL "EXCLUDE_FROM_ALL")
endif()
