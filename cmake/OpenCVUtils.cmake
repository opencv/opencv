include(CMakeParseArguments)

# Debugging function
function(ocv_cmake_dump_vars)
  get_cmake_property(__variableNames VARIABLES)
  cmake_parse_arguments(DUMP "" "TOFILE" "" ${ARGN})
  set(regex "${DUMP_UNPARSED_ARGUMENTS}")
  string(TOLOWER "${regex}" regex_lower)
  set(__VARS "")
  foreach(__variableName ${__variableNames})
    string(TOLOWER "${__variableName}" __variableName_lower)
    if((__variableName MATCHES "${regex}" OR __variableName_lower MATCHES "${regex_lower}")
        AND NOT __variableName_lower MATCHES "^__")
      set(__VARS "${__VARS}${__variableName}=${${__variableName}}\n")
    endif()
  endforeach()
  if(DUMP_TOFILE)
    file(WRITE ${CMAKE_BINARY_DIR}/${DUMP_TOFILE} "${__VARS}")
  else()
    message(AUTHOR_WARNING "${__VARS}")
  endif()
endfunction()

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
  configure_file(${file_name} "${CMAKE_BINARY_DIR}/CMakeConfig-${var_name}.cmake" ${ARGN})
  file(READ "${CMAKE_BINARY_DIR}/CMakeConfig-${var_name}.cmake" ${var_name})
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
#  string(REPLACE ";" " " __msg "${ARGN}")
#  message(STATUS "${__msg}")
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
  string(REGEX REPLACE "([/\\]?)[\\.][/\\]" "\\1" ${result_var} "${${result_var}}")
  if("${${result_var}}" STREQUAL "")
    set(${result_var} ".")
  endif()
  #message(STATUS "'${P1}' '${P2_}' => '${${result_var}}'")
endmacro()

# rename modules target to world if needed
macro(_ocv_fix_target target_var)
  if(BUILD_opencv_world)
    if(OPENCV_MODULE_${${target_var}}_IS_PART_OF_WORLD)
      set(${target_var} opencv_world)
    endif()
  endif()
endmacro()

function(ocv_is_opencv_directory result_var dir)
  get_filename_component(__abs_dir "${dir}" ABSOLUTE)
  if("${__abs_dir}" MATCHES "^${OpenCV_SOURCE_DIR}"
      OR "${__abs_dir}" MATCHES "^${OpenCV_BINARY_DIR}"
      OR (OPENCV_EXTRA_MODULES_PATH AND "${__abs_dir}" MATCHES "^${OPENCV_EXTRA_MODULES_PATH}"))
    set(${result_var} 1 PARENT_SCOPE)
  else()
    set(${result_var} 0 PARENT_SCOPE)
  endif()
endfunction()


# adds include directories in such a way that directories from the OpenCV source tree go first
function(ocv_include_directories)
  ocv_debug_message("ocv_include_directories( ${ARGN} )")
  set(__add_before "")
  foreach(dir ${ARGN})
    ocv_is_opencv_directory(__is_opencv_dir "${dir}")
    if(__is_opencv_dir)
      list(APPEND __add_before "${dir}")
    elseif(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0" AND
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

function(ocv_append_dependant_targets target)
  #ocv_debug_message("ocv_append_dependant_targets(${target} ${ARGN})")
  _ocv_fix_target(target)
  set(OPENCV_DEPENDANT_TARGETS_${target} "${OPENCV_DEPENDANT_TARGETS_${target}};${ARGN}" CACHE INTERNAL "" FORCE)
endfunction()

# adds include directories in such a way that directories from the OpenCV source tree go first
function(ocv_target_include_directories target)
  #ocv_debug_message("ocv_target_include_directories(${target} ${ARGN})")
  _ocv_fix_target(target)
  set(__params "")
  if(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0" AND
      ";${ARGN};" MATCHES "/usr/include;")
    return() # workaround for GCC 6.x bug
  endif()
  foreach(dir ${ARGN})
    get_filename_component(__abs_dir "${dir}" ABSOLUTE)
    ocv_is_opencv_directory(__is_opencv_dir "${dir}")
    if(__is_opencv_dir)
      list(APPEND __params "${__abs_dir}")
    else()
      list(APPEND __params "${dir}")
    endif()
  endforeach()
  if(HAVE_CUDA OR CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories(${__params})
  else()
    if(TARGET ${target})
      target_include_directories(${target} PRIVATE ${__params})
      if(OPENCV_DEPENDANT_TARGETS_${target})
        foreach(t ${OPENCV_DEPENDANT_TARGETS_${target}})
          target_include_directories(${t} PRIVATE ${__params})
        endforeach()
      endif()
    else()
      set(__new_inc "${OCV_TARGET_INCLUDE_DIRS_${target}};${__params}")
      set(OCV_TARGET_INCLUDE_DIRS_${target} "${__new_inc}" CACHE INTERNAL "")
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

set(OCV_COMPILER_FAIL_REGEX
    "command line option .* is valid for .* but not for C\\+\\+" # GNU
    "command line option .* is valid for .* but not for C" # GNU
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
      MESSAGE(STATUS "Performing Test ${RESULT}${__msg}")
      TRY_COMPILE(${RESULT}
        "${CMAKE_BINARY_DIR}"
        "${_fname}"
        CMAKE_FLAGS "-DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"   # CMP0056 do this on new CMake
        COMPILE_DEFINITIONS "${FLAG}"
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
    set(_lang OBJCXX)
  else()
    set(_lang ${lang})
  endif()

  string(TOUPPER "${flag}" ${varname})
  string(REGEX REPLACE "^(/|-)" "HAVE_${_lang}_" ${varname} "${${varname}}")
  string(REGEX REPLACE " -|-|=| |\\." "_" ${varname} "${${varname}}")

  ocv_check_compiler_flag("${_lang}" "${base_options} ${flag}" ${${varname}} ${ARGN})
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
    elseif((CMAKE_COMPILER_IS_GNUCXX OR (UNIX AND CV_ICC)) AND _gxx_warnings AND _flag_vars)
      foreach(var ${_flag_vars})
        foreach(warning ${_gxx_warnings})
          if(NOT warning MATCHES "^-Wno-")
            string(REPLACE "${warning}" "" ${var} "${${var}}")
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
#   option(<option_variable> "help string describing the option" <initial value or boolean expression> [IF <condition>])
macro(OCV_OPTION variable description value)
  set(__value ${value})
  set(__condition "")
  set(__varname "__value")
  foreach(arg ${ARGN})
    if(arg STREQUAL "IF" OR arg STREQUAL "if")
      set(__varname "__condition")
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
    unset(${variable} CACHE)
  endif()
  unset(__condition)
  unset(__value)
endmacro()

# Usage: ocv_append_build_options(HIGHGUI FFMPEG)
macro(ocv_append_build_options var_prefix pkg_prefix)
  foreach(suffix INCLUDE_DIRS LIBRARIES LIBRARY_DIRS)
    if(${pkg_prefix}_${suffix})
      list(APPEND ${var_prefix}_${suffix} ${${pkg_prefix}_${suffix}})
      list(REMOVE_DUPLICATES ${var_prefix}_${suffix})
    endif()
  endforeach()
endmacro()

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
  pkg_check_modules(${define} ${ARGN})
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
endmacro()


# Macro that checks if module has been installed.
# After it adds module to build and define
# constants passed as second arg
macro(CHECK_MODULE module_name define cv_module)
  set(${define} 0)
  if(PKG_CONFIG_FOUND)
    set(ALIAS               ALIASOF_${module_name})
    set(ALIAS_FOUND                 ${ALIAS}_FOUND)
    set(ALIAS_INCLUDE_DIRS   ${ALIAS}_INCLUDE_DIRS)
    set(ALIAS_LIBRARY_DIRS   ${ALIAS}_LIBRARY_DIRS)
    set(ALIAS_LIBRARIES         ${ALIAS}_LIBRARIES)

    PKG_CHECK_MODULES(${ALIAS} ${module_name})
    if(${ALIAS_FOUND})
      set(${define} 1)
      ocv_append_build_options(${cv_module} ${ALIAS})
    endif()
  endif()
endmacro()

if(NOT DEFINED CMAKE_ARGC) # Guard CMake standalone invocations

set(OPENCV_BUILD_INFO_FILE "${CMAKE_BINARY_DIR}/version_string.tmp")
file(REMOVE "${OPENCV_BUILD_INFO_FILE}")
function(ocv_output_status msg)
  message(STATUS "${msg}")
  string(REPLACE "\\" "\\\\" msg "${msg}")
  string(REPLACE "\"" "\\\"" msg "${msg}")
  file(APPEND "${OPENCV_BUILD_INFO_FILE}" "\"${msg}\\n\"\n")
endfunction()

macro(ocv_finalize_status)
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

# remove all matching elements from the list
macro(ocv_list_filterout lst regex)
  foreach(item ${${lst}})
    if(item MATCHES "${regex}")
      list(REMOVE_ITEM ${lst} "${item}")
    endif()
  endforeach()
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

  if(MSVS)
    if(NOT INSTALL_IGNORE_PDB AND
        (INSTALL_PDB OR
          (INSTALL_CREATE_DISTRIB AND NOT BUILD_SHARED_LIBS)
        ))
      set(__target "${ARGV0}")

      set(isArchive 0)
      set(isDst 0)
      unset(__dst)
      foreach(e ${ARGN})
        if(isDst EQUAL 1)
          set(__dst "${e}")
          break()
        endif()
        if(isArchive EQUAL 1 AND e STREQUAL "DESTINATION")
          set(isDst 1)
        endif()
        if(e STREQUAL "ARCHIVE")
          set(isArchive 1)
        else()
          set(isArchive 0)
        endif()
      endforeach()

#      message(STATUS "Process ${__target} dst=${__dst}...")
      if(DEFINED __dst)
        # If CMake version is >=3.1.0 or <2.8.12.
        if(NOT CMAKE_VERSION VERSION_LESS 3.1.0 OR CMAKE_VERSION VERSION_LESS 2.8.12)
          get_target_property(fname ${__target} LOCATION_DEBUG)
          if(fname MATCHES "\\.lib$")
            string(REGEX REPLACE "\\.lib$" ".pdb" fname "${fname}")
            install(FILES "${fname}" DESTINATION "${__dst}" CONFIGURATIONS Debug OPTIONAL)
          endif()

          get_target_property(fname ${__target} LOCATION_RELEASE)
          if(fname MATCHES "\\.lib$")
            string(REGEX REPLACE "\\.lib$" ".pdb" fname "${fname}")
            install(FILES "${fname}" DESTINATION "${__dst}" CONFIGURATIONS Release OPTIONAL)
          endif()
        else()
          # CMake 2.8.12 broke PDB support for STATIC libraries from MSVS, fix was introduced in CMake 3.1.0.
          message(WARNING "PDB's are not supported from this version of CMake, use CMake version later then 3.1.0 or before 2.8.12.")
        endif()
      endif()
    endif()
  endif()
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

# read single version info from the pkg file
macro(ocv_parse_pkg LIBNAME PKG_PATH SCOPE)
  if(EXISTS "${PKG_PATH}/${LIBNAME}.pc")
    file(STRINGS "${PKG_PATH}/${LIBNAME}.pc" line_to_parse REGEX "^Version:[ \t]+[0-9.]*.*$" LIMIT_COUNT 1)
    STRING(REGEX REPLACE ".*Version: ([^ ]+).*" "\\1" ALIASOF_${LIBNAME}_VERSION "${line_to_parse}" )
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
  set(LINK_MODE "LINK_PRIVATE")
  set(LINK_PENDING "")
  foreach(dep ${LINK_DEPS})
    if(" ${dep}" STREQUAL " ${target}")
      # prevent "link to itself" warning (world problem)
    elseif(" ${dep}" STREQUAL " LINK_PRIVATE" OR " ${dep}" STREQUAL "LINK_PUBLIC")
      if(NOT LINK_PENDING STREQUAL "")
        __ocv_push_target_link_libraries(${LINK_MODE} ${LINK_PENDING})
        set(LINK_PENDING "")
        set(LINK_MODE "${dep}")
      endif()
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
  target_compile_definitions(${target} ${ARGN})
endfunction()


function(_ocv_append_target_includes target)
  if(DEFINED OCV_TARGET_INCLUDE_DIRS_${target})
    target_include_directories(${target} PRIVATE ${OCV_TARGET_INCLUDE_DIRS_${target}})
    if (TARGET ${target}_object)
      target_include_directories(${target}_object PRIVATE ${OCV_TARGET_INCLUDE_DIRS_${target}})
    endif()
    if(OPENCV_DEPENDANT_TARGETS_${target})
      foreach(t ${OPENCV_DEPENDANT_TARGETS_${target}})
        target_include_directories(${t} PRIVATE ${OCV_TARGET_INCLUDE_DIRS_${target}})
      endforeach()
    endif()
    unset(OCV_TARGET_INCLUDE_DIRS_${target} CACHE)
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

  # Add OBJECT library (added in cmake 2.8.8) to use in compound modules
  if (NOT CMAKE_VERSION VERSION_LESS "2.8.8" AND OPENCV_ENABLE_OBJECT_TARGETS
      AND NOT OPENCV_MODULE_${target}_CHILDREN
      AND NOT OPENCV_MODULE_${target}_CLASS STREQUAL "BINDINGS"
      AND NOT ${target} STREQUAL "opencv_ts"
      AND (NOT BUILD_opencv_world OR NOT HAVE_CUDA)
    )
    set(sources ${ARGN})
    ocv_list_filterout(sources "\\\\.(cl|inc|cu)$")
    add_library(${target}_object OBJECT ${sources})
    set_target_properties(${target}_object PROPERTIES
      EXCLUDE_FROM_ALL True
      EXCLUDE_FROM_DEFAULT_BUILD True
      POSITION_INDEPENDENT_CODE True
      )
    if (ENABLE_SOLUTION_FOLDERS)
      set_target_properties(${target}_object PROPERTIES FOLDER "object_libraries")
    endif()
    unset(sources)
  endif()

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
  string(REGEX REPLACE "^lib(.+).(a|so)(.[.0-9]+)?$" "\\1" __libname "${__libname}")
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
          if(_type STREQUAL "STATIC_LIBRARY" AND BUILD_SHARED_LIBS)
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
