# Copyright (c) 2013-2015, Ruslan Baratov
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This is a gate file to Hunter package manager.
# Include this file using `include` command and add package you need, example:
#
#     cmake_minimum_required(VERSION 3.0)
#
#     include("cmake/HunterGate.cmake")
#     HunterGate(
#         URL "https://github.com/path/to/hunter/archive.tar.gz"
#         SHA1 "798501e983f14b28b10cda16afa4de69eee1da1d"
#     )
#
#     project(MyProject)
#
#     hunter_add_package(Foo)
#     hunter_add_package(Boo COMPONENTS Bar Baz)
#
# Projects:
#     * https://github.com/hunter-packages/gate/
#     * https://github.com/ruslo/hunter

cmake_minimum_required(VERSION 3.0) # Minimum for Hunter
include(CMakeParseArguments) # cmake_parse_arguments

option(HUNTER_ENABLED "Enable Hunter package manager support" ON)
option(HUNTER_STATUS_PRINT "Print working status" ON)
option(HUNTER_STATUS_DEBUG "Print a lot info" OFF)

set(HUNTER_WIKI "https://github.com/ruslo/hunter/wiki")

function(hunter_gate_status_print)
  foreach(print_message ${ARGV})
    if(HUNTER_STATUS_PRINT OR HUNTER_STATUS_DEBUG)
      message(STATUS "[hunter] ${print_message}")
    endif()
  endforeach()
endfunction()

function(hunter_gate_status_debug)
  foreach(print_message ${ARGV})
    if(HUNTER_STATUS_DEBUG)
      string(TIMESTAMP timestamp)
      message(STATUS "[hunter *** DEBUG *** ${timestamp}] ${print_message}")
    endif()
  endforeach()
endfunction()

function(hunter_gate_wiki wiki_page)
  message("------------------------------ WIKI -------------------------------")
  message("    ${HUNTER_WIKI}/${wiki_page}")
  message("-------------------------------------------------------------------")
  message("")
  message(FATAL_ERROR "")
endfunction()

function(hunter_gate_internal_error)
  message("")
  foreach(print_message ${ARGV})
    message("[hunter ** INTERNAL **] ${print_message}")
  endforeach()
  message("[hunter ** INTERNAL **] [Directory:${CMAKE_CURRENT_LIST_DIR}]")
  message("")
  hunter_gate_wiki("error.internal")
endfunction()

function(hunter_gate_fatal_error)
  cmake_parse_arguments(hunter "" "WIKI" "" "${ARGV}")
  string(COMPARE EQUAL "${hunter_WIKI}" "" have_no_wiki)
  if(have_no_wiki)
    hunter_gate_internal_error("Expected wiki")
  endif()
  message("")
  foreach(x ${hunter_UNPARSED_ARGUMENTS})
    message("[hunter ** FATAL ERROR **] ${x}")
  endforeach()
  message("[hunter ** FATAL ERROR **] [Directory:${CMAKE_CURRENT_LIST_DIR}]")
  message("")
  hunter_gate_wiki("${hunter_WIKI}")
endfunction()

function(hunter_gate_user_error)
  hunter_gate_fatal_error(${ARGV} WIKI "error.incorrect.input.data")
endfunction()

function(hunter_gate_self root version sha1 result)
  string(COMPARE EQUAL "${root}" "" is_bad)
  if(is_bad)
    hunter_gate_internal_error("root is empty")
  endif()

  string(COMPARE EQUAL "${version}" "" is_bad)
  if(is_bad)
    hunter_gate_internal_error("version is empty")
  endif()

  string(COMPARE EQUAL "${sha1}" "" is_bad)
  if(is_bad)
    hunter_gate_internal_error("sha1 is empty")
  endif()

  string(SUBSTRING "${sha1}" 0 7 archive_id)

  if(EXISTS "${root}/cmake/Hunter")
    set(hunter_self "${root}")
  else()
    set(
        hunter_self
        "${root}/_Base/Download/Hunter/${version}/${archive_id}/Unpacked"
    )
  endif()

  set("${result}" "${hunter_self}" PARENT_SCOPE)
endfunction()

# Set HUNTER_GATE_ROOT cmake variable to suitable value.
function(hunter_gate_detect_root)
  # Check CMake variable
  string(COMPARE NOTEQUAL "${HUNTER_ROOT}" "" not_empty)
  if(not_empty)
    set(HUNTER_GATE_ROOT "${HUNTER_ROOT}" PARENT_SCOPE)
    hunter_gate_status_debug("HUNTER_ROOT detected by cmake variable")
    return()
  endif()

  # Check environment variable
  string(COMPARE NOTEQUAL "$ENV{HUNTER_ROOT}" "" not_empty)
  if(not_empty)
    set(HUNTER_GATE_ROOT "$ENV{HUNTER_ROOT}" PARENT_SCOPE)
    hunter_gate_status_debug("HUNTER_ROOT detected by environment variable")
    return()
  endif()

  # Check HOME environment variable
  string(COMPARE NOTEQUAL "$ENV{HOME}" "" result)
  if(result)
    set(HUNTER_GATE_ROOT "$ENV{HOME}/.hunter" PARENT_SCOPE)
    hunter_gate_status_debug("HUNTER_ROOT set using HOME environment variable")
    return()
  endif()

  # Check SYSTEMDRIVE and USERPROFILE environment variable (windows only)
  if(WIN32)
    string(COMPARE NOTEQUAL "$ENV{SYSTEMDRIVE}" "" result)
    if(result)
      set(HUNTER_GATE_ROOT "$ENV{SYSTEMDRIVE}/.hunter" PARENT_SCOPE)
      hunter_gate_status_debug(
          "HUNTER_ROOT set using SYSTEMDRIVE environment variable"
      )
      return()
    endif()

    string(COMPARE NOTEQUAL "$ENV{USERPROFILE}" "" result)
    if(result)
      set(HUNTER_GATE_ROOT "$ENV{USERPROFILE}/.hunter" PARENT_SCOPE)
      hunter_gate_status_debug(
          "HUNTER_ROOT set using USERPROFILE environment variable"
      )
      return()
    endif()
  endif()

  hunter_gate_fatal_error(
      "Can't detect HUNTER_ROOT"
      WIKI "error.detect.hunter.root"
  )
endfunction()

macro(hunter_gate_lock dir)
  if(NOT HUNTER_SKIP_LOCK)
    if("${CMAKE_VERSION}" VERSION_LESS "3.2")
      hunter_gate_fatal_error(
          "Can't lock, upgrade to CMake 3.2 or use HUNTER_SKIP_LOCK"
          WIKI "error.can.not.lock"
      )
    endif()
    hunter_gate_status_debug("Locking directory: ${dir}")
    file(LOCK "${dir}" DIRECTORY GUARD FUNCTION)
    hunter_gate_status_debug("Lock done")
  endif()
endmacro()

function(hunter_gate_download dir)
  string(
      COMPARE
      NOTEQUAL
      "$ENV{HUNTER_DISABLE_AUTOINSTALL}"
      ""
      disable_autoinstall
  )
  if(disable_autoinstall AND NOT HUNTER_RUN_INSTALL)
    hunter_gate_fatal_error(
        "Hunter not found in '${dir}'"
        "Set HUNTER_RUN_INSTALL=ON to auto-install it from '${HUNTER_GATE_URL}'"
        "Settings:"
        "  HUNTER_ROOT: ${HUNTER_GATE_ROOT}"
        "  HUNTER_SHA1: ${HUNTER_GATE_SHA1}"
        WIKI "error.run.install"
    )
  endif()
  string(COMPARE EQUAL "${dir}" "" is_bad)
  if(is_bad)
    hunter_gate_internal_error("Empty 'dir' argument")
  endif()

  string(COMPARE EQUAL "${HUNTER_GATE_SHA1}" "" is_bad)
  if(is_bad)
    hunter_gate_internal_error("HUNTER_GATE_SHA1 empty")
  endif()

  string(COMPARE EQUAL "${HUNTER_GATE_URL}" "" is_bad)
  if(is_bad)
    hunter_gate_internal_error("HUNTER_GATE_URL empty")
  endif()

  set(done_location "${dir}/DONE")
  set(sha1_location "${dir}/SHA1")

  set(build_dir "${dir}/Build")
  set(cmakelists "${dir}/CMakeLists.txt")

  hunter_gate_lock("${dir}")
  if(EXISTS "${done_location}")
    # while waiting for lock other instance can do all the job
    hunter_gate_status_debug("File '${done_location}' found, skip install")
    return()
  endif()

  file(REMOVE_RECURSE "${build_dir}")
  file(REMOVE_RECURSE "${cmakelists}")

  file(MAKE_DIRECTORY "${build_dir}") # check directory permissions

  # Disabling languages speeds up a little bit, reduces noise in the output
  # and avoids path too long windows error
  file(
      WRITE
      "${cmakelists}"
      "cmake_minimum_required(VERSION 3.0)\n"
      "project(HunterDownload LANGUAGES NONE)\n"
      "include(ExternalProject)\n"
      "ExternalProject_Add(\n"
      "    Hunter\n"
      "    URL\n"
      "    \"${HUNTER_GATE_URL}\"\n"
      "    URL_HASH\n"
      "    SHA1=${HUNTER_GATE_SHA1}\n"
      "    DOWNLOAD_DIR\n"
      "    \"${dir}\"\n"
      "    SOURCE_DIR\n"
      "    \"${dir}/Unpacked\"\n"
      "    CONFIGURE_COMMAND\n"
      "    \"\"\n"
      "    BUILD_COMMAND\n"
      "    \"\"\n"
      "    INSTALL_COMMAND\n"
      "    \"\"\n"
      ")\n"
  )

  if(HUNTER_STATUS_DEBUG)
    set(logging_params "")
  else()
    set(logging_params OUTPUT_QUIET)
  endif()

  hunter_gate_status_debug("Run generate")
  execute_process(
      COMMAND "${CMAKE_COMMAND}" "-H${dir}" "-B${build_dir}"
      WORKING_DIRECTORY "${dir}"
      RESULT_VARIABLE download_result
      ${logging_params}
  )

  if(NOT download_result EQUAL 0)
    hunter_gate_internal_error("Configure project failed")
  endif()

  hunter_gate_status_print(
      "Initializing Hunter workspace (${HUNTER_GATE_SHA1})"
      "  ${HUNTER_GATE_URL}"
      "  -> ${dir}"
  )
  execute_process(
      COMMAND "${CMAKE_COMMAND}" --build "${build_dir}"
      WORKING_DIRECTORY "${dir}"
      RESULT_VARIABLE download_result
      ${logging_params}
  )

  if(NOT download_result EQUAL 0)
    hunter_gate_internal_error("Build project failed")
  endif()

  file(REMOVE_RECURSE "${build_dir}")
  file(REMOVE_RECURSE "${cmakelists}")

  file(WRITE "${sha1_location}" "${HUNTER_GATE_SHA1}")
  file(WRITE "${done_location}" "DONE")

  hunter_gate_status_debug("Finished")
endfunction()

# Must be a macro so master file 'cmake/Hunter' can
# apply all variables easily just by 'include' command
# (otherwise PARENT_SCOPE magic needed)
macro(HunterGate)
  if(HUNTER_GATE_DONE)
    # variable HUNTER_GATE_DONE set explicitly for external project
    # (see `hunter_download`)
    set_property(GLOBAL PROPERTY HUNTER_GATE_DONE YES)
  endif()

  # First HunterGate command will init Hunter, others will be ignored
  get_property(_hunter_gate_done GLOBAL PROPERTY HUNTER_GATE_DONE SET)

  if(NOT HUNTER_ENABLED)
    # Empty function to avoid error "unknown function"
    function(hunter_add_package)
    endfunction()
  elseif(_hunter_gate_done)
    hunter_gate_status_debug("Secondary HunterGate (use old settings)")
    hunter_gate_self(
        "${HUNTER_CACHED_ROOT}"
        "${HUNTER_VERSION}"
        "${HUNTER_SHA1}"
        _hunter_self
    )
    include("${_hunter_self}/cmake/Hunter")
  else()
    set(HUNTER_GATE_LOCATION "${CMAKE_CURRENT_LIST_DIR}")

    string(COMPARE NOTEQUAL "${PROJECT_NAME}" "" _have_project_name)
    if(_have_project_name)
      hunter_gate_fatal_error(
          "Please set HunterGate *before* 'project' command. "
          "Detected project: ${PROJECT_NAME}"
          WIKI "error.huntergate.before.project"
      )
    endif()

    cmake_parse_arguments(
        HUNTER_GATE "LOCAL" "URL;SHA1;GLOBAL;FILEPATH" "" ${ARGV}
    )

    string(COMPARE EQUAL "${HUNTER_GATE_SHA1}" "" _empty_sha1)
    string(COMPARE EQUAL "${HUNTER_GATE_URL}" "" _empty_url)
    string(
        COMPARE
        NOTEQUAL
        "${HUNTER_GATE_UNPARSED_ARGUMENTS}"
        ""
        _have_unparsed
    )
    string(COMPARE NOTEQUAL "${HUNTER_GATE_GLOBAL}" "" _have_global)
    string(COMPARE NOTEQUAL "${HUNTER_GATE_FILEPATH}" "" _have_filepath)

    if(_empty_sha1)
      hunter_gate_user_error("SHA1 suboption of HunterGate is mandatory")
    endif()
    if(_empty_url)
      hunter_gate_user_error("URL suboption of HunterGate is mandatory")
    endif()
    if(_have_unparsed)
      hunter_gate_user_error(
          "HunterGate unparsed arguments: ${HUNTER_GATE_UNPARSED_ARGUMENTS}"
      )
    endif()
    if(_have_global)
      if(HUNTER_GATE_LOCAL)
        hunter_gate_user_error("Unexpected LOCAL (already has GLOBAL)")
      endif()
      if(_have_filepath)
        hunter_gate_user_error("Unexpected FILEPATH (already has GLOBAL)")
      endif()
    endif()
    if(HUNTER_GATE_LOCAL)
      if(_have_global)
        hunter_gate_user_error("Unexpected GLOBAL (already has LOCAL)")
      endif()
      if(_have_filepath)
        hunter_gate_user_error("Unexpected FILEPATH (already has LOCAL)")
      endif()
    endif()
    if(_have_filepath)
      if(_have_global)
        hunter_gate_user_error("Unexpected GLOBAL (already has FILEPATH)")
      endif()
      if(HUNTER_GATE_LOCAL)
        hunter_gate_user_error("Unexpected LOCAL (already has FILEPATH)")
      endif()
    endif()

    hunter_gate_detect_root() # set HUNTER_GATE_ROOT

    # Beautify path, fix probable problems with windows path slashes
    get_filename_component(
        HUNTER_GATE_ROOT "${HUNTER_GATE_ROOT}" ABSOLUTE
    )
    hunter_gate_status_debug("HUNTER_ROOT: ${HUNTER_GATE_ROOT}")
    if(NOT HUNTER_ALLOW_SPACES_IN_PATH)
      string(FIND "${HUNTER_GATE_ROOT}" " " _contain_spaces)
      if(NOT _contain_spaces EQUAL -1)
        hunter_gate_fatal_error(
            "HUNTER_ROOT (${HUNTER_GATE_ROOT}) contains spaces."
            "Set HUNTER_ALLOW_SPACES_IN_PATH=ON to skip this error"
            "(Use at your own risk!)"
            WIKI "error.spaces.in.hunter.root"
        )
      endif()
    endif()

    string(
        REGEX
        MATCH
        "[0-9]+\\.[0-9]+\\.[0-9]+[-_a-z0-9]*"
        HUNTER_GATE_VERSION
        "${HUNTER_GATE_URL}"
    )
    string(COMPARE EQUAL "${HUNTER_GATE_VERSION}" "" _is_empty)
    if(_is_empty)
      set(HUNTER_GATE_VERSION "unknown")
    endif()

    hunter_gate_self(
        "${HUNTER_GATE_ROOT}"
        "${HUNTER_GATE_VERSION}"
        "${HUNTER_GATE_SHA1}"
        hunter_self_
    )

    set(_master_location "${hunter_self_}/cmake/Hunter")
    if(EXISTS "${HUNTER_GATE_ROOT}/cmake/Hunter")
      # Hunter downloaded manually (e.g. by 'git clone')
      set(_unused "xxxxxxxxxx")
      set(HUNTER_GATE_SHA1 "${_unused}")
      set(HUNTER_GATE_VERSION "${_unused}")
    else()
      get_filename_component(_archive_id_location "${hunter_self_}/.." ABSOLUTE)
      set(_done_location "${_archive_id_location}/DONE")
      set(_sha1_location "${_archive_id_location}/SHA1")

      # Check Hunter already downloaded by HunterGate
      if(NOT EXISTS "${_done_location}")
        hunter_gate_download("${_archive_id_location}")
      endif()

      if(NOT EXISTS "${_done_location}")
        hunter_gate_internal_error("hunter_gate_download failed")
      endif()

      if(NOT EXISTS "${_sha1_location}")
        hunter_gate_internal_error("${_sha1_location} not found")
      endif()
      file(READ "${_sha1_location}" _sha1_value)
      string(COMPARE EQUAL "${_sha1_value}" "${HUNTER_GATE_SHA1}" _is_equal)
      if(NOT _is_equal)
        hunter_gate_internal_error(
            "Short SHA1 collision:"
            "  ${_sha1_value} (from ${_sha1_location})"
            "  ${HUNTER_GATE_SHA1} (HunterGate)"
        )
      endif()
      if(NOT EXISTS "${_master_location}")
        hunter_gate_user_error(
            "Master file not found:"
            "  ${_master_location}"
            "try to update Hunter/HunterGate"
        )
      endif()
    endif()
    include("${_master_location}")
    set_property(GLOBAL PROPERTY HUNTER_GATE_DONE YES)
  endif()
endmacro()
