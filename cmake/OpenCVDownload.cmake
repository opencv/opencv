#
#  Download and optionally unpack a file
#
#  ocv_download(FILENAME p HASH h URL u1 [u2 ...] DESTINATION_DIR d [ID id] [STATUS s] [UNPACK] [RELATIVE_URL])
#    FILENAME - filename
#    HASH - MD5 hash
#    URL - full download url (first nonempty value will be chosen)
#    DESTINATION_DIR - file will be copied to this directory
#    ID     - identifier for project/group of downloaded files
#    STATUS - passed variable will be updated in parent scope,
#             function will not fail the build in case of download problem if this option is provided,
#             but will fail in case when other operations (copy, remove, etc.) failed
#    UNPACK - downloaded file will be unpacked to DESTINATION_DIR
#    RELATIVE_URL - if set, then URL is treated as a base, and FILENAME will be appended to it
#  Note: uses OPENCV_DOWNLOAD_PATH folder as cache, default is <opencv>/.cache

set(HELP_OPENCV_DOWNLOAD_PATH "Cache directory for downloaded files")
if(DEFINED ENV{OPENCV_DOWNLOAD_PATH})
  set(OPENCV_DOWNLOAD_PATH "$ENV{OPENCV_DOWNLOAD_PATH}" CACHE PATH "${HELP_OPENCV_DOWNLOAD_PATH}")
endif()
set(OPENCV_DOWNLOAD_PATH "${OpenCV_SOURCE_DIR}/.cache" CACHE PATH "${HELP_OPENCV_DOWNLOAD_PATH}")
set(OPENCV_DOWNLOAD_LOG "${OpenCV_BINARY_DIR}/CMakeDownloadLog.txt")

# Init download cache directory and log file
if(NOT EXISTS "${OPENCV_DOWNLOAD_PATH}")
  file(MAKE_DIRECTORY ${OPENCV_DOWNLOAD_PATH})
endif()
if(NOT EXISTS "${OPENCV_DOWNLOAD_PATH}/.gitignore")
  file(WRITE "${OPENCV_DOWNLOAD_PATH}/.gitignore" "*\n")
endif()
file(WRITE "${OPENCV_DOWNLOAD_LOG}" "use_cache \"${OPENCV_DOWNLOAD_PATH}\"\n")


function(ocv_download)
  cmake_parse_arguments(DL "UNPACK;RELATIVE_URL" "FILENAME;HASH;DESTINATION_DIR;ID;STATUS" "URL" ${ARGN})

  macro(ocv_download_log)
    file(APPEND "${OPENCV_DOWNLOAD_LOG}" "${ARGN}\n")
  endmacro()

  ocv_assert(DL_FILENAME)
  ocv_assert(DL_HASH)
  ocv_assert(DL_URL)
  ocv_assert(DL_DESTINATION_DIR)
  if((NOT " ${DL_UNPARSED_ARGUMENTS}" STREQUAL " ")
    OR DL_FILENAME STREQUAL ""
    OR DL_HASH STREQUAL ""
    OR DL_URL STREQUAL ""
    OR DL_DESTINATION_DIR STREQUAL ""
  )
    set(msg_level FATAL_ERROR)
    if(DEFINED DL_STATUS)
      set(${DL_STATUS} FALSE PARENT_SCOPE)
      set(msg_level WARNING)
    endif()
    message(${msg_level} "ERROR: ocv_download() unsupported arguments: ${ARGV}")
    return()
  endif()

  if(DEFINED DL_STATUS)
    set(${DL_STATUS} TRUE PARENT_SCOPE)
  endif()

  # Check CMake cache for already processed tasks
  string(FIND "${DL_DESTINATION_DIR}" "${CMAKE_BINARY_DIR}" DL_BINARY_PATH_POS)
  if(DL_BINARY_PATH_POS EQUAL 0)
    set(__file_id "${DL_DESTINATION_DIR}/${DL_FILENAME}")
    file(RELATIVE_PATH __file_id "${CMAKE_BINARY_DIR}" "${__file_id}")
    string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" __file_id "${__file_id}")
    if(DL_ID)
      string(TOUPPER ${DL_ID} __id)
      string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" __id "${__id}")
      set(OCV_DOWNLOAD_HASH_NAME "OCV_DOWNLOAD_${__id}_HASH_${__file_id}")
    else()
      set(OCV_DOWNLOAD_HASH_NAME "OCV_DOWNLOAD_HASH_${__file_id}")
    endif()
    if(" ${${OCV_DOWNLOAD_HASH_NAME}}" STREQUAL " ${DL_HASH}")
      ocv_download_log("#match_hash_in_cmake_cache \"${OCV_DOWNLOAD_HASH_NAME}\"")
      return()
    endif()
    unset("${OCV_DOWNLOAD_HASH_NAME}" CACHE)
  else()
    set(OCV_DOWNLOAD_HASH_NAME "")
    #message(WARNING "Download destination is not in CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}: ${DL_DESTINATION_DIR}")
  endif()

  # Select first non-empty url
  foreach(url ${DL_URL})
    if(url)
      set(DL_URL "${url}")
      break()
    endif()
  endforeach()

  # Append filename to url if needed
  if(DL_RELATIVE_URL)
    set(DL_URL "${DL_URL}${DL_FILENAME}")
  endif()

  set(mode "copy")
  if(DL_UNPACK)
    set(mode "unpack")
  endif()

  # Log all calls to file
  ocv_download_log("do_${mode} \"${DL_FILENAME}\" \"${DL_HASH}\" \"${DL_URL}\" \"${DL_DESTINATION_DIR}\"")
  # ... and to console
  set(__msg_prefix "")
  if(DL_ID)
    set(__msg_prefix "${DL_ID}: ")
  endif()
  message(STATUS "${__msg_prefix}Download: ${DL_FILENAME}")

  # Copy mode: check if copy destination exists and is correct
  if(NOT DL_UNPACK)
    set(COPY_DESTINATION "${DL_DESTINATION_DIR}/${DL_FILENAME}")
    if(EXISTS "${COPY_DESTINATION}")
      ocv_download_log("#check_md5 \"${COPY_DESTINATION}\"")
      file(MD5 "${COPY_DESTINATION}" target_md5)
      if(target_md5 STREQUAL DL_HASH)
        ocv_download_log("#match_md5 \"${COPY_DESTINATION}\" \"${target_md5}\"")
        if(OCV_DOWNLOAD_HASH_NAME)
          set(${OCV_DOWNLOAD_HASH_NAME} "${DL_HASH}" CACHE INTERNAL "")
        endif()
        return()
      endif()
      ocv_download_log("#mismatch_md5 \"${COPY_DESTINATION}\" \"${target_md5}\"")
    else()
      ocv_download_log("#missing \"${COPY_DESTINATION}\"")
    endif()
  endif()

  # Check cache first
  if(DL_ID)
    string(TOLOWER "${DL_ID}" __id)
    string(REGEX REPLACE "[^a-zA-Z0-9_/ ]" "_" __id "${__id}")
    set(CACHE_CANDIDATE "${OPENCV_DOWNLOAD_PATH}/${__id}/${DL_HASH}-${DL_FILENAME}")
  else()
    set(CACHE_CANDIDATE "${OPENCV_DOWNLOAD_PATH}/${DL_HASH}-${DL_FILENAME}")
  endif()
  if(EXISTS "${CACHE_CANDIDATE}")
    ocv_download_log("#check_md5 \"${CACHE_CANDIDATE}\"")
    file(MD5 "${CACHE_CANDIDATE}" target_md5)
    if(NOT target_md5 STREQUAL DL_HASH)
      ocv_download_log("#mismatch_md5 \"${CACHE_CANDIDATE}\" \"${target_md5}\"")
      ocv_download_log("#delete \"${CACHE_CANDIDATE}\"")
      file(REMOVE ${CACHE_CANDIDATE})
    endif()
  endif()

  # Download
  if(NOT EXISTS "${CACHE_CANDIDATE}")
    ocv_download_log("#cmake_download \"${CACHE_CANDIDATE}\" \"${DL_URL}\"")
    file(DOWNLOAD "${DL_URL}" "${CACHE_CANDIDATE}"
         INACTIVITY_TIMEOUT 60
         TIMEOUT 600
         STATUS status
         LOG __log)
    string(LENGTH "${__log}" __log_length)
    if(__log_length LESS 65536)
      string(REPLACE "\n" "\n# " __log "${__log}")
      ocv_download_log("# ${__log}\n")
    endif()
    if(NOT status EQUAL 0)
      set(msg_level FATAL_ERROR)
      if(DEFINED DL_STATUS)
        set(${DL_STATUS} FALSE PARENT_SCOPE)
        set(msg_level WARNING)
      endif()
      if(status MATCHES "Couldn't resolve host name")
        message(STATUS "
=======================================================================
  Couldn't download files from the Internet.
  Please check the Internet access on this host.
=======================================================================
")
      elseif(status MATCHES "Couldn't connect to server")
        message(STATUS "
=======================================================================
  Couldn't connect to server from the Internet.
  Perhaps direct connections are not allowed in the current network.
  To use proxy please check/specify these environment variables:
  - http_proxy/https_proxy
  - and/or HTTP_PROXY/HTTPS_PROXY
=======================================================================
")
      endif()
      message(${msg_level} "${__msg_prefix}Download failed: ${status}
For details please refer to the download log file:
${OPENCV_DOWNLOAD_LOG}
")
      return()
    endif()

    # Don't remove this code, because EXPECTED_MD5 parameter doesn't fail "file(DOWNLOAD)" step on wrong hash
    ocv_download_log("#check_md5 \"${CACHE_CANDIDATE}\"")
    file(MD5 "${CACHE_CANDIDATE}" target_md5)
    if(NOT target_md5 STREQUAL DL_HASH)
      ocv_download_log("#mismatch_md5 \"${CACHE_CANDIDATE}\" \"${target_md5}\"")
      set(msg_level FATAL_ERROR)
      if(DEFINED DL_STATUS)
        set(${DL_STATUS} FALSE PARENT_SCOPE)
        set(msg_level WARNING)
      endif()
      message(${msg_level} "${__msg_prefix}Hash mismatch: ${target_md5}")
      return()
    endif()
  endif()

  # Unpack or copy
  if(DL_UNPACK)
    if(EXISTS "${DL_DESTINATION_DIR}")
      ocv_download_log("#remove_unpack \"${DL_DESTINATION_DIR}\"")
      file(REMOVE_RECURSE "${DL_DESTINATION_DIR}")
    endif()
    ocv_download_log("#mkdir \"${DL_DESTINATION_DIR}\"")
    file(MAKE_DIRECTORY "${DL_DESTINATION_DIR}")
    ocv_download_log("#unpack \"${DL_DESTINATION_DIR}\" \"${CACHE_CANDIDATE}\"")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xzf "${CACHE_CANDIDATE}"
                    WORKING_DIRECTORY "${DL_DESTINATION_DIR}"
                    RESULT_VARIABLE res)
    if(NOT res EQUAL 0)
      message(FATAL_ERROR "${__msg_prefix}Unpack failed: ${res}")
    endif()
  else()
    ocv_download_log("#copy \"${COPY_DESTINATION}\" \"${CACHE_CANDIDATE}\"")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CACHE_CANDIDATE}" "${COPY_DESTINATION}"
                    RESULT_VARIABLE res)
    if(NOT res EQUAL 0)
      message(FATAL_ERROR "${__msg_prefix}Copy failed: ${res}")
    endif()
  endif()

  if(OCV_DOWNLOAD_HASH_NAME)
    set(${OCV_DOWNLOAD_HASH_NAME} "${DL_HASH}" CACHE INTERNAL "")
  endif()
endfunction()
