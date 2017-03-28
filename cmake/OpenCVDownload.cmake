#
#  Download and optionally unpack a file
#
#  ocv_download(FILENAME p HASH h URL u1 [u2 ...] DESTINATION_DIR d [STATUS s] [UNPACK] [RELATIVE])
#    FILENAME - filename
#    HASH - MD5 hash
#    URL - full download url (first nonempty value will be chosen)
#    DESTINATION_DIR - file will be copied to this directory
#    STATUS - passed variable will be updated in parent scope,
#             function will not fail the build in case of download problem if this option is provided,
#             but will fail in case when other operations (copy, remove, etc.) failed
#    UNPACK - downloaded file will be unpacked to DESTINATION_DIR
#    RELATIVE_URL - if set, then URL is treated as a base, and FILENAME will be appended to it
#  Note: uses OPENCV_DOWNLOAD_PATH folder as cache, default is <opencv>/.cache

set(OPENCV_DOWNLOAD_PATH "${OpenCV_SOURCE_DIR}/.cache" CACHE PATH "Cache directory for downloaded files")
set(OPENCV_DOWNLOAD_LOG "${OpenCV_BINARY_DIR}/CMakeDownloadLog.txt")

# Init download cache directory and log file
if(NOT EXISTS "${OPENCV_DOWNLOAD_PATH}")
  file(MAKE_DIRECTORY ${OPENCV_DOWNLOAD_PATH})
endif()
file(WRITE "${OPENCV_DOWNLOAD_LOG}" "use_cache \"${OPENCV_DOWNLOAD_PATH}\"\n")


function(ocv_download)
  cmake_parse_arguments(DL "UNPACK;RELATIVE_URL" "FILENAME;HASH;DESTINATION_DIR;STATUS" "URL" ${ARGN})

  ocv_assert(DEFINED DL_FILENAME)
  ocv_assert(DEFINED DL_HASH)
  ocv_assert(DEFINED DL_URL)
  ocv_assert(DEFINED DL_DESTINATION_DIR)

  if(DEFINED DL_STATUS)
    set(${DL_STATUS} TRUE PARENT_SCOPE)
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
  file(APPEND "${OPENCV_DOWNLOAD_LOG}" "do_${mode} \"${DL_FILENAME}\" \"${DL_HASH}\" \"${DL_URL}\" \"${DL_DESTINATION_DIR}\"\n")
  # ... and to console
  message(STATUS "Download: ${DL_FILENAME}")

  # Copy mode: check if copy destination exists and is correct
  if(NOT DL_UNPACK)
    set(COPY_DESTINATION "${DL_DESTINATION_DIR}/${DL_FILENAME}")
    if(EXISTS "${COPY_DESTINATION}")
      file(MD5 "${COPY_DESTINATION}" target_md5)
      if(target_md5 STREQUAL DL_HASH)
        return()
      endif()
    endif()
  endif()

  # Check cache first
  set(CACHE_CANDIDATE "${OPENCV_DOWNLOAD_PATH}/${DL_HASH}-${DL_FILENAME}")
  if(EXISTS "${CACHE_CANDIDATE}")
    file(MD5 "${CACHE_CANDIDATE}" target_md5)
    if(NOT target_md5 STREQUAL DL_HASH)
      file(REMOVE ${CACHE_CANDIDATE})
    endif()
  endif()

  # Download
  if(NOT EXISTS "${CACHE_CANDIDATE}")
    file(DOWNLOAD "${DL_URL}" "${CACHE_CANDIDATE}"
         INACTIVITY_TIMEOUT 60
         TIMEOUT 600
         STATUS status
         LOG log)
    string(REPLACE "\n" "\n# " log "# ${log}")
    file(APPEND "${OPENCV_DOWNLOAD_LOG}" "${log}\n\n")
    if(NOT status EQUAL 0)
      set(msg_level FATAL_ERROR)
      if(DEFINED DL_STATUS)
        set(${DL_STATUS} FALSE PARENT_SCOPE)
        set(msg_level WARNING)
      endif()
      message(${msg_level} "Download failed: ${status}")
      return()
    endif()

    # Don't remove this code, because EXPECTED_MD5 parameter doesn't fail "file(DOWNLOAD)" step on wrong hash
    file(MD5 "${CACHE_CANDIDATE}" target_md5)
    if(NOT target_md5 STREQUAL DL_HASH)
      set(msg_level FATAL_ERROR)
      if(DEFINED DL_STATUS)
        set(${DL_STATUS} FALSE PARENT_SCOPE)
        set(msg_level WARNING)
      endif()
      message(${msg_level} "Hash mismatch: ${target_md5}")
      return()
    endif()
  endif()

  # Unpack or copy
  if(DL_UNPACK)
    if(EXISTS "${DL_DESTINATION_DIR}")
      file(REMOVE_RECURSE "${DL_DESTINATION_DIR}")
    endif()
    file(MAKE_DIRECTORY "${DL_DESTINATION_DIR}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xz "${CACHE_CANDIDATE}"
                    WORKING_DIRECTORY "${DL_DESTINATION_DIR}"
                    RESULT_VARIABLE res)
    if(NOT res EQUAL 0)
      message(FATAL_ERROR "Unpack failed: ${res}")
    endif()
  else()
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CACHE_CANDIDATE}" "${COPY_DESTINATION}"
                    RESULT_VARIABLE res)
    if(NOT res EQUAL 0)
      message(FATAL_ERROR "Copy failed: ${res}")
    endif()
  endif()

endfunction()
