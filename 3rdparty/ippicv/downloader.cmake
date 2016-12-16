#
# The script downloads ICV package
#
# On return this will define:
# OPENCV_ICV_PATH - path to unpacked downloaded package
#

function(_icv_downloader)
  # Commit SHA in the opencv_3rdparty repo
  set(IPPICV_BINARIES_COMMIT "81a676001ca8075ada498583e4166079e5744668")
  # Define actual ICV versions
  if(APPLE)
    set(OPENCV_ICV_PACKAGE_NAME "ippicv_macosx_20151201.tgz")
    set(OPENCV_ICV_PACKAGE_HASH "4ff1fde9a7cfdfe7250bfcd8334e0f2f")
    set(OPENCV_ICV_PLATFORM "macosx")
    set(OPENCV_ICV_PACKAGE_SUBDIR "/ippicv_osx")
  elseif(UNIX)
    if(ANDROID AND NOT (ANDROID_ABI STREQUAL x86 OR ANDROID_ABI STREQUAL x86_64))
      return()
    endif()
    set(OPENCV_ICV_PACKAGE_NAME "ippicv_linux_20151201.tgz")
    set(OPENCV_ICV_PACKAGE_HASH "808b791a6eac9ed78d32a7666804320e")
    set(OPENCV_ICV_PLATFORM "linux")
    set(OPENCV_ICV_PACKAGE_SUBDIR "/ippicv_lnx")
  elseif(WIN32 AND NOT ARM)
    set(OPENCV_ICV_PACKAGE_NAME "ippicv_windows_20151201.zip")
    set(OPENCV_ICV_PACKAGE_HASH "04e81ce5d0e329c3fbc606ae32cad44d")
    set(OPENCV_ICV_PLATFORM "windows")
    set(OPENCV_ICV_PACKAGE_SUBDIR "/ippicv_win")
  else()
    return() # Not supported
  endif()

  set(OPENCV_ICV_UNPACK_PATH "${CMAKE_BINARY_DIR}/3rdparty/ippicv")
  set(OPENCV_ICV_PATH "${OPENCV_ICV_UNPACK_PATH}${OPENCV_ICV_PACKAGE_SUBDIR}")

  if(DEFINED OPENCV_ICV_PACKAGE_DOWNLOADED
       AND OPENCV_ICV_PACKAGE_DOWNLOADED STREQUAL OPENCV_ICV_PACKAGE_HASH
       AND EXISTS ${OPENCV_ICV_PATH})
    # Package has been downloaded and checked by the previous build
    set(OPENCV_ICV_PATH "${OPENCV_ICV_PATH}" PARENT_SCOPE)
    return()
  else()
    if(EXISTS ${OPENCV_ICV_UNPACK_PATH})
      message(STATUS "ICV: Removing previous unpacked package: ${OPENCV_ICV_UNPACK_PATH}")
      file(REMOVE_RECURSE ${OPENCV_ICV_UNPACK_PATH})
    endif()
  endif()
  unset(OPENCV_ICV_PACKAGE_DOWNLOADED CACHE)

  set(OPENCV_ICV_PACKAGE_ARCHIVE "${CMAKE_CURRENT_LIST_DIR}/downloads/${OPENCV_ICV_PLATFORM}-${OPENCV_ICV_PACKAGE_HASH}/${OPENCV_ICV_PACKAGE_NAME}")
  get_filename_component(OPENCV_ICV_PACKAGE_ARCHIVE_DIR "${OPENCV_ICV_PACKAGE_ARCHIVE}" PATH)
  if(EXISTS "${OPENCV_ICV_PACKAGE_ARCHIVE}")
    file(MD5 "${OPENCV_ICV_PACKAGE_ARCHIVE}" archive_md5)
    if(NOT archive_md5 STREQUAL OPENCV_ICV_PACKAGE_HASH)
      message(WARNING "ICV: Local copy of ICV package has invalid MD5 hash: ${archive_md5} (expected: ${OPENCV_ICV_PACKAGE_HASH})")
      file(REMOVE "${OPENCV_ICV_PACKAGE_ARCHIVE}")
      file(REMOVE_RECURSE "${OPENCV_ICV_PACKAGE_ARCHIVE_DIR}")
    endif()
  endif()

  if(NOT EXISTS "${OPENCV_ICV_PACKAGE_ARCHIVE}")
    if(NOT DEFINED OPENCV_ICV_URL)
      if(DEFINED ENV{OPENCV_ICV_URL})
        set(OPENCV_ICV_URL $ENV{OPENCV_ICV_URL})
      else()
        set(OPENCV_ICV_URL "https://raw.githubusercontent.com/opencv/opencv_3rdparty/${IPPICV_BINARIES_COMMIT}/ippicv")
      endif()
    endif()

    file(MAKE_DIRECTORY ${OPENCV_ICV_PACKAGE_ARCHIVE_DIR})
    message(STATUS "ICV: Downloading ${OPENCV_ICV_PACKAGE_NAME}...")
    file(DOWNLOAD "${OPENCV_ICV_URL}/${OPENCV_ICV_PACKAGE_NAME}" "${OPENCV_ICV_PACKAGE_ARCHIVE}"
         TIMEOUT 600 STATUS __status
         EXPECTED_MD5 ${OPENCV_ICV_PACKAGE_HASH})
    if(NOT __status EQUAL 0)
      message(FATAL_ERROR "ICV: Failed to download ICV package: ${OPENCV_ICV_PACKAGE_NAME}. Status=${__status}")
    else()
      # Don't remove this code, because EXPECTED_MD5 parameter doesn't fail "file(DOWNLOAD)" step
      # on wrong hash
      file(MD5 "${OPENCV_ICV_PACKAGE_ARCHIVE}" archive_md5)
      if(NOT archive_md5 STREQUAL OPENCV_ICV_PACKAGE_HASH)
        message(FATAL_ERROR "ICV: Downloaded copy of ICV package has invalid MD5 hash: ${archive_md5} (expected: ${OPENCV_ICV_PACKAGE_HASH})")
      endif()
    endif()
  endif()

  ocv_assert(EXISTS "${OPENCV_ICV_PACKAGE_ARCHIVE}")
  ocv_assert(NOT EXISTS "${OPENCV_ICV_UNPACK_PATH}")
  file(MAKE_DIRECTORY ${OPENCV_ICV_UNPACK_PATH})
  ocv_assert(EXISTS "${OPENCV_ICV_UNPACK_PATH}")

  message(STATUS "ICV: Unpacking ${OPENCV_ICV_PACKAGE_NAME} to ${OPENCV_ICV_UNPACK_PATH}...")
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xz "${OPENCV_ICV_PACKAGE_ARCHIVE}"
                  WORKING_DIRECTORY "${OPENCV_ICV_UNPACK_PATH}"
                  RESULT_VARIABLE __result)

  if(NOT __result EQUAL 0)
    message(FATAL_ERROR "ICV: Failed to unpack ICV package from ${OPENCV_ICV_PACKAGE_ARCHIVE} to ${OPENCV_ICV_UNPACK_PATH} with error ${__result}")
  endif()

  ocv_assert(EXISTS "${OPENCV_ICV_PATH}")

  set(OPENCV_ICV_PACKAGE_DOWNLOADED "${OPENCV_ICV_PACKAGE_HASH}" CACHE INTERNAL "ICV package hash")

  message(STATUS "ICV: Package successfully downloaded")
  set(OPENCV_ICV_PATH "${OPENCV_ICV_PATH}" PARENT_SCOPE)
endfunction()

_icv_downloader()
