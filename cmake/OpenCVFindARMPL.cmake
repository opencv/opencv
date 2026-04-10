if(NOT AARCH64 AND NOT ARM64 AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    return()
endif()

if(NOT WITH_ARMPL)
    return()
endif()

set(ARMPL_ROOT_DIR "" CACHE PATH "Path to ARM Performance Libraries root directory")

function(ocv_armpl_download)
  set(ARMPL_PACKAGE_DIR "${OpenCV_BINARY_DIR}/3rdparty/armpl")
  if(EXISTS "${OpenCV_SOURCE_DIR}/3rdparty/armpl/armpl.cmake")
    include("${OpenCV_SOURCE_DIR}/3rdparty/armpl/armpl.cmake")
  else()
    message(STATUS "ARM Performance Libraries: Download configuration not found")
    return()
  endif()

  if(NOT DEFINED ARMPL_DOWNLOAD_URL)
    return()
  endif()

  set(ARMPL_DOWNLOADED_FILE "${ARMPL_PACKAGE_DIR}/${ARMPL_BINARIES_ARCHIVE}")
  ocv_download(
    FILENAME        ${ARMPL_BINARIES_ARCHIVE}
    HASH            ${ARMPL_DOWNLOAD_HASH}
    URL             ${ARMPL_DOWNLOAD_URL}
    DESTINATION_DIR ${ARMPL_PACKAGE_DIR}
    ID              ARMPL
    STATUS          download_status
    RELATIVE_URL
  )

    if(NOT download_status)
    message(WARNING "ARM Performance Libraries: Download failed")
    return()
    endif()

    set(ARMPL_DOWNLOADED_FILE "${ARMPL_PACKAGE_DIR}/${ARMPL_BINARIES_ARCHIVE}")

  if(WIN32 AND ARMPL_BINARIES_ARCHIVE MATCHES "\\.msi$")
    set(ARMPL_EXTRACT_DIR "${ARMPL_PACKAGE_DIR}/${ARMPL_SUBDIR}")
    file(GLOB_RECURSE EXISTING_ARMPL_H "${ARMPL_EXTRACT_DIR}/**/armpl.h")
    if(EXISTING_ARMPL_H)
      list(GET EXISTING_ARMPL_H 0 FIRST_ARMPL_H)
      get_filename_component(ARMPL_INCLUDE_FOUND "${FIRST_ARMPL_H}" DIRECTORY)
      get_filename_component(FOUND_ARMPL_DIR "${ARMPL_INCLUDE_FOUND}" DIRECTORY)
      set(ARMPL_ROOT_DIR "${FOUND_ARMPL_DIR}" PARENT_SCOPE)
      return()
    endif()

    if(EXISTS "${ARMPL_EXTRACT_DIR}")
      file(REMOVE_RECURSE "${ARMPL_EXTRACT_DIR}")
    endif()
    file(MAKE_DIRECTORY "${ARMPL_EXTRACT_DIR}")
    get_filename_component(ARMPL_DOWNLOADED_FILE_ABS "${ARMPL_DOWNLOADED_FILE}" ABSOLUTE)
    get_filename_component(ARMPL_EXTRACT_DIR_ABS "${ARMPL_EXTRACT_DIR}" ABSOLUTE)
    file(TO_NATIVE_PATH "${ARMPL_DOWNLOADED_FILE_ABS}" ARMPL_MSI_NATIVE_PATH)
    file(TO_NATIVE_PATH "${ARMPL_EXTRACT_DIR_ABS}" ARMPL_TARGET_NATIVE_PATH)

    set(LESSMSI_DIR "${CMAKE_BINARY_DIR}/3rdparty/lessmsi")
    set(LESSMSI_EXE "${LESSMSI_DIR}/lessmsi.exe")

    find_program(LESSMSI_EXE_FOUND lessmsi)
    if(LESSMSI_EXE_FOUND)
      set(LESSMSI_EXE "${LESSMSI_EXE_FOUND}")
    elseif(NOT EXISTS "${LESSMSI_EXE}")
      message(STATUS "ARM Performance Libraries: Downloading lessmsi...")
      set(LESSMSI_ZIP "${LESSMSI_DIR}/lessmsi-v1.10.0.zip")
      file(MAKE_DIRECTORY "${LESSMSI_DIR}")
      ocv_download(
        FILENAME        "lessmsi-v1.10.0.zip"
        HASH            "230e7528132c98526d6d50490ddaa132"
        URL             "https://github.com/activescott/lessmsi/releases/download/v1.10.0/"
        DESTINATION_DIR "${LESSMSI_DIR}"
        ID              LESSMSI
        STATUS          LESSMSI_DOWNLOAD_STATUS
        RELATIVE_URL
      )
      if(NOT LESSMSI_DOWNLOAD_STATUS)
        message(WARNING
          "ARM Performance Libraries: Failed to download lessmsi. "
          "Please extract the MSI manually and set -DARMPL_ROOT_DIR=<path>")
        return()
      endif()

      file(ARCHIVE_EXTRACT INPUT "${LESSMSI_ZIP}" DESTINATION "${LESSMSI_DIR}")
      if(NOT EXISTS "${LESSMSI_DIR}/lessmsi.exe")
        message(WARNING
          "ARM Performance Libraries: Failed to unzip lessmsi. "
          "Please extract the MSI manually and set -DARMPL_ROOT_DIR=<path>")
        return()
      endif()
    endif()

    if(NOT EXISTS "${LESSMSI_EXE}")
      message(WARNING
        "ARM Performance Libraries: lessmsi.exe not found. "
        "Please extract the MSI manually and set -DARMPL_ROOT_DIR=<path>")
      return()
    endif()

    message(STATUS "ARM Performance Libraries: Extracting MSI...")
    execute_process(
      COMMAND         "${LESSMSI_EXE}" x "${ARMPL_MSI_NATIVE_PATH}" "${ARMPL_TARGET_NATIVE_PATH}\\"
      RESULT_VARIABLE EXTRACTION_RESULT
      OUTPUT_QUIET
      ERROR_VARIABLE  EXTRACTION_ERROR
      TIMEOUT         600
    )
    if(NOT EXTRACTION_RESULT EQUAL 0)
      message(WARNING
        "ARM Performance Libraries: MSI extraction failed. "
        "Error: ${EXTRACTION_ERROR}. "
        "Please extract manually and set -DARMPL_ROOT_DIR=<path>")
      return()
    endif()

    file(GLOB_RECURSE ARMPL_H_CHECK "${ARMPL_EXTRACT_DIR}/**/armpl.h")
    if(NOT ARMPL_H_CHECK)
      message(WARNING
        "ARM Performance Libraries: armpl.h not found after extraction. "
        "Please extract the MSI manually to ${ARMPL_EXTRACT_DIR} and set -DARMPL_ROOT_DIR=<path>")
      return()
    endif()

    list(GET ARMPL_H_CHECK 0 FIRST_ARMPL_H)
    get_filename_component(ARMPL_INCLUDE_FOUND "${FIRST_ARMPL_H}" DIRECTORY)
    get_filename_component(FOUND_ARMPL_DIR "${ARMPL_INCLUDE_FOUND}" DIRECTORY)
    if(EXISTS "${FOUND_ARMPL_DIR}/lib")
      set(ARMPL_ROOT_DIR "${FOUND_ARMPL_DIR}" PARENT_SCOPE)
    else()
      message(WARNING "ARM Performance Libraries: lib directory missing at ${FOUND_ARMPL_DIR}/lib")
    endif()
  endif()
endfunction()

if(NOT ARMPL_ROOT_DIR OR NOT EXISTS "${ARMPL_ROOT_DIR}")
    ocv_armpl_download()
endif()

if(NOT ARMPL_ROOT_DIR OR NOT EXISTS "${ARMPL_ROOT_DIR}")
    return()
endif()

set(ARMPL_INCLUDE_DIR "${ARMPL_ROOT_DIR}/include")
if(NOT EXISTS "${ARMPL_INCLUDE_DIR}")
    set(ARMPL_INCLUDE_DIR "${ARMPL_ROOT_DIR}/include-ilp64")
    if(NOT EXISTS "${ARMPL_INCLUDE_DIR}")
        return()
    endif()
endif()

if(NOT EXISTS "${ARMPL_INCLUDE_DIR}/armpl.h")
    return()
endif()

set(ARMPL_LIBRARY_DIR "${ARMPL_ROOT_DIR}/lib")
if(NOT EXISTS "${ARMPL_LIBRARY_DIR}")
    return()
endif()

set(ARMPL_LIB_CANDIDATES
    "armpl_lp64_mp"
    "armpl_ilp64_mp"
)

set(ARMPL_LIB_FOUND FALSE)
foreach(lib_candidate ${ARMPL_LIB_CANDIDATES})
    if(WIN32)
        set(ARMPL_LIB_FILE "${ARMPL_LIBRARY_DIR}/${lib_candidate}.lib")
        set(ARMPL_DLL_FILE "${ARMPL_ROOT_DIR}/bin/${lib_candidate}.dll")
    else()
        set(ARMPL_LIB_FILE "${ARMPL_LIBRARY_DIR}/lib${lib_candidate}.a")
    endif()
    if(EXISTS "${ARMPL_LIB_FILE}")
        set(ARMPL_LIB_NAME ${lib_candidate})
        set(ARMPL_LIB_FOUND TRUE)
        break()
    endif()
endforeach()

if(NOT ARMPL_LIB_FOUND)
    return()
endif()

string(REGEX MATCH "armpl[_-]([0-9]+\\.[0-9]+)" ARMPL_VERSION_MATCH "${ARMPL_ROOT_DIR}")
if(ARMPL_VERSION_MATCH)
    string(REGEX REPLACE "armpl[_-]" "" ARMPL_VERSION_STR "${ARMPL_VERSION_MATCH}")
else()
    set(ARMPL_VERSION_STR "unknown")
endif()

if(NOT TARGET armpl)
    if(WIN32)
        add_library(armpl SHARED IMPORTED)
        set_target_properties(armpl PROPERTIES
            IMPORTED_IMPLIB "${ARMPL_LIB_FILE}"
            IMPORTED_LOCATION "${ARMPL_DLL_FILE}"
            INTERFACE_INCLUDE_DIRECTORIES "${ARMPL_INCLUDE_DIR}"
        )
    else()
        add_library(armpl UNKNOWN IMPORTED)
        set_target_properties(armpl PROPERTIES
            IMPORTED_LOCATION "${ARMPL_LIB_FILE}"
            INTERFACE_INCLUDE_DIRECTORIES "${ARMPL_INCLUDE_DIR}"
        )
    endif()
endif()

if(WIN32)
    set(ARMPL_LIBOMP_PATH "${ARMPL_LIBRARY_DIR}/libomp.dll.lib")
    if(EXISTS "${ARMPL_LIBOMP_PATH}")
        set_target_properties(armpl PROPERTIES
            INTERFACE_LINK_LIBRARIES "${ARMPL_LIBOMP_PATH}"
        )
        file(TO_NATIVE_PATH "${ARMPL_LIBRARY_DIR}" ARMPL_LIB_DIR_NATIVE)
        set(ARMPL_LINKER_FLAGS "/LIBPATH:\"${ARMPL_LIB_DIR_NATIVE}\"")
        if(CMAKE_SHARED_LINKER_FLAGS)
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${ARMPL_LINKER_FLAGS}" CACHE STRING "Linker flags for shared libraries" FORCE)
        else()
            set(CMAKE_SHARED_LINKER_FLAGS "${ARMPL_LINKER_FLAGS}" CACHE STRING "Linker flags for shared libraries" FORCE)
        endif()
    else()
        message(WARNING "ARM Performance Libraries: libomp.dll.lib not found at ${ARMPL_LIBOMP_PATH}")
    endif()
endif()

set(ARMPL_LIBRARIES armpl CACHE INTERNAL "ArmPL libraries")
set(ARMPL_INCLUDE_DIRS "${ARMPL_INCLUDE_DIR}" CACHE INTERNAL "ArmPL include directories")
set(HAVE_ARMPL TRUE CACHE BOOL "ArmPL found and enabled" FORCE)
set(ARMPL_VERSION_STR "${ARMPL_VERSION_STR}" CACHE INTERNAL "ArmPL version")
set(ARMPL_LIB_NAME "${ARMPL_LIB_NAME}" CACHE INTERNAL "ArmPL library variant")

message(STATUS "ARM Performance Libraries: ENABLED (v${ARMPL_VERSION_STR}, ${ARMPL_LIB_NAME})")
