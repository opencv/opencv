# ----------------------------------------------------------------------------
#  Detect 3rd-party DICOM image IO GDCM library
# ----------------------------------------------------------------------------

# --- GDCM (required) ---
if(BUILD_GDCM)
  ocv_clear_vars(GDCM_FOUND)
  ocv_clear_vars(GDCM_LIBRARY GDCM_VERSION GDCM_LIBRARIES GDCM_INCLUDE_DIRS)

  set(GDCM_LIBRARY gdcmMSFF)
  add_subdirectory("${OpenCV_SOURCE_DIR}/3rdparty/gdcm")
  set(GDCM_VERSION "${GDCM_VERSION}")
  set(GDCM_ROOT_DIR ${GDCM_SOURCE_DIR})
  set(GDCM_INCLUDE_DIRS ${GDCM_INCLUDE_DIRS}
						${GDCM_ROOT_DIR}/Source/Common
						${GDCM_ROOT_DIR}/Source/DataStructureAndEncodingDefinition
						${GDCM_ROOT_DIR}/Source/MediaStorageAndFileFormat
						${CMAKE_BINARY_DIR}/3rdparty/gdcm/Source/Common
						"${GDCM_BINARY_DIR}"
						)
  set(GDCM_LIBRARIES ${GDCM_LIBRARY})

  # GDCM based DICOM decoder is available for reading DICOM image files.
  set(GDCM_FOUND YES)
  set(HAVE_DICOM YES)
endif()
