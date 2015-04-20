# Build the GDCM applications
# Namely:
# gdcmdump
# gdcminfo
# gdcmconv
# gdcmanon


if(WIN32 AND NOT CYGWIN)
  include_directories(
    "${GDCM_SOURCE_DIR}/Utilities/getopt"
  )
endif()

# Add the include paths
include_directories(
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  "${GDCM_SOURCE_DIR}/Source/MediaStorageAndFileFormat"
  "${GDCM_SOURCE_DIR}/Source/InformationObjectDefinition"
  "${GDCM_SOURCE_DIR}/Source/MessageExchangeDefinition"
# FIXME:
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
  "${GDCM_SOURCE_DIR}/Utilities"

  )

if(NOT GDCM_USE_SYSTEM_SOCKETXX)
  include_directories(
    "${GDCM_SOURCE_DIR}/Utilities/socketxx"
    "${GDCM_SOURCE_DIR}/Utilities/socketxx/socket++" # local.h
    "${GDCM_BINARY_DIR}/Utilities/socketxx/socket++" # config.h
    )
endif()

if(WIN32)
  if (BUILD_SHARED_LIBS)
    add_definitions(-DGETOPT_DLL)
  endif ()
endif()

set(GDCM_EXECUTABLE_NAME
  gdcmdump
  gdcmdiff
  gdcmraw
  gdcmscanner
  gdcmanon
  gdcmgendir
  #gdcmoverlay
  gdcmimg
  #deflate
  gdcmconv
  #gdcmstream
  gdcmtar
  gdcminfo
  gdcmscu
  gdcmxml
  gdcmpap3
  )
# poppler people have the worse API backward compatibility I know of.
# there is absolutely no way to check the version of poppler
# they change the API during a minor change of the version
if(GDCM_USE_SYSTEM_POPPLER)
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES ${POPPLER_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${POPPLER_LIBRARIES})
  CHECK_CXX_SOURCE_COMPILES(
    "\#include <poppler/GlobalParams.h>\nint main() { globalParams = new GlobalParams(0); return 0;}"
    LIBPOPPLER_GLOBALPARAMS_CSTOR_HAS_PARAM)
  set(libpoppler_flags)
  if(LIBPOPPLER_GLOBALPARAMS_CSTOR_HAS_PARAM)
    list(APPEND libpoppler_flags -DLIBPOPPLER_GLOBALPARAMS_CSTOR_HAS_PARAM)
  endif()
  CHECK_CXX_SOURCE_COMPILES(
    "\#include <poppler/PDFDoc.h>\nint main() { PDFDoc d((GooString*)NULL,(GooString*)NULL,(GooString*)NULL); d.getPDFVersion(); return 0;}"
    LIBPOPPLER_PDFDOC_HAS_PDFVERSION)
  if(LIBPOPPLER_PDFDOC_HAS_PDFVERSION)
    list(APPEND libpoppler_flags -DLIBPOPPLER_PDFDOC_HAS_PDFVERSION)
  endif()
  CHECK_CXX_SOURCE_COMPILES(
    "\#include <poppler/PDFDoc.h>\n#include <poppler/StructTreeRoot.h>\nint main() { Catalog c(NULL); c.getStructTreeRoot()->getDoc(); return 0;}"
    LIBPOPPLER_CATALOG_HAS_STRUCTTREEROOT)
  if(LIBPOPPLER_CATALOG_HAS_STRUCTTREEROOT)
    list(APPEND libpoppler_flags -DLIBPOPPLER_CATALOG_HAS_STRUCTTREEROOT)
  endif()
  if(libpoppler_flags)
    string(REPLACE ";" " " libpoppler_flags_string "${libpoppler_flags}")
    set_source_files_properties(
      ${CMAKE_CURRENT_SOURCE_DIR}/gdcminfo.cxx
      PROPERTIES COMPILE_FLAGS "${libpoppler_flags_string}")
    set_source_files_properties(
      ${CMAKE_CURRENT_SOURCE_DIR}/gdcmpdf.cxx
      PROPERTIES COMPILE_FLAGS "${libpoppler_flags_string}")
  endif()
  include_directories(${POPPLER_INCLUDE_DIRS})
  set(GDCM_EXECUTABLE_NAME
    ${GDCM_EXECUTABLE_NAME}
    gdcmpdf
  )
endif()

if(GDCM_USE_SYSTEM_LIBXML2)
  include_directories(${LIBXML2_INCLUDE_DIR})
endif()

if(GDCM_USE_SYSTEM_OPENJPEG)
  include_directories(${OPENJPEG_INCLUDE_DIRS} )
else()
  include_directories(
  "${GDCM_BINARY_DIR}/Utilities/gdcmopenjpeg"
    )
endif()

if(NOT BUILD_SHARED_LIBS)
  set_source_files_properties(gdcmstream.cxx
                              PROPERTIES
                              COMPILE_FLAGS -DOPJ_STATIC
                              )
endif()

if(GDCM_USE_SYSTEM_PAPYRUS3)
  include_directories(
    ${PAPYRUS3_INCLUDE_DIRS}
    )
endif()

foreach(exename ${GDCM_EXECUTABLE_NAME})
  if(${exename} STREQUAL "gdcminfo")
    add_executable(${exename} ${exename}.cxx puff.c)
  else()
    add_executable(${exename} ${exename}.cxx)
  endif()
  target_link_libraries(${exename} gdcmMSFF)
  if(${exename} STREQUAL "gdcmpdf")
    target_link_libraries(${exename} ${POPPLER_LIBRARIES})
  elseif(${exename} STREQUAL "gdcmxml")
    target_link_libraries(${exename} ${LIBXML2_LIBRARIES})
  elseif(${exename} STREQUAL "gdcmscu")
    if(NOT GDCM_USE_SYSTEM_SOCKETXX)
      target_link_libraries(${exename} gdcmMEXD socketxx)
    else()
      target_link_libraries(${exename} gdcmMEXD socket++)
    endif()
  elseif(${exename} STREQUAL "gdcmstream")
    target_link_libraries(${exename} ${GDCM_OPENJPEG_LIBRARIES})
  elseif(${exename} STREQUAL "gdcminfo")
    if(GDCM_USE_SYSTEM_POPPLER)
      target_link_libraries(${exename} ${POPPLER_LIBRARIES})
    endif()
  endif()
  if(GDCM_EXECUTABLE_PROPERTIES)
    set_target_properties(${exename} PROPERTIES ${GDCM_EXECUTABLE_PROPERTIES})
  endif()
  if(WIN32 AND NOT CYGWIN)
    target_link_libraries(${exename} gdcmgetopt)
  endif()
  if(NOT GDCM_INSTALL_NO_RUNTIME)
    install(TARGETS ${exename}
      EXPORT ${GDCM_TARGETS_NAME}
      RUNTIME DESTINATION ${GDCM_INSTALL_BIN_DIR} COMPONENT Applications
    )
  endif()
endforeach()

if(GDCM_USE_SYSTEM_PAPYRUS3)
  target_link_libraries(gdcmpap3 ${PAPYRUS3_LIBRARIES})
endif()

#if(BUILD_TESTING)
#  if(GDCM_DATA_ROOT)
#    file(MAKE_DIRECTORY ${GDCM_TEMP_DIRECTORY}/gdcmanon-gdcmdata)
#    add_test(NAME gdcmanon-gdcmdata COMMAND gdcmanon
#      --continue  # skip LIBIDO-16-ACR_NEMA-Volume.dcm
#      --certificate ${GDCM_SOURCE_DIR}/Testing/Source/Data/certificate.pem
#      ${GDCM_DATA_ROOT}
#      ${GDCM_TEMP_DIRECTORY}/gdcmanon-gdcmdata
#      )
#  endif()
#endif()

if(BUILD_TESTING)
  # http://www.na-mic.org/Wiki/index.php/CTSC:ARRA:Mockup
  # http://www.dicomserver.co.uk/
  # the NAMIC server is offline, Steve Pieper has volunteered his servers, but they are intermittent
  # IP Address: common.bwh.harvard.edu (155.52.248.49)
  # Port: 11112
  # AE Title: CTK_AE
  # and
  # IP Address: joe.bwh.harvard.edu (155.52.248.50)
  # Port: 5678
  # AE Title: CONQUESTSRV1
  #since these servers have no guaranteed uptime, these tests should be changed to some other, known-to-be-up server
  #add_test(NAME gdcmscu-echo-CONQUEST COMMAND gdcmscu --echo --call CONQUESTSRV1 joe.bwh.harvard.edu 5678)
  add_test(NAME gdcmscu-echo-dicomserver COMMAND gdcmscu --echo www.dicomserver.co.uk 11112)
  if(GDCM_DATA_ROOT)
    # CR-MONO1-10-chest.dcm gets rejected by www.dicomserver.co.uk:
    # Tag 0x0,0x902 reported as Incorrect VM (1) - Minimum value is 2 : 'Array of N Elements' (error 42752)
    # CR-MONO1-10-chest.dcm gets rejected by mi2b2.slicer.org
    # Tag 0x0,0x902 reported as Acceptance of objects without Patient ID is disabled (error 43264)
    #add_test(NAME gdcmscu-store-CONQUEST COMMAND gdcmscu --store --call CONQUESTSRV1 joe.bwh.harvard.edu 11112 ${GDCM_DATA_ROOT}/SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm)
    add_test(NAME gdcmscu-store-dicomserver COMMAND gdcmscu --store www.dicomserver.co.uk 11112 ${GDCM_DATA_ROOT}/SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm)
  endif()
  #add_test(NAME gdcmscu-find-CONQUEST COMMAND gdcmscu --find --patient --call CONQUESTSRV1 joe.bwh.harvard.edu 11112 --patientroot -k 10,10=X*)
  add_test(NAME gdcmscu-find-dicomserver COMMAND gdcmscu --find --patient www.dicomserver.co.uk 11112 --patientroot -k 10,20=*)

  if(GDCM_DICOM_SERVER_AETITLE)
  if(GDCM_DATA_ROOT)
    # Let's C-STORE a limited subset of gdcmData for now:
    set(CSTORE_DATA_FILES
      CR-MONO1-10-chest.dcm # Implicit VR Little Endian: Default Transfer Syntax for DICOM
      012345.002.050.dcm    # JPEG Lossless, Non-Hierarchical, First-Order Prediction
      )
    foreach(cstorefile ${CSTORE_DATA_FILES})
    add_test(NAME gdcmscu-echo-${cstorefile} COMMAND gdcmscu --echo  --aetitle ${GDCM_DICOM_CLIENT_AETITLE} --call ${GDCM_DICOM_SERVER_AETITLE} ${GDCM_DICOM_SERVER_PEER} ${GDCM_DICOM_SERVER_PORT})
    add_test(NAME gdcmscu-store-${cstorefile} COMMAND gdcmscu --store --aetitle ${GDCM_DICOM_CLIENT_AETITLE} --call ${GDCM_DICOM_SERVER_AETITLE} ${GDCM_DICOM_SERVER_PEER} ${GDCM_DICOM_SERVER_PORT} ${GDCM_DATA_ROOT}/${cstorefile})
    add_test(NAME gdcmscu-find-${cstorefile} COMMAND gdcmscu --find --patient --aetitle ${GDCM_DICOM_CLIENT_AETITLE} --call ${GDCM_DICOM_SERVER_AETITLE} ${GDCM_DICOM_SERVER_PEER} ${GDCM_DICOM_SERVER_PORT} --patientroot -k 10,10=A* -k 10,20)
    add_test(NAME gdcmscu-move1-${cstorefile} COMMAND gdcmscu --move  --patient --aetitle ${GDCM_DICOM_CLIENT_AETITLE} --call ${GDCM_DICOM_SERVER_AETITLE} ${GDCM_DICOM_SERVER_PEER} ${GDCM_DICOM_SERVER_PORT} -o ${CMAKE_CURRENT_BINARY_DIR} --patientroot -k 10,10=Anonymized --port-scp ${GDCM_DICOM_CLIENT_PORT})
    add_test(NAME gdcmscu-move2-${cstorefile} COMMAND gdcmscu --move  --patient --aetitle ${GDCM_DICOM_CLIENT_AETITLE} --call ${GDCM_DICOM_SERVER_AETITLE} ${GDCM_DICOM_SERVER_PEER} ${GDCM_DICOM_SERVER_PORT} -o ${CMAKE_CURRENT_BINARY_DIR} --patientroot -k 10,10=X* --port-scp ${GDCM_DICOM_CLIENT_PORT})
    endforeach()
  endif()
  endif()

endif()
