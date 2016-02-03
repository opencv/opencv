#-----------------------------------------------------------------------------
# GDCM uses KWStyle for checking the coding style
option(GDCM_USE_KWSTYLE "Run KWStyle in order to check for violations of the coding standard." OFF)
mark_as_advanced(GDCM_USE_KWSTYLE)

if(GDCM_USE_KWSTYLE)
  find_package(KWStyle)

  configure_file(${GDCM_SOURCE_DIR}/Utilities/KWStyle/GDCM.kws.xml.in
                 ${GDCM_BINARY_DIR}/Utilities/KWStyle/GDCM.kws.xml)
  #configure_file(${GDCM_SOURCE_DIR}/Utilities/KWStyle/GDCMMoreChecks.kws.xml.in
  #             ${GDCM_BINARY_DIR}/GDCMMoreChecks.kws.xml)

  configure_file(${GDCM_SOURCE_DIR}/Utilities/KWStyle/GDCMFiles.txt.in
                 ${GDCM_BINARY_DIR}/Utilities/KWStyle/GDCMFiles.txt)

  add_custom_command(
    OUTPUT ${GDCM_BINARY_DIR}/KWStyleReport.txt
    COMMAND ${KWSTYLE_EXECUTABLE}
    ARGS    -xml ${GDCM_BINARY_DIR}/Utilities/KWStyle/GDCM.kws.xml -o ${GDCM_SOURCE_DIR}/Utilities/KWStyle/GDCMOverwrite.txt -v ${KWSTYLE_ARGUMENTS} -D ${GDCM_BINARY_DIR}/Utilities/KWStyle/GDCMFiles.txt
    COMMENT "Coding Style Checker"
    )

  #add_custom_target(MoreStyleChecks
  #  COMMAND ${KWSTYLE_EXECUTABLE}
  #  -xml ${GDCM_BINARY_DIR}/GDCMMoreChecks.kws.xml -html ${GDCM_BINARY_DIR}/html -o ${GDCM_SOURCE_DIR}/Utilities/KWStyle/GDCMOverwrite.txt -v ${KWSTYLE_ARGUMENTS} -D ${GDCM_BINARY_DIR}/GDCMKWSFiles.txt
  #  COMMENT "Coding Style Checker, more checks enabled"
  #  )

  add_custom_target(StyleCheck DEPENDS ${GDCM_BINARY_DIR}/KWStyleReport.txt)
endif()
