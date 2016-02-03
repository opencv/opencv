# Required:
# sudo apt-get install libperl-dev

# http://www.swig.org/Doc2.0/Perl5.html
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

include_directories(
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  "${GDCM_SOURCE_DIR}/Source/InformationObjectDefinition"
  "${GDCM_SOURCE_DIR}/Source/MediaStorageAndFileFormat"
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
)

find_package(PerlLibs REQUIRED)
include_directories(
  ${PERL_INCLUDE_PATH}
)

set_source_files_properties(gdcm.i PROPERTIES CPLUSPLUS ON)

# TODO check convetion for perl module name:
SWIG_ADD_MODULE(perl_gdcm perl gdcm.i)
SWIG_LINK_LIBRARIES(perl_gdcm gdcmMSFF)

#set_target_properties(${SWIG_MODULE_perl_gdcm_REAL_NAME} PROPERTIES LINK_INTERFACE_LIBRARIES "")
# Remove 'lib' prefix :no-op on windows and valid for UNIX based syste
set_target_properties(${SWIG_MODULE_perl_gdcm_REAL_NAME} PROPERTIES PREFIX "")
set_property(TARGET ${SWIG_MODULE_perl_gdcm_REAL_NAME} PROPERTY NO_SONAME 1)

if(UNIX)
  set_target_properties(${SWIG_MODULE_perl_gdcm_REAL_NAME} PROPERTIES OUTPUT_NAME "gdcm")
endif()


# Let's copy gdcm.pm into the bin dir:
#add_custom_command(
#  OUTPUT ${LIBRARY_OUTPUT_PATH}/gdcm.pm
#  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/gdcm.php ${LIBRARY_OUTPUT_PATH}/gdcm.pm
#  DEPENDS "${swig_generated_file_fullname}"
#  COMMENT "copying gdcm.pm"
#)
#add_custom_target(GDCMPERL ALL
#  DEPENDS ${LIBRARY_OUTPUT_PATH}/gdcm.pm
#  COMMENT "building gdcm.pm"
#)

# Module are always place in the library destination
# but for poor win32 user I decided to place them
# right next to the other dlls
if(NOT GDCM_INSTALL_NO_LIBRARIES)
  install_swig_module(perl_gdcm Perl)
  # because gdcm.pm is constructed with custom commands, it need the install(FILES signature:
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gdcm.pm
    DESTINATION ${GDCM_INSTALL_LIB_DIR} COMPONENT PerlModule
  )
endif()
