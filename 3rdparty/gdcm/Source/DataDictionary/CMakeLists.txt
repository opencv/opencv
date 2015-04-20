# Define the srcs for Data Dictionary
# DICT

# Add the include paths
include_directories(
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
  "${GDCM_SOURCE_DIR}/Source/InformationObjectDefinition"
  )


#-----------------------------------------------------------------------------
# DICOM dictionary stuff
#set(PREP_DICT_SRCS
#  ${GDCM_SOURCE_DIR}/Source/DataDictionary/gdcmDictConverter.cxx
#  ${GDCM_SOURCE_DIR}/Source/DataDictionary/gdcmPrepDict.cxx
#  )
#
#add_executable(PrepDict ${PREP_DICT_SRCS}
#  "${GDCM_SOURCE_DIR}/Source/Common/gdcmSwapCode.cxx"
#  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition/gdcmVR.cxx"
#  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition/gdcmVM.cxx"
#  )
#
#add_executable(PrepGroupName
#  ${GDCM_SOURCE_DIR}/Source/DataDictionary/gdcmPrepGroupName.cxx
#  )
#
#  get_target_property( PREP_DICT
#    PrepDict
#    LOCATION
#    )
#  get_target_property( PREP_GROUPNAME
#    PrepGroupName
#    LOCATION
#    )
#set(DICT_LIST
#  DICOMV3
#  NIH
#  SPI
#  )
#
## List of file needed to get all dict into c++ code
#set(DICT_SRCS)
#set(DEFAULT_DICTS)
#foreach(dict ${DICT_LIST})
#  set(CXX_DICT ${GDCM_BINARY_DIR}/Source/DataDictionary/gdcm${dict}.cxx)
#  set(DICT_SRCS ${DICT_SRCS} ${CXX_DICT})
#  add_custom_command(
#    OUTPUT    ${CXX_DICT}
#    COMMAND   ${PREP_DICT}
#    ARGS      ${GDCM_SOURCE_DIR}/Source/DataDictionary/${dict}.dic
#              ${CXX_DICT}
#              ${dict}
#    DEPENDS   ${GDCM_SOURCE_DIR}/Source/DataDictionary/${dict}.dic
#              ${PREP_DICT}
#    COMMENT   "Generating gdcm${dict}.cxx based on ${dict}.dic"
#    )
#endforeach()
#
#add_custom_command(
#  OUTPUT    ${GDCM_BINARY_DIR}/Source/DataDictionary/gdcmDefaultGroupNames.cxx
#  COMMAND   ${PREP_GROUPNAME}
#  ARGS      ${GDCM_SOURCE_DIR}/Source/DataDictionary/GroupName.dic
#            ${GDCM_BINARY_DIR}/Source/DataDictionary/gdcmDefaultGroupNames.cxx
#  DEPENDS   ${GDCM_SOURCE_DIR}/Source/DataDictionary/GroupName.dic
#            ${PREP_GROUPNAME}
#  COMMENT   "Generating gdcmDefaultGroupNames.cxx based on GroupName.dic"
#  )
#

# Since the file gdcmDefaultDicts.cxx does not exist, mark it as GENERATED:
#set_source_files_properties(
#  "${GDCM_BINARY_DIR}/Source/gdcmDefaultDicts.cxx" GENERATED)

#configure_file(
#  ${GDCM_SOURCE_DIR}/Source/DataDictionary/gdcmDefaultDicts.cxx
#  ${GDCM_BINARY_DIR}/Source/DataDictionary/gdcmDefaultDicts.cxx
#  @ONLY
#  )

set(DICT_SRCS
  #${DICT_SRCS} # All the dicts cxx resources
  gdcmDicts.cxx
  gdcmDictEntry.cxx
  gdcmDefaultDicts.cxx # pseudo generated file
  gdcmPrivateDefaultDicts.cxx # pseudo generated file
  gdcmGlobal.cxx
  gdcmGroupDict.cxx
  gdcmDefaultGroupNames.cxx
  gdcmUIDs.cxx
  gdcmSOPClassUIDToIOD.cxx
  gdcmCSAHeaderDefaultDicts.cxx
  )

add_library(gdcmDICT ${DICT_SRCS})
set_target_properties(gdcmDICT PROPERTIES ${GDCM_LIBRARY_PROPERTIES} LINK_INTERFACE_LIBRARIES "")
target_link_libraries(gdcmDICT gdcmDSED gdcmIOD)

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(gdcmDICT PROPERTIES FOLDER "3rdparty")
endif()

# libs
install_library(gdcmDICT)
# PDB
install_pdb(gdcmDICT)
# include files
install_includes("*.h")

set(XML_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/CSAHeader.xml
  ${CMAKE_CURRENT_SOURCE_DIR}/Part6.xml
  ${CMAKE_CURRENT_SOURCE_DIR}/Part7.xml
  ${CMAKE_CURRENT_SOURCE_DIR}/UIDs.xml
  ${CMAKE_CURRENT_SOURCE_DIR}/cp699.xml
  )

set_source_files_properties(
  ${XML_FILES}
  PROPERTIES
  MACOSX_PACKAGE_LOCATION Resources
  )

#-----------------------------------------------------------------------------
# Install Part6.xml
install(FILES
  ${XML_FILES}
  DESTINATION ${GDCM_INSTALL_DATA_DIR}/XML COMPONENT Libraries
)
