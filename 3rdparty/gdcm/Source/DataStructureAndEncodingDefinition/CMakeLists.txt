# Define the srcs for Data Structure and Encoding Definitions
# DSED
set(DSED_SRCS
  gdcmTag.cxx
  gdcmTagToVR.cxx
  gdcmPrivateTag.cxx
  gdcmCodeString.cxx
  gdcmByteValue.cxx
  gdcmValue.cxx
  gdcmFileSet.cxx
  gdcmDataSet.cxx
  gdcmByteSwapFilter.cxx
  gdcmUNExplicitImplicitDataElement.cxx
  gdcmExplicitDataElement.cxx
  gdcmFileMetaInformation.cxx
  gdcmFragment.cxx
  gdcmImplicitDataElement.cxx
  gdcmItem.cxx
  gdcmReader.cxx
  gdcmWriter.cxx
  #gdcmParser.cxx
  gdcmCSAHeader.cxx
  gdcmPDBHeader.cxx
  gdcmSequenceOfFragments.cxx
  gdcmSequenceOfItems.cxx
  gdcmTransferSyntax.cxx
  gdcmMediaStorage.cxx
  gdcmVM.cxx
  gdcmVR.cxx
  gdcmFile.cxx
  gdcmPreamble.cxx
  gdcmParseException.cxx
  gdcmDataElement.cxx
  gdcmUNExplicitDataElement.cxx
  gdcmCP246ExplicitDataElement.cxx
  gdcmExplicitImplicitDataElement.cxx
  gdcmVR16ExplicitDataElement.cxx
  )

# Add the include paths
include_directories(
  # Bin:
  "${GDCM_BINARY_DIR}/Source/Common"

  # src:
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition/"

  "${GDCM_SOURCE_DIR}/Utilities"
 )

if(NOT GDCM_USE_SYSTEM_ZLIB)
  include_directories(
    "${GDCM_BINARY_DIR}/Utilities/gdcmzlib"
    )
endif()


add_library(gdcmDSED ${DSED_SRCS})
target_link_libraries(gdcmDSED gdcmCommon)
# zlib stuff are actually included (template) so we need to link them here.
target_link_libraries(gdcmDSED ${GDCM_ZLIB_LIBRARIES})
set_target_properties(gdcmDSED PROPERTIES ${GDCM_LIBRARY_PROPERTIES} LINK_INTERFACE_LIBRARIES "gdcmCommon")

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(gdcmDSED PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(gdcmDSED EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

# libs
install_library(gdcmDSED)
# PDB
install_pdb(gdcmDSED)
# include files
install_includes("*.h" "*.txx")
