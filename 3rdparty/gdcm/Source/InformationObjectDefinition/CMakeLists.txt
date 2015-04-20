# Define the srcs for the Information Object Definition
# IOD
set(IOD_SRCS
  gdcmModule.cxx
  gdcmMacro.cxx
  gdcmModules.cxx
  gdcmMacros.cxx
  gdcmNestedModuleEntries.cxx
  gdcmIODEntry.cxx
  gdcmTableReader.cxx
  gdcmTable.cxx
  gdcmSeries.cxx
  gdcmDefs.cxx
  gdcmDefinedTerms.cxx
  gdcmEnumeratedValues.cxx
  gdcmStudy.cxx
  gdcmPatient.cxx
  gdcmType.cxx
  gdcmUsage.cxx
  gdcmIOD.cxx
  #gdcmXMLDictReader.cxx
  #gdcmXMLPrivateDictReader.cxx
  )

# expat stuff
if(NOT BUILD_SHARED_LIBS)
  set_source_files_properties(gdcmTableReader.cxx
                              PROPERTIES
                              COMPILE_FLAGS -DXML_STATIC
                              )
endif()

# Add the include paths
include_directories(
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
  "${GDCM_SOURCE_DIR}/Utilities"
  )
if(GDCM_USE_SYSTEM_EXPAT)
  include_directories(
    ${EXPAT_INCLUDE_DIRS}
  )
endif()

add_library(gdcmIOD ${IOD_SRCS})
target_link_libraries(gdcmIOD gdcmDSED gdcmCommon ${GDCM_EXPAT_LIBRARIES})
set_target_properties(gdcmIOD PROPERTIES ${GDCM_LIBRARY_PROPERTIES} LINK_INTERFACE_LIBRARIES "gdcmDSED;gdcmCommon")

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(gdcmIOD PROPERTIES FOLDER "3rdparty")
endif()

# libs
install_library(gdcmIOD)
# PDB
install_pdb(gdcmIOD)
# include files
install_includes("*.h" "*.txx")

#-----------------------------------------------------------------------------
# Install Part3.xml / Part4.xml
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/Part3.xml
              ${CMAKE_CURRENT_SOURCE_DIR}/Part4.xml
  DESTINATION ${GDCM_INSTALL_DATA_DIR}/XML COMPONENT Libraries
)

#-----------------------------------------------------------------------------
file(GLOB GDCM_XML_TABLES_GLOB
  "${CMAKE_CURRENT_SOURCE_DIR}/*.xml"
  )

set(GDCM_XML_TABLES)
foreach(filename ${GDCM_XML_TABLES_GLOB})
  set(GDCM_XML_TABLES "${GDCM_XML_TABLES}\n\"${filename}\",")
endforeach()

# Populate GDCM_DATA_IMAGES:
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/gdcmTables.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/gdcmTables.h"
  )
