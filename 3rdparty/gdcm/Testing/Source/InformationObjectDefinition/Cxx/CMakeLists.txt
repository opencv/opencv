# Define the tests for Information Object Definition
# IOD
set(IOD_TEST_SRCS
  TestTableReader
  TestTable
  TestTableEntry
  TestType
  TestModule
  TestModules
  TestModuleEntry
  TestNestedModuleEntries
  TestIODEntry
  TestIOD
  TestIODs
  TestDefs
  TestPatient
  TestSeries
  TestStudy
  TestDefinedTerms
  TestEnumeratedValues
  TestUsage
  #TestXMLDictReader
  #TestXMLPrivateDictReader
  )

# Add the include paths
include_directories(
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
  "${GDCM_SOURCE_DIR}/Source/InformationObjectDefinition"
  "${GDCM_BINARY_DIR}/Source/InformationObjectDefinition" # gdcmTables.h
  )

create_test_sourcelist(IODTests gdcmIODTests.cxx ${IOD_TEST_SRCS}
  EXTRA_INCLUDE gdcmTestDriver.h
  )
add_executable(gdcmIODTests ${IODTests})
target_link_libraries(gdcmIODTests gdcmIOD gdcmMSFF)

# Loop over files and create executables
foreach(name ${IOD_TEST_SRCS})
  add_test(NAME ${name} COMMAND gdcmIODTests ${name})
endforeach()
