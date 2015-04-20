# Define the tests for Common
# Common
set(Common_TEST_SRCS
  TestVersion
  TestCommand
  TestCryptographicMessageSyntax
  TestDummyValueGenerator
  TestASN1
  TestTesting
  TestSystem3
  TestSwapper
  TestByteSwap
  TestString1
  TestString2
  TestTerminal
  TestFilenameGenerator
  TestObject
  TestSmartPointer
  TestSwapCode
  TestSystem1
  TestSystem2
  TestTrace
  TestTypes
  TestUnpacker12Bits
  TestBase64
  TestLog2
  )

if(GDCM_DATA_ROOT)
list(APPEND Common_TEST_SRCS
  TestDirectory
  TestFilename
  TestMD5
  )
if(GDCM_USE_SYSTEM_OPENSSL)
  list(APPEND Common_TEST_SRCS TestSHA1)
endif()
endif()

# Add the include paths
include_directories(
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  )

create_test_sourcelist(CommonTests gdcmCommonTests.cxx ${Common_TEST_SRCS}
  EXTRA_INCLUDE gdcmTestDriver.h
  )
add_executable(gdcmCommonTests ${CommonTests})
target_link_libraries(gdcmCommonTests gdcmCommon)

# Loop over files and create executables
foreach(name ${Common_TEST_SRCS})
  add_test(NAME ${name} COMMAND gdcmCommonTests ${name})
endforeach()
