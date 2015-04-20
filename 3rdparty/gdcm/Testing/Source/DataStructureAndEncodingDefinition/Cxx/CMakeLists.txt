# Define the tests for Data Structure and Encoding Definitions
# DSED
set(DSED_TEST_SRCS
  TestAttribute
  TestCodeString
  TestAttribute1
  TestAttribute7
  #TestReadPatientName
  TestLO
  TestCSAElement
  #TestByteBuffer
  #TestByteValue
  TestPreamble
  TestReader
  TestReader4
  TestReaderUpToTag1
  TestReaderUpToTag2
  TestReaderSelectedTags
  TestReaderSelectedPrivateGroups
  TestReaderCanRead
  TestWriter
  TestWriter2
  TestCSAHeader
  TestByteSwapFilter
  TestBasicOffsetTable
  TestComposite
  TestDataElement
  TestDataSet
  #TestVRDS
  TestDS
  TestVRLT
  TestVRUI
  TestElement1
  #TestExplicitDataElement
  TestFileMetaInformation
  TestFile
  TestFileSet
  TestFragment
  #TestImplicitDataElement
  TestItem
  #TestParser
  TestSequenceOfFragments
  TestSequenceOfItems
  TestTag
  TestPrivateTag
  TestTransferSyntax
  TestMediaStorage
  TestVL
  TestVM
  TestVR
  #TestValue
  #TestTorture
  TestElement2
  TestElement4
  TestElement5
  TestElement6
  )
if(GDCM_TESTING_USE_LC_NUMERIC)
# The test expect to be able to set fr as locale only turn if user
# set the appropriate & hidden cmake variable
set(DSED_TEST_SRCS
  ${DSED_TEST_SRCS}
  TestLCNumeric
)
endif()
if(GDCM_DATA_ROOT)
set(DSED_TEST_SRCS
  ${DSED_TEST_SRCS}
  TestReader2
  TestAttribute8
  TestPDBHeader
  TestSequenceOfItems2
  TestSequenceOfItems3
  )
endif()
if(GDCM_DATA_EXTRA_ROOT)
set(DSED_TEST_SRCS
  ${DSED_TEST_SRCS}
  TestInvalidDICOMFiles
  )
endif()

CHECK_INCLUDE_FILE("sys/mman.h"   GDCM_HAVE_SYS_MMAN_H)

# Need mmap
if(GDCM_HAVE_SYS_MMAN_H)
  set(DSED_TEST_SRCS ${DSED_TEST_SRCS}
    TestReader3
  )
endif()

option(SHARED_PTR "shared_ptr" OFF)
mark_as_advanced(SHARED_PTR)
if(SHARED_PTR)
  set(DSED_TEST_SRCS
    ${DSED_TEST_SRCS}
    TestCopyValue
    )
endif()

# Add the include paths
include_directories(
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  "${GDCM_BINARY_DIR}/Utilities/zlib"
  "${GDCM_SOURCE_DIR}/Utilities"
  )

create_test_sourcelist(DSEDTests gdcmDSEDTests.cxx ${DSED_TEST_SRCS}
  EXTRA_INCLUDE gdcmTestDriver.h
  )
add_executable(gdcmDSEDTests ${DSEDTests})
target_link_libraries(gdcmDSEDTests gdcmDSED)
if(CMAKE_COMPILER_IS_GNUCXX AND MINGW)
  set_target_properties( gdcmDSEDTests PROPERTIES LINK_FLAGS "-Wl,--allow-multiple-definition")
endif()

# Loop over files and create executables
foreach(name ${DSED_TEST_SRCS})
  add_test(NAME ${name} COMMAND gdcmDSEDTests ${name})
endforeach()

# FIXME
#add_executable(TestAttribute2 TestAttribute2.cxx)
#add_executable(TestAttribute3 TestAttribute3.cxx)
#add_executable(TestAttribute4 TestAttribute4.cxx)
#add_executable(TestAttribute5 TestAttribute5.cxx)
