# Define the tests for gdcm-python
# gdcm-python
set(GDCM_PYTHON_TEST_SRCS
  TestScanner
  TestImageReader
  TestUIDGenerator
  TestModifyFields
  TestAnonymizer
  #TestStringFilter
  TestOrientation
  TestIPPSorter
  #TestPythonFilter
  )
if(BUILD_APPLICATIONS)
  if(DCMTK_DCMDUMP_EXECUTABLE)
  if(UNIX)
    set(GDCM_PYTHON_TEST_SRCS
      ${GDCM_PYTHON_TEST_SRCS}
      TestDCMTKMD5 # need dcmdrle and dcmdjpeg
    )
  endif()
  endif()
endif()

# Loop over files and create executables
foreach(name ${GDCM_PYTHON_TEST_SRCS})
  ADD_PYTHON_TEST(${name}Python ${name}.py)
endforeach()

# Special test that need extra arg:
# TestKakaduDecompressionMD5
find_package(KAKADU)
if(BUILD_APPLICATIONS)
  if(KDU_EXPAND_EXECUTABLE)
  if(UNIX)
    ADD_PYTHON_TEST(TestKakaduDecompressionMD5Python TestKakaduDecompressionMD5.py ${KDU_EXPAND_EXECUTABLE})
  endif()
  endif()
endif()
