# Define the tests for Media Storage and File Format
# MSFF
set(MSFF_TEST_SRCS
  TestAnonymizer
  TestFileChangeTransferSyntax1
  TestFileChangeTransferSyntax2
  TestFileChangeTransferSyntax3
  TestFileChangeTransferSyntax4
  TestFileStreamer1
  TestFileStreamer2
  TestFileStreamer3
  TestFileStreamer4
  TestFileStreamer5
  TestFileStreamer6
  TestFileAnonymizer1
  TestFileAnonymizer2
  TestFileAnonymizer3
  TestFileAnonymizer4
  TestIconImageFilter
  TestIconImageGenerator
  TestIconImageGenerator2
  TestIconImageGenerator3
  TestIconImageGenerator4
  TestParseXPATH
  TestValidate
  TestAnonymizeEvent
  TestFileDerivation
  TestSegmentedPaletteColorLookupTable
  TestPNMCodec
  TestSpacing
  TestSerieHelper
  TestFileExplicitFilter
  TestImageFragmentSplitter
  TestTagPath
  TestOrientation
  TestIconImage
  TestImageHelper
  TestImageToImageFilter
  TestImageChangeTransferSyntax1
  #TestImageChangePhotometricInterpretation
  #TestImageChangePhotometricInterpretation2 # does not compile on mingw...
  TestImageChangeTransferSyntax2
  TestImageChangeTransferSyntax3
  TestImageChangeTransferSyntax4
  # see below
  TestImageChangeTransferSyntax6
  TestImageChangeTransferSyntax7
  TestImageApplyLookupTable
  TestImageChangePlanarConfiguration
  TestCoder
  TestDecoder
  TestRescaler1
  TestRescaler2
  TestDumper
  TestDictPrinter
  TestApplicationEntity
  TestStringFilter1
  TestStringFilter2
  TestStringFilter3
  TestUIDGenerator
  TestUUIDGenerator
  #TestUIDGenerator3
  TestXMLPrinter
  TestPrinter1
  TestPrint
  TestSorter
  TestImageReader
  TestStreamImageReader
  TestImageRegionReader1
  TestImageRegionReader2
  TestImageRegionReader3
  #TestStreamImageWriter
  TestImageReaderRandomEmpty
  TestDirectionCosines
  TestImageWriter
  TestCodec
  TestPDFCodec
  TestRLECodec
  TestAudioCodec
  TestImage
  TestPhotometricInterpretation
  TestLookupTable
  TestOverlay
  TestOverlay3
  TestCurve
  TestCurve2
  TestPixelFormat
  TestPersonName
  TestImageCodec
  TestImageConverter
  TestJPEGCodec
  TestRAWCodec
  TestDICOMDIR
  TestWaveform
  TestFiducials
  TestEncapsulatedDocument
  TestSpectroscopy
  TestSurfaceWriter
  TestSurfaceWriter2
  )

if(GDCM_DATA_ROOT)
  set(MSFF_TEST_SRCS
    ${MSFF_TEST_SRCS}
    TestScanner1
    TestScanner2
    TestPrinter2
    TestIPPSorter
    TestIPPSorter2
    TestIPPSorter3
    TestCopyDataSet
    TestDataElementValueAsSQ
    TestImageWriter2
    TestDICOMDIRGenerator1 # Must be after TestImageChangeTransferSyntax4
    TestDICOMDIRGenerator2 # Must be after TestImageChangeTransferSyntax4
    )
    # Those tests requires that openssl be linked in:
    if(GDCM_USE_SYSTEM_OPENSSL)
      set(MSFF_TEST_SRCS
        ${MSFF_TEST_SRCS}
        TestAnonymizer2
        TestAnonymizer3
        )
    endif()
    if(GDCM_USE_SYSTEM_JSON)
      set(MSFF_TEST_SRCS
        ${MSFF_TEST_SRCS}
        TestJSON1
      )
    endif()
endif()

# gdcmDataExtra
if(GDCM_DATA_EXTRA_ROOT)
  set(MSFF_TEST_SRCS
    ${MSFF_TEST_SRCS}
    TestSplitMosaicFilter
    TestOverlay2
    TestImageRegionReader4
    )
endif()

if(GDCM_USE_JPEGLS)
  set(MSFF_TEST_SRCS
    ${MSFF_TEST_SRCS}
    TestImageChangeTransferSyntax5
    )
endif()

if(GDCM_HAVE_PTHREAD_H)
  set(MSFF_TEST_SRCS
    ${MSFF_TEST_SRCS}
    TestUIDGenerator2
    )
endif()

if(GDCM_PIXEL_SPACING_DATA_ROOT)
  set(MSFF_TEST_SRCS
    ${MSFF_TEST_SRCS}
    TestImageReaderPixelSpacing
    )
endif()

# Add the include paths
include_directories(
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Testing/Source/Data"
  "${GDCM_BINARY_DIR}/Testing/Source/Data"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
  "${GDCM_SOURCE_DIR}/Source/MediaStorageAndFileFormat"
  )

create_test_sourcelist(MSFFTests gdcmMSFFTests.cxx ${MSFF_TEST_SRCS}
  EXTRA_INCLUDE gdcmTestDriver.h
  )
add_executable(gdcmMSFFTests ${MSFFTests})
target_link_libraries(gdcmMSFFTests gdcmMSFF)
if(GDCM_HAVE_PTHREAD_H)
  target_link_libraries(gdcmMSFFTests pthread)
endif()

# Loop over files and create executables
foreach(name ${MSFF_TEST_SRCS})
  add_test(NAME ${name} COMMAND gdcmMSFFTests ${name})
endforeach()
if(GDCM_DATA_ROOT)
  set_tests_properties(TestDICOMDIRGenerator2 PROPERTIES WILL_FAIL ON)
endif()

# We can only run the dcmtk cross-checking test *only* after all the tests have run
# in particular once the TestWriter is done.
foreach(filename ${gdcm_data_filenames_glob})
  get_filename_component(filename_name ${filename} NAME)
  string(REGEX MATCH ${filename_name} bad_dicom ${black_list_reader})
  if(NOT bad_dicom)
    if(GDCM_TEST_DCMTK)
      if(DCMTK_DCMDUMP_EXECUTABLE)
        # -M : load short tags
        # -dc: disable correction
        add_test(NAME "DCMDUMP-${filename_name}" COMMAND ${DCMTK_DCMDUMP_EXECUTABLE} -M -dc
          "${GDCM_TEMP_DIRECTORY}/TestWriter/${filename_name}")
      endif()
    endif()
  endif()
endforeach()

# Repeat for dcdump
foreach(filename ${gdcm_data_filenames_glob})
  get_filename_component(filename_name ${filename} NAME)
  string(REGEX MATCH ${filename_name} bad_dicom ${black_list_reader})
  if(NOT bad_dicom)
    if(GDCM_TEST_DICOM3TOOLS)
      if(DCDUMP_EXECUTABLE)
        add_test(NAME "DCDUMP-${filename_name}" COMMAND "${DCDUMP_EXECUTABLE}"
          "${GDCM_TEMP_DIRECTORY}/TestWriter/${filename_name}")
      endif()
    endif()
  endif()
endforeach()

# There is a new test that compress all images using the jpeg compression alg:
# try to decompress them with dcmtk:
file(MAKE_DIRECTORY "${GDCM_TEMP_DIRECTORY}/TestImageChangeTransferSyntax/dcmdjpeg/")

foreach(filename ${gdcm_data_filenames_glob})
  get_filename_component(filename_name ${filename} NAME)
  string(REGEX MATCH ${filename_name} bad_dicom ${black_list_reader})
  if(NOT bad_dicom)
    if(GDCM_TEST_DCMTK)
      if(DCMTK_DCMDJPEG_EXECUTABLE)
        add_test(NAME "DCMDJPEG-${filename_name}" COMMAND ${DCMTK_DCMDJPEG_EXECUTABLE}
          "${GDCM_TEMP_DIRECTORY}/TestImageChangeTransferSyntax/${filename_name}"
          "${GDCM_TEMP_DIRECTORY}/TestImageChangeTransferSyntax/dcmdjpeg/${filename_name}")
        # Special handling of the DICOMDIR files:
        string(REGEX MATCH ${filename_name} is_dicomdir ${gdcm_data_dicomdir_filenames_glob})
        if(is_dicomdir)
          #message("IS DICOMDIR ${filename_name}")
          set_tests_properties("DCMDJPEG-${filename_name}" PROPERTIES WILL_FAIL TRUE)
        endif()
      endif()
    endif()
  endif()
endforeach()

file(MAKE_DIRECTORY "${GDCM_TEMP_DIRECTORY}/TestImageChangeTransferSyntax3/dcmdrle/")

foreach(filename ${gdcm_data_filenames_glob})
  get_filename_component(filename_name ${filename} NAME)
  string(REGEX MATCH ${filename_name} bad_dicom ${black_list_reader})
  if(NOT bad_dicom)
    if(GDCM_TEST_DCMTK)
      if(DCMTK_DCMDRLE_EXECUTABLE)
        add_test(NAME "DCMDRLE-${filename_name}" COMMAND ${DCMTK_DCMDRLE_EXECUTABLE}
          "${GDCM_TEMP_DIRECTORY}/TestImageChangeTransferSyntax3/${filename_name}"
          "${GDCM_TEMP_DIRECTORY}/TestImageChangeTransferSyntax3/dcmdrle/${filename_name}")
        # Special handling of the DICOMDIR files:
        string(REGEX MATCH ${filename_name} is_dicomdir ${gdcm_data_dicomdir_filenames_glob})
        if(is_dicomdir)
          #message("IS DICOMDIR ${filename_name}")
          set_tests_properties("DCMDRLE-${filename_name}" PROPERTIES WILL_FAIL TRUE)
        endif()
        #
      endif()
    endif()
  endif()
endforeach()

file(MAKE_DIRECTORY "${GDCM_TEMP_DIRECTORY}/TestImageChangeTransferSyntax5/dcmdjpls/")

foreach(filename ${gdcm_data_filenames_glob})
  get_filename_component(filename_name ${filename} NAME)
  string(REGEX MATCH ${filename_name} bad_dicom ${black_list_reader})
  if(NOT bad_dicom)
    if(GDCM_TEST_DCMTK)
      if(DCMTK_DCMDJPLS_EXECUTABLE)
        #if(EXISTS "${GDCM_TEMP_DIRECTORY}/TestImageChangeTransferSyntax5/${filename_name}")
        add_test(NAME "DCMDJPLS-${filename_name}" COMMAND ${DCMTK_DCMDJPLS_EXECUTABLE}
          "${GDCM_TEMP_DIRECTORY}/TestImageChangeTransferSyntax5/${filename_name}"
          "${GDCM_TEMP_DIRECTORY}/TestImageChangeTransferSyntax5/dcmdjpls/${filename_name}")
        # Special handling of the DICOMDIR files:
        string(REGEX MATCH ${filename_name} is_dicomdir ${gdcm_data_dicomdir_filenames_glob})
        if(is_dicomdir)
          #message("IS DICOMDIR ${filename_name}")
          set_tests_properties("DCMDJPLS-${filename_name}" PROPERTIES WILL_FAIL TRUE)
        endif()
        #endif()
        #
      endif()
    endif()
  endif()
endforeach()

# Need to SET_TESTS_PROPERTIES only *after* the loop is done (test need to have been declared first)
if(GDCM_TEST_DCMTK)
  if(GDCM_DATA_ROOT)
    if(DCMTK_DCMDJPLS_EXECUTABLE)
      # No pixel data
      set_tests_properties("DCMDJPLS-ELSCINT1_PMSCT_RLE1.dcm" PROPERTIES WILL_FAIL TRUE)
    endif()
    if(DCMTK_DCMDJPEG_EXECUTABLE)
      # No pixel data
      set_tests_properties("DCMDJPEG-ELSCINT1_PMSCT_RLE1.dcm" PROPERTIES WILL_FAIL TRUE)
    endif()
    if(DCMTK_DCMDRLE_EXECUTABLE)
      # No pixel data
      set_tests_properties("DCMDRLE-ELSCINT1_PMSCT_RLE1.dcm" PROPERTIES WILL_FAIL TRUE)
      # It is missing the very last pixel, so technically is illegal
      set_tests_properties("DCMDRLE-ALOKA_SSD-8-MONO2-RLE-SQ.dcm" PROPERTIES WILL_FAIL TRUE)
    endif()
  endif()
endif()
