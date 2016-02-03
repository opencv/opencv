# Build the GDCM examples

# Add the include paths
include_directories(
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  "${GDCM_SOURCE_DIR}/Source/MediaStorageAndFileFormat"
  "${GDCM_SOURCE_DIR}/Source/InformationObjectDefinition"
  "${GDCM_SOURCE_DIR}/Source/MessageExchangeDefinition"

# FIXME:
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
  "${GDCM_SOURCE_DIR}/Utilities"
  )
# error C1083: Cannot open include file: 'zconf.h'
if(NOT GDCM_USE_SYSTEM_ZLIB)
include_directories(
  "${GDCM_BINARY_DIR}/Utilities/gdcmzlib"
)
endif()

if(GDCM_BINARY_DIR)
find_package(SQLITE3)

find_package(MAGIC)

find_package(Qt4 4.4.3 COMPONENTS QtCore QtGui)
# cmake 2.6.4 does not set it to advanced
mark_as_advanced(QT_QMAKE_EXECUTABLE)

endif()

set(EXAMPLES_SRCS
  DumpGEMSMovieGroup
  QIDO-RS
  DumpExamCard
  DumpPhilipsECHO
  ExtractIconFromFile
  CreateJPIPDataSet
  DumpADAC
  DumpImageHeaderInfo
  ReadMultiTimesException
  pmsct_rgb1
  GenAllVR
  ELSCINT1WaveToText
  DiscriminateVolume
  GetSubSequenceData
  GenLongSeqs
  ReadGEMSSDO
  DiffFile
  SimpleScanner
  TraverseModules
  GetSequenceUltrasound
  ReadAndPrintAttributes
  ChangeSequenceUltrasound
  ChangePrivateTags
  LargeVRDSExplicit
  ExtractEncryptedContent
  ReadAndDumpDICOMDIR
  GenerateStandardSOPClasses
  ClinicalTrialAnnotate
  CheckBigEndianBug
  DuplicatePCDE
  CreateARGBImage
  CreateCMYKImage
  CompressImage
  FixBrokenJ2K
  GenFakeImage
  GenFakeIdentifyFile
  HelloVizWorld
  HelloWorld
  MergeTwoFiles
  MrProtocol
  PatchFile
  ReadExplicitLengthSQIVR
  SortImage
  csa2img
  iU22tomultisc
  rle2img
  uid_unique
  PublicDict
  GenSeqs
  GetJPEGSamplePrecision
)
if(GDCM_USE_OPENJPEG_V2)
set(EXAMPLES_SRCS
  ${EXAMPLES_SRCS}
  StreamImageReaderTest
  Extracting_All_Resolution
  Fake_Image_Using_Stream_Image_Writer
  )
endif()
if(GDCM_USE_JPEGLS)
set(EXAMPLES_SRCS
  ${EXAMPLES_SRCS}
  FixJAIBugJPEGLS
  )
endif()

if(QT4_FOUND)
  include(${QT_USE_FILE})
  add_executable(ConvertToQImage ConvertToQImage.cxx)
  # qtGUI will resolved QImage symbol
  # qtcore will resolve qstring symbols
  target_link_libraries(ConvertToQImage gdcmMSFF ${QT_LIBRARIES})

  add_executable(ReadUTF8QtDir ReadUTF8QtDir.cxx)
  target_link_libraries(ReadUTF8QtDir gdcmMSFF ${QT_LIBRARIES})
  add_executable(CStoreQtProgress CStoreQtProgress.cxx)
  target_link_libraries(CStoreQtProgress gdcmMEXD gdcmMSFF ${QT_LIBRARIES})
endif()

if(MAGIC_FOUND)
  include_directories(${MAGIC_INCLUDE_DIRS})
  add_executable(EncapsulateFileInRawData EncapsulateFileInRawData.cxx)
  target_link_libraries(EncapsulateFileInRawData gdcmMSFF ${MAGIC_LIBRARIES})
endif()

if(SQLITE3_FOUND)
  include_directories(${SQLITE3_INCLUDE_DIRS})
  add_executable(DumpToSQLITE3 DumpToSQLITE3.cxx)
  target_link_libraries(DumpToSQLITE3 gdcmMSFF ${SQLITE3_LIBRARIES})
endif()

if(GDCM_BUILD_TESTING)
set(EXAMPLES_SRCS
  ${EXAMPLES_SRCS}
  VolumeSorter
)
endif()

if(BUILD_SHARED_LIBS)
  set_source_files_properties(FixJAIBugJPEGLS.cxx
                              PROPERTIES
                              COMPILE_FLAGS -DCHARLS_SHARED
                              )
endif()

foreach(example ${EXAMPLES_SRCS})
  add_executable(${example} ${example}.cxx)
  if(${example} STREQUAL "FixJAIBugJPEGLS")
    target_link_libraries(${example} gdcmMSFF ${GDCM_CHARLS_LIBRARIES})
  elseif(${example} STREQUAL "DumpPhilipsECHO")
    target_link_libraries(${example} gdcmMSFF ${GDCM_ZLIB_LIBRARIES})
  else()
    target_link_libraries(${example} gdcmMSFF)
  endif()
endforeach()

if(CMAKE_COMPILER_IS_GNUCXX AND MINGW)
#../../bin/libgdcmDSED.dll.a(d000469.o)(.text+0x0): multiple definition of `gdcm::VL gdcm::Item::GetLength<gdcm::ImplicitDataElement>() const'
#CMakeFiles/ReadExplicitLengthSQIVR.dir/ReadExplicitLengthSQIVR.obj(.text$_ZNK4gdcm4Item9GetLengthINS_19ImplicitDataElementEEENS_2VLEv[gdcm::VL gdcm::Item::GetLength<gdcm::ImplicitDataElement>() const]+0x0):ReadExplicitLengthSQIVR.cxx: first defined here
#../../bin/libgdcmDSED.dll.a(d000252.o)(.text+0x0): multiple definition of `std::istream& gdcm::DataSet::ReadNested<gdcm::ImplicitDataElement, gdcm::SwapperDoOp>(std::istream&)'
#CMakeFiles/ReadExplicitLengthSQIVR.dir/ReadExplicitLengthSQIVR.obj(.text$_ZN4gdcm7DataSet10ReadNestedINS_19ImplicitDataElementENS_11SwapperDoOpEEERSiS4_[std::basic_istream<char, std::char_traits<char> >& gdcm::DataSet::ReadNested<gdcm::ImplicitDataElement, gdcm::SwapperDoOp>(std::basic_istream<char, std::char_traits<char> >&)]+0x0):ReadExplicitLengthSQIVR.cxx: first defined here
#../../bin/libgdcmDSED.dll.a(d000253.o)(.text+0x0): multiple definition of `std::istream& gdcm::DataSet::ReadNested<gdcm::ImplicitDataElement, gdcm::SwapperNoOp>(std::istream&)'
#CMakeFiles/ReadExplicitLengthSQIVR.dir/ReadExplicitLengthSQIVR.obj(.text$_ZN4gdcm7DataSet10ReadNestedINS_19ImplicitDataElementENS_11SwapperNoOpEEERSiS4_[std::basic_istream<char, std::char_traits<char> >& gdcm::DataSet::ReadNested<gdcm::ImplicitDataElement, gdcm::SwapperNoOp>(std::basic_istream<char, std::char_traits<char> >&)]+0x0):ReadExplicitLengthSQIVR.cxx: first defined here
#../../bin/libgdcmDSED.dll.a(d000271.o)(.text+0x0): multiple definition of `std::istream& gdcm::DataSet::ReadWithLength<gdcm::ImplicitDataElement, gdcm::SwapperDoOp>(std::istream&, gdcm::VL&)'

  set_target_properties( ReadExplicitLengthSQIVR PROPERTIES LINK_FLAGS "-Wl,--allow-multiple-definition")
endif()
