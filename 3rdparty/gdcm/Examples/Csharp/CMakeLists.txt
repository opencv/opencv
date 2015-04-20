# C# examples
set(CSHARP_EXAMPLES
  SimplePrintPatientName
  ExtractOneFrame
  ExtractImageRegion
  ExtractImageRegionWithLUT
  ManipulateFile
  RescaleImage
  ExtractEncapsulatedFile
  DecompressImage
  DecompressImageMultiframe
  DecompressJPEGFile
  SimplePrint
  ScanDirectory
  #SortImage2
  GetArray
  NewSequence
  GenerateDICOMDIR
  StandardizeFiles
  ReformatFile
  CompressLossyJPEG
  SendFileSCU
  MpegVideoInfo
  BasicImageAnonymizer
  FileAnonymize
  FileStreaming
  FileChangeTS
  FileChangeTSLossy
  )

if(BUILD_TESTING)
  list(APPEND CSHARP_EXAMPLES
    BasicAnonymizer
    ClinicalTrialIdentificationWorkflow
    )
endif()

set(DEP)
# http://www.cmake.org/Wiki/CMake_FAQ#How_can_I_add_a_dependency_to_a_source_file_which_is_generated_in_a_subdirectory.3F
# Tell CMake the source won't be available until build time.
set_source_files_properties(${GDCM_LIBRARY_DIR}/gdcm-sharp.dll PROPERTIES GENERATED 1)
# Make sure the source is generated before the executable builds.
foreach(example ${CSHARP_EXAMPLES})
  file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${example}.cs result)
  add_custom_command(
    OUTPUT ${GDCM_EXECUTABLE_DIR}/${example}.exe
    COMMAND ${CMAKE_CSHARP_COMPILER} "${CSC_PLATFORM_FLAG}" "/r:${GDCM_LIBRARY_DIR}/gdcm-sharp.dll" "/out:${GDCM_EXECUTABLE_DIR}/${example}.exe" ${result}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${example}.cs
    COMMENT "Create ${example}.exe"
  )
  add_custom_target(gdcm_sharp_${example} DEPENDS ${GDCM_EXECUTABLE_DIR}/${example}.exe)
  # Make sure the gdcm-sharp.dll is generated before the exampls builds.
  add_dependencies(gdcm_sharp_${example} GDCMCSharp)

  list(APPEND DEP ${GDCM_EXECUTABLE_DIR}/${example}.exe)
endforeach()

add_custom_target(GDCMExampleCSharp ALL
  DEPENDS ${DEP}
  COMMENT "building examples"
)
add_dependencies(GDCMExampleCSharp GDCMCSharp)

if(BUILD_TESTING)
if(GDCM_DATA_ROOT)
  ADD_CSHARP_TEST(TestSimplePrintCSharp ${GDCM_EXECUTABLE_DIR}/SimplePrint.exe ${GDCM_DATA_ROOT}/012345.002.050.dcm)
# TODO:
#if(BUILD_EXAMPLES)
#  ADD_CSHARP_TEST(HelloWorldCSharp ${GDCM_EXECUTABLE_DIR}/HelloWorld.exe ${GDCM_DATA_ROOT}/012345.002.050.dcm)
#endif()
endif()
endif()
