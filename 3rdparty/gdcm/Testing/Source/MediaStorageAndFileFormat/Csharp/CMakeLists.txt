# C# MSFF tests
set(MSFF_TEST_SRCS
  FCTS_JPEG
  FCTS_JPEGLS
  FCTS_JPEG2000
  FCTS_RLE
  )

set(DEP)
set_source_files_properties(${GDCM_LIBRARY_DIR}/gdcm-sharp.dll PROPERTIES GENERATED 1)
foreach(test ${MSFF_TEST_SRCS})
  file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${test}.cs result)
  add_custom_command(
    OUTPUT ${GDCM_EXECUTABLE_DIR}/${test}.exe
    COMMAND ${CMAKE_CSHARP_COMPILER} "${CSC_PLATFORM_FLAG}" "/r:${GDCM_LIBRARY_DIR}/gdcm-sharp.dll" "/out:${GDCM_EXECUTABLE_DIR}/${test}.exe" ${result}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${test}.cs
    COMMENT "Create ${test}.exe"
    )
  add_custom_target(gdcm_sharp_${test} DEPENDS ${GDCM_EXECUTABLE_DIR}/${test}.exe)
  add_dependencies(gdcm_sharp_${test} GDCMCSharp)
  list(APPEND DEP ${GDCM_EXECUTABLE_DIR}/${test}.exe)

  # add test
  if(GDCM_DATA_ROOT)
    ADD_CSHARP_TEST(Test${test} ${GDCM_EXECUTABLE_DIR}/${test}.exe ${GDCM_DATA_ROOT}/GE_DLX-8-MONO2-Multiframe.dcm ${CMAKE_CURRENT_BINARY_DIR}/${test}.dcm)
  endif()
endforeach()

add_custom_target(gdcmMSFFTestsCsharp ALL
  DEPENDS ${DEP}
  COMMENT "building MSFF C# tests"
  )
add_dependencies(gdcmMSFFTestsCsharp GDCMCSharp)
