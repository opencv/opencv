set(DEP)
if(GDCM_USE_ACTIVIZ)
  set(examples
    #HelloActiviz
    HelloActiviz2
    HelloActiviz3
    HelloActiviz4
    HelloActiviz5
    RefCounting
    MetaImageMD5Activiz
    )
  foreach(example ${examples})
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${example}.cs result)
    add_custom_command(
      OUTPUT ${GDCM_EXECUTABLE_DIR}/${example}.exe
      COMMAND ${CMAKE_CSHARP_COMPILER} ARGS "/r:${ACTIVIZ_KITWARE_VTK_LIBRARY}" "/r:${ACTIVIZ_KITWARE_MUMMY_RUNTIME_LIBRARY}" "/r:${GDCM_LIBRARY_DIR}/Kitware.VTK.GDCM.dll" "/r:${GDCM_LIBRARY_DIR}/gdcm-sharp.dll" "/out:${GDCM_EXECUTABLE_DIR}/${example}.exe" ${result}
      DEPENDS ${GDCM_LIBRARY_DIR}/Kitware.VTK.GDCM.dll
      ${CMAKE_CURRENT_SOURCE_DIR}/${example}.cs
      COMMENT "Create ${example}.exe"
      )
    set(DEP ${DEP} ${GDCM_EXECUTABLE_DIR}/${example}.exe)
  endforeach()

  if(BUILD_TESTING)
    get_filename_component(runtimepath ${ACTIVIZ_KITWARE_VTK_LIBRARY} PATH)
    if(GDCM_DATA_ROOT)
      set_source_files_properties(${GDCM_EXECUTABLE_DIR}/HelloActiviz5.exe PROPERTIES RUNTIMEPATH ${runtimepath})
      ADD_CSHARP_TEST(TestHelloActiviz5CSharp ${GDCM_EXECUTABLE_DIR}/HelloActiviz5.exe)
    endif()
    set_source_files_properties(${GDCM_EXECUTABLE_DIR}/RefCounting.exe PROPERTIES RUNTIMEPATH ${runtimepath})
    ADD_CSHARP_TEST(TestRefCountingCSharp ${GDCM_EXECUTABLE_DIR}/RefCounting.exe)
  endif()

else()

  set(CSHARP_EXAMPLES
    HelloVTKWorld
    HelloVTKWorld2
    )

  foreach(example ${CSHARP_EXAMPLES})
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${example}.cs result)
    # WORKING_DIRECTORY is set to the src dir because of a strange issue with CSC compiler on Win32 system:
    # http://groups.google.com/group/microsoft.public.dotnet.languages.csharp/browse_thread/thread/9d3ac7eb9f7f56be
    add_custom_command(
      OUTPUT ${GDCM_EXECUTABLE_DIR}/${example}.exe
      COMMAND ${CMAKE_CSHARP_COMPILER} ARGS "/r:${GDCM_LIBRARY_DIR}/vtkgdcm-sharp.dll" "/out:${GDCM_EXECUTABLE_DIR}/${example}.exe" ${result}
      DEPENDS ${GDCM_LIBRARY_DIR}/vtkgdcm-sharp.dll
      ${CMAKE_CURRENT_SOURCE_DIR}/${example}.cs
      COMMENT "Create ${example}.exe"
      )
    set(DEP ${DEP} ${GDCM_EXECUTABLE_DIR}/${example}.exe)
  endforeach()
endif()

add_custom_target(VTKGDCMExampleCSharp ALL
  DEPENDS ${DEP}
  COMMENT "building examples"
)
