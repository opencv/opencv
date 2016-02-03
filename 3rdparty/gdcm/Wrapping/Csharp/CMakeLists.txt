# C# Cmd line options:
# http://msdn.microsoft.com/en-us/library/ms379563(VS.80).aspx
# http://msdn.microsoft.com/en-us/library/aa288436(VS.71).aspx
# http://msdn.microsoft.com/en-us/library/aa288481(VS.71).aspx

find_package(SWIG REQUIRED)
mark_as_advanced(SWIG_DIR SWIG_EXECUTABLE SWIG_VERSION)
include(${SWIG_USE_FILE})

include_directories(
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  "${GDCM_SOURCE_DIR}/Source/InformationObjectDefinition"
  "${GDCM_SOURCE_DIR}/Source/MediaStorageAndFileFormat"
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
  "${GDCM_SOURCE_DIR}/Source/MessageExchangeDefinition"
)

set_source_files_properties(gdcm.i PROPERTIES CPLUSPLUS ON)

# See sf.net bug #2895067
set(CMAKE_SWIG_FLAGS "-namespace gdcm -dllimport gdcmsharpglue")
separate_arguments(CMAKE_SWIG_FLAGS)
# libgdksharpglue.so
SWIG_ADD_MODULE(gdcmsharpglue csharp gdcm.i)
SWIG_LINK_LIBRARIES(gdcmsharpglue gdcmMSFF
  gdcmMEXD
)
set_target_properties(${SWIG_MODULE_gdcmsharpglue_REAL_NAME} PROPERTIES LINK_INTERFACE_LIBRARIES "")
set_property(TARGET ${SWIG_MODULE_gdcmsharpglue_REAL_NAME} PROPERTY NO_SONAME 1)

# Module are always place in the library destination but for poor win32 user I
# decided to place them right next to the other dlls
if(NOT GDCM_INSTALL_NO_LIBRARIES)
  install(TARGETS ${SWIG_MODULE_gdcmsharpglue_REAL_NAME}
    RUNTIME DESTINATION ${GDCM_INSTALL_BIN_DIR} COMPONENT Applications
    LIBRARY DESTINATION ${GDCM_INSTALL_LIB_DIR} COMPONENT Libraries
    ARCHIVE DESTINATION ${GDCM_INSTALL_LIB_DIR} COMPONENT DebugDevel
    ${CPACK_NAMELINK_TYPE}
    )
endif()

set(OLDSTYLECSHARP 1)

if(OLDSTYLECSHARP)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/AssemblyInfo.cs.in
    ${CMAKE_CURRENT_BINARY_DIR}/AssemblyInfo.cs
    @ONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/key.snk
    ${CMAKE_CURRENT_BINARY_DIR}/key.snk
    COPYONLY)

  # From Mirco Bauer on debian-cli
  # Hm? You wrote the C# binding for gdcm? Using C# in the lib name is
  # very wrong in all cases. The CIL bytecode can be used by any language
  # that targets the CLI. To give some examples: VB.NET, Boo, Nemerle (yes,
  # those are all in debian already :)).
  #
  # There is a common prefix of nFOO (like nunit or nant) and for
  # bindings even more common is the suffix of FOO-sharp (not -csharp)
  # like gtk-sharp, gnome-sharp, evolution-sharp.
  #
  # /usr/lib/mono/gtk-sharp/gtk-sharp.dll
  # /usr/lib/mono/gtk-sharp/gdk-sharp.dll
  # /usr/lib/mono/gtk-sharp/atk-sharp.dll

  add_custom_command(
    OUTPUT ${GDCM_LIBRARY_DIR}/gdcm-sharp.dll
    COMMAND ${CMAKE_CSHARP_COMPILER} ARGS "/t:library" "/doc:${GDCM_LIBRARY_DIR}/gdcm-sharp.dll.xml" "/out:${GDCM_LIBRARY_DIR}/gdcm-sharp.dll" "*.cs"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "${swig_generated_file_fullname}"
    ${CMAKE_CURRENT_BINARY_DIR}/AssemblyInfo.cs
    COMMENT "csc *.cs"
    )

  if(BUILD_EXAMPLES)
    # building HelloCSharpWorld:
    add_custom_command(
      OUTPUT ${GDCM_EXECUTABLE_DIR}/HelloCsharpWorld.exe
      COMMAND ${CMAKE_CSHARP_COMPILER} ARGS "/r:${GDCM_LIBRARY_DIR}/gdcm-sharp.dll" "/out:${GDCM_EXECUTABLE_DIR}/HelloCsharpWorld.exe" ${CMAKE_CURRENT_SOURCE_DIR}/HelloCsharpWorld.cs
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      DEPENDS ${GDCM_LIBRARY_DIR}/gdcm-sharp.dll
      ${CMAKE_CURRENT_SOURCE_DIR}/HelloCsharpWorld.cs
      COMMENT "Create HelloCsharpWorld.exe"
      )
  endif()

  # $ export MONO_LOG_LEVEL="debug"; export MONO_LOG_MASK="dll"
  # $ export LD_LIBRARY_PATH=/home/mmalaterre/Projects/gdcm/debug-gcc43/Wrapping/Csharp/:/home/mmalaterre/Projects/gdcm/debug-gcc43/bin
  add_custom_target(GDCMCSharp ALL
    DEPENDS
    ${GDCM_LIBRARY_DIR}/gdcm-sharp.dll
    ${SWIG_MODULE_gdcmsharpglue_REAL_NAME} # GDCMCSharp from the outside rebuild the C# binding not just the *.dll but the glue
    COMMENT "building gdcm-sharp.dll"
    )

  if(BUILD_EXAMPLES)
    add_custom_target(GDCMCSharpExample ALL
      DEPENDS
      ${GDCM_EXECUTABLE_DIR}/HelloCsharpWorld.exe
      COMMENT "building HelloCsharpWorld.exe"
      )
  endif()
endif()

if(NOT GDCM_INSTALL_NO_LIBRARIES)
  install_swig_module(gdcmsharpglue CSharp)
  set(GDCM_LIBRARY_DIR2 ${LIBRARY_OUTPUT_PATH}/\${BUILD_TYPE})
  # because gdcm-sharp.dll is constructed with custom commands, it need the
  # install(FILES signature:
  install(FILES
    ${GDCM_LIBRARY_DIR2}/gdcm-sharp.dll
    DESTINATION ${GDCM_INSTALL_CSHARPMODULE_DIR} COMPONENT CSharpModule
    )
endif()
