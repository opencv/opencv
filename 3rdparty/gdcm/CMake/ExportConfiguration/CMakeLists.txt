### Build the configuration file for external projects using GDCM and cmake:
configure_file( GDCMConfig.cmake.in
  ${GDCM_BINARY_DIR}/GDCMConfig.cmake
  @ONLY
)
install( FILES ${GDCM_BINARY_DIR}/GDCMConfig.cmake
  DESTINATION ${GDCM_INSTALL_PACKAGE_DIR} COMPONENT Headers
)
configure_file( GDCMConfigVersion.cmake.in
  ${GDCM_BINARY_DIR}/GDCMConfigVersion.cmake
  @ONLY
)
install( FILES ${GDCM_BINARY_DIR}/GDCMConfigVersion.cmake
  DESTINATION ${GDCM_INSTALL_PACKAGE_DIR} COMPONENT Headers
)
configure_file( UseGDCM.cmake.in
  ${GDCM_BINARY_DIR}/UseGDCM.cmake
  @ONLY
)
install( FILES ${GDCM_BINARY_DIR}/UseGDCM.cmake
  DESTINATION ${GDCM_INSTALL_PACKAGE_DIR} COMPONENT Headers
)

# install all targets referenced as ${GDCM_TARGETS_NAME}
if(GDCM_STANDALONE)
  install(EXPORT ${GDCM_TARGETS_NAME} DESTINATION ${GDCM_INSTALL_PACKAGE_DIR} COMPONENT Headers)
endif()

# [Prevent viral CMake dep]
# See the following ref, which explain this mess.
# http://cmake.org/Bug/view.php?id=9822
# 0009822: Please expose: cmGlobalGenerator::GetExportSet at cmake level

# I cannot retrieve the lists of exported targets, so read them from the file instead

#set(gdcmtarget_cmake
#  "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/Export/${GDCM_INSTALL_PACKAGE_DIR}/${GDCM_TARGETS_NAME}.cmake")

#configure_file(
# ${GDCM_BINARY_DIR}/CMakeCache.txt
#  ${CMAKE_CURRENT_BINARY_DIR}/gdcmtarget_cmake.dep
#COPYONLY
#)
# CMake Error at CMake/ExportConfiguration/CMakeLists.txt:59 (export):
#   export given target "gdcmopenjpeg" which is not built by this project.
#add_custom_command(
#  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gdcmtarget_cmake.dep
#  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GDCM_BINARY_DIR}/CMakeCache.txt ${CMAKE_CURRENT_BINARY_DIR}/gdcmtarget_cmake.dep
#  COMMAND ${CMAKE_COMMAND} -E remove ${gdcmtarget_cmake}
#  COMMAND ${CMAKE_COMMAND} -E remove ${GDCM_BINARY_DIR}/GDCMExports.cmake
#  #DEPENDS ${GDCM_BINARY_DIR}/CMakeCache.txt
#  #DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gdcmtarget_cmake.dep
#  DEPENDS ${gdcmtarget_cmake}
#)
#add_custom_target(clean_t ALL
#  COMMAND echo "Execute"
#  #COMMAND ${CMAKE_COMMAND} -E remove ${gdcmtarget_cmake}
#  COMMAND ${CMAKE_COMMAND} -E remove ${GDCM_BINARY_DIR}/GDCMExports.cmake
#  DEPENDS ${gdcmtarget_cmake} ${GDCM_BINARY_DIR}/GDCMExports.cmake
#  COMMENT "Crazy"
#)


#set( targets )
##file(REMOVE ${gdcmtarget_cmake})
#if(EXISTS ${gdcmtarget_cmake})
#  #message( "reading: ${gdcmtarget_cmake}" )
#  file(READ ${gdcmtarget_cmake} ENT)
#  string(REGEX REPLACE "\r?\n" ";" ENT "${ENT}")
#  foreach(line ${ENT})
#    string(REGEX MATCH "ADD_LIBRARY\\(.*SHARED IMPORTED\\)" m ${line})
#    if( m )
#      string(REGEX REPLACE "ADD_LIBRARY\\((.*) SHARED IMPORTED\\)" "\\1" out ${m})
#      list(APPEND targets ${out})
#    endif()
#  endforeach()
#endif()
#
#set( exports )
#set(gdcmexport_cmake ${GDCM_BINARY_DIR}/GDCMExports.cmake)
#if(EXISTS ${gdcmexport_cmake})
#  file(READ ${gdcmexport_cmake} ENT)
#  string(REGEX REPLACE "\r?\n" ";" ENT "${ENT}")
#  foreach(line ${ENT})
#    string(REGEX MATCH "ADD_LIBRARY\\(.*SHARED IMPORTED\\)" m ${line})
#    if( m )
#      string(REGEX REPLACE "ADD_LIBRARY\\((.*) SHARED IMPORTED\\)" "\\1" out ${m})
#      list(APPEND exports ${out})
#    endif()
#  endforeach()
#endif()

#list(LENGTH targets targets_len)
#list(LENGTH exports exports_len)
#if( ${targets_len} EQUAL ${exports_len} )
#  message("equal")
#else()
#  message("diff")
#  message( "${targets}" )
#  message( "${exports}" )
##set(targets ${exports})
##  execute_process(
##    COMMAND ${CMAKE_COMMAND} -E remove ${gdcmtarget_cmake}
##  )
#endif()

set(targets
  gdcmCommon
  gdcmDICT
  gdcmDSED
  gdcmIOD
  gdcmMSFF
  gdcmMEXD
  gdcmjpeg8
  gdcmjpeg12
  gdcmjpeg16
  gdcmexpat
  gdcmopenjpeg
  gdcmcharls
  gdcmmd5
  gdcmzlib
  gdcmuuid
  socketxx
  vtkgdcm
  php_vtkgdcm
  vtkgdcmsharpglue
  )
file(WRITE ${GDCM_BINARY_DIR}/GDCMExports.cmake "")
foreach(target ${targets})
  if(TARGET ${target})
    export(TARGETS ${target} FILE ${GDCM_BINARY_DIR}/GDCMExports.cmake APPEND)
  endif()
endforeach()
