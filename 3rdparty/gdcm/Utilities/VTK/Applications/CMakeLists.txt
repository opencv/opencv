# Build the vtk-gdcm simple viewer
include_directories(
  ${GDCM_SOURCE_DIR}/Utilities/VTK
  )

if(WIN32 AND NOT CYGWIN)
  include_directories(
    "${GDCM_SOURCE_DIR}/Utilities/getopt"
  )
endif()
if(WIN32)
  if (BUILD_SHARED_LIBS)
    add_definitions(-DGETOPT_DLL)
  endif ()
endif()

set(GDCM_VTK_APPS
  gdcm2vtk
  )
if(VTK_USE_RENDERING OR vtkRenderingCore_LOADED)
  set(GDCM_VTK_APPS
    ${GDCM_VTK_APPS}
    gdcmviewer # vtkImageViewer2
    gdcm2pnm
    )
endif()

foreach(app ${GDCM_VTK_APPS})
  add_executable(${app} ${app}.cxx)
  if(GDCM_EXECUTABLE_PROPERTIES)
    set_target_properties(${app} PROPERTIES ${GDCM_EXECUTABLE_PROPERTIES})
  endif()
  target_link_libraries(${app} ${VTKGDCM_NAME})
  target_link_libraries(${app} gdcmDSED gdcmMSFF gdcmCommon)
  if( "${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" LESS 6.0 )
    target_link_libraries(${app} vtkCommon vtkFiltering vtkIO vtkImaging)
    if(VTK_USE_RENDERING)
      target_link_libraries(${app} vtkRendering)
    endif()
    if( "${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" GREATER 5.0 )
      target_link_libraries(${app} vtkWidgets)
    endif()
  else()
    # >= 6.0
    target_link_libraries(${app} ${VTK_LIBRARIES} vtkIOXML)
  endif()
  if(WIN32 AND NOT CYGWIN)
    target_link_libraries(${app} gdcmgetopt)
  endif()
  if(NOT GDCM_INSTALL_NO_RUNTIME)
    install(TARGETS ${app}
      RUNTIME DESTINATION ${GDCM_INSTALL_BIN_DIR} COMPONENT VTKApplications
      LIBRARY DESTINATION ${GDCM_INSTALL_LIB_DIR} COMPONENT VTKLibraries
      ARCHIVE DESTINATION ${GDCM_INSTALL_LIB_DIR} COMPONENT VTKDebugDevel
    )
  endif()
endforeach()
