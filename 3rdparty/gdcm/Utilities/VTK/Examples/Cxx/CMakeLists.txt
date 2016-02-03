include_directories(
  ${GDCM_SOURCE_DIR}/Utilities/VTK
  )


if( "${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" GREATER 5.0 )
  set(GDCM_VTK_APPS
    CreateFakeRTDOSE
    ConvertSingleBitTo8Bits
    GenerateRTSTRUCT
    offscreenimage
  )
  if( ${VTK_MAJOR_VERSION} GREATER 5 )
    set(GDCM_VTK_APPS
      ${GDCM_VTK_APPS}
      Compute3DSpacing
      )
  endif()
  if(VTK_USE_RENDERING OR vtkRenderingCore_LOADED)
    set(GDCM_VTK_APPS
      ${GDCM_VTK_APPS}
      gdcmorthoplanes
      gdcmreslice
      gdcmrtionplan
      gdcmrtplan
      gdcmscene
      gdcmtexture
      gdcmvolume
      rtstructapp
      )
  endif()
  if(GDCM_BUILD_TESTING)
    set(GDCM_VTK_APPS
      ${GDCM_VTK_APPS}
      ConvertRGBToLuminance
      Convert16BitsTo8Bits
      ConvertMultiFrameToSingleFrame
      MagnifyFile
      )
    if(GDCM_DATA_EXTRA_ROOT)
      if(VTK_USE_RENDERING OR vtkRenderingCore_LOADED)
        set(GDCM_VTK_APPS
          ${GDCM_VTK_APPS}
          reslicesphere
          )
      endif()
    endif()
  endif()
endif()

foreach(app ${GDCM_VTK_APPS})
  add_executable(${app} ${app}.cxx)
  #set_target_properties(${app} PROPERTIES ${GDCM_EXECUTABLE_PROPERTIES})
  target_link_libraries(${app} ${VTKGDCM_NAME} ${vtkgdcm_LIBS})
  target_link_libraries(${app} gdcmDSED gdcmMSFF gdcmCommon)
  if(GDCM_BUILD_TESTING)
    # gdcmTesting is in Common:
    target_link_libraries(${app} gdcmCommon)
  endif()
  if( "${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" GREATER 6.0 )
    target_link_libraries(${app} ${VTK_LIBRARIES})
  else()
    if( "${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" GREATER 5.0 )
      if(VTK_USE_RENDERING)
        target_link_libraries(${app} vtkWidgets vtkVolumeRendering)
      endif()
    endif()
  endif()
endforeach()

if(GDCM_HAVE_PTHREAD_H)
# Dev only: I dont need to install that one:
  add_executable(threadgdcm threadgdcm.cxx)
  target_link_libraries(threadgdcm gdcmMSFF pthread )
  if( "${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" GREATER 6.0 )
    target_link_libraries(threadgdcm ${VTK_LIBRARIES})
  else()
    target_link_libraries(threadgdcm vtkIO)
  endif()
endif()

#add_executable(gdcm2vtk gdcm2vtk.cxx)
#target_link_libraries(gdcm2vtk gdcmMSFF pthread)
