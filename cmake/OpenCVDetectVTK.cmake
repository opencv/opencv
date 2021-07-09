if(NOT VTK_FOUND)
  find_package(VTK QUIET NAMES vtk VTK)
  if(VTK_FOUND)
    if(VTK_VERSION VERSION_EQUAL "9") # VTK 9.0
      find_package(VTK 9 QUIET NAMES vtk COMPONENTS
              FiltersExtraction
              FiltersSources
              FiltersTexture
              IOExport
              IOGeometry
              IOPLY
              InteractionStyle
              RenderingCore
              RenderingLOD
              RenderingOpenGL2
              NO_MODULE)
    elseif(VTK_VERSION VERSION_GREATER "5") # VTK 6.x components
      find_package(VTK QUIET COMPONENTS vtkInteractionStyle vtkRenderingLOD vtkIOPLY vtkFiltersTexture vtkRenderingFreeType vtkIOExport NO_MODULE)
      IF(VTK_FOUND)
        IF(VTK_RENDERING_BACKEND) #in vtk 7, the rendering backend is exported as a var.
          find_package(VTK QUIET COMPONENTS vtkRendering${VTK_RENDERING_BACKEND} vtkInteractionStyle vtkRenderingLOD vtkIOPLY vtkFiltersTexture vtkRenderingFreeType vtkIOExport vtkIOGeometry NO_MODULE)
        ELSE(VTK_RENDERING_BACKEND)
          find_package(VTK QUIET COMPONENTS vtkRenderingOpenGL vtkInteractionStyle vtkRenderingLOD vtkIOPLY vtkFiltersTexture vtkRenderingFreeType vtkIOExport NO_MODULE)
        ENDIF(VTK_RENDERING_BACKEND)
      ENDIF(VTK_FOUND)
    elseif(VTK_VERSION VERSION_EQUAL "5") # VTK 5.x components
      find_package(VTK QUIET COMPONENTS vtkCommon NO_MODULE)
    else()
      set(VTK_FOUND FALSE)
    endif()
  endif()
endif()

if(NOT VTK_FOUND)
  set(HAVE_VTK OFF)
  message(STATUS "VTK is not found. Please set -DVTK_DIR in CMake to VTK build directory, or to VTK install subdirectory with VTKConfig.cmake file")
  return()
endif()

# Don't support earlier VTKs
if(VTK_VERSION VERSION_LESS "5.8.0")
  message(STATUS "VTK support is disabled. VTK ver. 5.8.0 is minimum required, but found VTK ver. ${VTK_VERSION}")
  return()
endif()

# Different Qt versions can't be linked together
if(HAVE_QT5 AND VTK_VERSION VERSION_LESS "6.0.0")
  if(VTK_USE_QT)
    message(STATUS "VTK support is disabled. Incompatible combination: OpenCV + Qt5 and VTK ver.${VTK_VERSION} + Qt4")
  endif()
endif()

# Different Qt versions can't be linked together. VTK 6.0.0 doesn't provide a way to get Qt version it was linked with
if(HAVE_QT5 AND VTK_VERSION VERSION_EQUAL "6.0.0" AND NOT DEFINED FORCE_VTK)
  message(STATUS "VTK support is disabled. Possible incompatible combination: OpenCV+Qt5, and VTK ver.${VTK_VERSION} with Qt4")
  message(STATUS "If it is known that VTK was compiled without Qt4, please define '-DFORCE_VTK=TRUE' flag in CMake")
  return()
endif()

# Different Qt versions can't be linked together
if(HAVE_QT AND VTK_VERSION VERSION_GREATER "6.0.0" AND NOT ${VTK_QT_VERSION} STREQUAL "")
  if(HAVE_QT5 AND ${VTK_QT_VERSION} EQUAL "4")
    message(STATUS "VTK support is disabled. Incompatible combination: OpenCV + Qt5 and VTK ver.${VTK_VERSION} + Qt4")
    return()
  endif()

  if(NOT HAVE_QT5 AND ${VTK_QT_VERSION} EQUAL "5")
    message(STATUS "VTK support is disabled. Incompatible combination: OpenCV + Qt4 and VTK ver.${VTK_VERSION} + Qt5")
    return()
  endif()
endif()

try_compile(VTK_COMPILE_STATUS
    "${OpenCV_BINARY_DIR}"
    "${OpenCV_SOURCE_DIR}/cmake/checks/vtk_test.cpp"
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${VTK_INCLUDE_DIRS}"
    LINK_LIBRARIES ${VTK_LIBRARIES}
    OUTPUT_VARIABLE OUTPUT
)

if(NOT ${VTK_COMPILE_STATUS})
  message(STATUS "VTK support is disabled. Compilation of the sample code has failed.")
  return()
endif()

set(HAVE_VTK ON)
if (VTK_VERSION VERSION_LESS "8.90.0")
  message(STATUS "Found VTK ${VTK_VERSION} (${VTK_USE_FILE})")
else()
  message(STATUS "Found VTK ${VTK_VERSION}")
endif()
