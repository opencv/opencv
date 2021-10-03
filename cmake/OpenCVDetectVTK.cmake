# VTK 6.x components
find_package(VTK QUIET COMPONENTS vtkInteractionStyle vtkRenderingLOD vtkIOPLY vtkFiltersTexture vtkRenderingFreeType vtkIOExport NO_MODULE)
IF(VTK_FOUND)
  IF(VTK_RENDERING_BACKEND) #in vtk 7, the rendering backend is exported as a var.
      find_package(VTK QUIET COMPONENTS vtkRendering${VTK_RENDERING_BACKEND} vtkInteractionStyle vtkRenderingLOD vtkIOPLY vtkFiltersTexture vtkRenderingFreeType vtkIOExport vtkIOGeometry NO_MODULE)
  ELSE(VTK_RENDERING_BACKEND)
      find_package(VTK QUIET COMPONENTS vtkRenderingOpenGL vtkInteractionStyle vtkRenderingLOD vtkIOPLY vtkFiltersTexture vtkRenderingFreeType vtkIOExport NO_MODULE)
  ENDIF(VTK_RENDERING_BACKEND)
ENDIF(VTK_FOUND)

# VTK 5.x components
if(NOT VTK_FOUND)
  find_package(VTK QUIET COMPONENTS vtkCommon NO_MODULE)
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
if((HAVE_QT AND VTK_USE_QT)
    AND NOT DEFINED FORCE_VTK  # deprecated
    AND NOT DEFINED OPENCV_FORCE_VTK
)
  message(STATUS "VTK support is disabled. Possible incompatible combination: OpenCV+Qt, and VTK ver.${VTK_VERSION} with Qt")
  message(STATUS "If it is known that VTK was compiled without Qt4, please define '-DOPENCV_FORCE_VTK=TRUE' flag in CMake")
  return()
endif()

set(HAVE_VTK ON)
message(STATUS "Found VTK ${VTK_VERSION} (${VTK_USE_FILE})")
