if(NOT WITH_VTK OR ANDROID OR IOS)
  return()
endif()

if (HAVE_QT5)
  message(STATUS "VTK is disabled because OpenCV is linked with Q5. Some VTK disributives are compiled with Q4 and therefore can't be linked together Qt5.")
  return()
endif()

find_package(VTK 6.0 QUIET COMPONENTS vtkRenderingCore vtkInteractionWidgets vtkInteractionStyle vtkIOLegacy vtkIOPLY vtkRenderingFreeType vtkRenderingLOD vtkFiltersTexture vtkIOExport NO_MODULE)

if(NOT DEFINED VTK_FOUND OR NOT VTK_FOUND)
  find_package(VTK 5.10 QUIET COMPONENTS vtkCommon vtkFiltering vtkRendering vtkWidgets vtkImaging NO_MODULE)
endif()

if(NOT DEFINED VTK_FOUND OR NOT VTK_FOUND)
  find_package(VTK 5.8 QUIET COMPONENTS vtkCommon vtkFiltering vtkRendering vtkWidgets vtkImaging NO_MODULE)
endif()

if(VTK_FOUND)
  set(HAVE_VTK ON)
  message(STATUS "Found VTK ver. ${VTK_VERSION} (usefile: ${VTK_USE_FILE})")
else()
  set(HAVE_VTK OFF)
  message(STATUS "VTK is not found. Please set -DVTK_DIR in CMake to VTK build directory, or set $VTK_DIR enviroment variable to VTK install subdirectory with VTKConfig.cmake file (for windows)")
endif()
