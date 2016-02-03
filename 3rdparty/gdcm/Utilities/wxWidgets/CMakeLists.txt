cmake_minimum_required(VERSION 2.8.9)

project(WXGDCM)
# wxWidgets stuff changed a lot within the 2.4.2 to 2.4.3
# I wonder if this has something to do with adding features on a stable branch?

#-----------------------------------------------------------------------------
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

if(WIN32)
  set(GUI_EXECUTABLE WIN32)
else()
  if(APPLE)
    set(GUI_EXECUTABLE MACOSX_BUNDLE)
    if(VTK_USE_COCOA)
      set_source_files_properties(
        wxVTKRenderWindowInteractor.cxx
        PROPERTIES COMPILE_FLAGS "-ObjC++")
    endif()
  else()
    # Ok X11 for sure, but just check:
    if(NOT VTK_USE_X)
      message(FATAL_ERROR "You need to have VTK_USE_X")
    endif()
    # See also:
    # FindGTK.cmake update
    # http://www.cmake.org/Bug/bug.php?op=show&bugid=3582
    #find_package(GTK REQUIRED)
    #include_directories(${GTK_INCLUDE_DIR}
    #  #/usr/lib/wx/include/gtk-2.4/
    #)
    find_package(PkgConfig)
    pkg_check_modules (GTK2 gtk+-2.0)
    #message("${GTK2_INCLUDE_DIRS}")
    include_directories(${GTK2_INCLUDE_DIRS})
    link_libraries(${GTK2_LIBRARIES})

    # Can I require all my user to have the gl lib on linux, even if they do not really need it...
    set(WXGLCANVASLIBS "gl")
  endif()
endif()

# wxWidgets is required to build the project
# For GTK we need a couple of stuff:
# gl: GLCanvas
# adv: wxSashLayoutWindow and such...
find_package(wxWidgets COMPONENTS base core adv ${WXGLCANVASLIBS})

if(wxWidgets_FOUND)
  include( ${wxWidgets_USE_FILE} )
endif()

include_directories(
  ${GDCM_BINARY_DIR}/Source/Common # gdcmConfigure.h
  ${GDCM_SOURCE_DIR}/Source/Common
  ${GDCM_SOURCE_DIR}/Utilities/VTK
)

add_executable(wxGDCM main.cpp wxGDCMFrame.cpp wxGDCMFrameBase.cpp wxVTKRenderWindowInteractor.cxx)
target_link_libraries(wxGDCM vtkRendering ${wxWidgets_LIBRARIES} vtkgdcm)
install(TARGETS wxGDCM
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  ${CPACK_NAMELINK_TYPE}
  )
