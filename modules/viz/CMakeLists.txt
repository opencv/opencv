###############################################################################
# Find qvtk
# This sets the following variables:
# QVTK_FOUND - True if QVTK was found.
# QVTK_INCLUDE_DIR - Directory containing the QVTK include files.
# QVTK_LIBRARY - QVTK library.
# if QVTK_FOUND then QVTK_INCLUDE_DIR is appended to VTK_INCLUDE_DIRS and QVTK_LIBRARY is appended to QVTK_LIBRARY_DIR
macro(find_qvtk)
  find_library (QVTK_LIBRARY QVTK HINTS ${VTK_DIR} ${VTK_DIR}/bin)
  find_path (QVTK_INCLUDE_DIR QVTKWidget.h HINT ${VTK_INCLUDE_DIRS})
  find_package_handle_standard_args(QVTK DEFAULT_MSG QVTK_LIBRARY QVTK_INCLUDE_DIR)

  if(NOT QVTK_FOUND)
    set (VTK_USE_QVTK OFF)
  else(NOT QVTK_FOUND)
    get_filename_component (QVTK_LIBRARY_DIR ${QVTK_LIBRARY} PATH)
    set (VTK_LIBRARY_DIRS ${VTK_LIBRARY_DIRS} ${QVTK_LIBRARY_DIR})
    set (VTK_INCLUDE_DIRS ${VTK_INCLUDE_DIRS} ${QVTK_INCLUDE_DIR})
    set (VTK_USE_QVTK ON)
  endif()
endmacro()

macro(find_vtk)
  find_package(VTK 5.8.0 REQUIRED)
  if(VTK_FOUND)
    if (BUILD_SHARED_LIBS OR (NOT BUILD_SHARED_LIBS AND NOT VTK_BUILD_SHARED_LIBS))
      find_qvtk()
      message(STATUS "VTK found (include: ${VTK_INCLUDE_DIRS}, lib: ${VTK_LIBRARY_DIRS})")
      link_directories(${VTK_LIBRARY_DIRS})
      include_directories(SYSTEM ${VTK_INCLUDE_DIRS})
      set(HAVE_VTK ON)    
    else ()
      set(HAVE_VTK OFF)
      message ("Warning: VTK disabled. You are to build OpenCV in STATIC but VTK is SHARED!")
    endif ()
  endif()
endmacro()

macro(find_boost)
  # Disable the config mode of find_package(Boost)
  set(Boost_NO_BOOST_CMAKE ON)
  set(Boost_USE_STATIC_LIBS ON)

  find_package(Boost 1.49.0 REQUIRED COMPONENTS system thread)

  if(Boost_FOUND)
    set(HAVE_BOOST ON)

    # Obtain diagnostic information about Boost's automatic linking outputted during compilation time.
    add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
    message(STATUS "Boost found (include: ${Boost_INCLUDE_DIRS})")
  endif()
endmacro()

find_vtk()
find_boost()


find_package(OpenGL)
if (OPENGL_FOUND)
  if(OPENGL_INCLUDE_DIR)
    include_directories("${OPENGL_INCLUDE_DIR}")
  endif()
  if(OPENGL_DEFINITIONS)
    add_definitions("${OPENGL_DEFINITIONS}")
  endif()
endif()


if(NOT HAVE_VTK)
  set(DEFAULT FALSE)
  set(REASON "VTK was not found.")
else()
  set(DEFAULT TRUE)
  set(REASON)
  set(VTK_USE_FILE ${VTK_USE_FILE} CACHE INTERNAL "VTK_USE_FILE")
  include (${VTK_USE_FILE})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()


add_definitions(-DHAVE_VTK)


set(BUILD_opencv_viz_INIT OFF)

include_directories(src)
set(the_description "Viz")
ocv_define_module(viz opencv_core opencv_calib3d)
#${PCL_LIBRARIES}

target_link_libraries(opencv_viz vtkCommon vtkWidgets vtkHybrid vtkCharts vtkFiltering vtkRendering ${OPENGL_LIBRARIES})

if(APPLE)
    target_link_libraries(opencv_viz "-framework Cocoa")
endif()

