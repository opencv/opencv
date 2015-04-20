#
# Build the documentation
#
include (Documentation)

if (BUILD_DOCUMENTATION)
find_package(VTK REQUIRED)
mark_as_advanced(VTK_DIR)

  set(VTKGDCM_FINAL_PATH ${GDCM_BINARY_DIR}/Utilities/doxygen/html)
  set (INCLUDE_QT_DOCUMENTATION OFF CACHE BOOL "Specify if Qt documentation should be linked to.")
  mark_as_advanced(INCLUDE_QT_DOCUMENTATION)
  if (INCLUDE_QT_DOCUMENTATION)
    set(QT_DOC_DIR CACHE FILEPATH "Path to the directory containing Qt html documentation.")
    find_program (DOXYTAG
      doxytag
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\doxygen_is1;Inno Setup: App Path]/bin"
      /Applications/Doxygen.app/Contents/Resources
      /Applications/Doxygen.app/Contents/MacOS
      )
    mark_as_advanced(DOXYTAG QT_DOC_DIR)
    set (QT_TAG_FILE "${CMAKE_CURRENT_BINARY_DIR}/Qt.tag")
  endif ()
  set(BUILD_DOXYGEN ON)
  set(DOCUMENTATION_DOWNLOAD_VTK_TAGFILE ON)
  set(DOCUMENTATION_HTML_TARZ ON)
  set(DOXYGEN_PROJECT_NAME "vtkgdcm")
  if(VTK_SOURCE_DIR)
  message(${VTK_SOURCE_DIR})
  #set(DOXYGEN_SOURCE_DIR "${vtkgdcm_SOURCE_DIR}/VTK/Utilities/Doxygen")
  set(DOXYGEN_SOURCE_DIR "${VTK_SOURCE_DIR}/Utilities/Doxygen")
  else()
      # sudo apt-get install vtk-doc
  set(DOXYGEN_SOURCE_DIR "/usr/share/doc/vtk-doc/html/")
  endif()
  set(DOXYGEN_PROJECT_SOURCE_DIR "${vtkgdcm_SOURCE_DIR}")
  set(VTK_DOXYGEN_HOME "${vtkgdcm_SOURCE_DIR}/VTK/Utilities/Doxygen")
  set(DOXYGEN_CVSWEB_CHECKOUT "http://public.kitware.com/cgi-bin/cvsweb.cgi/~checkout~/vtkgdcm/")
  set(DOXYGEN_CVSWEB_CHECKOUT_SUFFIX "?cvsroot=vtkgdcm")
  set(DOXYGEN_SOURCE_LOCATIONS_DIR "${vtkgdcm_SOURCE_DIR}")
  set(DOXYGEN_SOURCE_LOCATIONS
       "\"${vtkgdcm_SOURCE_DIR}\" \\
        \"${vtkgdcm_SOURCE_DIR}/Utilities/Xdmf2/vtk\" \\
        \"${vtkgdcm_SOURCE_DIR}/Utilities/VTKPythonWrapping/Executable\" \\
        \"${vtkgdcm_SOURCE_DIR}/Utilities/VTKClientServer\" \\
        \"${vtkgdcm_SOURCE_DIR}/Qt/Chart\" \\
        \"${vtkgdcm_SOURCE_DIR}/Qt/Components\" \\
        \"${vtkgdcm_SOURCE_DIR}/Qt/Core\" \\
        \"${vtkgdcm_SOURCE_DIR}/Qt/Python\" \\
        \"${vtkgdcm_SOURCE_DIR}/Qt/Widgets\" \\
        \"${vtkgdcm_SOURCE_DIR}/Qt/Testing\" \\
        \"${vtkgdcm_SOURCE_DIR}/Applications/Client\" \\
        \"${vtkgdcm_SOURCE_DIR}/Applications/OverView\" \\
        \"${vtkgdcm_SOURCE_DIR}/Applications/OverView/Core\" \\
        \"${vtkgdcm_SOURCE_DIR}/Common/KWCommon\" \\
        \"${vtkgdcm_SOURCE_DIR}/Servers/Common\" \\
        \"${vtkgdcm_SOURCE_DIR}/Servers/Filters\" \\
        \"${vtkgdcm_SOURCE_DIR}/Servers/ServerManager\"")
  set(DOXYGEN_BINARY_LOCATIONS_DIR "${vtkgdcm_BINARY_DIR}")
  set(DOXYGEN_BINARY_LOCATIONS
       "")
  set(DOXYGEN_PARSED_INPUT_DIRS
      " \"${CMAKE_CURRENT_BINARY_DIR}/dox\" \\
        \"${CMAKE_CURRENT_BINARY_DIR}/dox/Utilities/Xdmf2/vtk\" \\
        \"${CMAKE_CURRENT_BINARY_DIR}/dox/Utilities/VTKPythonWrapping/Executable\" \\
        \"${vtkgdcm_SOURCE_DIR}/Utilities/VTKClientServer\" \\
        \"${vtkgdcm_SOURCE_DIR}/Qt/Chart\" \\
        \"${vtkgdcm_SOURCE_DIR}/Qt/Components\" \\
        \"${vtkgdcm_SOURCE_DIR}/Qt/Core\" \\
        \"${vtkgdcm_SOURCE_DIR}/Qt/Python\" \\
        \"${vtkgdcm_SOURCE_DIR}/Qt/Widgets\" \\
        \"${vtkgdcm_SOURCE_DIR}/Qt/Testing\" \\
        \"${vtkgdcm_SOURCE_DIR}/Applications/Client\" \\
        \"${vtkgdcm_SOURCE_DIR}/Applications/OverView\" \\
        \"${vtkgdcm_SOURCE_DIR}/Applications/OverView/Core\" \\
        \"${CMAKE_CURRENT_BINARY_DIR}/dox/Qt/Core\" \\
        \"${CMAKE_CURRENT_BINARY_DIR}/dox/Common/KWCommon\" \\
        \"${CMAKE_CURRENT_BINARY_DIR}/dox/Servers/Common\" \\
        \"${CMAKE_CURRENT_BINARY_DIR}/dox/Servers/Filters\" \\
        \"${CMAKE_CURRENT_BINARY_DIR}/dox/Servers/ServerManager\"")


  set(DOXYGEN_IGNORE_PREFIX "vtkPV vtkSM vtk pq")

  #
  # Configure the script and the doxyfile, then add target
  #
  if(NOT DOT_PATH)
    get_filename_component(DOT_PATH ${DOT} PATH)
  endif()

  set(TAGFILES "\"${CMAKE_CURRENT_BINARY_DIR}/vtkNightlyDoc.tag=http://www.vtk.org/doc/nightly/html\"")
  if(INCLUDE_QT_DOCUMENTATION)
    set (TAGFILES "${TAGFILES} \"${QT_TAG_FILE}=http://doc.trolltech.com/4.2\"")
  endif()

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/doxyfile @ONLY )

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/doc_makeall.sh.in
    ${CMAKE_CURRENT_BINARY_DIR}/doc_makeall.sh @ONLY )

  add_custom_target(${DOXYGEN_PROJECT_NAME}DoxygenDoc
    #ALL
    ${BASH}
    ${CMAKE_CURRENT_BINARY_DIR}/doc_makeall.sh)

  add_dependencies(
    ${DOXYGEN_PROJECT_NAME}DoxygenDoc
    GDCMDoxygenPDF
  )

  if(INCLUDE_QT_DOCUMENTATION)
    add_custom_target(QtDoxygenTags
      COMMAND ${DOXYTAG} -t ${QT_TAG_FILE} ${QT_DOC_DIR})
    add_dependencies(${DOXYGEN_PROJECT_NAME}DoxygenDoc
      QtDoxygenTags)
  endif()

endif ()
