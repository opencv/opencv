#
#  CMake file for OpenCV docs
#

if(BUILD_DOCS AND HAVE_SPHINX)

  project(opencv_docs)

  set(DOC_LIST
      "${OpenCV_SOURCE_DIR}/doc/opencv-logo.png"
      "${OpenCV_SOURCE_DIR}/doc/opencv-logo2.png"
      "${OpenCV_SOURCE_DIR}/doc/opencv-logo-white.png"
      "${OpenCV_SOURCE_DIR}/doc/opencv.ico"
      "${OpenCV_SOURCE_DIR}/doc/pattern.png"
      "${OpenCV_SOURCE_DIR}/doc/acircles_pattern.png")
  if(NOT INSTALL_CREATE_DISTRIB)
    list(APPEND DOC_LIST "${OpenCV_SOURCE_DIR}/doc/haartraining.htm")
  endif()

  set(OPTIONAL_DOC_LIST "")


  # build lists of modules to be documented
  set(BASE_MODULES "")
  set(EXTRA_MODULES "")

  foreach(mod ${OPENCV_MODULES_BUILD} ${OPENCV_MODULES_DISABLED_USER} ${OPENCV_MODULES_DISABLED_AUTO} ${OPENCV_MODULES_DISABLED_FORCE})
    string(REGEX REPLACE "^opencv_" "" mod "${mod}")
    if("${OPENCV_MODULE_opencv_${mod}_LOCATION}" STREQUAL "${OpenCV_SOURCE_DIR}/modules/${mod}")
      list(APPEND BASE_MODULES ${mod})
    else()
      list(APPEND EXTRA_MODULES ${mod})
    endif()
  endforeach()

  set(FIXED_ORDER_MODULES core imgproc imgcodecs videoio highgui video calib3d features2d objdetect ml flann photo stitching nonfree contrib legacy)

  list(REMOVE_ITEM BASE_MODULES ${FIXED_ORDER_MODULES})

  ocv_list_sort(BASE_MODULES)
  ocv_list_sort(EXTRA_MODULES)

  set(BASE_MODULES ${FIXED_ORDER_MODULES} ${BASE_MODULES})

  # build lists of documentation files and generate table of contents for reference manual

  set(DOC_FAKE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/fake-root")
  set(DOC_FAKE_ROOT_FILES "")

  function(ocv_doc_add_file_to_fake_root source destination)
    add_custom_command(
      OUTPUT "${DOC_FAKE_ROOT}/${destination}"
      COMMAND "${CMAKE_COMMAND}" -E copy "${source}" "${DOC_FAKE_ROOT}/${destination}"
      DEPENDS "${source}"
      COMMENT "Copying ${destination} to fake root..."
      VERBATIM
    )
    list(APPEND DOC_FAKE_ROOT_FILES "${DOC_FAKE_ROOT}/${destination}")
    set(DOC_FAKE_ROOT_FILES "${DOC_FAKE_ROOT_FILES}" PARENT_SCOPE)
  endfunction()

  function(ocv_doc_add_to_fake_root source)
    if(ARGC GREATER 1)
      set(destination "${ARGV1}")
    else()
      file(RELATIVE_PATH destination "${OpenCV_SOURCE_DIR}" "${source}")
    endif()

    if(IS_DIRECTORY "${source}")
      file(GLOB_RECURSE files RELATIVE "${source}" "${source}/*")

      foreach(file ${files})
        ocv_doc_add_file_to_fake_root("${source}/${file}" "${destination}/${file}")
      endforeach()
    else()
      ocv_doc_add_file_to_fake_root("${source}" "${destination}")
    endif()

    set(DOC_FAKE_ROOT_FILES "${DOC_FAKE_ROOT_FILES}" PARENT_SCOPE)
  endfunction()

  set(OPENCV_REFMAN_TOC "")

  foreach(mod ${BASE_MODULES} ${EXTRA_MODULES})
    if(EXISTS "${OPENCV_MODULE_opencv_${mod}_LOCATION}/doc/${mod}.rst")
      ocv_doc_add_to_fake_root("${OPENCV_MODULE_opencv_${mod}_LOCATION}/doc" modules/${mod}/doc)
      set(OPENCV_REFMAN_TOC "${OPENCV_REFMAN_TOC}   ${mod}/doc/${mod}.rst\n")
    endif()
  endforeach()

  configure_file("${OpenCV_SOURCE_DIR}/modules/refman.rst.in" "${DOC_FAKE_ROOT}/modules/refman.rst" @ONLY)

  ocv_doc_add_to_fake_root("${OpenCV_SOURCE_DIR}/index.rst")
  ocv_doc_add_to_fake_root("${OpenCV_SOURCE_DIR}/doc")
  ocv_doc_add_to_fake_root("${OpenCV_SOURCE_DIR}/platforms/android")
  ocv_doc_add_to_fake_root("${OpenCV_SOURCE_DIR}/samples")

  set(BUILD_PLANTUML "")
  if(PLANTUML)
    set(BUILD_PLANTUML "-tplantuml")
  endif()

  if(PDFLATEX_COMPILER)
    add_custom_target(docs
      COMMAND ${SPHINX_BUILD} ${BUILD_PLANTUML} -b latex -c "${CMAKE_CURRENT_SOURCE_DIR}" "${DOC_FAKE_ROOT}" .
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/pics ${CMAKE_CURRENT_BINARY_DIR}/doc/opencv1/pics
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/mymath.sty ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND ${PYTHON_DEFAULT_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/patch_refman_latex.py" opencv2refman.tex
      COMMAND ${PYTHON_DEFAULT_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/patch_refman_latex.py" opencv2manager.tex
      COMMAND ${CMAKE_COMMAND} -E echo "Generating opencv2refman.pdf"
      COMMAND ${PDFLATEX_COMPILER} -interaction=batchmode opencv2refman.tex
      COMMAND ${PDFLATEX_COMPILER} -interaction=batchmode opencv2refman.tex
      COMMAND ${CMAKE_COMMAND} -E echo "Generating opencv2manager.pdf"
      COMMAND ${PDFLATEX_COMPILER} -interaction=batchmode opencv2manager.tex
      COMMAND ${PDFLATEX_COMPILER} -interaction=batchmode opencv2manager.tex
      COMMAND ${CMAKE_COMMAND} -E echo "Generating opencv_user.pdf"
      COMMAND ${PDFLATEX_COMPILER} -interaction=batchmode opencv_user.tex
      COMMAND ${PDFLATEX_COMPILER} -interaction=batchmode opencv_user.tex
      COMMAND ${CMAKE_COMMAND} -E echo "Generating opencv_tutorials.pdf"
      COMMAND ${PDFLATEX_COMPILER} -interaction=batchmode opencv_tutorials.tex
      COMMAND ${PDFLATEX_COMPILER} -interaction=batchmode opencv_tutorials.tex
      COMMAND ${CMAKE_COMMAND} -E echo "Generating opencv_cheatsheet.pdf"
      COMMAND ${PDFLATEX_COMPILER} -interaction=batchmode "${CMAKE_CURRENT_SOURCE_DIR}/opencv_cheatsheet.tex"
      COMMAND ${PDFLATEX_COMPILER} -interaction=batchmode "${CMAKE_CURRENT_SOURCE_DIR}/opencv_cheatsheet.tex"
      DEPENDS ${DOC_FAKE_ROOT_FILES}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating the PDF Manuals"
      )

    LIST(APPEND OPTIONAL_DOC_LIST "${CMAKE_BINARY_DIR}/doc/opencv2refman.pdf" "${CMAKE_BINARY_DIR}/doc/opencv2manager.pdf"
          "${CMAKE_BINARY_DIR}/doc/opencv_user.pdf" "${CMAKE_BINARY_DIR}/doc/opencv_tutorials.pdf" "${CMAKE_BINARY_DIR}/doc/opencv_cheatsheet.pdf")

    if(ENABLE_SOLUTION_FOLDERS)
      set_target_properties(docs PROPERTIES FOLDER "documentation")
    endif()
  endif()

  add_custom_target(html_docs
    COMMAND "${SPHINX_BUILD}" ${BUILD_PLANTUML} -b html -c "${CMAKE_CURRENT_SOURCE_DIR}" "${DOC_FAKE_ROOT}" ./_html
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/mymath.sty ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${DOC_FAKE_ROOT_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating Online Documentation"
    )

  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(html_docs PROPERTIES FOLDER "documentation")
  endif()

  foreach(f ${DOC_LIST})
    install(FILES "${f}" DESTINATION "${OPENCV_DOC_INSTALL_PATH}" COMPONENT docs)
  endforeach()

  foreach(f ${OPTIONAL_DOC_LIST})
    install(FILES "${f}" DESTINATION "${OPENCV_DOC_INSTALL_PATH}" OPTIONAL COMPONENT docs)
  endforeach()

endif()
