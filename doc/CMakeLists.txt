#
#  CMake file for OpenCV docs
#

file(GLOB FILES_DOC *.htm *.txt *.jpg *.png *.pdf)
file(GLOB FILES_DOC_VS vidsurv/*.doc)
file(GLOB FILES_TEX *.tex *.sty *.bib)
file(GLOB FILES_TEX_PICS pics/*.png pics/*.jpg)

if(BUILD_DOCS AND HAVE_SPHINX)

  project(opencv_docs)

  set(OPENCV2_BASE_MODULES core imgproc highgui video calib3d features2d objdetect ml flann gpu photo stitching nonfree contrib legacy)

  # build lists of modules to be documented
  set(OPENCV2_MODULES "")
  set(OPENCV_MODULES "")

  foreach(mod ${OPENCV_MODULES_BUILD} ${OPENCV_MODULES_DISABLED_USER} ${OPENCV_MODULES_DISABLED_AUTO} ${OPENCV_MODULES_DISABLED_FORCE})
    string(REGEX REPLACE "^opencv_" "" mod "${mod}")
    if("${OPENCV_MODULE_opencv_${mod}_LOCATION}" STREQUAL "${OpenCV_SOURCE_DIR}/modules/${mod}")
      list(APPEND OPENCV2_MODULES ${mod})
    else()
      list(APPEND OPENCV_MODULES ${mod})
    endif()
  endforeach()
  list(REMOVE_ITEM OPENCV2_MODULES ${OPENCV2_BASE_MODULES})
  ocv_list_sort(OPENCV2_MODULES)
  ocv_list_sort(OPENCV_MODULES)

  # build lists of documentation files and generate table of contents for reference manual
  set(OPENCV_FILES_REF "")
  set(OPENCV_FILES_REF_PICT "")
  set(OPENCV_REFMAN_TOC "")

  foreach(mod ${OPENCV2_BASE_MODULES} ${OPENCV2_MODULES} ${OPENCV_MODULES})
    file(GLOB_RECURSE _OPENCV_FILES_REF "${OPENCV_MODULE_opencv_${mod}_LOCATION}/doc/*.rst")
    file(GLOB_RECURSE _OPENCV_FILES_REF_PICT "${OPENCV_MODULE_opencv_${mod}_LOCATION}/doc/*.png" "${OPENCV_MODULE_opencv_${mod}_LOCATION}/doc/*.jpg")
    list(APPEND OPENCV_FILES_REF ${_OPENCV_FILES_REF})
    list(APPEND OPENCV_FILES_REF_PICT ${_OPENCV_FILES_REF_PICT})

    set(toc_file "${OPENCV_MODULE_opencv_${mod}_LOCATION}/doc/${mod}.rst")
    if(EXISTS "${toc_file}")
      file(RELATIVE_PATH toc_file "${OpenCV_SOURCE_DIR}/modules" "${toc_file}")
      set(OPENCV_REFMAN_TOC "${OPENCV_REFMAN_TOC}   ${toc_file}\r\n")
    endif()
  endforeach()

  configure_file("${OpenCV_SOURCE_DIR}/modules/refman.rst.in" "${OpenCV_SOURCE_DIR}/modules/refman.rst" IMMEDIATE @ONLY)

  file(GLOB_RECURSE OPENCV_FILES_UG user_guide/*.rst)
  file(GLOB_RECURSE OPENCV_FILES_TUT tutorials/*.rst)
  file(GLOB_RECURSE OPENCV_FILES_TUT_PICT tutorials/*.png tutorials/*.jpg)

  set(OPENCV_DOC_DEPS conf.py ${OPENCV_FILES_REF} ${OPENCV_FILES_REF_PICT}
           ${OPENCV_FILES_UG} ${OPENCV_FILES_TUT} ${OPENCV_FILES_TUT_PICT})

  if(PDFLATEX_COMPILER)
    add_custom_target(docs
      COMMAND ${SPHINX_BUILD} -b latex -c ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. .
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/pics ${CMAKE_CURRENT_BINARY_DIR}/doc/opencv1/pics
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/mymath.sty ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/patch_refman_latex.py" opencv2refman.tex
      COMMAND ${PDFLATEX_COMPILER} opencv2refman.tex
      COMMAND ${PDFLATEX_COMPILER} opencv2refman.tex
      COMMAND ${PDFLATEX_COMPILER} opencv_user.tex
      COMMAND ${PDFLATEX_COMPILER} opencv_user.tex
      COMMAND ${PDFLATEX_COMPILER} opencv_tutorials.tex
      COMMAND ${PDFLATEX_COMPILER} opencv_tutorials.tex
      DEPENDS ${OPENCV_DOC_DEPS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating the PDF Manuals"
      )

    if(ENABLE_SOLUTION_FOLDERS)
      set_target_properties(docs PROPERTIES FOLDER "documentation")
    endif()
  endif()

  add_custom_target(html_docs
    COMMAND ${SPHINX_BUILD} -b html -c ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ./_html
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/mymath.sty ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${OPENCV_DOC_DEPS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating Online Documentation"
    )

  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(html_docs PROPERTIES FOLDER "documentation")
  endif()
endif()

install(FILES ${FILES_DOC} DESTINATION "${OPENCV_DOC_INSTALL_PATH}" COMPONENT main)
install(FILES ${FILES_DOC_VS} DESTINATION "${OPENCV_DOC_INSTALL_PATH}/vidsurv" COMPONENT main)
