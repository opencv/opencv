#
#  CMake file for OpenCV docs
#

file(GLOB FILES_TEX *.tex *.sty *.bib)
file(GLOB FILES_TEX_PICS pics/*.png pics/*.jpg)

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


  set(OPENCV2_BASE_MODULES core imgproc highgui video calib3d features2d objdetect ml flann photo stitching nonfree contrib legacy bioinspired)

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
      set(OPENCV_REFMAN_TOC "${OPENCV_REFMAN_TOC}   ${toc_file}\n")
    endif()
  endforeach()

  file(GLOB_RECURSE _OPENCV_FILES_REF "${OpenCV_SOURCE_DIR}/platforms/android/service/doc/*.rst")
  file(GLOB_RECURSE _OPENCV_FILES_REF_PICT "${OpenCV_SOURCE_DIR}/platforms/android/service/doc/*.png" "${OpenCV_SOURCE_DIR}/platforms/android/service/doc/*.jpg")
  list(APPEND OPENCV_FILES_REF ${_OPENCV_FILES_REF})
  list(APPEND OPENCV_FILES_REF_PICT ${_OPENCV_FILES_REF_PICT})

  configure_file("${OpenCV_SOURCE_DIR}/modules/refman.rst.in" "${OpenCV_SOURCE_DIR}/modules/refman.rst" IMMEDIATE @ONLY)

  file(GLOB_RECURSE OPENCV_FILES_UG  user_guide/*.rst)
  file(GLOB_RECURSE OPENCV_FILES_TUT tutorials/*.rst)
  file(GLOB_RECURSE OPENCV_FILES_TUT_PICT tutorials/*.png tutorials/*.jpg)

  set(OPENCV_DOC_DEPS conf.py ${OPENCV_FILES_REF} ${OPENCV_FILES_REF_PICT}
           ${OPENCV_FILES_UG} ${OPENCV_FILES_TUT} ${OPENCV_FILES_TUT_PICT})

  set(BUILD_PLANTUML "")
  if(PLANTUML)
    set(BUILD_PLANTUML "-tplantuml")
  endif()

  if(PDFLATEX_COMPILER)
    add_custom_target(docs
      COMMAND ${SPHINX_BUILD} ${BUILD_PLANTUML} -b latex -c ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. .
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/pics ${CMAKE_CURRENT_BINARY_DIR}/doc/opencv1/pics
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/mymath.sty ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/patch_refman_latex.py" opencv2refman.tex
      COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/patch_refman_latex.py" opencv2manager.tex
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
      DEPENDS ${OPENCV_DOC_DEPS}
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
    COMMAND ${SPHINX_BUILD} ${BUILD_PLANTUML} -b html -c ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ./_html
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/mymath.sty ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${OPENCV_DOC_DEPS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating Online Documentation"
    )

  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(html_docs PROPERTIES FOLDER "documentation")
  endif()

  foreach(f ${DOC_LIST})
    install(FILES "${f}" DESTINATION "${OPENCV_DOC_INSTALL_PATH}" COMPONENT main)
  endforeach()

  foreach(f ${OPTIONAL_DOC_LIST})
    install(FILES "${f}" DESTINATION "${OPENCV_DOC_INSTALL_PATH}" OPTIONAL)
  endforeach()

endif()
