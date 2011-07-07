#
#  CMake file for OpenCV docs
#

file(GLOB FILES_DOC *.htm *.txt *.jpg *.png *.pdf)
file(GLOB FILES_DOC_VS vidsurv/*.doc)
file(GLOB FILES_TEX *.tex *.sty *.bib)
file(GLOB FILES_TEX_PICS pics/*.png pics/*.jpg)

if(BUILD_DOCS AND PDFLATEX_COMPILER AND HAVE_SPHINX)

project(opencv_docs)

file(GLOB_RECURSE OPENCV_FILES_REF ../modules/*.rst)
file(GLOB_RECURSE OPENCV_FILES_REF_PICT ../modules/*.png ../modules/*.jpg)
file(GLOB_RECURSE OPENCV_FILES_UG user_guide/*.rst)
file(GLOB_RECURSE OPENCV_FILES_TUT tutorials/*.rst)
file(GLOB_RECURSE OPENCV_FILES_TUT_PICT tutorials/*.png tutorials/*.jpg)

set(OPENCV_DOC_DEPS conf.py ${OPENCV_FILES_REF} ${OPENCV_FILES_REF_PICT}
          ${OPENCV_FILES_UG} ${OPENCV_FILES_TUT} ${OPENCV_FILES_TUT_PICT})

add_custom_target(docs
                  ${SPHINX_BUILD}
                    -b latex -c ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/.. .
                  COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_CURRENT_SOURCE_DIR}/pics ${CMAKE_CURRENT_BINARY_DIR}/doc/opencv1/pics
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${CMAKE_CURRENT_SOURCE_DIR}/mymath.sty ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${PDFLATEX_COMPILER} opencv2refman.tex
                  COMMAND ${PDFLATEX_COMPILER} opencv2refman.tex
                  COMMAND ${PDFLATEX_COMPILER} opencv_user.tex
                  COMMAND ${PDFLATEX_COMPILER} opencv_user.tex
                  COMMAND ${PDFLATEX_COMPILER} opencv_tutorials.tex
                  COMMAND ${PDFLATEX_COMPILER} opencv_tutorials.tex
                  DEPENDS ${OPENCV_DOC_DEPS}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMENT "Generating the PDF Manuals")

add_custom_target(html_docs
                ${SPHINX_BUILD}
                  -b html -c ${CMAKE_CURRENT_SOURCE_DIR}
                  ${CMAKE_CURRENT_SOURCE_DIR}/.. ./_html
                COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_SOURCE_DIR}/mymath.sty ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS ${OPENCV_DOC_DEPS}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Generating Online Documentation")

endif()
        
install(FILES ${FILES_DOC} DESTINATION "${OPENCV_DOC_INSTALL_PATH}" COMPONENT main)
install(FILES ${FILES_DOC_VS} DESTINATION "${OPENCV_DOC_INSTALL_PATH}/vidsurv" COMPONENT main)

########################################### Doxygen Docs #########################################

if(0)
#if(BUILD_DOXYGEN_DOCS AND DOXYGEN_FOUND)

set(DOXYGEN_LANGUAGE "English" CACHE STRING "Language used by doxygen")
mark_as_advanced(DOXYGEN_LANGUAGE)

set(CMAKE_DOXYGEN_IMAGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/pics")

set(module_names core imgproc calib3d features2d objdetect video highgui ml legacy contrib)
set(DOXYGEN_SOURCE_DIR)
set(doxy_html_dep_list)

foreach(m ${module_names})
    set(module_hdr_path "${OpenCV_SOURCE_DIR}/modules/${m}/include/opencv2/${m}")
    set(DOXYGEN_SOURCE_DIR ${DOXYGEN_SOURCE_DIR} ${module_hdr_path})
    file(GLOB module_hdrs "${module_hdr_path}/*.h*")
    set(doxy_html_dep_list ${doxy_html_dep_list} ${module_hdrs})
endforeach()

set(DOXYGEN_OUTPUT_DIR html)
string(REGEX REPLACE ";" " " CMAKE_DOXYGEN_INPUT_LIST "${DOXYGEN_SOURCE_DIR}")

configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
set(HTML_TARGET "DOXY_HTML" )

add_custom_target(${HTML_TARGET} ALL
	${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
	DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile ${doxy_html_dep_list})
exec_program(mkdir ARGS "-p \"${CMAKE_CURRENT_BINARY_DIR}/html/\"")
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html" DESTINATION "${OPENCV_DOC_INSTALL_PATH}/html" COMPONENT main)

endif()
