#
#  CMake file for OpenCV docs
#

file(GLOB FILES_DOC *.htm *.txt *.jpg *.png *.pdf)
file(GLOB FILES_DOC_PAPERS papers/*.pdf papers/*.ps)
file(GLOB FILES_DOC_VS vidsurv/*.doc)
file(GLOB FILES_TEX *.tex *.sty *.bib go)
file(GLOB FILES_TEX_PICS pics/*.png pics/*.jpg)
file(GLOB_RECURSE FILES_PLASTEX plastex/*)

if(BUILD_LATEX_DOCS AND PDFLATEX_COMPILER)
        
project(opencv_refman)

set(refman_copied_srcs)

#foreach(f ${refman_m4_srcs})
#    get_filename_component(basef ${f} NAME_WE)
#    set(outputf ${CMAKE_CURRENT_BINARY_DIR}/${basef}-c.tex)
#    add_custom_command(
#        OUTPUT ${outputf}
#        DEPENDS ${f}
#        COMMAND m4
#        ARGS -DTARGET_LANGUAGE=c ${f} > ${outputf}
#        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#        VERBATIM
#        )
#    list(APPEND refman_copied_srcs ${outputf})
#endforeach()

foreach(f ${FILES_TEX})
    get_filename_component(basef ${f} NAME)
    set(outputf ${CMAKE_CURRENT_BINARY_DIR}/${basef})
    add_custom_command(
        OUTPUT ${outputf}
        DEPENDS ${f}
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy ${f} ${outputf}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    list(APPEND refman_copied_srcs ${outputf})
endforeach()

set(pdflatex_args -interaction=batchmode
"\"\\newcommand{\\genc}{true}\\newcommand{\\genpy}{true}\\newcommand{\\targetlang}{c}\\input{opencv.tex}\"")

add_custom_command(
  OUTPUT    ${CMAKE_CURRENT_BINARY_DIR}/opencv.aux
  DEPENDS   ${refman_copied_srcs}
  COMMAND   ${PDFLATEX_COMPILER}
  ARGS      ${pdflatex_args}
  COMMENT   "Latex (first pass)"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

add_custom_command(
  OUTPUT    ${CMAKE_CURRENT_BINARY_DIR}/opencv.bbl
  DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/opencv.aux
  COMMAND   ${BIBTEX_COMPILER}
  ARGS      ${CMAKE_CURRENT_BINARY_DIR}/opencv
  COMMENT   "Bibtex"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

add_custom_command(
  OUTPUT    ${CMAKE_CURRENT_BINARY_DIR}/opencv.pdf
  DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/opencv.bbl
  COMMAND   ${PDFLATEX_COMPILER}
  ARGS      ${pdflatex_args}
  COMMENT   "Latex (second pass)"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

add_custom_command(
  OUTPUT    ${CMAKE_CURRENT_BINARY_DIR}/opencv.log
  DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/opencv.bbl
            ${CMAKE_CURRENT_BINARY_DIR}/opencv.pdf
  COMMAND   ${PDFLATEX_COMPILER}
  ARGS      ${pdflatex_args}
  COMMENT   "Latex (third pass)"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

set(PICS_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/pics")
message(STATUS "Copying pictures to the build directory")
make_directory(${PICS_OUT_DIR})

file(GLOB pics RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/pics" pics/*.jpg pics/*.png)

foreach(p ${pics})
    configure_file(
           "pics/${p}"
           "${PICS_OUT_DIR}/${p}"
           COPYONLY)
endforeach()

set(PYTHON_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_fragments")
message(STATUS "Copying python_fragments to the build directory")
make_directory(${PYTHON_OUT_DIR})

file(GLOB python_fragments RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/python_fragments" python_fragments/*.py)

foreach(p ${python_fragments})
    configure_file(
           "python_fragments/${p}"
           "${PYTHON_OUT_DIR}/${p}"
           COPYONLY)
endforeach()

#add_custom_target(MakePicsLink
#   COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/pics ${CMAKE_CURRENT_BINARY_DIR}/pics
#   )
  
# Eventually trigger the whole process
add_custom_target(ReferenceManual ALL echo
   DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/opencv.log
   )

#add_dependencies(ReferenceManual MakePicsLink)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/opencv.pdf"
        DESTINATION "${OPENCV_DOC_INSTALL_PATH}" COMPONENT main)
        
endif() # BUILD_LATEX_DOCS AND PDFLATEX_COMPILER
        
install(FILES ${FILES_DOC} DESTINATION "${OPENCV_DOC_INSTALL_PATH}" COMPONENT main)
install(FILES ${FILES_DOC_PAPERS} DESTINATION "${OPENCV_DOC_INSTALL_PATH}/papers" COMPONENT main)
install(FILES ${FILES_DOC_VS} DESTINATION "${OPENCV_DOC_INSTALL_PATH}/vidsurv" COMPONENT main)
