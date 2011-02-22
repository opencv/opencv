if(BUILD_REFMAN AND PDFLATEX_COMPILER AND HAVE_SPHINX)

project(opencv_refman)

file(GLOB_RECURSE FILES_PICT ../modules/*.png ../modules/*.jpg)
file(GLOB_RECURSE FILES_RST *.rst ../modules/*.rst)

add_custom_target(refman ${SPHINX_BUILD} -b latex -c ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../modules .
                  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/doc/pics ${CMAKE_CURRENT_BINARY_DIR}/pics
                  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/doc/latex2sphinx/mymath.sty ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${PDFLATEX_COMPILER} opencv
                  COMMAND ${PDFLATEX_COMPILER} opencv
                  DEPENDS conf.py ${FILES_RST} ${FILES_PICT}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMENT "Generating the Reference Manual")

install(FILES ${CURRENT_BINARY_DIR}/opencv.pdf DESTINATION "${OPENCV_DOC_INSTALL_PATH}" COMPONENT main)

endif()
