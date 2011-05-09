project(opencv_refman1)

file(GLOB_RECURSE OPENCV1_FILES_PICT pics/*.png pics/*.jpg)
file(GLOB_RECURSE OPENCV1_FILES_RST *.rst)

add_custom_target(refman1
                  ${SPHINX_BUILD}
                    -b latex -c ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR} .
                  COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_CURRENT_SOURCE_DIR}/../pics ${CMAKE_CURRENT_BINARY_DIR}/pics
                  COMMAND ${CMAKE_COMMAND} -E copy
                    ${CMAKE_CURRENT_SOURCE_DIR}/../mymath.sty ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${PDFLATEX_COMPILER} opencv1x
                  COMMAND ${PDFLATEX_COMPILER} opencv1x
                  DEPENDS conf.py ${OPENCV1_FILES_RST} ${OPENCV1_FILES_PICT}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMENT "Generating the OpenCV 1.x Reference Manual")

#install(FILES ${CURRENT_BINARY_DIR}/opencv1x.pdf DESTINATION "${OPENCV_DOC_INSTALL_PATH}" COMPONENT main)
