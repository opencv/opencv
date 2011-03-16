project(opencv_refman)

file(GLOB_RECURSE OPENCV2_FILES_PICT ../../modules/*.png ../../modules/*.jpg)
file(GLOB_RECURSE OPENCV2_FILES_RST ../../modules/*.rst)

add_custom_target(refman
                  ${SPHINX_BUILD}
                    -b latex -c ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../modules .
                  COMMAND ${CMAKE_COMMAND} -E copy
                    ${CMAKE_CURRENT_SOURCE_DIR}/../mymath.sty ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${PDFLATEX_COMPILER} opencv
                  COMMAND ${PDFLATEX_COMPILER} opencv
                  DEPENDS conf.py ${OPENCV2_FILES_RST} ${OPENCV2_FILES_PICT}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMENT "Generating the OpenCV Reference Manual")

#install(FILES ${CURRENT_BINARY_DIR}/opencv.pdf DESTINATION "${OPENCV_DOC_INSTALL_PATH}" COMPONENT main)
