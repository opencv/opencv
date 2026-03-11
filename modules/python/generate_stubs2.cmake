add_custom_target(enable_doxygen_xml)
add_custom_command(
    TARGET enable_doxygen_xml
    POST_BUILD
    COMMAND ${PYTHON_DEFAULT_EXECUTABLE} ${PYTHON_SOURCE_DIR}/src2/enable_doxygen_xml.py
    "${CMAKE_BINARY_DIR}/doc/Doxyfile"
)

if(TARGET doxygen)
    add_dependencies(doxygen enable_doxygen_xml)
endif()

add_custom_target(generate_opencv_python_stubs2 ALL)
add_dependencies(generate_opencv_python_stubs2 enable_doxygen_xml doxygen opencv_python3)
add_custom_command(
    TARGET generate_opencv_python_stubs2
    POST_BUILD
    COMMAND ${PYTHON_DEFAULT_EXECUTABLE} ${PYTHON_SOURCE_DIR}/src2/types-opencv-python/gen_stubs2.py
        "${CMAKE_BINARY_DIR}"
        "${CMAKE_BINARY_DIR}/types-opencv-python/genout"
        "${CMAKE_BINARY_DIR}/lib/python3"
)
add_custom_command(
    TARGET generate_opencv_python_stubs2
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_BINARY_DIR}/types-opencv-python/genout/cv2"
        "${OPENCV_PYTHON_BINDINGS_DIR}/cv2"
)

add_custom_command(
    TARGET generate_opencv_python_stubs2
    POST_BUILD
    COMMAND ${PYTHON_DEFAULT_EXECUTABLE} ${PYTHON_SOURCE_DIR}/src2/cleanup_stubs.py "${OPENCV_PYTHON_BINDINGS_DIR}/cv2"
)
