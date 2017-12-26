add_definitions(-D__OPENCV_BUILD=1)
add_definitions(-D__OPENCV_APPS=1)

link_libraries(${OPENCV_LINKER_LIBS})

add_subdirectory(traincascade)
add_subdirectory(createsamples)
add_subdirectory(annotation)
add_subdirectory(visualisation)
add_subdirectory(interactive-calibration)
add_subdirectory(version)
