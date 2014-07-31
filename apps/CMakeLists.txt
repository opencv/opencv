add_definitions(-D__OPENCV_BUILD=1)
link_libraries(${OPENCV_LINKER_LIBS})

add_subdirectory(traincascade)
