file(GLOB sources "*.h" "*.c" "*.hpp" "*.cpp")

add_library(utility STATIC ${sources})

target_link_libraries(utility ${OpenCV_LIBS})
