cmake_minimum_required(VERSION 2.8)
project( PNP_DEMO )

find_package( OpenCV REQUIRED )

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_library(pnp_lib
    src/CsvReader.cpp
    src/CsvWriter.cpp
    src/ModelRegistration.cpp
    src/Mesh.cpp
    src/Model.cpp
    src/PnPProblem.cpp
    src/Utils.cpp
    src/RobustMatcher.cpp)

add_executable( pnp_registration src/main_registration.cpp )
add_executable( pnp_verification src/main_verification.cpp )
add_executable( pnp_detection src/main_detection.cpp )
add_executable( pnp_test src/test_pnp.cpp )

target_link_libraries( pnp_registration pnp_lib ${OpenCV_LIBS} )
target_link_libraries( pnp_verification pnp_lib ${OpenCV_LIBS} )
target_link_libraries( pnp_detection pnp_lib ${OpenCV_LIBS} )
target_link_libraries( pnp_test pnp_lib ${OpenCV_LIBS} )
