cmake_minimum_required(VERSION 2.8)
project( PNP_DEMO )

ADD_DEFINITIONS(
    -std=c++11 
    -std=c++0x
    # Other flags
)

find_package( OpenCV REQUIRED )

include_directories( 
	$(PNP_DEMO_SOURCE_DIR)/include 
	${OpenCV_INCLUDE_DIRS} 
)

add_executable( 
pnp_registration 
	src/main_registration.cpp
	src/CsvReader.cpp 
	src/CsvWriter.cpp
	src/ModelRegistration.cpp
	src/Mesh.cpp
	src/Model.cpp
	src/PnPProblem.cpp
	src/Utils.cpp
	src/RobustMatcher.cpp
)

add_executable( 
pnp_verification 
	src/main_verification.cpp
	src/CsvReader.cpp 
	src/CsvWriter.cpp
	src/ModelRegistration.cpp
	src/Mesh.cpp
	src/Model.cpp
	src/PnPProblem.cpp
	src/Utils.cpp
	src/RobustMatcher.cpp
)

add_executable( 
pnp_detection 
	src/main_detection.cpp
	src/CsvReader.cpp 
	src/CsvWriter.cpp
	src/ModelRegistration.cpp
	src/Mesh.cpp
	src/Model.cpp
	src/PnPProblem.cpp
	src/Utils.cpp
	src/RobustMatcher.cpp
)

add_executable( 
pnp_test 
	src/test_pnp.cpp
)

target_link_libraries( pnp_registration ${OpenCV_LIBS} )
target_link_libraries( pnp_verification ${OpenCV_LIBS} )
target_link_libraries( pnp_detection ${OpenCV_LIBS} )
target_link_libraries( pnp_test ${OpenCV_LIBS} )
