cmake_minimum_required(VERSION 2.8)
project( Qt_sample )
FIND_PACKAGE(Qt4)

if(Qt4_FOUND)

  include_directories( 	${OpenCV_INCLUDE_DIRS}
						  ${QT_INCLUDES}
						  ./
						  ${CMAKE_CURRENT_BINARY_DIR}
						  )

  INCLUDE(${QT_USE_FILE})
  ADD_DEFINITIONS(${QT_DEFINITIONS})


  FILE( GLOB source_files ./*.cpp )
  FILE( GLOB moc_files ./*.h )
  FILE( GLOB ui_files ./*.ui )
  #QT4_ADD_RESOURCES(QtApp_RCC_SRCS resources/pictures.qrc)

  QT4_WRAP_UI(SOURCES_UI_HEADERS ${ui_files})
  QT4_WRAP_CPP(MOC ${moc_files})


  add_executable( Qt_sample ${MOC} ${source_files} ${SOURCES_UI_HEADERS})
  target_link_libraries( Qt_sample ${OPENCV_LINKER_LIBS} ${OPENCV_CPP_SAMPLES_REQUIRED_DEPS} ${QT_LIBRARIES}) 
endif()