# glob the source files
file(GLOB SOURCE_FILES *.cpp)

# compile each source
foreach(SOURCE_FILE_ ${SOURCE_FILES})
  get_filename_component(SOURCE_FILE_NAME_ ${SOURCE_FILE_} NAME_WE)
  add_custom_target(${SOURCE_FILE_NAME_} ALL ${MATLAB_MEX_SCRIPT} ${SOURCE_FILE_})
endforeach()
