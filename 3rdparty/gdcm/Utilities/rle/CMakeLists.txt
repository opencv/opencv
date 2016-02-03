project(RLE)

add_library(rle rle.c rlelib.c)
#set_target_properties(rle PROPERTIES ${GDCM_LIBRARY_PROPERTIES})

add_executable(example example.c)
target_link_libraries(example rle)
install(TARGETS rle example
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/static
  ${CPACK_NAMELINK_TYPE}
  )

add_executable(rledump rledump.c)
target_link_libraries(rledump rle)
