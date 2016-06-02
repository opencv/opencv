# helper file for Android samples build

file(GLOB_RECURSE LIBS RELATIVE ${SRC_DIR} "*.so")

foreach(l ${LIBS})
  message(STATUS "  Copying: ${l} ...")
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_DIR}/${l} ${DST_DIR}/${l})
endforeach()
