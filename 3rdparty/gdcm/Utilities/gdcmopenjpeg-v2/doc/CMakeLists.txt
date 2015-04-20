find_package(Doxygen REQUIRED)

# The Doxyfile.dox is poorly defined and produce output
# in the source dir
add_custom_target(doxygen
# By default doxygen target is added to the 'all' target. Project is small
# thus running doxygen is not too time consuming
  ALL
  ${DOXYGEN}
  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.dox
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
