# jp3d

# Headers file are located here:
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/codec
  ${CMAKE_CURRENT_SOURCE_DIR}/libjp3dvm
  )


subdirs(
  libjp3dvm
  codec
)


