set(HAVE_FLATBUFFERS FALSE)

if(NOT WITH_FLATBUFFERS)
  return()
endif()

list(APPEND CUSTOM_STATUS flatbuffers)

find_package(flatbuffers QUIET)
if(flatbuffers_FOUND)
  set(HAVE_FLATBUFFERS 1)
  list(APPEND CUSTOM_STATUS_flatbuffers "    FlatBuffers:" "${flatbuffers_VERSION}")
else()
  list(APPEND CUSTOM_STATUS_flatbuffers "    FlatBuffers:" "NO")
endif()
