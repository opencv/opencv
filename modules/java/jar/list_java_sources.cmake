file(GLOB_RECURSE java_sources "${OPENCV_JAVA_DIR}/*.java")

set(__sources "")

foreach(dst ${java_sources})
    set(__sources "${__sources}${dst}\n")
endforeach()

function(ocv_update_file filepath content)
  if(EXISTS "${filepath}")
    file(READ "${filepath}" actual_content)
  else()
    set(actual_content "")
  endif()
  if(NOT ("${actual_content}" STREQUAL "${content}"))
    file(WRITE "${filepath}" "${content}")
  endif()
endfunction()
ocv_update_file("${OUTPUT}" "${__sources}")
