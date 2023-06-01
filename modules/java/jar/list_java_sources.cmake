file(GLOB_RECURSE java_sources "${OPENCV_JAVA_DIR}/java/*.java")

set(__sources "")

foreach(dst ${java_sources})
    set(__sources "${__sources}${dst}\n")
endforeach()

file(WRITE "${OUTPUT}" "${__sources}")
