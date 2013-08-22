file(GLOB cl_list "${CL_DIR}/*.cl" )

file(WRITE ${OUTPUT} "// This file is auto-generated. Do not edit!

namespace cv
{
namespace ocl
{
")

foreach(cl ${cl_list})
  get_filename_component(cl_filename "${cl}" NAME_WE)
  #message("${cl_filename}")

  file(READ "${cl}" lines)

  string(REPLACE "\r" "" lines "${lines}\n")
  string(REPLACE "\t" "  " lines "${lines}")

  string(REGEX REPLACE "/\\*([^*]/|\\*[^/]|[^*/])*\\*/" ""   lines "${lines}") # multiline comments
  string(REGEX REPLACE "/\\*([^\n])*\\*/"               ""   lines "${lines}") # single-line comments
  string(REGEX REPLACE "[ ]*//[^\n]*\n"                 "\n" lines "${lines}") # single-line comments
  string(REGEX REPLACE "\n[ ]*(\n[ ]*)*"                "\n" lines "${lines}") # empty lines & leading whitespace
  string(REGEX REPLACE "^\n"                            ""   lines "${lines}") # leading new line

  string(REPLACE "\\" "\\\\" lines "${lines}")
  string(REPLACE "\"" "\\\"" lines "${lines}")
  string(REPLACE "\n" "\\n\"\n\"" lines "${lines}")

  string(REGEX REPLACE "\"$" "" lines "${lines}") # unneeded " at the eof

  file(APPEND ${OUTPUT} "const char* ${cl_filename}=\"${lines};\n")
endforeach()

file(APPEND ${OUTPUT} "}\n}\n")
