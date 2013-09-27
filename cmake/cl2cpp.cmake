file(GLOB cl_list "${CL_DIR}/*.cl" )
list(SORT cl_list)

string(REPLACE ".cpp" ".hpp" OUTPUT_HPP "${OUTPUT}")
get_filename_component(OUTPUT_HPP_NAME "${OUTPUT_HPP}" NAME)

set(STR_CPP "// This file is auto-generated. Do not edit!

#include \"${OUTPUT_HPP_NAME}\"

namespace cv
{
namespace ocl
{
")

set(STR_HPP "// This file is auto-generated. Do not edit!

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

  set(STR_CPP "${STR_CPP}const char* ${cl_filename}=\"${lines};\n")
  set(STR_HPP "${STR_HPP}extern const char* ${cl_filename};\n")
endforeach()

set(STR_CPP "${STR_CPP}}\n}\n")
set(STR_HPP "${STR_HPP}}\n}\n")

file(WRITE ${OUTPUT} "${STR_CPP}")
file(WRITE ${OUTPUT_HPP} "${STR_HPP}")
