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

  string(MD5 hash "${lines}")

  set(STR_CPP "${STR_CPP}const struct ProgramEntry ${cl_filename}={\"${cl_filename}\",\n\"${lines}, \"${hash}\"};\n")
  set(STR_HPP "${STR_HPP}extern const struct ProgramEntry ${cl_filename};\n")
endforeach()

set(STR_CPP "${STR_CPP}}\n}\n")
set(STR_HPP "${STR_HPP}}\n}\n")

file(WRITE "${OUTPUT}" "${STR_CPP}")

if(EXISTS "${OUTPUT_HPP}")
  file(READ "${OUTPUT_HPP}" hpp_lines)
endif()
if("${hpp_lines}" STREQUAL "${STR_HPP}")
  message(STATUS "${OUTPUT_HPP} contains same content")
else()
  file(WRITE "${OUTPUT_HPP}" "${STR_HPP}")
endif()
