file(GLOB cl_list "${CL_DIR}/*.cl" )
list(SORT cl_list)

string(REPLACE ".cpp" ".hpp" OUTPUT_HPP "${OUTPUT}")
get_filename_component(OUTPUT_HPP_NAME "${OUTPUT_HPP}" NAME)

if("${MODULE_NAME}" STREQUAL "ocl")
    set(nested_namespace_start "")
    set(nested_namespace_end "")
else()
    set(new_mode ON)
    set(nested_namespace_start "namespace ${MODULE_NAME}\n{")
    set(nested_namespace_end "}")
endif()

set(STR_CPP "// This file is auto-generated. Do not edit!

#include \"precomp.hpp\"
#include \"${OUTPUT_HPP_NAME}\"

namespace cv
{
namespace ocl
{
${nested_namespace_start}

")

set(STR_HPP "// This file is auto-generated. Do not edit!

#include \"opencv2/core/ocl_genbase.hpp\"

namespace cv
{
namespace ocl
{
${nested_namespace_start}

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

  set(STR_CPP_DECL "const struct ProgramEntry ${cl_filename}={\"${cl_filename}\",\n\"${lines}, \"${hash}\"};\n")
  set(STR_HPP_DECL "extern const struct ProgramEntry ${cl_filename};\n")
  if(new_mode)
    set(STR_CPP_DECL "${STR_CPP_DECL}ProgramSource2 ${cl_filename}_oclsrc(${cl_filename}.programStr);\n")
    set(STR_HPP_DECL "${STR_HPP_DECL}extern ProgramSource2 ${cl_filename}_oclsrc;\n")
  endif()

  set(STR_CPP "${STR_CPP}${STR_CPP_DECL}")
  set(STR_HPP "${STR_HPP}${STR_HPP_DECL}")
endforeach()

set(STR_CPP "${STR_CPP}}\n${nested_namespace_end}}\n")
set(STR_HPP "${STR_HPP}}\n${nested_namespace_end}}\n")

file(WRITE "${OUTPUT}" "${STR_CPP}")

if(EXISTS "${OUTPUT_HPP}")
  file(READ "${OUTPUT_HPP}" hpp_lines)
endif()
if("${hpp_lines}" STREQUAL "${STR_HPP}")
  message(STATUS "${OUTPUT_HPP} contains same content")
else()
  file(WRITE "${OUTPUT_HPP}" "${STR_HPP}")
endif()
