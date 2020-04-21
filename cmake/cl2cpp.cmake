if (NOT EXISTS "${CL_DIR}")
  message(FATAL_ERROR "Specified wrong OpenCL kernels directory: ${CL_DIR}")
endif()

file(GLOB cl_list "${CL_DIR}/*.cl" )
list(SORT cl_list)

if (NOT cl_list)
  message(FATAL_ERROR "Can't find OpenCL kernels in directory: ${CL_DIR}")
endif()

string(REGEX REPLACE "\\.cpp$" ".hpp" OUTPUT_HPP "${OUTPUT}")
get_filename_component(OUTPUT_HPP_NAME "${OUTPUT_HPP}" NAME)

set(nested_namespace_start "namespace ${MODULE_NAME}\n{")
set(nested_namespace_end "}")

set(STR_CPP "// This file is auto-generated. Do not edit!

#include \"opencv2/core.hpp\"
#include \"cvconfig.h\"
#include \"${OUTPUT_HPP_NAME}\"

#ifdef HAVE_OPENCL

namespace cv
{
namespace ocl
{
${nested_namespace_start}

static const char* const moduleName = \"${MODULE_NAME}\";

")

set(STR_HPP "// This file is auto-generated. Do not edit!

#include \"opencv2/core/ocl.hpp\"
#include \"opencv2/core/ocl_genbase.hpp\"
#include \"opencv2/core/opencl/ocl_defs.hpp\"

#ifdef HAVE_OPENCL

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

  set(STR_CPP_DECL "struct cv::ocl::internal::ProgramEntry ${cl_filename}_oclsrc={moduleName, \"${cl_filename}\",\n\"${lines}, \"${hash}\", NULL};\n")
  set(STR_HPP_DECL "extern struct cv::ocl::internal::ProgramEntry ${cl_filename}_oclsrc;\n")

  set(STR_CPP "${STR_CPP}${STR_CPP_DECL}")
  set(STR_HPP "${STR_HPP}${STR_HPP_DECL}")
endforeach()

set(STR_CPP "${STR_CPP}\n${nested_namespace_end}}}\n#endif\n")
set(STR_HPP "${STR_HPP}\n${nested_namespace_end}}}\n#endif\n")

file(WRITE "${OUTPUT}" "${STR_CPP}")

if(EXISTS "${OUTPUT_HPP}")
  file(READ "${OUTPUT_HPP}" hpp_lines)
endif()
if("${hpp_lines}" STREQUAL "${STR_HPP}")
  message(STATUS "${OUTPUT_HPP} contains the same content")
else()
  file(WRITE "${OUTPUT_HPP}" "${STR_HPP}")
endif()
