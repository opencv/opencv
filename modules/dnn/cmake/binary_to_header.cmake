# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
#
# Copyright (C) 2017, Intel Corporation, all rights reserved.
# Third party copyrights are property of their respective owners.

# Function that converts binary file to header with hexadecimal array.
# Works like `xxd -i` on Linux.
function(binary_to_header binary_file output_header)
  get_filename_component(binary_file_name ${binary_file} NAME_WE)
  string(TOUPPER ${binary_file_name} binary_file_name)

  set(header_from_binary
      "#ifndef __OPENCV_DNN_${binary_file_name}_HPP__\n"
      "#define __OPENCV_DNN_${binary_file_name}_HPP__\n")

  string(TOLOWER ${binary_file_name} binary_file_name)

  set(header_from_binary "${header_from_binary}char ${binary_file_name}[] = {\n")

  file(READ ${binary_file} binary_file_content HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "'\\\\x\\1', "
         binary_file_content ${binary_file_content})
  string(REGEX REPLACE
         "('.x..', '.x..', '.x..', '.x..', '.x..', '.x..', '.x..', '.x..', '.x..',) "
         "  \\1\n"
         binary_file_content ${binary_file_content})
  string(REGEX REPLACE ", $" "\n" binary_file_content ${binary_file_content})

  set(header_from_binary "${header_from_binary}${binary_file_content}}\;\n#endif\n")

  file(WRITE ${output_header} ${header_from_binary})
endfunction(binary_to_header)
