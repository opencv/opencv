# Iterate all GLSL shaders (with suffix '.comp') in current directory.
#
# Use glslangValidator to compile them to SPIR-V shaders and write them
# into .cpp files as unsigned int array.
#
# Also generate a header file 'spv_shader.hpp' to extern declare these shaders.

import re
import os
import sys

dir = "./"
license_decl = \
'// This file is part of OpenCV project.\n'\
'// It is subject to the license terms in the LICENSE file found in the top-level directory\n'\
'// of this distribution and at http://opencv.org/license.html.\n'\
'//\n'\
'// Copyright (C) 2018, Intel Corporation, all rights reserved.\n'\
'// Third party copyrights are property of their respective owners.\n\n'
precomp = '#include \"../../precomp.hpp\"\n'
ns_head = '\nnamespace cv { namespace dnn { namespace vkcom {\n\n'
ns_tail = '\n}}} // namespace cv::dnn::vkcom\n'

headfile = open('spv_shader.hpp', 'w')
headfile.write(license_decl)
headfile.write('#ifndef OPENCV_DNN_SPV_SHADER_HPP\n')
headfile.write('#define OPENCV_DNN_SPV_SHADER_HPP\n\n')
headfile.write(ns_head)

cmd_remove = ''
null_out = ''
if sys.platform.find('win32') != -1:
    cmd_remove = 'del'
    null_out = ' >>nul 2>nul'
elif sys.platform.find('linux') != -1:
    cmd_remove = 'rm'
    null_out = ' > /dev/null 2>&1'

list = os.listdir(dir)
for i in range(0, len(list)):
    if (os.path.splitext(list[i])[-1] != '.comp'):
        continue
    prefix = os.path.splitext(list[i])[0];
    path = os.path.join(dir, list[i])


    bin_file = prefix + '.tmp'
    cmd = ' glslangValidator -V ' + path + ' -S comp -o ' + bin_file
    print('compiling')
    if os.system(cmd) != 0:
        continue;
    size = os.path.getsize(bin_file)

    spv_txt_file = prefix + '.spv'
    cmd = 'glslangValidator -V ' + path + ' -S comp -o ' + spv_txt_file + ' -x' + null_out
    os.system(cmd)

    infile_name = spv_txt_file
    outfile_name = prefix + '_spv.cpp'
    array_name = prefix + '_spv'
    infile = open(infile_name, 'r')
    outfile = open(outfile_name, 'w')

    outfile.write(license_decl)
    outfile.write(precomp)
    outfile.write(ns_head)
    # xxx.spv ==> xxx_spv.cpp
    fmt = 'extern const unsigned int %s[%d] = {\n' % (array_name, size/4)
    outfile.write(fmt)
    for eachLine in infile:
        if(re.match(r'^.*\/\/', eachLine)):
            continue
        newline = '    ' + eachLine.replace('\t','')
        outfile.write(newline)
    infile.close()
    outfile.write("};\n")
    outfile.write(ns_tail)

    # write a line into header file
    fmt = 'extern const unsigned int %s[%d];\n' % (array_name, size/4)
    headfile.write(fmt)

    os.system(cmd_remove + ' ' + bin_file)
    os.system(cmd_remove + ' ' + spv_txt_file)

headfile.write(ns_tail)
headfile.write('\n#endif /* OPENCV_DNN_SPV_SHADER_HPP */\n')
headfile.close();
