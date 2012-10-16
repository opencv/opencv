import os, os.path, sys, glob

indir = sys.argv[1]
outname = sys.argv[2]
#indir = "/Users/vp/work/ocv/opencv/modules/ocl/src/kernels"
#outname = "/Users/vp/work/ocv.build/xcode/modules/ocl/kernels.cpp"

try:
    os.mkdir(os.path.dirname(outname))
except OSError:
    pass

cl_list = glob.glob(os.path.join(indir, "*.cl"))
kfile = open(outname, "wt")

kfile.write("""// This file is auto-generated. Do not edit!

namespace cv
{
namespace ocl
{
""")

for cl in cl_list:
    cl_file = open(cl, "rt")
    cl_filename = os.path.basename(cl)
    cl_filename = cl_filename[:cl_filename.rfind(".")]
    kfile.write("const char* %s=" % cl_filename)
    state = 0

    for cl_line in cl_file.readlines():
        l = cl_line.strip()
        # skip the leading comments
        if l.startswith("//") and l.find("*/") < 0:
            if state == 0:
                state = 1
        else:
            if state == 1 or l.find("*/") >= 0:
                state = 2

        if state == 1:
            continue

        l = l.replace("\\", "\\\\")
        l = l.replace("\r", "")
        l = l.replace("\"", "\\\"")
        l = l.replace("\t", "  ")
        kfile.write("\"%s\\n\"\n" % l)
    kfile.write(";\n")
    cl_file.close()

kfile.write("""}
}
""")
kfile.close()
