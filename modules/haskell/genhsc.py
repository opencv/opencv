from __future__ import print_function
from cbits import genc
import sys
import re

if sys.version_info[0] >= 3:
    from io import StringIO
else:
    from cStringIO import StringIO

simple_types = {"int": "CInt",
                "int64": "CLong",
                "bool": "CInt",
                "float": "CFloat",
                "double": "CDouble",
                "char*": "CString",
                "char": "CChar",
                "size_t": "CSize",
                "c_string": "CString",
                "void": "()"}
exceptions = {"flann_Index": "Index",
              "SimpleBlobDetector_Params": "Params",
              "cvflann_flann_distance_t": "flann_distance_t",
              "cvflann_flann_algorithm_t": "flann_algorithm_t",
              "flann_IndexParams": "IndexParams",
              "flann_SearchParams": "SearchParams"}


def toHSType(t):
    if t in simple_types:
        return simple_types[t]
    # check if we have a pointer to a simple_type
    elif t[:-1] in simple_types:
        return "Ptr " + simple_types[t[:-1]]

    t = re.sub(r"Ptr_(\w+)", r"\1*", t)

    if t in exceptions:
        t = exceptions[t]
    elif t[:-1] in exceptions:
        t = exceptions[t[:-1]]

    ptr = t.endswith("*")
    t = "<" + t + ">"
    if ptr:
        t = re.sub(r"<(\w+)\*>", r"Ptr <\1>", t)
    else:
        t = re.sub(r"(.+)", r"Ptr \1", t)

    return t


class HSCWrapperGen(object):
    def __init__(self):
        self.hsc_types = StringIO()
        self.hsc_funcs = StringIO()
        self.hsc_consts = StringIO()
        self.types = {}

    def gen_const(self, constinfo):
        if constinfo.isfractional:
            self.hsc_consts.write("#fractional %s\n" % constinfo.cname,)
        else:
            self.hsc_consts.write("#num %s\n" % constinfo.name,)

    def gen_type(self, typeinfo):
        if not typeinfo.name:
            return None

        name = typeinfo.name.replace("cv.", "")
        name = name.replace("*", "")
        name = name.replace("struct ", "")
        name = name.replace(".", "_")
        name = name.replace("Ptr_", "")

        if name in exceptions:
            name = exceptions[name]

        if not (name in self.types or name in simple_types):
            self.hsc_types.write("#opaque_t %s\n" % name)
            self.types[name] = typeinfo

    def gen_func(self, func):
        code = "#ccall %s , " % (func.get_wrapper_name(),)
        for a in func.args:
            code += "%s -> " % toHSType(a.tp)

        ret = func.classname + "*" if func.isconstructor else func.rettype
        hsc_ret = toHSType(ret)
        if " " in hsc_ret:
            hsc_ret = "(" + hsc_ret + ")"
        code += "IO %s\n" % hsc_ret

        self.hsc_funcs.write(code)

    def prep_hsc(self):
        for hsc in [self.hsc_types, self.hsc_consts, self.hsc_funcs]:
            hsc.write("{-# LANGUAGE ForeignFunctionInterface #-}\n")
            hsc.write("#include <bindings.dsl.h>\n")
            hsc.write("#include <opencv_generated.hpp>\n")

        self.hsc_types.write("module OpenCV.Types where\n")
        self.hsc_consts.write("module OpenCV.Consts where\n")
        self.hsc_funcs.write("module OpenCV.Funcs where\n")

        for hsc in [self.hsc_types, self.hsc_consts, self.hsc_funcs]:
            hsc.write("#strict_import\n")
            hsc.write("import Foreign.C\n")
            hsc.write("import Foreign.C.Types\n")

        self.hsc_funcs.write("import OpenCV.Types\n")

    def save(self, dstdir, outfile, buf):
        f = open(dstdir + outfile + ".hsc", "wt")
        f.write(buf.getvalue())
        f.close()

    def gen(self, cgen, dstdir):
        self.prep_hsc()

        # Generate the code for consts, types, and functions
        constlist = list(cgen.consts.items())
        constlist.sort()
        for n, c in constlist:
            self.gen_const(c)

        typelist = list(cgen.types.items())
        typelist.sort()
        for n, t in typelist:
                self.gen_type(t)

        funclist = list(cgen.funcs.items())
        funclist.sort()
        for n, f in funclist:
            self.gen_func(f)

        if not dstdir.endswith("/"):
            dstdir += "/"

        self.save(dstdir, "Types", self.hsc_types)
        self.save(dstdir, "Consts", self.hsc_consts)
        self.save(dstdir, "Funcs", self.hsc_funcs)

if __name__ == "__main__":
    header_dir = "/usr/local/include/"
    dstdir = "cbits/"
    hscdstdir = "OpenCV/"
    headers = None
    if len(sys.argv) > 1:
        header_dir = sys.argv[1]
    if len(sys.argv) > 2:
        dstdir = sys.argv[2]
    if len(sys.argv) > 3:
        hscdstdir = sys.argv[3]
    if len(sys.argv) > 4:
        headers = sys.argv[4:]

    cgen = genc.CWrapperGenerator()
    cgen.gen(header_dir, headers, dstdir)

    hsc = HSCWrapperGen()
    hsc.gen(cgen, hscdstdir)
