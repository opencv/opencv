#!/usr/bin/python
from __future__ import print_function
import hdr_parser
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


class TypeInfo(object):
    ptr = re.compile(r"Ptr_(\w+)")

    def __init__(self, name, decl):
        self.name = name.replace("Ptr_", "")
        self.fields = {}

        if decl:
            for p in decl[3]:
                self.fields[p[1]] = p[0]


class ArgInfo(object):
    def __init__(self, arg_tuple):
        self.tp = TypeInfo.ptr.sub(r"\1*", arg_tuple[0])
        self.truetp = arg_tuple[0]
        self.name = arg_tuple[1]
        self.defval = arg_tuple[2]
        self.isarray = False
        self.arraylen = 0
        self.arraycvt = None
        self.inputarg = True
        self.outputarg = False
        self.returnarg = False
        for m in arg_tuple[3]:
            if m == "/O":
                self.inputarg = False
                self.outputarg = True
                self.returnarg = True
            elif m == "/IO":
                self.inputarg = True
                self.outputarg = True
                self.returnarg = True
            elif m.startswith("/A"):
                self.isarray = True
                self.arraylen = m[2:].strip()
            elif m.startswith("/CA"):
                self.isarray = True
                self.arraycvt = m[2:].strip()
        self.py_inputarg = False
        self.py_outputarg = False

    def isbig(self):
        return self.tp == "Mat" or self.tp == "vector_Mat"

    def crepr(self):
        return "ArgInfo(\"%s\", %d)" % (self.name, self.outputarg)


class ConstInfo(object):
    def __init__(self, name, val):
        self.cname = name.replace(".", "::")
        self.name = name.replace(".", "_")
        self.name = re.sub(r"cv_([a-z])([A-Z])", r"cv_\1_\2", self.name)
        self.name = self.name.upper()
        if self.name.startswith("CV") and self.name[2] != "_":
            self.name = "CV_" + self.name[2:]
        self.value = val
        self.isfractional = re.match(r"^d+?\.\d+?$", self.value)


class FuncInfo(object):
    def __init__(self, classname, name, cname,
                 rettype, isconstructor, ismethod, args):
        self.classname = classname
        self.name = name
        self.cname = cname
        self.isconstructor = isconstructor
        self.variants = []
        self.args = args
        self.rettype = rettype
        if ismethod:
            self_arg = ArgInfo((classname + "*", "self", None, []))
            self.args = [self_arg] + self.args
        self.ismethod = ismethod

    def get_wrapper_name(self):
        name = self.name
        if self.classname:
            classname = self.classname + "_"
            if "[" in name:
                name = re.sub(r"operator\[\]", "getelem", name)
            elif "(" in name:
                name = re.sub(r"operator \(\)", "call", name)
        else:
            classname = ""

        if self.isconstructor:
            return "cv_create_" + name
        else:
            return "cv_" + classname + name


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
        self.types = {}
        self.funcs = {}
        self.consts = {}
        self.gentypes = {}
        self.hsc_types = StringIO()
        self.hsc_funcs = StringIO()
        self.hsc_consts = StringIO()

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

        if not (name in self.gentypes or name in simple_types):
            self.hsc_types.write("#opaque_t %s\n" % name)
            self.gentypes[name] = typeinfo

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
            hsc.write("#include <opencv2/opencv.h>\n")

        self.hsc_types.write("module OpenCV.Types where\n")
        self.hsc_consts.write("module OpenCV.Consts where\n")
        self.hsc_funcs.write("module OpenCV.Funcs where\n")

        for hsc in [self.hsc_types, self.hsc_consts, self.hsc_funcs]:
            hsc.write("#strict_import\n")
            hsc.write("import Foreign.C\n")
            hsc.write("import Foreign.C.Types\n")

        self.hsc_funcs.write("import OpenCV.Types\n")

    def add_type(self, name, decl):
        typeinfo = TypeInfo(name, decl)

        if not typeinfo.name in self.types:
            self.types[typeinfo.name] = typeinfo

    def add_func(self, decl):
        classname = bareclassname = ""
        name = decl[0]           # looks like cv{.classname}*.func
        dpos = name.rfind(".")
        if dpos >= 0 and name[:dpos] != "cv":
            classname = bareclassname = re.sub(r"^cv\.", "", name[:dpos])
            name = name[dpos + 1:]
            dpos = classname.rfind(".")
            if dpos >= 0:
                bareclassname = classname[dpos + 1:]
                classname = classname.replace(".", "_")

        cname = name
        name = re.sub(r"^cv\.", "", name)
        isconstructor = cname == bareclassname
        ismethod = not isconstructor and bareclassname != ""

        cname = cname.replace(".", "::")

        args = list(map(ArgInfo, decl[3]))
        for a in args:
            self.add_type(a.tp, None)

        if name in self.funcs.keys():
            name = self.fix_overloaded_func(name, len(args))

        self.funcs[name] = FuncInfo(bareclassname, name, cname,
                                    decl[1], isconstructor, ismethod, args)
        self.add_type(decl[1], None)

    def fix_overloaded_func(self, n, nargs):
        name = n

        name += str(nargs)

        #ugly, but keeps with the old behavior and adds to it.
        if name in self.funcs.keys():
            i = 0
            while True:
                #overloaded function with the same number of args :/
                if not name + "_" + str(i) in self.funcs.keys():
                    name += "_" + str(i)
                    break
                else:
                    i += 1
        return name

    def add_const(self, name, decl):
        constinfo = ConstInfo(name, decl[1])

        i = 0
        while constinfo.name + str(i) in self.consts:
            i += 1

        constinfo.name += str(i)
        self.consts[constinfo.name] = constinfo

    def readHeaders(self, srcfiles):
        parser = hdr_parser.CppHeaderParser()

        for hdr in srcfiles:
            decls = parser.parse(hdr)
            for decl in decls:
                name = decl[0]
                if name.startswith("struct") or name.startswith("class"):
                    self.add_type(name.replace("class ", "").strip(), decl)
                elif name.startswith("const"):
                    self.add_const(name.replace("const ", "").strip(), decl)
                else:
                    self.add_func(decl)

    def save(self, dstdir, outfile, buf):
        f = open(dstdir + outfile + ".hsc", "wt")
        f.write(buf.getvalue())
        f.close()

    def gen(self, srcfiles, dstdir):
        if not srcfiles:
            srcfiles = hdr_parser.opencv_hdr_list
        self.readHeaders(srcfiles)
        self.prep_hsc()

        # Generate the code for consts, types, and functions
        constlist = list(self.consts.items())
        constlist.sort()
        for n, c in constlist:
            self.gen_const(c)

        typelist = list(self.types.items())
        typelist.sort()
        for n, t in typelist:
                self.gen_type(t)

        funclist = list(self.funcs.items())
        funclist.sort()
        for n, f in funclist:
            self.gen_func(f)

        if not dstdir.endswith("/"):
            dstdir += "/"

        self.save(dstdir, "Types", self.hsc_types)
        self.save(dstdir, "Consts", self.hsc_consts)
        self.save(dstdir, "Funcs", self.hsc_funcs)

if __name__ == "__main__":
    hscdstdir = "OpenCV/"
    headers = None
    if len(sys.argv) > 1:
        hscdstdir = sys.argv[1]
    if len(sys.argv) > 2:
        headers = sys.argv[2:]

    hsc = HSCWrapperGen()
    hsc.gen(headers, hscdstdir)
