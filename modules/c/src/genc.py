#!/usr/bin/python
from __future__ import print_function
import hdr_parser
import re
import sys

if sys.version_info[0] >= 3:
    from io import StringIO
else:
    from cStringIO import StringIO

_types = ["CvANN", "flann", "c"]                 # literally, underscore types.
namespaces = ["SimpleBlobDetector"]
empty_types = ["cvflann", "flann"]

exceptions = {"distance_t": "flann_distance_t",
              "algorithm_t": "flann_algorithm_t",
              # The following was taken care of previously by _types,
              # But for some reason, it doesn't work with the typedefs.
              r"CvANN<(\w+?)<(\w+)>>": r"CvANN_\1_\2"}

simple_types = ["int", "int64", "bool", "float", "double",
                "char*", "char", "size_t", "c_string", "void"]


class TypeInfo(object):
    _types = list(map(lambda t: re.compile(r"" + t + r"_(\w+)"), _types))
    generic_type = re.compile(r"(\w+?)_((\w{2,}))")
    nss = list(zip(namespaces, map(lambda ns: re.compile(r"" + ns + r"_(\w+)"),
                                   namespaces)))
    empty_types = list(map(lambda et: re.compile(r"" + et + r"_(\w+)"),
                           empty_types))
    ptr = re.compile(r"Ptr_(\w+)")

    def __init__(self, name, decl):
        self.name = name.replace("Ptr_", "")
        self.cname = TypeInfo.gen_cname(self.name)
        self.fields = {}

        if decl:
            for p in decl[3]:
                self.fields[p[1]] = p[0]

    @staticmethod
    def gen_cname(name):
        cname = name
        # make basic substitutions to get rid of basic types
        # and correct namespaces
        for m in TypeInfo.empty_types:
            cname = m.sub(r"\1", cname)
        for ns, m in TypeInfo.nss:
            cname = m.sub(r"" + ns + r"::\1", cname)

        if TypeInfo.ptr.match(cname):
            TypeInfo.ptr.sub(r"\1*", cname)

        # fix templated types
        while (TypeInfo.generic_type.search(cname) and
               not any(t.match(cname) for t in TypeInfo._types)):
            cname = TypeInfo.generic_type.sub(r"\1<\2>", cname)

        # fix any exceptional type issues and type1<type2<type3>> issues
        for e in exceptions:
            cname = re.sub(e, exceptions[e], cname)
        cname = re.sub(r"(\w+)<(\w+)<(\w+)>>", r"\1<\2<\3> >", cname)
        return cname


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
                name = "getelem"
            elif "(" in name:
                name = "call"
        else:
            classname = ""

        if self.isconstructor:
            return "cv_create_" + name
        else:
            return "cv_" + classname + name

    def get_wrapper_prototype(self):
        full_fname = self.get_wrapper_name()

        ptr_template = self.rettype.startswith("Ptr_")
        ptr = ptr_template or self.rettype.endswith("*")
        s = "" if self.rettype in simple_types or ptr else "*"
        if ptr_template:
            rettype = TypeInfo.ptr.sub(r"\1*", self.rettype)
        else:
            rettype = self.rettype + s
        ret = self.classname + "*" if self.isconstructor else rettype

        proto = "%s %s(" % (ret, full_fname)
        for arg in self.args:
            t = arg.tp
            simple = t in simple_types
            pointer = t.endswith("*")
            if not simple and not pointer and arg.name != self or arg.isarray:
                format_s = "%s* %s, "
            else:
                format_s = "%s %s, "

            proto += format_s % (t, arg.name)

        if proto.endswith("("):
            return proto + ");"
        else:
            return proto[:-2] + ");"

    def fix_call(self, call):
        if not call.endswith("("):
            call = call[:-2]

        void = self.rettype == "void"
        simple = self.rettype in simple_types
        pointer = self.rettype.endswith("*")
        # The following looks weird, but it's to deref the smart pointer
        # and return the simple pointer.
        if self.rettype.startswith("Ptr_"):
            call = "&*" + call
        elif not (void or simple or pointer or self.isconstructor):
            call = "new " + self.rettype + "(" + call + ")"

        return call + ");"

    def gen_code(self):
        proto = self.get_wrapper_prototype()[:-1]
        code = "%s {\n" % (proto,)

        ret = "" if self.rettype == "void" else "return "
        prefix = ""
        postfix = self.cname
        args = self.args
        if self.isconstructor:
            prefix = "new "
            postfix = self.classname
        elif self.ismethod:
            prefix = "self->"
            args = args[1:]

        call = prefix + "%s(" % (postfix,)

        for arg in args:
            smart_ptr = arg.truetp.startswith("Ptr_")
            name = arg.name
            if smart_ptr:
                name = "Ptr<" + arg.tp[:-1] + ">(" + arg.name + ")"

            simple = arg.tp in simple_types
            ptr = arg.tp.endswith("*")
            s = "" if simple or ptr else "*"
            call += s + name + ", "

        call = self.fix_call(call)
        code += "\t" + ret + call
        code += "\n}\n"

        return code


class CWrapperGenerator(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.funcs = {}
        self.consts = {}
        self.types = {}
        self.source = StringIO()
        self.header = StringIO()

    def add_const(self, name, decl):
        constinfo = ConstInfo(name, decl[1])

        i = 0
        while constinfo.name + str(i) in self.consts:
            i += 1

        constinfo.name += str(i)
        self.consts[constinfo.name] = constinfo

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

        i = 0
        #ugly, but keeps with the old behavior and adds to it.
        if name in self.funcs.keys():
            while True:
                #overloaded function with the same number of args :/
                if not name + "_" + str(i) in self.funcs.keys():
                    break
                else:
                    i += 1
        if i != 0:
            name += "_" + str(i)
        return name

    def add_type(self, name, decl):
        typeinfo = TypeInfo(name, decl)

        if not typeinfo.name in self.types:
            self.types[typeinfo.name] = typeinfo

    def save(self, path, name, buf):
        f = open(path + "/" + name, "wt")
        f.write(buf.getvalue())
        f.close()

    def gen_const_reg(self, constinfo):
        self.header.write("#define %s %s\n"
                          % (constinfo.name, constinfo.cname))

    def gen_typedef(self, typeinfo):
        self.header.write("typedef %s %s;\n"
                          % (typeinfo.cname.replace("*", ""),
                             typeinfo.name.replace("*", "")))

    def prep_src(self):
        self.source.write("using namespace cv;\n")
        self.source.write("using namespace std;\n")
        self.source.write("using namespace flann;\n")
        self.source.write("using namespace cvflann;\n")
        self.source.write("extern \"C\" {\n")

    def prep_header(self):
        self.header.write("#ifndef __OPENCV_GENERATED_HPP\n")
        self.header.write("#define __OPENCV_GENERATED_HPP\n")
        self.header.write("#include <opencv2/opencv.hpp>\n")
        self.header.write("#include <opencv2/nonfree.hpp>\n")
        self.header.write("#include <vector>\n")
        self.header.write("using namespace cv;\n")
        self.header.write("using namespace std;\n")
        self.header.write("using namespace flann;\n")
        self.header.write("using namespace cvflann;\n")
        self.header.write("extern \"C\" {\n")
        self.header.write("typedef char* c_string;\n")
        self.header.write("typedef SimpleBlobDetector::Params Params;\n")
        self.header.write("typedef linemod::Detector Detector;\n")

    def finalize_and_write(self, output_path):
        self.header.write("}\n")
        self.header.write("#endif //__OPENCV_GENERATED_HPP")
        self.source.write("}")
        self.save(output_path, "opencv_generated.hpp", self.header)
        self.save(output_path, "opencv_generated.cpp", self.source)

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

    def gen(self, srcfiles, output_path):
        self.clear()

        if not srcfiles:
            srcfiles = hdr_parser.opencv_hdr_list
        self.readHeaders(srcfiles)
        self.prep_header()
        self.prep_src()

        typelist = list(self.types.items())
        typelist.sort()
        for name, typeinfo in typelist:
            if typeinfo.name != typeinfo.cname:
                self.gen_typedef(typeinfo)

        constlist = list(self.consts.items())
        constlist.sort()
        for name, const in constlist:
            self.gen_const_reg(const)

        funclist = list(self.funcs.items())
        funclist.sort()
        for name, func in funclist:
            prototype = func.get_wrapper_prototype() + "\n"
            code = func.gen_code()
            self.header.write(prototype)
            self.source.write(code)

        self.finalize_and_write(output_path)


if __name__ == "__main__":
    srcfiles = None
    dstdir = "."
    if len(sys.argv) > 1:
        dstdir = sys.argv[1]
    if len(sys.argv) > 2:
        srcfiles = sys.argv[2:]

    generator = CWrapperGenerator()
    generator.gen(srcfiles, dstdir)
