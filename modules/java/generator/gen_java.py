#!/usr/bin/env python

import sys, re, os.path
import json
import logging
from pprint import pformat
from string import Template

if sys.version_info[0] >= 3:
    from io import StringIO
else:
    from cStringIO import StringIO

# list of class names, which should be skipped by wrapper generator
# the list is loaded from misc/java/gen_dict.json defined for the module and its dependencies
class_ignore_list = []

# list of constant names, which should be skipped by wrapper generator
# ignored constants can be defined using regular expressions
const_ignore_list = []

# list of private constants
const_private_list = []

# { Module : { public : [[name, val],...], private : [[]...] } }
missing_consts = {}

# c_type    : { java/jni correspondence }
# Complex data types are configured for each module using misc/java/gen_dict.json

type_dict = {
# "simple"  : { j_type : "?", jn_type : "?", jni_type : "?", suffix : "?" },
    ""        : { "j_type" : "", "jn_type" : "long", "jni_type" : "jlong" }, # c-tor ret_type
    "void"    : { "j_type" : "void", "jn_type" : "void", "jni_type" : "void" },
    "env"     : { "j_type" : "", "jn_type" : "", "jni_type" : "JNIEnv*"},
    "cls"     : { "j_type" : "", "jn_type" : "", "jni_type" : "jclass"},
    "bool"    : { "j_type" : "boolean", "jn_type" : "boolean", "jni_type" : "jboolean", "suffix" : "Z" },
    "char"    : { "j_type" : "char", "jn_type" : "char", "jni_type" : "jchar", "suffix" : "C" },
    "int"     : { "j_type" : "int", "jn_type" : "int", "jni_type" : "jint", "suffix" : "I" },
    "long"    : { "j_type" : "int", "jn_type" : "int", "jni_type" : "jint", "suffix" : "I" },
    "float"   : { "j_type" : "float", "jn_type" : "float", "jni_type" : "jfloat", "suffix" : "F" },
    "double"  : { "j_type" : "double", "jn_type" : "double", "jni_type" : "jdouble", "suffix" : "D" },
    "size_t"  : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "__int64" : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "int64"   : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "double[]": { "j_type" : "double[]", "jn_type" : "double[]", "jni_type" : "jdoubleArray", "suffix" : "_3D" }
}

# { class : { func : {j_code, jn_code, cpp_code} } }
ManualFuncs = {}

# { class : { func : { arg_name : {"ctype" : ctype, "attrib" : [attrib]} } } }
func_arg_fix = {}

def getLibVersion(version_hpp_path):
    version_file = open(version_hpp_path, "rt").read()
    major = re.search("^W*#\W*define\W+CV_VERSION_MAJOR\W+(\d+)\W*$", version_file, re.MULTILINE).group(1)
    minor = re.search("^W*#\W*define\W+CV_VERSION_MINOR\W+(\d+)\W*$", version_file, re.MULTILINE).group(1)
    revision = re.search("^W*#\W*define\W+CV_VERSION_REVISION\W+(\d+)\W*$", version_file, re.MULTILINE).group(1)
    status = re.search("^W*#\W*define\W+CV_VERSION_STATUS\W+\"(.*?)\"\W*$", version_file, re.MULTILINE).group(1)
    return (major, minor, revision, status)

def libVersionBlock():
    (major, minor, revision, status) = getLibVersion(
    (os.path.dirname(__file__) or '.') + '/../../core/include/opencv2/core/version.hpp')
    version_str    = '.'.join( (major, minor, revision) ) + status
    version_suffix =  ''.join( (major, minor, revision) )
    return """
    // these constants are wrapped inside functions to prevent inlining
    private static String getVersion() { return "%(v)s"; }
    private static String getNativeLibraryName() { return "opencv_java%(vs)s"; }
    private static int getVersionMajor() { return %(ma)s; }
    private static int getVersionMinor() { return %(mi)s; }
    private static int getVersionRevision() { return %(re)s; }
    private static String getVersionStatus() { return "%(st)s"; }

    public static final String VERSION = getVersion();
    public static final String NATIVE_LIBRARY_NAME = getNativeLibraryName();
    public static final int VERSION_MAJOR = getVersionMajor();
    public static final int VERSION_MINOR = getVersionMinor();
    public static final int VERSION_REVISION = getVersionRevision();
    public static final String VERSION_STATUS = getVersionStatus();
""" % { 'v' : version_str, 'vs' : version_suffix, 'ma' : major, 'mi' : minor, 're' : revision, 'st': status }


T_JAVA_START_INHERITED = """
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.$module;

$imports

// C++: class $name
//javadoc: $name
public class $jname extends $base {

    protected $jname(long addr) { super(addr); }

"""

T_JAVA_START_ORPHAN = """
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.$module;

$imports

// C++: class $name
//javadoc: $name
public class $jname {

    protected final long nativeObj;
    protected $jname(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }
"""

T_JAVA_START_MODULE = """
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.$module;

$imports

public class $jname {
"""

T_CPP_MODULE = """
//
// This file is auto-generated, please don't edit!
//

#define LOG_TAG "org.opencv.$m"

#include "common.h"

#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_$M

#include <string>

#include "opencv2/$m.hpp"

$includes

using namespace cv;

/// throw java exception
static void throwJavaException(JNIEnv *env, const std::exception *e, const char *method) {
  std::string what = "unknown exception";
  jclass je = 0;

  if(e) {
    std::string exception_type = "std::exception";

    if(dynamic_cast<const cv::Exception*>(e)) {
      exception_type = "cv::Exception";
      je = env->FindClass("org/opencv/core/CvException");
    }

    what = exception_type + ": " + e->what();
  }

  if(!je) je = env->FindClass("java/lang/Exception");
  env->ThrowNew(je, what.c_str());

  LOGE("%s caught %s", method, what.c_str());
  (void)method;        // avoid "unused" warning
}


extern "C" {

$code

} // extern "C"

#endif // HAVE_OPENCV_$M
"""

class GeneralInfo():
    def __init__(self, name, namespaces):
        self.namespace, self.classpath, self.classname, self.name = self.parseName(name, namespaces)

    def parseName(self, name, namespaces):
        '''
        input: full name and available namespaces
        returns: (namespace, classpath, classname, name)
        '''
        name = name[name.find(" ")+1:].strip() # remove struct/class/const prefix
        spaceName = ""
        localName = name # <classes>.<name>
        for namespace in sorted(namespaces, key=len, reverse=True):
            if name.startswith(namespace + "."):
                spaceName = namespace
                localName = name.replace(namespace + ".", "")
                break
        pieces = localName.split(".")
        if len(pieces) > 2: # <class>.<class>.<class>.<name>
            return spaceName, ".".join(pieces[:-1]), pieces[-2], pieces[-1]
        elif len(pieces) == 2: # <class>.<name>
            return spaceName, pieces[0], pieces[0], pieces[1]
        elif len(pieces) == 1: # <name>
            return spaceName, "", "", pieces[0]
        else:
            return spaceName, "", "" # error?!

    def fullName(self, isCPP=False):
        result = ".".join([self.fullClass(), self.name])
        return result if not isCPP else result.replace(".", "::")

    def fullClass(self, isCPP=False):
        result = ".".join([f for f in [self.namespace] + self.classpath.split(".") if len(f)>0])
        return result if not isCPP else result.replace(".", "::")

class ConstInfo(GeneralInfo):
    def __init__(self, decl, addedManually=False, namespaces=[]):
        GeneralInfo.__init__(self, decl[0], namespaces)
        self.cname = self.name.replace(".", "::")
        self.value = decl[1]
        self.addedManually = addedManually

    def __repr__(self):
        return Template("CONST $name=$value$manual").substitute(name=self.name,
                                                                 value=self.value,
                                                                 manual="(manual)" if self.addedManually else "")

    def isIgnored(self):
        for c in const_ignore_list:
            if re.match(c, self.name):
                return True
        return False

class ClassPropInfo():
    def __init__(self, decl): # [f_ctype, f_name, '', '/RW']
        self.ctype = decl[0]
        self.name = decl[1]
        self.rw = "/RW" in decl[3]

    def __repr__(self):
        return Template("PROP $ctype $name").substitute(ctype=self.ctype, name=self.name)

class ClassInfo(GeneralInfo):
    def __init__(self, decl, namespaces=[]): # [ 'class/struct cname', ': base', [modlist] ]
        GeneralInfo.__init__(self, decl[0], namespaces)
        self.cname = self.name.replace(".", "::")
        self.methods = []
        self.methods_suffixes = {}
        self.consts = [] # using a list to save the occurence order
        self.private_consts = []
        self.imports = set()
        self.props= []
        self.jname = self.name
        self.smart = None # True if class stores Ptr<T>* instead of T* in nativeObj field
        self.j_code = None # java code stream
        self.jn_code = None # jni code stream
        self.cpp_code = None # cpp code stream
        for m in decl[2]:
            if m.startswith("="):
                self.jname = m[1:]
        self.base = ''
        if decl[1]:
            #self.base = re.sub(r"\b"+self.jname+r"\b", "", decl[1].replace(":", "")).strip()
            self.base = re.sub(r"^.*:", "", decl[1].split(",")[0]).strip().replace(self.jname, "")

    def __repr__(self):
        return Template("CLASS $namespace::$classpath.$name : $base").substitute(**self.__dict__)

    def getAllImports(self, module):
        return ["import %s;" % c for c in sorted(self.imports) if not c.startswith('org.opencv.'+module)]

    def addImports(self, ctype):
        if ctype in type_dict:
            if "j_import" in type_dict[ctype]:
                self.imports.add(type_dict[ctype]["j_import"])
            if "v_type" in type_dict[ctype]:
                self.imports.add("java.util.List")
                self.imports.add("java.util.ArrayList")
                self.imports.add("org.opencv.utils.Converters")
                if type_dict[ctype]["v_type"] in ("Mat", "vector_Mat"):
                    self.imports.add("org.opencv.core.Mat")

    def getAllMethods(self):
        result = []
        result.extend([fi for fi in sorted(self.methods) if fi.isconstructor])
        result.extend([fi for fi in sorted(self.methods) if not fi.isconstructor])
        return result

    def addMethod(self, fi):
        self.methods.append(fi)

    def getConst(self, name):
        for cand in self.consts + self.private_consts:
            if cand.name == name:
                return cand
        return None

    def addConst(self, constinfo):
        # choose right list (public or private)
        consts = self.consts
        for c in const_private_list:
            if re.match(c, constinfo.name):
                consts = self.private_consts
                break
        consts.append(constinfo)

    def initCodeStreams(self, Module):
        self.j_code = StringIO()
        self.jn_code = StringIO()
        self.cpp_code = StringIO();
        if self.base:
            self.j_code.write(T_JAVA_START_INHERITED)
        else:
            if self.name != Module:
                self.j_code.write(T_JAVA_START_ORPHAN)
            else:
                self.j_code.write(T_JAVA_START_MODULE)
        # misc handling
        if self.name == 'Core':
            self.imports.add("java.lang.String")
            self.j_code.write(libVersionBlock())

    def cleanupCodeStreams(self):
        self.j_code.close()
        self.jn_code.close()
        self.cpp_code.close()

    def generateJavaCode(self, m, M):
        return Template(self.j_code.getvalue() + "\n\n" + \
                         self.jn_code.getvalue() + "\n}\n").substitute(\
                            module = m,
                            name = self.name,
                            jname = self.jname,
                            imports = "\n".join(self.getAllImports(M)),
                            base = self.base)

    def generateCppCode(self):
        return self.cpp_code.getvalue()

class ArgInfo():
    def __init__(self, arg_tuple): # [ ctype, name, def val, [mod], argno ]
        self.pointer = False
        ctype = arg_tuple[0]
        if ctype.endswith("*"):
            ctype = ctype[:-1]
            self.pointer = True
        self.ctype = ctype
        self.name = arg_tuple[1]
        self.defval = arg_tuple[2]
        self.out = ""
        if "/O" in arg_tuple[3]:
            self.out = "O"
        if "/IO" in arg_tuple[3]:
            self.out = "IO"

    def __repr__(self):
        return Template("ARG $ctype$p $name=$defval").substitute(ctype=self.ctype,
                                                                  p=" *" if self.pointer else "",
                                                                  name=self.name,
                                                                  defval=self.defval)

class FuncInfo(GeneralInfo):
    def __init__(self, decl, namespaces=[]): # [ funcname, return_ctype, [modifiers], [args] ]
        GeneralInfo.__init__(self, decl[0], namespaces)
        self.cname = self.name.replace(".", "::")
        self.jname = self.name
        self.isconstructor = self.name == self.classname
        if "[" in self.name:
            self.jname = "getelem"
        for m in decl[2]:
            if m.startswith("="):
                self.jname = m[1:]
        self.static = ["","static"][ "/S" in decl[2] ]
        self.ctype = re.sub(r"^CvTermCriteria", "TermCriteria", decl[1] or "")
        self.args = []
        func_fix_map = func_arg_fix.get(self.jname, {})
        for a in decl[3]:
            arg = a[:]
            arg_fix_map = func_fix_map.get(arg[1], {})
            arg[0] = arg_fix_map.get('ctype',  arg[0]) #fixing arg type
            arg[3] = arg_fix_map.get('attrib', arg[3]) #fixing arg attrib
            self.args.append(ArgInfo(arg))

    def __repr__(self):
        return Template("FUNC <$ctype $namespace.$classpath.$name $args>").substitute(**self.__dict__)

    def __lt__(self, other):
        return self.__repr__() < other.__repr__()


class JavaWrapperGenerator(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.namespaces = set(["cv"])
        self.classes = { "Mat" : ClassInfo([ 'class Mat', '', [], [] ], self.namespaces) }
        self.module = ""
        self.Module = ""
        self.ported_func_list = []
        self.skipped_func_list = []
        self.def_args_hist = {} # { def_args_cnt : funcs_cnt }

    def add_class(self, decl):
        classinfo = ClassInfo(decl, namespaces=self.namespaces)
        if classinfo.name in class_ignore_list:
            logging.info('ignored: %s', classinfo)
            return
        name = classinfo.name
        if self.isWrapped(name) and not classinfo.base:
            logging.warning('duplicated: %s', classinfo)
            return
        self.classes[name] = classinfo
        if name in type_dict and not classinfo.base:
            logging.warning('duplicated: %s', classinfo)
            return
        type_dict[name] = \
            { "j_type" : classinfo.jname,
              "jn_type" : "long", "jn_args" : (("__int64", ".nativeObj"),),
              "jni_name" : "(*("+classinfo.fullName(isCPP=True)+"*)%(n)s_nativeObj)", "jni_type" : "jlong",
              "suffix" : "J" }
        type_dict[name+'*'] = \
            { "j_type" : classinfo.jname,
              "jn_type" : "long", "jn_args" : (("__int64", ".nativeObj"),),
              "jni_name" : "("+classinfo.fullName(isCPP=True)+"*)%(n)s_nativeObj", "jni_type" : "jlong",
              "suffix" : "J" }

        # missing_consts { Module : { public : [[name, val],...], private : [[]...] } }
        if name in missing_consts:
            if 'private' in missing_consts[name]:
                for (n, val) in missing_consts[name]['private']:
                    classinfo.private_consts.append( ConstInfo([n, val], addedManually=True) )
            if 'public' in missing_consts[name]:
                for (n, val) in missing_consts[name]['public']:
                    classinfo.consts.append( ConstInfo([n, val], addedManually=True) )

        # class props
        for p in decl[3]:
            if True: #"vector" not in p[0]:
                classinfo.props.append( ClassPropInfo(p) )
            else:
                logging.warning("Skipped property: [%s]" % name, p)

        if classinfo.base:
            classinfo.addImports(classinfo.base)
        type_dict["Ptr_"+name] = \
            { "j_type" : classinfo.jname,
              "jn_type" : "long", "jn_args" : (("__int64", ".getNativeObjAddr()"),),
              "jni_name" : "*((Ptr<"+classinfo.fullName(isCPP=True)+">*)%(n)s_nativeObj)", "jni_type" : "jlong",
              "suffix" : "J" }
        logging.info('ok: class %s, name: %s, base: %s', classinfo, name, classinfo.base)

    def add_const(self, decl): # [ "const cname", val, [], [] ]
        constinfo = ConstInfo(decl, namespaces=self.namespaces)
        if constinfo.isIgnored():
            logging.info('ignored: %s', constinfo)
        elif not self.isWrapped(constinfo.classname):
            logging.info('class not found: %s', constinfo)
        else:
            ci = self.getClass(constinfo.classname)
            duplicate = ci.getConst(constinfo.name)
            if duplicate:
                if duplicate.addedManually:
                    logging.info('manual: %s', constinfo)
                else:
                    logging.warning('duplicated: %s', constinfo)
            else:
                ci.addConst(constinfo)
                logging.info('ok: %s', constinfo)

    def add_func(self, decl):
        fi = FuncInfo(decl, namespaces=self.namespaces)
        classname = fi.classname or self.Module
        if classname in class_ignore_list:
            logging.info('ignored: %s', fi)
        elif classname in ManualFuncs and fi.jname in ManualFuncs[classname]:
            logging.info('manual: %s', fi)
        elif not self.isWrapped(classname):
            logging.warning('not found: %s', fi)
        else:
            self.getClass(classname).addMethod(fi)
            logging.info('ok: %s', fi)
            # calc args with def val
            cnt = len([a for a in fi.args if a.defval])
            self.def_args_hist[cnt] = self.def_args_hist.get(cnt, 0) + 1

    def save(self, path, buf):
        f = open(path, "wt")
        f.write(buf)
        f.close()

    def gen(self, srcfiles, module, output_path, common_headers):
        self.clear()
        self.module = module
        self.Module = module.capitalize()
        # TODO: support UMat versions of declarations (implement UMat-wrapper for Java)
        parser = hdr_parser.CppHeaderParser(generate_umat_decls=False)

        self.add_class( ['class ' + self.Module, '', [], []] ) # [ 'class/struct cname', ':bases', [modlist] [props] ]

        # scan the headers and build more descriptive maps of classes, consts, functions
        includes = [];
        for hdr in common_headers:
            logging.info("\n===== Common header : %s =====", hdr)
            includes.append('#include "' + hdr + '"')
        for hdr in srcfiles:
            decls = parser.parse(hdr)
            self.namespaces = parser.namespaces
            logging.info("\n\n===== Header: %s =====", hdr)
            logging.info("Namespaces: %s", parser.namespaces)
            if decls:
                includes.append('#include "' + hdr + '"')
            else:
                logging.info("Ignore header: %s", hdr)
            for decl in decls:
                logging.info("\n--- Incoming ---\n%s", pformat(decl, 4))
                name = decl[0]
                if name.startswith("struct") or name.startswith("class"):
                    self.add_class(decl)
                elif name.startswith("const"):
                    self.add_const(decl)
                else: # function
                    self.add_func(decl)

        logging.info("\n\n===== Generating... =====")
        moduleCppCode = StringIO()
        for ci in self.classes.values():
            if ci.name == "Mat":
                continue
            ci.initCodeStreams(self.Module)
            self.gen_class(ci)
            classJavaCode = ci.generateJavaCode(self.module, self.Module)
            self.save("%s/%s+%s.java" % (output_path, module, ci.jname), classJavaCode)
            moduleCppCode.write(ci.generateCppCode())
            ci.cleanupCodeStreams()
        self.save(output_path+"/"+module+".cpp", Template(T_CPP_MODULE).substitute(m = module, M = module.upper(), code = moduleCppCode.getvalue(), includes = "\n".join(includes)))
        self.save(output_path+"/"+module+".txt", self.makeReport())

    def makeReport(self):
        '''
        Returns string with generator report
        '''
        report = StringIO()
        total_count = len(self.ported_func_list)+ len(self.skipped_func_list)
        report.write("PORTED FUNCs LIST (%i of %i):\n\n" % (len(self.ported_func_list), total_count))
        report.write("\n".join(self.ported_func_list))
        report.write("\n\nSKIPPED FUNCs LIST (%i of %i):\n\n" % (len(self.skipped_func_list), total_count))
        report.write("".join(self.skipped_func_list))
        for i in self.def_args_hist.keys():
            report.write("\n%i def args - %i funcs" % (i, self.def_args_hist[i]))
        return report.getvalue()

    def fullTypeName(self, t):
        if self.isWrapped(t):
            return self.getClass(t).fullName(isCPP=True)
        else:
            return t

    def gen_func(self, ci, fi, prop_name=''):
        logging.info("%s", fi)
        j_code   = ci.j_code
        jn_code  = ci.jn_code
        cpp_code = ci.cpp_code

        # c_decl
        # e.g: void add(Mat src1, Mat src2, Mat dst, Mat mask = Mat(), int dtype = -1)
        if prop_name:
            c_decl = "%s %s::%s" % (fi.ctype, fi.classname, prop_name)
        else:
            decl_args = []
            for a in fi.args:
                s = a.ctype or ' _hidden_ '
                if a.pointer:
                    s += "*"
                elif a.out:
                    s += "&"
                s += " " + a.name
                if a.defval:
                    s += " = "+a.defval
                decl_args.append(s)
            c_decl = "%s %s %s(%s)" % ( fi.static, fi.ctype, fi.cname, ", ".join(decl_args) )

        # java comment
        j_code.write( "\n    //\n    // C++: %s\n    //\n\n" % c_decl )
        # check if we 'know' all the types
        if fi.ctype not in type_dict: # unsupported ret type
            msg = "// Return type '%s' is not supported, skipping the function\n\n" % fi.ctype
            self.skipped_func_list.append(c_decl + "\n" + msg)
            j_code.write( " "*4 + msg )
            logging.warning("SKIP:" + c_decl.strip() + "\t due to RET type" + fi.ctype)
            return
        for a in fi.args:
            if a.ctype not in type_dict:
                if not a.defval and a.ctype.endswith("*"):
                    a.defval = 0
                if a.defval:
                    a.ctype = ''
                    continue
                msg = "// Unknown type '%s' (%s), skipping the function\n\n" % (a.ctype, a.out or "I")
                self.skipped_func_list.append(c_decl + "\n" + msg)
                j_code.write( " "*4 + msg )
                logging.warning("SKIP:" + c_decl.strip() + "\t due to ARG type" + a.ctype + "/" + (a.out or "I"))
                return

        self.ported_func_list.append(c_decl)

        # jn & cpp comment
        jn_code.write( "\n    // C++: %s\n" % c_decl )
        cpp_code.write( "\n//\n// %s\n//\n" % c_decl )

        # java args
        args = fi.args[:] # copy
        j_signatures=[]
        suffix_counter = int(ci.methods_suffixes.get(fi.jname, -1))
        while True:
            suffix_counter += 1
            ci.methods_suffixes[fi.jname] = suffix_counter
             # java native method args
            jn_args = []
            # jni (cpp) function args
            jni_args = [ArgInfo([ "env", "env", "", [], "" ]), ArgInfo([ "cls", "", "", [], "" ])]
            j_prologue = []
            j_epilogue = []
            c_prologue = []
            c_epilogue = []
            if type_dict[fi.ctype]["jni_type"] == "jdoubleArray":
                fields = type_dict[fi.ctype]["jn_args"]
                c_epilogue.append( \
                    ("jdoubleArray _da_retval_ = env->NewDoubleArray(%(cnt)i);  " +
                     "jdouble _tmp_retval_[%(cnt)i] = {%(args)s}; " +
                     "env->SetDoubleArrayRegion(_da_retval_, 0, %(cnt)i, _tmp_retval_);") %
                    { "cnt" : len(fields), "args" : ", ".join(["(jdouble)_retval_" + f[1] for f in fields]) } )
            if fi.classname and fi.ctype and not fi.static: # non-static class method except c-tor
                # adding 'self'
                jn_args.append ( ArgInfo([ "__int64", "nativeObj", "", [], "" ]) )
                jni_args.append( ArgInfo([ "__int64", "self", "", [], "" ]) )
            ci.addImports(fi.ctype)
            for a in args:
                if not a.ctype: # hidden
                    continue
                ci.addImports(a.ctype)
                if "v_type" in type_dict[a.ctype]: # pass as vector
                    if type_dict[a.ctype]["v_type"] in ("Mat", "vector_Mat"): #pass as Mat or vector_Mat
                        jn_args.append  ( ArgInfo([ "__int64", "%s_mat.nativeObj" % a.name, "", [], "" ]) )
                        jni_args.append ( ArgInfo([ "__int64", "%s_mat_nativeObj" % a.name, "", [], "" ]) )
                        c_prologue.append( type_dict[a.ctype]["jni_var"] % {"n" : a.name} + ";" )
                        c_prologue.append( "Mat& %(n)s_mat = *((Mat*)%(n)s_mat_nativeObj)" % {"n" : a.name} + ";" )
                        if "I" in a.out or not a.out:
                            if type_dict[a.ctype]["v_type"] == "vector_Mat":
                                j_prologue.append( "List<Mat> %(n)s_tmplm = new ArrayList<Mat>((%(n)s != null) ? %(n)s.size() : 0);" % {"n" : a.name } )
                                j_prologue.append( "Mat %(n)s_mat = Converters.%(t)s_to_Mat(%(n)s, %(n)s_tmplm);" % {"n" : a.name, "t" : a.ctype} )
                            else:
                                if not type_dict[a.ctype]["j_type"].startswith("MatOf"):
                                    j_prologue.append( "Mat %(n)s_mat = Converters.%(t)s_to_Mat(%(n)s);" % {"n" : a.name, "t" : a.ctype} )
                                else:
                                    j_prologue.append( "Mat %s_mat = %s;" % (a.name, a.name) )
                            c_prologue.append( "Mat_to_%(t)s( %(n)s_mat, %(n)s );" % {"n" : a.name, "t" : a.ctype} )
                        else:
                            if not type_dict[a.ctype]["j_type"].startswith("MatOf"):
                                j_prologue.append( "Mat %s_mat = new Mat();" % a.name )
                            else:
                                j_prologue.append( "Mat %s_mat = %s;" % (a.name, a.name) )
                        if "O" in a.out:
                            if not type_dict[a.ctype]["j_type"].startswith("MatOf"):
                                j_epilogue.append("Converters.Mat_to_%(t)s(%(n)s_mat, %(n)s);" % {"t" : a.ctype, "n" : a.name})
                                j_epilogue.append( "%s_mat.release();" % a.name )
                            c_epilogue.append( "%(t)s_to_Mat( %(n)s, %(n)s_mat );" % {"n" : a.name, "t" : a.ctype} )
                    else: #pass as list
                        jn_args.append  ( ArgInfo([ a.ctype, a.name, "", [], "" ]) )
                        jni_args.append ( ArgInfo([ a.ctype, "%s_list" % a.name , "", [], "" ]) )
                        c_prologue.append(type_dict[a.ctype]["jni_var"] % {"n" : a.name} + ";")
                        if "I" in a.out or not a.out:
                            c_prologue.append("%(n)s = List_to_%(t)s(env, %(n)s_list);" % {"n" : a.name, "t" : a.ctype})
                        if "O" in a.out:
                            c_epilogue.append("Copy_%s_to_List(env,%s,%s_list);" % (a.ctype, a.name, a.name))
                else:
                    fields = type_dict[a.ctype].get("jn_args", ((a.ctype, ""),))
                    if "I" in a.out or not a.out or self.isWrapped(a.ctype): # input arg, pass by primitive fields
                        for f in fields:
                            jn_args.append ( ArgInfo([ f[0], a.name + f[1], "", [], "" ]) )
                            jni_args.append( ArgInfo([ f[0], a.name + f[1].replace(".","_").replace("[","").replace("]","").replace("_getNativeObjAddr()","_nativeObj"), "", [], "" ]) )
                    if "O" in a.out and not self.isWrapped(a.ctype): # out arg, pass as double[]
                        jn_args.append ( ArgInfo([ "double[]", "%s_out" % a.name, "", [], "" ]) )
                        jni_args.append ( ArgInfo([ "double[]", "%s_out" % a.name, "", [], "" ]) )
                        j_prologue.append( "double[] %s_out = new double[%i];" % (a.name, len(fields)) )
                        c_epilogue.append( \
                            "jdouble tmp_%(n)s[%(cnt)i] = {%(args)s}; env->SetDoubleArrayRegion(%(n)s_out, 0, %(cnt)i, tmp_%(n)s);" %
                            { "n" : a.name, "cnt" : len(fields), "args" : ", ".join(["(jdouble)" + a.name + f[1] for f in fields]) } )
                        if type_dict[a.ctype]["j_type"] in ('bool', 'int', 'long', 'float', 'double'):
                            j_epilogue.append('if(%(n)s!=null) %(n)s[0] = (%(t)s)%(n)s_out[0];' % {'n':a.name,'t':type_dict[a.ctype]["j_type"]})
                        else:
                            set_vals = []
                            i = 0
                            for f in fields:
                                set_vals.append( "%(n)s%(f)s = %(t)s%(n)s_out[%(i)i]" %
                                    {"n" : a.name, "t": ("("+type_dict[f[0]]["j_type"]+")", "")[f[0]=="double"], "f" : f[1], "i" : i}
                                )
                                i += 1
                            j_epilogue.append( "if("+a.name+"!=null){ " + "; ".join(set_vals) + "; } ")

            # calculate java method signature to check for uniqueness
            j_args = []
            for a in args:
                if not a.ctype: #hidden
                    continue
                jt = type_dict[a.ctype]["j_type"]
                if a.out and jt in ('bool', 'int', 'long', 'float', 'double'):
                    jt += '[]'
                j_args.append( jt + ' ' + a.name )
            j_signature = type_dict[fi.ctype]["j_type"] + " " + \
                fi.jname + "(" + ", ".join(j_args) + ")"
            logging.info("java: " + j_signature)

            if(j_signature in j_signatures):
                if args:
                    pop(args)
                    continue
                else:
                    break

            # java part:
            # private java NATIVE method decl
            # e.g.
            # private static native void add_0(long src1, long src2, long dst, long mask, int dtype);
            jn_code.write( Template(\
                "    private static native $type $name($args);\n").substitute(\
                type = type_dict[fi.ctype].get("jn_type", "double[]"), \
                name = fi.jname + '_' + str(suffix_counter), \
                args = ", ".join(["%s %s" % (type_dict[a.ctype]["jn_type"], a.name.replace(".","_").replace("[","").replace("]","").replace("_getNativeObjAddr()","_nativeObj")) for a in jn_args])
            ) );

            # java part:

            #java doc comment
            f_name = fi.name
            if fi.classname:
                f_name = fi.classname + "::" + fi.name
            java_doc = "//javadoc: " + f_name + "(%s)" % ", ".join([a.name for a in args if a.ctype])
            j_code.write(" "*4 + java_doc + "\n")

            # public java wrapper method impl (calling native one above)
            # e.g.
            # public static void add( Mat src1, Mat src2, Mat dst, Mat mask, int dtype )
            # { add_0( src1.nativeObj, src2.nativeObj, dst.nativeObj, mask.nativeObj, dtype );  }
            ret_type = fi.ctype
            if fi.ctype.endswith('*'):
                ret_type = ret_type[:-1]
            ret_val = type_dict[ret_type]["j_type"] + " retVal = "
            tail = ""
            ret = "return retVal;"
            if "v_type" in type_dict[ret_type]:
                j_type = type_dict[ret_type]["j_type"]
                if type_dict[ret_type]["v_type"] in ("Mat", "vector_Mat"):
                    tail = ")"
                    if j_type.startswith('MatOf'):
                        ret_val += j_type + ".fromNativeAddr("
                    else:
                        ret_val = "Mat retValMat = new Mat("
                        j_prologue.append( j_type + ' retVal = new Array' + j_type+'();')
                        j_epilogue.append('Converters.Mat_to_' + ret_type + '(retValMat, retVal);')
            elif ret_type.startswith("Ptr_"):
                ret_val = type_dict[fi.ctype]["j_type"] + " retVal = new " + type_dict[ret_type]["j_type"] + "("
                tail = ")"
            elif ret_type == "void":
                ret_val = ""
                ret = "return;"
            elif ret_type == "": # c-tor
                if fi.classname and ci.base:
                    ret_val = "super( "
                    tail = " )"
                else:
                    ret_val = "nativeObj = "
                ret = "return;"
            elif self.isWrapped(ret_type): # wrapped class
                ret_val = type_dict[ret_type]["j_type"] + " retVal = new " + self.getClass(ret_type).jname + "("
                tail = ")"
            elif "jn_type" not in type_dict[ret_type]:
                ret_val = type_dict[fi.ctype]["j_type"] + " retVal = new " + type_dict[ret_type]["j_type"] + "("
                tail = ")"

            static = "static"
            if fi.classname:
                static = fi.static

            j_code.write( Template(\
"""    public $static $j_type $j_name($j_args)
    {
        $prologue
        $ret_val$jn_name($jn_args_call)$tail;
        $epilogue
        $ret
    }

"""
                ).substitute(\
                    ret = ret, \
                    ret_val = ret_val, \
                    tail = tail, \
                    prologue = "\n        ".join(j_prologue), \
                    epilogue = "\n        ".join(j_epilogue), \
                    static=static, \
                    j_type=type_dict[fi.ctype]["j_type"], \
                    j_name=fi.jname, \
                    j_args=", ".join(j_args), \
                    jn_name=fi.jname + '_' + str(suffix_counter), \
                    jn_args_call=", ".join( [a.name for a in jn_args] ),\
                )
            )


            # cpp part:
            # jni_func(..) { _retval_ = cv_func(..); return _retval_; }
            ret = "return _retval_;"
            default = "return 0;"
            if fi.ctype == "void":
                ret = "return;"
                default = "return;"
            elif not fi.ctype: # c-tor
                ret = "return (jlong) _retval_;"
            elif "v_type" in type_dict[fi.ctype]: # c-tor
                if type_dict[fi.ctype]["v_type"] in ("Mat", "vector_Mat"):
                    ret = "return (jlong) _retval_;"
                else: # returned as jobject
                    ret = "return _retval_;"
            elif fi.ctype == "String":
                ret = "return env->NewStringUTF(_retval_.c_str());"
                default = 'return env->NewStringUTF("");'
            elif self.isWrapped(fi.ctype): # wrapped class:
                ret = "return (jlong) new %s(_retval_);" % self.fullTypeName(fi.ctype)
            elif fi.ctype.startswith('Ptr_'):
                c_prologue.append("typedef Ptr<%s> %s;" % (self.fullTypeName(fi.ctype[4:]), fi.ctype))
                ret = "return (jlong)(new %(ctype)s(_retval_));" % { 'ctype':fi.ctype }
            elif self.isWrapped(ret_type): # pointer to wrapped class:
                ret = "return (jlong) _retval_;"
            elif type_dict[fi.ctype]["jni_type"] == "jdoubleArray":
                ret = "return _da_retval_;"

            # hack: replacing func call with property set/get
            name = fi.name
            if prop_name:
                if args:
                    name = prop_name + " = "
                else:
                    name = prop_name + ";//"

            cvname = fi.fullName(isCPP=True)
            retval = self.fullTypeName(fi.ctype) + " _retval_ = "
            if fi.ctype == "void":
                retval = ""
            elif fi.ctype == "String":
                retval = "cv::" + retval
            elif "v_type" in type_dict[fi.ctype]: # vector is returned
                retval = type_dict[fi.ctype]['jni_var'] % {"n" : '_ret_val_vector_'} + " = "
                if type_dict[fi.ctype]["v_type"] in ("Mat", "vector_Mat"):
                    c_epilogue.append("Mat* _retval_ = new Mat();")
                    c_epilogue.append(fi.ctype+"_to_Mat(_ret_val_vector_, *_retval_);")
                else:
                    c_epilogue.append("jobject _retval_ = " + fi.ctype + "_to_List(env, _ret_val_vector_);")
            if len(fi.classname)>0:
                if not fi.ctype: # c-tor
                    retval = fi.fullClass(isCPP=True) + "* _retval_ = "
                    cvname = "new " + fi.fullClass(isCPP=True)
                elif fi.static:
                    cvname = fi.fullName(isCPP=True)
                else:
                    cvname = ("me->" if  not self.isSmartClass(ci) else "(*me)->") + name
                    c_prologue.append(\
                        "%(cls)s* me = (%(cls)s*) self; //TODO: check for NULL" \
                            % { "cls" : self.smartWrap(ci, fi.fullClass(isCPP=True))} \
                    )
            cvargs = []
            for a in args:
                if a.pointer:
                    jni_name = "&%(n)s"
                else:
                    jni_name = "%(n)s"
                    if not a.out and not "jni_var" in type_dict[a.ctype]:
                        # explicit cast to C type to avoid ambiguous call error on platforms (mingw)
                        # where jni types are different from native types (e.g. jint is not the same as int)
                        jni_name  = "(%s)%s" % (a.ctype, jni_name)
                if not a.ctype: # hidden
                    jni_name = a.defval
                cvargs.append( type_dict[a.ctype].get("jni_name", jni_name) % {"n" : a.name})
                if "v_type" not in type_dict[a.ctype]:
                    if ("I" in a.out or not a.out or self.isWrapped(a.ctype)) and "jni_var" in type_dict[a.ctype]: # complex type
                        c_prologue.append(type_dict[a.ctype]["jni_var"] % {"n" : a.name} + ";")
                    if a.out and "I" not in a.out and not self.isWrapped(a.ctype) and a.ctype:
                        c_prologue.append("%s %s;" % (a.ctype, a.name))

            rtype = type_dict[fi.ctype].get("jni_type", "jdoubleArray")
            clazz = ci.jname
            cpp_code.write ( Template( \
"""
${namespace}

JNIEXPORT $rtype JNICALL Java_org_opencv_${module}_${clazz}_$fname ($argst);

JNIEXPORT $rtype JNICALL Java_org_opencv_${module}_${clazz}_$fname
  ($args)
{
    static const char method_name[] = "$module::$fname()";
    try {
        LOGD("%s", method_name);
        $prologue
        $retval$cvname( $cvargs );
        $epilogue$ret
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    $default
}


""" ).substitute( \
        rtype = rtype, \
        module = self.module.replace('_', '_1'), \
        clazz = clazz.replace('_', '_1'), \
        fname = (fi.jname + '_' + str(suffix_counter)).replace('_', '_1'), \
        args  = ", ".join(["%s %s" % (type_dict[a.ctype].get("jni_type"), a.name) for a in jni_args]), \
        argst = ", ".join([type_dict[a.ctype].get("jni_type") for a in jni_args]), \
        prologue = "\n        ".join(c_prologue), \
        epilogue = "  ".join(c_epilogue) + ("\n        " if c_epilogue else ""), \
        ret = ret, \
        cvname = cvname, \
        cvargs = ", ".join(cvargs), \
        default = default, \
        retval = retval, \
        namespace = ('using namespace ' + ci.namespace.replace('.', '::') + ';') if ci.namespace else ''
    ) )

            # adding method signature to dictionarry
            j_signatures.append(j_signature)

            # processing args with default values
            if not args or not args[-1].defval:
                break
            while args and args[-1].defval:
                # 'smart' overloads filtering
                a = args.pop()
                if a.name in ('mask', 'dtype', 'ddepth', 'lineType', 'borderType', 'borderMode', 'criteria'):
                    break



    def gen_class(self, ci):
        logging.info("%s", ci)
        # constants
        if ci.private_consts:
            logging.info("%s", ci.private_consts)
            ci.j_code.write("""
    private static final int
            %s;\n\n""" % (",\n"+" "*12).join(["%s = %s" % (c.name, c.value) for c in ci.private_consts])
            )
        if ci.consts:
            logging.info("%s", ci.consts)
            ci.j_code.write("""
    public static final int
            %s;\n\n""" % (",\n"+" "*12).join(["%s = %s" % (c.name, c.value) for c in ci.consts])
            )
        # methods
        for fi in ci.getAllMethods():
            self.gen_func(ci, fi)
        # props
        for pi in ci.props:
            # getter
            getter_name = ci.fullName() + ".get_" + pi.name
            fi = FuncInfo( [getter_name, pi.ctype, [], []], self.namespaces ) # [ funcname, return_ctype, [modifiers], [args] ]
            self.gen_func(ci, fi, pi.name)
            if pi.rw:
                #setter
                setter_name = ci.fullName() + ".set_" + pi.name
                fi = FuncInfo( [ setter_name, "void", [], [ [pi.ctype, pi.name, "", [], ""] ] ], self.namespaces)
                self.gen_func(ci, fi, pi.name)

        # manual ports
        if ci.name in ManualFuncs:
            for func in ManualFuncs[ci.name].keys():
                ci.j_code.write ( "\n".join(ManualFuncs[ci.name][func]["j_code"]) )
                ci.jn_code.write( "\n".join(ManualFuncs[ci.name][func]["jn_code"]) )
                ci.cpp_code.write( "\n".join(ManualFuncs[ci.name][func]["cpp_code"]) )

        if ci.name != self.Module or ci.base:
            # finalize()
            ci.j_code.write(
"""
    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }
""" )

            ci.jn_code.write(
"""
    // native support for java finalize()
    private static native void delete(long nativeObj);
""" )

            # native support for java finalize()
            ci.cpp_code.write( \
"""
//
//  native support for java finalize()
//  static void %(cls)s::delete( __int64 self )
//
JNIEXPORT void JNICALL Java_org_opencv_%(module)s_%(j_cls)s_delete(JNIEnv*, jclass, jlong);

JNIEXPORT void JNICALL Java_org_opencv_%(module)s_%(j_cls)s_delete
  (JNIEnv*, jclass, jlong self)
{
    delete (%(cls)s*) self;
}

""" % {"module" : module.replace('_', '_1'), "cls" : self.smartWrap(ci, ci.fullName(isCPP=True)), "j_cls" : ci.jname.replace('_', '_1')}
            )

    def getClass(self, classname):
        return self.classes[classname or self.Module]

    def isWrapped(self, classname):
        name = classname or self.Module
        return name in self.classes

    def isSmartClass(self, ci):
        '''
        Check if class stores Ptr<T>* instead of T* in nativeObj field
        '''
        if ci.smart != None:
            return ci.smart

        # if parents are smart (we hope) then children are!
        # if not we believe the class is smart if it has "create" method
        ci.smart = False
        if ci.base or ci.name == 'Algorithm':
            ci.smart = True
        else:
            for fi in ci.methods:
                if fi.name == "create":
                    ci.smart = True
                    break

        return ci.smart

    def smartWrap(self, ci, fullname):
        '''
        Wraps fullname with Ptr<> if needed
        '''
        if self.isSmartClass(ci):
            return "Ptr<" + fullname + ">"
        return fullname


if __name__ == "__main__":

    # parse command line parameters
    import argparse
    arg_parser = argparse.ArgumentParser(description='OpenCV Java Wrapper Generator')
    arg_parser.add_argument('-p', '--parser', required=True, help='OpenCV header parser')
    arg_parser.add_argument('-m', '--module', required=True, help='OpenCV module name')
    arg_parser.add_argument('-s', '--srcfiles', required=True, nargs='+', help='Source headers to be wrapped')
    arg_parser.add_argument('-c', '--common', nargs='*', help='Common headers')
    arg_parser.add_argument('-t', '--gendict', nargs='*', help='Custom module dictionaries for C++ to Java conversion')

    args=arg_parser.parse_args()

    # import header parser
    hdr_parser_path = os.path.abspath(args.parser)
    if hdr_parser_path.endswith(".py"):
        hdr_parser_path = os.path.dirname(hdr_parser_path)
    sys.path.append(hdr_parser_path)
    import hdr_parser

    module = args.module
    srcfiles = args.srcfiles
    common_headers= args.common
    gen_dict_files = args.gendict

    dstdir = "."

    # initialize logger
    logging.basicConfig(filename='%s/%s.log' % (dstdir, module), format=None, filemode='w', level=logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(handler)

    # load dictionaries
    for gdf in gen_dict_files:
        with open(gdf) as f:
            gen_type_dict = json.load(f)
            if "class_ignore_list" in gen_type_dict:
                class_ignore_list += gen_type_dict["class_ignore_list"]
            if "const_ignore_list" in gen_type_dict:
                const_ignore_list += gen_type_dict["const_ignore_list"]
            if "const_private_list" in gen_type_dict:
                const_private_list += gen_type_dict["const_private_list"]
            if "missing_consts" in gen_type_dict:
                missing_consts.update(gen_type_dict["missing_consts"])
            if "type_dict" in gen_type_dict:
                type_dict.update(gen_type_dict["type_dict"])
            if "ManualFuncs" in gen_type_dict:
                ManualFuncs.update(gen_type_dict["ManualFuncs"])
            if "func_arg_fix" in gen_type_dict:
                func_arg_fix.update(gen_type_dict["func_arg_fix"])

    # launch Java Wrapper generator
    generator = JavaWrapperGenerator()
    generator.gen(srcfiles, module, dstdir, common_headers)
