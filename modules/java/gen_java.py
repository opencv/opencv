import sys, re, os.path
from string import Template

try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO

# c_type    : { java/jni correspondence }
type_dict = {
# "simple"  : { j_type : "?", jn_type : "?", jni_type : "?", suffix : "?" },
    ""        : { "j_type" : "", "jn_type" : "long", "jni_type" : "jlong" }, # c-tor ret_type
    "void"    : { "j_type" : "void", "jn_type" : "void", "jni_type" : "void" },
    "env"     : { "j_type" : "", "jn_type" : "", "jni_type" : "JNIEnv*"},
    "cls"     : { "j_type" : "", "jn_type" : "", "jni_type" : "jclass"},
    "bool"    : { "j_type" : "boolean", "jn_type" : "boolean", "jni_type" : "jboolean", "suffix" : "Z" },
    "int"     : { "j_type" : "int", "jn_type" : "int", "jni_type" : "int", "suffix" : "I" },
    "long"    : { "j_type" : "int", "jn_type" : "int", "jni_type" : "int", "suffix" : "I" },
    "float"   : { "j_type" : "float", "jn_type" : "float", "jni_type" : "jfloat", "suffix" : "F" },
    "double"  : { "j_type" : "double", "jn_type" : "double", "jni_type" : "jdouble", "suffix" : "D" },
    "size_t"  : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "__int64" : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
# "complex" : { j_type : "?", jn_args : (("", ""),), jn_name : "", jni_var : "", jni_name : "", "suffix" : "?" },
    "Mat"     : { "j_type" : "Mat", "jn_type" : "long", "jn_args" : (("__int64", ".nativeObj"),),
                  "jni_var" : "Mat& %(n)s = *((Mat*)%(n)s_nativeObj)",
                  "jni_type" : "jlong", #"jni_name" : "*%(n)s",
                  "suffix" : "J" },
    "Point"   : { "j_type" : "Point", "jn_args" : (("double", ".x"), ("double", ".y")),
                  "jni_var" : "cv::Point %(n)s((int)%(n)s_x, (int)%(n)s_y)",
                  "suffix" : "DD"},
    "Point2f" : { "j_type" : "Point", "jn_args" : (("double", ".x"), ("double", ".y")),
                  "jni_var" : "cv::Point2f %(n)s((float)%(n)s_x, (float)%(n)s_y)",
                  "suffix" : "DD"},
    "Point2d" : { "j_type" : "Point", "jn_args" : (("double", ".x"), ("double", ".y")),
                  "jni_var" : "cv::Point2d %(n)s(%(n)s_x, %(n)s_y)",
                  "suffix" : "DD"},
    "Point3i" : { "j_type" : "Point", "jn_args" : (("double", ".x"), ("double", ".y"), ("double", ".z")),
                  "jni_var" : "cv::Point3i %(n)s((int)%(n)s_x, (int)%(n)s_y, (int)%(n)s_z)",
                  "suffix" : "DDD"},
    "Point3f" : { "j_type" : "Point", "jn_args" : (("double", ".x"), ("double", ".y"), ("double", ".z")),
                  "jni_var" : "cv::Point3f %(n)s((float)%(n)s_x, (float)%(n)s_y, (float)%(n)s_z)",
                  "suffix" : "DDD"},
    "Point3d" : { "j_type" : "Point", "jn_args" : (("double", ".x"), ("double", ".y"), ("double", ".z")),
                  "jni_var" : "cv::Point3d %(n)s(%(n)s_x, %(n)s_y, %(n)s_z)",
                  "suffix" : "DDD"},
    "Rect"    : { "j_type" : "Rect",  "jn_args" : (("int", ".x"), ("int", ".y"), ("int", ".width"), ("int", ".height")),
                  "jni_var" : "cv::Rect %(n)s(%(n)s_x, %(n)s_y, %(n)s_width, %(n)s_height)",
                  "suffix" : "IIII"},
    "Size"    : { "j_type" : "Size",  "jn_args" : (("double", ".width"), ("double", ".height")),
                  "jni_var" : "cv::Size %(n)s((int)%(n)s_width, (int)%(n)s_height)",
                  "suffix" : "DD"},
    "Size2f"  : { "j_type" : "Size",  "jn_args" : (("double", ".width"), ("double", ".height")),
                  "jni_var" : "cv::Size2f %(n)s((float)%(n)s_width, (float)%(n)s_height)",
                  "suffix" : "DD"},
 "RotatedRect": { "j_type" : "RotatedRect",  "jn_args" : (("double", ".center.x"), ("double", ".center.y"), ("double", ".size.width"), ("double", ".size.height"), ("double", ".angle")),
                  "jni_var" : "cv::RotatedRect %(n)s(cv::Point2f(%(n)s_center_x, %(n)s_center_y), cv::Size2f(%(n)s_size_width, %(n)s_size_height), %(n)s_angle)",
                  "suffix" : "DDDDD"},
    "Scalar"  : { "j_type" : "Scalar",  "jn_args" : (("double", ".v0"), ("double", ".v1"), ("double", ".v2"), ("double", ".v3")),
                  "jni_var" : "cv::Scalar %(n)s(%(n)s_v0, %(n)s_v1, %(n)s_v2, %(n)s_v3)",
                  "suffix" : "DDDD"},
    "Range"   : { "j_type" : "Range",  "jn_args" : (("int", ".start"), ("int", ".end")),
                  "jni_var" : "cv::Range %(n)s(%(n)s_start, %(n)s_end)",
                  "suffix" : "II"},
    "CvSlice"   : { "j_type" : "Range",  "jn_args" : (("int", ".start"), ("int", ".end")),
                  "jni_var" : "cv::Range %(n)s(%(n)s_start, %(n)s_end)",
                  "suffix" : "II"},
    "string"  : { "j_type" : "java.lang.String",  "jn_type" : "java.lang.String",
                  "jni_type" : "jstring", "jni_name" : "n_%(n)s",
                  "jni_var" : 'const char* utf_%(n)s = env->GetStringUTFChars(%(n)s, 0); std::string n_%(n)s( utf_%(n)s ? utf_%(n)s : "" ); env->ReleaseStringUTFChars(%(n)s, utf_%(n)s)',
                  "suffix" : "Ljava_lang_String_2"},
    "String"  : { "j_type" : "java.lang.String",  "jn_type" : "java.lang.String",
                  "jni_type" : "jstring", "jni_name" : "n_%(n)s",
                  "jni_var" : 'const char* utf_%(n)s = env->GetStringUTFChars(%(n)s, 0); String n_%(n)s( utf_%(n)s ? utf_%(n)s : "" ); env->ReleaseStringUTFChars(%(n)s, utf_%(n)s)',
                  "suffix" : "Ljava_lang_String_2"},
    "c_string": { "j_type" : "java.lang.String",  "jn_type" : "java.lang.String",
                  "jni_type" : "jstring", "jni_name" : "n_%(n)s.c_str()",
                  "jni_var" : 'const char* utf_%(n)s = env->GetStringUTFChars(%(n)s, 0); std::string n_%(n)s( utf_%(n)s ? utf_%(n)s : "" ); env->ReleaseStringUTFChars(%(n)s, utf_%(n)s)',
                  "suffix" : "Ljava_lang_String_2"},

}

class ConstInfo(object):
    def __init__(self, cname, name, val):
##        self.name = re.sub(r"^cv\.", "", name).replace(".", "_")
        self.cname = cname
        self.name =  re.sub(r"^Cv", "", name)
        #self.name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
        #self.name = self.name.upper()
        self.value = val


class ClassInfo(object):
    def __init__(self, decl): # [ 'class/struct cname', [bases], [modlist] ]
        name = decl[0]
        name = name[name.find(" ")+1:].strip()
        self.cname = self.name = self.jname = re.sub(r"^cv\.", "", name)
        self.cname =self.cname.replace(".", "::")
        #self.jname =  re.sub(r"^Cv", "", self.jname)
        self.methods = {}
        self.consts = [] # using a list to save the occurence order
        for m in decl[2]:
            if m.startswith("="):
                self.jname = m[1:]


class ArgInfo(object):
    def __init__(self, arg_tuple): # [ ctype, name, def val, [mod], argno ]
        self.ctype = arg_tuple[0]
        self.name = arg_tuple[1]
        self.defval = arg_tuple[2]
        self.out = ""
        if "/O" in arg_tuple[3]:
            self.out = "O"
        if "/IO" in arg_tuple[3]:
            self.out = "IO"


class FuncInfo(object):
    def __init__(self, decl): # [ funcname, return_ctype, [modifiers], [args] ]
        name = re.sub(r"^cv\.", "", decl[0])
        self.cname = name.replace(".", "::")
        classname = ""
        dpos = name.rfind(".")
        if dpos >= 0:
            classname = name[:dpos]
            name = name[dpos+1:]
        self.classname = classname
        self.jname = self.name = name
        if "[" in name:
            self.jname = "getelem"
        for m in decl[2]:
            if m.startswith("="):
                self.jname = m[1:]
        self.jn_name = "n_" + self.jname
        self.jni_name= re.sub(r"_", "_1", self.jn_name)
        if self.classname:
            self.jni_name = "00024" + self.classname + "_" + self.jni_name
        self.static = ["","static"][ "/S" in decl[2] ]
        self.ctype = decl[1] or ""
        self.args = []
        #self.jni_suffix = "__"
        #if self.classname and self.ctype and not self.static: # non-static class methods except c-tors
        #    self.jni_suffix += "J" # artifical 'self'
        for a in decl[3]:
            ai = ArgInfo(a)
            self.args.append(ai)
        #    self.jni_suffix += ctype2j.get(ai.ctype, ["","","",""])[3]



class FuncFamilyInfo(object):
    def __init__(self, decl): # [ funcname, return_ctype, [modifiers], [args] ]
        self.funcs = []
        self.funcs.append( FuncInfo(decl) )
        self.jname = self.funcs[0].jname
        self.isconstructor = self.funcs[0].name == self.funcs[0].classname



    def add_func(self, fi):
        self.funcs.append( fi )


class JavaWrapperGenerator(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.classes = { "Mat" : ClassInfo([ 'class Mat', [], [] ]) }
        self.funcs = {}
        self.consts = [] # using a list to save the occurence order
        self.module = ""
        self.java_code = StringIO()
        self.jn_code = StringIO()
        self.cpp_code = StringIO()
        self.ported_func_counter = 0
        self.ported_func_list = []
        self.skipped_func_list = []
        self.total_func_counter = 0

    def add_class(self, decl):
        classinfo = ClassInfo(decl)
        if classinfo.name in self.classes:
            print "Generator error: class %s (%s) is duplicated" % \
                    (classinfo.name, classinfo.cname)
            sys.exit(-1)
        self.classes[classinfo.name] = classinfo
        if classinfo.name in type_dict:
            print "Duplicated class: " + classinfo.name
            sys.exit(-1)
        type_dict[classinfo.name] = \
            { "j_type" : classinfo.name,  "jn_args" : (("__int64", ".nativeObj"),),
              "jni_name" : "(*("+classinfo.name+"*)%(n)s_nativeObj)",
              "suffix" : "J" }


    def add_const(self, decl): # [ "const cname", val, [], [] ]
        consts = self.consts
        name = decl[0].replace("const ", "").strip()
        name = re.sub(r"^cv\.", "", name)
        cname = name.replace(".", "::")
        # check if it's a class member
        dpos = name.rfind(".")
        if dpos >= 0:
            classname = name[:dpos]
            name = name[dpos+1:]
            if classname in self.classes:
                consts = self.classes[classname].consts
            else:
                # this class isn't wrapped
                # skipping this const
                return
        constinfo = ConstInfo(cname, name, decl[1])
        # checking duplication
        for c in consts:
            if c.name == constinfo.name:
                print "Generator error: constant %s (%s) is duplicated" \
                        % (constinfo.name, constinfo.cname)
                sys.exit(-1)
        consts.append(constinfo)

    def add_func(self, decl):
        ffi = FuncFamilyInfo(decl)
        func_map = self.funcs
        classname = ffi.funcs[0].classname
        if classname:
            if classname in self.classes:
                func_map = self.classes[classname].methods
            else:
                print "Generator error: the class %s for method %s is missing" % \
                        (classname, ffi.jname)
                sys.exit(-1)
        if ffi.jname in func_map:
            func_map[ffi.jname].add_func(ffi.funcs[0])
        else:
            func_map[ffi.jname] = ffi

    def save(self, path, name, buf):
        f = open(path + "/" + name, "wt")
        f.write(buf.getvalue())
        f.close()

    def gen(self, srcfiles, module, output_path):
        self.clear()
        self.module = module
        parser = hdr_parser.CppHeaderParser()

        # step 1: scan the headers and build more descriptive maps of classes, consts, functions
        for hdr in srcfiles:
            decls = parser.parse(hdr)
            for decl in decls:
                name = decl[0]
                if name.startswith("struct") or name.startswith("class"):
                    self.add_class(decl)
                    pass
                elif name.startswith("const"):
                    self.add_const(decl)
                else: # function
                    self.add_func(decl)
                    pass

        # java module header
        self.java_code.write("package org.opencv;\n\npublic class %s {\n" % module)

        if module == "core":
            self.java_code.write(\
"""
    private static final int
            CV_8U  = 0,
            CV_8S  = 1,
            CV_16U = 2,
            CV_16S = 3,
            CV_32S = 4,
            CV_32F = 5,
            CV_64F = 6,
            CV_USRTYPE1 = 7;

""" )

        if module == "imgproc":
            self.java_code.write(\
"""
    public static final int
            IPL_BORDER_CONSTANT = 0,
            IPL_BORDER_REPLICATE = 1,
            IPL_BORDER_REFLECT = 2,
            IPL_BORDER_WRAP = 3,
            IPL_BORDER_REFLECT_101 = 4,
            IPL_BORDER_TRANSPARENT = 5;
""" )

        if module == "calib3d":
            self.java_code.write(\
"""
    public static final int
            CV_LMEDS = 4,
            CV_RANSAC = 8,
            CV_FM_LMEDS = CV_LMEDS,
            CV_FM_RANSAC = CV_RANSAC;

    public static final int
            CV_FM_7POINT = 1,
            CV_FM_8POINT = 2;

    public static final int
            CV_CALIB_USE_INTRINSIC_GUESS = 1,
            CV_CALIB_FIX_ASPECT_RATIO = 2,
            CV_CALIB_FIX_PRINCIPAL_POINT = 4,
            CV_CALIB_ZERO_TANGENT_DIST = 8,
            CV_CALIB_FIX_FOCAL_LENGTH = 16,
            CV_CALIB_FIX_K1 = 32,
            CV_CALIB_FIX_K2 = 64,
            CV_CALIB_FIX_K3 = 128,
            CV_CALIB_FIX_K4 = 2048,
            CV_CALIB_FIX_K5 = 4096,
            CV_CALIB_FIX_K6 = 8192,
            CV_CALIB_RATIONAL_MODEL = 16384,
            CV_CALIB_FIX_INTRINSIC = 256,
            CV_CALIB_SAME_FOCAL_LENGTH = 512,
            CV_CALIB_ZERO_DISPARITY = 1024;
""" )

        # java native stuff
        self.jn_code.write("""
    //
    // native stuff
    //
    static { System.loadLibrary("opencv_java");	}
""")

        # cpp module header
        self.cpp_code.write(\
"""//
// This file is auto-generated, please don't edit!
//

#include <jni.h>

#ifdef DEBUG
#include <android/log.h>
#define MODULE_LOG_TAG "OpenCV.%s"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, MODULE_LOG_TAG, __VA_ARGS__))
#endif // DEBUG

""" % module)
        self.cpp_code.write( "\n".join(['#include "opencv2/%s/%s"' % (module, os.path.basename(f)) \
                            for f in srcfiles]) )
        self.cpp_code.write('\nusing namespace cv;\n')
        self.cpp_code.write('\n\nextern "C" {\n\n')

        # step 2: generate the code for global constants
        self.gen_consts()

        # step 3: generate the code for all the global functions
        self.gen_funcs()

        # step 4: generate code for the classes
        self.gen_classes()

        # module tail
        self.java_code.write("\n\n" + self.jn_code.getvalue() + "\n")
        self.java_code.write("}\n")
        self.cpp_code.write('} // extern "C"\n')

        self.save(output_path, module+".java", self.java_code)
        self.save(output_path, module+".cpp",  self.cpp_code)
        # report
        report = StringIO()
        report.write("PORTED FUNCs LIST (%i of %i):\n\n" % \
            (self.ported_func_counter, self.total_func_counter) \
        )
        report.write("\n".join(self.ported_func_list))
        report.write("\n\nSKIPPED FUNCs LIST (%i of %i):\n\n" % \
            (self.total_func_counter - self.ported_func_counter, self.total_func_counter) \
        )
        report.write("".join(self.skipped_func_list))
        self.save(output_path, module+".txt", report)

        print "Done %i of %i funcs." % (self.ported_func_counter, self.total_func_counter)


    def gen_consts(self):
        # generate the code for global constants
        if self.consts:
            self.java_code.write("""
    public static final int
            """ + """,
            """.join(["%s = %s" % (c.name, c.value) for c in self.consts]) + \
            ";\n\n")


    def gen_func(self, fi, isoverload, jn_code):
        self.total_func_counter += 1

        # // C++: c_decl
        # e.g:
        # //  C++: void add(Mat src1, Mat src2, Mat dst, Mat mask = Mat(), int dtype = -1)
        c_decl = "%s %s %s(%s)" % \
            ( fi.static, fi.ctype, fi.cname, \
              ", ".join(a.ctype + " " + a.name + [""," = "+a.defval][bool(a.defval)] for a in fi.args) )
        indent = " " * 4
        if fi.classname:
            indent += " " * 4
        # java comment
        self.java_code.write( "\n%s// C++: %s\n" % (indent, c_decl) )
        # check if we 'know' all the types
        type_info = type_dict.get(fi.ctype)
        if not (type_info and type_info.get("jn_type")): # unsupported ret type
            msg = "// Return type '%s' is not supported, skipping the function\n\n" % fi.ctype
            self.skipped_func_list.append(c_decl + "\n" + msg)
            self.java_code.write( indent + msg )
            #self.cpp_code.write( msg )
            print "SKIP:", c_decl, "\n\tdue to RET type", fi.ctype
            return
        for a in fi.args:
            if a.ctype not in type_dict:
                msg = "// Unknown type '%s', skipping the function\n\n" % a.ctype
                self.skipped_func_list.append(c_decl + "\n" + msg)
                self.java_code.write( indent + msg )
                #self.cpp_code.write( msg )
                print "SKIP:", c_decl, "\n\tdue to ARG type", a.ctype, a.out
                return
            if a.ctype != "Mat" and "jn_args" in type_dict[a.ctype] and a.out: # complex out args not yet supported
                msg = "// Unsupported type '%s&', skipping the function\n\n" % a.ctype
                self.skipped_func_list.append(c_decl + "\n" + msg)
                self.java_code.write( indent + msg )
                #self.cpp_code.write( msg )
                print "SKIP:", c_decl, "\n\tdue to OUT ARG of type", a.ctype, a.out
                return

        self.ported_func_counter += 1
        self.ported_func_list.append(c_decl)

        # jn & cpp comment
        jn_code.write( "\n%s// C++: %s\n" % (indent, c_decl) )
        self.cpp_code.write( "\n//\n// %s\n//\n" % c_decl )

        # java args
        args = fi.args[:] # copy
        if args and args[-1].defval:
            isoverload = True

        while True:

             # java native method args
            jn_args = []
            # jni (cpp) function args
            jni_args = [ArgInfo([ "env", "env", "", [], "" ]), ArgInfo([ "cls", "cls", "", [], "" ])]
            suffix = "__"
            if fi.classname and fi.ctype and not fi.static: # non-static class method except c-tor
                # adding 'self'
                jn_args.append ( ArgInfo([ "__int64", "nativeObj", "", [], "" ]) )
                jni_args.append( ArgInfo([ "__int64", "self", "", [], "" ]) )
                suffix += "J"
            for a in args:
                suffix += type_dict[a.ctype].get("suffix") or ""
                fields = type_dict[a.ctype].get("jn_args") or []
                if fields: # complex type
                    for f in fields:
                        jn_args.append ( ArgInfo([ f[0], a.name + f[1], "", [], "" ]) )
                        jni_args.append( ArgInfo([ f[0], a.name + f[1].replace(".","_"), "", [], "" ]) )
                else:
                    jn_args.append(a)
                    jni_args.append(a)

            # java part:
            # private java NATIVE method decl
            # e.g.
            # private static native void n_add(long src1, long src2, long dst, long mask, int dtype);
            jn_code.write( Template(\
                "${indent}private static native $jn_type $jn_name($jn_args);\n").substitute(\
                indent = indent, \
                jn_type = type_dict[fi.ctype]["jn_type"], \
                jn_name = fi.jn_name, \
                jn_args = ", ".join(["%s %s" % (type_dict[a.ctype]["jn_type"], a.name.replace(".","_")) for a in jn_args])
            ) );

            # java part:

            #java doc comment
            f_name = fi.name
            if fi.classname:
                f_name = fi.classname + "::" + fi.name
            self.java_code.write(indent + "//javadoc: " + f_name + "(%s)\n" % \
                ", ".join([a.name for a in args])
            )

            # public java wrapper method impl (calling native one above)
            # e.g.
            # public static void add( Mat src1, Mat src2, Mat dst, Mat mask, int dtype )
            # { n_add( src1.nativeObj, src2.nativeObj, dst.nativeObj, mask.nativeObj, dtype );  }
            impl_code = "return $jn_name($jn_args_call);"
            if fi.ctype == "void":
                impl_code = "$jn_name($jn_args_call);"
            elif fi.ctype == "": # c-tor
                impl_code = "nativeObj = $jn_name($jn_args_call);"
            elif fi.ctype in self.classes: # wrapped class
                impl_code = " return new %s( $jn_name($jn_args_call) ); " % \
                    self.classes[fi.ctype].jname

            static = "static"
            if fi.classname:
                static = fi.static

            self.java_code.write( Template(\
                "${indent}public $static $j_type $j_name($j_args)").substitute(\
                indent = indent, \
                static=static, \
                j_type=type_dict[fi.ctype]["j_type"], \
                j_name=fi.jname, \
                j_args=", ".join(["%s %s" % (type_dict[a.ctype]["j_type"], a.name) for a in args]) \
            ) )

            self.java_code.write( Template("\n$indent{ " + impl_code + " }\n").substitute(\
                indent = indent, \
                jn_name=fi.jn_name, \
                jn_args_call=", ".join( [a.name for a in jn_args] )\
            ) )

            # cpp part:
            # jni_func(..) { return cv_func(..); }
            ret = "return "
            ext = ""
            default = "return 0;"
            if fi.ctype == "void":
                ret = ""
                default = ""
            elif fi.ctype == "string":
                ret = "return env->NewStringUTF"
                ext = ".c_str()"
                default = 'return env->NewStringUTF("");'
            elif fi.ctype in self.classes: # wrapped class:
                ret = "return (jlong) new " + self.classes[fi.ctype].jname

            cvname = "cv::" + fi.name
            j2cvargs = []
            if fi.classname:
                if not fi.ctype: # c-tor
                    cvname = "(jlong) new " + fi.classname
                elif fi.static:
                    cvname = "%s::%s" % (fi.classname, fi.name)
                else:
                    cvname = "me->" + fi.name
                    j2cvargs.append(\
                        "%(cls)s* me = (%(cls)s*) self; //TODO: check for NULL" \
                            % { "cls" : fi.classname} \
                    )
            cvargs = []
            for a in args:
                cvargs.append( type_dict[a.ctype].get("jni_name", "%(n)s") % {"n" : a.name})
                if "jni_var" in type_dict[a.ctype]: # complex type
                    j2cvargs.append(type_dict[a.ctype]["jni_var"] % {"n" : a.name} + ";")

            rtype = type_dict[fi.ctype]["jni_type"]
            self.cpp_code.write ( Template( \
"""

JNIEXPORT $rtype JNICALL Java_org_opencv_${module}_$fname
  ($args)
{
    try {
#ifdef DEBUG
        LOGD("$module::$fname()");
#endif // DEBUG
        $j2cv
        $ret( $cvname( $cvargs )$ext );
    } catch(cv::Exception e) {
#ifdef DEBUG
        LOGD("$module::$fname() catched cv::Exception: %s", e.what());
#endif // DEBUG
        jclass je = env->FindClass("org/opencv/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        $default
    } catch (...) {
#ifdef DEBUG
        LOGD("$module::$fname() catched unknown exception (...)");
#endif // DEBUG
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {$module::$fname()}");
        $default
    }
}


""" ).substitute( \
        rtype = rtype, \
        module = self.module, \
        fname = fi.jni_name + ["",suffix][isoverload], \
        args = ", ".join(["%s %s" % (type_dict[a.ctype].get("jni_type"), a.name) for a in jni_args]), \
        j2cv = "\n        ".join([a for a in j2cvargs]), \
        ret = ret, \
        cvname = cvname, \
        cvargs = ", ".join([a for a in cvargs]), \
        ext = ext, \
        default = default
    ) )

            # processing args with default values
            if args and args[-1].defval:
                a = args.pop()
            else:
                break



    def gen_funcs(self):
        # generate the code for all the global functions
        indent = "\t"
        fflist = self.funcs.items()
        fflist.sort()
        for name, ffi in fflist:
            assert not ffi.funcs[0].classname, "Error: global func is a class member - "+name
            for fi in ffi.funcs:
                self.gen_func(fi, len(ffi.funcs)>1, self.jn_code)


    def gen_classes(self):
        # generate code for the classes (their methods and consts)
        indent = " " * 4
        indent_m = indent + " " * 4
        classlist = self.classes.items()
        classlist.sort()
        for name, ci in classlist:
            if name == "Mat":
                continue
            self.java_code.write( "\n\n" + indent + "// C++: class %s" % (ci.cname) + "\n" )
            self.java_code.write( indent + "//javadoc: " + name + "\n" ) #java doc comment
            self.java_code.write( indent + "public static class %s {\n\n" % (ci.jname) )
            # self
            self.java_code.write( indent_m + "protected final long nativeObj;\n" )
            self.java_code.write( indent_m + "protected %s(long addr) { nativeObj = addr; }\n\n" \
                % name );
            # constants
            if ci.consts:
                prefix = "\n" + indent_m + "\t"
                s = indent_m + "public static final int" + prefix +\
                    ("," + prefix).join(["%s = %s" % (c.name, c.value) for c in ci.consts]) + ";\n\n"
                self.java_code.write( s )
            # methods
            jn_code = StringIO()
            # c-tors
            fflist = ci.methods.items()
            fflist.sort()
            for n, ffi in fflist:
                if ffi.isconstructor:
                    for fi in ffi.funcs:
                        self.gen_func(fi, len(ffi.funcs)>1, jn_code)
            self.java_code.write( "\n" )
            for n, ffi in fflist:
                if not ffi.isconstructor:
                    for fi in ffi.funcs:
                        self.gen_func(fi, len(ffi.funcs)>1, jn_code)

            # finalize()
            self.java_code.write(
"""
        @Override
        protected void finalize() throws Throwable {
            n_delete(nativeObj);
            super.finalize();
        }

"""
            )

            self.java_code.write(indent_m + "// native stuff\n\n")
            self.java_code.write(indent_m + 'static { System.loadLibrary("opencv_java"); }\n')
            self.java_code.write( jn_code.getvalue() )
            self.java_code.write(
"""
        // native support for java finalize()
        private static native void n_delete(long nativeObj);
"""
            )
            self.java_code.write("\n" + indent + "}\n\n")

            # native support for java finalize()
            self.cpp_code.write( \
"""
//
//  native support for java finalize()
//  static void %(cls)s::n_delete( __int64 self )
//

JNIEXPORT void JNICALL Java_org_opencv_%(module)s_00024%(cls)s_n_1delete
  (JNIEnv* env, jclass cls, jlong self)
{
    delete (%(cls)s*) self;
}

""" % {"module" : module, "cls" : name}
            )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Usage:\n", \
            os.path.basename(sys.argv[0]), \
            "<full path to hdr_parser.py> <module name> <C++ header> [<C++ header>...]"
        print "Current args are: ", ", ".join(["'"+a+"'" for a in sys.argv])
        exit(0)

    dstdir = "."
    hdr_parser_path = os.path.abspath(sys.argv[1])
    if hdr_parser_path.endswith(".py"):
        hdr_parser_path = os.path.dirname(hdr_parser_path)
    sys.path.append(hdr_parser_path)
    import hdr_parser
    module = sys.argv[2]
    srcfiles = sys.argv[3:]
    print "Generating module '" + module + "' from headers:\n\t" + "\n\t".join(srcfiles)
    generator = JavaWrapperGenerator()
    generator.gen(srcfiles, module, dstdir)

