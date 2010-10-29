import opencv_parser, sys, re, cStringIO
from string import Template

gen_template_check_self = Template("""    if(!PyObject_TypeCheck(self, &pyopencv_${name}_Type))
        return failmsg("Incorrect type of self (must be '${name}' or its derivative)");
    $cname* _self_ = ((pyopencv_${name}_t*)self)->v;
""")

gen_template_call_constructor = Template("""self = PyObject_NEW(pyopencv_${name}_t, &pyopencv_${name}_Type);
        if(self) ERRWRAP2(self->v = new $cname""")

gen_template_parse_args = Template("""const char* keywords[] = { $kw_list, NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "$fmtspec", (char**)keywords, $parse_arglist)$code_cvt )""")

gen_template_func_body = Template("""$code_decl
    $code_parse
    {
        $code_fcall;
        return $code_retval;
    }
""")

gen_template_set_prop_from_map = Template("""
    if( PyMapping_HasKeyString(src, "$propname") )
    {
        tmp = PyMapping_GetItemString(src, "$propname");
        ok = tmp && pyopencv_to_$proptype(tmp, dst.$propname);
        Py_DECREF(tmp);
        if(!ok) return false;
    }""")

gen_template_decl_type = Template("""
/*
  $cname is the OpenCV C struct
  pyopencv_${name}_t is the Python object
*/

struct pyopencv_${name}_t
{
    PyObject_HEAD
    ${cname}* v;
};

static void pyopencv_${name}_dealloc(PyObject* self)
{
    delete ((pyopencv_${name}_t*)self)->v;
    PyObject_Del(self);
}

static PyObject* pyopencv_${name}_repr(PyObject* self)
{
    char str[1000];
    sprintf(str, "<$wname %p>", self);
    return PyString_FromString(str);
}

${getset_code}

static PyGetSetDef pyopencv_${name}_getseters[] =
{${getset_inits}
    {NULL}  /* Sentinel */
};

${methods_code}

static PyMethodDef pyopencv_${name}_methods[] =
{
${methods_inits}
    {NULL,          NULL}
};

static PyTypeObject pyopencv_${name}_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
    0,
    MODULESTR".$wname",
    sizeof(pyopencv_${name}_t),
};

static void pyopencv_${name}_specials(void)
{
    pyopencv_${name}_Type.tp_base = ${baseptr};
    pyopencv_${name}_Type.tp_dealloc = pyopencv_${name}_dealloc;
    pyopencv_${name}_Type.tp_repr = pyopencv_${name}_repr;
    pyopencv_${name}_Type.tp_getset = pyopencv_${name}_getseters;
    pyopencv_${name}_Type.tp_methods = pyopencv_${name}_methods;${extra_specials}
}

static PyObject* pyopencv_from_${name}_ptr(<$cname>* r)
{
    pyopencv_${name}_t *m = PyObject_NEW(pyopencv_${name}_t, &pyopencv_${name}_Type);
    m->v = r;
    return (PyObject*)m;
}

static bool pyopencv_to_${name}_ptr(PyObject* src, <$cname>*& dst, const char* name="")
{
    if( src == NULL or src == Py_None )
    {
        dst = 0;
        return true;
    }
    if(!PyObject_TypeCheck(src, &pyopencv_${name}_Type))
        return failmsg("Expected ${cname} for argument '%s'", name);
    dst = ((pyopencv_${name}_t*)src)->v;
    return true;
}
""")

gen_template_get_prop = Template("""
static PyObject* pyopencv_${name}_get_${member}(pyopencv_${name}_t* p, void *closure)
{
    return pyopencv_from_${membertype}(p->v->${member});
}
""")

gen_template_set_prop = Template("""
static int pyopencv_${name}_set_${member}(pyopencv_${name}_t* p, PyObject *value, void *closure)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the ${member} attribute");
        return -1;
    }
    return pyopencv_to_${membertype}(value, p->v->${member}) ? 0 : -1;
}
""")

gen_template_prop_init = Template("""
    {(char*)"${member}", (getter)pyopencv_${name}_get_${member}, NULL, (char*)"${member}", NULL},""")

gen_template_rw_prop_init = Template("""
    {(char*)"${member}", (getter)pyopencv_${name}_get_${member}, (setter)pyopencv_${name}_set_${member}, (char*)"${member}", NULL},""")

simple_argtype_mapping = {
    "bool": ("bool", "b", "pyopencv_from_bool", "0"),
    "int": ("int", "i", "pyopencv_from_int", "0"),
    "float": ("float", "f", "pyopencv_from_float", "0.f"),
    "double": ("double", "d", "pyopencv_from_double", "0"),
    "c_string": ("char*", "s", "pyopencv_from_c_string", '""')
}

class ClassProp(object):
    def __init__(self, decl):
        self.tp = decl[0]
        self.name = decl[1]
        self.readonly = True
        if "/RW" in decl[3]:
            self.readonly = False

class ClassInfo(object):
    def __init__(self, name, decl=None):
        self.cname = name.replace(".", "::")
        self.name = self.wname = re.sub(r"^cv\.", "", name)
        self.ismap = False
        self.methods = {}
        self.props = []
        self.consts = {}
        customname = False
        
        if decl:
            self.bases = decl[1].split()[1:]
            if len(self.bases) > 1:
                print "Error: class %s has more than 1 base class (not supported by Python C extensions)" % (self.name,)
                print "Bases: ", self.bases
                return sys.exit(-1)
            for m in decl[2]:
                if m.startswith("="):
                    self.wname = m[1:]
                    customname = True
                elif m == "/Map":
                    self.ismap = True
            self.props = [ClassProp(p) for p in decl[3]]
        
        if not customname and self.wname.startswith("Cv"):
            self.wname = self.wname[2:]
        
    def gen_map_code(self):
        code = "static bool pyopencv_to_%s(PyObject* src, %s& dst)\n{\n    PyObject* tmp;\n    bool ok;\n" % (self.name, self.cname)
        code += "".join([gen_template_set_prop_from_map.substitute(propname=p.name,proptype=p.tp) for p in self.props])
        code += "\n    return true;\n}"
        return code
        
    def gen_code(self, all_classes):
        if self.ismap:
            return self.gen_map_code()
        
        getset_code = ""
        getset_inits = ""
        
        sorted_props = [(p.name, p) for p in self.props]
        sorted_props.sort()
        
        for pname, p in sorted_props:
            getset_code += gen_template_get_prop.substitute(name=self.name, member=pname, membertype=p.tp)
            if p.readonly:
                getset_inits += gen_template_prop_init.substitute(name=self.name, member=pname)
            else:
                getset_code += gen_template_set_prop.substitute(name=self.name, member=pname, membertype=p.tp)
                getset_inits += gen_template_rw_prop_init.substitute(name=self.name, member=pname)
                
        methods_code = ""
        methods_inits = ""
        
        sorted_methods = self.methods.items()
        sorted_methods.sort()
        
        for mname, m in sorted_methods:
            methods_code += m.gen_code(all_classes)
            methods_inits += m.get_tab_entry()
        
        baseptr = "NULL"
        if self.bases and all_classes.has_key(self.bases[0]):
            baseptr = "&pyopencv_" + all_classes[self.bases[0]].name + "_Type"
        
        code = gen_template_decl_type.substitute(name=self.name, wname=self.wname, cname=self.cname,
            getset_code=getset_code, getset_inits=getset_inits,
            methods_code=methods_code, methods_inits=methods_inits,
            baseptr=baseptr, extra_specials="")
        
        return code
            
            
class ConstInfo(object):
    def __init__(self, name, val):
        self.cname = name.replace(".", "::")
        self.name = re.sub(r"^cv\.", "", name).replace(".", "_")
        if self.name.startswith("Cv"):
            self.name = self.name[2:]
        self.name = re.sub(r"([a-z])([A-Z])", r"\1_\2", self.name)
        self.name = self.name.upper()
        self.value = val
     
class ArgInfo(object):
    def __init__(self, arg_tuple):
        self.tp = arg_tuple[0]
        self.name = arg_tuple[1]
        self.defval = arg_tuple[2]
        self.isarray = False
        self.arraylen = 0
        self.arraycvt = None
        self.inputarg = True
        self.outputarg = False
        for m in arg_tuple[3]:
            if m == "/O":
                self.inputarg = False
                self.outputarg = True
            elif m == "/IO":
                self.inputarg = True
                self.outputarg = True
            elif m.startswith("/A"):
                self.isarray = True
                self.arraylen = m[2:].strip()
            elif m.startswith("/CA"):
                self.isarray = True
                self.arraycvt = m[2:].strip()
        self.py_inputarg = False
        self.py_outputarg = False
        
    def isbig(self):
        return self.tp == "Mat" or self.tp.startswith("vector")


class FuncVariant(object):
    def __init__(self, name, decl, isconstructor):
        self.name = name
        self.isconstructor = isconstructor
        self.rettype = decl[1]
        if self.rettype == "void":
            self.rettype = ""
        self.args = []
        self.array_counters = {}
        for a in decl[3]:
            ainfo = ArgInfo(a)
            if ainfo.isarray and not ainfo.arraycvt:
                c = ainfo.arraylen
                c_arrlist = self.array_counters.get(c, [])
                if c_arrlist:
                    c_arrlist.append(ainfo.name)
                else:
                    self.array_counters[c] = [ainfo.name]
            self.args.append(ainfo)
        self.init_pyproto()
        
    def init_pyproto(self):
        # string representation of argument list, with '[', ']' symbols denoting optional arguments, e.g.
        # "src1, src2[, dst[, mask]]" for cv.add
        argstr = ""
        
        # list of all input arguments of the Python function, with the argument numbers:
        #    [("src1", 0), ("src2", 1), ("dst", 2), ("mask", 3)]
        # we keep an argument number to find the respective argument quickly, because
        # some of the arguments of C function may not present in the Python function (such as array counters)
        # or even go in a different order ("heavy" output parameters of the C function
        # become the first optional input parameters of the Python function, and thus they are placed right after
        # non-optional input parameters) 
        arglist = []
        
        # the list of "heavy" output parameters. Heavy parameters are the parameters
        # that can be expensive to allocate each time, such as vectors and matrices (see isbig).
        outarr_list = []
        
        # the list of output parameters. Also includes input/output parameters.
        outlist = []
        
        firstoptarg = 0
        argno = -1
        for a in self.args:
            argno += 1
            if a.name in self.array_counters:
                continue
            if a.outputarg:
                outlist.append((a.name, argno))
            if not a.inputarg:
                if a.isbig():
                    outarr_list.append((a.name, argno))
                continue
            if not a.defval:
                arglist.append((a.name, argno))
                firstoptarg = argno+1
            else:
                # if there are some array output parameters before the first default parameter, they
                # are added as optional parameters before the first optional parameter
                if outarr_list:
                    arglist += outarr_list
                    outarr_list = []
                arglist.append((a.name, argno))
                
        if outarr_list:
            arglist += outarr_list
        noptargs = len(arglist) - firstoptarg
        argnamelist = [aname for aname, argno in arglist]
        argstr = ", ".join(argnamelist[:firstoptarg])
        argstr = "[, ".join([argstr] + argnamelist[firstoptarg:])
        argstr += "]" * noptargs
        if self.rettype:
            outlist = [("retval", -1)] + outlist
        elif self.isconstructor:
            assert outlist == []
            outlist = [("self", -1)]
        if outlist:
            outstr = ", ".join([o[0] for o in outlist])
        elif self.isconstructor:
            outstr = self.classname + " object"
        else:
            outstr = "None"
        self.py_docstring = "%s(%s) -> %s" % (self.name, argstr, outstr)
        self.py_noptargs = noptargs
        self.py_arglist = arglist
        for aname, argno in arglist:
            self.args[argno].py_inputarg = True
        for aname, argno in outlist:
            if argno >= 0:
                self.args[argno].py_outputarg = True
        self.py_outlist = outlist


class FuncInfo(object):
    def __init__(self, classname, name, cname, isconstructor):
        self.classname = classname
        self.name = name
        self.cname = cname
        self.isconstructor = isconstructor
        self.variants = []
    
    def add_variant(self, decl):
        self.variants.append(FuncVariant(self.name, decl, self.isconstructor))
        
    def get_wrapper_name(self):
        if self.classname:
            cn = self.classname + "_"
        else:
            cn = ""
        return "pyopencv_" + cn + self.name
    
    def get_wrapper_prototype(self):
        full_fname = self.get_wrapper_name()
        if self.classname and not self.isconstructor:
            self_arg = "self"
        else:
            self_arg = ""
        return "static PyObject* %s(PyObject* %s, PyObject* args, PyObject* kw)" % (full_fname, self_arg)
    
    def get_tab_entry(self):
        docstring_list = []
        have_empty_constructor = False
        for v in self.variants:
            s = v.py_docstring
            if (not v.py_arglist) and self.isconstructor:
                have_empty_constructor = True
            if s not in docstring_list:
                docstring_list.append(s)
        # if there are just 2 constructors: default one and some other,
        # we simplify the notation.
        # Instead of ClassName(args ...) -> object or ClassName() -> object
        # we write ClassName([args ...]) -> object
        if have_empty_constructor and len(self.variants) == 2:
            idx = self.variants[1].arglist != []
            docstring_list = ["[" + self.variants[idx].py_docstring + "]"]
                        
        return Template('    {"$py_funcname", (PyCFunction)$wrap_funcname, METH_KEYWORDS, "$py_docstring"},\n'
                        ).substitute(py_funcname = self.name, wrap_funcname=self.get_wrapper_name(),
                                     py_docstring = "  or  ".join(docstring_list))
        
    def gen_code(self, all_classes):
        proto = self.get_wrapper_prototype()
        code = "%s\n{\n" % (proto,)

        selfinfo = ClassInfo("")
        ismethod = self.classname != "" and not self.isconstructor
        # full name is needed for error diagnostic in PyArg_ParseTupleAndKeywords
        fullname = self.name

        if self.classname:
            selfinfo = all_classes[self.classname]
            if not self.isconstructor:
                code += gen_template_check_self.substitute(name=selfinfo.name, cname=selfinfo.cname)
                fullname = selfinfo.wname + "." + fullname

        all_code_variants = []
        declno = -1
        for v in self.variants:
            code_decl = ""
            code_fcall = ""
            code_ret = ""
            code_cvt_list = []

            if self.isconstructor:
                code_decl += "    pyopencv_%s_t* self = 0;\n" % selfinfo.name
                code_fcall = gen_template_call_constructor.substitute(name=selfinfo.name, cname=selfinfo.cname)
            else:
                code_fcall = "ERRWRAP2( "
                if v.rettype:
                    code_decl += "    " + v.rettype + " retval;\n"
                    code_fcall += "retval = "
                if ismethod:
                    code_fcall += "_self_->" + self.cname
                else:
                    code_fcall += self.cname
            code_fcall += "("
            all_cargs = []
            parse_arglist = []

            # declare all the C function arguments,
            # add necessary conversions from Python objects to code_cvt_list,
            # form the function/method call,
            # for the list of type mappings
            for a in v.args:
                tp1 = tp = a.tp
                amp = ""
                defval0 = ""
                if tp.endswith("*"):
                    tp = tp1 = tp[:-1]
                    amp = "&"
                    if tp.endswith("*"):
                        defval0 = "0"
                        tp1 = tp.replace("*", "_ptr")
                if tp1.endswith("*"):
                    print "Error: type with star: a.tp=%s, tp=%s, tp1=%s" % (a.tp, tp, tp1)
                    sys.exit(-1)

                amapping = simple_argtype_mapping.get(tp, (tp, "O", "pyopencv_from_" + tp1, defval0))
                all_cargs.append(amapping)
                if a.py_inputarg:
                    if amapping[1] == "O":
                        code_decl += "    PyObject* pyobj_%s = NULL;\n" % (a.name,)
                        parse_arglist.append("pyobj_" + a.name)
                        code_cvt_list.append("pyopencv_to_%s(pyobj_%s, %s)" % (tp1, a.name, a.name))
                    else:
                        parse_arglist.append(a.name)

                defval = a.defval
                if not defval:
                    defval = amapping[3]
                # "tp arg = tp();" is equivalent to "tp arg;" in the case of complex types
                if defval == tp + "()" and amapping[1] == "O":
                    defval = ""
                if defval:
                    code_decl += "    %s %s=%s;\n" % (amapping[0], a.name, defval)
                else:
                    code_decl += "    %s %s;\n" % (amapping[0], a.name)

                if not code_fcall.endswith("("):
                    code_fcall += ", "
                code_fcall += amp + a.name

            code_fcall += "))"

            if code_cvt_list:
                code_cvt_list = [""] + code_cvt_list

            # add info about return value, if any, to all_cargs. if there non-void return value,
            # it is encoded in v.py_outlist as ("retval", -1) pair.
            # As [-1] in Python accesses the last element of a list, we automatically handle the return value by
            # adding the necessary info to the end of all_cargs list.
            if v.rettype:
                tp = v.rettype
                tp1 = tp.replace("*", "_ptr")
                amapping = simple_argtype_mapping.get(tp, (tp, "O", "pyopencv_from_" + tp1, "0"))
                all_cargs.append(amapping)

            if v.args:
                # form the format spec for PyArg_ParseTupleAndKeywords
                fmtspec = "".join([all_cargs[argno][1] for aname, argno in v.py_arglist])
                if v.py_noptargs > 0:
                    fmtspec = fmtspec[:-v.py_noptargs] + "|" + fmtspec[-v.py_noptargs:]
                fmtspec += ":" + fullname

                # form the argument parse code that:
                #   - declares the list of keyword parameters
                #   - calls PyArg_ParseTupleAndKeywords
                #   - converts complex arguments from PyObject's to native OpenCV types
                code_parse = gen_template_parse_args.substitute(
                    kw_list = ", ".join(['"' + aname + '"' for aname, argno in v.py_arglist]),
                    fmtspec = fmtspec,
                    parse_arglist = ", ".join(["&" + aname for aname in parse_arglist]),
                    code_cvt = " &&\n        ".join(code_cvt_list))
            else:
                code_parse = "if(PyObject_Size(args) == 0 && PyObject_Size(kw) == 0)"

            if len(v.py_outlist) == 0:
                code_retval = "Py_RETURN_NONE"
            elif len(v.py_outlist) == 1:
                if self.isconstructor:
                    code_retval = "self"
                else:
                    aname, argno = v.py_outlist[0]
                    code_retval = "%s(%s)" % (all_cargs[argno][2], aname)
            else:
                # ther is more than 1 return parameter; form the tuple out of them
                fmtspec = "N"*len(v.py_outlist)
                backcvt_arg_list = []
                for aname, argno in v.py_outlist:
                    amapping = all_cargs[argno]
                    backcvt_arg_list.append("%s(%s)" % (amapping[2], aname))
                code_retval = "Py_BuildTuple(\"(%s)\", %s)" % \
                    (fmtspec, ", ".join([all_cargs[argno][2] + "(" + aname + ")" for aname, argno in v.py_outlist]))                    

            all_code_variants.append(gen_template_func_body.substitute(code_decl=code_decl,
                code_parse=code_parse, code_fcall=code_fcall, code_retval=code_retval))

        if len(all_code_variants)==1:
            # if the function/method has only 1 signature, then just put it
            code += all_code_variants[0]
        else:
            # try to execute each signature
            code += "    PyErr_Clear();\n\n".join(["    {\n" + v + "    }\n" for v in all_code_variants])
        code += "\n    return NULL;\n}\n\n"
        return code    
  
    
class PythonWrapperGenerator(object):
    def __init__(self):
        self.clear()
        
    def clear(self):
        self.classes = {}
        self.funcs = {}
        self.consts = {}
        self.code_types = cStringIO.StringIO()
        self.code_funcs = cStringIO.StringIO()
        self.code_functab = cStringIO.StringIO()
        self.code_type_reg = cStringIO.StringIO()
        self.code_const_reg = cStringIO.StringIO()

    def add_class(self, stype, name, decl):
        classinfo = ClassInfo(name, decl)
        
        if self.classes.has_key(classinfo.name):
            print "Generator error: class %s (cname=%s) already exists" \
                % (classinfo.name, classinfo.cname)
            sys.exit(-1) 
        self.classes[classinfo.name] = classinfo
        
    def add_const(self, name, decl):
        constinfo = ConstInfo(name, decl[1])
        
        if self.consts.has_key(constinfo.name):
            print "Generator error: constant %s (cname=%s) already exists" \
                % (constinfo.name, constinfo.cname)
            sys.exit(-1) 
        self.consts[constinfo.name] = constinfo

    def add_func(self, decl):
        classname = ""
        name = decl[0]
        dpos = name.rfind(".")
        if dpos >= 0 and name[:dpos] != "cv":
            classname = re.sub(r"^cv\.", "", name[:dpos])
            name = name[dpos+1:]
        cname = name
        name = re.sub(r"^cv\.", "", name)
        isconstructor = cname == classname
        cname = cname.replace(".", "::")
        isclassmethod = False
        customname = False
        for m in decl[2]:
            if m == "/S":
                isclassmethod = True
            elif m.startswith("="):
                name = m[1:]
                customname = True
        func_map = self.funcs
        
        if not classname or isconstructor:
            pass
        elif isclassmethod:
            if not customname:
                name = classname + "_" + name
            cname = classname + "::" + cname
            classname = ""
        else:
            classinfo = self.classes.get(classname, ClassInfo(""))
            if not classinfo.name:
                print "Generator error: the class for method %s is missing" % (name,)
                sys.exit(-1)
            func_map = classinfo.methods
            
        func = func_map.get(name, FuncInfo(classname, name, cname, isconstructor))
        func.add_variant(decl)
        if len(func.variants) == 1:
            func_map[name] = func
    
    def gen_const_reg(self, constinfo):
        self.code_const_reg.write("PUBLISH2(%s,%s);\n" % (constinfo.name, constinfo.cname))
    
    def save(self, path, name, buf):
        f = open(path + "/" + name, "wt")
        f.write(buf.getvalue())
        f.close()
            
    def gen(self, api_list, output_path):
        self.clear()
        
        # step 1: scan the list of declarations and build more descriptive maps of classes, consts, functions
        for decl in api_list:
            name = decl[0]
            if name.startswith("struct") or name.startswith("class"):
                # class/struct
                p = name.find(" ")
                stype = name[:p]
                name = name[p+1:].strip()
                self.add_class(stype, name, decl)
            elif name.startswith("const"):
                # constant
                self.add_const(name.replace("const ", "").strip(), decl)
            else:
                # function
                self.add_func(decl)
        
        # step 2: generate code for the classes and their methods
        classlist = self.classes.items()
        classlist.sort()
        for name, classinfo in classlist:
            code = classinfo.gen_code(self.classes)
            self.code_types.write(code)
            if not classinfo.ismap:
                self.code_type_reg.write("MKTYPE2(%s);\n" % (classinfo.name,) )
            
        # step 3: generate the code for all the global functions
        funclist = self.funcs.items()
        funclist.sort()
        for name, func in funclist:
            code = func.gen_code(self.classes)
            self.code_funcs.write(code)
            
        # step 4: generate the code for constants
        constlist = self.consts.items()
        constlist.sort()
        for name, constinfo in constlist:
            self.gen_const_reg(constinfo)
            
        # That's it. Now save all the files
        self.save(output_path, "pyopencv_generated_funcs.h", self.code_funcs)
        self.save(output_path, "pyopencv_generated_func_tab.h", self.code_functab)
        self.save(output_path, "pyopencv_generated_const_reg.h", self.code_const_reg)
        self.save(output_path, "pyopencv_generated_types.h", self.code_types)
        self.save(output_path, "pyopencv_generated_type_reg.h", self.code_type_reg)

def generate_all():
    decls = opencv_parser.parse_all()
    generator = PythonWrapperGenerator()
    generator.gen(decls, "/Users/vp/tmp")
    
if __name__ == "__main__":
    generate_all()

    
    
