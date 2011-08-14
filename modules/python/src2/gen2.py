import hdr_parser, sys, re, os, cStringIO
from string import Template

gen_template_check_self = Template("""    if(!PyObject_TypeCheck(self, &pyopencv_${name}_Type))
        return failmsgp("Incorrect type of self (must be '${name}' or its derivative)");
    $cname* _self_ = ${amp}((pyopencv_${name}_t*)self)->v;
""")

gen_template_call_constructor = Template("""self = PyObject_NEW(pyopencv_${name}_t, &pyopencv_${name}_Type);
        if(self) ERRWRAP2(self->v = $op$cname""")

gen_template_parse_args = Template("""const char* keywords[] = { $kw_list, NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "$fmtspec", (char**)keywords, $parse_arglist)$code_cvt )""")

gen_template_func_body = Template("""$code_decl
    $code_parse
    {
        $code_fcall;
        $code_ret;
    }
""")

gen_template_simple_type_decl = Template("""
struct pyopencv_${name}_t
{
    PyObject_HEAD
    ${cname} v;
};

static PyTypeObject pyopencv_${name}_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
    0,
    MODULESTR".$wname",
    sizeof(pyopencv_${name}_t),
};

static void pyopencv_${name}_dealloc(PyObject* self)
{
    PyObject_Del(self);
}

static PyObject* pyopencv_from(const ${cname}& r)
{
    pyopencv_${name}_t *m = PyObject_NEW(pyopencv_${name}_t, &pyopencv_${name}_Type);
    m->v = r;
    return (PyObject*)m;
}

static bool pyopencv_to(PyObject* src, ${cname}& dst, const char* name="<unknown>")
{
    if( src == NULL || src == Py_None )
        return true;
    if(!PyObject_TypeCheck(src, &pyopencv_${name}_Type))
    {
        failmsg("Expected ${cname} for argument '%s'", name);
        return false;
    }
    dst = ((pyopencv_${name}_t*)src)->v;
    return true;
}
""")


gen_template_type_decl = Template("""
struct pyopencv_${name}_t
{
    PyObject_HEAD
    ${cname}* v;
};

static PyTypeObject pyopencv_${name}_Type =
{
    PyObject_HEAD_INIT(&PyType_Type)
    0,
    MODULESTR".$wname",
    sizeof(pyopencv_${name}_t),
};

static void pyopencv_${name}_dealloc(PyObject* self)
{
    delete ((pyopencv_${name}_t*)self)->v;
    PyObject_Del(self);
}
""")

gen_template_map_type_cvt = Template("""
static bool pyopencv_to(PyObject* src, ${cname}& dst, const char* name="<unknown>");
""")

gen_template_set_prop_from_map = Template("""
    if( PyMapping_HasKeyString(src, (char*)"$propname") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"$propname");
        ok = tmp && pyopencv_to(tmp, dst.$propname);
        Py_DECREF(tmp);
        if(!ok) return false;
    }""")

gen_template_type_impl = Template("""
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

static void pyopencv_${name}_specials(void)
{
    pyopencv_${name}_Type.tp_base = ${baseptr};
    pyopencv_${name}_Type.tp_dealloc = pyopencv_${name}_dealloc;
    pyopencv_${name}_Type.tp_repr = pyopencv_${name}_repr;
    pyopencv_${name}_Type.tp_getset = pyopencv_${name}_getseters;
    pyopencv_${name}_Type.tp_methods = pyopencv_${name}_methods;${extra_specials}
}
""")


gen_template_get_prop = Template("""
static PyObject* pyopencv_${name}_get_${member}(pyopencv_${name}_t* p, void *closure)
{
    return pyopencv_from(p->v${access}${member});
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
    return pyopencv_to(value, p->v${access}${member}) ? 0 : -1;
}
""")

gen_template_prop_init = Template("""
    {(char*)"${member}", (getter)pyopencv_${name}_get_${member}, NULL, (char*)"${member}", NULL},""")

gen_template_rw_prop_init = Template("""
    {(char*)"${member}", (getter)pyopencv_${name}_get_${member}, (setter)pyopencv_${name}_set_${member}, (char*)"${member}", NULL},""")

simple_argtype_mapping = {
    "bool": ("bool", "b", "0"),
    "int": ("int", "i", "0"),
    "float": ("float", "f", "0.f"),
    "double": ("double", "d", "0"),
    "c_string": ("char*", "s", '(char*)""')
}

def normalize_class_name(name):
    return re.sub(r"^cv\.", "", name).replace(".", "_")

class ClassProp(object):
    def __init__(self, decl):
        self.tp = decl[0].replace("*", "_ptr")
        self.name = decl[1]
        self.readonly = True
        if "/RW" in decl[3]:
            self.readonly = False

class ClassInfo(object):
    def __init__(self, name, decl=None):
        self.cname = name.replace(".", "::")
        self.name = self.wname = normalize_class_name(name)
        self.ismap = False
        self.issimple = False
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
                elif m == "/Simple":
                    self.issimple = True
            self.props = [ClassProp(p) for p in decl[3]]
        
        if not customname and self.wname.startswith("Cv"):
            self.wname = self.wname[2:]
        
    def gen_map_code(self, all_classes):
        code = "static bool pyopencv_to(PyObject* src, %s& dst, const char* name)\n{\n    PyObject* tmp;\n    bool ok;\n" % (self.cname)
        code += "".join([gen_template_set_prop_from_map.substitute(propname=p.name,proptype=p.tp) for p in self.props])
        if self.bases:
            code += "\n    return pyopencv_to(src, (%s&)dst, name);\n}\n" % all_classes[self.bases[0]].cname
        else:
            code += "\n    return true;\n}\n"
        return code
        
    def gen_code(self, all_classes):
        if self.ismap:
            return self.gen_map_code(all_classes)
        
        getset_code = cStringIO.StringIO()
        getset_inits = cStringIO.StringIO()
        
        sorted_props = [(p.name, p) for p in self.props]
        sorted_props.sort()
        
        access_op = "->"
        if self.issimple:
            access_op = "."
        
        for pname, p in sorted_props:
            getset_code.write(gen_template_get_prop.substitute(name=self.name, member=pname, membertype=p.tp, access=access_op))
            if p.readonly:
                getset_inits.write(gen_template_prop_init.substitute(name=self.name, member=pname))
            else:
                getset_code.write(gen_template_set_prop.substitute(name=self.name, member=pname, membertype=p.tp, access=access_op))
                getset_inits.write(gen_template_rw_prop_init.substitute(name=self.name, member=pname))
                
        methods_code = cStringIO.StringIO()
        methods_inits = cStringIO.StringIO()
        
        sorted_methods = self.methods.items()
        sorted_methods.sort()
        
        for mname, m in sorted_methods:
            methods_code.write(m.gen_code(all_classes))
            methods_inits.write(m.get_tab_entry())
        
        baseptr = "NULL"
        if self.bases and all_classes.has_key(self.bases[0]):
            baseptr = "&pyopencv_" + all_classes[self.bases[0]].name + "_Type"
        
        code = gen_template_type_impl.substitute(name=self.name, wname=self.wname, cname=self.cname,
            getset_code=getset_code.getvalue(), getset_inits=getset_inits.getvalue(),
            methods_code=methods_code.getvalue(), methods_inits=methods_inits.getvalue(),
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
        self.returnarg = False
        for m in arg_tuple[3]:
            if m == "/O":
                self.inputarg = False
                self.outputarg = True
                self.returnarg = True
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
        return self.tp == "Mat" or self.tp == "vector_Mat"# or self.tp.startswith("vector")


class FuncVariant(object):
    def __init__(self, classname, name, decl, isconstructor):
        self.classname = classname
        self.name = self.wname = name
        self.isconstructor = isconstructor
        if self.isconstructor:
            if self.wname.startswith("Cv"):
                self.wname = self.wname[2:]
            else:
                self.wname = self.classname
            
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
        
        firstoptarg = 1000000
        argno = -1
        for a in self.args:
            argno += 1
            if a.name in self.array_counters:
                continue
            if a.returnarg:
                outlist.append((a.name, argno))
            if (not a.inputarg or a.returnarg) and a.isbig():
                outarr_list.append((a.name, argno))
            if not a.inputarg:
                continue
            if not a.defval:
                arglist.append((a.name, argno))
            else:
                firstoptarg = min(firstoptarg, len(arglist))
                # if there are some array output parameters before the first default parameter, they
                # are added as optional parameters before the first optional parameter
                if outarr_list:
                    arglist += outarr_list
                    outarr_list = []
                arglist.append((a.name, argno))
                
        if outarr_list:
            firstoptarg = min(firstoptarg, len(arglist))
            arglist += outarr_list
        firstoptarg = min(firstoptarg, len(arglist))
        
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
        if self.isconstructor:
            classname = self.classname
            if classname.startswith("Cv"):
                classname=classname[2:]
            outstr = "<%s object>" % (classname,)
        elif outlist:
            outstr = ", ".join([o[0] for o in outlist])
        else:
            outstr = "None"
            
        self.py_docstring = "%s(%s) -> %s" % (self.wname, argstr, outstr)
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
        self.variants.append(FuncVariant(self.classname, self.name, decl, self.isconstructor))
        
    def get_wrapper_name(self):
        name = self.name
        if self.classname:
            classname = self.classname + "_"
            if "[" in name:
                name = "getelem"
        else:
            classname = ""
        return "pyopencv_" + classname + name
    
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
            idx = self.variants[1].py_arglist != []
            s = self.variants[idx].py_docstring
            p1 = s.find("(")
            p2 = s.rfind(")")
            docstring_list = [s[:p1+1] + "[" + s[p1+1:p2] + "]" + s[p2:]]
            
        return Template('    {"$py_funcname", (PyCFunction)$wrap_funcname, METH_KEYWORDS, "$py_docstring"},\n'
                        ).substitute(py_funcname = self.variants[0].wname, wrap_funcname=self.get_wrapper_name(),
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
                amp = ""
                if selfinfo.issimple:
                    amp = "&"
                code += gen_template_check_self.substitute(name=selfinfo.name, cname=selfinfo.cname, amp=amp)
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
                op = "new "
                if selfinfo.issimple:
                    op = ""
                code_fcall = gen_template_call_constructor.substitute(name=selfinfo.name, cname=selfinfo.cname, op=op)
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

                amapping = simple_argtype_mapping.get(tp, (tp, "O", defval0))
                parse_name = a.name
                if a.py_inputarg:
                    if amapping[1] == "O":
                        code_decl += "    PyObject* pyobj_%s = NULL;\n" % (a.name,)
                        parse_name = "pyobj_" + a.name
                        code_cvt_list.append("pyopencv_to(pyobj_%s, %s)" % (a.name, a.name))
                
                all_cargs.append([amapping, parse_name])
                
                defval = a.defval
                if not defval:
                    defval = amapping[2]
                # "tp arg = tp();" is equivalent to "tp arg;" in the case of complex types
                if defval == tp + "()" and amapping[1] == "O":
                    defval = ""
                if a.outputarg and not a.inputarg:
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
                amapping = simple_argtype_mapping.get(tp, (tp, "O", "0"))
                all_cargs.append(amapping)

            if v.args and v.py_arglist:
                # form the format spec for PyArg_ParseTupleAndKeywords
                fmtspec = "".join([all_cargs[argno][0][1] for aname, argno in v.py_arglist])
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
                    parse_arglist = ", ".join(["&" + all_cargs[argno][1] for aname, argno in v.py_arglist]),
                    code_cvt = " &&\n        ".join(code_cvt_list))
            else:
                code_parse = "if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))"

            if len(v.py_outlist) == 0:
                code_ret = "Py_RETURN_NONE"
            elif len(v.py_outlist) == 1:
                if self.isconstructor:
                    code_ret = "return (PyObject*)self"
                else:
                    aname, argno = v.py_outlist[0]
                    code_ret = "return pyopencv_from(%s)" % (aname,)
            else:
                # ther is more than 1 return parameter; form the tuple out of them
                fmtspec = "N"*len(v.py_outlist)
                backcvt_arg_list = []
                for aname, argno in v.py_outlist:
                    amapping = all_cargs[argno][0]
                    backcvt_arg_list.append("%s(%s)" % (amapping[2], aname))
                code_ret = "return Py_BuildValue(\"(%s)\", %s)" % \
                    (fmtspec, ", ".join(["pyopencv_from(" + aname + ")" for aname, argno in v.py_outlist]))                    

            all_code_variants.append(gen_template_func_body.substitute(code_decl=code_decl,
                code_parse=code_parse, code_fcall=code_fcall, code_ret=code_ret))

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
        self.code_func_tab = cStringIO.StringIO()
        self.code_type_reg = cStringIO.StringIO()
        self.code_const_reg = cStringIO.StringIO()
        self.class_idx = 0

    def add_class(self, stype, name, decl):
        classinfo = ClassInfo(name, decl)
        classinfo.decl_idx = self.class_idx
        self.class_idx += 1
        
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
        classname = bareclassname = ""
        name = decl[0]
        dpos = name.rfind(".")
        if dpos >= 0 and name[:dpos] != "cv":
            classname = bareclassname = re.sub(r"^cv\.", "", name[:dpos])
            name = name[dpos+1:]
            dpos = classname.rfind(".")
            if dpos >= 0:
                bareclassname = classname[dpos+1:]
                classname = classname.replace(".", "_")
        cname = name
        name = re.sub(r"^cv\.", "", name)
        isconstructor = cname == bareclassname
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
            
    def gen(self, srcfiles, output_path):
        self.clear()
        parser = hdr_parser.CppHeaderParser()
        
        # step 1: scan the headers and build more descriptive maps of classes, consts, functions
        for hdr in srcfiles:
            decls = parser.parse(hdr)
            for decl in decls:
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
            if classinfo.ismap:
                self.code_types.write(gen_template_map_type_cvt.substitute(name=name, cname=classinfo.cname))
            else:
                if classinfo.issimple:
                    templ = gen_template_simple_type_decl
                else:
                    templ = gen_template_type_decl
                self.code_types.write(templ.substitute(name=name, wname=classinfo.wname, cname=classinfo.cname))
        
        # register classes in the same order as they have been declared.
        # this way, base classes will be registered in Python before their derivatives.
        classlist1 = [(classinfo.decl_idx, name, classinfo) for name, classinfo in classlist]
        classlist1.sort()
        
        for decl_idx, name, classinfo in classlist1:
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
            self.code_func_tab.write(func.get_tab_entry())
            
        # step 4: generate the code for constants
        constlist = self.consts.items()
        constlist.sort()
        for name, constinfo in constlist:
            self.gen_const_reg(constinfo)
            
        # That's it. Now save all the files
        self.save(output_path, "pyopencv_generated_funcs.h", self.code_funcs)
        self.save(output_path, "pyopencv_generated_func_tab.h", self.code_func_tab)
        self.save(output_path, "pyopencv_generated_const_reg.h", self.code_const_reg)
        self.save(output_path, "pyopencv_generated_types.h", self.code_types)
        self.save(output_path, "pyopencv_generated_type_reg.h", self.code_type_reg)

if __name__ == "__main__":
    srcfiles = hdr_parser.opencv_hdr_list
    dstdir = "/Users/vp/tmp"
    if len(sys.argv) > 1:
        dstdir = sys.argv[1]
    if len(sys.argv) > 2:
        srcfiles = sys.argv[2:]
    generator = PythonWrapperGenerator()
    generator.gen(srcfiles, dstdir)

    
    
