#!/usr/bin/env python

from __future__ import print_function
import hdr_parser, sys, re
from string import Template
from collections import namedtuple
from itertools import chain

from typing_stubs_generator import TypingStubsGenerator

if sys.version_info[0] >= 3:
    from io import StringIO

else:
    from cStringIO import StringIO

if sys.version_info >= (3, 6):
    from typing_stubs_generation import SymbolName
else:
    SymbolName = namedtuple('SymbolName', ('namespaces', 'classes', 'name'))

    def parse_symbol_name(cls, full_symbol_name, known_namespaces):
        chunks = full_symbol_name.split('.')
        namespaces, name = chunks[:-1], chunks[-1]
        classes = []
        while len(namespaces) > 0 and '.'.join(namespaces) not in known_namespaces:
            classes.insert(0, namespaces.pop())
        return cls(tuple(namespaces), tuple(classes), name)

    setattr(SymbolName, "parse", classmethod(parse_symbol_name))


forbidden_arg_types = ["void*"]

ignored_arg_types = ["RNG*"]

pass_by_val_types = ["Point*", "Point2f*", "Rect*", "String*", "double*", "float*", "int*"]

gen_template_check_self = Template("""
    ${cname} * self1 = 0;
    if (!pyopencv_${name}_getp(self, self1))
        return failmsgp("Incorrect type of self (must be '${name}' or its derivative)");
    ${pname} _self_ = ${cvt}(self1);
""")
gen_template_call_constructor_prelude = Template("""new (&(self->v)) Ptr<$cname>(); // init Ptr with placement new
        if(self) """)

gen_template_call_constructor = Template("""self->v.reset(new ${cname}${py_args})""")

gen_template_simple_call_constructor_prelude = Template("""if(self) """)

gen_template_simple_call_constructor = Template("""new (&(self->v)) ${cname}${py_args}""")

gen_template_parse_args = Template("""const char* keywords[] = { $kw_list, NULL };
    if( PyArg_ParseTupleAndKeywords(py_args, kw, "$fmtspec", (char**)keywords, $parse_arglist)$code_cvt )""")

gen_template_func_body = Template("""$code_decl
    $code_parse
    {
        ${code_prelude}ERRWRAP2($code_fcall);
        $code_ret;
    }
""")

gen_template_mappable = Template("""
    {
        ${mappable} _src;
        if (pyopencv_to_safe(src, _src, info))
        {
            return cv_mappable_to(_src, dst);
        }
    }
""")

gen_template_type_decl = Template("""
// Converter (${name})

template<>
struct PyOpenCV_Converter< ${cname} >
{
    static PyObject* from(const ${cname}& r)
    {
        return pyopencv_${name}_Instance(r);
    }
    static bool to(PyObject* src, ${cname}& dst, const ArgInfo& info)
    {
        if(!src || src == Py_None)
            return true;
        ${cname} * dst_;
        if (pyopencv_${name}_getp(src, dst_))
        {
            dst = *dst_;
            return true;
        }
        ${mappable_code}
        failmsg("Expected ${cname} for argument '%s'", info.name);
        return false;
    }
};

""")

gen_template_map_type_cvt = Template("""
template<> bool pyopencv_to(PyObject* src, ${cname}& dst, const ArgInfo& info);

""")

gen_template_set_prop_from_map = Template("""
    if( PyMapping_HasKeyString(src, (char*)"$propname") )
    {
        tmp = PyMapping_GetItemString(src, (char*)"$propname");
        ok = tmp && pyopencv_to_safe(tmp, dst.$propname, ArgInfo("$propname", 0));
        Py_DECREF(tmp);
        if(!ok) return false;
    }""")

gen_template_type_impl = Template("""
// GetSet (${name})

${getset_code}

// Methods (${name})

${methods_code}

// Tables (${name})

static PyGetSetDef pyopencv_${name}_getseters[] =
{${getset_inits}
    {NULL}  /* Sentinel */
};

static PyMethodDef pyopencv_${name}_methods[] =
{
${methods_inits}
    {NULL,          NULL}
};
""")


gen_template_get_prop = Template("""
static PyObject* pyopencv_${name}_get_${member}(pyopencv_${name}_t* p, void *closure)
{
    return pyopencv_from(p->v${access}${member});
}
""")

gen_template_get_prop_algo = Template("""
static PyObject* pyopencv_${name}_get_${member}(pyopencv_${name}_t* p, void *closure)
{
    $cname* _self_ = dynamic_cast<$cname*>(p->v.get());
    if (!_self_)
        return failmsgp("Incorrect type of object (must be '${name}' or its derivative)");
    return pyopencv_from(_self_${access}${member});
}
""")

gen_template_set_prop = Template("""
static int pyopencv_${name}_set_${member}(pyopencv_${name}_t* p, PyObject *value, void *closure)
{
    if (!value)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the ${member} attribute");
        return -1;
    }
    return pyopencv_to_safe(value, p->v${access}${member}, ArgInfo("value", 0)) ? 0 : -1;
}
""")

gen_template_set_prop_algo = Template("""
static int pyopencv_${name}_set_${member}(pyopencv_${name}_t* p, PyObject *value, void *closure)
{
    if (!value)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the ${member} attribute");
        return -1;
    }
    $cname* _self_ = dynamic_cast<$cname*>(p->v.get());
    if (!_self_)
    {
        failmsgp("Incorrect type of object (must be '${name}' or its derivative)");
        return -1;
    }
    return pyopencv_to_safe(value, _self_${access}${member}, ArgInfo("value", 0)) ? 0 : -1;
}
""")


gen_template_prop_init = Template("""
    {(char*)"${export_member_name}", (getter)pyopencv_${name}_get_${member}, NULL, (char*)"${export_member_name}", NULL},""")

gen_template_rw_prop_init = Template("""
    {(char*)"${export_member_name}", (getter)pyopencv_${name}_get_${member}, (setter)pyopencv_${name}_set_${member}, (char*)"${export_member_name}", NULL},""")

gen_template_overloaded_function_call = Template("""
    {
${variant}

        pyPopulateArgumentConversionErrors();
    }
""")


class FormatStrings:
    string = 's'
    unsigned_char = 'b'
    short_int = 'h'
    int = 'i'
    unsigned_int = 'I'
    long = 'l'
    unsigned_long = 'k'
    long_long = 'L'
    unsigned_long_long = 'K'
    size_t = 'n'
    float = 'f'
    double = 'd'
    object = 'O'


ArgTypeInfo = namedtuple('ArgTypeInfo',
                         ['atype', 'format_str', 'default_value', 'strict_conversion'])
# strict_conversion is False by default
ArgTypeInfo.__new__.__defaults__ = (False,)

simple_argtype_mapping = {
    "bool": ArgTypeInfo("bool", FormatStrings.unsigned_char, "0", True),
    "size_t": ArgTypeInfo("size_t", FormatStrings.unsigned_long_long, "0", True),
    "int": ArgTypeInfo("int", FormatStrings.int, "0", True),
    "float": ArgTypeInfo("float", FormatStrings.float, "0.f", True),
    "double": ArgTypeInfo("double", FormatStrings.double, "0", True),
    "c_string": ArgTypeInfo("char*", FormatStrings.string, '(char*)""'),
    "string": ArgTypeInfo("std::string", FormatStrings.object, None, True),
    "Stream": ArgTypeInfo("Stream", FormatStrings.object, 'Stream::Null()', True),
    "cuda_Stream": ArgTypeInfo("cuda::Stream", FormatStrings.object, "cuda::Stream::Null()", True),
    "cuda_GpuMat": ArgTypeInfo("cuda::GpuMat", FormatStrings.object, "cuda::GpuMat()", True),
    "UMat": ArgTypeInfo("UMat", FormatStrings.object, 'UMat()', True),  # FIXIT: switch to CV_EXPORTS_W_SIMPLE as UMat is already a some kind of smart pointer
    "Ptr<streambuf>": ArgTypeInfo("Ptr<std::streambuf>", FormatStrings.object, None, True),
}

# Set of reserved keywords for Python. Can be acquired via the following call
# $ python -c "help('keywords')"
# Keywords that are reserved in C/C++ are excluded because they can not be
# used as variables identifiers
python_reserved_keywords = {
    "True", "None", "False", "as", "assert", "def", "del", "elif", "except", "exec",
    "finally", "from", "global",  "import", "in", "is", "lambda", "nonlocal",
    "pass", "print", "raise", "with", "yield"
}


def normalize_class_name(name):
    return re.sub(r"^cv\.", "", name).replace(".", "_")


def get_type_format_string(arg_type_info):
    if arg_type_info.strict_conversion:
        return FormatStrings.object
    else:
        return arg_type_info.format_str


class ClassProp(object):
    def __init__(self, decl):
        self.tp = decl[0].replace("*", "_ptr")
        self.name = decl[1]
        self.default_value = decl[2]
        self.readonly = True
        if "/RW" in decl[3]:
            self.readonly = False

    @property
    def export_name(self):
        if self.name in python_reserved_keywords:
            return self.name + "_"
        return self.name


class ClassInfo(object):
    def __init__(self, name, decl=None, codegen=None):
        # Scope name can be a module or other class e.g. cv::SimpleBlobDetector::Params
        self.original_scope_name, self.original_name = name.rsplit(".", 1)

        # In case scope refer the outer class exported with different name
        if codegen:
            self.export_scope_name = codegen.get_export_scope_name(
                self.original_scope_name
            )
        else:
            self.export_scope_name = self.original_scope_name
        self.export_scope_name = re.sub(r"^cv\.?", "", self.export_scope_name)

        self.export_name = self.original_name

        self.class_id = normalize_class_name(name)

        self.cname = name.replace(".", "::")
        self.ismap = False
        self.is_parameters = False
        self.issimple = False
        self.isalgorithm = False
        self.methods = {}
        self.props = []
        self.mappables = []
        self.consts = {}
        self.base = None
        self.constructor = None

        if decl:
            bases = decl[1].split()[1:]
            if len(bases) > 1:
                print("Note: Class %s has more than 1 base class (not supported by Python C extensions)" % (self.cname,))
                print("      Bases: ", " ".join(bases))
                print("      Only the first base class will be used")
                #return sys.exit(-1)
            elif len(bases) == 1:
                self.base = bases[0].strip(",")
                if self.base.startswith("cv::"):
                    self.base = self.base[4:]
                if self.base == "Algorithm":
                    self.isalgorithm = True
                self.base = self.base.replace("::", "_")

            for m in decl[2]:
                if m.startswith("="):
                    # Aliasing only affects the exported class name, not class identifier
                    self.export_name = m[1:]
                elif m == "/Map":
                    self.ismap = True
                elif m == "/Simple":
                    self.issimple = True
                elif m == "/Params":
                    self.is_parameters = True
                    self.issimple = True
            self.props = [ClassProp(p) for p in decl[3]]

        if not self.has_export_alias and self.original_name.startswith("Cv"):
            self.export_name = self.export_name[2:]

    @property
    def wname(self):
        if len(self.export_scope_name) > 0:
            return self.export_scope_name.replace(".", "_") + "_" + self.export_name

        return self.export_name

    @property
    def name(self):
        return self.class_id

    @property
    def full_export_scope_name(self):
        return "cv." + self.export_scope_name if len(self.export_scope_name) else "cv"

    @property
    def full_export_name(self):
        return self.full_export_scope_name + "." + self.export_name

    @property
    def full_original_name(self):
        return self.original_scope_name + "." + self.original_name

    @property
    def has_export_alias(self):
        return self.export_name != self.original_name

    def gen_map_code(self, codegen):
        all_classes = codegen.classes
        code = "static bool pyopencv_to(PyObject* src, %s& dst, const ArgInfo& info)\n{\n    PyObject* tmp;\n    bool ok;\n" % (self.cname)
        code += "".join([gen_template_set_prop_from_map.substitute(propname=p.name,proptype=p.tp) for p in self.props])
        if self.base:
            code += "\n    return pyopencv_to_safe(src, (%s&)dst, info);\n}\n" % all_classes[self.base].cname
        else:
            code += "\n    return true;\n}\n"
        return code

    def gen_code(self, codegen):
        all_classes = codegen.classes
        if self.ismap:
            return self.gen_map_code(codegen)

        getset_code = StringIO()
        getset_inits = StringIO()

        sorted_props = [(p.name, p) for p in self.props]
        sorted_props.sort()

        access_op = "->"
        if self.issimple:
            access_op = "."

        for pname, p in sorted_props:
            if self.isalgorithm:
                getset_code.write(gen_template_get_prop_algo.substitute(name=self.name, cname=self.cname, member=pname, membertype=p.tp, access=access_op))
            else:
                getset_code.write(gen_template_get_prop.substitute(name=self.name, member=pname, membertype=p.tp, access=access_op))
            if p.readonly:
                getset_inits.write(gen_template_prop_init.substitute(name=self.name, member=pname, export_member_name=p.export_name))
            else:
                if self.isalgorithm:
                    getset_code.write(gen_template_set_prop_algo.substitute(name=self.name, cname=self.cname, member=pname, membertype=p.tp, access=access_op))
                else:
                    getset_code.write(gen_template_set_prop.substitute(name=self.name, member=pname, membertype=p.tp, access=access_op))
                getset_inits.write(gen_template_rw_prop_init.substitute(name=self.name, member=pname, export_member_name=p.export_name))

        methods_code = StringIO()
        methods_inits = StringIO()

        sorted_methods = list(self.methods.items())
        sorted_methods.sort()

        if self.constructor is not None:
            methods_code.write(self.constructor.gen_code(codegen))

        for mname, m in sorted_methods:
            methods_code.write(m.gen_code(codegen))
            methods_inits.write(m.get_tab_entry())

        code = gen_template_type_impl.substitute(name=self.name,
                                                 getset_code=getset_code.getvalue(),
                                                 getset_inits=getset_inits.getvalue(),
                                                 methods_code=methods_code.getvalue(),
                                                 methods_inits=methods_inits.getvalue())

        return code

    def gen_def(self, codegen):
        all_classes = codegen.classes
        baseptr = "NoBase"
        if self.base and self.base in all_classes:
            baseptr = all_classes[self.base].name

        constructor_name = "0"
        if self.constructor is not None:
            constructor_name = self.constructor.get_wrapper_name()

        return 'CVPY_TYPE({}, {}, {}, {}, {}, {}, "{}")\n'.format(
            self.export_name,
            self.class_id,
            self.cname if self.issimple else "Ptr<{}>".format(self.cname),
            self.original_name if self.issimple else "Ptr",
            baseptr,
            constructor_name,
            # Leading dot is required to provide correct class naming
            "." + self.export_scope_name if len(self.export_scope_name) > 0 else self.export_scope_name
        )


def handle_ptr(tp):
    if tp.startswith('Ptr_'):
        tp = 'Ptr<' + "::".join(tp.split('_')[1:]) + '>'
    return tp


class ArgInfo(object):
    def __init__(self, atype, name, default_value, modifiers=(),
                 enclosing_arg=None):
        # type: (ArgInfo, str, str, str, tuple[str, ...], ArgInfo | None) -> None
        self.tp = handle_ptr(atype)
        self.name = name
        self.defval = default_value
        self._modifiers = tuple(modifiers)
        self.isarray = False
        self.is_smart_ptr = self.tp.startswith('Ptr<')  # FIXIT: handle through modifiers - need to modify parser
        self.arraylen = 0
        self.arraycvt = None
        for m in self._modifiers:
            if m.startswith("/A"):
                self.isarray = True
                self.arraylen = m[2:].strip()
            elif m.startswith("/CA"):
                self.isarray = True
                self.arraycvt = m[2:].strip()
        self.py_inputarg = False
        self.py_outputarg = False
        self.enclosing_arg = enclosing_arg

    def __str__(self):
        return 'ArgInfo("{}", tp="{}", default="{}", in={}, out={})'.format(
            self.name, self.tp, self.defval, self.inputarg,
            self.outputarg
        )

    def __repr__(self):
        return str(self)

    @property
    def export_name(self):
        if self.name in python_reserved_keywords:
            return self.name + '_'
        return self.name

    @property
    def nd_mat(self):
        return '/ND' in self._modifiers

    @property
    def inputarg(self):
        return '/O' not in self._modifiers

    @property
    def arithm_op_src_arg(self):
        return '/AOS' in self._modifiers

    @property
    def outputarg(self):
        return '/O' in self._modifiers or '/IO' in self._modifiers

    @property
    def pathlike(self):
        return '/PATH' in self._modifiers

    @property
    def returnarg(self):
        return self.outputarg

    @property
    def isrvalueref(self):
        return '/RRef' in self._modifiers

    @property
    def full_name(self):
        if self.enclosing_arg is None:
            return self.name
        return self.enclosing_arg.name + '.' + self.name

    def isbig(self):
        return self.tp in ["Mat", "vector_Mat",
                           "cuda::GpuMat", "cuda_GpuMat", "GpuMat",
                           "vector_GpuMat", "vector_cuda_GpuMat",
                           "UMat", "vector_UMat"] # or self.tp.startswith("vector")

    def crepr(self):
        arg  = 0x01 if self.outputarg else 0x0
        arg += 0x02 if self.arithm_op_src_arg else 0x0
        arg += 0x04 if self.pathlike else 0x0
        arg += 0x08 if self.nd_mat else 0x0
        return "ArgInfo(\"%s\", %d)" % (self.name, arg)


def find_argument_class_info(argument_type, function_namespace,
                             function_class_name, known_classes):
    # type: (str, str, str, dict[str, ClassInfo]) -> ClassInfo | None
    """Tries to find corresponding class info for the provided argument type

    Args:
        argument_type (str): Function argument type
        function_namespace (str): Namespace of the function declaration
        function_class_name (str): Name of the class if function is a method of class
        known_classes (dict[str, ClassInfo]): Mapping between string class
            identifier and ClassInfo struct.

    Returns:
        Optional[ClassInfo]: class info struct if the provided argument type
            refers to a known C++ class, None otherwise.
    """

    possible_classes = tuple(filter(lambda cls: cls.endswith(argument_type), known_classes))
    # If argument type is not a known class - just skip it
    if not possible_classes:
        return None
    if len(possible_classes) == 1:
        return known_classes[possible_classes[0]]

    # If there is more than 1 matched class, try to select the most probable one
    # Look for a matched class name in different scope, starting from the
    # narrowest one

    # First try to find argument inside class scope of the function (if any)
    if function_class_name:
        type_to_match = function_class_name + '_' + argument_type
        if type_to_match in possible_classes:
            return known_classes[type_to_match]
    else:
        type_to_match = argument_type

    # Trying to find argument type in the namespace of the function
    type_to_match = '{}_{}'.format(
        function_namespace.lstrip('cv.').replace('.', '_'), type_to_match
    )
    if type_to_match in possible_classes:
        return known_classes[type_to_match]

    # Try to find argument name as is
    if argument_type in possible_classes:
        return known_classes[argument_type]

    # NOTE: parser is broken - some classes might not be visible, depending on
    # the order of parsed headers.
    # print("[WARNING] Can't select an appropriate class for argument: '",
    #       argument_type, "'. Possible matches: '", possible_classes, "'")
    return None


class FuncVariant(object):
    def __init__(self, namespace, classname, name, decl, isconstructor, known_classes, isphantom=False):
        self.name = self.wname = name
        self.isconstructor = isconstructor
        self.isphantom = isphantom

        self.docstring = decl[5]

        self.rettype = decl[4] or handle_ptr(decl[1])
        if self.rettype == "void":
            self.rettype = ""
        self.args = []
        self.array_counters = {}
        for arg_decl in decl[3]:
            assert len(arg_decl) == 4, \
                'ArgInfo contract is violated. Arg declaration should contain:' \
                '"arg_type", "name", "default_value", "modifiers". '\
                'Got tuple: {}'.format(arg_decl)

            ainfo = ArgInfo(atype=arg_decl[0], name=arg_decl[1],
                            default_value=arg_decl[2], modifiers=arg_decl[3])
            if ainfo.isarray and not ainfo.arraycvt:
                c = ainfo.arraylen
                c_arrlist = self.array_counters.get(c, [])
                if c_arrlist:
                    c_arrlist.append(ainfo.name)
                else:
                    self.array_counters[c] = [ainfo.name]
            self.args.append(ainfo)
        self.init_pyproto(namespace, classname, known_classes)

    def is_arg_optional(self, py_arg_index):
        # type: (FuncVariant, int) -> bool
        return py_arg_index >= len(self.py_arglist) - self.py_noptargs

    def init_pyproto(self, namespace, classname, known_classes):
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

        # Check if there is params structure in arguments
        arguments = []
        for arg in self.args:
            arg_class_info = find_argument_class_info(
                arg.tp, namespace, classname, known_classes
            )
            # If argument refers to the 'named arguments' structure - instead of
            # the argument put its properties
            if arg_class_info is not None and arg_class_info.is_parameters:
                for prop in arg_class_info.props:
                    # Convert property to ArgIfno and mark that argument is
                    # a part of the parameters structure:
                    arguments.append(
                        ArgInfo(prop.tp, prop.name, prop.default_value,
                                enclosing_arg=arg)
                    )
            else:
                arguments.append(arg)
        # Prevent names duplication after named arguments are merged
        # to the main arguments list
        argument_names = tuple(arg.name for arg in arguments)
        assert len(set(argument_names)) == len(argument_names), \
            "Duplicate arguments with names '{}' in function '{}'. "\
            "Please, check named arguments used in function interface".format(
                argument_names, self.name
            )

        self.args = arguments

        for argno, a in enumerate(self.args):
            if a.name in self.array_counters:
                continue
            assert a.tp not in forbidden_arg_types, \
                'Forbidden type "{}" for argument "{}" in "{}" ("{}")'.format(
                    a.tp, a.name, self.name, self.classname
                )

            if a.tp in ignored_arg_types:
                continue
            if a.returnarg:
                outlist.append((a.name, argno))
            if (not a.inputarg) and a.isbig():
                outarr_list.append((a.name, argno))
                continue
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
        argnamelist = [self.args[argno].export_name for _, argno in arglist]
        argstr = ", ".join(argnamelist[:firstoptarg])
        argstr = "[, ".join([argstr] + argnamelist[firstoptarg:])
        argstr += "]" * noptargs
        if self.rettype:
            outlist = [("retval", -1)] + outlist
        elif self.isconstructor:
            assert outlist == []
            outlist = [("self", -1)]
        if self.isconstructor:
            if classname.startswith("Cv"):
                classname = classname[2:]
            outstr = "<%s object>" % (classname,)
        elif outlist:
            outstr = ", ".join([o[0] for o in outlist])
        else:
            outstr = "None"

        self.py_arg_str = argstr
        self.py_return_str = outstr
        self.py_prototype = "%s(%s) -> %s" % (self.wname, argstr, outstr)
        self.py_noptargs = noptargs
        self.py_arglist = arglist
        for _, argno in arglist:
            self.args[argno].py_inputarg = True
        for _, argno in outlist:
            if argno >= 0:
                self.args[argno].py_outputarg = True
        self.py_outlist = outlist


class FuncInfo(object):
    def __init__(self, classname, name, cname, isconstructor, namespace, is_static):
        self.classname = classname
        self.name = name
        self.cname = cname
        self.isconstructor = isconstructor
        self.namespace = namespace
        self.is_static = is_static
        self.variants = []

    def add_variant(self, decl, known_classes, isphantom=False):
        self.variants.append(
            FuncVariant(self.namespace, self.classname, self.name, decl,
                        self.isconstructor, known_classes, isphantom)
        )

    def get_wrapper_name(self):
        name = self.name
        if self.classname:
            classname = self.classname + "_"
            if "[" in name:
                name = "getelem"
        else:
            classname = ""

        if self.is_static:
            name += "_static"

        return "pyopencv_" + self.namespace.replace('.','_') + '_' + classname + name

    def get_wrapper_prototype(self, codegen):
        full_fname = self.get_wrapper_name()
        if self.isconstructor:
            return "static int {fn_name}(pyopencv_{type_name}_t* self, PyObject* py_args, PyObject* kw)".format(
                    fn_name=full_fname, type_name=codegen.classes[self.classname].name)

        if self.classname:
            self_arg = "self"
        else:
            self_arg = ""
        return "static PyObject* %s(PyObject* %s, PyObject* py_args, PyObject* kw)" % (full_fname, self_arg)

    def get_tab_entry(self):
        prototype_list = []
        docstring_list = []

        have_empty_constructor = False
        for v in self.variants:
            s = v.py_prototype
            if (not v.py_arglist) and self.isconstructor:
                have_empty_constructor = True
            if s not in prototype_list:
                prototype_list.append(s)
                docstring_list.append(v.docstring)

        # if there are just 2 constructors: default one and some other,
        # we simplify the notation.
        # Instead of ClassName(args ...) -> object or ClassName() -> object
        # we write ClassName([args ...]) -> object
        if have_empty_constructor and len(self.variants) == 2:
            idx = self.variants[1].py_arglist != []
            s = self.variants[idx].py_prototype
            p1 = s.find("(")
            p2 = s.rfind(")")
            prototype_list = [s[:p1+1] + "[" + s[p1+1:p2] + "]" + s[p2:]]

        # The final docstring will be: Each prototype, followed by
        # their relevant doxygen comment
        full_docstring = ""
        for prototype, body in zip(prototype_list, docstring_list):
            full_docstring += Template("$prototype\n$docstring\n\n\n\n").substitute(
                prototype=prototype,
                docstring='\n'.join(
                    ['.   ' + line
                     for line in body.split('\n')]
                )
            )

        # Escape backslashes, newlines, and double quotes
        full_docstring = full_docstring.strip().replace("\\", "\\\\").replace('\n', '\\n').replace("\"", "\\\"")
        # Convert unicode chars to xml representation, but keep as string instead of bytes
        full_docstring = full_docstring.encode('ascii', errors='xmlcharrefreplace').decode()

        return Template('    {"$py_funcname", CV_PY_FN_WITH_KW_($wrap_funcname, $flags), "$py_docstring"},\n'
                        ).substitute(py_funcname = self.variants[0].wname, wrap_funcname=self.get_wrapper_name(),
                                     flags = 'METH_STATIC' if self.is_static else '0', py_docstring = full_docstring)

    def gen_code(self, codegen):
        all_classes = codegen.classes
        proto = self.get_wrapper_prototype(codegen)
        code = "%s\n{\n" % (proto,)
        code += "    using namespace %s;\n\n" % self.namespace.replace('.', '::')

        selfinfo = None
        ismethod = self.classname != "" and not self.isconstructor
        # full name is needed for error diagnostic in PyArg_ParseTupleAndKeywords
        fullname = self.name

        if self.classname:
            selfinfo = all_classes[self.classname]
            if not self.isconstructor:
                if not self.is_static:
                    code += gen_template_check_self.substitute(
                        name=selfinfo.name,
                        cname=selfinfo.cname if selfinfo.issimple else "Ptr<{}>".format(selfinfo.cname),
                        pname=(selfinfo.cname + '*') if selfinfo.issimple else "Ptr<{}>".format(selfinfo.cname),
                        cvt='' if selfinfo.issimple else '*'
                    )
                fullname = selfinfo.wname + "." + fullname

        all_code_variants = []

        # See https://github.com/opencv/opencv/issues/25928
        # Conversion to UMat is expensive more than conversion to Mat.
        # To reduce this cost, conversion to Mat is prefer than to UMat.
        variants = []
        variants_umat = []
        for v in self.variants:
            hasUMat = False
            for a in v.args:
                hasUMat = hasUMat or "UMat" in a.tp
            if hasUMat :
                variants_umat.append(v)
            else:
                variants.append(v)
        variants.extend(variants_umat)

        for v in variants:
            code_decl = ""
            code_ret = ""
            code_cvt_list = []

            code_args = "("
            all_cargs = []

            if v.isphantom and ismethod and not self.is_static:
                code_args += "_self_"

            # declare all the C function arguments,
            # add necessary conversions from Python objects to code_cvt_list,
            # form the function/method call,
            # for the list of type mappings
            instantiated_args = set()
            for a in v.args:
                if a.tp in ignored_arg_types:
                    defval = a.defval
                    if not defval and a.tp.endswith("*"):
                        defval = "0"
                    assert defval
                    if not code_args.endswith("("):
                        code_args += ", "
                    code_args += defval
                    all_cargs.append([[None, ""], ""])
                    continue
                tp1 = tp = a.tp
                amp = ""
                defval0 = ""
                if tp in pass_by_val_types:
                    tp = tp1 = tp[:-1]
                    amp = "&"
                    if tp.endswith("*"):
                        defval0 = "0"
                        tp1 = tp.replace("*", "_ptr")
                tp_candidates = [a.tp, normalize_class_name(self.namespace + "." + a.tp)]
                if any(tp in codegen.enums.keys() for tp in tp_candidates):
                    defval0 = "static_cast<%s>(%d)" % (a.tp, 0)

                if tp in simple_argtype_mapping:
                    arg_type_info = simple_argtype_mapping[tp]
                else:
                    if tp in all_classes:
                        tp_classinfo = all_classes[tp]
                        cname_of_value = tp_classinfo.cname if tp_classinfo.issimple else "Ptr<{}>".format(tp_classinfo.cname)
                        arg_type_info = ArgTypeInfo(cname_of_value, FormatStrings.object, defval0, True)
                        assert not (a.is_smart_ptr and tp_classinfo.issimple), "Can't pass 'simple' type as Ptr<>"
                        if not a.is_smart_ptr and not tp_classinfo.issimple:
                            assert amp == ''
                            amp = '*'
                    else:
                        # FIXIT: Ptr_ / vector_ / enums / nested types
                        arg_type_info = ArgTypeInfo(tp, FormatStrings.object, defval0, True)

                parse_name = a.name
                if a.py_inputarg and arg_type_info.strict_conversion:
                    parse_name = "pyobj_" + a.full_name.replace('.', '_')
                    code_decl += "    PyObject* %s = NULL;\n" % (parse_name,)
                    if a.tp == 'char':
                        code_cvt_list.append("convert_to_char(%s, &%s, %s)" % (parse_name, a.full_name, a.crepr()))
                    else:
                        code_cvt_list.append("pyopencv_to_safe(%s, %s, %s)" % (parse_name, a.full_name, a.crepr()))

                all_cargs.append([arg_type_info, parse_name])

                # Argument is actually a part of the named arguments structure,
                # but it is possible to mimic further processing like it is normal arg
                if a.enclosing_arg:
                    a = a.enclosing_arg
                    arg_type_info = ArgTypeInfo(a.tp, FormatStrings.object,
                                                default_value=a.defval,
                                                strict_conversion=True)
                    # Skip further actions if enclosing argument is already instantiated
                    # by its another field
                    if a.name in instantiated_args:
                        continue
                    instantiated_args.add(a.name)

                defval = a.defval
                if not defval:
                    defval = arg_type_info.default_value
                else:
                    if "UMat" in tp:
                        if "Mat" in defval and "UMat" not in defval:
                            defval = defval.replace("Mat", "UMat")
                    if "cuda::GpuMat" in tp:
                        if "Mat" in defval and "GpuMat" not in defval:
                            defval = defval.replace("Mat", "cuda::GpuMat")
                # "tp arg = tp();" is equivalent to "tp arg;" in the case of complex types
                if defval == tp + "()" and arg_type_info.format_str == FormatStrings.object:
                    defval = ""
                if a.outputarg and not a.inputarg:
                    defval = ""
                if defval:
                    code_decl += "    %s %s=%s;\n" % (arg_type_info.atype, a.name, defval)
                else:
                    code_decl += "    %s %s;\n" % (arg_type_info.atype, a.name)

                if not code_args.endswith("("):
                    code_args += ", "

                if a.isrvalueref:
                    code_args += amp + 'std::move(' + a.name + ')'
                else:
                    code_args += amp + a.name

            code_args += ")"

            if self.isconstructor:
                if selfinfo.issimple:
                    templ_prelude = gen_template_simple_call_constructor_prelude
                    templ = gen_template_simple_call_constructor
                else:
                    templ_prelude = gen_template_call_constructor_prelude
                    templ = gen_template_call_constructor

                code_prelude = templ_prelude.substitute(name=selfinfo.name, cname=selfinfo.cname)
                code_fcall = templ.substitute(name=selfinfo.name, cname=selfinfo.cname, py_args=code_args)
                if v.isphantom:
                    code_fcall = code_fcall.replace("new " + selfinfo.cname, self.cname.replace("::", "_"))
            else:
                code_prelude = ""
                code_fcall = ""
                if v.rettype:
                    code_decl += "    " + v.rettype + " retval;\n"
                    code_fcall += "retval = "
                if not v.isphantom and ismethod and not self.is_static:
                    code_fcall += "_self_->" + self.cname
                else:
                    code_fcall += self.cname
                code_fcall += code_args

            if code_cvt_list:
                code_cvt_list = [""] + code_cvt_list

            # add info about return value, if any, to all_cargs. if there non-void return value,
            # it is encoded in v.py_outlist as ("retval", -1) pair.
            # As [-1] in Python accesses the last element of a list, we automatically handle the return value by
            # adding the necessary info to the end of all_cargs list.
            if v.rettype:
                tp = v.rettype
                tp1 = tp.replace("*", "_ptr")
                default_info = ArgTypeInfo(tp, FormatStrings.object, "0")
                arg_type_info = simple_argtype_mapping.get(tp, default_info)
                all_cargs.append(arg_type_info)

            if v.args and v.py_arglist:
                # form the format spec for PyArg_ParseTupleAndKeywords
                fmtspec = "".join([
                    get_type_format_string(all_cargs[argno][0])
                    for _, argno in v.py_arglist
                ])
                if v.py_noptargs > 0:
                    fmtspec = fmtspec[:-v.py_noptargs] + "|" + fmtspec[-v.py_noptargs:]
                fmtspec += ":" + fullname

                # form the argument parse code that:
                #   - declares the list of keyword parameters
                #   - calls PyArg_ParseTupleAndKeywords
                #   - converts complex arguments from PyObject's to native OpenCV types
                code_parse = gen_template_parse_args.substitute(
                    kw_list=", ".join(['"' + v.args[argno].export_name + '"' for _, argno in v.py_arglist]),
                    fmtspec=fmtspec,
                    parse_arglist=", ".join(["&" + all_cargs[argno][1] for _, argno in v.py_arglist]),
                    code_cvt=" &&\n        ".join(code_cvt_list))
            else:
                code_parse = "if(PyObject_Size(py_args) == 0 && (!kw || PyObject_Size(kw) == 0))"

            if len(v.py_outlist) == 0:
                code_ret = "Py_RETURN_NONE"
            elif len(v.py_outlist) == 1:
                if self.isconstructor:
                    code_ret = "return 0"
                else:
                    aname, argno = v.py_outlist[0]
                    code_ret = "return pyopencv_from(%s)" % (aname,)
            else:
                # there is more than 1 return parameter; form the tuple out of them
                fmtspec = "N"*len(v.py_outlist)
                code_ret = "return Py_BuildValue(\"(%s)\", %s)" % \
                    (fmtspec, ", ".join(["pyopencv_from(" + aname + ")" for aname, argno in v.py_outlist]))

            all_code_variants.append(gen_template_func_body.substitute(code_decl=code_decl,
                code_parse=code_parse, code_prelude=code_prelude, code_fcall=code_fcall, code_ret=code_ret))

        if len(all_code_variants)==1:
            # if the function/method has only 1 signature, then just put it
            code += all_code_variants[0]
        else:
            # try to execute each signature, add an interlude between function
            # calls to collect error from all conversions
            code += '    pyPrepareArgumentConversionErrorsStorage({});\n'.format(len(all_code_variants))
            code += '    \n'.join(gen_template_overloaded_function_call.substitute(variant=v)
                                  for v in all_code_variants)
            code += '    pyRaiseCVOverloadException("{}");\n'.format(self.name)

        def_ret = "NULL"
        if self.isconstructor:
            def_ret = "-1"
        code += "\n    return %s;\n}\n\n" % def_ret

        cname = self.cname
        classinfo = None
        #dump = False
        #if dump: pprint(vars(self))
        #if dump: pprint(vars(self.variants[0]))
        if self.classname:
            classinfo = all_classes[self.classname]
            #if dump: pprint(vars(classinfo))
            if self.isconstructor:
                py_name = classinfo.full_export_name
            else:
                py_name = classinfo.full_export_name + "." + self.variants[0].wname

            if not self.is_static and not self.isconstructor:
                cname = classinfo.cname + '::' + cname
        else:
            py_name = '.'.join([self.namespace, self.variants[0].wname])
        #if dump: print(cname + " => " + py_name)
        py_signatures = codegen.py_signatures.setdefault(cname, [])
        for v in self.variants:
            s = dict(name=py_name, arg=v.py_arg_str, ret=v.py_return_str)
            for old in py_signatures:
                if s == old:
                    break
            else:
                py_signatures.append(s)

        return code


class Namespace(object):
    def __init__(self):
        self.funcs = {}
        self.consts = {}


class PythonWrapperGenerator(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.classes = {}
        self.namespaces = {}
        self.consts = {}
        self.enums = {}
        self.typing_stubs_generator = TypingStubsGenerator()
        self.code_include = StringIO()
        self.code_enums = StringIO()
        self.code_types = StringIO()
        self.code_funcs = StringIO()
        self.code_ns_reg = StringIO()
        self.code_ns_init = StringIO()
        self.code_type_publish = StringIO()
        self.py_signatures = dict()
        self.class_idx = 0

    def add_class(self, stype, name, decl):
        classinfo = ClassInfo(name, decl, self)
        classinfo.decl_idx = self.class_idx
        self.class_idx += 1

        if classinfo.name in self.classes:
            print("Generator error: class %s (cname=%s) already exists" \
                % (classinfo.name, classinfo.cname))
            sys.exit(-1)
        self.classes[classinfo.name] = classinfo

        namespace, _, _ = self.split_decl_name(name)
        namespace = '.'.join(namespace)
        # Registering a namespace if it is not already handled or
        # doesn't have anything except classes defined in it
        self.namespaces.setdefault(namespace, Namespace())

        # Add Class to json file.
        py_name = classinfo.full_export_name  # use wrapper name
        py_signatures = self.py_signatures.setdefault(classinfo.cname, [])
        py_signatures.append(dict(name=py_name))
        #print('class: ' + classinfo.cname + " => " + py_name)

    def get_export_scope_name(self, original_scope_name):
        # Outer classes should be registered before their content - inner classes in this case
        class_scope = self.classes.get(normalize_class_name(original_scope_name), None)

        if class_scope:
            return class_scope.full_export_name

        # Otherwise it is a namespace.
        # If something is messed up at this point - it will be revelead during
        # library import
        return original_scope_name

    def split_decl_name(self, name):
        return SymbolName.parse(name, self.parser.namespaces)

    def add_const(self, name, decl):
        cname = name.replace('.','::')
        namespace, classes, name = self.split_decl_name(name)
        namespace = '.'.join(namespace)
        name = '_'.join(chain(classes, (name, )))
        ns = self.namespaces.setdefault(namespace, Namespace())
        if name in ns.consts:
            print("Generator error: constant %s (cname=%s) already exists" \
                % (name, cname))
            sys.exit(-1)
        ns.consts[name] = cname
        value = decl[1]
        py_name = '.'.join([namespace, name])
        py_signatures = self.py_signatures.setdefault(cname, [])
        py_signatures.append(dict(name=py_name, value=value))
        #print(cname + ' => ' + str(py_name) + ' (value=' + value + ')')

    def add_enum(self, name, decl):
        enumeration_name = SymbolName.parse(name, self.parser.namespaces)
        is_scoped_enum = decl[0].startswith("enum class") \
            or decl[0].startswith("enum struct")

        wname = normalize_class_name(name)
        if wname.endswith("<unnamed>"):
            wname = None
        else:
            self.enums[wname] = name
        const_decls = decl[3]
        enum_entries = {}
        for decl in const_decls:
            enum_entries[decl[0].split(".")[-1]] = decl[1]

            self.add_const(decl[0].replace("const ", "").strip(), decl)

        # Extra enumerations tracking is required to generate stubs for
        # all enumerations, including <unnamed> once, otherwise they
        # will be forgiven
        self.typing_stubs_generator.add_enum(enumeration_name, is_scoped_enum,
                                             enum_entries)

    def add_func(self, decl):
        namespace, classes, barename = self.split_decl_name(decl[0])
        cname = "::".join(chain(namespace, classes, (barename, )))
        name = barename
        classname = ''
        bareclassname = ''
        if classes:
            classname = normalize_class_name('.'.join(namespace+classes))
            bareclassname = classes[-1]
        namespace_str = '.'.join(namespace)

        isconstructor = name == bareclassname
        is_static = False
        isphantom = False
        mappable = None
        for m in decl[2]:
            if m == "/S":
                is_static = True
            elif m == "/phantom":
                isphantom = True
                cname = cname.replace("::", "_")
            elif m.startswith("="):
                name = m[1:]
            elif m.startswith("/mappable="):
                mappable = m[10:]
                self.classes[classname].mappables.append(mappable)
                return

        if isconstructor:
            name = "_".join(chain(classes[:-1], (name, )))

        if is_static:
            # Add it as a method to the class
            func_map = self.classes[classname].methods
            func = func_map.setdefault(name, FuncInfo(classname, name, cname, isconstructor, namespace_str, is_static))
            func.add_variant(decl, self.classes, isphantom)

            # Add it as global function
            g_name = "_".join(chain(classes, (name, )))
            w_classes = []
            for i in range(0, len(classes)):
                classes_i = classes[:i+1]
                classname_i = normalize_class_name('.'.join(namespace+classes_i))
                w_classname = self.classes[classname_i].wname
                namespace_prefix = normalize_class_name('.'.join(namespace)) + '_'
                if w_classname.startswith(namespace_prefix):
                    w_classname = w_classname[len(namespace_prefix):]
                w_classes.append(w_classname)
            g_wname = "_".join(w_classes+[name])
            func_map = self.namespaces.setdefault(namespace_str, Namespace()).funcs
            # Static functions should be called using class names, not like
            # module-level functions, so first step is to remove them from
            # type hints.
            self.typing_stubs_generator.add_ignored_function_name(g_name)
            # Exports static function with internal name (backward compatibility)
            func = func_map.setdefault(g_name, FuncInfo("", g_name, cname, isconstructor, namespace_str, False))
            func.add_variant(decl, self.classes, isphantom)
            if g_wname != g_name:  # TODO OpenCV 5.0
                self.typing_stubs_generator.add_ignored_function_name(g_wname)
                wfunc = func_map.setdefault(g_wname, FuncInfo("", g_wname, cname, isconstructor, namespace_str, False))
                wfunc.add_variant(decl, self.classes, isphantom)
        else:
            if classname and not isconstructor:
                if not isphantom:
                    cname = barename
                func_map = self.classes[classname].methods
            else:
                func_map = self.namespaces.setdefault(namespace_str, Namespace()).funcs

            func = func_map.setdefault(name, FuncInfo(classname, name, cname, isconstructor, namespace_str, is_static))
            func.add_variant(decl, self.classes, isphantom)

        if classname and isconstructor:
            self.classes[classname].constructor = func

    def gen_namespace(self, ns_name):
        ns = self.namespaces[ns_name]
        wname = normalize_class_name(ns_name)

        self.code_ns_reg.write('static PyMethodDef methods_%s[] = {\n'%wname)
        for name, func in sorted(ns.funcs.items()):
            if func.isconstructor:
                continue
            self.code_ns_reg.write(func.get_tab_entry())
        custom_entries_macro = 'PYOPENCV_EXTRA_METHODS_{}'.format(wname.upper())
        self.code_ns_reg.write('#ifdef {}\n    {}\n#endif\n'.format(custom_entries_macro, custom_entries_macro))
        self.code_ns_reg.write('    {NULL, NULL}\n};\n\n')

        self.code_ns_reg.write('static ConstDef consts_%s[] = {\n'%wname)
        for name, cname in sorted(ns.consts.items()):
            self.code_ns_reg.write('    {"%s", static_cast<long>(%s)},\n'%(name, cname))
            compat_name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name).upper()
            if name != compat_name:
                self.code_ns_reg.write('    {"%s", static_cast<long>(%s)},\n'%(compat_name, cname))
        custom_entries_macro = 'PYOPENCV_EXTRA_CONSTANTS_{}'.format(wname.upper())
        self.code_ns_reg.write('#ifdef {}\n    {}\n#endif\n'.format(custom_entries_macro, custom_entries_macro))
        self.code_ns_reg.write('    {NULL, 0}\n};\n\n')

    def gen_enum_reg(self, enum_name):
        name_seg = enum_name.split(".")
        is_enum_class = False
        if len(name_seg) >= 2 and name_seg[-1] == name_seg[-2]:
            enum_name = ".".join(name_seg[:-1])
            is_enum_class = True

        wname = normalize_class_name(enum_name)
        cname = enum_name.replace(".", "::")

        code = ""
        if re.sub(r"^cv\.", "", enum_name) != wname:
            code += "typedef {0} {1};\n".format(cname, wname)
        code += "CV_PY_FROM_ENUM({0})\nCV_PY_TO_ENUM({0})\n\n".format(wname)
        self.code_enums.write(code)

    def save(self, path, name, buf):
        with open(path + "/" + name, "wt") as f:
            f.write(buf.getvalue())

    def save_json(self, path, name, value):
        import json
        with open(path + "/" + name, "wt") as f:
            json.dump(value, f)

    def gen(self, srcfiles, output_path):
        self.clear()
        self.parser = hdr_parser.CppHeaderParser(generate_umat_decls=True, generate_gpumat_decls=True)


        # step 1: scan the headers and build more descriptive maps of classes, consts, functions
        for hdr in srcfiles:
            decls = self.parser.parse(hdr)
            if len(decls) == 0:
                continue

            if hdr.find('misc/python/shadow_') < 0:  # Avoid including the "shadow_" files
                if hdr.find('opencv2/') >= 0:
                    # put relative path
                    self.code_include.write('#include "{0}"\n'.format(hdr[hdr.rindex('opencv2/'):]))
                else:
                    self.code_include.write('#include "{0}"\n'.format(hdr))

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
                elif name.startswith("enum"):
                    # enum
                    self.add_enum(name.rsplit(" ", 1)[1], decl)
                else:
                    # function
                    self.add_func(decl)

        # step 1.5 check if all base classes exist
        for name, classinfo in self.classes.items():
            if classinfo.base:
                chunks = classinfo.base.split('_')
                base = '_'.join(chunks)
                while base not in self.classes and len(chunks)>1:
                    del chunks[-2]
                    base = '_'.join(chunks)
                if base not in self.classes:
                    print("Generator error: unable to resolve base %s for %s"
                        % (classinfo.base, classinfo.name))
                    sys.exit(-1)
                base_instance = self.classes[base]
                classinfo.base = base
                classinfo.isalgorithm |= base_instance.isalgorithm  # wrong processing of 'isalgorithm' flag:
                                                                    # doesn't work for trees(graphs) with depth > 2
                self.classes[name] = classinfo

        # tree-based propagation of 'isalgorithm'
        processed = dict()
        def process_isalgorithm(classinfo):
            if classinfo.isalgorithm or classinfo in processed:
                return classinfo.isalgorithm
            res = False
            if classinfo.base:
                res = process_isalgorithm(self.classes[classinfo.base])
                #assert not (res == True or classinfo.isalgorithm is False), "Internal error: " + classinfo.name + " => " + classinfo.base
                classinfo.isalgorithm |= res
                res = classinfo.isalgorithm
            processed[classinfo] = True
            return res
        for name, classinfo in self.classes.items():
            process_isalgorithm(classinfo)

        # step 2: generate code for the classes and their methods
        classlist = list(self.classes.items())
        classlist.sort()
        for name, classinfo in classlist:
            self.code_types.write("//{}\n".format(80*"="))
            self.code_types.write("// {} ({})\n".format(name, 'Map' if classinfo.ismap else 'Generic'))
            self.code_types.write("//{}\n".format(80*"="))
            self.code_types.write(classinfo.gen_code(self))
            if classinfo.ismap:
                self.code_types.write(gen_template_map_type_cvt.substitute(name=classinfo.name, cname=classinfo.cname))
            else:
                mappable_code = "\n".join([
                                      gen_template_mappable.substitute(cname=classinfo.cname, mappable=mappable)
                                          for mappable in classinfo.mappables])
                code = gen_template_type_decl.substitute(
                    name=classinfo.name,
                    cname=classinfo.cname if classinfo.issimple else "Ptr<{}>".format(classinfo.cname),
                    mappable_code=mappable_code
                )
                self.code_types.write(code)

        # register classes in the same order as they have been declared.
        # this way, base classes will be registered in Python before their derivatives.
        classlist1 = [(classinfo.decl_idx, name, classinfo) for name, classinfo in classlist]
        classlist1.sort()

        published_types = set()  # ensure toposort with base classes
        for decl_idx, name, classinfo in classlist1:
            if classinfo.ismap:
                continue

            def _registerType(classinfo):
                if classinfo.decl_idx in published_types:
                    #print(classinfo.decl_idx, classinfo.name, ' - already published')
                    # If class already registered it means that there is
                    # a correponding node in the AST. This check is partically
                    # useful for base classes.
                    return self.typing_stubs_generator.find_class_node(
                        classinfo, self.parser.namespaces
                    )
                published_types.add(classinfo.decl_idx)

                # Registering a class means creation of the AST node from the
                # given class information
                class_node = self.typing_stubs_generator.create_class_node(
                    classinfo, self.parser.namespaces
                )

                if classinfo.base and classinfo.base in self.classes:
                    base_classinfo = self.classes[classinfo.base]
                    # print(classinfo.decl_idx, classinfo.name, ' - request publishing of base type ', base_classinfo.decl_idx, base_classinfo.name)
                    base_node = _registerType(base_classinfo)
                    class_node.add_base(base_node)

                # print(classinfo.decl_idx, classinfo.name, ' - published!')
                self.code_type_publish.write(classinfo.gen_def(self))
                return class_node

            _registerType(classinfo)

        # step 3: generate the code for all the global functions
        for ns_name, ns in sorted(self.namespaces.items()):
            if ns_name.split('.')[0] != 'cv':
                continue
            for name, func in sorted(ns.funcs.items()):
                if func.isconstructor:
                    continue
                code = func.gen_code(self)
                self.code_funcs.write(code)
                # If function is not ignored - create an AST node for it
                if name not in self.typing_stubs_generator.type_hints_ignored_functions:
                    self.typing_stubs_generator.create_function_node(func)

            self.gen_namespace(ns_name)
            self.code_ns_init.write('CVPY_MODULE("{}", {});\n'.format(ns_name[2:], normalize_class_name(ns_name)))

        # step 4: generate the code for enum types
        enumlist = list(self.enums.values())
        enumlist.sort()
        for name in enumlist:
            self.gen_enum_reg(name)

        # step 5: generate the code for constants
        constlist = list(self.consts.items())
        constlist.sort()
        for name, constinfo in constlist:
            self.gen_const_reg(constinfo)

        # All symbols are collected and AST is reconstructed, generating
        # typing stubs...
        self.typing_stubs_generator.generate(output_path)

        # That's it. Now save all the files
        self.save(output_path, "pyopencv_generated_include.h", self.code_include)
        self.save(output_path, "pyopencv_generated_funcs.h", self.code_funcs)
        self.save(output_path, "pyopencv_generated_enums.h", self.code_enums)
        self.save(output_path, "pyopencv_generated_types.h", self.code_type_publish)
        self.save(output_path, "pyopencv_generated_types_content.h", self.code_types)
        self.save(output_path, "pyopencv_generated_modules.h", self.code_ns_init)
        self.save(output_path, "pyopencv_generated_modules_content.h", self.code_ns_reg)
        self.save_json(output_path, "pyopencv_signatures.json", self.py_signatures)


if __name__ == "__main__":
    srcfiles = hdr_parser.opencv_hdr_list
    dstdir = "/Users/vp/tmp"
    if len(sys.argv) > 1:
        dstdir = sys.argv[1]
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'r') as f:
            srcfiles = [l.strip() for l in f.readlines()]
    generator = PythonWrapperGenerator()
    generator.gen(srcfiles, dstdir)
