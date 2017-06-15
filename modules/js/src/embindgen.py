###############################################################################
# AUTHOR: Sajjad Taheri sajjadt[at]uci[at]edu
#
#                             LICENSE AGREEMENT
# Copyright (c) 2015, University of California, Irvine
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. All advertising materials mentioning features or use of this software
#    must display the following acknowledgement:
#    This product includes software developed by the UC Irvine.
# 4. Neither the name of the UC Irvine nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY UC IRVINE ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL UC IRVINE OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

from __future__ import print_function
import hdr_parser, sys, re
from templates import *

if sys.version_info[0] >= 3:
    from io import StringIO
else:
    from cStringIO import StringIO


func_table = {}

# Ignore these functions due to Embind limitations for now
ignore_list = ['locate',  #int&
               'minEnclosingCircle',  #float&
               'checkRange',
               'minMaxLoc',
               'floodFill',
               'phaseCorrelate',
               'randShuffle'
               ]

# Features to be exported
export_enums = True
with_wrapped_functions = True
with_default_params = with_wrapped_functions and False  # Emscripten::val is used to support default parameters

wrapper_namespace = "Wrappers"
type_dict = {
    'InputArray': 'const cv::Mat&',
    'OutputArray': 'cv::Mat&',
    'InputOutputArray': 'cv::Mat&',
    'InputArrayOfArrays': 'const std::vector<cv::Mat>&',
    'OutputArrayOfArrays': 'std::vector<cv::Mat>&',
    'String': 'std::string',
    'const String&':'const std::string&'
}

def normalize_class_name(name):
    return re.sub(r"^cv\.", "", name).replace(".", "_")


class ClassProp(object):
    def __init__(self, decl):
        self.tp = decl[0].replace("*", "_ptr").strip()
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
        self.isalgorithm = False
        self.methods = {}
        self.ext_constructors = {}
        self.props = []
        self.consts = {}
        customname = False
        self.jsfuncs = {}

        self.has_smart_ptr = False

        if decl:
            self.bases = decl[1].split()[1:]
            if len(self.bases) > 1:
                self.bases = [self.bases[0].strip(",")]
                # return sys.exit(-1)
            if self.bases and self.bases[0].startswith("cv::"):
                self.bases[0] = self.bases[0][4:]
            if self.bases and self.bases[0] == "Algorithm":
                self.isalgorithm = True
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


def handle_ptr(tp):
    if tp.startswith('Ptr_'):
        tp = 'Ptr<' + "::".join(tp.split('_')[1:]) + '>'
    return tp


class ArgInfo(object):
    def __init__(self, arg_tuple):
        self.tp = handle_ptr(arg_tuple[0]).strip()
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
        return self.tp == "Mat" or self.tp == "vector_Mat"  # or self.tp.startswith("vector")

    def crepr(self):
        return "ArgInfo(\"%s\", %d)" % (self.name, self.outputarg)



class FuncVariant(object):
    def __init__(self, class_name, name, decl, is_constructor, is_class_method, is_const, is_virtual, is_pure_virtual, ref_return, const_return):
        self.class_name = class_name
        self.name = self.wname = name
        self.is_constructor = is_constructor
        self.is_class_method = is_class_method
        self.is_const = is_const
        self.is_virtual = is_virtual
        self.is_pure_virtual = is_pure_virtual
        self.refret = ref_return
        self.constret = const_return
        self.rettype = handle_ptr(decl[1]).strip()
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


class FuncInfo(object):
    def __init__(self, class_name, name, cname, namespace, isconstructor):
        self.class_name = class_name
        self.name = name
        self.cname = cname
        self.namespace = namespace
        self.variants = []
        self.is_constructor = isconstructor

    def add_variant(self, variant):
        self.variants.append(variant)


class Namespace(object):
    def __init__(self):
        self.funcs = {}
        self.enums = {}


class JSWrapperGenerator(object):
    def __init__(self):

        self.bindings = []
        self.wrapper_funcs = []

        self.classes = {}
        self.namespaces = {}
        self.enums = {}

        self.parser = hdr_parser.CppHeaderParser()
        self.class_idx = 0

    def add_class(self, stype, name, decl):
        class_info = ClassInfo(name, decl)
        class_info.decl_idx = self.class_idx
        self.class_idx += 1

        if class_info.name in self.classes:
            print("Generator error: class %s (cpp_name=%s) already exists" \
                  % (class_info.name, class_info.cname))
            sys.exit(-1)
        self.classes[class_info.name] = class_info

        if class_info.bases:
            chunks = class_info.bases[0].split('::')
            base = '_'.join(chunks)
            while base not in self.classes and len(chunks) > 1:
                del chunks[-2]
                base = '_'.join(chunks)
            if base not in self.classes:
                print("Generator error: unable to resolve base %s for %s"
                      % (class_info.bases[0], class_info.name))
                sys.exit(-1)
            else:
                class_info.bases[0] = "::".join(chunks)
                class_info.isalgorithm |= self.classes[base].isalgorithm

    def split_decl_name(self, name):
        chunks = name.split('.')
        namespace = chunks[:-1]
        classes = []
        while namespace and '.'.join(namespace) not in self.parser.namespaces:
            classes.insert(0, namespace.pop())
        return namespace, classes, chunks[-1]

    def add_enum(self, decl):
        name = decl[1]
        namespace, classes, val = self.split_decl_name(name)
        namespace = '.'.join(namespace)
        val = '_'.join(classes + [name])
        cname = name.replace('.', '::')
        ns = self.namespaces.setdefault(namespace, Namespace())
        if name in ns.enums:
            print("Generator warning: constant %s (cname=%s) already exists" \
                  % (name, cname))
            # sys.exit(-1)
        else:
            ns.enums[name] = []
        for item in decl[3]:
            ns.enums[name].append(item)

    def add_func(self, decl):
        namespace, classes, barename = self.split_decl_name(decl[0])
        cpp_name = "::".join(namespace + classes + [barename])
        name = barename
        class_name = ''
        bare_class_name = ''
        if classes:
            class_name = normalize_class_name('.'.join(namespace + classes))
            bare_class_name = classes[-1]
        namespace = '.'.join(namespace)

        is_constructor = name == bare_class_name
        is_class_method = False
        is_const_method = False
        is_virtual_method = False
        is_pure_virtual_method = False
        const_return = False
        ref_return = False

        for m in decl[2]:
            if m == "/S":
                is_class_method = True
            elif m == "/C":
                is_const_method = True
            elif m == "/V":
                is_virtual_method = True
            elif m == "/PV":
                is_pure_virtual_method = True
            elif m == "/Ref":
                ref_return = True
            elif m == "/CRet":
                const_return = True
            elif m.startswith("="):
                name = m[1:]

        if class_name:
            cpp_name = barename
            func_map = self.classes[class_name].methods
        else:
            func_map = self.namespaces.setdefault(namespace, Namespace()).funcs

        func = func_map.setdefault(name, FuncInfo(class_name, name, cpp_name, namespace, is_constructor))

        variant = FuncVariant(class_name, name, decl, is_constructor, is_class_method, is_const_method,
                        is_virtual_method, is_pure_virtual_method, ref_return, const_return)
        func.add_variant(variant)

    def save(self, path, name, buf):
        f = open(path + "/" + name, "wt")
        f.write(buf.getvalue())
        f.close()

    def gen_function_binding_with_wrapper(self, func, class_info):

        binding_text = None
        wrapper_func_text = None

        bindings = []
        wrappers = []

        for index, variant in enumerate(func.variants):

            factory = False
            if class_info and 'Ptr<' in variant.rettype:

                factory = True
                base_class_name = variant.rettype
                base_class_name = base_class_name.replace("Ptr<","").replace(">","").strip()
                if base_class_name in self.classes:
                    self.classes[base_class_name].has_smart_ptr = True
                else:
                    print(base_class_name, ' not found in classes for registering smart pointer using ', class_info.name, 'instead')
                    self.classes[class_info.name].has_smart_ptr = True

            def_args = []
            has_def_param = False

            # Return type
            ret_type = 'void' if variant.rettype.strip() == '' else variant.rettype
            if ret_type.startswith('Ptr'): #smart pointer
                ptr_type = ret_type.replace('Ptr<', '').replace('>', '')
                if ptr_type in type_dict:
                    ret_type = type_dict[ptr_type]
            for key in type_dict:
                if key in ret_type:
                    ret_type = ret_type.replace(key, type_dict[key])

            arg_types = []
            unwrapped_arg_types = []
            for arg in variant.args:
                arg_type = None
                if arg.tp in type_dict:
                    arg_type = type_dict[arg.tp]
                else:
                    arg_type = arg.tp
                # Add default value
                if with_default_params and arg.defval != '':
                    def_args.append(arg.defval)
                    arg_types.append('emscripten::val')
                else:
                    arg_types.append(arg_type)
                unwrapped_arg_types.append(arg_type)

            # Function attribure
            func_attribs = ''
            if '*' in ''.join(arg_types):
                func_attribs += ', allow_raw_pointers()'

            if variant.is_pure_virtual:
                func_attribs += ', pure_virtual()'


            # Wrapper function
            wrap_func_name = (func.class_name+"_" if class_info != None else "") + func.cname.split("::")[-1] + "_wrapper"
            js_func_name = func.name

            # TODO: Name functions based wrap directives or based on arguments list
            if index > 0:
                wrap_func_name += str(index)
                js_func_name += str(index)

            c_func_name = 'Wrappers::' + wrap_func_name

            # Binding template-
            arg_names = ['arg' + str(i + 1) for i in range(0, len(variant.args))]
            w_signature = [" ".join(pair) for pair in zip(arg_types, arg_names)]


            # Argument list, signature
            arg_names_casted = [c if a == b else c + '.as<' + a + '>()' for a, b, c in
                                zip(unwrapped_arg_types, arg_types, arg_names)]


            for arg in variant.args:
                if arg.defval != "":
                    has_def_param = True


            # Add self object to the parameters
            if class_info and not  factory:
                arg_types = [class_info.cname + '&'] + arg_types
                w_signature = [class_info.cname + '& arg0 '] + w_signature

            ###################################
            # Wrapper
            if has_def_param and with_default_params:
                # Generate default arguments
                i = 0
                while i <= len(def_args):
                    check_arg = wrapper_overload_def_values[i]

                    if i == 0:
                        arg_list = arg_names_casted
                    else:
                        arg_list = arg_names_casted[0:-i] + def_args[-i:]

                    if global_func:
                        cpp_call_code = call_template.substitute(func=func.cname,
                                                                 args=', '.join(arg_list))
                    else:
                        cpp_call_code = class_call_template.substitute(obj='arg0',
                                                                       func=func.cname,
                                                                       args=', '.join(arg_list))
                        arg_list = [class_info.cname + '&'] + arg_list

                    if i > 0:
                        next_code = check_arg_text
                    if i == 0:
                        check_arg_text = check_arg.substitute(cpp_call=cpp_call_code)
                    elif i == 1:
                        check_arg_text = check_arg.substitute(next=next_code,
                                                              cpp_call=cpp_call_code,
                                                              arg0=arg_names[-1])
                    elif i == 2:
                        check_arg_text = check_arg.substitute(next=next_code,
                                                              cpp_call=cpp_call_code,
                                                              arg0=arg_names[-2],
                                                              arg1=arg_names[-1])
                    elif i == 3:
                        check_arg_text = check_arg.substitute(next=next_code,
                                                              cpp_call=cpp_call_code,
                                                              arg0=arg_names[-3],
                                                              arg1=arg_names[-2],
                                                              arg2=arg_names[-1])
                    elif i == 4:
                        check_arg_text = check_arg.substitute(next=next_code,
                                                              cpp_call=cpp_call_code,
                                                              arg0=arg_names[-4],
                                                              arg1=arg_names[-3],
                                                              arg2=arg_names[-2],
                                                              arg3=arg_names[-1])
                    elif i == 5:
                        check_arg_text = check_arg.substitute(next=next_code,
                                                              cpp_call=cpp_call_code,
                                                              arg0=arg_names[-5],
                                                              arg1=arg_names[-4],
                                                              arg2=arg_names[-3],
                                                              arg3=arg_names[-2],
                                                              arg4=arg_names[-1])
                    elif i == 6:
                        check_arg_text = check_arg.substitute(next=next_code,
                                                              cpp_call=cpp_call_code,
                                                              arg0=arg_names[-6],
                                                              arg1=arg_names[-5],
                                                              arg2=arg_names[-4],
                                                              arg3=arg_names[-3],
                                                              arg4=arg_names[-2],
                                                              arg5=arg_names[-1])
                    else:
                        err = 1

                    i += 1

                wrapper_func_text = wrapper_function_with_def_args_template.substitute(ret_val=ret_type,
                                                                                       func=wrap_func_name,
                                                                                       signature=', '.join(w_signature),
                                                                                       check_args=check_arg_text,
                                                                                       const='' if variant.is_const else '')
            else:

                if factory: # TODO or static
                    name = class_info.cname+'::' if variant.class_name else ""
                    cpp_call_text = static_class_call_template.substitute(scope=name,
                                                                   func=func.cname,
                                                                   args=', '.join(arg_names))
                elif class_info:
                    cpp_call_text = class_call_template.substitute(obj='arg0',
                                                                   func=func.cname,
                                                                   args=', '.join(arg_names))
                else:
                    cpp_call_text = call_template.substitute(func=func.cname,
                                                             args=', '.join(arg_names))


                wrapper_func_text = wrapper_function_template.substitute(ret_val=ret_type,
                                                                             func=wrap_func_name,
                                                                             signature=', '.join(w_signature),
                                                                             cpp_call=cpp_call_text,
                                                                             const='' if variant.is_const else '')
            ###################################
            # Binding
            if class_info:
                if factory:
                    binding_text = ctr_template.substitute(const='const' if variant.is_const else '',
                                                       cpp_name=c_func_name,
                                                       ret=ret_type,
                                                       args=','.join(arg_types),
                                                       optional=func_attribs)
                else:
                    binding_template = overload_class_static_function_template if variant.is_class_method else \
                        overload_class_function_template
                    binding_text = binding_template.substitute(js_name=js_func_name,
                                                       const='' if variant.is_const else '',
                                                       cpp_name=c_func_name,
                                                       ret=ret_type,
                                                       args=','.join(arg_types),
                                                       optional=func_attribs)
            else:
                binding_text = overload_function_template.substitute(js_name=js_func_name,
                                                   cpp_name=c_func_name,
                                                   const='const' if variant.is_const else '',
                                                   ret=ret_type,
                                                   args=', '.join(arg_types),
                                                   optional=func_attribs)

            bindings.append(binding_text)
            wrappers.append(wrapper_func_text)

        return [bindings, wrappers]


    def gen_function_binding(self, func, class_info):

        if not class_info == None :
            func_name = class_info.cname+'::'+func.cname
        else :
            func_name = func.cname

        binding_text = None
        binding_text_list = []

        for index, variant in enumerate(func.variants):
            factory = False
            #TODO if variant.is_class_method and variant.rettype == ('Ptr<' + class_info.name + '>'):
            if (not class_info == None) and variant.rettype == ('Ptr<' + class_info.name + '>') or (func.name.startswith("create") and variant.rettype):
                factory = True
                base_class_name = variant.rettype
                base_class_name = base_class_name.replace("Ptr<","").replace(">","").strip()
                if base_class_name in self.classes:
                    self.classes[base_class_name].has_smart_ptr = True
                else:
                    print(base_class_name, ' not found in classes for registering smart pointer using ', class_info.name, 'instead')
                    self.classes[class_info.name].has_smart_ptr = True


            # Return type
            ret_type = 'void' if variant.rettype.strip() == '' else variant.rettype

            ret_type = ret_type.strip()

            if ret_type.startswith('Ptr'): #smart pointer
                ptr_type = ret_type.replace('Ptr<', '').replace('>', '')
                if ptr_type in type_dict:
                    ret_type = type_dict[ptr_type]
            for key in type_dict:
                if key in ret_type:
                    ret_type = ret_type.replace(key, type_dict[key])

            if variant.constret and ret_type.startswith('const') == False:
                ret_type = 'const ' + ret_type
            if variant.refret and ret_type.endswith('&') == False:
                ret_type += '&'

            arg_types = []
            orig_arg_types = []
            for arg in variant.args:
                if arg.tp in type_dict:
                    arg_type = type_dict[arg.tp]
                else:
                    arg_type = arg.tp

                #if arg.outputarg:
                #    arg_type += '&'
                orig_arg_types.append(arg_type)
                if with_default_params and arg.defval != '':
                    arg_types.append('emscripten::val')
                else:
                    arg_types.append(orig_arg_types[-1])

            # Function attribure
            func_attribs = ''
            if '*' in ''.join(orig_arg_types):
                func_attribs += ', allow_raw_pointers()'

            if variant.is_pure_virtual:
                func_attribs += ', pure_virtual()'

            #TODO better naming
            #if variant.name in self.jsfunctions:
            #else
            js_func_name = variant.name


            c_func_name = func.cname if (factory and variant.is_class_method == False) else func_name


            ################################### Binding

            if factory:
                binding_text = ctr_template.substitute(const='const' if variant.is_const else '',
                                                       cpp_name=c_func_name,
                                                       ret=ret_type,
                                                       args=','.join(arg_types),
                                                       optional=func_attribs)
            else:
                binding_template = overload_class_static_function_template if variant.is_class_method else \
                        overload_function_template if class_info == None else overload_class_function_template
                binding_text = binding_template.substitute(js_name=js_func_name,
                                                           const='const' if variant.is_const else '',
                                                           cpp_name=c_func_name,
                                                           ret=ret_type,
                                                           args=','.join(arg_types),
                                                           optional=func_attribs)

            binding_text_list.append(binding_text)

        return binding_text_list



    def gen(self, src_files):

        global with_default_params

        # step 1: scan the headers and extract classes, enums and functions
        for hdr in src_files:
            decls = self.parser.parse(hdr)
            if len(decls) == 0:
                continue
            for decl in decls:
                name = decl[0]
                type = name[:name.find(" ")]
                if type == "struct" or type == "class":  # class/structure case
                    name = name[name.find(" ") + 1:].strip()
                    self.add_class(type, name, decl)
                elif name.startswith("enum"):  # enumerations
                    self.add_enum(decl)
                else:  # class/global function
                    self.add_func(decl)

        # step 2: generate bindings
        # Global functions
        for ns_name, ns in sorted(self.namespaces.items()):
            if ns_name.split('.')[0] != 'cv':
                continue
            for name, func in sorted(ns.funcs.items()):
                if name in ignore_list:
                    continue

                ext_cnst = False
                # Check if the method is an external constructor
                for variant in func.variants:
                    if "Ptr<" in variant.rettype:

                        # Register the smart pointer
                        base_class_name = variant.rettype
                        base_class_name = base_class_name.replace("Ptr<","").replace(">","").strip()
                        self.classes[base_class_name].has_smart_ptr = True

                        # Adds the external constructor
                        class_name = func.name.replace("create", "")
                        if not class_name in self.classes:
                            self.classes[base_class_name].methods[func.cname] = func
                        else:
                            self.classes[class_name].methods[func.cname] = func
                        ext_cnst = True
                if ext_cnst:
                    continue

                if with_wrapped_functions:
                    binding, wrapper = self.gen_function_binding_with_wrapper(func, class_info=None)
                    self.bindings += binding
                    self.wrapper_funcs += wrapper
                else:
                    binding = self.gen_function_binding(func, class_info=None)
                    self.bindings+=binding

        # generate code for the classes and their methods
        class_list = list(self.classes.items())

        with_default_params = False
        for name, class_info in class_list:
            class_bindings = []

            # Generate bindings for methods
            for method_name, method in class_info.methods.iteritems():
                if method.cname in ignore_list:
                    continue
                if method.is_constructor:
                    for variant in method.variants:
                        args = []
                        for arg in variant.args:
                            args.append(arg.tp)
                        class_bindings.append(constructor_template.substitute(signature=', '.join(args)))
                else:
                    if with_wrapped_functions and (len(method.variants) > 1 or len(method.variants[0].args)>0 or "String" in method.variants[0].rettype):
                        binding, wrapper = self.gen_function_binding_with_wrapper(method, class_info=class_info)
                        self.wrapper_funcs = self.wrapper_funcs + wrapper
                        class_bindings = class_bindings + binding
                    else:
                        binding = self.gen_function_binding(method, class_info=class_info)
                        class_bindings = class_bindings + binding

            # Regiseter Smart pointer
            if class_info.has_smart_ptr:
                class_bindings.append(smart_ptr_reg_template.substitute(cname=class_info.cname, name=class_info.name))

            # Attach external constructors
            for method_name, method in class_info.ext_constructors.iteritems():
                print("ext constructor", method_name)
            #if class_info.ext_constructors:



            # Generate bindings for properties
            for property in class_info.props:
                class_bindings.append(class_property_template.substitute(js_name=property.name, cpp_name='::'.join(
                    [class_info.cname, property.name])))

            dv = ''
            base = Template("""base<$base$isPoly>""")

            assert len(class_info.bases) <= 1 , "multiple inheritance not supported"

            if len(class_info.bases) == 1:
                dv = "," + base.substitute(base=', '.join(class_info.bases),
                                           isPoly = " ,true" if class_info.name=="Feature2D" else "")

            self.bindings.append(class_template.substitute(cpp_name=class_info.cname,
                                                           js_name=name,
                                                           class_templates=''.join(class_bindings),
                                                           derivation=dv))

        if export_enums:
            # step 4: generate bindings for enums
            # TODO anonymous enums are ignored for now.
            for ns_name, ns in sorted(self.namespaces.items()):
                if ns_name.split('.')[0] != 'cv':
                    continue
                for name, enum in sorted(ns.enums.items()):
                    if not name.endswith('.anonymous'):
                        name = name.replace("cv.", "")
                        enum_values = []
                        for enum_val in enum:
                            value = enum_val[0][enum_val[0].rfind(".")+1:]
                            enum_values.append(enum_item_template.substitute(val=value,
                                                                             cpp_val=name.replace('.', '::')+'::'+value))

                        self.bindings.append(enum_template.substitute(cpp_name=name.replace(".", "::"),
                                                                      js_name=name.replace(".", "_"),
                                                                      enum_items=''.join(enum_values)))
                    else:
                        print(name)
                        #TODO: represent anonymous enums with constants

        with open("core_bindings.cpp") as f:
            ret = f.read()

        defis = '\n'.join(self.wrapper_funcs)
        ret += wrapper_codes_template.substitute(ns=wrapper_namespace, defs=defis)
        ret += emscripten_binding_template.substitute(binding_name='testBinding', bindings=''.join(self.bindings))


        print(ret)
        text_file = open("../bindings.cpp", "w")
        text_file.write(ret)
        text_file.close()


if __name__ == "__main__":
    srcFiles = opencv_hdr_list = [
                    "../opencv/Modules/core/include/opencv2/core.hpp",
                    "../opencv/Modules/core/include/opencv2/core/types.hpp",
                    "../opencv/Modules/core/include/opencv2/core/ocl.hpp",
                    "../opencv/Modules/core/include/opencv2/core/mat.hpp",
                    "../opencv/Modules/flann/include/opencv2/flann.hpp",
                    "../opencv/Modules/flann/include/opencv2/flann/miniflann.hpp",
                    "../opencv/Modules/ml/include/opencv2/ml.hpp",
                    "../opencv/Modules/photo/include/opencv2/photo.hpp",
                    "../opencv/Modules/imgproc/include/opencv2/imgproc.hpp",
                    "../opencv/Modules/shape/include/opencv2/shape.hpp",
                    "../opencv/Modules/shape/include/opencv2/shape/hist_cost.hpp",
                    "../opencv/Modules/shape/include/opencv2/shape/shape_distance.hpp",
                    "../opencv/Modules/shape/include/opencv2/shape/shape_transformer.hpp",
                    "../opencv/Modules/features2d/include/opencv2/features2d.hpp",
                    "../opencv/Modules/video/include/opencv2/video/tracking.hpp",
                    "../opencv/Modules/video/include/opencv2/video/background_segm.hpp",
                    "../opencv/Modules/objdetect/include/opencv2/objdetect.hpp",
                    "../opencv/Modules/imgcodecs/include/opencv2/imgcodecs.hpp"
                    ]

    if len(sys.argv) > 1:
        dstDir = sys.argv[1]
    if len(sys.argv) > 2:
        srcFiles = open(sys.argv[2], 'r').read().split(';')
    generator = JSWrapperGenerator()
    generator.gen(srcFiles)
