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

from string import Template

wrapper_codes_template = Template("namespace $ns {\n$defs\n}")

call_template = Template("""$func($args)""")
class_call_template = Template("""$obj.$func($args)""")
static_class_call_template = Template("""$scope$func($args)""")

wrapper_function_template = Template("""    $ret_val $func($signature)$const {
        return $cpp_call;
    }
    """)

wrapper_function_with_def_args_template = Template("""    $ret_val $func($signature)$const {
        $check_args
    }
    """)

wrapper_overload_def_values = [
    Template("""return $cpp_call;"""), Template("""if ($arg0.isUndefined())
            return $cpp_call;
        else
            $next"""),
    Template("""if ($arg0.isUndefined() && $arg1.isUndefined())
            return $cpp_call;
        else $next"""),
    Template("""if ($arg0.isUndefined() && $arg1.isUndefined() && $arg2.isUndefined())
            return $cpp_call;
        else $next"""),
    Template("""if ($arg0.isUndefined() && $arg1.isUndefined() && $arg2.isUndefined() && $arg3.isUndefined())
            return $cpp_call;
        else $next"""),
    Template("""if ($arg0.isUndefined() && $arg1.isUndefined() && $arg2.isUndefined() && $arg3.isUndefined() &&
                    $arg4.isUndefined())
            return $cpp_call;
        else $next"""),
    Template("""if ($arg0.isUndefined() && $arg1.isUndefined() && $arg2.isUndefined() && $arg3.isUndefined() &&
                    $arg4.isUndefined() && $arg5.isUndefined() )
            return $cpp_call;
        else $next"""),
    Template("""if ($arg0.isUndefined() && $arg1.isUndefined() && $arg2.isUndefined() && $arg3.isUndefined() &&
                    $arg4.isUndefined() && $arg5.isUndefined() && $arg6.isUndefined() )
            return $cpp_call;
        else $next"""),
    Template("""if ($arg0.isUndefined() && $arg1.isUndefined() && $arg2.isUndefined() && $arg3.isUndefined() &&
                    $arg4.isUndefined() && $arg5.isUndefined()&& $arg6.isUndefined()  && $arg7.isUndefined())
            return $cpp_call;
        else $next"""),
    Template("""if ($arg0.isUndefined() && $arg1.isUndefined() && $arg2.isUndefined() && $arg3.isUndefined() &&
                    $arg4.isUndefined() && $arg5.isUndefined()&& $arg6.isUndefined()  && $arg7.isUndefined() &&
                    $arg8.isUndefined())
            return $cpp_call;
        else $next"""),
    Template("""if ($arg0.isUndefined() && $arg1.isUndefined() && $arg2.isUndefined() && $arg3.isUndefined() &&
                    $arg4.isUndefined() && $arg5.isUndefined()&& $arg6.isUndefined()  && $arg7.isUndefined()&&
                    $arg8.isUndefined() && $arg9.isUndefined())
            return $cpp_call;
        else $next""")]

emscripten_binding_template = Template("""

EMSCRIPTEN_BINDINGS($binding_name) {$bindings
}
""")

simple_function_template = Template("""
    emscripten::function("$js_name", &$cpp_name);
""")

smart_ptr_reg_template = Template("""
        .smart_ptr<Ptr<$cname>>("Ptr<$name>")
""")

overload_function_template = Template("""
    function("$js_name", select_overload<$ret($args)>(&$cpp_name)$optional);
""")

overload_class_function_template = Template("""
        .function("$js_name", select_overload<$ret($args)$const>(&$cpp_name)$optional)""")

overload_class_static_function_template = Template("""
        .class_function("$js_name", select_overload<$ret($args)$const>(&$cpp_name)$optional)""")

class_property_template = Template("""
        .property("$js_name", &$cpp_name)""")

ctr_template = Template("""
        .constructor(select_overload<$ret($args)$const>(&$cpp_name)$optional)""")

smart_ptr_ctr_overload_template = Template("""
        .smart_ptr_constructor("$ptr_type", select_overload<$ret($args)$const>(&$cpp_name)$optional)""")

function_template = Template("""
        .function("$js_name", &$cpp_name)""")

static_function_template = Template("""
        .class_function("$js_name", &$cpp_name)""")

constructor_template = Template("""
        .constructor<$signature>()""")

enum_item_template = Template("""
        .value("$val", $cpp_val)""")

enum_template = Template("""
    emscripten::enum_<$cpp_name>("$js_name")$enum_items;
""")

vector_template = Template("""
     emscripten::register_vector<$cType>("$js_name");
""")

map_template = Template("""
     emscripten::register_map<cpp_type_key,$cpp_type_val>("$js_name");
""")

class_template = Template("""
    emscripten::class_<$cpp_name $derivation>("$js_name")$class_templates;
""")
