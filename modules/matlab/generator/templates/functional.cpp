/*
 * compose
 * compose a function call
 * This macro takes as input a Method object and composes
 * a function call by inspecting the types and argument names
 */
{% macro compose(fun) %}
  {# ----------- Return type ------------- #}
  {%- if not fun.rtp|void and not fun.constructor -%} retval = {% endif -%}
  {%- if fun.constructor -%}{{fun.clss}} obj = {% endif -%}
  {%- if fun.clss and not fun.constructor -%}inst.{%- else -%} cv:: {%- endif -%}
  {{fun.name}}(
  {#- ----------- Required ------------- -#}
  {%- for arg in fun.req -%}
    {%- if arg.ref == '*' -%}&{%- endif -%}
    {{arg.name}}
    {%- if not loop.last %}, {% endif %}
  {% endfor %}
  {#- ----------- Optional ------------- -#}
  {% if fun.req and fun.opt %}, {% endif %}
  {%- for opt in fun.opt -%}
    {%- if opt.ref == '*' -%}&{%- endif -%}
    {{opt.name}}
    {%- if not loop.last -%}, {% endif %}
  {%- endfor -%}
  );
{%- endmacro %}


/*
 * composeMatlab
 * compose a Matlab function call
 * This macro takes as input a Method object and composes
 * a Matlab function call by inspecting the types and argument names
 */
{% macro composeMatlab(fun) %}
  {# ----------- Return type ------------- #}
  {%- if fun|noutputs > 1 -%}[{% endif -%}
  {%- if not fun.rtp|void -%}LVALUE{% endif -%}
  {%- if not fun.rtp|void and fun|noutputs > 1 -%},{% endif -%}
  {# ------------- Outputs ------------- -#}
  {%- for arg in fun.req|outputs + fun.opt|outputs -%}
    {{arg.name}}
    {%- if arg.I -%}_out{%- endif -%}
    {%- if not loop.last %}, {% endif %}
  {% endfor %}
  {%- if fun|noutputs > 1 -%}]{% endif -%}
  {%- if fun|noutputs %} = {% endif -%}
  cv.{{fun.name}}(
  {#- ------------ Inputs -------------- -#}
  {%- for arg in fun.req|inputs + fun.opt|inputs -%}
    {{arg.name}}
    {%- if arg.O -%}_in{%- endif -%}
    {%- if not loop.last %}, {% endif -%}
  {% endfor -%}
  );
{%- endmacro %}


/*
 * composeVariant
 * compose a variant call for the ArgumentParser
 */
{% macro composeVariant(fun) %}
addVariant("{{ fun.name }}", {{ fun.req|inputs|length }}, {{ fun.opt|inputs|length }}
{%- if fun.opt|inputs|length %}, {% endif -%}
{%- for arg in fun.opt|inputs -%}
  "{{arg.name}}"
  {%- if not loop.last %}, {% endif -%}
{% endfor -%}
)
{%- endmacro %}


/*
 * composeWithExceptionHandler
 * compose a function call wrapped in exception traps
 * This macro takes an input a Method object and composes a function
 * call through the compose() macro, then wraps the return in traps
 * for cv::Exceptions, std::exceptions, and all generic exceptions
 * and returns a useful error message to the Matlab interpreter
 */
{%- macro composeWithExceptionHandler(fun) -%}
  // call the opencv function
  // [out =] namespace.fun(src1, ..., srcn, dst1, ..., dstn, opt1, ..., optn);
  try {
    {{ compose(fun) }}
  } catch(cv::Exception& e) {
    error(std::string("cv::exception caught: ").append(e.what()).c_str());
  } catch(std::exception& e) {
    error(std::string("std::exception caught: ").append(e.what()).c_str());
  } catch(...) {
    error("Uncaught exception occurred in {{fun.name}}");
  }
{%- endmacro %}


/*
 * handleInputs
 * unpack input arguments from the Bridge
 * Given an input Bridge object, this unpacks the object from the Bridge and
 * casts them into the correct type
 */
{%- macro handleInputs(fun) %}

  {% if fun|ninputs or (fun|noutputs and not fun.constructor) %}
  // unpack the arguments
  {# ----------- Inputs ------------- #}
  {% for arg in fun.req|inputs %}
  {{arg.tp}} {{arg.name}} = inputs[{{ loop.index0 }}].to{{arg.tp|toUpperCamelCase}}();
  {% endfor %}
  {% for opt in fun.opt|inputs %}
  {{opt.tp}} {{opt.name}} = inputs[{{loop.index0 + fun.req|inputs|length}}].empty() ? {% if opt.ref == '*' -%} {{opt.tp}}() {%- else -%} {{opt.default}} {%- endif %} : inputs[{{loop.index0 + fun.req|inputs|length}}].to{{opt.tp|toUpperCamelCase}}();
  {% endfor %}
  {# ----------- Outputs ------------ #}
  {% for arg in fun.req|only|outputs %}
  {{arg.tp}} {{arg.name}};
  {% endfor %}
  {% for opt in fun.opt|only|outputs %}
  {{opt.tp}} {{opt.name}};
  {% endfor %}
  {% if not fun.rtp|void and not fun.constructor %}
  {{fun.rtp}} retval;
  {% endif %}
  {% endif %}

{%- endmacro %}

/*
 * handleOutputs
 * pack outputs into the bridge
 * Given a set of outputs, this methods assigns them into the bridge for
 * return to the calling method
 */
{%- macro handleOutputs(fun) %}

  {% if fun|noutputs %}
  // assign the outputs into the bridge
  {% if not fun.rtp|void and not fun.constructor %}
  outputs[0] = retval;
  {% endif %}
  {% for arg in fun.req|outputs %}
  outputs[{{loop.index0 + fun.rtp|void|not}}] = {{arg.name}};
  {% endfor %}
  {% for opt in fun.opt|outputs %}
  outputs[{{loop.index0 + fun.rtp|void|not + fun.req|outputs|length}}] = {{opt.name}};
  {% endfor %}
  {% endif %}
{%- endmacro %}
