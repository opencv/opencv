/* 
 * compose
 * compose a function call
 * This macro takes as input a Function object and composes
 * a function call by inspecting the types and argument names
 */
/
{% macro compose(fun) %}
  {# ----------- Return type ------------- #}
  {%- if not fun.rtp|void -%} retval = {% endif -%}
  {%- if fun.clss -%}inst.{%- else -%} cv:: {%- endif -%}
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

// create a full function invocation
{%- macro generate(fun) -%}

  // unpack the arguments
  {# ----------- Inputs ------------- #}
  {% for arg in fun.req|inputs %}
  {{arg.tp}} {{arg.name}} = inputs[{{ loop.index0 }}].to{{arg.tp|toUpperCamelCase}}();
  {% endfor %}
  {% for opt in fun.opt|inputs %}
  {{opt.tp}} {{opt.name}} = (nrhs > {{loop.index0 + fun.req|inputs|length}}) ? inputs[{{loop.index0 + fun.req|inputs|length}}].to{{opt.tp|toUpperCamelCase}}() : {% if opt.ref == '*' -%} {{opt.tp}}() {%- else -%} {{opt.default}} {%- endif %};
  {% endfor %}
  {# ----------- Outputs ------------ #}
  {% for arg in fun.req|only|outputs %}
  {{arg.tp}} {{arg.name}};
  {% endfor %}
  {% for opt in fun.opt|only|outputs %}
  {{opt.tp}} {{opt.name}};
  {% endfor %}
  {% if not fun.rtp|void %}
  {{fun.rtp}} retval;
  {% endif %}

  // call the opencv function
  // [out =] namespace.fun(src1, ..., srcn, dst1, ..., dstn, opt1, ..., optn);
  try {
    {{ compose(fun) }}
  } catch(cv::Exception& e) {
    mexErrMsgTxt(std::string("cv::exception caught: ").append(e.what()).c_str());
  } catch(std::exception& e) {
    mexErrMsgTxt(std::string("std::exception caught: ").append(e.what()).c_str());
  } catch(...) {
    mexErrMsgTxt("Uncaught exception occurred in {{fun.name}}");
  }

  // assign the outputs into the bridge
  {% if not fun.rtp|void %}
  outputs[0] = retval;
  {% endif %}
  {% for arg in fun.req|outputs %}
  outputs[{{loop.index0 + fun.rtp|void|not}}] = {{arg.name}};
  {% endfor %}
  {% for opt in fun.opt|outputs %}
  outputs[{{loop.index0 + fun.rtp|void|not + fun.req|outputs|length}}] = {{opt.name}};
  {% endfor %}
  
{% endmacro %}
