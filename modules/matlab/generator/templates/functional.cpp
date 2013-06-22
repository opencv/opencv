// compose a function
{% macro compose(fun, retname="ret") %}
  {%- if not fun.rtp == "void" -%} {{fun.rtp}} retname = {% endif -%}
  {{fun.name}}(
  {%- for arg in fun.req -%} 
    {{arg.name}} 
    {%- if not loop.last %}, {% endif %}
  {% endfor %}
  {% if fun.req and fun.opt %}, {% endif %}
  {%- for opt in fun.opt -%} 
    {{opt.name}} 
    {%- if not loop.last -%}, {% endif %}
  {%- endfor -%}
  );
{%- endmacro %}

// create a full function invocation
{%- macro generate(fun) -%}

  // unpack the arguments
  // inputs
  {% for arg in fun.req|inputs %}
  {{arg.tp}} {{arg.name}} = inputs[{{ loop.index0 }}];
  {% endfor %}
  {% for opt in fun.opt|inputs %}
  {{opt.tp}} {{opt.name}} = (nrhs > {{loop.index0 + fun.req|ninputs}}) ? inputs[{{loop.index0 + fun.req|ninputs}}] : {{opt.default}};
  {% endfor %}

  // outputs
  {% for arg in fun.req|outputs %}
  {{arg.tp}} {{arg.name}};
  {% endfor %}
  {% for opt in fun.opt|outputs %}
  {{opt.tp}} {{opt.name}};
  {% endfor %}

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
  {% for arg in fun.req|outputs %}
  outputs[{{loop.index0}}] = {{arg.name}};
  {% endfor %}
  {% for opt in fun.opt|outputs %}
  outputs[{{loop.index0 + fun.req|noutputs}}] = {{opt.name}};
  {% endfor %}

{%- endmacro -%}
