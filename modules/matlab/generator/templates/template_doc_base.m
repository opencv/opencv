{% import 'functional.cpp' as functional %}
%CV.{{ fun.name | upper }}
%   {{ functional.composeMatlab(fun) | upper }}
%
{% if fun.rtp|void|not or fun.req|outputs|length or fun.opt|outputs|length %}
%   Returns:
{% endif %}
{% if fun.rtp|void|not %}
%      LVALUE
{% endif %}
{% for arg in fun.req|outputs + fun.opt|outputs %}
%      {{arg.name | upper}}{%- if arg.I -%}_OUT{% endif %}

{% endfor %}
%
{% if fun.req|inputs|length %}
%   Required Inputs:
{% endif %}
{% for arg in fun.req|inputs %}
%      {{arg.name | upper}}{%- if arg.O -%}_OUT{% endif %}

{% endfor %}
%
{% if fun.opt|inputs|length %}
%   Optional Inputs:
{% endif %}
{% for arg in fun.opt|inputs %}
%      {{arg.name | upper}}{%- if arg.O -%}_OUT{% endif %}

{% endfor %}
%
%   See also:
%
%   Official documentation: http://docs.opencv.org
%   Copyright {{ time.strftime("%Y", time.localtime()) }} The OpenCV Foundation
%
