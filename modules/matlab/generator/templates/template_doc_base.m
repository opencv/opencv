{% import 'functional.cpp' as functional %}
{{ ('CV.' + fun.name | upper + ' ' + doc.brief | stripTags) | comment(75, '%') | matlabURL }}
%
%   {{ functional.composeMatlab(fun) | upper }}
{% if doc.long %}
{{ doc.long | stripTags | qualify(fun.name) | comment(75, '%   ') | matlabURL }}
{% endif %}
%
{# ----------------------- Returns --------------------- #}
{% if fun.rtp|void|not or fun.req|outputs|length or fun.opt|outputs|length %}
%   Returns:
{% if fun.rtp|void|not %}
%      LVALUE
{% endif %}
{% for arg in fun.req|outputs + fun.opt|outputs %}
{% set uname = arg.name | upper + ('_OUT' if arg.I else '') %}
{% if arg.name in doc.params %}
{{ (uname + ' ' + doc.params[arg.name]) | stripTags | comment(75, '%     ') }}
{% else %}
{{ uname }}
{% endif %}
{% endfor %}
%
{% endif %}
{# ----------------- Required Inputs ------------------- #}
{% if fun.req|inputs|length %}
%   Required Inputs:
{% for arg in fun.req|inputs %}
{% set uname = arg.name | upper + ('_IN' if arg.O else '') %}
{% if arg.name in doc.params %}
{{ (uname + ' ' + doc.params[arg.name]) | stripTags | comment(75, '%     ') }}
{% else %}
{% endif %}
{% endfor %}
%
{% endif %}
{# ------------------ Optional Inputs ------------------- #}
{% if fun.opt|inputs|length %}
%   Optional Inputs:
{% for arg in fun.opt|inputs %}
{% set uname = arg.name | upper + ('_IN' if arg.O else '') + ' (default: ' + arg.default + ')' %}
{% if arg.name in doc.params %}
{{ (uname + ' ' + doc.params[arg.name]) | stripTags | comment(75, '%     ') }}
{% else %}
{{ uname }}
{% endif %}
{% endfor %}
%
{% endif %}
{# ---------------------- See also --------------------- #}
{% if 'seealso' in doc %}
%   See also: {% for item in doc['seealso'] %}
cv.{{ item }}{% if not loop.last %}, {% endif %}
{% endfor %}

%
{% endif %}
{# ----------------------- Online ---------------------- #}
{% set url = 'http://docs.opencv.org/modules/' + doc.module  + '/doc/' + (doc.file|filename) + '.html#' + (fun.name|slugify) %}
%   Online docs: {{ url | matlabURL }}
%   Copyright {{ time.strftime("%Y", time.localtime()) }} The OpenCV Foundation
%
