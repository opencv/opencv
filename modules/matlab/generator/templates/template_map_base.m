classdef cv
    properties (Constant = true)
    {% for key, val in constants.items() %}
        {{key}} = {{val}};
    {% endfor %}
    end
end
