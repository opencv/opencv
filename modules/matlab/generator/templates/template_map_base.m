% CV
% This class enumerates all OpenCV constants, stripping them
% out of classes where necessary. The constants can then be 
% used in OpenCV functions by prefixing the class name
% e.g.
%   cv.dft(x, xf, cv.DFT_FORWARD);
%
% The properties are all declared Constant, so they cannot be
% changed, however they can be accidentally aliased if you 
% declare a variable of the same name first. If you're 
% particularly afraid of aliasing, you can call cv() before
% calling constants to parse the variable 'cv' as this class
%
% Note that calls to this class and calls to methods contained
% in the namespace cv can happily coexist
%
% Users also have the option of calling the constants as strings
% e.g.
%   cv.dft(x, xf, "DFT_FORWARD");
% 
% This tends to be faster as it is hashed in C++, but the
% values of the constants cannot be introspected
classdef cv
    properties (Constant = true)
    {% for key, val in constants.items() %}
        {% if val|convertibleToInt %}
        {{key}} = {{val}};
        {% else %}
        {{key}} = {{constants[val]}};
        {% endif %}
    {% endfor %}
    end
end
