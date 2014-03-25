% ------------------------------------------------------------------------
%                             <strong>OpenCV Toolbox</strong>
%                  Matlab bindings for the OpenCV library
% ------------------------------------------------------------------------
%
% The OpenCV Toolbox allows you to make calls to native OpenCV methods
% and classes directly from within Matlab.
%
% <strong>PATHS</strong>
% To call OpenCV methods from anywhere in your workspace, add the
% directory containing this file to the path:
%
%     addpath(fileparts(which('cv')));
%
% The OpenCV Toolbox contains two important locations:
%     cv.m - This file, containing OpenCV enums
%     +cv/ - The directory containing the OpenCV methods and classes
%
% <strong>CALLING SYNTAX</strong>
% To call an OpenCV method, class or enum, it must be prefixed with the
% 'cv' qualifier. For example:
%
%     % perform a Fourier transform
%     Xf = cv.dft(X, cv.DFT_COMPLEX_OUTPUT);
%
%     % create a VideoCapture object, and open a file
%     camera = cv.VideoCapture();
%     camera.open('/path/to/file');
%
% You can specify optional arguments by name, similar to how python
% and many builtin Matlab functions work. For example, the cv.dft
% method used above has an optional 'nonzeroRows' argument. If
% you want to specify that, but keep the default 'flags' behaviour,
% simply call the method as:
%
%     Xf = cv.dft(X, 'nonzeroRows', 7);
%
% <strong>HELP</strong>
% Each method has its own help file containing information about the
% arguments, return values, and what operation the method performs.
% You can access this help information by typing:
%
%     help cv.methodName
%
% The full list of methods can be found by inspecting the +cv/
% directory. Note that the methods available to you will depend
% on which modules you configured OpenCV to build.
%
% <strong>DIAGNOSTICS</strong>
% If you are having problems with the OpenCV Toolbox and need to send a
% bug report to the OpenCV team, you can get a printout of diagnostic
% information to submit along with your report by typing:
%
%     <a href="matlab: cv.buildInformation()">cv.buildInformation();</a>
%
% <strong>OTHER RESOURCES</strong>
% OpenCV documentation online: <a href="matlab: web('http://docs.opencv.org', '-browser')">http://docs.opencv.org</a>
% OpenCV issue tracker: <a href="matlab: web('http://code.opencv.org', '-browser')">http://code.opencv.org</a>
% OpenCV Q&A: <a href="matlab: web('http://answers.opencv.org', '-browser')">http://answers.opencv.org</a>
%
% See also: cv.help, <a href="matlab: cv.buildInformation()">cv.buildInformation</a>
%
% Copyright {{ time.strftime("%Y", time.localtime()) }} The OpenCV Foundation
%
classdef cv
    properties (Constant = true)
    {% for key, val in constants.items() %}
        {{key}} = {{val|formatMatlabConstant(constants)}};
    {% endfor %}
    end
end
