% ------------------------------------------------------------------------
%                              OpenCV Toolbox
%                  Matlab bindings for the OpenCV library
% ------------------------------------------------------------------------
%
% The OpenCV Toolbox allows you to make calls to native OpenCV methods
% and classes directly from within Matlab. 
%
% PATHS
% To call OpenCV methods from anywhere in your workspace, add the
% directory containing this file to the path: 
%
%     addpath(fileparts(which('cv')));
%
% The OpenCV Toolbox contains two important locations:
%     cv.m - This file, containing OpenCV enums
%     +cv/ - The directory containing the OpenCV methods and classes
%
% CALLING SYNTAX
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
% HELP
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
% DIAGNOSTICS
% If you are having problems with the OpenCV Toolbox and need to send a 
% bug report to the OpenCV team, you can get a printout of diagnostic 
% information to submit along with your report by typing:
%
%     cv.buildInformation();
%
% OTHER RESOURCES
% OpenCV documentation online: http://docs.opencv.org
% OpenCV issue tracker: http://code.opencv.org
% OpenCV Q&A: http://answers.opencv.org
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
