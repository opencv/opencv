function buildInformation()
%CV.BUILDINFORMATION display OpenCV Toolbox build information
%
%   Call CV.BUILDINFORMATION() to get a printout of diagonstic information
%   pertaining to your particular build of the OpenCV Toolbox. If you ever
%   run into issues with the Toolbox, it is useful to submit this
%   information alongside a bug report to the OpenCV team.
%
%   Copyright 2017 The OpenCV Foundation
%
info = {
'  ------------------------------------------------------------------------'
'                              <strong>OpenCV Toolbox</strong>'
'                     Build and diagnostic information'
'  ------------------------------------------------------------------------'
''
'  <strong>Platform</strong>'
'  OS:            Darwin-15.6.0'
'  Architecture:  64-bit x86_64'
'  Compiler:      Clang 8.0.0.8000042'
''
'  <strong>Matlab</strong>'
['  Version:       ' version()]
['  Mex extension: ' mexext()]
'  Architecture:  maci64'
'  Mex path:      /Applications/MATLAB_R2016b.app/bin/mex'
'  Mex flags:     -largeArrayDims'
'  CXX flags:     -fsigned-char -W -Wall -Werror=return-type -Werror=non-'
'                 virtual-dtor -Werror=address -Werror=sequence-point -Wformat'
'                 -Werror=format-security -Wmissing-declarations -Wmissing-'
'                 prototypes -Wstrict-prototypes -Wundef -Winit-self'
'                 -Wpointer-arith -Wshadow -Wsign-promo -Wno-narrowing -Wno-'
'                 delete-non-virtual-dtor -Wno-unnamed-type-template-args'
'                 -Wno-comment -fdiagnostics-show-option -Wno-long-long'
'                 -Qunused-arguments -Wno-semicolon-before-method-body -fno-'
'                 omit-frame-pointer -msse -msse2 -mno-avx -msse3 -mno-ssse3'
'                 -mno-sse4.1 -mno-sse4.2'
''
'  <strong>OpenCV</strong>'
'  Version:       3.2.0-dev'
'  Commit:        dcbed8d676a4f0879fc25c31aeaf22bd738f2c63'
'  Configuration: Release'
'  Modules:       dnn, core, imgproc, ml, imgcodecs, videoio, highgui,'
'                 objdetect, flann, features2d, photo, video, videostab,'
'                 calib3d, stitching, superres, xfeatures2d'
''
};

info = cellfun(@(x) [x '\n'], info, 'UniformOutput', false);
info = horzcat(info{:});
fprintf(info);
end