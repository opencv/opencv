function buildInformation()
%CV.BUILDINFORMATION display OpenCV Toolbox build information
%
%   Call CV.BUILDINFORMATION() to get a printout of diagonstic information
%   pertaining to your particular build of the OpenCV Toolbox. If you ever
%   run into issues with the Toolbox, it is useful to submit this
%   information alongside a bug report to the OpenCV team.
%
%   Copyright {{ time.strftime("%Y", time.localtime()) }} The OpenCV Foundation
%
info = {
'  ------------------------------------------------------------------------'
'                              <strong>OpenCV Toolbox</strong>'
'                     Build and diagnostic information'
'  ------------------------------------------------------------------------'
''
'  <strong>Platform</strong>'
'  OS:            {{ build.os }}'
'  Architecture:  {{ build.arch[0] }}-bit {{ build.arch[1] }}'
'  Compiler:      {{ build.compiler | csv(' ') }}'
''
'  <strong>Matlab</strong>'
['  Version:       ' version()]
['  Mex extension: ' mexext()]
'  Architecture:  {{ build.mex_arch }}'
'  Mex path:      {{ build.mex_script }}'
'  Mex flags:     {{ build.mex_opts | csv(' ') }}'
'  CXX flags:     {{ build.cxx_flags | csv(' ') | stripExtraSpaces | wordwrap(60, True, '\'\n\'                 ') }}'
''
'  <strong>OpenCV</strong>'
'  Version:       {{ build.opencv_version }}'
'  Commit:        {{ build.commit }}'
'  Configuration: {{ build.configuration }}'
'  Modules:       {{ build.modules | csv | wordwrap(60, True, '\'\n\'                 ') }}'
''
};

info = cellfun(@(x) [x '\n'], info, 'UniformOutput', false);
info = horzcat(info{:});
fprintf(info);
end
