function mex(varargin)
%CV.MEX compile MEX-function with OpenCV linkages
%
%   Usage:
%       CV.MEX [options ...] file [file file ...]
%
%   Description:
%       CV.MEX compiles one or more C/C++ source files into a shared-library
%       called a mex-file. This function is equivalent to the builtin MEX
%       routine, with the notable exception that it automatically resolves
%       OpenCV includes, and links in the OpenCV libraries where appropriate.
%       It also forwards the flags used to build OpenCV, so architecture-
%       specific optimizations can be used.
%
%       CV.MEX is designed to be used in situations where the source(s) you
%       are compiling contain OpenCV definitions. In such cases, it streamlines
%       the finding and including of appropriate OpenCV libraries.
%
%   See also: mex
%
%   Copyright {{ time.strftime("%Y", time.localtime()) }} The OpenCV Foundation
%

  % forward the OpenCV build flags (C++ only)
  EXTRA_FLAGS  = ['"CXXFLAGS="\$CXXFLAGS '...
                  '{{ cv.flags | trim | wordwrap(60, false, '\'...\n                  \'') }}""'];

  % add the OpenCV include dirs
  INCLUDE_DIRS = {{ cv.include_dirs | split | cellarray | wordwrap(60, false, '...\n                  ') }};

  % add the lib dir (singular in both build tree and install tree)
  LIB_DIR      = '{{ cv.lib_dir }}';

  % add the OpenCV libs. Only the used libs will actually be linked
  LIBS         = {{ cv.libs | split | cellarray | wordwrap(60, false, '...\n                  ') }};

  % add the mex opts (usually at least -largeArrayDims)
  OPTS         = {{ cv.opts | split | cellarray | wordwrap(60, false, '...\n                  ') }};

  % merge all of the default options (EXTRA_FLAGS, LIBS, etc) and the options
  % and files passed by the user (varargin) into a single cell array
  merged       = [ {EXTRA_FLAGS}, INCLUDE_DIRS, {LIB_DIR}, LIBS, OPTS, varargin ];

  % expand the merged argument list into the builtin mex utility
  mex(merged{:});
end
