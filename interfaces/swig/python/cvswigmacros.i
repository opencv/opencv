/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/* This file contains swig macros that are used in several typemap files */


%define %myshadow(function)
%ignore function;
%rename (function) function##_Shadow;
%enddef

// Elsewhere in this wrapper, the cvRelease* functions are mapped to 
// the destructors for the corresponding OpenCV object wrapper.  This
// is done in order to let Python handle memory management.  If the 
// reference count of the Python object wrapping the OpenCV object 
// goes to 0, the garbage collector will call the destructor, and 
// therefore the cvRelease* function, before freeing the Python object.
// However, if the user explicitly calls the cvRelease* function, we 
// must prevent the Python garbage collector from calling it again when
// the refcount reaches 0 -- otherwise a double-free error occurs.
//
// Thus, below, we redirect each cvRelease* function to the 
// corresponding OpenCV object's destructor.  This has the effect of:
// (1) Calling the corresponding cvRelease* function, and therefore 
//     immediately releasing the OpenCV object.
// (2) Telling SWIG to disown memory management for this OpenCV object.  
//
// Thus, when the refcount for the Python object reaches 0, the Python
// object is garbage collected, but since it no longer owns the OpenCV 
// object, this is not freed again.
%define %myrelease(module, Function, Type)
%ignore Function;
%rename (Function) Function##_Shadow;
%pythoncode %{
Function = _##module##.delete_##Type
%}
%enddef
