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


// 2004-03-16, Gabriel Schreiber <schreiber@ient.rwth-aachen.de>
//             Mark Asbach       <asbach@ient.rwth-aachen.de>
//             Institute of Communications Engineering, RWTH Aachen University


/**
 * We set our own error handling function using cvRedirectError.
 * (See also error.h)
 * This function throws an error (OpenCV itself never throws) which 
 * we catch here. The error handling function (SendErrorToPython)
 * sets the Python error.
 * We need to return 0 here instead of an PyObject to tell Python
 * that an error has occured.
 */
%exception
    {
    try { $action } 
    catch (...) 
        {
	  SWIG_fail;
        } 
    }


/* include exception.i, so we can generate exceptions when we found errors */
%include "exception.i"

%include "sizeof.i"

/**
 * IplImage has no reference counting of underlying data, which creates problems with double 
 * frees after accessing subarrays in python -- instead, replace IplImage with CvMat, which
 * should be functionally equivalent, but has reference counting.
 * The potential shortcomings of this method are 
 * 1. no ROI
 * 2. IplImage fields missing or named something else.
 */
%typemap(in) IplImage * (IplImage header){
	void * vptr;
	int res = SWIG_ConvertPtr($input, (&vptr), $descriptor( CvMat * ), 0);
	if ( res == -1 ){
		SWIG_exception( SWIG_TypeError, "%%typemap(in) IplImage * : could not convert to CvMat");
		SWIG_fail;
	}
	$1 = cvGetImage((CvMat *)vptr, &header);
}

/** For IplImage * return type, there are cases in which the memory should be freed and 
 * some not.  To avoid leaks and segfaults, deprecate this return type and handle cases 
 * individually
 */
%typemap(out) IplImage * {
 	SWIG_exception( SWIG_TypeError, "IplImage * return type is deprecated. Please file a bug report at www.sourceforge.net/opencvlibrary if you see this error message.");
	SWIG_fail;
}

/** macro to convert IplImage return type to CvMat.  Note that this is only covers the case
 *  where the returned IplImage need not be freed.  If the IplImage header needs to be freed,
 *  then CvMat must take ownership of underlying data.  Instead, just handle these limited cases
 *  with CvMat equivalent.
 */
%define %typemap_out_CvMat(func, decl, call)
%rename (func##__Deprecated) func;
%rename (func) func##__CvMat;
%inline %{
CvMat * func##__CvMat##decl{
	IplImage * im = func##call;
	if(im){
		CvMat * mat = (CvMat *)cvAlloc(sizeof(CvMat));
		mat = cvGetMat(im, mat);
		return mat;
	}
	return false;
}
%}
%enddef
