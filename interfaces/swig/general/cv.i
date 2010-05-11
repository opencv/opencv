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


%module(package="opencv") cv


%{
#include "cxtypes.h"
#include "cxcore.h"
#include "cvtypes.h"
#include "cv.h"
%}

// The preprocessor "gcc -E" may generate compiler specific storage specifiers that confuse SWIG
// We just define them away
#define __attribute__(arg)
#define __inline  inline
#define __const   const

// SWIG needs this to be parsed before cv.h
%ignore CV_SET_IMAGE_IO_FUNCTIONS;
%include "./cvmacros.i"

// A couple of typemaps helps wrapping OpenCV functions in a sensible way
%include "./memory.i"
%include "./typemaps.i"
%include "./doublepointers.i"

// hide COI and ROI functions
%ignore cvSetImageCOI;
%ignore cvSetImageROI;
%ignore cvGetImageROI;
%ignore cvGetImageCOI;

// mask some functions that return IplImage *
%ignore cvInitImageHeader;
%ignore cvGetImage;
%ignore cvCreateImageHeader;

// adapt others to return CvMat * instead
%ignore cvCreateImage;
%rename (cvCreateImage) cvCreateImageMat;
%ignore cvCloneImage;
%rename (cvCloneImage) cvCloneImageMat;
%inline %{
CvMat * cvCreateImageMat( CvSize size, int depth, int channels ){
    static const signed char icvDepthToType[]=
    {
        -1, -1, CV_8U, CV_8S, CV_16U, CV_16S, -1, -1,
        CV_32F, CV_32S, -1, -1, -1, -1, -1, -1, CV_64F, -1
    };

	depth = icvDepthToType[((depth & 255) >> 2) + (depth < 0)];
	return cvCreateMat( size.height, size.width, CV_MAKE_TYPE(depth, channels));
}
#define cvCloneImageMat( mat ) cvCloneMat( mat )

#ifdef WIN32

CvModuleInfo *CvModule::first=0;
CvModuleInfo *CvModule::last=0;
CvTypeInfo *CvType::first=0;
CvTypeInfo *CvType::last=0;

#endif

%}
CvMat * cvCloneImageMat( CvMat * mat );


// Now include the filtered OpenCV constants and prototypes (includes cxcore as well)
%include "../filtered/constants.h"
%include "../filtered/cv.h"

%include "./extensions.i"
%include "./cvarr_operators.i"
