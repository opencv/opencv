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


%module(package="opencv") highgui

%{
#include "highgui.h"
%}

%import "./cv.i"

%include "./memory.i"
%include "./typemaps.i"

%newobject cvLoadImage;
%newobject cvLoadImageM;
%newobject cvLoadImageMat;

%nodefault CvCapture;
%newobject cvCaptureFromFile;
%newobject cvCaptureFromCAM;

%nodefault CvVideoWriter;
%newobject cvCreateVideoWriter;

/** modify the following to return CvMat instead of IplImage */
%ignore cvLoadImage;
%rename (cvLoadImage) cvLoadImageMat;
%inline %{
CvMat * cvLoadImageMat(const char* filename, int iscolor=CV_LOAD_IMAGE_COLOR ){
	return cvLoadImageM(filename, iscolor);
}
%}

%typemap_out_CvMat(cvRetrieveFrame, ( CvCapture* capture ), (capture));
%typemap_out_CvMat(cvQueryFrame, ( CvCapture * capture ), (capture));

%include "highgui.h"

struct CvCapture {
};
struct CvVideoWriter {
};
%extend CvCapture     { ~CvCapture ()     { CvCapture *     dummy = self; cvReleaseCapture     (& dummy); } }
%extend CvVideoWriter { ~CvVideoWriter () { CvVideoWriter * dummy = self; cvReleaseVideoWriter (& dummy); } }


