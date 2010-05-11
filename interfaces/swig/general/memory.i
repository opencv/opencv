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
// 2008-05-15, Xavier Delacour   <xavier.delacour@gmail.com>


/****************************************************************************************\
*          Array allocation, deallocation, initialization and access to elements         *
\****************************************************************************************/

%nodefault _IplImage;
%newobject cvCreateImage;
%newobject cvCreateImageMat;
%newobject cvCreateImageHeader;
%newobject cvCloneImage;
%newobject cvCloneImageMat;

%nodefault CvMat;
%newobject cvCreateMat;
%newobject cvCreateMatHeader;
%newobject cvCloneMat;
%newobject cvGetSubRect;
%newobject cvGetRow;
%newobject cvGetRows;
%newobject cvGetCol;
%newobject cvGetCols;
%newobject cvGetDiag;

%nodefault CvMatND;
%newobject cvCreateMatND;
%newobject cvCreateMatHeaderND;
%newobject cvCloneMatND;

%nodefault CvSparseMat;
%newobject cvCreateSparseMat;
%newobject cvCloneSparseMat;


/****************************************************************************************\
*                              Dynamic data structures                                   *
\****************************************************************************************/

%nodefault CvMemStorage;
%newobject cvCreateMemStorage;
%newobject cvCreateChildMemStorage;

%nodefault CvGraphScanner;
%newobject cvCreateGraphScanner;


/****************************************************************************************\
*                                    Data Persistence                                    *
\****************************************************************************************/

%nodefault CvFileStorage;
%newobject cvOpenFileStorage;



/// cv.h

/****************************************************************************************\
*                                    Image Processing                                    *
\****************************************************************************************/

%nodefault _IplConvKernel;
%newobject cvCreateStructuringElement;


/****************************************************************************************\
*                                   Structural Analysis                                  *
\****************************************************************************************/

%newobject cvFitEllipse2;


/****************************************************************************************\
*                                       Tracking                                         *
\****************************************************************************************/

%nodefault CvKalman;
%newobject cvCreateKalman;

/****************************************************************************************\
*                                  Histogram functions                                   *
\****************************************************************************************/

%nodefault CvHistogram;
%newobject cvCreateHist;


/****************************************************************************************\
*                         Haar-like Object Detection functions                           *
\****************************************************************************************/

%nodefault CvHaarClassifierCascade;
%newobject cvLoadHaarClassifierCascade;

%nodefault CvPOSITObject;
%newobject cvCreatePOSITObject;

%nodefault CvFeatureTree;
%newobject cvCreateFeatureTree;

%nodefault CvLSH;
%newobject cvCreateLSH;
%newobject cvCreateMemoryLSH;



/// This hides all members of the IplImage which OpenCV doesn't use.
%ignore _IplImage::nSize;
%ignore _IplImage::alphaChannel;
%ignore _IplImage::colorModel;
%ignore _IplImage::channelSeq;
%ignore _IplImage::maskROI;
%ignore _IplImage::imageId;
%ignore _IplImage::tileInfo;
%ignore _IplImage::BorderMode;
%ignore _IplImage::BorderConst;
%ignore _IplImage::imageDataOrigin;

/**
 * imageData is hidden because the accessors produced by SWIG are not 
 * working correct. Use imageData_set and imageData_get instead 
 * (they are defined in "imagedata.i")
 */
%ignore _IplImage::imageData;
