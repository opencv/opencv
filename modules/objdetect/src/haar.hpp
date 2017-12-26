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

/* Haar features calculation */

#ifndef OPENCV_OBJDETECT_HAAR_HPP
#define OPENCV_OBJDETECT_HAAR_HPP

#define CV_HAAR_FEATURE_MAX_LOCAL 3

typedef int sumtype;
typedef double sqsumtype;

typedef struct CvHidHaarFeature
{
    struct
    {
        sumtype *p0, *p1, *p2, *p3;
        float weight;
    }
    rect[CV_HAAR_FEATURE_MAX_LOCAL];
} CvHidHaarFeature;


typedef struct CvHidHaarTreeNode
{
    CvHidHaarFeature feature;
    float threshold;
    int left;
    int right;
} CvHidHaarTreeNode;


typedef struct CvHidHaarClassifier
{
    int count;
    //CvHaarFeature* orig_feature;
    CvHidHaarTreeNode* node;
    float* alpha;
} CvHidHaarClassifier;

#define calc_sumf(rect,offset) \
    static_cast<float>((rect).p0[offset] - (rect).p1[offset] - (rect).p2[offset] + (rect).p3[offset])

namespace cv_haar_avx
{
#if 0 /*CV_TRY_AVX*/
    #define CV_HAAR_USE_AVX 1
#else
    #define CV_HAAR_USE_AVX 0
#endif

#if CV_HAAR_USE_AVX
    // AVX version icvEvalHidHaarClassifier.  Process 8 CvHidHaarClassifiers per call. Check AVX support before invocation!!
    double icvEvalHidHaarClassifierAVX(CvHidHaarClassifier* classifier, double variance_norm_factor, size_t p_offset);
    double icvEvalHidHaarStumpClassifierAVX(CvHidHaarClassifier* classifier, double variance_norm_factor, size_t p_offset);
    double icvEvalHidHaarStumpClassifierTwoRectAVX(CvHidHaarClassifier* classifier, double variance_norm_factor, size_t p_offset);
#endif
}

#endif

/* End of file. */
