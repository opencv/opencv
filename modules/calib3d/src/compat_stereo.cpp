//M*//////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#include "precomp.hpp"

CvStereoBMState* cvCreateStereoBMState( int /*preset*/, int numberOfDisparities )
{
    CvStereoBMState* state = (CvStereoBMState*)cvAlloc( sizeof(*state) );
    if( !state )
        return 0;

    state->preFilterType = CV_STEREO_BM_XSOBEL; //CV_STEREO_BM_NORMALIZED_RESPONSE;
    state->preFilterSize = 9;
    state->preFilterCap = 31;
    state->SADWindowSize = 15;
    state->minDisparity = 0;
    state->numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : 64;
    state->textureThreshold = 10;
    state->uniquenessRatio = 15;
    state->speckleRange = state->speckleWindowSize = 0;
    state->trySmallerWindows = 0;
    state->roi1 = state->roi2 = cvRect(0,0,0,0);
    state->disp12MaxDiff = -1;

    state->preFilteredImg0 = state->preFilteredImg1 = state->slidingSumBuf =
    state->disp = state->cost = 0;

    return state;
}

void cvReleaseStereoBMState( CvStereoBMState** state )
{
    if( !state )
        CV_Error( CV_StsNullPtr, "" );

    if( !*state )
        return;

    cvReleaseMat( &(*state)->preFilteredImg0 );
    cvReleaseMat( &(*state)->preFilteredImg1 );
    cvReleaseMat( &(*state)->slidingSumBuf );
    cvReleaseMat( &(*state)->disp );
    cvReleaseMat( &(*state)->cost );
    cvFree( state );
}

template<> void cv::Ptr<CvStereoBMState>::delete_obj()
{ cvReleaseStereoBMState(&obj); }


void cvFindStereoCorrespondenceBM( const CvArr* leftarr, const CvArr* rightarr,
                                   CvArr* disparr, CvStereoBMState* state )
{
    cv::Mat left = cv::cvarrToMat(leftarr), right = cv::cvarrToMat(rightarr);
    const cv::Mat disp = cv::cvarrToMat(disparr);

    CV_Assert( state != 0 );

    cv::Ptr<cv::StereoMatcher> sm = cv::createStereoBM(state->numberOfDisparities,
                                                       state->SADWindowSize);
    sm->set("preFilterType", state->preFilterType);
    sm->set("preFilterSize", state->preFilterSize);
    sm->set("preFilterCap", state->preFilterCap);
    sm->set("SADWindowSize", state->SADWindowSize);
    sm->set("numDisparities", state->numberOfDisparities > 0 ? state->numberOfDisparities : 64);
    sm->set("textureThreshold", state->textureThreshold);
    sm->set("uniquenessRatio", state->uniquenessRatio);
    sm->set("speckleRange", state->speckleRange);
    sm->set("speckleWindowSize", state->speckleWindowSize);
    sm->set("disp12MaxDiff", state->disp12MaxDiff);

    sm->compute(left, right, disp);
}

CvRect cvGetValidDisparityROI( CvRect roi1, CvRect roi2, int minDisparity,
                              int numberOfDisparities, int SADWindowSize )
{
    return (CvRect)cv::getValidDisparityROI( roi1, roi2, minDisparity,
                                            numberOfDisparities, SADWindowSize );
}

void cvValidateDisparity( CvArr* _disp, const CvArr* _cost, int minDisparity,
                         int numberOfDisparities, int disp12MaxDiff )
{
    cv::Mat disp = cv::cvarrToMat(_disp), cost = cv::cvarrToMat(_cost);
    cv::validateDisparity( disp, cost, minDisparity, numberOfDisparities, disp12MaxDiff );
}

namespace cv
{

StereoBM::StereoBM()
{ init(BASIC_PRESET); }

StereoBM::StereoBM(int _preset, int _ndisparities, int _SADWindowSize)
{ init(_preset, _ndisparities, _SADWindowSize); }

void StereoBM::init(int _preset, int _ndisparities, int _SADWindowSize)
{
    state = cvCreateStereoBMState(_preset, _ndisparities);
    state->SADWindowSize = _SADWindowSize;
}

void StereoBM::operator()( InputArray _left, InputArray _right,
                          OutputArray _disparity, int disptype )
{
    Mat left = _left.getMat(), right = _right.getMat();
    CV_Assert( disptype == CV_16S || disptype == CV_32F );
    _disparity.create(left.size(), disptype);
    Mat disp = _disparity.getMat();

    CvMat left_c = left, right_c = right, disp_c = disp;
    cvFindStereoCorrespondenceBM(&left_c, &right_c, &disp_c, state);
}


StereoSGBM::StereoSGBM()
{
    minDisparity = numberOfDisparities = 0;
    SADWindowSize = 0;
    P1 = P2 = 0;
    disp12MaxDiff = 0;
    preFilterCap = 0;
    uniquenessRatio = 0;
    speckleWindowSize = 0;
    speckleRange = 0;
    fullDP = false;

    sm = createStereoSGBM(0, 0, 0);
}

StereoSGBM::StereoSGBM( int _minDisparity, int _numDisparities, int _SADWindowSize,
                       int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
                       int _uniquenessRatio, int _speckleWindowSize, int _speckleRange,
                       bool _fullDP )
{
    minDisparity = _minDisparity;
    numberOfDisparities = _numDisparities;
    SADWindowSize = _SADWindowSize;
    P1 = _P1;
    P2 = _P2;
    disp12MaxDiff = _disp12MaxDiff;
    preFilterCap = _preFilterCap;
    uniquenessRatio = _uniquenessRatio;
    speckleWindowSize = _speckleWindowSize;
    speckleRange = _speckleRange;
    fullDP = _fullDP;

    sm = createStereoSGBM(0, 0, 0);
}

StereoSGBM::~StereoSGBM()
{
}

void StereoSGBM::operator ()( InputArray _left, InputArray _right,
                             OutputArray _disp )
{
    sm->set("minDisparity", minDisparity);
    sm->set("numDisparities", numberOfDisparities);
    sm->set("SADWindowSize", SADWindowSize);
    sm->set("P1", P1);
    sm->set("P2", P2);
    sm->set("disp12MaxDiff", disp12MaxDiff);
    sm->set("preFilterCap", preFilterCap);
    sm->set("uniquenessRatio", uniquenessRatio);
    sm->set("speckleWindowSize", speckleWindowSize);
    sm->set("speckleRange", speckleRange);
    sm->set("fullDP", fullDP);

    sm->compute(_left, _right, _disp);
}

}



