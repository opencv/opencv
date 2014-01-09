/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install, copy or use the software.
//
// Copyright (C) 2009, Farhad Dadgostar
// Intel Corporation and third party copyrights are property of their respective owners.
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

#include "precomp.hpp"

#define ASD_INTENSITY_SET_PIXEL(pointer, qq) {(*pointer) = (unsigned char)qq;}

#define ASD_IS_IN_MOTION(pointer, v, threshold)	((abs((*(pointer)) - (v)) > (threshold)) ? true : false)

void CvAdaptiveSkinDetector::initData(IplImage *src, int widthDivider, int heightDivider)
{
    CvSize imageSize = cvSize(src->width/widthDivider, src->height/heightDivider);

    imgHueFrame = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
    imgShrinked = cvCreateImage(imageSize, IPL_DEPTH_8U, src->nChannels);
    imgSaturationFrame = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
    imgMotionFrame = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
    imgTemp = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
    imgFilteredFrame = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
    imgGrayFrame = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
    imgLastGrayFrame = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
    imgHSVFrame = cvCreateImage(imageSize, IPL_DEPTH_8U, 3);
}

CvAdaptiveSkinDetector::CvAdaptiveSkinDetector(int samplingDivider, int morphingMethod)
{
    nSkinHueLowerBound = GSD_HUE_LT;
    nSkinHueUpperBound = GSD_HUE_UT;

    fHistogramMergeFactor = 0.05;  	// empirical result
    fHuePercentCovered = 0.95;		// empirical result

    nMorphingMethod = morphingMethod;
    nSamplingDivider = samplingDivider;

    nFrameCount = 0;
    nStartCounter = 0;

    imgHueFrame = NULL;
    imgMotionFrame = NULL;
    imgTemp = NULL;
    imgFilteredFrame = NULL;
    imgShrinked = NULL;
    imgGrayFrame = NULL;
    imgLastGrayFrame = NULL;
    imgSaturationFrame = NULL;
    imgHSVFrame = NULL;
}

CvAdaptiveSkinDetector::~CvAdaptiveSkinDetector()
{
    cvReleaseImage(&imgHueFrame);
    cvReleaseImage(&imgSaturationFrame);
    cvReleaseImage(&imgMotionFrame);
    cvReleaseImage(&imgTemp);
    cvReleaseImage(&imgFilteredFrame);
    cvReleaseImage(&imgShrinked);
    cvReleaseImage(&imgGrayFrame);
    cvReleaseImage(&imgLastGrayFrame);
    cvReleaseImage(&imgHSVFrame);
}

void CvAdaptiveSkinDetector::process(IplImage *inputBGRImage, IplImage *outputHueMask)
{
    IplImage *src = inputBGRImage;

    int h, v, i, l;
    bool isInit = false;

    nFrameCount++;

    if (imgHueFrame == NULL)
    {
        isInit = true;
        initData(src, nSamplingDivider, nSamplingDivider);
    }

    unsigned char *pShrinked, *pHueFrame, *pMotionFrame, *pLastGrayFrame, *pFilteredFrame, *pGrayFrame;
    pShrinked = (unsigned char *)imgShrinked->imageData;
    pHueFrame = (unsigned char *)imgHueFrame->imageData;
    pMotionFrame = (unsigned char *)imgMotionFrame->imageData;
    pLastGrayFrame = (unsigned char *)imgLastGrayFrame->imageData;
    pFilteredFrame = (unsigned char *)imgFilteredFrame->imageData;
    pGrayFrame = (unsigned char *)imgGrayFrame->imageData;

    if ((src->width != imgHueFrame->width) || (src->height != imgHueFrame->height))
    {
        cvResize(src, imgShrinked);
        cvCvtColor(imgShrinked, imgHSVFrame, CV_BGR2HSV);
    }
    else
    {
        cvCvtColor(src, imgHSVFrame, CV_BGR2HSV);
    }

    cvSplit(imgHSVFrame, imgHueFrame, imgSaturationFrame, imgGrayFrame, 0);

    cvSetZero(imgMotionFrame);
    cvSetZero(imgFilteredFrame);

    l = imgHueFrame->height * imgHueFrame->width;

    for (i = 0; i < l; i++)
    {
        v = (*pGrayFrame);
        if ((v >= GSD_INTENSITY_LT) && (v <= GSD_INTENSITY_UT))
        {
            h = (*pHueFrame);
            if ((h >= GSD_HUE_LT) && (h <= GSD_HUE_UT))
            {
                if ((h >= nSkinHueLowerBound) && (h <= nSkinHueUpperBound))
                    ASD_INTENSITY_SET_PIXEL(pFilteredFrame, h);

                if (ASD_IS_IN_MOTION(pLastGrayFrame, v, 7))
                    ASD_INTENSITY_SET_PIXEL(pMotionFrame, h);
            }
        }
        pShrinked += 3;
        pGrayFrame++;
        pLastGrayFrame++;
        pMotionFrame++;
        pHueFrame++;
        pFilteredFrame++;
    }

    if (isInit)
        cvCalcHist(&imgHueFrame, skinHueHistogram.fHistogram);

    cvCopy(imgGrayFrame, imgLastGrayFrame);

    cvErode(imgMotionFrame, imgTemp);  // eliminate disperse pixels, which occur because of the camera noise
    cvDilate(imgTemp, imgMotionFrame);

    cvCalcHist(&imgMotionFrame, histogramHueMotion.fHistogram);

    skinHueHistogram.mergeWith(&histogramHueMotion, fHistogramMergeFactor);

    skinHueHistogram.findCurveThresholds(nSkinHueLowerBound, nSkinHueUpperBound, 1 - fHuePercentCovered);

    switch (nMorphingMethod)
    {
        case MORPHING_METHOD_ERODE :
            cvErode(imgFilteredFrame, imgTemp);
            cvCopy(imgTemp, imgFilteredFrame);
            break;
        case MORPHING_METHOD_ERODE_ERODE :
            cvErode(imgFilteredFrame, imgTemp);
            cvErode(imgTemp, imgFilteredFrame);
            break;
        case MORPHING_METHOD_ERODE_DILATE :
            cvErode(imgFilteredFrame, imgTemp);
            cvDilate(imgTemp, imgFilteredFrame);
            break;
    }

    if (outputHueMask != NULL)
        cvCopy(imgFilteredFrame, outputHueMask);
}


//------------------------- Histogram for Adaptive Skin Detector -------------------------//

CvAdaptiveSkinDetector::Histogram::Histogram()
{
    int histogramSize[] = { HistogramSize };
    float range[] = { GSD_HUE_LT, GSD_HUE_UT };
    float *ranges[] = { range };
    fHistogram = cvCreateHist(1, histogramSize, CV_HIST_ARRAY, ranges, 1);
    cvClearHist(fHistogram);
}

CvAdaptiveSkinDetector::Histogram::~Histogram()
{
    cvReleaseHist(&fHistogram);
}

int CvAdaptiveSkinDetector::Histogram::findCoverageIndex(double surfaceToCover, int defaultValue)
{
    double s = 0;
    for (int i = 0; i < HistogramSize; i++)
    {
        s += cvGetReal1D( fHistogram->bins, i );
        if (s >= surfaceToCover)
        {
            return i;
        }
    }
    return defaultValue;
}

void CvAdaptiveSkinDetector::Histogram::findCurveThresholds(int &x1, int &x2, double percent)
{
    double sum = 0;

    for (int i = 0; i < HistogramSize; i++)
    {
        sum += cvGetReal1D( fHistogram->bins, i );
    }

    x1 = findCoverageIndex(sum * percent, -1);
    x2 = findCoverageIndex(sum * (1-percent), -1);

    if (x1 == -1)
        x1 = GSD_HUE_LT;
    else
        x1 += GSD_HUE_LT;

    if (x2 == -1)
        x2 = GSD_HUE_UT;
    else
        x2 += GSD_HUE_LT;
}

void CvAdaptiveSkinDetector::Histogram::mergeWith(CvAdaptiveSkinDetector::Histogram *source, double weight)
{
    float myweight = (float)(1-weight);
    float maxVal1 = 0, maxVal2 = 0, *f1, *f2, ff1, ff2;

    cvGetMinMaxHistValue(source->fHistogram, NULL, &maxVal2);

    if (maxVal2 > 0 )
    {
        cvGetMinMaxHistValue(fHistogram, NULL, &maxVal1);
        if (maxVal1 <= 0)
        {
            for (int i = 0; i < HistogramSize; i++)
            {
                f1 = (float*)cvPtr1D(fHistogram->bins, i);
                f2 = (float*)cvPtr1D(source->fHistogram->bins, i);
                (*f1) = (*f2);
            }
        }
        else
        {
            for (int i = 0; i < HistogramSize; i++)
            {
                f1 = (float*)cvPtr1D(fHistogram->bins, i);
                f2 = (float*)cvPtr1D(source->fHistogram->bins, i);

                ff1 = ((*f1)/maxVal1)*myweight;
                if (ff1 < 0)
                    ff1 = -ff1;

                ff2 = (float)(((*f2)/maxVal2)*weight);
                if (ff2 < 0)
                    ff2 = -ff2;

                (*f1) = (ff1 + ff2);

            }
        }
    }
}
