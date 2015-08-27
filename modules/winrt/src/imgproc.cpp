// Copyright (c) 2015, Microsoft Open Technologies, Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of Microsoft Open Technologies, Inc. nor the names
//   of its contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "precomp.hpp"
#include "collection.h"
#include "opencvrt/imgproc.hpp"

using namespace cvRT;

//
// proxy for cv::cvtColor()
//
void ImgProc::cvtColor(Mat^ srcImg, Mat^ destImg, ColorConversionCodes conversionCode)
{
    cv::cvtColor(srcImg->Get(), destImg->Get(), (int)conversionCode);
}

//
// proxy for cv::cvtColor()
//
void ImgProc::GaussianBlur(Mat^ src, Mat^ dst, cvRT::Size^ ksize, double sigmaX)
{
    GaussianBlur(src, dst, ksize, sigmaX, 0);
}

//
// proxy for cv::GaussianBlur()
//
void ImgProc::GaussianBlur(Mat ^ src, Mat ^ dst, cvRT::Size^ ksize, double sigmaX, double sigmaY)
{    
    cv::GaussianBlur(src->Get(), dst->Get(), ksize->GetCvSize(), sigmaX, sigmaY);
}

//
// proxy for cv::Canny()
//
void ImgProc::Canny(Mat^ src, Mat^ dst, double threshold1, double threshold2)
{
    Canny(src, dst, threshold1, threshold2, 3, false);
}

//
// proxy for cv::Canny()
//
void ImgProc::Canny(Mat^ src, Mat^ dst, double threshold1, double threshold2, int apertureSize)
{
    Canny(src, dst, threshold1, threshold2, apertureSize, false);
}

//
// proxy for cv::Canny()
//
void ImgProc::Canny(Mat^ src, Mat^ dst, double threshold1, double threshold2, int apertureSize, bool L2gradient)
{
    cv::Canny(src->Get(), dst->Get(), threshold1, threshold2, apertureSize, L2gradient);
}

//
// proxy for cv::EqualizeHist()
//
void cvRT::ImgProc::EqualizeHist(Mat^ src, Mat^ dst)
{
    cv::equalizeHist(src->Get(), dst->Get());
}

//
// proxy for cv::ellipse()
//
void cvRT::ImgProc::Ellipse(Mat^ src, Point^ center, cvRT::Size^ axes, double angle, double start_angle, double end_angle, Scalar^ scalar, int thickness, int line_type, int shift)
{
    cv::ellipse(src->Get(), center->Get(), axes->GetCvSize(), angle, start_angle, end_angle, scalar->Get(), thickness, line_type, shift);
}

//
// proxy for cv::circle()
//
void cvRT::ImgProc::Circle(Mat^ src, Point^ center, int radius, Scalar^ scalar, int thickness, int line_type, int shift)
{
    cv::circle(src->Get(), center->Get(), radius, scalar->Get(), thickness, line_type, shift);
}

//
// proxy for cv::findContours()
//
void cvRT::ImgProc::FindContours(Mat^ image, VectorOfVectorOfPoint^ contours, VectorOfVec4i^ hierarchy, ContourRetrievalAlgorithm mode, ContourApproximationModes method, Point^ offset)
{   
    cv::findContours(image->Get(), contours->Get(), hierarchy->Get(), (int)mode, (int)method, offset->Get());
}

//
// proxy for cv::drawContours()
//
void cvRT::ImgProc::DrawContours(Mat^ image, VectorOfVectorOfPoint^ contours, int contourIdx, Scalar^ color, int thickness, int lineType, VectorOfVec4i^ hierarchy, int maxLevel, Point^ offset)
{
    cv::drawContours(image->Get(), contours->Get(), contourIdx, color->Get(), thickness, lineType, hierarchy->Get(), maxLevel, offset->Get());
}