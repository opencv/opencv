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

#pragma once

#ifndef _OPENCV_RT_IMGPROC_HPP_
#define _OPENCV_RT_IMGPROC_HPP_

namespace cvRT
{
    public enum class ColorConversionCodes
    {
        COLOR_RGBA2GRAY = cv::COLOR_RGBA2GRAY,
        COLOR_GRAY2RGB  = cv::COLOR_GRAY2RGB
    };

    public enum class ContourRetrievalAlgorithm
    {
        RETR_EXTERNAL = cv::RETR_EXTERNAL,
        RETR_LIST = cv::RETR_LIST,
        RETR_CCOMP = cv::RETR_CCOMP,
        RETR_TREE = cv::RETR_TREE,
        RETR_FLOODFILL = cv::RETR_FLOODFILL
    };

    public enum class ContourApproximationModes
    {
        CHAIN_APPROX_NONE = cv::CHAIN_APPROX_NONE,
        CHAIN_APPROX_SIMPLE = cv::CHAIN_APPROX_SIMPLE,
        CHAIN_APPROX_TC89_L1 = cv::CHAIN_APPROX_TC89_L1,
        CHAIN_APPROX_TC89_KCOS = cv::CHAIN_APPROX_TC89_KCOS
    };

    public ref class ImgProc sealed
    {
    public:
        void cvtColor(Mat^ src, Mat^ dst, ColorConversionCodes conversionCode);        
        void GaussianBlur(Mat^ src, Mat^ dst, Size^ ksize, double sigmaX);
        void GaussianBlur(Mat^ src, Mat^ dst, Size^ ksize, double sigmaX, double sigmaY);
        
        void Canny(Mat^ src, Mat^ dst, double threshold1, double threshold2);
        void Canny(Mat^ src, Mat^ dst, double threshold1, double threshold2, int apertureSize);
        void Canny(Mat^ src, Mat^ dst, double threshold1, double threshold2, int apertureSize, bool L2gradient);
        void EqualizeHist(Mat^ src, Mat^ dst);
        void Ellipse(Mat^ src, Point^ center, Size^ axes, double angle, double start_angle, double end_angle, Scalar^ scalar, int thickness, int line_type, int shift);
        void Circle(Mat^ src, Point^ center, int radius, Scalar^ scalar, int thickness, int line_type, int shift);     
        
	    // Contour		
        void FindContours(Mat^ image, VectorOfVectorOfPoint^ contours, VectorOfVec4i^ hierarchy, ContourRetrievalAlgorithm mode, ContourApproximationModes method, Point^ offset);
        void DrawContours(Mat^ image, VectorOfVectorOfPoint^ contours, int contourIdx, Scalar^ color, int thickness, int lineType, VectorOfVec4i^ hierarchy, int maxLevel, Point^ offset);
    };
}
#endif