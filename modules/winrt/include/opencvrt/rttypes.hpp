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
#ifndef _OPENCV_RT_TYPES_HPP_
#define _OPENCV_RT_TYPES_HPP_

namespace cvRT
{    
    // 
    // cv::Scalar    
    //
    public ref class Scalar sealed
    {
    public:
        Scalar(double val0, double val1, double val2, double val3) { cvScalar = cv::Scalar(val0, val1, val2, val3); }        
        Scalar(double val0, double val1, double val2) { cvScalar = cv::Scalar(val0, val1, val2); }
        Scalar(double val0, double val1) { cvScalar = cv::Scalar(val0, val1); }       
        Scalar(double val0) { cvScalar = cv::Scalar(val0); }
    internal:
        cv::Scalar& Get() { return cvScalar; }
    private:
        cv::Scalar cvScalar;
    };

    //
    // cv::Point 
    //
    public ref class Point sealed
    {
    public:
        Point(int p0, int p1) { cvPoint = cv::Point(p0, p1); }
    internal:
        cv::Point& Get() { return cvPoint; }
    private:
        cv::Point cvPoint;
    };

    //
    // cv::Rect 
    //
    public ref class Rect sealed
    {
    public:
        property int X { int get() { return cvRect.x; }}
        property int Y { int get() { return cvRect.y; }}
        property int Width { int get() { return cvRect.width; }}
        property int Height {int get() { return cvRect.height; }}
        Rect(int x, int y, int width, int height) { cvRect = cv::Rect(x, y, width, height); }
    internal:
        cv::Rect& Get() { return cvRect; }
    private:
        cv::Rect cvRect;
    };
    
    //
    // cv::Size
    //
    public ref class Size sealed
    {
    public:
        property int Width { int get() {return cvSize.width; }}
        property int Height { int get() { return cvSize.height; }}
        Size(int width, int height) {cvSize = cv::Size(width, height);}
    internal:
        cv::Size& GetCvSize() { return cvSize; }
    private:
        cv::Size cvSize;
    };  

    //
    //  for cv::InputArray or cv::OutputArray
    //
    public ref class VectorOfPoint sealed
    {
    public:
        VectorOfPoint() { ; }
    internal:
    private:
        std::vector<cv::Point> cvVoPoint;
    };

    //
    // for cv::InputArray or cv::OutputArray
    //
    public ref class VectorOfVec4i sealed
    {
    public:
        VectorOfVec4i() { ; }
    internal:
        std::vector<cv::Vec4i>& Get() { return cvVecV4i; }
    private:
        std::vector<cv::Vec4i> cvVecV4i;
    };
    
    //
    // for cv::InputArrayOfArrays or cv::OutputArrayOfArrays
    // 
    public ref class VectorOfVectorOfPoint sealed
    {
    public:
        VectorOfVectorOfPoint() { ; }
        int Count() { return cvVoVoPoint.size(); }

        // tbd this is equivalent to [] overload.
        void Index(int index) 
        { 
            
        }        
    internal:
        std::vector<std::vector<cv::Point>>& Get() { return cvVoVoPoint; }
    private:
        std::vector<std::vector<cv::Point>> cvVoVoPoint;
    };
}
#endif