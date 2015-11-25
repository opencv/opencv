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

#ifndef _OPENCV_RT_CASCADE_CLASSIFIER_HPP_
#define _OPENCV_RT_CASCADE_CLASSIFIER_HPP_

namespace cvRT
{
    public enum class CV_HAAR
    {
        DO_CANNY_PRUNING = CV_HAAR_DO_CANNY_PRUNING,
        SCALE_IMAGE = CV_HAAR_SCALE_IMAGE,
        FIND_BIGGEST_OBJECT = CV_HAAR_FIND_BIGGEST_OBJECT,
        DO_ROUGH_SEARCH = CV_HAAR_DO_ROUGH_SEARCH
    };    
    
    public enum class CASCADE_FLAG
    {      
        CASCADE_DO_CANNY_PRUNING = 1,
        CASCADE_SCALE_IMAGE = 2,
        CASCADE_FIND_BIGGEST_OBJECT = 4,
        CASCADE_DO_ROUGH_SEARCH = 8
    };    

    public ref class CascadeClassifier sealed
    {
    public:
        CascadeClassifier();
        
        bool load(String^ fileName);
           
        void detectMultiScale(
            Mat^ image,
            IVector<Rect^>^ objects,
            double scaleFactor,
            int minNeighbors, 
            int flags,
            Size^ minSize);      

    protected:


    private:
        cv::CascadeClassifier casCla;
    };
}
#endif