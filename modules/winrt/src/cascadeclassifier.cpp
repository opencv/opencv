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
#include <locale>
#include <codecvt>
#include "opencvrt/cascadeclassifier.hpp"

using namespace Windows::System;
using namespace cvRT;

CascadeClassifier::CascadeClassifier()
{
}

// Map to cv::CascadeClassifier.Load()
bool CascadeClassifier::load(String^ fileName)
{      
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::string filenameNative = converter.to_bytes(fileName->Data());    
    return casCla.load(cv::String(filenameNative));    
}

// Map to cv::CascadeClassifier.detectMultiScale()
void CascadeClassifier::detectMultiScale(Mat^ image, IVector<Rect^>^ objects, double scaleFactor, int minNeighbors, int flags, Size^ minSize)
{
    std::vector<cv::Rect> returnedObjects;

    casCla.detectMultiScale(
        image->Get(),
        returnedObjects,
        scaleFactor,
        minNeighbors,
        flags,
        cv::Size(minSize->Width, minSize->Height)); 

    for (size_t i = 0; i < returnedObjects.size(); i++)
    {
        cv::Rect point = returnedObjects[i];
        objects->Append(ref new Rect(point.x, point.y, point.width, point.height));
    }
}
