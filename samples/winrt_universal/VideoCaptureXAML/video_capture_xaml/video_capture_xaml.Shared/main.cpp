// main.cpp

// Copyright (c) Microsoft Open Technologies, Inc.
// All rights reserved.
//
// (3 - clause BSD License)
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
// following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
// following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
// promote products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "pch.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/cap_winrt.hpp>

using namespace cv;

namespace video_capture_xaml {

    void cvMain()
    {
        VideoCapture cam;

        // open the default camera
        cam.open(0);

        Mat edges;
        Mat frame;

        // process frames
        while (1)
        {
            // get a new frame from camera - this is non-blocking per spec
            cam >> frame;

            // don't reprocess the same frame again
            // nb if commented then flashing may occur
            if (!cam.grab()) continue;

            // image processing calculations here
            // nb Mat frame is in RGB24 format (8UC3)

            // select processing type 1 or 2
    #if 0
            // image manipulation example #1
            // write color bar at row 100 for 200 rows
            auto ar = frame.ptr(100);
            int bytesPerPixel = 3;
            int adjust = (int)(((float)30 / 100.0f) * 255.0);
            for (int i = 0; i < 640 * 100 * bytesPerPixel;)
            {
                ar[i++] = adjust;           // R
                i++;                        // G
                ar[i++] = 255 - adjust;     // B
            }
    #else
            // image processing example #2
            // apply 'canny' filter
            cvtColor(frame, edges, COLOR_RGB2GRAY);
            GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
            Canny(edges, edges, 0, 30, 3);
            cvtColor(edges, frame, COLOR_GRAY2RGB);
    #endif
            // important step to get XAML image component updated
            winrt_imshow();
        }
    }
}