/*M///////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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
#include <iostream>
#include "opencv2/core/utility.hpp"
namespace cv
{
void TickMeter::stop()
{
int64 time = cv::getTickCount();
if (startTime == 0)
return;
++counter;
sumTime += (time - startTime);
startTime = 0;
}

void TickMeter::reset()
{
startTime = 0;
sumTime = 0;
counter = 0;
}

double TickMeter::getFps(int reset_point)
{
static double stat_fps = 0;
int local_counter = (counter%reset_point)+1;
static double static_start = 0;
double local_timesec = getTimeSec() - static_start;
if(local_counter == reset_point)
    {
    static_start = getTimeSec();
    }
double fps = local_counter/local_timesec;
if(std::isnan(fps))
    return stat_fps;
    else stat_fps = fps;
return stat_fps;
}

void TickMeter::incrementCounter(int by)
{
counter += by;
}
}