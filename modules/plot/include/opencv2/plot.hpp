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
// Copyright (C) 2009-2012, Willow Garage Inc., all rights reserved.
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
//################################################################################
//
//                    Created by Nuno Moutinho
//
//################################################################################

#ifndef _OPENCV_PLOT_H_
#define _OPENCV_PLOT_H_
#ifdef __cplusplus

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>

/*This plot class allows you to easily plot data from a Mat or a vector. You can plot 1D or 2D plots, change the window size and the axis limits. It's simple yet very effective and usefull. //*/

namespace cv
{
    class CV_EXPORTS Plot{

    public:

    Plot(cv::Mat Data, const char * FigureName, int FigureWidth=600, int FigureHeight=400, double XAxisMin=1e8, double XAxisMax=1e8, double YAxisMin=1e8, double YAxisMax=1e8);
    Plot(std::vector<double> Data, const char * FigureName, int FigureWidth=600, int FigureHeight=400, double XAxisMin=1e8, double XAxisMax=1e8, double YAxisMin=1e8, double YAxisMax=1e8);
    Plot(cv::Mat Xdata, cv::Mat Ydata, const char * FigureName, int FigureWidth=600, int FigureHeight=400, double XAxisMin=1e8, double XAxisMax=1e8, double YAxisMin=1e8, double YAxisMax=1e8);
    Plot(std::vector<double> Xdata, std::vector<double> Ydata, const char * FigureName, int FigureWidth=600, int FigureHeight=400, double XAxisMin=1e8, double XAxisMax=1e8, double YAxisMin=1e8, double YAxisMax=1e8);

    private:

    void constructorHelper(cv::Mat Xdata, cv::Mat Ydata, const char * FigureName, int FigureWidth, int FigureHeight, double XAxisMin, double XAxisMax, double YAxisMin, double YAxisMax);
    void constructorHelper(cv::Mat Data, const char * FigureName, int FigureWidth, int FigureHeight, double XAxisMin, double XAxisMax, double YAxisMin, double YAxisMax);

    cv::Mat linearInterpolation(double Xa, double Xb, double Ya, double Yb, cv::Mat Xdata);
    void drawAxis(double MinX, double MaxX, double MinY, double MaxY, double ImageXzero, double ImageYzero, double CurrentX, double CurrentY, int NumVecElements, int FigW, int FigH, cv::Mat &FigureRed,    cv::Mat &FigureGreen, cv::Mat &FigureBlue);
    void drawValuesAsText(double Value, int Xloc, int Yloc, int XMargin, int YMargin, cv::Mat &FigureRed, cv::Mat &FigureGreen, cv::Mat &FigureBlue);
    void drawValuesAsText(const char * Text, double Value, int Xloc, int Yloc, int XMargin, int YMargin, cv::Mat &FigureRed, cv::Mat &FigureGreen, cv::Mat &FigureBlue);
    void drawLine(int Xstart, int Xend, int Ystart, int Yend, cv::Mat &FigureRed, cv::Mat &FigureGreen, cv::Mat &FigureBlue, int LineWidth, cv::Point3d RGBLineColor);
};
}

#endif
#endif
