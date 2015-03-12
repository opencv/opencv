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
    class CV_EXPORTS Plot
{
    protected:
        cv::Mat plotDataX;
        cv::Mat plotDataY;
        cv::Mat plotDataX_plusZero;
        cv::Mat plotDataY_plusZero;
        const char * plotName;

        ///dimensions and limits of the plot
        int plotSizeWidth;
        int plotSizeHeight;
        double plotMinX;
        double plotMaxX;
        double plotMinY;
        double plotMaxY;
        double plotMinX_plusZero;
        double plotMaxX_plusZero;
        double plotMinY_plusZero;
        double plotMaxY_plusZero;
        int plotLineWidth;

        ///colors of each plot element
        cv::Scalar plotLineColor;
        cv::Scalar plotBackgroundColor;
        cv::Scalar plotAxisColor;
        cv::Scalar plotTextColor;

        ///the final plot result
        cv::Mat plotResult;

    public:

        ///constructor accepting only Y data to be plotted
        Plot(cv::Mat _plotData)
        {
            plotDataY=_plotData;
            plotDataX = plotDataY*0;
            for (int i=0; i<plotDataY.rows; i++)
            {
                plotDataX.at<double>(i,0) = i;
            }

            ///calling the main constructor
            plotHelper(plotDataX, plotDataY);
        }

        ///constructor accepting X data and Y data to be plotted
        Plot(cv::Mat _plotDataX, cv::Mat _plotDataY)
        {
            plotHelper(_plotDataX, _plotDataY);
        }

        ///set functions
        void setMinX(double _plotMinX)
        {
            plotMinX = _plotMinX;
            plotMinX_plusZero = _plotMinX;
        }
        void setMaxX(double _plotMaxX)
        {
            plotMaxX = _plotMaxX;
            plotMaxX_plusZero = _plotMaxX;
        }
        void setMinY(double _plotMinY)
        {
            plotMinY = _plotMinY;
            plotMinY_plusZero = _plotMinY;
        }
        void setMaxY(double _plotMaxY)
        {
            plotMaxY = _plotMaxY;
            plotMaxY_plusZero = _plotMaxY;
        }
        void setPlotLineWidth(int _plotLineWidth)
        {
            plotLineWidth=_plotLineWidth;
        }
        void setPlotLineColor(cv::Scalar _plotLineColor)
        {
            plotLineColor=_plotLineColor;
        }
        void setPlotBackgroundColor(cv::Scalar _plotBackgroundColor)
        {
            plotBackgroundColor=_plotBackgroundColor;
        }
        void setPlotAxisColor(cv::Scalar _plotAxisColor)
        {
            plotAxisColor=_plotAxisColor;
        }
        void setPlotTextColor(cv::Scalar _plotTextColor)
        {
            plotTextColor=_plotTextColor;
        }
        void setPlotSize(int _plotSizeWidth, int _plotSizeHeight)
        {
            if(_plotSizeWidth > 400)
                plotSizeWidth = _plotSizeWidth;
            else
                plotSizeWidth = 400;

            if(_plotSizeHeight > 300)
                plotSizeHeight = _plotSizeHeight;
            else
                plotSizeHeight = 300;
        }

        ///render the plotResult to a Mat
        void render(cv::Mat &_plotResult);

        ///show the plotResult from within the class
        void show(const char * _plotName);

        ///save the plotResult as a .png image
        void save(const char * _plotFileName);

    protected:

        ///a helper method to be used in the constructor
        void plotHelper(cv::Mat _plotDataX, cv::Mat _plotDataY)
        {
            plotDataX=_plotDataX;
            plotDataY=_plotDataY;

            int NumVecElements = plotDataX.rows;

            plotDataX_plusZero = cv::Mat::zeros(NumVecElements+1,1,CV_64F);
            plotDataY_plusZero = cv::Mat::zeros(NumVecElements+1,1,CV_64F);

            for(int i=0; i<NumVecElements; i++){
                plotDataX_plusZero.at<double>(i,0) = plotDataX.at<double>(i,0);
                plotDataY_plusZero.at<double>(i,0) = plotDataY.at<double>(i,0);
            }

            double MinX;
            double MaxX;
            double MinY;
            double MaxY;
            double MinX_plusZero;
            double MaxX_plusZero;
            double MinY_plusZero;
            double MaxY_plusZero;

            ///Obtain the minimum and maximum values of Xdata
            minMaxLoc(plotDataX,&MinX,&MaxX);

            ///Obtain the minimum and maximum values of Ydata
            minMaxLoc(plotDataY,&MinY,&MaxY);

            ///Obtain the minimum and maximum values of Xdata plus zero
            minMaxLoc(plotDataX_plusZero,&MinX_plusZero,&MaxX_plusZero);

            ///Obtain the minimum and maximum values of Ydata plus zero
            minMaxLoc(plotDataY_plusZero,&MinY_plusZero,&MaxY_plusZero);

            ///setting the min and max values for each axis
            plotMinX = MinX;
            plotMaxX = MaxX;
            plotMinY = MinY;
            plotMaxY = MaxY;
            plotMinX_plusZero = MinX_plusZero;
            plotMaxX_plusZero = MaxX_plusZero;
            plotMinY_plusZero = MinY_plusZero;
            plotMaxY_plusZero = MaxY_plusZero;

            ///setting the default size of a plot figure
            setPlotSize(600, 400);

            ///setting the default plot line size
            setPlotLineWidth(1);

            ///setting default colors for the different elements of the plot
            setPlotAxisColor(cv::Scalar(0, 0, 255));
            setPlotBackgroundColor(cv::Scalar(0, 0, 0));
            setPlotLineColor(cv::Scalar(0, 255, 255));
            setPlotTextColor(cv::Scalar(255, 255, 255));
        }

        cv::Mat linearInterpolation(double Xa, double Xb, double Ya, double Yb, cv::Mat Xdata);
        void drawAxis(double ImageXzero, double ImageYzero, double CurrentX, double CurrentY, cv::Scalar axisColor);
        void drawValuesAsText(double Value, int Xloc, int Yloc, int XMargin, int YMargin);
        void drawValuesAsText(const char * Text, double Value, int Xloc, int Yloc, int XMargin, int YMargin);
        void drawLine(int Xstart, int Xend, int Ystart, int Yend, cv::Scalar lineColor);
};
}

#endif
#endif
