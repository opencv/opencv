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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"

using namespace cv;
using namespace std;

///render the plotResult to a Mat
void Plot::render(cv::Mat &_plotResult){

    ///create the plot result
    plotResult = cv::Mat::zeros(plotSizeHeight, plotSizeWidth, CV_8UC3);

    int NumVecElements = plotDataX.rows;

    Mat InterpXdata = linearInterpolation(plotMinX, plotMaxX, 0, plotSizeWidth, plotDataX);
    Mat InterpYdata = linearInterpolation(plotMinY, plotMaxY, 0, plotSizeHeight, plotDataY);

    ///Find the zeros in image coordinates
    Mat InterpXdataFindZero = linearInterpolation(plotMinX_plusZero, plotMaxX_plusZero, 0, plotSizeWidth, plotDataX_plusZero);
    Mat InterpYdataFindZero = linearInterpolation(plotMinY_plusZero, plotMaxY_plusZero, 0, plotSizeHeight, plotDataY_plusZero);

    int ImageXzero = (int)InterpXdataFindZero.at<double>(NumVecElements,0);
    int ImageYzero = (int)InterpYdataFindZero.at<double>(NumVecElements,0);

    double CurrentX = plotDataX.at<double>(NumVecElements-1,0);
    double CurrentY = plotDataY.at<double>(NumVecElements-1,0);

    //Draw the plot by connecting lines between the points
    cv::Point p1;
    p1.x = (int)InterpXdata.at<double>(0,0);
    p1.y = (int)InterpYdata.at<double>(0,0);

    drawAxis(ImageXzero,ImageYzero, CurrentX, CurrentY, plotAxisColor, plotGridColor);

    for (int r=1; r<InterpXdata.rows; r++){

        cv::Point p2;
        p2.x = (int)InterpXdata.at<double>(r,0);
        p2.y = (int)InterpYdata.at<double>(r,0);

        line(plotResult, p1, p2, plotLineColor, plotLineWidth, 8, 0);

        p1 = p2;

    }

    _plotResult = plotResult.clone();

}

///show the plotResult from within the class
void Plot::show(const char * _plotName)
{
    namedWindow(_plotName);
    imshow(_plotName, plotResult);
    waitKey(5);
}

///save the plotResult as a .png image
void Plot::save(const char * _plotFileName)
{
    imwrite(_plotFileName, plotResult);
}

void Plot::drawAxis(int ImageXzero, int ImageYzero, double CurrentX, double CurrentY, Scalar axisColor, Scalar gridColor){

    drawValuesAsText(0, ImageXzero, ImageYzero, 10, 20);
    drawValuesAsText(0, ImageXzero, ImageYzero, -20, 20);
    drawValuesAsText(0, ImageXzero, ImageYzero, 10, -10);
    drawValuesAsText(0, ImageXzero, ImageYzero, -20, -10);
    drawValuesAsText("X = %g",CurrentX, 0, 0, 40, 20);
    drawValuesAsText("Y = %g",CurrentY, 0, 20, 40, 20);

    //Horizontal X axis and equispaced horizontal lines
    int LineSpace = 50;
    int TraceSize = 5;
    drawLine(0, plotSizeWidth, ImageYzero, ImageYzero, axisColor);

    for(int i=-plotSizeHeight; i<plotSizeHeight; i=i+LineSpace){

        if(i!=0){
            int Trace=0;
            while(Trace<plotSizeWidth){
                drawLine(Trace, Trace+TraceSize, ImageYzero+i, ImageYzero+i, gridColor);
                Trace = Trace+2*TraceSize;
            }
        }
    }


    //Vertical Y axis
    drawLine(ImageXzero, ImageXzero, 0, plotSizeHeight, axisColor);

    for(int i=-plotSizeWidth; i<plotSizeWidth; i=i+LineSpace){

        if(i!=0){
            int Trace=0;
            while(Trace<plotSizeHeight){
                drawLine(ImageXzero+i, ImageXzero+i, Trace, Trace+TraceSize, gridColor);
                Trace = Trace+2*TraceSize;
            }
        }
    }
}

Mat Plot::linearInterpolation(double Xa, double Xb, double Ya, double Yb, cv::Mat Xdata){

    Mat Ydata = Xdata*0;

    for (int i=0; i<Xdata.rows; i++){

        double X = Xdata.at<double>(i,0);
        Ydata.at<double>(i,0) = int(Ya + (Yb-Ya)*(X-Xa)/(Xb-Xa));

        if(Ydata.at<double>(i,0)<0)
            Ydata.at<double>(i,0)=0;

    }

    return Ydata;

}

void Plot::drawValuesAsText(double Value, int Xloc, int Yloc, int XMargin, int YMargin){

    char AxisX_Min_Text[20];
    double TextSize = 1;

    sprintf(AxisX_Min_Text, "%g", Value);
    cv::Point AxisX_Min_Loc;
    AxisX_Min_Loc.x = Xloc+XMargin;
    AxisX_Min_Loc.y = Yloc+YMargin;

    putText(plotResult,AxisX_Min_Text, AxisX_Min_Loc, FONT_HERSHEY_COMPLEX_SMALL, TextSize, plotTextColor, 1, 8);
}

void Plot::drawValuesAsText(const char *Text, double Value, int Xloc, int Yloc, int XMargin, int YMargin){

    char AxisX_Min_Text[20];
    int TextSize = 1;

    sprintf(AxisX_Min_Text, Text, Value);
    cv::Point AxisX_Min_Loc;
    AxisX_Min_Loc.x = Xloc+XMargin;
    AxisX_Min_Loc.y = Yloc+YMargin;

    putText(plotResult,AxisX_Min_Text, AxisX_Min_Loc, FONT_HERSHEY_COMPLEX_SMALL, TextSize, plotTextColor, 1, 8);

}


void Plot::drawLine(int Xstart, int Xend, int Ystart, int Yend, Scalar lineColor){

    cv::Point Axis_start;
    cv::Point Axis_end;
    Axis_start.x = Xstart;
    Axis_start.y = Ystart;
    Axis_end.x = Xend;
    Axis_end.y = Yend;

    line(plotResult, Axis_start, Axis_end, lineColor, plotLineWidth, 8, 0);

}
