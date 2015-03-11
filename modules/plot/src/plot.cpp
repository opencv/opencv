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

#include "precomp.hpp"

using namespace cv;
using namespace std;


Plot::Plot(cv::Mat Data, const char * FigureName, int FigureWidth, int FigureHeight, double XAxisMin, double XAxisMax, double YAxisMin, double YAxisMax){
        constructorHelper(Data, FigureName, FigureWidth, FigureHeight, XAxisMin, XAxisMax, YAxisMin, YAxisMax);
}

Plot::Plot(std::vector<double> Data, const char * FigureName, int FigureWidth, int FigureHeight, double XAxisMin, double XAxisMax, double YAxisMin, double YAxisMax){

	Mat DataMat = Mat(Data).clone();

	constructorHelper(DataMat, FigureName, FigureWidth, FigureHeight, XAxisMin, XAxisMax, YAxisMin, YAxisMax);
}

Plot::Plot(std::vector<double> Xdata, std::vector<double> Ydata, const char * FigureName, int FigureWidth, int FigureHeight,
		   double XAxisMin, double XAxisMax, double YAxisMin, double YAxisMax){

	Mat XdataMat = Mat(Xdata).clone();
	Mat YdataMat = Mat(Ydata).clone();

	constructorHelper(XdataMat, YdataMat, FigureName, FigureWidth, FigureHeight, XAxisMin, XAxisMax, YAxisMin, YAxisMax);
}

Plot::Plot(cv::Mat Xdata, cv::Mat Ydata, const char * FigureName, int FigureWidth, int FigureHeight, double XAxisMin, double XAxisMax, double YAxisMin, double YAxisMax){
  constructorHelper(Xdata, Ydata, FigureName, FigureWidth, FigureHeight, XAxisMin, XAxisMax, YAxisMin, YAxisMax);
}


void Plot::constructorHelper(cv::Mat Data, const char * FigureName, int FigureWidth, int FigureHeight, double XAxisMin, double XAxisMax, double YAxisMin, double YAxisMax)
{
	Mat Xdata = Data*0;
	for (int i=0; i<Data.rows; i++){

		Xdata.at<double>(i,0) = i;
	}

	constructorHelper(Xdata, Data, FigureName, FigureWidth, FigureHeight, XAxisMin, XAxisMax, YAxisMin, YAxisMax);
}


void Plot::constructorHelper(cv::Mat Xdata, cv::Mat Ydata, const char * FigureName, int FigureWidth, int FigureHeight, double XAxisMin, double XAxisMax, double YAxisMin, double YAxisMax){

	///Auxiliar variables used in the minMaxLoc function
	cv::Point MinXLoc;
	cv::Point MaxXLoc;
	cv::Point MinYLoc;
	cv::Point MaxYLoc;
	cv::Point MinXLocFindZero;
	cv::Point MaxXLocFindZero;
	cv::Point MinYLocFindZero;
	cv::Point MaxYLocFindZero;

	double MinX;
	double MaxX;
	double MinY;
	double MaxY;
	double MinXFindZero;
	double MaxXFindZero;
	double MinYFindZero;
	double MaxYFindZero;

	int FigW = FigureWidth;
	int FigH = FigureHeight;

	int NumVecElements = Xdata.rows;

	//Obtain the minimum and maximum values of Xdata
	minMaxLoc(Xdata,&MinX,&MaxX,&MinXLoc,&MaxXLoc);

	//Obtain the minimum and maximum values of Ydata
	minMaxLoc(Ydata,&MinY,&MaxY,&MinYLoc,&MaxYLoc);

	if(XAxisMin != 1e8 && XAxisMax != 1e8 && YAxisMin != 1e8 && YAxisMax != 1e8)
	{
		MinX = XAxisMin;
		MaxX = XAxisMax;
		MinY = YAxisMin;
		MaxY = YAxisMax;
	}

	Mat InterpXdata = linearInterpolation(MinX, MaxX, 0, FigW, Xdata);
	Mat InterpYdata = linearInterpolation(MaxY, MinY, 0, FigH, Ydata);

	//Find the zeros in image coordinates
	Mat XdataPlusZero = Mat::zeros(Xdata.rows+1,1,CV_64F);
	Mat YdataPlusZero = Mat::zeros(Ydata.rows+1,1,CV_64F);

	for(int i=0; i<Xdata.rows; i++){
		XdataPlusZero.at<double>(i,0) = Xdata.at<double>(i,0);
		YdataPlusZero.at<double>(i,0) = Ydata.at<double>(i,0);
	}

	minMaxLoc(XdataPlusZero,&MinXFindZero,&MaxXFindZero,&MinXLocFindZero,&MaxXLocFindZero);
	minMaxLoc(YdataPlusZero,&MinYFindZero,&MaxYFindZero,&MinYLocFindZero,&MaxXLocFindZero);

	if(XAxisMin != 1e8 && XAxisMax != 1e8 && YAxisMin != 1e8 && YAxisMax != 1e8)
	{
		MinXFindZero = XAxisMin;
		MaxXFindZero = XAxisMax;
		MinYFindZero = YAxisMin;
		MaxYFindZero = YAxisMax;
	}

	Mat InterpXdataFindZero = linearInterpolation(MinXFindZero, MaxXFindZero, 0, FigW, XdataPlusZero);
	Mat InterpYdataFindZero = linearInterpolation(MaxYFindZero, MinYFindZero, 0, FigH, YdataPlusZero);

	double ImageXzero = InterpXdataFindZero.at<double>(NumVecElements,0);
	double ImageYzero = InterpYdataFindZero.at<double>(NumVecElements,0);

	double CurrentX = Xdata.at<double>(NumVecElements-1,0);
	double CurrentY = Ydata.at<double>(NumVecElements-1,0);

	Mat FigureRed	= Mat::zeros(FigH, FigW, CV_64F);
	Mat FigureGreen = Mat::zeros(FigH, FigW, CV_64F);
	Mat FigureBlue	= Mat::zeros(FigH, FigW, CV_64F);
	std::vector<cv::Mat> FigureVec;
	Mat Figure		= Mat::zeros(FigH, FigW, CV_64F);

	//Draw the plot by connecting lines between the points
	cv::Point p1;
	p1.x = InterpXdata.at<double>(0,0);
	p1.y = InterpYdata.at<double>(0,0);

	drawAxis(MinX, MaxX, MinY, MaxY, ImageXzero,ImageYzero, CurrentX, CurrentY, NumVecElements, FigW, FigH, FigureRed, FigureGreen, FigureBlue);

	for (int r=1; r<InterpXdata.rows; r++){

		cv::Point p2;
		p2.x = InterpXdata.at<double>(r,0);
		p2.y = InterpYdata.at<double>(r,0);

		line(FigureRed, p1, p2, 255, 2, 8, 0);
		line(FigureGreen, p1, p2, 255, 2, 8, 0);
		line(FigureBlue, p1, p2, 0, 2, 8, 0);

		p1 = p2;

	}

	FigureVec.push_back(FigureBlue);
	FigureVec.push_back(FigureGreen);
	FigureVec.push_back(FigureRed);

	merge(FigureVec,Figure);

	//Show the plot
	namedWindow(FigureName);
	imshow(FigureName, Figure);

	waitKey(5);

}

void Plot::drawAxis(double /*MinX*/, double /*MaxX*/, double /*MinY*/, double /*MaxY*/, double ImageXzero, double ImageYzero, double CurrentX, double CurrentY, int /*NumVecElements*/, int FigW, int FigH, cv::Mat &FigureRed, cv::Mat &FigureGreen, cv::Mat &FigureBlue){

	drawValuesAsText(0, ImageXzero, ImageYzero, 10, 20, FigureRed, FigureGreen, FigureBlue);
	drawValuesAsText(0, ImageXzero, ImageYzero, -20, 20, FigureRed, FigureGreen, FigureBlue);
	drawValuesAsText(0, ImageXzero, ImageYzero, 10, -10, FigureRed, FigureGreen, FigureBlue);
	drawValuesAsText(0, ImageXzero, ImageYzero, -20, -10, FigureRed, FigureGreen, FigureBlue);
	drawValuesAsText("X = %g",CurrentX, 0, 0, 40, 20, FigureRed, FigureGreen, FigureBlue);
	drawValuesAsText("Y = %g",CurrentY, 0, 20, 40, 20, FigureRed, FigureGreen, FigureBlue);

	//Horizontal X axis and equispaced horizontal lines
	int LineSpace = 50;
	int TraceSize = 5;
	cv::Point3d RGBLineColor;
	RGBLineColor.x = 255;
	RGBLineColor.y = 0;
	RGBLineColor.z = 0;
	drawLine(0, FigW, ImageYzero, ImageYzero, FigureRed, FigureGreen, FigureBlue, 1, RGBLineColor);

	RGBLineColor.x = 255;
	RGBLineColor.y = 255;
	RGBLineColor.z = 255;
	for(int i=-FigH; i<FigH; i=i+LineSpace){

		if(i!=0){
			int Trace=0;
			while(Trace<FigW){
				drawLine(Trace, Trace+TraceSize, ImageYzero+i, ImageYzero+i, FigureRed, FigureGreen, FigureBlue, 1, RGBLineColor);
				Trace = Trace+2*TraceSize;
			}
		}
	}


	//Vertical Y axis
	RGBLineColor.x = 255;
	RGBLineColor.y = 0;
	RGBLineColor.z = 0;
	drawLine(ImageXzero, ImageXzero, 0, FigH, FigureRed, FigureGreen, FigureBlue, 1, RGBLineColor);

	RGBLineColor.x = 255;
	RGBLineColor.y = 255;
	RGBLineColor.z = 255;
	for(int i=-FigW; i<FigW; i=i+LineSpace){

		if(i!=0){
			int Trace=0;
			while(Trace<FigH){
				drawLine(ImageXzero+i, ImageXzero+i, Trace, Trace+TraceSize, FigureRed, FigureGreen, FigureBlue, 1, RGBLineColor);
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

void Plot::drawValuesAsText(double Value, int Xloc, int Yloc, int XMargin, int YMargin, Mat &FigureRed, Mat &FigureGreen, Mat &FigureBlue){

	char AxisX_Min_Text[20];
	int TextSize = 1;

	sprintf(AxisX_Min_Text, "%g", Value);
	cv::Point AxisX_Min_Loc;
	AxisX_Min_Loc.x = Xloc+XMargin;
	AxisX_Min_Loc.y = Yloc+YMargin;
	putText(FigureRed,AxisX_Min_Text, AxisX_Min_Loc, FONT_HERSHEY_COMPLEX_SMALL, TextSize, 255, 1, 8);
	putText(FigureGreen,AxisX_Min_Text, AxisX_Min_Loc, FONT_HERSHEY_COMPLEX_SMALL, TextSize, 255, 1, 8);
	putText(FigureBlue,AxisX_Min_Text, AxisX_Min_Loc, FONT_HERSHEY_COMPLEX_SMALL, TextSize, 255, 1, 8);

}

void Plot::drawValuesAsText(const char *Text, double Value, int Xloc, int Yloc, int XMargin, int YMargin, Mat &FigureRed, Mat &FigureGreen, Mat &FigureBlue){

	char AxisX_Min_Text[20];
	int TextSize = 1;

	sprintf(AxisX_Min_Text, Text, Value);
	cv::Point AxisX_Min_Loc;
	AxisX_Min_Loc.x = Xloc+XMargin;
	AxisX_Min_Loc.y = Yloc+YMargin;
	putText(FigureRed,AxisX_Min_Text, AxisX_Min_Loc, FONT_HERSHEY_COMPLEX_SMALL, TextSize, 255, 1, 8);
	putText(FigureGreen,AxisX_Min_Text, AxisX_Min_Loc, FONT_HERSHEY_COMPLEX_SMALL, TextSize, 255, 1, 8);
	putText(FigureBlue,AxisX_Min_Text, AxisX_Min_Loc, FONT_HERSHEY_COMPLEX_SMALL, TextSize, 255, 1, 8);

}


void Plot::drawLine(int Xstart, int Xend, int Ystart, int Yend, Mat &FigureRed, Mat &FigureGreen, Mat &FigureBlue, int LineWidth, cv::Point3d RGBLineColor){

	cv::Point Axis_start;
	cv::Point Axis_end;
	Axis_start.x = Xstart;
	Axis_start.y = Ystart;
	Axis_end.x = Xend;
	Axis_end.y = Yend;

	int RedColor = RGBLineColor.x;
	int GreenColor = RGBLineColor.y;
	int BlueColor = RGBLineColor.z;

	line(FigureRed, Axis_start, Axis_end, RedColor, LineWidth, 8, 0);
	line(FigureGreen, Axis_start, Axis_end, GreenColor, LineWidth, 8, 0);
	line(FigureBlue, Axis_start, Axis_end, BlueColor, LineWidth, 8, 0);

}
