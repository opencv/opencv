#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <stdlib.h>

#include "seamless_cloning.hpp"

using namespace std;
using namespace cv;

Mat img0, img1, img2, res, res1, final, final1, blend;

Point point;
int drag = 0;
int destx, desty;

int numpts = 100;
Point* pts = new Point[100];
Point* pts1 = new Point[100];
Point* pts2 = new Point[100];


int var = 0;
int flag = 0;
int flag1 = 0;

int minx,miny,maxx,maxy,lenx,leny;
int minxd,minyd,maxxd,maxyd,lenxd,lenyd;

int channel,num;

float alpha,beta;

float red, green, blue;
void mouseHandler(int , int , int , int, void*);
void mouseHandler1(int , int , int , int, void*);
void mouseHandler(int event, int x, int y, int, void*)
{

	if (event == EVENT_LBUTTONDOWN && !drag)
	{
		if(flag1 == 0)
		{
			if(var==0)
				img1 = img0.clone();
			point = Point(x, y);
			circle(img1,point,2,Scalar(0, 0, 255),-1, 8, 0);
			pts[var] = point;
			var++;
			drag  = 1;
			if(var>1)
				line(img1,pts[var-2], point, Scalar(0, 0, 255), 2, 8, 0);

			imshow("Source", img1);
		}
	}


	if (event == EVENT_LBUTTONUP && drag)
	{
		imshow("Source", img1);

		drag = 0;
	}
	if (event == EVENT_RBUTTONDOWN)
	{
		flag1 = 1;
		img1 = img0.clone();
		for(int i = var; i < numpts ; i++)
			pts[i] = point;

		if(var!=0)
		{
			const Point* pts3[1] = {&pts[0]};
			polylines( img1, pts3, &numpts,1, 1, Scalar(0,0,0), 2, 8, 0);
		}

		for(int i=0;i<var;i++)
		{
			minx = min(minx,pts[i].x);
			maxx = max(maxx,pts[i].x);
			miny = min(miny,pts[i].y);
			maxy = max(maxy,pts[i].y);
		}
		lenx = maxx - minx;
		leny = maxy - miny;

		imshow("Source", img1);
	}

	if (event == EVENT_RBUTTONUP)
	{
		flag = var;

		final = Mat::zeros(img0.size(),CV_8UC3);
		res1 = Mat::zeros(img0.size(),CV_8UC1);
		const Point* pts4[1] = {&pts[0]};

		fillPoly(res1, pts4,&numpts, 1, Scalar(255, 255, 255), 8, 0);

		bitwise_and(img0, img0, final,res1);

		imshow("Source", img1);
		if(num == 4)
		{
			Cloning obj;
			obj.local_color_change(img0,final,res1,blend,num,red,green,blue);

			namedWindow("Color Change Image");
			imshow("Color Change Image", blend);
			waitKey(0);

		}
		else if(num == 5)
		{
			Cloning obj;
			Mat img3 = Mat(img0.size(),CV_8UC1); 
			Mat img4 = Mat(img0.size(),CV_8UC3); 
			cvtColor(img0,img3,COLOR_BGR2GRAY);

			for(int i=0;i<img0.size().height;i++)
				for(int j=0;j<img0.size().width;j++)
				{
					img4.at<uchar>(i,j*3+0) = img3.at<uchar>(i,j);
					img4.at<uchar>(i,j*3+1) = img3.at<uchar>(i,j);
					img4.at<uchar>(i,j*3+2) = img3.at<uchar>(i,j);
				}

			obj.local_color_change(img4,final,res1,blend,num);

			namedWindow("Background Decolor Image");
			imshow("Background Decolor Image", blend);
			waitKey(0);
		}
		else if(num == 6)
		{
			Cloning obj;
			obj.illum_change(img0,final,res1,blend,alpha,beta);

			namedWindow("Illum Change Image");
			imshow("Illum Change Image", blend);
			waitKey(0);

		}

	}
	if (event == EVENT_MBUTTONDOWN)
	{
		for(int i = 0; i < numpts ; i++)
		{
			pts[i].x=0;
			pts[i].y=0;
		}
		var = 0;
		flag1 = 0;
		minx = INT_MAX; miny = INT_MAX; maxx = INT_MIN; maxy = INT_MIN;
		imshow("Source", img0);
		drag = 0;
	}
}


void mouseHandler1(int event, int x, int y, int, void*)
{


	Mat im1;
	minxd = INT_MAX; minyd = INT_MAX; maxxd = INT_MIN; maxyd = INT_MIN;
	im1 = img2.clone();
	if (event == EVENT_LBUTTONDOWN)
	{
		if(flag1 == 1)
		{
			point = Point(x, y);

			for(int i =0; i < numpts;i++)
				pts1[i] = pts[i];

			int tempx;
			int tempy;
			for(int i =0; i < flag; i++)
			{
				tempx = pts1[i+1].x - pts1[i].x;
				tempy = pts1[i+1].y - pts1[i].y;
				if(i==0)
				{
					pts2[i+1].x = point.x + tempx;
					pts2[i+1].y = point.y + tempy;
				}
				else if(i>0)
				{
					pts2[i+1].x = pts2[i].x + tempx;
					pts2[i+1].y = pts2[i].y + tempy;
				}

			}	

			for(int i=flag;i<numpts;i++)
				pts2[i] = pts2[flag-1]; 

			pts2[0] = point;

			const Point* pts5[1] = {&pts2[0]};
			polylines( im1, pts5, &numpts,1, 1, Scalar(0,0,255), 2, 8, 0);

			destx = x;
			desty = y;

			imshow("Destination", im1);
		}
	}
	if (event == EVENT_RBUTTONUP)
	{
		for(int i=0;i<flag;i++)
		{
			minxd = min(minxd,pts2[i].x);
			maxxd = max(maxxd,pts2[i].x);
			minyd = min(minyd,pts2[i].y);
			maxyd = max(maxyd,pts2[i].y);
		}

		if(maxxd > im1.size().width || maxyd > im1.size().height || minxd < 0 || minyd < 0)
		{
			cout << "Index out of range" << endl;
			exit(0);
		}

		final1 = Mat::zeros(img2.size(),CV_8UC3);
		res = Mat::zeros(img2.size(),CV_8UC1);
		for(int i=miny, k=minyd;i<(miny+leny);i++,k++)
			for(int j=minx,l=minxd ;j<(minx+lenx);j++,l++)
			{
				for(int c=0;c<channel;c++)
				{
					final1.at<uchar>(k,l*channel+c) = final.at<uchar>(i,j*channel+c);

				}
			}


		const Point* pts6[1] = {&pts2[0]};
		fillPoly(res, pts6, &numpts, 1, Scalar(255, 255, 255), 8, 0);

		if(num == 1 || num == 2 || num == 3)
		{
			Cloning obj;
			obj.normal_clone(img2,final1,res,blend,num);
			namedWindow("Cloned Image");
			imshow("Cloned Image", blend);
			waitKey(0);
		}

		for(int i = 0; i < flag ; i++)
		{
			pts2[i].x=0;
			pts2[i].y=0;
		}

		minxd = INT_MAX; minyd = INT_MAX; maxxd = INT_MIN; maxyd = INT_MIN;
	}

	im1.release();
}

void cv::seamlessClone(InputArray _src, InputArray _dst, OutputArray _blend, int flags)
{
	Mat src  = _src.getMat();
	Mat dest = _dst.getMat();
	_blend.create(dest.size(), CV_8UC3);
	blend = _blend.getMat();

	num = flags;

	minx = INT_MAX; miny = INT_MAX; maxx = INT_MIN; maxy = INT_MIN;

	minxd = INT_MAX; minyd = INT_MAX; maxxd = INT_MIN; maxyd = INT_MIN;

	img0 = src;
	img2 = dest;

	channel = img0.channels();

	res = Mat::zeros(img2.size(),CV_8UC1);
	res1 = Mat::zeros(img0.size(),CV_8UC1);
	final = Mat::zeros(img0.size(),CV_8UC3);
	final1 = Mat::zeros(img2.size(),CV_8UC3);
	//////////// source image ///////////////////

	namedWindow("Source", 1);
	setMouseCallback("Source", mouseHandler, NULL);
	imshow("Source", img0);

	/////////// destination image ///////////////

	namedWindow("Destination", 1);
	setMouseCallback("Destination", mouseHandler1, NULL);
	imshow("Destination",img2);
	waitKey(0);

	img0.release();
	img1.release();
	img2.release();
}

void cv::colorChange(InputArray _src, OutputArray _dst, int flags, float r, float g, float b)
{
	Mat src  = _src.getMat();
	_dst.create(src.size(), src.type());
	blend = _dst.getMat();
	
	minx = INT_MAX; miny = INT_MAX; maxx = INT_MIN; maxy = INT_MIN;

	minxd = INT_MAX; minyd = INT_MAX; maxxd = INT_MIN; maxyd = INT_MIN;

	num = flags;
	red = r;
	green = g;
	blue = b;

	img0 = src;
	res1 = Mat::zeros(img0.size(),CV_8UC1);
	final = Mat::zeros(img0.size(),CV_8UC3);

	namedWindow("Source");
	setMouseCallback("Source", mouseHandler, NULL);
	imshow("Source", img0);

	waitKey(0);

	img0.release();
}


void cv::illuminationChange(InputArray _src, OutputArray _dst, float a, float b)
{

	Mat src  = _src.getMat();
	_dst.create(src.size(), src.type());
	blend = _dst.getMat();
	num = 6;
	alpha = a;
	beta = b;

	img0 = src;

	res1 = Mat::zeros(img0.size(),CV_8UC1);
	final = Mat::zeros(img0.size(),CV_8UC3);

	namedWindow("Source");
	setMouseCallback("Source", mouseHandler, NULL);
	imshow("Source", img0);

	waitKey(0);
}
void cv::textureFlattening(InputArray _src, OutputArray _dst)
{

	Mat src  = _src.getMat();
	_dst.create(src.size(), src.type());
	blend = _dst.getMat();
	img0 = src;

	Cloning obj;
	obj.texture_flatten(img0,blend);

	namedWindow("Texture Flattened Image");
	imshow("Texture Flattened Image", blend);
	waitKey(0);

}

