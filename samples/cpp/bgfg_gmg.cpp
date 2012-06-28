/*
 * FGBGTest.cpp
 *
 *  Created on: May 7, 2012
 *      Author: Andrew B. Godbehere
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>

using namespace cv;

static void help()
{
	std::cout <<
	"\nA program demonstrating the use and capabilities of a particular BackgroundSubtraction\n"
	"algorithm described in A. Godbehere, A. Matsukawa, K. Goldberg, \n"
	"\"Visual Tracking of Human Visitors under Variable-Lighting Conditions for a Responsive\n"
	"Audio Art Installation\", American Control Conference, 2012, used in an interactive\n"
	"installation at the Contemporary Jewish Museum in San Francisco, CA from March 31 through\n"
	"July 31, 2011.\n"
	"Call:\n"
	"./BackgroundSubtractorGMG_sample\n"
	"Using OpenCV version " << CV_VERSION << "\n"<<std::endl;
}

int main(int argc, char** argv)
{
	help();
	setUseOptimized(true);
	setNumThreads(8);

	Ptr<BackgroundSubtractorGMG> fgbg = Algorithm::create<BackgroundSubtractorGMG>("BackgroundSubtractor.GMG");
	if (fgbg == NULL)
	{
		CV_Error(CV_StsError,"Failed to create Algorithm\n");
	}
	fgbg->set("smoothingRadius",7);
	fgbg->set("decisionThreshold",0.7);

	VideoCapture cap;
    if( argc > 1 )
        cap.open(argv[1]);
    else
        cap.open(0);
    
	if (!cap.isOpened())
	{
        std::cout << "error: cannot read video. Try moving video file to sample directory.\n";
		return -1;
	}

	Mat img, downimg, downimg2, fgmask, upfgmask, posterior, upposterior;

	bool first = true;
	namedWindow("posterior");
	namedWindow("fgmask");
	namedWindow("FG Segmentation");
	int i = 0;
	for (;;)
	{
		std::stringstream txt;
		txt << "frame: ";
		txt << i++;

		cap >> img;
		putText(img,txt.str(),Point(20,40),FONT_HERSHEY_SIMPLEX,0.8,Scalar(1.0,0.0,0.0));

		resize(img,downimg,Size(160,120),0,0,INTER_NEAREST);   // Size(cols, rows) or Size(width,height)
		if (first)
		{
			fgbg->initializeType(downimg,0,255);
			first = false;
		}
		if (img.empty())
		{
			return 0;
		}
		(*fgbg)(downimg,fgmask);
		fgbg->updateBackgroundModel(Mat::zeros(120,160,CV_8U));
		fgbg->getPosteriorImage(posterior);
		resize(fgmask,upfgmask,Size(640,480),0,0,INTER_NEAREST);
		Mat coloredFG = Mat::zeros(480,640,CV_8UC3);
		coloredFG.setTo(Scalar(100,100,0),upfgmask);

		resize(posterior,upposterior,Size(640,480),0,0,INTER_NEAREST);
		imshow("posterior",upposterior);
		imshow("fgmask",upfgmask);
        resize(img, downimg2, Size(640, 480),0,0,INTER_LINEAR);
		imshow("FG Segmentation",downimg2 + coloredFG);
        int c = waitKey(30);
        if( c == 'q' || c == 'Q' || (c & 255) == 27 )
			break;
	}
}

