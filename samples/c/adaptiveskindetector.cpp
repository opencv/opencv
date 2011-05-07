/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install, copy or use the software.
//
// Copyright (C) 2009, Farhad Dadgostar
// Intel Corporation and third party copyrights are property of their respective owners.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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


#include <iostream>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>

void help(char **argv)
{
	std::cout << "\nThis program demonstrates the contributed flesh detector CvAdaptiveSkinDetector which can be found in contrib.cpp\n"
			<< "Usage: " << std::endl <<
		argv[0] << " fileMask firstFrame lastFrame" << std::endl << std::endl <<
		"Example: " << std::endl <<
		argv[0] << " C:\\VideoSequences\\sample1\\right_view\\temp_%05d.jpg  0  1000" << std::endl <<
		"	iterates through temp_00000.jpg  to  temp_01000.jpg" << std::endl << std::endl <<
		"If no parameter specified, this application will try to capture from the default Webcam." << std::endl <<
		"Please note: Background should not contain large surfaces with skin tone." <<
		"\n\n ESC will stop\n"
		"Using OpenCV version %s\n" << CV_VERSION << "\n"
		<< std::endl;
}

class ASDFrameHolder
{
private:
	IplImage *image;
	double timeStamp;

public:
	ASDFrameHolder();
	virtual ~ASDFrameHolder();
	virtual void assignFrame(IplImage *sourceImage, double frameTime);
	inline IplImage *getImage();
	inline double getTimeStamp();
	virtual void setImage(IplImage *sourceImage);
};

class ASDFrameSequencer
{
public:
	virtual ~ASDFrameSequencer();
	virtual IplImage *getNextImage();
	virtual void close();
	virtual bool isOpen();
	virtual void getFrameCaption(char *caption);
};

class ASDCVFrameSequencer : public ASDFrameSequencer
{
protected:
	CvCapture *capture;

public:
	virtual IplImage *getNextImage();
	virtual void close();
	virtual bool isOpen();
};

class ASDFrameSequencerWebCam : public ASDCVFrameSequencer
{
public:
	virtual bool open(int cameraIndex);
};

class ASDFrameSequencerVideoFile : public ASDCVFrameSequencer
{
public:
	virtual bool open(const char *fileName);
};

class ASDFrameSequencerImageFile : public ASDFrameSequencer {
private:
	char sFileNameMask[2048];
	int nCurrentIndex, nStartIndex, nEndIndex;

public:
	virtual void open(const char *fileNameMask, int startIndex, int endIndex);
	virtual void getFrameCaption(char *caption);
	virtual IplImage *getNextImage();
	virtual void close();
	virtual bool isOpen();
};

//-------------------- ASDFrameHolder -----------------------//
ASDFrameHolder::ASDFrameHolder( )
{
	image = NULL;
	timeStamp = 0;
};

ASDFrameHolder::~ASDFrameHolder( )
{
	cvReleaseImage(&image);
};

void ASDFrameHolder::assignFrame(IplImage *sourceImage, double frameTime)
{
	if (image != NULL)
	{
		cvReleaseImage(&image);
		image = NULL;
	}

	image = cvCloneImage(sourceImage);
	timeStamp = frameTime;
};

IplImage *ASDFrameHolder::getImage()
{
	return image;
};

double ASDFrameHolder::getTimeStamp()
{
	return timeStamp;
};

void ASDFrameHolder::setImage(IplImage *sourceImage)
{
	image = sourceImage;
};


//-------------------- ASDFrameSequencer -----------------------//

ASDFrameSequencer::~ASDFrameSequencer()
{
	close();
};

IplImage *ASDFrameSequencer::getNextImage()
{
	return NULL;
};

void ASDFrameSequencer::close()
{

};

bool ASDFrameSequencer::isOpen()
{
	return false;
};

void ASDFrameSequencer::getFrameCaption(char* /*caption*/) {
	return;
};

IplImage* ASDCVFrameSequencer::getNextImage()
{
	IplImage *image;

	image = cvQueryFrame(capture);

	if (image != NULL)
	{
		return cvCloneImage(image);
	}
	else
	{
		return NULL;
	}
};

void ASDCVFrameSequencer::close()
{
	if (capture != NULL)
	{
		cvReleaseCapture(&capture);
	}
};

bool ASDCVFrameSequencer::isOpen()
{
	return (capture != NULL);
};


//-------------------- ASDFrameSequencerWebCam -----------------------//

bool ASDFrameSequencerWebCam::open(int cameraIndex)
{
	close();

	capture = cvCaptureFromCAM(cameraIndex);

	if (!capture)
	{
		return false;
	}
	else
	{
		return true;
	}
};


//-------------------- ASDFrameSequencerVideoFile -----------------------//

bool ASDFrameSequencerVideoFile::open(const char *fileName)
{
	close();

	capture = cvCaptureFromFile(fileName);
	if (!capture)
	{
		return false;
	}
	else
	{
		return true;
	}
};


//-------------------- ASDFrameSequencerImageFile -----------------------//

void ASDFrameSequencerImageFile::open(const char *fileNameMask, int startIndex, int endIndex)
{
	nCurrentIndex = startIndex-1;
	nStartIndex = startIndex;
	nEndIndex = endIndex;

	std::sprintf(sFileNameMask, "%s", fileNameMask);
};

void ASDFrameSequencerImageFile::getFrameCaption(char *caption) {
	std::sprintf(caption, sFileNameMask, nCurrentIndex);
};

IplImage* ASDFrameSequencerImageFile::getNextImage()
{
	char fileName[2048];

	nCurrentIndex++;

	if (nCurrentIndex > nEndIndex)
		return NULL;

	std::sprintf(fileName, sFileNameMask, nCurrentIndex);

	IplImage* img = cvLoadImage(fileName);

	return img;
};

void ASDFrameSequencerImageFile::close()
{
	nCurrentIndex = nEndIndex+1;
};

bool ASDFrameSequencerImageFile::isOpen()
{
	return (nCurrentIndex <= nEndIndex);
};

void putTextWithShadow(IplImage *img, const char *str, CvPoint point, CvFont *font, CvScalar color = CV_RGB(255, 255, 128))
{
	cvPutText(img, str, cvPoint(point.x-1,point.y-1), font, CV_RGB(0, 0, 0));
	cvPutText(img, str, point, font, color);
};

#define ASD_RGB_SET_PIXEL(pointer, r, g, b)	{ (*pointer) = (unsigned char)b; (*(pointer+1)) = (unsigned char)g;	(*(pointer+2)) = (unsigned char)r; }

#define ASD_RGB_GET_PIXEL(pointer, r, g, b) {b = (unsigned char)(*(pointer)); g = (unsigned char)(*(pointer+1)); r = (unsigned char)(*(pointer+2));}

void displayBuffer(IplImage *rgbDestImage, IplImage *buffer, int rValue, int gValue, int bValue)
{
	int x, y, nWidth, nHeight;
	double destX, destY, dx, dy;
	uchar c;
	unsigned char *pSrc;

	nWidth = buffer->width;
	nHeight = buffer->height;

	dx = double(rgbDestImage->width)/double(nWidth);
	dy = double(rgbDestImage->height)/double(nHeight);

	destX = 0;
	for (x = 0; x < nWidth; x++)
	{
		destY = 0;
		for (y = 0; y < nHeight; y++)
		{
			c = ((uchar*)(buffer->imageData + buffer->widthStep*y))[x];

			if (c)
			{
				pSrc = (unsigned char *)rgbDestImage->imageData + rgbDestImage->widthStep*int(destY) + (int(destX)*rgbDestImage->nChannels);
				ASD_RGB_SET_PIXEL(pSrc, rValue, gValue, bValue);
			}
			destY += dy;
		}
		destY = 0;
		destX += dx;
	}
};

int main(int argc, char** argv )
{
	IplImage *img, *filterMask = NULL;
	CvAdaptiveSkinDetector filter(1, CvAdaptiveSkinDetector::MORPHING_METHOD_ERODE_DILATE);
	ASDFrameSequencer *sequencer;
	CvFont base_font;
	char caption[2048], s[256], windowName[256];
	long int clockTotal = 0, numFrames = 0;
	std::clock_t clock;

	if (argc < 4)
	{
		help(argv);
		sequencer = new ASDFrameSequencerWebCam();
		(dynamic_cast<ASDFrameSequencerWebCam*>(sequencer))->open(-1);

		if (! sequencer->isOpen())
		{
			std::cout << std::endl << "Error: Cannot initialize the default Webcam" << std::endl << std::endl;
		}
	}
	else
	{
		sequencer = new ASDFrameSequencerImageFile();
		(dynamic_cast<ASDFrameSequencerImageFile*>(sequencer))->open(argv[1], std::atoi(argv[2]), std::atoi(argv[3]) ); // A sequence of images captured from video source, is stored here

	}
	std::sprintf(windowName, "%s", "Adaptive Skin Detection Algorithm for Video Sequences");

	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
	cvInitFont( &base_font, CV_FONT_VECTOR0, 0.5, 0.5);

	// Usage:
	//		c:\>CvASDSample "C:\VideoSequences\sample1\right_view\temp_%05d.jpg" 0 1000

	std::cout << "Press ESC to stop." << std::endl << std::endl;
	while ((img = sequencer->getNextImage()) != 0)
	{
		numFrames++;

		if (filterMask == NULL)
		{
			filterMask = cvCreateImage( cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
		}
		clock = std::clock();
		filter.process(img, filterMask);	// DETECT SKIN
		clockTotal += (std::clock() - clock);

		displayBuffer(img, filterMask, 0, 255, 0);

		sequencer->getFrameCaption(caption);
		std::sprintf(s, "%s - %d x %d", caption, img->width, img->height);
		putTextWithShadow(img, s, cvPoint(10, img->height-35), &base_font);

		std::sprintf(s, "Average processing time per frame: %5.2fms", (double(clockTotal*1000/CLOCKS_PER_SEC))/numFrames);
		putTextWithShadow(img, s, cvPoint(10, img->height-15), &base_font);

		cvShowImage (windowName, img);
		cvReleaseImage(&img);

		if (cvWaitKey(1) == 27)
			break;
	}

	sequencer->close();
	delete sequencer;

	cvReleaseImage(&filterMask);

	cvDestroyWindow(windowName);

	std::cout << "Finished, " << numFrames << " frames processed." << std::endl;

	return 0;
}

