#include "cvtest.h"

#if 0

#define debug //enables showing images.
#define CONSOLEOUTPUT //enables printing rms error and percentage of bad pixels to console.

//#define LOAD //enables skipping computing disparities and load them from images.
//#define SAVEIMAGES //enables saving computed disparity and red-marked disparity images.

void MarkPixel(const IplImage* markedDisparity, const int h, const int w)
{
	uchar* data = (uchar*)&markedDisparity->imageData[h*markedDisparity->widthStep + w*3];
	data[0] = 0;
	data[1] = 0;
	data[2] = 255;
}

int CalculateErrors(const IplImage* disparity,const IplImage* groundTruth, IplImage* markedDisparity,
					  double &rms_error, double &percentage_of_bad_pixels, 
					  const int maxDisparity CV_DEFAULT(16), const int eval_ignore_border CV_DEFAULT(10))
{
	if (disparity->width != groundTruth->width)
		return CvTS::FAIL_INVALID_TEST_DATA;
	if (disparity->height != groundTruth->height)
		return CvTS::FAIL_INVALID_TEST_DATA;

	const double eval_bad_thresh = 1.0;

	char* DC = disparity->imageData;
	char* DT = groundTruth->imageData;

	double currSum = 0;
	unsigned int bad_pixels_counter=0;
	
	double diff=0;

	int w = disparity->width;
	int h = disparity->height;
	unsigned int numPixels = w*h;
	
	for(int i=eval_ignore_border; i<h-eval_ignore_border; i++)
		for(int j=eval_ignore_border; j<w-eval_ignore_border; j++)
		{
			diff = (double)abs(DC[i*disparity->widthStep+j] - DT[i*groundTruth->widthStep+j])/(double)maxDisparity;
			currSum += diff*diff;

			if ( diff > eval_bad_thresh )
			{
				bad_pixels_counter++;		
				MarkPixel(markedDisparity, i, j);
			}
		}

	currSum /=(double)numPixels;
	rms_error = sqrt(currSum);

	percentage_of_bad_pixels = (double)bad_pixels_counter/(double)numPixels * 100;

	return 0;
}

class CV_StereoCorrespondenceTestGC : public CvTest
{
public:
    CV_StereoCorrespondenceTestGC();
protected:
    void run(int);
};

	
CV_StereoCorrespondenceTestGC::CV_StereoCorrespondenceTestGC():
CvTest( "stereo-gc", "cvFindStereoCorrespondenceGC" )
{
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
}

/* ///////////////////// stereo_correspondece_test ///////////////////////// */
void CV_StereoCorrespondenceTestGC::run( int )
{
	int code = CvTS::OK;

	const double rms_error_thresh = 1000.0; 
	const double percentage_of_bad_pixels_thresh = 90.0; 

	double rms_error[2];
	double percentage_of_bad_pixels[2];

	/* test parameters */
    char   filepath[1000];
    char   filename[1000];
	//char   extension[5];

	IplImage* left ;
	IplImage* right;
	IplImage* disparity_left;
	IplImage* disparity_right;
	IplImage* groundTruthLeft;
	IplImage* groundTruthRight;

    sprintf( filepath, "%sstereocorrespondence/", ts->get_data_path() );
    sprintf( filename, "%sstereocorrespondence_list.txt", filepath );

	FILE* f = fopen(filename,"r");
	int numImages=0;
	fscanf(f,"%d\n",&numImages);

	for(int i=0; i<numImages; i++)
	{
		/*Load left and right image from the storage*/
		char dataName[100];
		int maxDisparity=0;
		
		fscanf(f,"%s %d\n",dataName,&maxDisparity);
		sprintf(filename,"%s%sL.png",filepath,dataName);
		left =  cvLoadImage(filename,0);
		sprintf(filename,"%s%sR.png",filepath,dataName);
		right = cvLoadImage(filename,0);

		if (!left || !right)
		{
			ts->printf( CvTS::LOG, "Left or right image doesn't exist" );
			code = CvTS::FAIL_MISSING_TEST_DATA;
			goto _exit_;
		}		
		if ((cvGetSize(left).height != cvGetSize(right).height) 
			|| ((cvGetSize(left).width != cvGetSize(right).width)))
		{
			ts->printf( CvTS::LOG, "Left and right image sizes aren't equal" );
			code = CvTS::FAIL_MISSING_TEST_DATA;
			goto _exit_;
		}

		sprintf(filename,"%s%s_gtL.png",filepath,dataName);
		groundTruthLeft = cvLoadImage(filename,0);		
		sprintf(filename,"%s%s_gtR.png",filepath,dataName);
		groundTruthRight = cvLoadImage(filename,0);
		
		if (!groundTruthLeft && !groundTruthRight)
		{
			ts->printf( CvTS::LOG, "Left and right ground truth images don't exist" );
			code = CvTS::FAIL_MISSING_TEST_DATA;
			goto _exit_;
		}	

		for(int i=0; i<2; i++)
		{
			IplImage*& groundTruth = (i == 0) ? groundTruthLeft : groundTruthRight;
			if (groundTruth)
				if (groundTruth->nChannels != 1)
				{
					IplImage* tmp = groundTruth;
					groundTruth = cvCreateImage(cvGetSize(left),IPL_DEPTH_8U,1);
					cvCvtColor(tmp, groundTruth,CV_BGR2GRAY);
				}
		}

		/*Find disparity map for current image pair*/
#ifndef LOAD
		disparity_left = cvCreateImage( cvGetSize(left), IPL_DEPTH_32S, 1 );
		disparity_right = cvCreateImage( cvGetSize(left), IPL_DEPTH_32S, 1 );
		
		CvStereoGCState* state = cvCreateStereoGCState(maxDisparity, 2);
		cvFindStereoCorrespondenceGC( left, right,
                                  disparity_left, disparity_right, state);

		double scale = 256/maxDisparity ;
		if (!strcmp(dataName,"sawtooth") || !strcmp(dataName,"map") || !strcmp(dataName,"poster")
		|| !strcmp(dataName,"bull") || !strcmp(dataName,"barn1") || !strcmp(dataName,"barn2"))
			scale = 8.0;
		
		IplImage* temp;
		temp = disparity_left;
		disparity_left = cvCreateImage(cvGetSize(temp), IPL_DEPTH_8U,1);
		cvConvertScale(temp, disparity_left, -scale);
		temp = disparity_right;
		disparity_right = cvCreateImage(cvGetSize(temp), IPL_DEPTH_8U,1);
		cvConvertScale(temp, disparity_right, scale );
#endif
#ifdef LOAD
		disparity_left;
		disparity_right;
		sprintf(filename,"%s%s_dLgc.png",filepath,dataName);
		disparity_left = cvLoadImage(filename,0);
		sprintf(filename,"%s%s_dRgc.png",filepath,dataName);
		disparity_right = cvLoadImage(filename,0);
#endif
#ifdef debug
		cvNamedWindow("disparity_left");
		cvNamedWindow("disparity_right");
		cvNamedWindow("ground_truth_left");
		cvNamedWindow("ground_truth_right");

		cvShowImage("disparity_left",disparity_left);
		cvShowImage("disparity_right",disparity_right);
		cvShowImage("ground_truth_left", groundTruthLeft);
		cvShowImage("ground_truth_right", groundTruthRight);
#endif

		/*Calculate RMS error and percentage of bad pixels*/
		IplImage* markedDisparity_left = cvCreateImage(cvGetSize(left), IPL_DEPTH_8U, 3);
		IplImage* markedDisparity_right = cvCreateImage(cvGetSize(left), IPL_DEPTH_8U, 3);
		cvCvtColor(disparity_left,markedDisparity_left,CV_GRAY2RGB);
		cvCvtColor(disparity_right,markedDisparity_right,CV_GRAY2RGB);
		
		int eval_ignore_border = 10;
		if (strcmp(dataName,"tsukuba") == 0)
			eval_ignore_border = 18;

		/*Left*/
        int retcode[2] = {0,0};
		if (groundTruthLeft)
			retcode[0] = CalculateErrors(disparity_left,groundTruthLeft, markedDisparity_left,
										rms_error[0], percentage_of_bad_pixels[0], maxDisparity, eval_ignore_border);
		/*Right*/
		if (groundTruthRight)
			retcode[1] = CalculateErrors(disparity_right,groundTruthRight, markedDisparity_right,
										rms_error[1], percentage_of_bad_pixels[1], maxDisparity, eval_ignore_border);

#ifdef SAVEIMAGES
#ifndef LOAD	
		sprintf(filename,"%s%s_dLgc.png",filepath,dataName);
		cvSaveImage(filename,disparity_left);
		sprintf(filename,"%s%s_dRgc.png",filepath,dataName);
		cvSaveImage(filename,disparity_right);

		sprintf(filename,"%s%s_mdLgc.png",filepath,dataName);
		cvSaveImage(filename,markedDisparity_left);
		sprintf(filename,"%s%s_mdRgc.png",filepath,dataName);
		cvSaveImage(filename,markedDisparity_right);
#endif
#endif
#ifdef debug
		cvNamedWindow("markedDisparity_left");
		cvNamedWindow("markedDisparity_right");
		cvShowImage("markedDisparity_left",markedDisparity_left);
		cvShowImage("markedDisparity_right",markedDisparity_right);
		cvWaitKey(1000);
#endif
		if (retcode[0])
		{
			ts->printf(CvTS::LOG,"Calculation error");
			code = retcode[0];
			//goto _exit_;
		}
		if (retcode[1])
		{
			ts->printf(CvTS::LOG,"Calculation error");
			code = retcode[1];
			//goto _exit_;
		}
#ifdef CONSOLEOUTPUT
		printf("\n%s\n",dataName);
		if (groundTruthLeft)
			printf("L rms error = %f\npercentage of bad pixels = %f\n",
					rms_error[0], percentage_of_bad_pixels[0]);
		if(groundTruthRight)
			printf("R rms error = %f\npercentage of bad pixels = %f\n",
					rms_error[1], percentage_of_bad_pixels[1]);
#endif
		for(int i=0; i<2; i++)
		{
			IplImage* groundTruth = (i == 0) ? groundTruthLeft : groundTruthRight;
			if (groundTruth)
			{
				if (rms_error[i] > rms_error_thresh)
				{
					ts->printf( CvTS::LOG, "Big RMS error" );
					code = CvTS::FAIL_BAD_ACCURACY;
					//goto _exit_;
				}
				if (percentage_of_bad_pixels[i] > percentage_of_bad_pixels_thresh)
				{
					ts->printf( CvTS::LOG, "Big percentage of bad pixels" );
					code = CvTS::FAIL_BAD_ACCURACY;
					//goto _exit_;
				}
			}
		}
	}
_exit_:
		cvReleaseImage(&left);
		cvReleaseImage(&right);
		cvReleaseImage(&disparity_left);
		cvReleaseImage(&disparity_right);
		cvReleaseImage(&groundTruthLeft);
		cvReleaseImage(&groundTruthRight);
#ifndef LOAD
		//cvReleaseStereoCorrespondenceGCState(&stereoMatcher);
#endif
	if( code < 0 )
        ts->set_failed_test_info( code );
}

CV_StereoCorrespondenceTestGC stereo_correspondece_test_gc;

#endif
