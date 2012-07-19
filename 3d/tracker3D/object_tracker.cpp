/*
 * A Demo of automatic samples capturing
 * Author: Alexey Latyshev
 */


#ifdef WIN32
#include "cv.h"
#include "highgui.h"
#else
#include "opencv/cv.h"
#include "opencv/highgui.h"
#endif
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <ctype.h>
#include "tracker_calibration.h"
#include "tracker3D.h"
#include "object_tracker.h"


int VIDEO_CALIBRATION_DELAY = 1000;
bool AUTOFIND_CHESSBOARD=true;
bool USE_DEINTERLACING=false;
bool DRAW_CORNERS = true;
bool DRAW_CHESSBOARD = true;
bool USE_UNDISTORTION = false;
bool CALIBRATE_CAMERA = false;
bool IS_VIDEO_CAPTURE = false;
bool SAVE_DRAWINGS = false;
bool SAVE_SAMPLES = false;
bool SAVE_ALL_FRAMES = false;
bool SHOW_TEST_SQUARE = false;
bool USE_3D = false;
char* SAMPLES_PATH;
char* INPUT_VIDEO;
char* OUTPUT_DIRECTORY;
char* CAMERA_PARAMETERS_PATH;

CvCapture* capture;
IplImage* frame;
IplImage* workImage;
CvSize innerCornersCount;
CvMat* IntrinsicMatrix;
CvMat* DistortionCoeffs;



// Image deinterlacing
IplImage* Deinterlace(IplImage* src)
{
	IplImage* res = cvCloneImage(src);
	uchar* linea;
	uchar* lineb;
	uchar* linec;

	for (int i = 1; i < res->height-1; i+=2)
	{
		linea = (uchar*)res->imageData + ((i-1) * res->widthStep);
		lineb = (uchar*)res->imageData + ((i) * res->widthStep);
		linec = (uchar*)res->imageData + ((i+1) * res->widthStep);

		for (int j = 0; j < res->width * res->nChannels; j++)
		{
			lineb[j] = (uchar)((linea[j] + linec[j])/2);
		}
	}

	if (res->height > 1 && res->height % 2 == 0)
	{
		linea = (uchar*)res->imageData + ((res->height-2) * res->widthStep);
		lineb = (uchar*)res->imageData + ((res->height-1) * res->widthStep);
		memcpy(lineb, linea, res->width);
	}
	return res;

}

void InverseUndistortMap(const CvMat* mapx,const CvMat* mapy, CvMat** invmapx, CvMat** invmapy, bool interpolate)
{
	*invmapx=cvCreateMat(mapx->rows,mapx->cols,mapx->type);
	*invmapy=cvCreateMat(mapy->rows,mapy->cols,mapy->type);
	int x,y;
	for (int i=0;i<mapx->cols;i++)
		for (int j=0;j<mapy->rows;j++)
		{
			(*invmapx)->data.fl[i+(*invmapx)->cols*j]=-1e20f;
			(*invmapy)->data.fl[i+(*invmapy)->cols*j]=-1e20f;
		}

		for (int i=0;i<mapx->cols;i++)
			for (int j=0;j<mapy->rows;j++)
			{
				x = (int) mapx->data.fl[i+mapx->cols*j];
				y = (int) mapy->data.fl[i+mapy->cols*j];
				if ((x>=0) && (x<(*invmapx)->cols) && (y>=0) && (y < (*invmapy)->rows))
				{
					(*invmapx)->data.fl[x+(*invmapx)->cols*y]=(float)i;
					(*invmapy)->data.fl[x+(*invmapy)->cols*y]=(float)j;
				}
			}
			if (interpolate)
			{
				for (int i=1;i<mapx->cols-1;i++)
					for (int j=1;j<mapy->rows-1;j++)
					{
						if ((*invmapx)->data.fl[i+(*invmapx)->cols*j]==-1e20)
						{
							(*invmapx)->data.fl[i+(*invmapx)->cols*j] = ((*invmapx)->data.fl[i-1+(*invmapx)->cols*j]+(*invmapx)->data.fl[i+1+(*invmapx)->cols*j]
							+ (*invmapx)->data.fl[i+(*invmapx)->cols*(j-1)]+(*invmapx)->data.fl[i+(*invmapx)->cols*(j+1)])/4;
						}
						if ((*invmapy)->data.fl[i+(*invmapy)->cols*j]==-1e20)
						{
							(*invmapy)->data.fl[i+(*invmapy)->cols*j] = ((*invmapy)->data.fl[i-1+(*invmapy)->cols*j]+(*invmapy)->data.fl[i+1+(*invmapy)->cols*j]
							+ (*invmapy)->data.fl[i+(*invmapy)->cols*(j-1)]+(*invmapy)->data.fl[i+(*invmapy)->cols*(j+1)])/4;
						}
					}
			}
}

// left button click sets new point;
// (!!!removed!!!)right button click removes last point
void on_mouse( int event, int x, int y, int flags, void* param )
{
	switch( event )
	{
	case CV_EVENT_LBUTTONUP:
		{

			int n=0;
			//for (n=0;(n<4) && ((*((CvPoint**)param))[n].x != -1);n++);
			for (n=0;(n<4) && (((CvPoint*)param)[n].x != -1);n++);
			if (n<4)
			{
				((CvPoint*)param)[n].x = x;
				((CvPoint*)param)[n].y = y;
			}
		}
		break;
	}
}


//Load camera params from yaml file
int LoadCameraParams(char* filename, CvMat** intrinsic_matrix, CvMat** distortion_coeffs)
{
	CvFileStorage* fs = cvOpenFileStorage( filename, 0, CV_STORAGE_READ );
	if (fs==NULL) return 0;

	*intrinsic_matrix = (CvMat*)cvReadByName( fs,0,"camera_matrix");
	*distortion_coeffs = (CvMat*)cvReadByName( fs,0,"distortion_coefficients");

	return 1;
}


int LoadApplicationParams(char* filename)
{
	CvFileStorage* fs = cvOpenFileStorage( filename, 0, CV_STORAGE_READ );
	if (fs==NULL) return 0;

	USE_DEINTERLACING = cvReadIntByName( fs,0,"USE_DEINTERLACING",0) != 0;
	USE_UNDISTORTION = cvReadIntByName( fs,0,"USE_UNDISTORTION",0) != 0;
	CALIBRATE_CAMERA = cvReadIntByName( fs,0,"CALIBRATE_CAMERA",0) != 0;
	DRAW_CORNERS = cvReadIntByName( fs,0,"DRAW_CORNERS",1) != 0;
	DRAW_CHESSBOARD = cvReadIntByName( fs,0,"DRAW_CHESSBOARD",1) != 0;
	AUTOFIND_CHESSBOARD = cvReadIntByName( fs,0,"AUTOFIND_CHESSBOARD",1) != 0;
	IS_VIDEO_CAPTURE = cvReadIntByName( fs,0,"IS_VIDEO_CAPTURE",0) != 0;
	SAVE_DRAWINGS = cvReadIntByName( fs,0,"SAVE_DRAWINGS",0) != 0;
	SAVE_SAMPLES = cvReadIntByName( fs,0,"SAVE_SAMPLES",0) != 0;
	SAVE_ALL_FRAMES = cvReadIntByName( fs,0,"SAVE_ALL_FRAMES",0) != 0;
	SHOW_TEST_SQUARE = cvReadIntByName( fs,0,"SHOW_TEST_SQUARE",0) != 0;
	USE_3D = cvReadIntByName( fs,0,"USE_3D",0) != 0;
	if (SAVE_SAMPLES)
	{
		SAMPLES_PATH = new char[500];
		if (cvReadStringByName( fs,0,"SAMPLES_PATH"))
			strcpy(SAMPLES_PATH, cvReadStringByName( fs,0,"SAMPLES_PATH"));
		else return 0;
	}
	if (IS_VIDEO_CAPTURE)
	{
		INPUT_VIDEO = new char[500];
		if (cvReadStringByName( fs,0,"INPUT_VIDEO"))
			strcpy(INPUT_VIDEO,cvReadStringByName( fs,0,"INPUT_VIDEO") );//input video filename
		else return 0;
	}

	OUTPUT_DIRECTORY = new char[500];
	if (cvReadStringByName( fs,0,"OUTPUT_DIRECTORY"))
		strcpy(OUTPUT_DIRECTORY,cvReadStringByName( fs,0,"OUTPUT_DIRECTORY"));//output directory for images and selected regions path
	else return 0;
	//chessboard inner corners count (width x height)
	int CHESSBOARD_WIDTH=cvReadIntByName( fs,0,"CHESSBOARD_WIDTH");
	int CHESSBOARD_HEIGHT=cvReadIntByName( fs,0,"CHESSBOARD_HEIGHT");
	innerCornersCount=cvSize(CHESSBOARD_WIDTH,CHESSBOARD_HEIGHT);


	CAMERA_PARAMETERS_PATH = new char[500];
	if (cvReadStringByName( fs,0,"CAMERA_PARAMETERS_PATH"))
		strcpy(CAMERA_PARAMETERS_PATH, cvReadStringByName( fs,0,"CAMERA_PARAMETERS_PATH"));//camera parameters filename
	else 
		return 0;
	if (LoadCameraParams(CAMERA_PARAMETERS_PATH,&IntrinsicMatrix,&DistortionCoeffs))
	{
		printf("\nCamera parameters loaded successfully\n");
	}
	else
	{
		printf("\nUnable to load parameters\n");
		if (USE_UNDISTORTION && (!CALIBRATE_CAMERA))
			return 0;
	}


	//cvReleaseFileStorage(&fs);
	return 1;
}


IplImage* Undistort(IplImage* src,const CvMat* intrinsic_matrix,const CvMat* distortion_coeffs)
{
	IplImage* undistortImg;
	undistortImg = cvCreateImage(cvSize(src->width,src->height),src->depth,src->nChannels);
	cvUndistort2(src,undistortImg,intrinsic_matrix,distortion_coeffs);

	return undistortImg;
}

int ShowFrame(int pos)
{
	cvSetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES,pos);
	frame = cvQueryFrame(capture);
	if (frame)
	{
		IplImage* undistortImg = 0;
		cvReleaseImage(&workImage);
		if (USE_DEINTERLACING)
			workImage =  Deinterlace(frame);
		else
			workImage = cvCloneImage(frame);
		if (USE_UNDISTORTION)
		{

			undistortImg = Undistort(workImage,IntrinsicMatrix,DistortionCoeffs);		
			workImage = cvCloneImage(undistortImg);
			cvReleaseImage(&undistortImg);
		}

		cvShowImage(FRAME_WINDOW,workImage);
		return 1;
	}
	return 0;
}


//Calculates selected points coordinates (by x and y) in coordinate system connected with chessboard
//(zero point is in the upper-left corner of the chessboard, axises are horizontal and vertical chessboard edges)
// in: nFrame (number of frame on which points were selected), points (CvPoint[4] array with selected points)
// out: relCoords (pointer to the array[4] with (x1,y1) coordinate of every point)
// Function returns 0 if there is no chessboard found and array 2xN with corners otherwise
// Camera Matrix is for 3D relative position estimation with z start coordinate
// We also can give found Chessboard to the method instead of running chessboard finding algorithm
CvMat* CalcRelativePosition(IplImage* workImage, CvPoint* points, CvPoint2D32f** relCoords, CvMat* cameraMatrix=NULL, float z = 0, CvMat* Chessboard=NULL)
{
	CvPoint2D32f* corners = new CvPoint2D32f[innerCornersCount.height*innerCornersCount.width];
	int count;

	if (!Chessboard)
	{
		if (cvFindChessboardCorners(workImage,innerCornersCount,corners,&count)==0)
		{
			delete[] corners;
			return NULL;
		}

		IplImage* view_gray = cvCreateImage( cvGetSize(workImage), 8, 1 );
		cvCvtColor(workImage, view_gray, CV_BGR2GRAY );
		cvFindCornerSubPix( view_gray, corners, count, cvSize(11,11),cvSize(-1,-1), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
		cvReleaseImage( &view_gray );
	}
	else
	{
		for (int i=0;i<Chessboard->cols;i++)
		{
			corners[i].x = (float)Chessboard->data.db[i];
			corners[i].y = (float)Chessboard->data.db[Chessboard->cols+i];
		}	
	}

	if (!cameraMatrix)
	{
		CvPoint2D32f src[3];
		CvPoint2D32f dst[3];
		src[0].x = corners[0].x;
		src[0].y = corners[0].y;
		src[1].x = corners[innerCornersCount.width-1].x;
		src[1].y = corners[innerCornersCount.width-1].y;
		src[2].x = corners[innerCornersCount.width*(innerCornersCount.height-1)].x;
		src[2].y = corners[innerCornersCount.width*(innerCornersCount.height-1)].y;
		dst[0].x = 0;
		dst[0].y = 0;
		dst[1].x = innerCornersCount.width-1.f;
		dst[1].y = 0;
		dst[2].x = 0;
		dst[2].y = innerCornersCount.height-1.f;

		CvMat* map_matrix = cvCreateMat(2,3,CV_64F);
		cvGetAffineTransform(src,dst,map_matrix);

		for (int i=0;i<4;i++)
		{
			(*relCoords)[i].x=(float)(points[i].x*(map_matrix->data.db[0])+points[i].y*(map_matrix->data.db[1])+(map_matrix->data.db[2]));
			(*relCoords)[i].y=(float)(points[i].x*(map_matrix->data.db[3])+points[i].y*(map_matrix->data.db[4])+(map_matrix->data.db[5]));
		}
		cvReleaseMat(&map_matrix);
	}
	else
	{
		CvMat* image_points = cvCreateMat(2,innerCornersCount.height*innerCornersCount.width,CV_64FC1);
		CvMat* object_points = cvCreateMat(3,innerCornersCount.height*innerCornersCount.width,CV_64FC1);

		// Sets object points and image points
		for (int i=0; i< innerCornersCount.height;i++)
			for (int j=0; j < innerCornersCount.width;j++)
			{
				object_points->data.db[(i*innerCornersCount.width+j)]=j;
				object_points->data.db[(i*innerCornersCount.width+j)+innerCornersCount.width*innerCornersCount.height]=i;
				object_points->data.db[(i*innerCornersCount.width+j)+2*innerCornersCount.width*innerCornersCount.height]=0.0f;

				image_points->data.db[(i*innerCornersCount.width+j)]=corners[(i*innerCornersCount.width+j)].x;
				image_points->data.db[(i*innerCornersCount.width+j)+innerCornersCount.width*innerCornersCount.height]=corners[(i*innerCornersCount.width+j)].y;
			}

		CvMat* R = cvCreateMat(3, 3, CV_64FC1);
		CvMat* T = cvCreateMat(3, 1, CV_64FC1);
		CvMat* rotation_vector = cvCreateMat(3,1,CV_64FC1);
		//Calculating Exrinsic camera parameters
		CvMat* distCoeffs = cvCreateMat(5,1,CV_64FC1);
		cvZero(distCoeffs);
		cvFindExtrinsicCameraParams2(object_points,image_points,cameraMatrix, distCoeffs,rotation_vector,T);
		cvReleaseMat(&distCoeffs);
		cvRodrigues2(rotation_vector,R);
		CvMat* M = cvCreateMat(3, 4, CV_64FC1);
		CvMat* A = cvCreateMat(3, 3, CV_64FC1);
		CvMat* invA = cvCreateMat(3, 3, CV_64FC1);
		CvMat* point3D = cvCreateMat(3, 1, CV_64FC1);
		CvMat* t = cvCreateMat(3, 1, CV_64FC1);

		cvGEMM(cameraMatrix,R,1.0,NULL,0,A);
		cvGEMM(cameraMatrix,T,1.0,NULL,0,t);

		t->data.db[0]+=z*A->data.db[2];
		t->data.db[1]+=z*A->data.db[5];
		t->data.db[2]+=z*A->data.db[8];

		t->data.db[0]=-t->data.db[0];
		t->data.db[1]=-t->data.db[1];
		t->data.db[2]=-t->data.db[2];


		for (int i=0;i<4;i++)
		{
			A->data.db[2]=-points[i].x;
			A->data.db[5]=-points[i].y;
			A->data.db[8]=-1.0f;

			cvInvert(A,invA);
			cvGEMM(invA,t,1.0,NULL,0,point3D);
			(*relCoords)[i].x = (float)(point3D->data.db[0]);
			(*relCoords)[i].y = (float)(point3D->data.db[1]);

			//TEST
			CvMat* d3 = cvCreateMat(1,1,CV_64FC3);
			d3->data.db[0] = point3D->data.db[0];
			d3->data.db[1] = point3D->data.db[1];
			d3->data.db[2] = z;
			distCoeffs = cvCreateMat(5,1,CV_64FC1);
			CvMat* imP = cvCreateMat(1,1,CV_64FC2);
			cvZero(distCoeffs);
			
			cvProjectPoints2(d3,rotation_vector,T,cameraMatrix,distCoeffs,imP);
			cvReleaseMat(&imP);
			cvReleaseMat(&d3);
			cvReleaseMat(&distCoeffs);
			//END OF
		}

		cvReleaseMat(&T);
		cvReleaseMat(&t);
		cvReleaseMat(&R);
		cvReleaseMat(&rotation_vector);
		cvReleaseMat(&M);
		cvReleaseMat(&A);
		cvReleaseMat(&invA);
		cvReleaseMat(&point3D);
		
	}
	if (!Chessboard)
	{
		CvMat* result = cvCreateMat(2,innerCornersCount.height*innerCornersCount.width,CV_64FC1);
		for (int i=0;i<result->cols;i++)
		{
			result->data.db[i]=corners[i].x;
			result->data.db[result->cols+i]=corners[i].y;
		}	
		
		delete[] corners;
		return result;
	}
	else
	{
		return NULL;
	}

}

// if chessboardPoints = NULL using simle affine transform otherwise we must have correct oldPoints pointer
// chessBoardPoints is array 2xN
//  returns new points location
CvPoint* GetCurrentPointsPosition(IplImage* workImage, CvPoint2D32f* relCoords, CvMat* chessboardPoints, CvPoint* oldPoints, CvPoint2D32f* outCorners)
{
	CvPoint2D32f* corners = new CvPoint2D32f[innerCornersCount.height*innerCornersCount.width];
	int count;

	if (cvFindChessboardCorners(workImage,innerCornersCount,corners,&count)==0)
	{
		delete[] corners;
		corners = NULL;
		return NULL;
	}

	IplImage* view_gray = cvCreateImage( cvGetSize(workImage), 8, 1 );
	cvCvtColor(workImage, view_gray, CV_BGR2GRAY );
	cvFindCornerSubPix( view_gray, corners, count, cvSize(11,11),cvSize(-1,-1), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
	cvReleaseImage( &view_gray );

	CvPoint* result = new CvPoint[4];

	if (chessboardPoints && oldPoints)
	{
		CvMat* chessboardPoints2 = cvCreateMat(2,innerCornersCount.height*innerCornersCount.width,CV_64FC1);
		for (int i=0;i<(chessboardPoints2)->cols;i++)
		{
			chessboardPoints2->data.db[i]=(corners)[i].x;
			chessboardPoints2->data.db[chessboardPoints2->cols+i]=(corners)[i].y;
		}
		CvMat* homography = cvCreateMat(3,3,CV_64FC1);

		cvFindHomography(chessboardPoints,chessboardPoints2,homography);

		for (int i=0;i<4;i++)
		{
			double t = oldPoints[i].x*(homography->data.db[6])+oldPoints[i].y*(homography->data.db[7])+(homography->data.db[8]);
			result[i].x = (int)((oldPoints[i].x*(homography->data.db[0])+oldPoints[i].y*(homography->data.db[1])+(homography->data.db[2]))/t);
			result[i].y = (int)((oldPoints[i].x*(homography->data.db[3])+oldPoints[i].y*(homography->data.db[4])+(homography->data.db[5]))/t);

		}
		cvReleaseMat(&homography);
	}

	else
	{
		CvPoint2D32f src[3];
		CvPoint2D32f dst[3];
		dst[0].x = (corners)[0].x;
		dst[0].y = (corners)[0].y;
		dst[1].x = (corners)[innerCornersCount.width-1].x;
		dst[1].y = (corners)[innerCornersCount.width-1].y;
		dst[2].x = (corners)[innerCornersCount.width*(innerCornersCount.height-1)].x;
		dst[2].y = (corners)[innerCornersCount.width*(innerCornersCount.height-1)].y;
		src[0].x = 0;
		src[0].y = 0;
		src[1].x = innerCornersCount.width-1.f;
		src[1].y = 0;
		src[2].x = 0;
		src[2].y = innerCornersCount.height-1.f;

		CvMat* map_matrix = cvCreateMat(2,3,CV_64F);
		cvGetAffineTransform(src,dst,map_matrix);

		for (int i=0;i<4;i++)
		{
			result[i].x=(int)(relCoords[i].x*(map_matrix->data.db[0])+relCoords[i].y*(map_matrix->data.db[1])+(map_matrix->data.db[2]));
			result[i].y=(int)(relCoords[i].x*(map_matrix->data.db[3])+relCoords[i].y*(map_matrix->data.db[4])+(map_matrix->data.db[5]));
		}
		cvReleaseMat(&map_matrix);
	}

	if (outCorners)
	{
		for (int i=0;i<innerCornersCount.height*innerCornersCount.width;i++)
		{
			outCorners[i].x=corners[i].x;
			outCorners[i].y=corners[i].y;
		}
	}

	delete[] corners;

	return result;
}




IplImage* GetSample(const IplImage* src,CvSize innerCornersCount, const CvPoint* points, CvPoint2D32f* chessboardCorners)
{
	CvPoint2D32f* corners = new CvPoint2D32f[innerCornersCount.height*innerCornersCount.width];
	int count;

	if (!chessboardCorners)
	{
		if (cvFindChessboardCorners(src,innerCornersCount,corners,&count)==0)
		{
			delete[] corners;
			return NULL;
		}

		IplImage* view_gray = cvCreateImage( cvGetSize(src), 8, 1 );
		cvCvtColor(src, view_gray, CV_BGR2GRAY );
		cvFindCornerSubPix( view_gray, corners, count, cvSize(11,11),cvSize(-1,-1), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
		cvReleaseImage( &view_gray );
	}
	else
	{
		for (int i=0;i<innerCornersCount.height*innerCornersCount.width;i++)
		{
			corners[i]=chessboardCorners[i];
		}

	}

	CvPoint result[4];
	CvPoint2D32f top;
	CvPoint2D32f bot;
	top.x = corners[0].x < corners[(innerCornersCount.height-1)*innerCornersCount.width].x ? corners[0].x : corners[(innerCornersCount.height-1)*innerCornersCount.width].x;
	top.y = corners[0].y < corners[innerCornersCount.width-1].y ? corners[0].y : corners[innerCornersCount.width-1].y;
	bot.x = corners[innerCornersCount.width-1].x < corners[innerCornersCount.width*innerCornersCount.height-1].x ? corners[innerCornersCount.width*innerCornersCount.height-1].x : corners[innerCornersCount.width-1].x;
	bot.y = corners[innerCornersCount.height*innerCornersCount.width-1].y < corners[(innerCornersCount.height-1)*innerCornersCount.width].y ?
		corners[(innerCornersCount.height-1)*innerCornersCount.width].y : corners[innerCornersCount.height*innerCornersCount.width-1].y;

	CvMat* chessboardPoints = cvCreateMat(2,innerCornersCount.height*innerCornersCount.width,CV_64FC1);
	CvMat* chessboardPoints2 = cvCreateMat(2,innerCornersCount.height*innerCornersCount.width,CV_64FC1);

	//assumes that chessboard squares are squared
	float step = (bot.x-top.x)/(innerCornersCount.width-1) > (bot.y-top.y)/(innerCornersCount.height-1) ? (bot.x-top.x)/(innerCornersCount.width-1) : (bot.y-top.y)/(innerCornersCount.height-1);
	for (int i=0;i<innerCornersCount.width;i++)
		for (int j=0;j<innerCornersCount.height;j++)
		{
			chessboardPoints2->data.db[i+j*innerCornersCount.width]=top.x+i*step;
			chessboardPoints2->data.db[i+j*(innerCornersCount.width)+innerCornersCount.width*innerCornersCount.height]=
				top.y+j*step;
		}

		for (int i=0;i<(chessboardPoints)->cols;i++)
		{
			chessboardPoints->data.db[i]=(corners)[i].x;
			chessboardPoints->data.db[chessboardPoints->cols+i]=(corners)[i].y;
		}
		CvMat* homography = cvCreateMat(3,3,CV_64FC1);

		cvFindHomography(chessboardPoints,chessboardPoints2,homography);

		for (int i=0;i<4;i++)
		{
			double t = points[i].x*(homography->data.db[6])+points[i].y*(homography->data.db[7])+(homography->data.db[8]);
			result[i].x = (int)((points[i].x*(homography->data.db[0])+points[i].y*(homography->data.db[1])+(homography->data.db[2]))/t);
			result[i].y = (int)((points[i].x*(homography->data.db[3])+points[i].y*(homography->data.db[4])+(homography->data.db[5]))/t);
		}

		IplImage* resImage= cvCloneImage(src);
		cvWarpPerspective( src, resImage, homography);
		CvRect rect;
		rect.x = result[0].x<result[2].x ? result[0].x : result[2].x;
		rect.y = result[0].y<result[1].y ? result[0].y : result[1].y;
		rect.width = result[1].x<result[3].x ? (result[3].x-rect.x) : (result[1].x-rect.x);
		rect.height = result[2].y<result[3].y ? (result[3].y-rect.y) : (result[2].y-rect.y);
		if ((rect.x>=0) && (rect.y >=0) && ((rect.x+rect.width)<resImage->width)&&((rect.y+rect.height)<resImage->height))
			cvSetImageROI(resImage,rect);
		else
			return NULL;

		cvReleaseMat(&homography);
		return resImage;
		delete[] corners;
}

void createSamples2DObject(int argc, char** argv)
{
	CvPoint points[4];
	CvPoint pointsChessboardSquare[4];

	CvPoint2D32f* relCoords = new CvPoint2D32f[4];
	CvPoint2D32f* relCoordsChessboardSquare = new CvPoint2D32f[4];
	CvPoint * newPoints;
	CvPoint * newPointsChessboardSquare;
	int currentFrame=0;
	int key;
	IplImage* undistortedImg;

	if (CALIBRATE_CAMERA)
	{
		int v = IS_VIDEO_CAPTURE ? 10 : 7;
		char** c = new char*[v];
		c[0]=new char[11]; c[0]="calibration";
		c[1]=new char[2]; c[1]="-w";
		c[2] = new char[2]; sprintf(c[2],"%d",innerCornersCount.width);
		c[3]=new char[2]; c[3]="-h";
		c[4] = new char[2];  sprintf(c[4],"%d",innerCornersCount.height);
		c[5] = new char[2]; c[5]="-d";
		c[6] = new char[5];  sprintf(c[6],"%d",VIDEO_CALIBRATION_DELAY);
		c[7]=new char[2]; c[7]="-o";
		c[8]=new char[100];
		sprintf(c[8],"%scamera.yml",OUTPUT_DIRECTORY);
		if (IS_VIDEO_CAPTURE)
		{
			c[9]=new char[strlen(INPUT_VIDEO)+1];
			strcpy(c[9],INPUT_VIDEO);
		}

		calibrate(v,c);
		LoadCameraParams(c[8],&IntrinsicMatrix, &DistortionCoeffs);

		// Add functionality for camera calibration
	}

	cvNamedWindow(FRAME_WINDOW,1);
	capture = IS_VIDEO_CAPTURE ? cvCreateFileCapture(INPUT_VIDEO) : cvCreateCameraCapture(0);

	if (!IS_VIDEO_CAPTURE)
	{
		cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH,640);
		cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT,480);
	}

	cvResizeWindow(FRAME_WINDOW,(int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH),(int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT));

	CvPoint2D32f* corners = new CvPoint2D32f[innerCornersCount.height*innerCornersCount.width];
	int count;

	do
	{
		if (IS_VIDEO_CAPTURE)
			cvSetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES,++currentFrame);
		if (workImage) cvReleaseImage(&workImage);
		frame = cvQueryFrame(capture);
		if (frame)
		{
			if (USE_DEINTERLACING)
				workImage =  Deinterlace(frame);
			else
				workImage = cvCloneImage(frame);

			if (USE_UNDISTORTION)
			{

				undistortedImg = Undistort(workImage,IntrinsicMatrix,DistortionCoeffs);
				cvReleaseImage(&workImage);
				workImage = cvCloneImage(undistortedImg);
				cvReleaseImage(&undistortedImg);
			}
			//if (!IS_VIDEO_CAPTURE)
			cvShowImage(FRAME_WINDOW,workImage);
			key = cvWaitKey(30);
			if (key==27) return;
		}

	}
	while ( frame && (((AUTOFIND_CHESSBOARD && IS_VIDEO_CAPTURE)|| (!IS_VIDEO_CAPTURE)) && cvFindChessboardCorners(workImage,innerCornersCount,corners,&count)==0));

	if (frame == NULL)
	{
		printf("\n Unable to load video with chessboard or connect to the camera");
		return;
	}

	delete[] corners;
	currentFrame--;

	IplImage* firstImage=cvCloneImage(workImage);

	cvShowImage(FRAME_WINDOW,workImage);

	points[0].x =-1;
	points[1].x =-1;
	points[2].x =-1;
	points[3].x =-1;

	printf("\n Select the quadrangle region by selectig its vertices on the frame with a mouse\n");
	cvSetMouseCallback(FRAME_WINDOW, on_mouse, &points);

	// Getting initial points position
	for (int currPoint=0;currPoint<4;)
	{
		if (points[currPoint].x != -1)
		{
			cvCircle(workImage,cvPoint(points[currPoint].x,points[currPoint].y),2,cvScalar(255,255,255));
			cvCircle(workImage,cvPoint(points[currPoint].x,points[currPoint].y),3,cvScalar(0));
			cvShowImage(FRAME_WINDOW,workImage);
			currPoint++;
		}
		key = cvWaitKey(30);
		if (IS_VIDEO_CAPTURE)
		{
			switch (key)
			{
			case 32: // Space symbol
				if (ShowFrame(++currentFrame))
				{
					points[0].x =-1;
					points[1].x =-1;
					points[2].x =-1;
					points[3].x =-1;
					currPoint = 0;
				}
				else
				{
					currentFrame--;
				}
				break;
			case 8: // Backspace symbol
				if (currentFrame>0)
				{
					if (ShowFrame(--currentFrame))
					{
						points[0].x =-1;
						points[1].x =-1;
						points[2].x =-1;
						points[3].x =-1;
						currPoint = 0;
					}
					else
					{
						currentFrame++;
					}
				}
				break;
			}
		}
	}

	// sorting points

	for (int i=1;i<4;i++)
	{

		if (points[i].y<points[0].y)
		{
			CvPoint temp;
			temp = points[0];
			points[0]=points[i];
			points[i]=temp;
		}
	}
	if (points[1].x<points[0].x)
	{
		CvPoint temp;
		temp = points[0];
		points[0]=points[1];
		points[1]=temp;
	}

	if (points[3].x<points[2].x)
	{
		CvPoint temp;
		temp = points[3];
		points[3]=points[2];
		points[2]=temp;
	}

	//end of sorting

	if (workImage) cvReleaseImage(&workImage);
	workImage = cvCloneImage(firstImage);
	if (USE_DEINTERLACING)
		workImage =  Deinterlace(frame);
	else
		workImage = cvCloneImage(frame);
	if (USE_UNDISTORTION)
	{
		undistortedImg = Undistort(workImage,IntrinsicMatrix,DistortionCoeffs);
		cvReleaseImage(&workImage);
		workImage = cvCloneImage(undistortedImg);
		cvReleaseImage(&undistortedImg);
	}
	cvReleaseImage(&firstImage);


	CvMat* chessboardPointsMat = CalcRelativePosition(workImage, points,&relCoords);
	if (SHOW_TEST_SQUARE && chessboardPointsMat)
	{
		pointsChessboardSquare[0].x = (int)chessboardPointsMat->data.db[(innerCornersCount.height-1)*innerCornersCount.width-2];
		pointsChessboardSquare[1].x = (int)chessboardPointsMat->data.db[(innerCornersCount.height-1)*innerCornersCount.width-1];
		pointsChessboardSquare[3].x = (int)chessboardPointsMat->data.db[innerCornersCount.height*innerCornersCount.width-2];
		pointsChessboardSquare[2].x = (int)chessboardPointsMat->data.db[innerCornersCount.height*innerCornersCount.width-1];

		pointsChessboardSquare[0].y = (int)chessboardPointsMat->data.db[chessboardPointsMat->cols+(innerCornersCount.height-1)*innerCornersCount.width-2];
		pointsChessboardSquare[1].y = (int)chessboardPointsMat->data.db[chessboardPointsMat->cols+(innerCornersCount.height-1)*innerCornersCount.width-1];
		pointsChessboardSquare[3].y = (int)chessboardPointsMat->data.db[chessboardPointsMat->cols+innerCornersCount.height*innerCornersCount.width-2];
		pointsChessboardSquare[2].y = (int)chessboardPointsMat->data.db[chessboardPointsMat->cols+innerCornersCount.height*innerCornersCount.width-1];
		CalcRelativePosition(workImage,pointsChessboardSquare,&relCoordsChessboardSquare);
	}

	char* PATH = OUTPUT_DIRECTORY;
	FILE* f;
	char path[100];
	char path_frames[100];
	sprintf(path_frames,"%sframes.txt",PATH);
	remove(path_frames);
	char cmd[100];
#ifdef WIN32
	sprintf(cmd,"mkdir %s",PATH);
	system(cmd);
	sprintf(cmd,"del %s*.* /q",PATH);
	system(cmd);
	if (SAVE_SAMPLES)
	{
		sprintf(cmd,"mkdir %s",SAMPLES_PATH);
		system(cmd);
		sprintf(cmd,"del %s*.* /q",SAMPLES_PATH);
		system(cmd);
	}
#else

	sprintf(cmd,"mkdir %s",PATH);
	system(cmd);
	sprintf(cmd,"rm -f %s*.*",PATH);
	system(cmd);
	if (SAVE_SAMPLES)
	{
		sprintf(cmd,"mkdir %s",SAMPLES_PATH);
		system(cmd);
		sprintf(cmd,"rm -f %s*.*",SAMPLES_PATH);
		system(cmd);
	}
#endif

	do
	{
		if (workImage) cvReleaseImage(&workImage);
		frame = cvQueryFrame(capture);
		if (frame)
		{
			if (USE_DEINTERLACING)
				workImage =  Deinterlace(frame);
			else
				workImage = cvCloneImage(frame);

			if (USE_UNDISTORTION)
			{
				undistortedImg = Undistort(workImage,IntrinsicMatrix,DistortionCoeffs);
				cvReleaseImage(&workImage);
				workImage = cvCloneImage(undistortedImg);
				cvReleaseImage(&undistortedImg);
			}

			if (chessboardPointsMat)
			{
				CvPoint2D32f* ChessboardCorners = new CvPoint2D32f[innerCornersCount.height*innerCornersCount.width];
				newPoints = GetCurrentPointsPosition(workImage,relCoords,chessboardPointsMat,points,ChessboardCorners/*&newPointsMat*/);

				if (newPoints)
				{
					if (SHOW_TEST_SQUARE)
					{
						newPointsChessboardSquare = GetCurrentPointsPosition(workImage,relCoordsChessboardSquare,chessboardPointsMat,pointsChessboardSquare,ChessboardCorners);
						cvFillConvexPoly(workImage,newPointsChessboardSquare,4,cvScalar(10,10,250));
					}
					bool areOnFrame=true; // are points on frame or not
					for (int i=0;i<4;i++)
					{
						if ((newPoints[i].x>workImage->width)||(newPoints[i].x<0)||(newPoints[i].y>workImage->height)||(newPoints[i].y<0))
							areOnFrame=false;
					}

					if (areOnFrame)
					{
						f = fopen(path_frames,"a");
						sprintf(path,"%s%d.jpg",PATH,currentFrame+1);
						fprintf(f,"%s,%d,%d,%d,%d,%d,%d,%d,%d\n",path,newPoints[0].x,newPoints[0].y,newPoints[1].x,newPoints[1].y,newPoints[2].x,newPoints[2].y,newPoints[3].x,newPoints[3].y);
						fclose(f);

						if (DRAW_CHESSBOARD)
							cvDrawChessboardCorners(workImage,innerCornersCount,ChessboardCorners,innerCornersCount.height*innerCornersCount.width,1);
						if (DRAW_CORNERS)
						{
							cvCircle(workImage,newPoints[0],2,cvScalar(255,255,255));
							cvCircle(workImage,newPoints[0],3,cvScalar(0));
							cvCircle(workImage,newPoints[1],2,cvScalar(255,255,255));
							cvCircle(workImage,newPoints[1],3,cvScalar(0));
							cvCircle(workImage,newPoints[2],2,cvScalar(255,255,255));
							cvCircle(workImage,newPoints[2],3,cvScalar(0));
							cvCircle(workImage,newPoints[3],2,cvScalar(255,255,255));
							cvCircle(workImage,newPoints[3],3,cvScalar(0));
						}
						if (SAVE_DRAWINGS)
							cvSaveImage(path,workImage);
						else
						{
							if (USE_DEINTERLACING)
							{
								IplImage* tmp = Deinterlace(frame);
								if (USE_UNDISTORTION)
								{
									undistortedImg = Undistort(tmp,IntrinsicMatrix,DistortionCoeffs);
									cvReleaseImage(&tmp);
									tmp = cvCloneImage(undistortedImg);
									cvReleaseImage(&undistortedImg);
								}
								cvSaveImage(path,tmp);
								cvReleaseImage(&tmp);
							}
							else
							{
								if (!USE_UNDISTORTION)
									cvSaveImage(path,frame);
								else
								{
									undistortedImg = Undistort(frame,IntrinsicMatrix,DistortionCoeffs);
									cvSaveImage(path,undistortedImg);
									cvReleaseImage(&undistortedImg);
								}
							}
						}
						if (SAVE_SAMPLES)
						{
							sprintf(path,"%s%d.jpg",SAMPLES_PATH,currentFrame+1);
							IplImage* tmp;
							if (!USE_UNDISTORTION)
								tmp = GetSample(frame,innerCornersCount,newPoints,ChessboardCorners);
							else
							{
								undistortedImg = Undistort(frame,IntrinsicMatrix,DistortionCoeffs);
								tmp = GetSample(undistortedImg,innerCornersCount,newPoints,ChessboardCorners);
								cvReleaseImage(&undistortedImg);
							}
							if (tmp)
							{
								cvSaveImage(path,tmp);
								cvReleaseImage(&tmp);
							}
						}
						printf("Frame %d successfully saved to %s\n",currentFrame+1,path);
						delete[] ChessboardCorners;
						ChessboardCorners = NULL;
					}

					delete[] newPoints;
					newPoints = NULL;
				}
				else
					if (SAVE_ALL_FRAMES)
					{
						sprintf(path,"%s%d.jpg",PATH,currentFrame+1);
						cvSaveImage(path,workImage);
					}

			}


			cvShowImage(FRAME_WINDOW,workImage);
			key = cvWaitKey(30);
		}
		currentFrame++;
	}
	while (frame && (key!=27));

	if (capture)
		cvReleaseCapture(&capture);
}
//--------------------------

void createSamples3DObject(int argc, char** argv)
{
	CvPoint points1[4];
	CvPoint points2[4];

	IplImage* i1;
	IplImage* i2;
	IplImage* undistortedImg;

	CvPoint3D32f* points3D = new CvPoint3D32f[8];
	CvPoint* newPoints = new CvPoint[8];

	int currentFrame=0;
	int key;
	int boxDepthValue=10;
	int maxErrorValue=10;


	if (CALIBRATE_CAMERA)
	{
		int v = 10;/*IS_VIDEO_CAPTURE ? 10 : 7*/;
		char** c = new char*[v];
		c[0] = new char[11]; c[0]="calibration";
		c[1] = new char[2]; c[1]="-w";
		c[2] = new char[2]; sprintf(c[2],"%d",innerCornersCount.width);
		c[3] = new char[2]; c[3]="-h";
		c[4] = new char[2];  sprintf(c[4],"%d",innerCornersCount.height);
		c[5] = new char[2]; c[5]="-d";
		c[6] = new char[5];  sprintf(c[6],"%d",VIDEO_CALIBRATION_DELAY);
		c[7] = new char[2]; c[7]="-o";
		c[8] = new char[100];
		sprintf(c[8],"%scamera.yml",OUTPUT_DIRECTORY);
		c[9]=new char[strlen(INPUT_VIDEO)+1];
		strcpy(c[9],INPUT_VIDEO);

		calibrate(v,c);
		LoadCameraParams(c[8],&IntrinsicMatrix, &DistortionCoeffs);

		// Add functionality for camera calibration
	}

	cvNamedWindow(FRAME_WINDOW,1);
	cvCreateTrackbar(DEPTH_TRACKBAR,FRAME_WINDOW,&boxDepthValue,MAX_DEPTH_VALUE,NULL);
	cvCreateTrackbar(ERROR_TRACKBAR,FRAME_WINDOW,&maxErrorValue,MAX_ERROR_VALUE,NULL);
	capture =cvCreateFileCapture(INPUT_VIDEO);

	cvResizeWindow(FRAME_WINDOW,(int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH),(int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT));

	CvPoint2D32f* corners = new CvPoint2D32f[innerCornersCount.height*innerCornersCount.width];

	if (capture == NULL)
	{
		printf("\n Unable to load video with chessboard or connect to the camera");
		return;
	}

	delete[] corners;

	if (workImage) cvReleaseImage(&workImage);
	frame = cvQueryFrame(capture);
	if (frame)
	{
		if (USE_DEINTERLACING)
			workImage =  Deinterlace(frame);
		else
			workImage = cvCloneImage(frame);

		if (USE_UNDISTORTION)
		{

			undistortedImg = Undistort(workImage,IntrinsicMatrix,DistortionCoeffs);
			cvReleaseImage(&workImage);
			workImage = cvCloneImage(undistortedImg);
			cvReleaseImage(&undistortedImg);
		}
	}

	IplImage* firstImage=cvCloneImage(workImage);

	cvShowImage(FRAME_WINDOW,workImage);

	do
	{
		points1[0].x =-1;
		points1[1].x =-1;
		points1[2].x =-1;
		points1[3].x =-1;
		points2[0].x =-1;
		points2[1].x =-1;
		points2[2].x =-1;
		points2[3].x =-1;

		printf("\nSelect the quadrangle region by selecting its vertices on the frame with a mouse\nUse Space and Backspace to go forward or backward\nPress Esc key to exit");

		cvSetMouseCallback(FRAME_WINDOW, on_mouse, &points1);

		// Getting initial points position
		for (int currPoint=0;currPoint<4;)
		{
			if (points1[currPoint].x != -1)
			{
				cvCircle(workImage,cvPoint(points1[currPoint].x,points1[currPoint].y),2,cvScalar(255,255,255));
				cvCircle(workImage,cvPoint(points1[currPoint].x,points1[currPoint].y),3,cvScalar(0));
				cvShowImage(FRAME_WINDOW,workImage);
				currPoint++;
			}
			key = cvWaitKey(30);
			switch (key)
			{
			case 32: // Space symbol
				if (ShowFrame(++currentFrame))
				{
					points1[0].x =-1;
					points1[1].x =-1;
					points1[2].x =-1;
					points1[3].x =-1;
					currPoint = 0;
				}
				else
				{
					currentFrame--;
				}
				break;
			case 8: // Backspace symbol
				if (currentFrame>0)
				{
					if (ShowFrame(--currentFrame))
					{
						points1[0].x =-1;
						points1[1].x =-1;
						points1[2].x =-1;
						points1[3].x =-1;

						currPoint = 0;
					}
					else
					{
						currentFrame++;
					}
				}
				break;
			}
		}

		i1 = cvCloneImage(workImage);

		// sorting points

		for (int i=1;i<4;i++)
		{

			if (points1[i].y<points1[0].y)
			{
				CvPoint temp;
				temp = points1[0];
				points1[0]=points1[i];
				points1[i]=temp;
			}
		}
		if (points1[1].x<points1[0].x)
		{
			CvPoint temp;
			temp = points1[0];
			points1[0]=points1[1];
			points1[1]=temp;
		}

		if (points1[3].x<points1[2].x)
		{
			CvPoint temp;
			temp = points1[3];
			points1[3]=points1[2];
			points1[2]=temp;
		}

		//end of sorting


		ShowFrame(currentFrame);
		//second frame
		cvSetMouseCallback(FRAME_WINDOW, on_mouse, &points2);

		printf("\nSelect one more time on another frame\n");
		// Getting initial points position
		for (int currPoint=0;currPoint<4;)
		{
			if (points2[currPoint].x != -1)
			{
				cvCircle(workImage,cvPoint(points2[currPoint].x,points2[currPoint].y),2,cvScalar(255,255,255));
				cvCircle(workImage,cvPoint(points2[currPoint].x,points2[currPoint].y),3,cvScalar(0));
				cvShowImage(FRAME_WINDOW,workImage);
				currPoint++;
			}
			key = cvWaitKey(30);
			switch (key)
			{
			case 32: // Space symbol
				if (ShowFrame(++currentFrame))
				{
					points2[0].x =-1;
					points2[1].x =-1;
					points2[2].x =-1;
					points2[3].x =-1;
					currPoint = 0;
				}
				else
				{
					currentFrame--;
				}
				break;
			case 8: // Backspace symbol
				if (currentFrame>0)
				{
					if (ShowFrame(--currentFrame))
					{
						points2[0].x =-1;
						points2[1].x =-1;
						points2[2].x =-1;
						points2[3].x =-1;
						currPoint = 0;
					}
					else
					{
						currentFrame++;
					}
				}
				break;
			}
		}

		i2 = cvCloneImage(workImage);
		// sorting points

		for (int i=1;i<4;i++)
		{

			if (points2[i].y<points2[0].y)
			{
				CvPoint temp;
				temp = points2[0];
				points2[0]=points2[i];
				points2[i]=temp;
			}
		}
		if (points2[1].x<points2[0].x)
		{
			CvPoint temp;
			temp = points2[0];
			points2[0]=points2[1];
			points2[1]=temp;
		}

		if (points2[3].x<points2[2].x)
		{
			CvPoint temp;
			temp = points2[3];
			points2[3]=points2[2];
			points2[2]=temp;
		}

		//end of sorting



		if (workImage) cvReleaseImage(&workImage);
		if (USE_DEINTERLACING)
		{
			workImage =  Deinterlace(i1);
			cvReleaseImage(&i1);
			i1=cvCloneImage(workImage);
			cvReleaseImage(&workImage);
			workImage =  Deinterlace(i2);
			cvReleaseImage(&i2);
			i2=cvCloneImage(workImage);
			cvReleaseImage(&workImage);
		}

		if (USE_UNDISTORTION)
		{
			workImage =  Undistort(i1,IntrinsicMatrix,DistortionCoeffs);
			cvReleaseImage(&i1);
			i1=cvCloneImage(workImage);
			cvReleaseImage(&workImage);
			workImage =  Undistort(i2,IntrinsicMatrix,DistortionCoeffs);
			cvReleaseImage(&i2);
			i2=cvCloneImage(workImage);
			cvReleaseImage(&workImage);
		}

		points3D = calc3DPoints(i1,i2,points1,points2,innerCornersCount,IntrinsicMatrix,DistortionCoeffs,
			!USE_UNDISTORTION,(float)(cvGetTrackbarPos(DEPTH_TRACKBAR,FRAME_WINDOW))/IDENTITY_DEPTH_VALUE,(float)(cvGetTrackbarPos(ERROR_TRACKBAR,FRAME_WINDOW))/IDENTITY_ERROR_VALUE);
		if (!points3D)
		{
			printf("\nUnable to find correct stereo correspondence, please try again\n");
			ShowFrame(currentFrame);
		}
	}
	while (!points3D);



	workImage = cvCloneImage(firstImage);
	if (USE_DEINTERLACING)
		workImage =  Deinterlace(frame);
	else
		workImage = cvCloneImage(frame);
	if (USE_UNDISTORTION)
	{
		undistortedImg = Undistort(workImage,IntrinsicMatrix,DistortionCoeffs);
		cvReleaseImage(&workImage);
		workImage = cvCloneImage(undistortedImg);
		cvReleaseImage(&undistortedImg);
	}
	cvReleaseImage(&firstImage);


	char* PATH = OUTPUT_DIRECTORY;
	FILE* f;
	char path[100];
	char path_frames[100];
	sprintf(path_frames,"%sframes.txt",PATH);
	remove(path_frames);
	char cmd[100];
#ifdef WIN32
	sprintf(cmd,"mkdir %s",PATH);
	system(cmd);
	sprintf(cmd,"del %s*.* /q",PATH);
	system(cmd);
	if (SAVE_SAMPLES)
	{
		sprintf(cmd,"mkdir %s",SAMPLES_PATH);
		system(cmd);
		sprintf(cmd,"del %s*.* /q",SAMPLES_PATH);
		system(cmd);
	}
#else

	sprintf(cmd,"mkdir %s",PATH);
	system(cmd);
	sprintf(cmd,"rm -f %s*.*",PATH);
	system(cmd);
	if (SAVE_SAMPLES)
	{
		sprintf(cmd,"mkdir %s",SAMPLES_PATH);
		system(cmd);
		sprintf(cmd,"rm -f %s*.*",SAMPLES_PATH);
		system(cmd);
	}
#endif

	do
	{
		if (workImage) cvReleaseImage(&workImage);
		frame = cvQueryFrame(capture);
		if (frame)
		{
			if (USE_DEINTERLACING)
				workImage =  Deinterlace(frame);
			else
				workImage = cvCloneImage(frame);

			if (USE_UNDISTORTION)
			{
				undistortedImg = Undistort(workImage,IntrinsicMatrix,DistortionCoeffs);
				cvReleaseImage(&workImage);
				workImage = cvCloneImage(undistortedImg);
				cvReleaseImage(&undistortedImg);
			}

			if (points3D)
			{

				newPoints = Find3DObject(workImage,points3D,innerCornersCount,IntrinsicMatrix,DistortionCoeffs,!USE_UNDISTORTION);

				if (newPoints)
				{
					bool areOnFrame=true; // are points on frame or not
					if (areOnFrame=true)
					{
						f = fopen(path_frames,"a");
						sprintf(path,"%s%d.jpg",PATH,currentFrame+1);
						fprintf(f,"%s,%d,%d,%d,%d,%d,%d,%d,%d,",path,newPoints[0].x,newPoints[0].y,newPoints[1].x,newPoints[1].y,newPoints[2].x,newPoints[2].y,newPoints[3].x,newPoints[3].y);
						fprintf(f,"%d,%d,%d,%d,%d,%d,%d,%d\n",newPoints[4].x,newPoints[4].y,newPoints[5].x,newPoints[5].y,newPoints[6].x,newPoints[6].y,newPoints[7].x,newPoints[7].y);
						fclose(f);

						if (DRAW_CHESSBOARD)
						{	
							CvPoint2D32f* ChessboardCorners = new CvPoint2D32f[innerCornersCount.height*innerCornersCount.width];
							cvFindChessboardCorners(workImage,innerCornersCount,ChessboardCorners);
							cvDrawChessboardCorners(workImage,innerCornersCount,ChessboardCorners,innerCornersCount.height*innerCornersCount.width,1);
							delete[] ChessboardCorners;
						}
						if (DRAW_CORNERS)
						{
							for (int i=0;i<8;i++)
							{
								cvCircle(workImage,newPoints[i],2,cvScalar(255,255,255));
								cvCircle(workImage,newPoints[i],3,cvScalar(0));	
							}
							cvLine(workImage,newPoints[0],newPoints[1], cvScalar(100,255,100));
							cvLine(workImage,newPoints[0],newPoints[2], cvScalar(100,255,100));
							cvLine(workImage,newPoints[1],newPoints[3], cvScalar(100,255,100));
							cvLine(workImage,newPoints[2],newPoints[3], cvScalar(100,255,100));
							cvLine(workImage,newPoints[4],newPoints[5], cvScalar(100,255,100));
							cvLine(workImage,newPoints[5],newPoints[7], cvScalar(100,255,100));
							cvLine(workImage,newPoints[6],newPoints[7], cvScalar(100,255,100));
							cvLine(workImage,newPoints[4],newPoints[6], cvScalar(100,255,100));
							cvLine(workImage,newPoints[0],newPoints[4], cvScalar(100,255,100));
							cvLine(workImage,newPoints[1],newPoints[5], cvScalar(100,255,100));
							cvLine(workImage,newPoints[2],newPoints[6], cvScalar(100,255,100));
							cvLine(workImage,newPoints[3],newPoints[7], cvScalar(100,255,100));
						}
						if (SAVE_DRAWINGS)
							cvSaveImage(path,workImage);
						else
						{
							if (USE_DEINTERLACING)
							{
								IplImage* tmp = Deinterlace(frame);
								if (USE_UNDISTORTION)
								{
									undistortedImg = Undistort(tmp,IntrinsicMatrix,DistortionCoeffs);
									cvReleaseImage(&tmp);
									tmp = cvCloneImage(undistortedImg);
									cvReleaseImage(&undistortedImg);
								}
								cvSaveImage(path,tmp);
								cvReleaseImage(&tmp);
							}
							else
							{
								if (!USE_UNDISTORTION)
									cvSaveImage(path,frame);
								else
								{
									undistortedImg = Undistort(frame,IntrinsicMatrix,DistortionCoeffs);
									cvSaveImage(path,undistortedImg);
									cvReleaseImage(&undistortedImg);
								}
							}
						}
						if (SAVE_SAMPLES)
						{
							sprintf(path,"%s%d.jpg",SAMPLES_PATH,currentFrame+1);
							IplImage* tmp;
							if (!USE_UNDISTORTION)
								tmp = GetSample3D(frame,newPoints);
							else
							{
								undistortedImg = Undistort(frame,IntrinsicMatrix,DistortionCoeffs);
								tmp = GetSample3D(undistortedImg,newPoints);
								cvReleaseImage(&undistortedImg);
							}
							if (tmp)
							{
								cvSaveImage(path,tmp);
								cvReleaseImage(&tmp);
							}
						}
						printf("Frame %d successfully saved to %s\n",currentFrame+1,path);
					}

					delete[] newPoints;
					newPoints = NULL;
				}
				else
					if (SAVE_ALL_FRAMES)
					{
						sprintf(path,"%s%d.jpg",PATH,currentFrame+1);
						cvSaveImage(path,workImage);
					}

			}


			cvShowImage(FRAME_WINDOW,workImage);
			key = cvWaitKey(30);
		}
		currentFrame++;
	}
	while (frame && (key!=27));

	if (capture)
		cvReleaseCapture(&capture);
}

//-------------

void createSamples3DObject2(int argc, char** argv)
{
	CvPoint points[4];

	CvPoint2D32f* relCoords = new CvPoint2D32f[4];
	CvPoint3D32f* objectPoints = new CvPoint3D32f[8];

	CvPoint * newPoints;
	int currentFrame=0;
	int key;
	IplImage* undistortedImg;

	if (CALIBRATE_CAMERA)
	{
		int v = IS_VIDEO_CAPTURE ? 10 : 7;
		char** c = new char*[v];
		c[0]=new char[11]; c[0]="calibration";
		c[1]=new char[2]; c[1]="-w";
		c[2] = new char[2]; sprintf(c[2],"%d",innerCornersCount.width);
		c[3]=new char[2]; c[3]="-h";
		c[4] = new char[2];  sprintf(c[4],"%d",innerCornersCount.height);
		c[5] = new char[2]; c[5]="-d";
		c[6] = new char[5];  sprintf(c[6],"%d",VIDEO_CALIBRATION_DELAY);
		c[7]=new char[2]; c[7]="-o";
		c[8]=new char[100];
		sprintf(c[8],"%scamera.yml",OUTPUT_DIRECTORY);
		if (IS_VIDEO_CAPTURE)
		{
			c[9]=new char[strlen(INPUT_VIDEO)+1];
			strcpy(c[9],INPUT_VIDEO);
		}

		calibrate(v,c);
		LoadCameraParams(c[8],&IntrinsicMatrix, &DistortionCoeffs);

		// Add functionality for camera calibration
	}

	int boxDepthValue = IDENTITY_DEPTH_VALUE;
	int initialBoxDepthValue = (MAX_INITAL_DEPTH_VALUE - MIN_INITAL_DEPTH_VALUE)/2;
	cvNamedWindow(FRAME_WINDOW,1);
	cvCreateTrackbar(DEPTH_TRACKBAR,FRAME_WINDOW,&boxDepthValue,MAX_DEPTH_VALUE,NULL);
	cvCreateTrackbar(INITIAL_DEPTH_TRACKBAR,FRAME_WINDOW,&initialBoxDepthValue,MAX_INITAL_DEPTH_VALUE-MIN_INITAL_DEPTH_VALUE,NULL);


	capture = IS_VIDEO_CAPTURE ? cvCreateFileCapture(INPUT_VIDEO) : cvCreateCameraCapture(0);

	if (!IS_VIDEO_CAPTURE)
	{
		cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH,640);
		cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT,480);
	}

	cvResizeWindow(FRAME_WINDOW,(int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH),(int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT));

	CvPoint2D32f* corners = new CvPoint2D32f[innerCornersCount.height*innerCornersCount.width];
	int count;

	do
	{
		if (IS_VIDEO_CAPTURE)
			cvSetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES,++currentFrame);
		if (workImage) cvReleaseImage(&workImage);
		frame = cvQueryFrame(capture);
		if (frame)
		{
			if (USE_DEINTERLACING)
				workImage =  Deinterlace(frame);
			else
				workImage = cvCloneImage(frame);


			if (USE_UNDISTORTION)
			{
			
				undistortedImg = Undistort(workImage,IntrinsicMatrix,DistortionCoeffs);
				cvReleaseImage(&workImage);
				workImage = cvCloneImage(undistortedImg);
				cvReleaseImage(&undistortedImg);
			}
			cvShowImage(FRAME_WINDOW,workImage);
			key = cvWaitKey(30);
			if (key==27) return;
		}

	}
	while ( frame && (((AUTOFIND_CHESSBOARD && IS_VIDEO_CAPTURE)|| (!IS_VIDEO_CAPTURE)) && cvFindChessboardCorners(workImage,innerCornersCount,corners,&count)==0));

	if (frame == NULL)
	{
		printf("\n Unable to load video with chessboard or connect to the camera");
		return;
	}

	delete[] corners;
	currentFrame--;

	IplImage* firstImage=cvCloneImage(workImage);

	cvShowImage(FRAME_WINDOW,workImage);

	points[0].x =-1;
	points[1].x =-1;
	points[2].x =-1;
	points[3].x =-1;

	printf("\nSelect the quadrangle region by selecting its vertices on the frame with a mouse\nUse Space and Backspace to go forward or backward (video only)\nPress Esc key to exit");
	cvSetMouseCallback(FRAME_WINDOW, on_mouse, &points);

	// Getting initial points position
	for (int currPoint=0;currPoint<4;)
	{
		if (points[currPoint].x != -1)
		{
			cvCircle(workImage,cvPoint(points[currPoint].x,points[currPoint].y),2,cvScalar(255,255,255));
			cvCircle(workImage,cvPoint(points[currPoint].x,points[currPoint].y),3,cvScalar(0));
			cvShowImage(FRAME_WINDOW,workImage);
			currPoint++;
		}
		key = cvWaitKey(30);
		if (IS_VIDEO_CAPTURE)
		{
			switch (key)
			{
			case 32: // Space symbol
				if (ShowFrame(++currentFrame))
				{
					points[0].x =-1;
					points[1].x =-1;
					points[2].x =-1;
					points[3].x =-1;
					currPoint = 0;
				}
				else
				{
					currentFrame--;
				}
				break;
			case 8: // Backspace symbol
				if (currentFrame>0)
				{
					if (ShowFrame(--currentFrame))
					{
						points[0].x =-1;
						points[1].x =-1;
						points[2].x =-1;
						points[3].x =-1;
						currPoint = 0;
					}
					else
					{
						currentFrame++;
					}
				}
				break;
			}
		}
	}

	// sorting points

	for (int i=1;i<4;i++)
	{

		if (points[i].y<points[0].y)
		{
			CvPoint temp;
			temp = points[0];
			points[0]=points[i];
			points[i]=temp;
		}
	}
	if (points[1].x<points[0].x)
	{
		CvPoint temp;
		temp = points[0];
		points[0]=points[1];
		points[1]=temp;
	}

	if (points[3].x<points[2].x)
	{
		CvPoint temp;
		temp = points[3];
		points[3]=points[2];
		points[2]=temp;
	}

	//end of sorting



	if (workImage) cvReleaseImage(&workImage);
	workImage = cvCloneImage(firstImage);
	if (USE_DEINTERLACING)
		workImage =  Deinterlace(frame);
	else
		workImage = cvCloneImage(frame);
	if (USE_UNDISTORTION)
	{
		undistortedImg = Undistort(workImage,IntrinsicMatrix,DistortionCoeffs);
		cvReleaseImage(&workImage);
		workImage = cvCloneImage(undistortedImg);
		cvReleaseImage(&undistortedImg);
	}
	cvReleaseImage(&firstImage);


	CvMat* Chessboard = CalcRelativePosition(workImage,points,&relCoords,IntrinsicMatrix, (float)(initialBoxDepthValue+MIN_INITAL_DEPTH_VALUE)/IDENTITY_INITIAL_DEPTH_VALUE);

	if (relCoords)
	{
		for (int i=0;i<4;i++)
		{
			objectPoints[i+4].x = objectPoints[i].x = relCoords[i].x;
			objectPoints[i+4].y = objectPoints[i].y = relCoords[i].y;
			objectPoints[i].z = (float)(initialBoxDepthValue+MIN_INITAL_DEPTH_VALUE)/IDENTITY_INITIAL_DEPTH_VALUE;
			objectPoints[i+4].z = objectPoints[i].z + (float)(boxDepthValue)/IDENTITY_DEPTH_VALUE; 
		}
	}
	else
	{
		delete[] objectPoints;
		objectPoints = NULL;
		return;
	}


	printf("\n\nCheck box depth and position\nUse Space and Backspace to go forward or backward (video only)\nPress Esc key to exit and Enter to approve");
	if (IS_VIDEO_CAPTURE)
		currentFrame--;
	//waiting for box depth approving
	IplImage* startImage = cvCloneImage(workImage);
	key=-1;
	while (key != 13)
	{
		key = cvWaitKey(30);
		if (key==27) 
			return;
		if (IS_VIDEO_CAPTURE)
		{
			switch (key)
			{
			case 32: // Space symbol
				if (ShowFrame(++currentFrame))
				{

				}
				else
				{
					currentFrame--;
				}
				break;
			case 8: // Backspace symbol
				if (currentFrame>0)
				{
					if (ShowFrame(--currentFrame))
					{

					}
					else
					{
						currentFrame++;
					}
				}
				break;
			default:
				ShowFrame(currentFrame);
				break;
			}
		}
		else
		{
			frame = cvQueryFrame(capture);
			if (frame)
			{
				if (USE_DEINTERLACING)
					workImage =  Deinterlace(frame);
				else
					workImage = cvCloneImage(frame);

				if (USE_UNDISTORTION)
				{
					undistortedImg = Undistort(workImage,IntrinsicMatrix,DistortionCoeffs);
					cvReleaseImage(&workImage);
					workImage = cvCloneImage(undistortedImg);
					cvReleaseImage(&undistortedImg);
				}


			}
			else
				return;
		}


		CalcRelativePosition(workImage,points,&relCoords,IntrinsicMatrix, (float)(initialBoxDepthValue+MIN_INITAL_DEPTH_VALUE)/IDENTITY_INITIAL_DEPTH_VALUE,Chessboard);


		for (int i=0;i<4;i++)
		{
			objectPoints[i+4].x = objectPoints[i].x = relCoords[i].x;
			objectPoints[i+4].y = objectPoints[i].y = relCoords[i].y;
			objectPoints[i].z = (float)(initialBoxDepthValue+MIN_INITAL_DEPTH_VALUE)/IDENTITY_INITIAL_DEPTH_VALUE;
			objectPoints[i+4].z = objectPoints[i].z + (float)(boxDepthValue)/IDENTITY_DEPTH_VALUE; 
		}



		for (int i=0;i<4;i++)
		{
			objectPoints[i].z = (float)(initialBoxDepthValue+MIN_INITAL_DEPTH_VALUE)/IDENTITY_INITIAL_DEPTH_VALUE;
			objectPoints[i+4].z = objectPoints[i].z +(float)(boxDepthValue)/IDENTITY_DEPTH_VALUE; 
		}
		newPoints = Find3DObject(workImage,objectPoints,innerCornersCount,IntrinsicMatrix,DistortionCoeffs,false);
		if (newPoints)
		{
 			for (int i=0;i<8;i++)
			{
				cvCircle(workImage,newPoints[i],2,cvScalar(255,255,255));
				cvCircle(workImage,newPoints[i],3,cvScalar(0));	
			}
			cvLine(workImage,newPoints[0],newPoints[1], cvScalar(100,255,100));
			cvLine(workImage,newPoints[0],newPoints[2], cvScalar(100,255,100));
			cvLine(workImage,newPoints[1],newPoints[3], cvScalar(100,255,100));
			cvLine(workImage,newPoints[2],newPoints[3], cvScalar(100,255,100));

			cvLine(workImage,newPoints[4],newPoints[5], cvScalar(50,150,50));
			cvLine(workImage,newPoints[5],newPoints[7], cvScalar(50,150,50));
			cvLine(workImage,newPoints[6],newPoints[7], cvScalar(50,150,50));
			cvLine(workImage,newPoints[4],newPoints[6], cvScalar(50,150,50));

			cvLine(workImage,newPoints[0],newPoints[4], cvScalar(50,150,50));
			cvLine(workImage,newPoints[1],newPoints[5], cvScalar(50,150,50));
			cvLine(workImage,newPoints[2],newPoints[6], cvScalar(50,150,50));
			cvLine(workImage,newPoints[3],newPoints[7], cvScalar(50,150,50));
		}
		cvShowImage(FRAME_WINDOW,workImage);
	}
	cvReleaseImage(&startImage);


	char* PATH = OUTPUT_DIRECTORY;
	FILE* f;
	char path[100];
	char path_frames[100];
	sprintf(path_frames,"%sframes.txt",PATH);
	remove(path_frames);
	char cmd[100];
#ifdef WIN32
	sprintf(cmd,"mkdir %s",PATH);
	system(cmd);
	sprintf(cmd,"del %s*.* /q",PATH);
	system(cmd);
	if (SAVE_SAMPLES)
	{
		sprintf(cmd,"mkdir %s",SAMPLES_PATH);
		system(cmd);
		sprintf(cmd,"del %s*.* /q",SAMPLES_PATH);
		system(cmd);
	}
#else

	sprintf(cmd,"mkdir %s",PATH);
	system(cmd);
	sprintf(cmd,"rm -f %s*.*",PATH);
	system(cmd);
	if (SAVE_SAMPLES)
	{
		sprintf(cmd,"mkdir %s",SAMPLES_PATH);
		system(cmd);
		sprintf(cmd,"rm -f %s*.*",SAMPLES_PATH);
		system(cmd);
	}
#endif
	if (IS_VIDEO_CAPTURE)
	{
		currentFrame = 0;
		cvSetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES,currentFrame);
	}
	do
	{
		if (workImage) cvReleaseImage(&workImage);
		frame = cvQueryFrame(capture);
		if (frame)
		{
			if (USE_DEINTERLACING)
				workImage =  Deinterlace(frame);
			else
				workImage = cvCloneImage(frame);

			if (USE_UNDISTORTION)
			{
				undistortedImg = Undistort(workImage,IntrinsicMatrix,DistortionCoeffs);
				cvReleaseImage(&workImage);
				workImage = cvCloneImage(undistortedImg);
				cvReleaseImage(&undistortedImg);
			}

			if (objectPoints)
			{
				newPoints = Find3DObject(workImage,objectPoints,innerCornersCount,IntrinsicMatrix,DistortionCoeffs,false);

				if (newPoints)
				{
					bool areOnFrame=true; // are points on frame or not
					if (areOnFrame=true)
					{
						f = fopen(path_frames,"a");
						sprintf(path,"%s%d.jpg",PATH,currentFrame+1);
						fprintf(f,"%s,%d,%d,%d,%d,%d,%d,%d,%d,",path,newPoints[0].x,newPoints[0].y,newPoints[1].x,newPoints[1].y,newPoints[2].x,newPoints[2].y,newPoints[3].x,newPoints[3].y);
						fprintf(f,"%d,%d,%d,%d,%d,%d,%d,%d\n",newPoints[4].x,newPoints[4].y,newPoints[5].x,newPoints[5].y,newPoints[6].x,newPoints[6].y,newPoints[7].x,newPoints[7].y);
						fclose(f);

						if (DRAW_CHESSBOARD)
						{	
							CvPoint2D32f* ChessboardCorners = new CvPoint2D32f[innerCornersCount.height*innerCornersCount.width];
							cvFindChessboardCorners(workImage,innerCornersCount,ChessboardCorners);
							cvDrawChessboardCorners(workImage,innerCornersCount,ChessboardCorners,innerCornersCount.height*innerCornersCount.width,1);
							delete[] ChessboardCorners;
						}
						if (DRAW_CORNERS)
						{
							for (int i=0;i<8;i++)
							{
								cvCircle(workImage,newPoints[i],2,cvScalar(255,255,255));
								cvCircle(workImage,newPoints[i],3,cvScalar(0));	
							}
							cvLine(workImage,newPoints[0],newPoints[1], cvScalar(100,255,100));
							cvLine(workImage,newPoints[0],newPoints[2], cvScalar(100,255,100));
							cvLine(workImage,newPoints[1],newPoints[3], cvScalar(100,255,100));
							cvLine(workImage,newPoints[2],newPoints[3], cvScalar(100,255,100));

							cvLine(workImage,newPoints[4],newPoints[5], cvScalar(50,150,50));
							cvLine(workImage,newPoints[5],newPoints[7], cvScalar(50,150,50));
							cvLine(workImage,newPoints[6],newPoints[7], cvScalar(50,150,50));
							cvLine(workImage,newPoints[4],newPoints[6], cvScalar(50,150,50));

							cvLine(workImage,newPoints[0],newPoints[4], cvScalar(50,150,50));
							cvLine(workImage,newPoints[1],newPoints[5], cvScalar(50,150,50));
							cvLine(workImage,newPoints[2],newPoints[6], cvScalar(50,150,50));
							cvLine(workImage,newPoints[3],newPoints[7], cvScalar(50,150,50));
						}
						if (SAVE_DRAWINGS)
							cvSaveImage(path,workImage);
						else
						{
							if (USE_DEINTERLACING)
							{
								IplImage* tmp = Deinterlace(frame);
								if (USE_UNDISTORTION)
								{
									undistortedImg = Undistort(tmp,IntrinsicMatrix,DistortionCoeffs);
									cvReleaseImage(&tmp);
									tmp = cvCloneImage(undistortedImg);
									cvReleaseImage(&undistortedImg);
								}
								cvSaveImage(path,tmp);
								cvReleaseImage(&tmp);
							}
							else
							{
								if (!USE_UNDISTORTION)
									cvSaveImage(path,frame);
								else
								{
									undistortedImg = Undistort(frame,IntrinsicMatrix,DistortionCoeffs);
									cvSaveImage(path,undistortedImg);
									cvReleaseImage(&undistortedImg);
								}
							}
						}
						if (SAVE_SAMPLES)
						{
							sprintf(path,"%s%d.jpg",SAMPLES_PATH,currentFrame+1);
							IplImage* tmp;
							if (!USE_UNDISTORTION)
								tmp = GetSample3D(frame,newPoints);
							else
							{
								undistortedImg = Undistort(frame,IntrinsicMatrix,DistortionCoeffs);
								tmp = GetSample3D(undistortedImg,newPoints);
								cvReleaseImage(&undistortedImg);
							}
							if (tmp)
							{
								cvSaveImage(path,tmp);
								cvReleaseImage(&tmp);
							}
						}
						printf("Frame %d successfully saved to %s\n",currentFrame+1,path);
					}

					delete[] newPoints;
					newPoints = NULL;
				}
				else
					if (SAVE_ALL_FRAMES)
					{
						sprintf(path,"%s%d.jpg",PATH,currentFrame+1);
						cvSaveImage(path,workImage);
					}

			}


			cvShowImage(FRAME_WINDOW,workImage);
			key = cvWaitKey(30);
		}
		currentFrame++;
	}
	while (frame && (key!=27));
	if (capture)
		cvReleaseCapture(&capture);
}
//-------------

// Command line arguments: <config_file>
int main( int argc, char** argv )
{
	if(argc != 2)
	{
		
		printf("Usage: createsamples arguments.yml\n");
		printf("Read readme.txt to get an additional information about the sample\n");
		return 0;
	}
	if (LoadApplicationParams(argv[1]))
	{
		if (USE_3D)
			createSamples3DObject2(argc,argv);
		else
			createSamples2DObject(argc,argv);
	}
	else
	{
		printf("\nUnable to load configuration file from %s\nClosing application...\n",argv[1]);
	}

	return 0;
}
