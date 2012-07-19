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
#include <float.h>
#include "object_tracker.h"
#include "tracker3D.h"

/////////----------------------------------------------------------------
/////////-----------------------------------------------
// return NULL if no there was no chessboard founded

// input: the same points on two images
// output: 3D coordinates of selected points
// boxDepthCoeff is box depth (relative to max of width or height)
// maxRelativeError - max reprojection error of upper-left corner of the schessboard (in chessboard square sizes)
CvPoint3D32f* calc3DPoints(const IplImage* _img1, const IplImage* _img2,CvPoint* points1,  CvPoint* points2, CvSize innerCornersCount,
							const CvMat* intrinsic_matrix, const CvMat* _distortion_coeffs, bool undistortImage,
							float boxDepthCoeff, float maxRelativeError)
{
	bool isOK = true;
	IplImage* img1 = cvCreateImage(cvSize(_img1->width,_img1->height),_img1->depth,_img1->nChannels);
	IplImage* img2 = cvCreateImage(cvSize(_img2->width,_img2->height),_img2->depth,_img2->nChannels);
	if (undistortImage)
	{
		cvUndistort2(_img1,img1,intrinsic_matrix,_distortion_coeffs);
		cvUndistort2(_img2,img2,intrinsic_matrix,_distortion_coeffs);
	}
	else
	{
		img1=cvCloneImage(_img1);
		img2=cvCloneImage(_img2);
	}

	CvMat* image_points1 = cvCreateMat(2,innerCornersCount.height*innerCornersCount.width,CV_64FC1);
	CvMat* image_points2 = cvCreateMat(2,innerCornersCount.height*innerCornersCount.width,CV_64FC1);
	CvMat* object_points = cvCreateMat(3,innerCornersCount.height*innerCornersCount.width,CV_64FC1);

	CvPoint2D32f* corners1 = new CvPoint2D32f[ innerCornersCount.height*innerCornersCount.width ];
	CvPoint2D32f* corners2 = new CvPoint2D32f[ innerCornersCount.height*innerCornersCount.width ];
	int count1 = 0;
	int count2 = 0;

	//Find chessboard corners
	int found1 = cvFindChessboardCorners(img1,innerCornersCount,corners1,&count1);
	int found2 = cvFindChessboardCorners(img2,innerCornersCount,corners2,&count2);
	if ((found1 == 0)||(found2 == 0))
	{
		delete[] corners1;
		corners1 = NULL;
		delete[] corners2;
		corners2 = NULL;
		cvReleaseMat(&image_points1);
		cvReleaseMat(&image_points2);
		cvReleaseMat(&object_points);
		cvReleaseImage(&img1);
		cvReleaseImage(&img2);
		return NULL;
	}

	CvMat* distortion_coeffs = cvCloneMat(_distortion_coeffs);
	cvZero(distortion_coeffs);

	IplImage* view_gray = cvCreateImage( cvGetSize(img1), 8, 1 );
	cvCvtColor(img1, view_gray, CV_BGR2GRAY );
	cvFindCornerSubPix( view_gray, corners1, count1, cvSize(11,11),cvSize(-1,-1), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
	cvReleaseImage( &view_gray );

	view_gray = cvCreateImage( cvGetSize(img2), 8, 1 );
	cvCvtColor(img2, view_gray, CV_BGR2GRAY );
	cvFindCornerSubPix( view_gray, corners2, count2, cvSize(11,11),cvSize(-1,-1), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
	cvReleaseImage( &view_gray );

	//assumes that chessboard squares are squared
	float step = 1.0f;

	// Sets object points and image points
	for (int i=0; i< innerCornersCount.height;i++)
		for (int j=0; j < innerCornersCount.width;j++)
		{
			object_points->data.db[(i*innerCornersCount.width+j)]=j*step;
			object_points->data.db[(i*innerCornersCount.width+j)+innerCornersCount.width*innerCornersCount.height]=i*step;
			object_points->data.db[(i*innerCornersCount.width+j)+2*innerCornersCount.width*innerCornersCount.height]=0.0f;

			image_points1->data.db[(i*innerCornersCount.width+j)]=(int)corners1[(i*innerCornersCount.width+j)].x;
			image_points1->data.db[(i*innerCornersCount.width+j)+innerCornersCount.width*innerCornersCount.height]=(int)corners1[(i*innerCornersCount.width+j)].y;

			image_points2->data.db[(i*innerCornersCount.width+j)]=(int)corners2[(i*innerCornersCount.width+j)].x;
			image_points2->data.db[(i*innerCornersCount.width+j)+innerCornersCount.width*innerCornersCount.height]=(int)corners2[(i*innerCornersCount.width+j)].y;

		}

		CvMat* R = cvCreateMat(3, 3, CV_64FC1);
		CvMat* T = cvCreateMat(3, 1, CV_64FC1);
		CvMat* R1 = cvCreateMat(3, 3, CV_64FC1);
		CvMat* R2 = cvCreateMat(3, 3, CV_64FC1);
		CvMat* R_1 = cvCreateMat(3, 3, CV_64FC1);
		CvMat* R_2 = cvCreateMat(3, 3, CV_64FC1);
		CvMat* P1 = cvCreateMat(3, 4, CV_64FC1);
		CvMat* P2 = cvCreateMat(3, 4, CV_64FC1);
		CvMat* T1 = cvCreateMat(3, 1, CV_64FC1);
		CvMat* T2 = cvCreateMat(3, 1, CV_64FC1);
		CvMat* Q = cvCreateMat(4, 4, CV_64FC1);
		CvMat* rotation_vector = cvCreateMat(3,1,CV_64FC1);
		CvMat* new_camera1 = cvCreateMat(3, 3, CV_64FC1);
		CvMat* new_camera2 = cvCreateMat(3, 3, CV_64FC1);


		//Calculating Exrinsic camera parameters
		cvFindExtrinsicCameraParams2(object_points,image_points1,intrinsic_matrix, distortion_coeffs,rotation_vector,T1);
		cvRodrigues2(rotation_vector,R1);

		cvFindExtrinsicCameraParams2(object_points,image_points2,intrinsic_matrix, distortion_coeffs,rotation_vector,T2);
		cvRodrigues2(rotation_vector,R2);

		//Finding rotation and translation vectors between two cameras
		cvGEMM(R2,R1,1,NULL,0,R,CV_GEMM_B_T);
		cvGEMM(R,T1,-1.0,T2,1,T);


		//Rectifying cameras
		cvStereoRectify( intrinsic_matrix, intrinsic_matrix,
			distortion_coeffs, distortion_coeffs,
			cvSize(img1->width,img1->height), R, T,
			R_1, R_2, P1, P2, Q, CV_CALIB_ZERO_DISPARITY);

		for (int i=0;i<3;i++)
			for (int j=0;j<3;j++)
			{
				new_camera1->data.db[i*new_camera1->cols+j]=P1->data.db[i*P1->cols+j];
				new_camera2->data.db[i*new_camera2->cols+j]=P2->data.db[i*P2->cols+j];
			}

		CvMat* oldPoints1 = cvCreateMat( 5,1, CV_32FC2 );
		CvMat* oldPoints2 = cvCreateMat( 5,1, CV_32FC2 );
		CvMat* newPoints1 = cvCreateMat( 5,1, CV_32FC2 );
		CvMat* newPoints2 = cvCreateMat( 5,1, CV_32FC2 );
		CvMat* t = cvCreateMat(3,1,CV_64FC1);

		CvPoint3D32f* xyd = new CvPoint3D32f[4];
		CvPoint3D32f test_point;

		//finding new x,y points and disparity
		for (int i=0;i<4;i++)
		{
			oldPoints1->data.fl[2*i]=(float)points1[i].x;
			oldPoints1->data.fl[2*i+1]=(float)points1[i].y;
			oldPoints2->data.fl[2*i]=(float)points2[i].x;
			oldPoints2->data.fl[2*i+1]=(float)points2[i].y;
		}
		oldPoints1->data.fl[8]=(float)(image_points1->data.db[0]);
		oldPoints1->data.fl[9]=(float)(image_points1->data.db[innerCornersCount.height*innerCornersCount.width]);
		oldPoints2->data.fl[8]=(float)(image_points2->data.db[0]);
		oldPoints2->data.fl[9]=(float)(image_points2->data.db[innerCornersCount.height*innerCornersCount.width]);
		cvUndistortPoints(oldPoints1,newPoints1,intrinsic_matrix,distortion_coeffs,R_1,P1);
		cvUndistortPoints(oldPoints2,newPoints2,intrinsic_matrix,distortion_coeffs,R_2,P2);

		for (int i=0;i<4;i++)
		{
			xyd[i].x=newPoints1->data.fl[2*i];
			xyd[i].y=newPoints1->data.fl[2*i+1];
			if (fabs(T->data.db[1]) < fabs(T->data.db[0])) 
				xyd[i].z=(newPoints2->data.fl[2*i])-(newPoints1->data.fl[2*i]);
			else
				xyd[i].z=(newPoints2->data.fl[2*i+1])-(newPoints1->data.fl[2*i+1]);
		}
		test_point.x = newPoints1->data.fl[8];
		test_point.y = newPoints1->data.fl[9];
		if (fabs(T->data.db[1]) < fabs(T->data.db[0])) 
			test_point.z=(newPoints2->data.fl[8])-(newPoints1->data.fl[8]);
		else
			test_point.z=(newPoints2->data.fl[9])-(newPoints1->data.fl[9]);

		CvPoint3D32f* result = new CvPoint3D32f[8];
		double x, y, z, w;

		//calculating 3D points
		for (int i=0;i<4;i++)
		{
			float d = xyd[i].z;
			x=(Q->data.db[0])*(xyd[i].x)+(Q->data.db[1])*(xyd[i].y)+(Q->data.db[2])*(d)+(Q->data.db[3]);
			y=(Q->data.db[4])*(xyd[i].x)+(Q->data.db[5])*(xyd[i].y)+(Q->data.db[6])*(d)+(Q->data.db[7]);
			z=(Q->data.db[8])*(xyd[i].x)+(Q->data.db[9])*(xyd[i].y)+(Q->data.db[10])*(d)+(Q->data.db[11]);
			w=(Q->data.db[12])*(xyd[i].x)+(Q->data.db[13])*(xyd[i].y)+(Q->data.db[14])*(d)+(Q->data.db[15]);
			if (w != 0)
			{
				result[i].x = (float)(x/w);
				result[i].y = (float)(y/w);
				result[i].z = (float)(z/w);
			}
			else
			{
				result[i].x = result[i].y = result[i].z = FLT_MAX;
				isOK=false;
			}

			// Calculating points coordinates in chessboard coordinate system

			t->data.db[0]=result[i].x;
			t->data.db[1]=result[i].y;
			t->data.db[2]=result[i].z;

			cvGEMM(R_1,t,1.0,T1,-1.0,t,CV_GEMM_A_T);
			cvGEMM(R1,t,1.0,NULL,0,t,CV_GEMM_A_T);


			result[i].x = (float)t->data.db[0];
			result[i].y = (float)t->data.db[1];
			result[i].z = (float)t->data.db[2];
		}

		x=(Q->data.db[0])*(test_point.x)+(Q->data.db[1])*(test_point.y)+(Q->data.db[2])*(test_point.z)+(Q->data.db[3]);
		y=(Q->data.db[4])*(test_point.x)+(Q->data.db[5])*(test_point.y)+(Q->data.db[6])*(test_point.z)+(Q->data.db[7]);
		z=(Q->data.db[8])*(test_point.x)+(Q->data.db[9])*(test_point.y)+(Q->data.db[10])*(test_point.z)+(Q->data.db[11]);
		w=(Q->data.db[12])*(test_point.x)+(Q->data.db[13])*(test_point.y)+(Q->data.db[14])*(test_point.z)+(Q->data.db[15]);
		if (w != 0)
		{
			x/=w;
			y/=w;
			z/=w;
		}
		else
		{
			x = y = z = FLT_MAX;
			isOK=false;
		}
		// Calculating test points coordinates in chessboard coordinate system

		t->data.db[0]=x;
		t->data.db[1]=y;
		t->data.db[2]=z;

		cvGEMM(R_1,t,1.0,T1,-1.0,t,CV_GEMM_A_T);
		cvGEMM(R1,t,1.0,NULL,0,t,CV_GEMM_A_T);


		x = t->data.db[0];
		y = t->data.db[1];
		z = t->data.db[2];

		if ((abs(x) > maxRelativeError) || (abs(y) > maxRelativeError) || (abs(z) > maxRelativeError))
			isOK = false;

		cvReleaseMat(&t);

		float maxSideSize = (fabs(result[0].x-result[1].x) > fabs(result[0].y-result[2].y)) ? fabs(result[0].x-result[1].x) : fabs(result[0].y-result[2].y);

		for (int i=0;i<4;i++)
		{
			result[i+4].x = result[i].x;
			result[i+4].y = result[i].y;
			result[i+4].z = (result[i].z > 0)? (result[i].z - boxDepthCoeff*maxSideSize) : (result[i].z + boxDepthCoeff*maxSideSize);
		}

		//Memory Cleanup
		delete[] xyd;
		delete[] corners1;
		delete[] corners2;
		cvReleaseMat(&oldPoints1);
		cvReleaseMat(&oldPoints2);
		cvReleaseMat(&newPoints1);
		cvReleaseMat(&newPoints2);
		cvReleaseMat(&object_points);
		cvReleaseMat(&T1);
		cvReleaseMat(&T2);
		cvReleaseMat(&rotation_vector);
		cvReleaseMat(&R1);
		cvReleaseMat(&R2);
		cvReleaseMat(&R_1);
		cvReleaseMat(&R_2);
		cvReleaseMat(&P1);
		cvReleaseMat(&P2);
		cvReleaseMat(&R);
		cvReleaseMat(&T);
		cvReleaseMat(&Q);
		cvReleaseMat(&new_camera1);
		cvReleaseMat(&new_camera2);
		cvReleaseImage(&img1);
		cvReleaseImage(&img2);
		cvReleaseMat(&distortion_coeffs);
		cvReleaseMat(&image_points1);
		cvReleaseMat(&image_points2);

		if (isOK)
			return result;
		delete[] result;
		return NULL;
}

CvPoint* Find3DObject(const IplImage* _img, const CvPoint3D32f* points, CvSize innerCornersCount,
					   const CvMat* intrinsic_matrix, const CvMat* _distortion_coeffs, bool undistortImage)
{
	IplImage* img;
	CvMat* mx = cvCreateMat( _img->height,_img->width, CV_32F );
	CvMat* my = cvCreateMat( _img->height,_img->width, CV_32F );
	CvMat* invmx = NULL;
	CvMat* invmy = NULL;


	if (undistortImage)
	{
		img = cvCreateImage(cvSize(_img->width,_img->height),_img->depth,_img->nChannels);
		cvInitUndistortMap(intrinsic_matrix,_distortion_coeffs,mx,my);
		cvRemap(_img,img,mx,my);
		InverseUndistortMap(mx,my,&invmx,&invmy,true);
	}
	else
	{
		img = cvCloneImage(_img);
	}

	CvMat* chessBoardPoints = cvCreateMat(2,innerCornersCount.height*innerCornersCount.width,CV_64FC1);

	CvPoint2D32f* corners = new CvPoint2D32f[ innerCornersCount.height*innerCornersCount.width ];
	int count = 0;


	//Find chessboard corners
	if (cvFindChessboardCorners(img,innerCornersCount,corners,&count)==0)
	{
		delete[] corners;
		corners = NULL;
		cvReleaseMat(&chessBoardPoints);
		cvReleaseImage(&img);
		cvReleaseMat(&mx);
		cvReleaseMat(&my);
		if (invmx)
			cvReleaseMat(&invmx);
		if (invmy)
			cvReleaseMat(&invmy);
		chessBoardPoints = NULL;

		return NULL;
	}

	CvMat* distortion_coeffs = cvCloneMat(_distortion_coeffs);
	cvZero(distortion_coeffs);

	IplImage* view_gray = cvCreateImage( cvGetSize(img), 8, 1 );
	cvCvtColor(img, view_gray, CV_BGR2GRAY );
	cvFindCornerSubPix( view_gray, corners, count, cvSize(11,11),cvSize(-1,-1), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
	cvReleaseImage( &view_gray );

	//assumes that chessboard squares are squared
	float step = 1.0f;

	// Sets object points and image points
	CvMat* object_points = cvCreateMat(3,innerCornersCount.height*innerCornersCount.width,CV_64FC1);
	for (int i=0; i< innerCornersCount.height;i++)
	{
		for (int j=0; j < innerCornersCount.width;j++)
		{
			object_points->data.db[(i*innerCornersCount.width+j)]=j*step;
			object_points->data.db[(i*innerCornersCount.width+j)+innerCornersCount.width*innerCornersCount.height]=i*step;
			object_points->data.db[(i*innerCornersCount.width+j)+2*innerCornersCount.width*innerCornersCount.height]=0.0f;

			chessBoardPoints->data.db[(i*innerCornersCount.width+j)]=corners[(i*innerCornersCount.width+j)].x;
			chessBoardPoints->data.db[(i*innerCornersCount.width+j)+innerCornersCount.width*innerCornersCount.height]=corners[(i*innerCornersCount.width+j)].y;

		}
	}

	CvMat* T = cvCreateMat(3, 1, CV_64FC1);
	CvMat* rotation_vector = cvCreateMat(3,1,CV_64FC1);

	cvFindExtrinsicCameraParams2(object_points,chessBoardPoints,intrinsic_matrix, distortion_coeffs,rotation_vector,T);

	CvMat* image_points3D = cvCreateMat(3,8,CV_64FC1);
	CvMat* image_points2D = cvCreateMat(2,8,CV_64FC1);

	for (int i=0;i<8;i++)
	{
		image_points3D->data.db[i]=points[i].x;
		image_points3D->data.db[i+8]=points[i].y;
		image_points3D->data.db[i+16]=points[i].z;

	}


	cvProjectPoints2(image_points3D,rotation_vector,T,intrinsic_matrix,distortion_coeffs,image_points2D);

	CvPoint* result = new CvPoint[8];

	for (int i=0;i<8;i++)
	{
		if (!undistortImage)
		{
			result[i].x=(int)(image_points2D->data.db[i]);
			result[i].y=(int)(image_points2D->data.db[i+8]);
		}
		else
		{
			result[i].x=(int)(invmx->data.fl[(int)(image_points2D->data.db[i])+(invmx->cols)*(int)(image_points2D->data.db[i+8])]);
			result[i].y=(int)(invmy->data.fl[(int)(image_points2D->data.db[i])+(invmx->cols)*(int)(image_points2D->data.db[i+8])]);
			if ((result[i].x < 0)||(result[i].y < 0))
			{
				delete[] result;
				result = NULL;
				break;
			}
		}
	}


	cvReleaseMat(&image_points3D);
	cvReleaseMat(&image_points2D);
	cvReleaseMat(&rotation_vector);
	cvReleaseMat(&T);
	cvReleaseMat(&object_points);
	cvReleaseMat(&chessBoardPoints);
	cvReleaseMat(&distortion_coeffs);
	cvReleaseImage(&img);
	cvReleaseMat(&mx);
	cvReleaseMat(&my);
	cvReleaseMat(&invmx);
	cvReleaseMat(&invmy);
	delete[] corners;

	return result;

}
//----------
IplImage* GetSample3D(const IplImage* img, CvPoint* points)
{
	int minx = img->width;
	int miny = img->height;
	int maxx = 0;
	int maxy = 0;
	for (int i=0;i<8;i++)
	{
		if (points[i].x < minx)
			minx=points[i].x;
		if (points[i].y < miny)
			miny=points[i].y;
		if (points[i].x > maxx)
			maxx=points[i].x;
		if (points[i].y > maxy)
			maxy=points[i].y;
	}
	IplImage* result;
	if ((maxx < img->width) && (maxy< img->height) && (maxx > minx) && (maxy > miny))
	{
		IplImage* workImage = cvCloneImage(img);
		cvSetImageROI(workImage,cvRect(minx,miny,maxx-minx,maxy-miny));
		result = cvCreateImage(cvSize(maxx-minx,maxy-miny),workImage->depth,workImage->nChannels);
		cvConvert(workImage,result);
		cvReleaseImage(&workImage);
	}
	else 
		result = NULL;
	return result;
}


#if 0
// test code
int main( int argc, char** argv )
{

	CvMat* IntrinsicMatrix;
	CvMat* DistortionCoeffs;
	if (LoadCameraParams("f:\\_camera.yml",&IntrinsicMatrix,&DistortionCoeffs))
	{
		CvPoint* p1 = new CvPoint[4];
		CvPoint* p2 = new CvPoint[4];

		p1[0].x=481;
		p1[0].y=251;
		p1[1].x=630;
		p1[1].y=283;
		p1[2].x=487;
		p1[2].y=308;
		p1[3].x=619;
		p1[3].y=354;

		p2[0].x=504;
		p2[0].y=242;
		p2[1].x=632;
		p2[1].y=299;
		p2[2].x=496;
		p2[2].y=297;
		p2[3].x=621;
		p2[3].y=359;


		CvSize _innerCornersCount = cvSize(8,6);

		IplImage* i2= cvLoadImage("f:\\451.jpg");
		IplImage* i1= cvLoadImage("f:\\2.jpg");
		CvCapture* capture = cvCreateFileCapture("f:\\DATA\\DoorHandle_xvid.avi");
		cvSetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES,200);

		IplImage* itest;
		IplImage* _itest = cvLoadImage("f:/2.jpg",1);
		CvPoint3D32f* points = new CvPoint3D32f[8];
		CvPoint* newPoints = new CvPoint[8];
		points = calc3DPoints(i1,i2,p1,p2,_innerCornersCount,IntrinsicMatrix,DistortionCoeffs,false);
		cvNamedWindow("res",1);
		while (points && ((_itest = cvQueryFrame(capture))!=NULL))
		{

			itest = cvCloneImage(_itest);
			newPoints = Find3DObject(itest,points,_innerCornersCount,IntrinsicMatrix,DistortionCoeffs,false);

			if (newPoints)
			{
				for (int i=0;i<8;i++)
				{
					cvCircle(itest,newPoints[i],2,cvScalar(255,255,255));
					cvCircle(itest,newPoints[i],3,cvScalar(0));	
				}
				cvLine(itest,newPoints[0],newPoints[1], cvScalar(100,255,100));
				cvLine(itest,newPoints[0],newPoints[2], cvScalar(100,255,100));
				cvLine(itest,newPoints[1],newPoints[3], cvScalar(100,255,100));
				cvLine(itest,newPoints[2],newPoints[3], cvScalar(100,255,100));
				cvLine(itest,newPoints[4],newPoints[5], cvScalar(100,255,100));
				cvLine(itest,newPoints[5],newPoints[7], cvScalar(100,255,100));
				cvLine(itest,newPoints[6],newPoints[7], cvScalar(100,255,100));
				cvLine(itest,newPoints[4],newPoints[6], cvScalar(100,255,100));
				cvLine(itest,newPoints[0],newPoints[4], cvScalar(100,255,100));
				cvLine(itest,newPoints[1],newPoints[5], cvScalar(100,255,100));
				cvLine(itest,newPoints[2],newPoints[6], cvScalar(100,255,100));
				cvLine(itest,newPoints[3],newPoints[7], cvScalar(100,255,100));

			}
			cvShowImage("res",itest);
			cvWaitKey(10);
			cvReleaseImage(&itest);
			delete[] newPoints;
			newPoints=NULL;
		}
	}


	return 0;
}
#endif
