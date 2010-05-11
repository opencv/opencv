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

#include "cvtest.h"


class CV_DefaultNewCameraMatrixTest : public CvArrTest
{
public:
	CV_DefaultNewCameraMatrixTest();
protected:
	int prepare_test_case (int test_case_idx);
	void prepare_to_validation( int test_case_idx );
	void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
	void run_func();

private:
	cv::Size img_size;
	cv::Mat camera_mat;
	cv::Mat new_camera_mat;

	int matrix_type;

	bool center_principal_point;

	static const int MAX_X = 2048;
	static const int MAX_Y = 2048;
	static const int MAX_VAL = 10000;
};

CV_DefaultNewCameraMatrixTest::CV_DefaultNewCameraMatrixTest() : CvArrTest("undistort-getDefaultNewCameraMatrix","getDefaultNewCameraMatrix")
{
	test_array[INPUT].push(NULL);
	test_array[OUTPUT].push(NULL);
	test_array[REF_OUTPUT].push(NULL);
}

void CV_DefaultNewCameraMatrixTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
	CvArrTest::get_test_array_types_and_sizes(test_case_idx,sizes,types);
	CvRNG* rng = ts->get_rng();
	matrix_type = types[INPUT][0] = types[OUTPUT][0]= types[REF_OUTPUT][0] = cvTsRandInt(rng)%2 ? CV_64F : CV_32F;
	sizes[INPUT][0] = sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize(3,3);
}

int CV_DefaultNewCameraMatrixTest::prepare_test_case(int test_case_idx)
{
	int code = CvArrTest::prepare_test_case( test_case_idx );

	if (code <= 0)
		return code;

	CvRNG* rng = ts->get_rng();

	img_size.width = cvTsRandInt(rng) % MAX_X + 1;
	img_size.height = cvTsRandInt(rng) % MAX_Y + 1;

	center_principal_point = ((cvTsRandInt(rng) % 2)!=0);

	// Generating camera_mat matrix
	double sz = MAX(img_size.width, img_size.height);
	double aspect_ratio = cvTsRandReal(rng)*0.6 + 0.7;
	double a[9] = {0,0,0,0,0,0,0,0,1};
	CvMat _a = cvMat(3,3,CV_64F,a);
	a[2] = (img_size.width - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
	a[5] = (img_size.height - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
	a[0] = sz/(0.9 - cvTsRandReal(rng)*0.6);
	a[4] = aspect_ratio*a[0];

	//Copying into input array
	CvMat* _a0 = &test_mat[INPUT][0];
	cvTsConvert( &_a, _a0 );
	camera_mat = _a0;
	//new_camera_mat = camera_mat;

	return code;

}

void CV_DefaultNewCameraMatrixTest::run_func()
{
	new_camera_mat = cv::getDefaultNewCameraMatrix(camera_mat,img_size,center_principal_point);
}

void CV_DefaultNewCameraMatrixTest::prepare_to_validation( int /*test_case_idx*/ )
{
	const CvMat* src = &test_mat[INPUT][0];
	CvMat* dst = &test_mat[REF_OUTPUT][0];
	CvMat* test_output = &test_mat[OUTPUT][0];
	CvMat output = new_camera_mat;
	cvTsConvert( &output, test_output );
	if (!center_principal_point)
	{
		cvCopy(src,dst);
	}
	else
	{
		double a[9] = {0,0,0,0,0,0,0,0,1};
		CvMat _a = cvMat(3,3,CV_64F,a);
		if (matrix_type == CV_64F)
		{
			a[0] = ((double*)(src->data.ptr + src->step*0))[0];
			a[4] = ((double*)(src->data.ptr + src->step*1))[1];
		}
		else
		{
			a[0] = (double)((float*)(src->data.ptr + src->step*0))[0];
			a[4] = (double)((float*)(src->data.ptr + src->step*1))[1];
		}
		a[2] = (img_size.width - 1)*0.5;
		a[5] = (img_size.height - 1)*0.5;
		cvTsConvert( &_a, dst );
	}
}

CV_DefaultNewCameraMatrixTest default_new_camera_matrix_test; 

//---------

class CV_UndistortPointsTest : public CvArrTest
{
public:
	CV_UndistortPointsTest();
protected:
	int prepare_test_case (int test_case_idx);
	void prepare_to_validation( int test_case_idx );
	void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
	double get_success_error_level( int test_case_idx, int i, int j );
	void run_func();
	void cvTsDistortPoints(const CvMat* _src, CvMat* _dst, const CvMat* _cameraMatrix,
		const CvMat* _distCoeffs,
		const CvMat* matR, const CvMat* matP);

private:
	bool useCPlus;
	bool useDstMat;
	static const int N_POINTS = 10;
	static const int MAX_X = 2048;
	static const int MAX_Y = 2048;

	bool zero_new_cam;
	bool zero_distortion;
	bool zero_R;

	cv::Size img_size;
	cv::Mat dst_points_mat;

	cv::Mat camera_mat;
	cv::Mat R;
	cv::Mat P;
	cv::Mat distortion_coeffs;
	cv::Mat src_points;
	std::vector<cv::Point2f> dst_points;
};

CV_UndistortPointsTest::CV_UndistortPointsTest() : CvArrTest("undistort-points","cvUndistortPoints")
{
	test_array[INPUT].push(NULL); // points matrix
	test_array[INPUT].push(NULL); // camera matrix
	test_array[INPUT].push(NULL); // distortion coeffs
	test_array[INPUT].push(NULL); // R matrix
	test_array[INPUT].push(NULL); // P matrix
	test_array[OUTPUT].push(NULL); // distorted dst points
	test_array[TEMP].push(NULL); // dst points
	test_array[REF_OUTPUT].push(NULL);
}

void CV_UndistortPointsTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
	CvArrTest::get_test_array_types_and_sizes(test_case_idx,sizes,types);
	CvRNG* rng = ts->get_rng();
	useCPlus = ((cvTsRandInt(rng) % 2)!=0);
	//useCPlus = 0;
	if (useCPlus)
	{
		types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = types[TEMP][0]= CV_32FC2;
	}
	else
	{
		types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = types[TEMP][0]= cvTsRandInt(rng)%2 ? CV_64FC2 : CV_32FC2;
	}
	types[INPUT][1] = cvTsRandInt(rng)%2 ? CV_64F : CV_32F;
	types[INPUT][2] = cvTsRandInt(rng)%2 ? CV_64F : CV_32F;
	types[INPUT][3] = cvTsRandInt(rng)%2 ? CV_64F : CV_32F;
	types[INPUT][4] = cvTsRandInt(rng)%2 ? CV_64F : CV_32F;

	sizes[INPUT][0] = sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = sizes[TEMP][0]= cvTsRandInt(rng)%2 ? cvSize(1,N_POINTS) : cvSize(N_POINTS,1); 
	sizes[INPUT][1] = sizes[INPUT][3] = cvSize(3,3);
	sizes[INPUT][4] = cvTsRandInt(rng)%2 ? cvSize(3,3) : cvSize(4,3);

	if (cvTsRandInt(rng)%2)
	{
		if (cvTsRandInt(rng)%2)
		{
			sizes[INPUT][2] = cvSize(1,4);
		}
		else
		{
			sizes[INPUT][2] = cvSize(1,5);
		}
	}
	else
	{
		if (cvTsRandInt(rng)%2) 
		{
			sizes[INPUT][2] = cvSize(4,1);
		}
		else
		{
			sizes[INPUT][2] = cvSize(5,1);
		}
	}
}

int CV_UndistortPointsTest::prepare_test_case(int test_case_idx)
{
	CvRNG* rng = ts->get_rng();
	int code = CvArrTest::prepare_test_case( test_case_idx );

	if (code <= 0)
		return code;

	useDstMat = (cvTsRandInt(rng) % 2) == 0;

	img_size.width = cvTsRandInt(rng) % MAX_X + 1;
	img_size.height = cvTsRandInt(rng) % MAX_Y + 1;
	int dist_size = test_mat[INPUT][2].cols > test_mat[INPUT][2].rows ? test_mat[INPUT][2].cols : test_mat[INPUT][2].rows;
	double cam[9] = {0,0,0,0,0,0,0,0,1};
	double* dist = new double[dist_size ];
	double* proj = new double[test_mat[INPUT][4].cols * test_mat[INPUT][4].rows];
	double* points = new double[N_POINTS*2];

	CvMat _camera = cvMat(3,3,CV_64F,cam);
	CvMat _distort = cvMat(test_mat[INPUT][2].rows,test_mat[INPUT][2].cols,CV_64F,dist);
	CvMat _proj = cvMat(test_mat[INPUT][4].rows,test_mat[INPUT][4].cols,CV_64F,proj);
	CvMat _points= cvMat(test_mat[INPUT][0].rows,test_mat[INPUT][0].cols,CV_64FC2, points);

	for (int i=0;i<test_mat[INPUT][4].cols * test_mat[INPUT][4].rows;i++)
	{
		proj[i] = 0;
	}

	//Generating points
	for (int i=0;i<N_POINTS;i++)
	{
		points[2*i] = cvTsRandReal(rng)*img_size.width;
		points[2*i+1] = cvTsRandReal(rng)*img_size.height;
	}



	//Generating camera matrix
	double sz = MAX(img_size.width,img_size.height);
	double aspect_ratio = cvTsRandReal(rng)*0.6 + 0.7;
	cam[2] = (img_size.width - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
	cam[5] = (img_size.height - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
	cam[0] = sz/(0.9 - cvTsRandReal(rng)*0.6);
	cam[4] = aspect_ratio*cam[0];

	//Generating distortion coeffs
	dist[0] = cvTsRandReal(rng)*0.06 - 0.03;
	dist[1] = cvTsRandReal(rng)*0.06 - 0.03;
	if( dist[0]*dist[1] > 0 )
		dist[1] = -dist[1];
	if( cvTsRandInt(rng)%4 != 0 )
	{
		dist[2] = cvTsRandReal(rng)*0.004 - 0.002;
		dist[3] = cvTsRandReal(rng)*0.004 - 0.002;
		if (dist_size > 4)
			dist[4] = cvTsRandReal(rng)*0.004 - 0.002;
	}
	else
	{
		dist[2] = dist[3] = 0;
		if (dist_size > 4)
			dist[4] = 0;
	}

	//Generating P matrix (projection)
	if ( test_mat[INPUT][4].cols != 4)
	{
		proj[8] = 1;
		if (cvTsRandInt(rng)%2 == 0) // use identity new camera matrix
		{
			proj[0] = 1;
			proj[4] = 1;
		}
		else
		{
			proj[0] = cam[0] + (cvTsRandReal(rng) - (double)0.5)*0.2*cam[0]; //10%
			proj[4] = cam[4] + (cvTsRandReal(rng) - (double)0.5)*0.2*cam[4]; //10%
			proj[2] = cam[2] + (cvTsRandReal(rng) - (double)0.5)*0.3*img_size.width; //15%
			proj[5] = cam[5] + (cvTsRandReal(rng) - (double)0.5)*0.3*img_size.height; //15%
		}
	}
	else
	{
		proj[10] = 1;
		proj[0] = cam[0] + (cvTsRandReal(rng) - (double)0.5)*0.2*cam[0]; //10%
		proj[5] = cam[4] + (cvTsRandReal(rng) - (double)0.5)*0.2*cam[4]; //10%
		proj[2] = cam[2] + (cvTsRandReal(rng) - (double)0.5)*0.3*img_size.width; //15%
		proj[6] = cam[5] + (cvTsRandReal(rng) - (double)0.5)*0.3*img_size.height; //15%

		proj[3] = (img_size.height + img_size.width - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
		proj[7] = (img_size.height + img_size.width - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
		proj[11] = (img_size.height + img_size.width - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
	}

	//Generating R matrix
	CvMat* _rot = cvCreateMat(3,3,CV_64F);
	CvMat* rotation = cvCreateMat(1,3,CV_64F);
	rotation->data.db[0] = CV_PI*(cvTsRandReal(rng) - (double)0.5); // phi
	rotation->data.db[1] = CV_PI*(cvTsRandReal(rng) - (double)0.5); // ksi
	rotation->data.db[2] = CV_PI*(cvTsRandReal(rng) - (double)0.5); //khi
	cvRodrigues2(rotation,_rot);
	cvReleaseMat(&rotation);

	//copying data
	//src_points = &_points;
	CvMat* dst = &test_mat[INPUT][0];
	cvTsConvert( &_points, dst);
	dst = &test_mat[INPUT][1];
	cvTsConvert( &_camera, dst);
	dst = &test_mat[INPUT][2];
	cvTsConvert( &_distort, dst);
	dst = &test_mat[INPUT][3];
	cvTsConvert( _rot, dst);
	dst = &test_mat[INPUT][4];
	cvTsConvert( &_proj, dst);

	zero_distortion = (cvRandInt(rng)%2) == 0 ? false : true;
	zero_new_cam = (cvRandInt(rng)%2) == 0 ? false : true;
	zero_R = (cvRandInt(rng)%2) == 0 ? false : true;


	cvReleaseMat(&_rot);

	if (useCPlus)
	{
		CvMat* temp = cvCreateMat(test_mat[INPUT][0].rows,test_mat[INPUT][0].cols,CV_32FC2);
		for (int i=0;i<test_mat[INPUT][0].rows*test_mat[INPUT][0].cols*2;i++)
			temp->data.fl[i] = (float)_points.data.db[i];


		src_points = cv::Mat(temp,true);

		cvReleaseMat(&temp);

		camera_mat = &test_mat[INPUT][1];
		distortion_coeffs = &test_mat[INPUT][2];
		R = &test_mat[INPUT][3];
		P = &test_mat[INPUT][4];
	}
	delete[] dist;
	delete[] proj;
	delete[] points;

	return code;
}

void CV_UndistortPointsTest::prepare_to_validation(int /*test_case_idx*/)
{
	int dist_size = test_mat[INPUT][2].cols > test_mat[INPUT][2].rows ? test_mat[INPUT][2].cols : test_mat[INPUT][2].rows;
	double cam[9] = {0,0,0,0,0,0,0,0,1};
	double rot[9] = {1,0,0,0,1,0,0,0,1};
	double* dist = new double[dist_size ];
	double* proj = new double[test_mat[INPUT][4].cols * test_mat[INPUT][4].rows];
	double* points = new double[N_POINTS*2];
	double* r_points = new double[N_POINTS*2];
	//Run reference calculations
	CvMat ref_points= cvMat(test_mat[INPUT][0].rows,test_mat[INPUT][0].cols,CV_64FC2,r_points);
	CvMat _camera = cvMat(3,3,CV_64F,cam);
	CvMat _rot = cvMat(3,3,CV_64F,rot);
	CvMat _distort = cvMat(test_mat[INPUT][2].rows,test_mat[INPUT][2].cols,CV_64F,dist);
	CvMat _proj = cvMat(test_mat[INPUT][4].rows,test_mat[INPUT][4].cols,CV_64F,proj);
	CvMat _points= cvMat(test_mat[TEMP][0].rows,test_mat[TEMP][0].cols,CV_64FC2,points);


	cvTsConvert(&test_mat[INPUT][1],&_camera);
	cvTsConvert(&test_mat[INPUT][2],&_distort);
	cvTsConvert(&test_mat[INPUT][3],&_rot);
	cvTsConvert(&test_mat[INPUT][4],&_proj);

	if (useCPlus)
	{
		if (useDstMat)
		{
			CvMat temp = dst_points_mat;
			for (int i=0;i<N_POINTS*2;i++)
			{
				points[i] = temp.data.fl[i];
			}		
		}

		else
		{

			for (int i=0;i<N_POINTS;i++)
			{
				points[2*i] = dst_points[i].x;
				points[2*i+1] = dst_points[i].y;
			}
		}
	}
	else
	{
		cvTsConvert(&test_mat[TEMP][0],&_points);
	}

	CvMat* input2;
	CvMat* input3;
	CvMat* input4;
	input2 = zero_distortion ? 0 : &_distort;
	input3 = zero_R ? 0 : &_rot;
	input4 = zero_new_cam ? 0 : &_proj;
	cvTsDistortPoints(&_points,&ref_points,&_camera,input2,input3,input4);

	CvMat* dst = &test_mat[REF_OUTPUT][0];
	cvTsConvert(&ref_points,dst);

	cvCopy(&test_mat[INPUT][0],&test_mat[OUTPUT][0]);

	delete[] dist;
	delete[] proj;
	delete[] points;
	delete[] r_points;
}

void CV_UndistortPointsTest::run_func()
{

	if (useCPlus)
	{
		cv::Mat input2,input3,input4;
		input2 = zero_distortion ? cv::Mat() : cv::Mat(&test_mat[INPUT][2]);
		input3 = zero_R ? cv::Mat() : cv::Mat(&test_mat[INPUT][3]);
		input4 = zero_new_cam ? cv::Mat() : cv::Mat(&test_mat[INPUT][4]);

		if (useDstMat)
		{
			//cv::undistortPoints(src_points,dst_points_mat,camera_mat,distortion_coeffs,R,P);
			cv::undistortPoints(src_points,dst_points_mat,camera_mat,input2,input3,input4);
		}
		else
		{
			//cv::undistortPoints(src_points,dst_points,camera_mat,distortion_coeffs,R,P);
			cv::undistortPoints(src_points,dst_points,camera_mat,input2,input3,input4);
		}
	}
	else
	{
		CvMat* input2;
		CvMat* input3;
		CvMat* input4;
		input2 = zero_distortion ? 0 : &test_mat[INPUT][2];
		input3 = zero_R ? 0 : &test_mat[INPUT][3];
		input4 = zero_new_cam ? 0 : &test_mat[INPUT][4];
		cvUndistortPoints(&test_mat[INPUT][0],&test_mat[TEMP][0],&test_mat[INPUT][1],input2,input3,input4);
	}

}

void CV_UndistortPointsTest::cvTsDistortPoints(const CvMat* _src, CvMat* _dst, const CvMat* _cameraMatrix,
											   const CvMat* _distCoeffs,
											   const CvMat* matR, const CvMat* matP)
{
	double a[9];

	CvMat* __P;
	if ((!matP)||(matP->cols == 3))
		__P = cvCreateMat(3,3,CV_64F);
	else
		__P = cvCreateMat(3,4,CV_64F);
	if (matP)
	{
		cvTsConvert(matP,__P);
	}
	else
	{
		cvZero(__P);
		__P->data.db[0] = 1;
		__P->data.db[4] = 1;
		__P->data.db[8] = 1;
	}
	CvMat* __R = cvCreateMat(3,3,CV_64F);;
	if (matR)
	{
		cvCopy(matR,__R);
	}
	else
	{
		cvZero(__R);
		__R->data.db[0] = 1;
		__R->data.db[4] = 1;
		__R->data.db[8] = 1;
	}
	for (int i=0;i<N_POINTS;i++)
	{
		int movement = __P->cols > 3 ? 1 : 0;
		double x = (_src->data.db[2*i]-__P->data.db[2])/__P->data.db[0];
		double y = (_src->data.db[2*i+1]-__P->data.db[5+movement])/__P->data.db[4+movement];
		CvMat inverse = cvMat(3,3,CV_64F,a);
		cvInvert(__R,&inverse);
		double w1 = x*inverse.data.db[6]+y*inverse.data.db[7]+inverse.data.db[8];
		double _x = (x*inverse.data.db[0]+y*inverse.data.db[1]+inverse.data.db[2])/w1;
		double _y = (x*inverse.data.db[3]+y*inverse.data.db[4]+inverse.data.db[5])/w1;

		//Distortions

		double __x = _x;
		double __y = _y;
		if (_distCoeffs)
		{
			double r2 = _x*_x+_y*_y;

			__x = _x*(1+_distCoeffs->data.db[0]*r2+_distCoeffs->data.db[1]*r2*r2)+
				2*_distCoeffs->data.db[2]*_x*_y+_distCoeffs->data.db[3]*(r2+2*_x*_x);
			__y = _y*(1+_distCoeffs->data.db[0]*r2+_distCoeffs->data.db[1]*r2*r2)+
				2*_distCoeffs->data.db[3]*_x*_y+_distCoeffs->data.db[2]*(r2+2*_y*_y);
			if ((_distCoeffs->cols > 4) || (_distCoeffs->rows > 4))
			{
				__x+=_x*_distCoeffs->data.db[4]*r2*r2*r2;
				__y+=_y*_distCoeffs->data.db[4]*r2*r2*r2;
			}
		}


		_dst->data.db[2*i] = __x*_cameraMatrix->data.db[0]+_cameraMatrix->data.db[2];
		_dst->data.db[2*i+1] = __y*_cameraMatrix->data.db[4]+_cameraMatrix->data.db[5];

	}

	cvReleaseMat(&__R);
	cvReleaseMat(&__P);

}

double CV_UndistortPointsTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
	return 5e-2;
}

CV_UndistortPointsTest undistort_points_test;

//------------------------------------------------------

class CV_InitUndistortRectifyMapTest : public CvArrTest
{
public:
	CV_InitUndistortRectifyMapTest();
protected:
	int prepare_test_case (int test_case_idx);
	void prepare_to_validation( int test_case_idx );
	void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
	double get_success_error_level( int test_case_idx, int i, int j );
	void run_func();

private:
	bool useCPlus;
	static const int N_POINTS = 100;
	static const int MAX_X = 2048;
	static const int MAX_Y = 2048;
	bool zero_new_cam;
	bool zero_distortion;
	bool zero_R;


	cv::Size img_size;

	cv::Mat camera_mat;
	cv::Mat R;
	cv::Mat new_camera_mat;
	cv::Mat distortion_coeffs;
	cv::Mat mapx;
	cv::Mat mapy;
	CvMat* _mapx;
	CvMat* _mapy;
	int mat_type;
};

CV_InitUndistortRectifyMapTest::CV_InitUndistortRectifyMapTest() : CvArrTest("undistort-undistort_rectify_map","cvInitUndistortRectifyMap")
{
	test_array[INPUT].push(NULL); // test points matrix
	test_array[INPUT].push(NULL); // camera matrix
	test_array[INPUT].push(NULL); // distortion coeffs
	test_array[INPUT].push(NULL); // R matrix
	test_array[INPUT].push(NULL); // new camera matrix
	test_array[OUTPUT].push(NULL); // distorted dst points
	test_array[REF_OUTPUT].push(NULL);
}

void CV_InitUndistortRectifyMapTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
	CvArrTest::get_test_array_types_and_sizes(test_case_idx,sizes,types);
	CvRNG* rng = ts->get_rng();
	useCPlus = ((cvTsRandInt(rng) % 2)!=0);
	//useCPlus = 0;
	types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_64FC2;

	types[INPUT][1] = cvTsRandInt(rng)%2 ? CV_64F : CV_32F;
	types[INPUT][2] = cvTsRandInt(rng)%2 ? CV_64F : CV_32F;
	types[INPUT][3] = cvTsRandInt(rng)%2 ? CV_64F : CV_32F;
	types[INPUT][4] = cvTsRandInt(rng)%2 ? CV_64F : CV_32F;

	sizes[INPUT][0] = sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize(N_POINTS,1); 
	sizes[INPUT][1] = sizes[INPUT][3] = cvSize(3,3);
	sizes[INPUT][4] = cvSize(3,3);

	if (cvTsRandInt(rng)%2)
	{
		if (cvTsRandInt(rng)%2)
		{
			sizes[INPUT][2] = cvSize(1,4);
		}
		else
		{
			sizes[INPUT][2] = cvSize(1,5);
		}
	}
	else
	{
		if (cvTsRandInt(rng)%2) 
		{
			sizes[INPUT][2] = cvSize(4,1);
		}
		else
		{
			sizes[INPUT][2] = cvSize(5,1);
		}
	}
}

int CV_InitUndistortRectifyMapTest::prepare_test_case(int test_case_idx)
{
	CvRNG* rng = ts->get_rng();
	int code = CvArrTest::prepare_test_case( test_case_idx );

	if (code <= 0)
		return code;

	img_size.width = cvTsRandInt(rng) % MAX_X + 1;
	img_size.height = cvTsRandInt(rng) % MAX_Y + 1;

	if (useCPlus)
	{
		mat_type = (cvTsRandInt(rng) % 2) == 0 ? CV_32FC1 : CV_16SC2;
		if ((cvTsRandInt(rng) % 4) == 0)
			mat_type = -1;
		if ((cvTsRandInt(rng) % 4) == 0)
			mat_type = CV_32FC2;
		_mapx = 0;
		_mapy = 0;
	}
	else
	{
		int typex = (cvTsRandInt(rng) % 2) == 0 ? CV_32FC1 : CV_16SC2;
		//typex = CV_32FC1; ///!!!!!!!!!!!!!!!!
		int typey = (typex == CV_32FC1) ? CV_32FC1 : CV_16UC1;

		_mapx = cvCreateMat(img_size.height,img_size.width,typex);
		_mapy = cvCreateMat(img_size.height,img_size.width,typey);


	}

	int dist_size = test_mat[INPUT][2].cols > test_mat[INPUT][2].rows ? test_mat[INPUT][2].cols : test_mat[INPUT][2].rows;
	double cam[9] = {0,0,0,0,0,0,0,0,1};
	double* dist = new double[dist_size ];
	double* new_cam = new double[test_mat[INPUT][4].cols * test_mat[INPUT][4].rows];
	double* points = new double[N_POINTS*2];

	CvMat _camera = cvMat(3,3,CV_64F,cam);
	CvMat _distort = cvMat(test_mat[INPUT][2].rows,test_mat[INPUT][2].cols,CV_64F,dist);
	CvMat _new_cam = cvMat(test_mat[INPUT][4].rows,test_mat[INPUT][4].cols,CV_64F,new_cam);
	CvMat _points= cvMat(test_mat[INPUT][0].rows,test_mat[INPUT][0].cols,CV_64FC2, points);

	for (int i=0;i<test_mat[INPUT][4].cols * test_mat[INPUT][4].rows;i++)
	{
		new_cam[i] = 0;
	}

	//Generating points
	for (int i=0;i<N_POINTS;i++)
	{
		points[2*i] = cvTsRandReal(rng)*img_size.width;
		points[2*i+1] = cvTsRandReal(rng)*img_size.height;
	}



	//Generating camera matrix
	double sz = MAX(img_size.width,img_size.height);
	double aspect_ratio = cvTsRandReal(rng)*0.6 + 0.7;
	cam[2] = (img_size.width - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
	cam[5] = (img_size.height - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
	cam[0] = sz/(0.9 - cvTsRandReal(rng)*0.6);
	cam[4] = aspect_ratio*cam[0];

	//Generating distortion coeffs
	dist[0] = cvTsRandReal(rng)*0.06 - 0.03;
	dist[1] = cvTsRandReal(rng)*0.06 - 0.03;
	if( dist[0]*dist[1] > 0 )
		dist[1] = -dist[1];
	if( cvTsRandInt(rng)%4 != 0 )
	{
		dist[2] = cvTsRandReal(rng)*0.004 - 0.002;
		dist[3] = cvTsRandReal(rng)*0.004 - 0.002;
		if (dist_size > 4)
			dist[4] = cvTsRandReal(rng)*0.004 - 0.002;
	}
	else
	{
		dist[2] = dist[3] = 0;
		if (dist_size > 4)
			dist[4] = 0;
	}

	//Generating new camera matrix

	new_cam[8] = 1;

	//new_cam[0] = cam[0];
	//new_cam[4] = cam[4];
	//new_cam[2] = cam[2];
	//new_cam[5] = cam[5];

	new_cam[0] = cam[0] + (cvTsRandReal(rng) - (double)0.5)*0.2*cam[0]; //10%
	new_cam[4] = cam[4] + (cvTsRandReal(rng) - (double)0.5)*0.2*cam[4]; //10%
	new_cam[2] = cam[2] + (cvTsRandReal(rng) - (double)0.5)*0.3*img_size.width; //15%
	new_cam[5] = cam[5] + (cvTsRandReal(rng) - (double)0.5)*0.3*img_size.height; //15%


	//Generating R matrix
	CvMat* _rot = cvCreateMat(3,3,CV_64F);
	CvMat* rotation = cvCreateMat(1,3,CV_64F);
	rotation->data.db[0] = CV_PI/8*(cvTsRandReal(rng) - (double)0.5); // phi
	rotation->data.db[1] = CV_PI/8*(cvTsRandReal(rng) - (double)0.5); // ksi
	rotation->data.db[2] = CV_PI/3*(cvTsRandReal(rng) - (double)0.5); //khi
	cvRodrigues2(rotation,_rot);
	cvReleaseMat(&rotation);

	//cvSetIdentity(_rot);
	//copying data
	CvMat* dst = &test_mat[INPUT][0];
	cvTsConvert( &_points, dst);
	dst = &test_mat[INPUT][1];
	cvTsConvert( &_camera, dst);
	dst = &test_mat[INPUT][2];
	cvTsConvert( &_distort, dst);
	dst = &test_mat[INPUT][3];
	cvTsConvert( _rot, dst);
	dst = &test_mat[INPUT][4];
	cvTsConvert( &_new_cam, dst);

	zero_distortion = (cvRandInt(rng)%2) == 0 ? false : true;
	zero_new_cam = (cvRandInt(rng)%2) == 0 ? false : true;
	zero_R = (cvRandInt(rng)%2) == 0 ? false : true;

	cvReleaseMat(&_rot);

	if (useCPlus)
	{
		camera_mat = &test_mat[INPUT][1];
		distortion_coeffs = &test_mat[INPUT][2];
		R = &test_mat[INPUT][3];
		new_camera_mat = &test_mat[INPUT][4];
	}
	delete[] dist;
	delete[] new_cam;
	delete[] points;

	return code;
}

void CV_InitUndistortRectifyMapTest::prepare_to_validation(int/* test_case_idx*/)
{
	int dist_size = test_mat[INPUT][2].cols > test_mat[INPUT][2].rows ? test_mat[INPUT][2].cols : test_mat[INPUT][2].rows;
	double cam[9] = {0,0,0,0,0,0,0,0,1};
	double rot[9] = {1,0,0,0,1,0,0,0,1};
	double* dist = new double[dist_size ];
	double* new_cam = new double[test_mat[INPUT][4].cols * test_mat[INPUT][4].rows];
	double* points = new double[N_POINTS*2];
	double* r_points = new double[N_POINTS*2];
	//Run reference calculations
	CvMat ref_points= cvMat(test_mat[INPUT][0].rows,test_mat[INPUT][0].cols,CV_64FC2,r_points);
	CvMat _camera = cvMat(3,3,CV_64F,cam);
	CvMat _rot = cvMat(3,3,CV_64F,rot);
	CvMat _distort = cvMat(test_mat[INPUT][2].rows,test_mat[INPUT][2].cols,CV_64F,dist);
	CvMat _new_cam = cvMat(test_mat[INPUT][4].rows,test_mat[INPUT][4].cols,CV_64F,new_cam);
	CvMat _points= cvMat(test_mat[INPUT][0].rows,test_mat[INPUT][0].cols,CV_64FC2,points);

	cvTsConvert(&test_mat[INPUT][1],&_camera);
	cvTsConvert(&test_mat[INPUT][2],&_distort);
	cvTsConvert(&test_mat[INPUT][3],&_rot);
	cvTsConvert(&test_mat[INPUT][4],&_new_cam);

	//Applying precalculated undistort rectify map
	if (!useCPlus)
	{
		mapx = cv::Mat(_mapx);
		mapy = cv::Mat(_mapy);
	}
	cv::Mat map1,map2;
	cv::convertMaps(mapx,mapy,map1,map2,CV_32FC1);
	CvMat _map1 = map1;
	CvMat _map2 = map2;
	for (int i=0;i<N_POINTS;i++)
	{
		double u = test_mat[INPUT][0].data.db[2*i];
		double v = test_mat[INPUT][0].data.db[2*i+1];
		_points.data.db[2*i] = (double)_map1.data.fl[(int)v*_map1.cols+(int)u];
		_points.data.db[2*i+1] = (double)_map2.data.fl[(int)v*_map2.cols+(int)u];
	}

	//---

	cvUndistortPoints(&_points,&ref_points,&_camera,
		zero_distortion ? 0 : &_distort, zero_R ? 0 : &_rot, zero_new_cam ? &_camera : &_new_cam);
	//cvTsDistortPoints(&_points,&ref_points,&_camera,&_distort,&_rot,&_new_cam);
	CvMat* dst = &test_mat[REF_OUTPUT][0];
	cvTsConvert(&ref_points,dst);

	cvCopy(&test_mat[INPUT][0],&test_mat[OUTPUT][0]);

	delete[] dist;
	delete[] new_cam;
	delete[] points;
	delete[] r_points;
	if (_mapx)
	{
		cvReleaseMat(&_mapx);
		_mapx = 0;
	}
	if (_mapy)
	{
		cvReleaseMat(&_mapy);
		_mapy = 0;
	}
}

void CV_InitUndistortRectifyMapTest::run_func()
{
	if (useCPlus)
	{
		cv::Mat input2,input3,input4;
		input2 = zero_distortion ? cv::Mat() : cv::Mat(&test_mat[INPUT][2]);
		input3 = zero_R ? cv::Mat() : cv::Mat(&test_mat[INPUT][3]);
		input4 = zero_new_cam ? cv::Mat() : cv::Mat(&test_mat[INPUT][4]);
		cv::initUndistortRectifyMap(camera_mat,input2,input3,input4,img_size,mat_type,mapx,mapy);
	}
	else
	{
		CvMat* input2;
		CvMat* input3;
		CvMat* input4;
		input2 = zero_distortion ? 0 : &test_mat[INPUT][2];
		input3 = zero_R ? 0 : &test_mat[INPUT][3];
		input4 = zero_new_cam ? 0 : &test_mat[INPUT][4];
		cvInitUndistortRectifyMap(&test_mat[INPUT][1],input2,input3,input4,_mapx,_mapy);
	}
}

double CV_InitUndistortRectifyMapTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
	return 8;
}

CV_InitUndistortRectifyMapTest init_undistort_rectify_map_test;
