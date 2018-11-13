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

#include "test_precomp.hpp"
#include "opencv2/imgproc/imgproc_c.h"

namespace opencv_test { namespace {

class CV_DefaultNewCameraMatrixTest : public cvtest::ArrayTest
{
public:
    CV_DefaultNewCameraMatrixTest();
protected:
    int prepare_test_case (int test_case_idx);
    void prepare_to_validation( int test_case_idx );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void run_func();

private:
    cv::Size img_size;
    cv::Mat camera_mat;
    cv::Mat new_camera_mat;

    int matrix_type;

    bool center_principal_point;

    static const int MAX_X = 2048;
    static const int MAX_Y = 2048;
    //static const int MAX_VAL = 10000;
};

CV_DefaultNewCameraMatrixTest::CV_DefaultNewCameraMatrixTest()
{
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);

    matrix_type = 0;
    center_principal_point = false;
}

void CV_DefaultNewCameraMatrixTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    cvtest::ArrayTest::get_test_array_types_and_sizes(test_case_idx,sizes,types);
    RNG& rng = ts->get_rng();
    matrix_type = types[INPUT][0] = types[OUTPUT][0]= types[REF_OUTPUT][0] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;
    sizes[INPUT][0] = sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize(3,3);
}

int CV_DefaultNewCameraMatrixTest::prepare_test_case(int test_case_idx)
{
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );

    if (code <= 0)
        return code;

    RNG& rng = ts->get_rng();

    img_size.width = cvtest::randInt(rng) % MAX_X + 1;
    img_size.height = cvtest::randInt(rng) % MAX_Y + 1;

    center_principal_point = ((cvtest::randInt(rng) % 2)!=0);

    // Generating camera_mat matrix
    double sz = MAX(img_size.width, img_size.height);
    double aspect_ratio = cvtest::randReal(rng)*0.6 + 0.7;
    double a[9] = {0,0,0,0,0,0,0,0,1};
    Mat _a(3,3,CV_64F,a);
    a[2] = (img_size.width - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    a[5] = (img_size.height - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    a[0] = sz/(0.9 - cvtest::randReal(rng)*0.6);
    a[4] = aspect_ratio*a[0];

    Mat& _a0 = test_mat[INPUT][0];
    cvtest::convert(_a, _a0, _a0.type());
    camera_mat = _a0;

    return code;

}

void CV_DefaultNewCameraMatrixTest::run_func()
{
    new_camera_mat = cv::getDefaultNewCameraMatrix(camera_mat,img_size,center_principal_point);
}

void CV_DefaultNewCameraMatrixTest::prepare_to_validation( int /*test_case_idx*/ )
{
    const Mat& src = test_mat[INPUT][0];
    Mat& dst = test_mat[REF_OUTPUT][0];
    Mat& test_output = test_mat[OUTPUT][0];
    Mat& output = new_camera_mat;
    cvtest::convert( output, test_output, test_output.type() );
    if (!center_principal_point)
    {
        cvtest::copy(src, dst);
    }
    else
    {
        double a[9] = {0,0,0,0,0,0,0,0,1};
        Mat _a(3,3,CV_64F,a);
        if (matrix_type == CV_64F)
        {
            a[0] = src.at<double>(0,0);
            a[4] = src.at<double>(1,1);
        }
        else
        {
            a[0] = src.at<float>(0,0);
            a[4] = src.at<float>(1,1);
        }
        a[2] = (img_size.width - 1)*0.5;
        a[5] = (img_size.height - 1)*0.5;
        cvtest::convert( _a, dst, dst.type() );
    }
}

//---------

class CV_UndistortPointsTest : public cvtest::ArrayTest
{
public:
    CV_UndistortPointsTest();
protected:
    int prepare_test_case (int test_case_idx);
    void prepare_to_validation( int test_case_idx );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void distortPoints(const CvMat* _src, CvMat* _dst, const CvMat* _cameraMatrix,
                       const CvMat* _distCoeffs, const CvMat* matR, const CvMat* matP);

private:
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

CV_UndistortPointsTest::CV_UndistortPointsTest()
{
    test_array[INPUT].push_back(NULL); // points matrix
    test_array[INPUT].push_back(NULL); // camera matrix
    test_array[INPUT].push_back(NULL); // distortion coeffs
    test_array[INPUT].push_back(NULL); // R matrix
    test_array[INPUT].push_back(NULL); // P matrix
    test_array[OUTPUT].push_back(NULL); // distorted dst points
    test_array[TEMP].push_back(NULL); // dst points
    test_array[REF_OUTPUT].push_back(NULL);

    useDstMat = false;
    zero_new_cam = zero_distortion = zero_R = false;
}

void CV_UndistortPointsTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    cvtest::ArrayTest::get_test_array_types_and_sizes(test_case_idx,sizes,types);
    RNG& rng = ts->get_rng();
    //rng.next();
    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = types[TEMP][0]= CV_32FC2;
    types[INPUT][1] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;
    types[INPUT][2] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;
    types[INPUT][3] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;
    types[INPUT][4] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;

    sizes[INPUT][0] = sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = sizes[TEMP][0]= cvtest::randInt(rng)%2 ? cvSize(1,N_POINTS) : cvSize(N_POINTS,1);
    sizes[INPUT][1] = sizes[INPUT][3] = cvSize(3,3);
    sizes[INPUT][4] = cvtest::randInt(rng)%2 ? cvSize(3,3) : cvSize(4,3);

    if (cvtest::randInt(rng)%2)
    {
        if (cvtest::randInt(rng)%2)
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
        if (cvtest::randInt(rng)%2)
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
    RNG& rng = ts->get_rng();
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );

    if (code <= 0)
        return code;

    useDstMat = (cvtest::randInt(rng) % 2) == 0;

    img_size.width = cvtest::randInt(rng) % MAX_X + 1;
    img_size.height = cvtest::randInt(rng) % MAX_Y + 1;
    int dist_size = test_mat[INPUT][2].cols > test_mat[INPUT][2].rows ? test_mat[INPUT][2].cols : test_mat[INPUT][2].rows;
    double cam[9] = {0,0,0,0,0,0,0,0,1};
    vector<double> dist(dist_size);
    vector<double> proj(test_mat[INPUT][4].cols * test_mat[INPUT][4].rows);
    vector<Point2d> points(N_POINTS);

    Mat _camera(3,3,CV_64F,cam);
    Mat _distort(test_mat[INPUT][2].rows,test_mat[INPUT][2].cols,CV_64F,&dist[0]);
    Mat _proj(test_mat[INPUT][4].size(), CV_64F, &proj[0]);
    Mat _points(test_mat[INPUT][0].size(), CV_64FC2, &points[0]);

    _proj = Scalar::all(0);

    //Generating points
    for( int i = 0; i < N_POINTS; i++ )
    {
        points[i].x = cvtest::randReal(rng)*img_size.width;
        points[i].y = cvtest::randReal(rng)*img_size.height;
    }

    //Generating camera matrix
    double sz = MAX(img_size.width,img_size.height);
    double aspect_ratio = cvtest::randReal(rng)*0.6 + 0.7;
    cam[2] = (img_size.width - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    cam[5] = (img_size.height - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    cam[0] = sz/(0.9 - cvtest::randReal(rng)*0.6);
    cam[4] = aspect_ratio*cam[0];

    //Generating distortion coeffs
    dist[0] = cvtest::randReal(rng)*0.06 - 0.03;
    dist[1] = cvtest::randReal(rng)*0.06 - 0.03;
    if( dist[0]*dist[1] > 0 )
        dist[1] = -dist[1];
    if( cvtest::randInt(rng)%4 != 0 )
    {
        dist[2] = cvtest::randReal(rng)*0.004 - 0.002;
        dist[3] = cvtest::randReal(rng)*0.004 - 0.002;
        if (dist_size > 4)
            dist[4] = cvtest::randReal(rng)*0.004 - 0.002;
    }
    else
    {
        dist[2] = dist[3] = 0;
        if (dist_size > 4)
            dist[4] = 0;
    }

    //Generating P matrix (projection)
    if( test_mat[INPUT][4].cols != 4 )
    {
        proj[8] = 1;
        if (cvtest::randInt(rng)%2 == 0) // use identity new camera matrix
        {
            proj[0] = 1;
            proj[4] = 1;
        }
        else
        {
            proj[0] = cam[0] + (cvtest::randReal(rng) - (double)0.5)*0.2*cam[0]; //10%
            proj[4] = cam[4] + (cvtest::randReal(rng) - (double)0.5)*0.2*cam[4]; //10%
            proj[2] = cam[2] + (cvtest::randReal(rng) - (double)0.5)*0.3*img_size.width; //15%
            proj[5] = cam[5] + (cvtest::randReal(rng) - (double)0.5)*0.3*img_size.height; //15%
        }
    }
    else
    {
        proj[10] = 1;
        proj[0] = cam[0] + (cvtest::randReal(rng) - (double)0.5)*0.2*cam[0]; //10%
        proj[5] = cam[4] + (cvtest::randReal(rng) - (double)0.5)*0.2*cam[4]; //10%
        proj[2] = cam[2] + (cvtest::randReal(rng) - (double)0.5)*0.3*img_size.width; //15%
        proj[6] = cam[5] + (cvtest::randReal(rng) - (double)0.5)*0.3*img_size.height; //15%

        proj[3] = (img_size.height + img_size.width - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
        proj[7] = (img_size.height + img_size.width - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
        proj[11] = (img_size.height + img_size.width - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    }

    //Generating R matrix
    Mat _rot(3,3,CV_64F);
    Mat rotation(1,3,CV_64F);
    rotation.at<double>(0) = CV_PI*(cvtest::randReal(rng) - (double)0.5); // phi
    rotation.at<double>(1) = CV_PI*(cvtest::randReal(rng) - (double)0.5); // ksi
    rotation.at<double>(2) = CV_PI*(cvtest::randReal(rng) - (double)0.5); //khi
    cvtest::Rodrigues(rotation, _rot);

    //copying data
    //src_points = &_points;
    _points.convertTo(test_mat[INPUT][0], test_mat[INPUT][0].type());
    _camera.convertTo(test_mat[INPUT][1], test_mat[INPUT][1].type());
    _distort.convertTo(test_mat[INPUT][2], test_mat[INPUT][2].type());
    _rot.convertTo(test_mat[INPUT][3], test_mat[INPUT][3].type());
    _proj.convertTo(test_mat[INPUT][4], test_mat[INPUT][4].type());

    zero_distortion = (cvtest::randInt(rng)%2) == 0 ? false : true;
    zero_new_cam = (cvtest::randInt(rng)%2) == 0 ? false : true;
    zero_R = (cvtest::randInt(rng)%2) == 0 ? false : true;

    _points.convertTo(src_points, CV_32F);

    camera_mat = test_mat[INPUT][1];
    distortion_coeffs = test_mat[INPUT][2];
    R = test_mat[INPUT][3];
    P = test_mat[INPUT][4];

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

    Mat __camera = cvarrToMat(&_camera);
    Mat __distort = cvarrToMat(&_distort);
    Mat __rot = cvarrToMat(&_rot);
    Mat __proj = cvarrToMat(&_proj);
    Mat __points = cvarrToMat(&_points);
    Mat _ref_points = cvarrToMat(&ref_points);

    cvtest::convert(test_mat[INPUT][1], __camera, __camera.type());
    cvtest::convert(test_mat[INPUT][2], __distort, __distort.type());
    cvtest::convert(test_mat[INPUT][3], __rot, __rot.type());
    cvtest::convert(test_mat[INPUT][4], __proj, __proj.type());

    if (useDstMat)
    {
        CvMat temp = cvMat(dst_points_mat);
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

    CvMat* input2 = zero_distortion ? 0 : &_distort;
    CvMat* input3 = zero_R ? 0 : &_rot;
    CvMat* input4 = zero_new_cam ? 0 : &_proj;
    distortPoints(&_points,&ref_points,&_camera,input2,input3,input4);

    Mat& dst = test_mat[REF_OUTPUT][0];
    cvtest::convert(_ref_points, dst, dst.type());

    cvtest::copy(test_mat[INPUT][0], test_mat[OUTPUT][0]);

    delete[] dist;
    delete[] proj;
    delete[] points;
    delete[] r_points;
}

void CV_UndistortPointsTest::run_func()
{
    cv::Mat input2,input3,input4;
    input2 = zero_distortion ? cv::Mat() : cv::Mat(test_mat[INPUT][2]);
    input3 = zero_R ? cv::Mat() : cv::Mat(test_mat[INPUT][3]);
    input4 = zero_new_cam ? cv::Mat() : cv::Mat(test_mat[INPUT][4]);

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

void CV_UndistortPointsTest::distortPoints(const CvMat* _src, CvMat* _dst, const CvMat* _cameraMatrix,
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
        cvtest::convert(cvarrToMat(matP), cvarrToMat(__P), -1);
    }
    else
    {
        cvZero(__P);
        __P->data.db[0] = 1;
        __P->data.db[4] = 1;
        __P->data.db[8] = 1;
    }
    CvMat* __R = cvCreateMat(3,3,CV_64F);
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

//------------------------------------------------------

class CV_InitUndistortRectifyMapTest : public cvtest::ArrayTest
{
public:
    CV_InitUndistortRectifyMapTest();
protected:
    int prepare_test_case (int test_case_idx);
    void prepare_to_validation( int test_case_idx );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();

private:
    static const int MAX_X = 1024;
    static const int MAX_Y = 1024;
    bool zero_new_cam;
    bool zero_distortion;
    bool zero_R;

    cv::Size img_size;
    int map_type;
};

CV_InitUndistortRectifyMapTest::CV_InitUndistortRectifyMapTest()
{
    test_array[INPUT].push_back(NULL); // camera matrix
    test_array[INPUT].push_back(NULL); // distortion coeffs
    test_array[INPUT].push_back(NULL); // R matrix
    test_array[INPUT].push_back(NULL); // new camera matrix
    test_array[OUTPUT].push_back(NULL); // distorted mapx
    test_array[OUTPUT].push_back(NULL); // distorted mapy
    test_array[REF_OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);

    zero_distortion = zero_new_cam = zero_R = false;
    map_type = 0;
}

void CV_InitUndistortRectifyMapTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    cvtest::ArrayTest::get_test_array_types_and_sizes(test_case_idx,sizes,types);
    RNG& rng = ts->get_rng();
    //rng.next();

    map_type = CV_32F;
    types[OUTPUT][0] = types[OUTPUT][1] = types[REF_OUTPUT][0] = types[REF_OUTPUT][1] = map_type;

    img_size.width = cvtest::randInt(rng) % MAX_X + 1;
    img_size.height = cvtest::randInt(rng) % MAX_Y + 1;

    types[INPUT][0] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;
    types[INPUT][1] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;
    types[INPUT][2] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;
    types[INPUT][3] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;

    sizes[OUTPUT][0] = sizes[OUTPUT][1] = sizes[REF_OUTPUT][0] = sizes[REF_OUTPUT][1] = img_size;
    sizes[INPUT][0] = sizes[INPUT][2] = sizes[INPUT][3] = cvSize(3,3);

    Size dsize;

    if (cvtest::randInt(rng)%2)
    {
        if (cvtest::randInt(rng)%2)
        {
            dsize = Size(1,4);
        }
        else
        {
            dsize = Size(1,5);
        }
    }
    else
    {
        if (cvtest::randInt(rng)%2)
        {
            dsize = Size(4,1);
        }
        else
        {
            dsize = Size(5,1);
        }
    }
    sizes[INPUT][1] = dsize;
}


int CV_InitUndistortRectifyMapTest::prepare_test_case(int test_case_idx)
{
    RNG& rng = ts->get_rng();
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );

    if (code <= 0)
        return code;

    int dist_size = test_mat[INPUT][1].cols > test_mat[INPUT][1].rows ? test_mat[INPUT][1].cols : test_mat[INPUT][1].rows;
    double cam[9] = {0,0,0,0,0,0,0,0,1};
    vector<double> dist(dist_size);
    vector<double> new_cam(test_mat[INPUT][3].cols * test_mat[INPUT][3].rows);

    Mat _camera(3,3,CV_64F,cam);
    Mat _distort(test_mat[INPUT][1].size(),CV_64F,&dist[0]);
    Mat _new_cam(test_mat[INPUT][3].size(),CV_64F,&new_cam[0]);

    //Generating camera matrix
    double sz = MAX(img_size.width,img_size.height);
    double aspect_ratio = cvtest::randReal(rng)*0.6 + 0.7;
    cam[2] = (img_size.width - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    cam[5] = (img_size.height - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    cam[0] = sz/(0.9 - cvtest::randReal(rng)*0.6);
    cam[4] = aspect_ratio*cam[0];

    //Generating distortion coeffs
    dist[0] = cvtest::randReal(rng)*0.06 - 0.03;
    dist[1] = cvtest::randReal(rng)*0.06 - 0.03;
    if( dist[0]*dist[1] > 0 )
        dist[1] = -dist[1];
    if( cvtest::randInt(rng)%4 != 0 )
    {
        dist[2] = cvtest::randReal(rng)*0.004 - 0.002;
        dist[3] = cvtest::randReal(rng)*0.004 - 0.002;
        if (dist_size > 4)
            dist[4] = cvtest::randReal(rng)*0.004 - 0.002;
    }
    else
    {
        dist[2] = dist[3] = 0;
        if (dist_size > 4)
            dist[4] = 0;
    }

    //Generating new camera matrix
    _new_cam = Scalar::all(0);
    new_cam[8] = 1;

    //new_cam[0] = cam[0];
    //new_cam[4] = cam[4];
    //new_cam[2] = cam[2];
    //new_cam[5] = cam[5];

    new_cam[0] = cam[0] + (cvtest::randReal(rng) - (double)0.5)*0.2*cam[0]; //10%
    new_cam[4] = cam[4] + (cvtest::randReal(rng) - (double)0.5)*0.2*cam[4]; //10%
    new_cam[2] = cam[2] + (cvtest::randReal(rng) - (double)0.5)*0.3*img_size.width; //15%
    new_cam[5] = cam[5] + (cvtest::randReal(rng) - (double)0.5)*0.3*img_size.height; //15%

    //Generating R matrix
    Mat _rot(3,3,CV_64F);
    Mat rotation(1,3,CV_64F);
    rotation.at<double>(0) = CV_PI/8*(cvtest::randReal(rng) - (double)0.5); // phi
    rotation.at<double>(1) = CV_PI/8*(cvtest::randReal(rng) - (double)0.5); // ksi
    rotation.at<double>(2) = CV_PI/3*(cvtest::randReal(rng) - (double)0.5); //khi
    cvtest::Rodrigues(rotation, _rot);

    //cvSetIdentity(_rot);
    //copying data
    cvtest::convert( _camera, test_mat[INPUT][0], test_mat[INPUT][0].type());
    cvtest::convert( _distort, test_mat[INPUT][1], test_mat[INPUT][1].type());
    cvtest::convert( _rot, test_mat[INPUT][2], test_mat[INPUT][2].type());
    cvtest::convert( _new_cam, test_mat[INPUT][3], test_mat[INPUT][3].type());

    zero_distortion = (cvtest::randInt(rng)%2) == 0 ? false : true;
    zero_new_cam = (cvtest::randInt(rng)%2) == 0 ? false : true;
    zero_R = (cvtest::randInt(rng)%2) == 0 ? false : true;

    return code;
}

void CV_InitUndistortRectifyMapTest::prepare_to_validation(int/* test_case_idx*/)
{
    cvtest::initUndistortMap(test_mat[INPUT][0],
                             zero_distortion ? cv::Mat() : test_mat[INPUT][1],
                             zero_R ? cv::Mat() : test_mat[INPUT][2],
                             zero_new_cam ? test_mat[INPUT][0] : test_mat[INPUT][3],
                             img_size, test_mat[REF_OUTPUT][0], test_mat[REF_OUTPUT][1],
                             test_mat[REF_OUTPUT][0].type());
}

void CV_InitUndistortRectifyMapTest::run_func()
{
    cv::Mat camera_mat = test_mat[INPUT][0];
    cv::Mat dist = zero_distortion ? cv::Mat() : test_mat[INPUT][1];
    cv::Mat R = zero_R ? cv::Mat() : test_mat[INPUT][2];
    cv::Mat new_cam = zero_new_cam ? cv::Mat() : test_mat[INPUT][3];
    cv::Mat& mapx = test_mat[OUTPUT][0], &mapy = test_mat[OUTPUT][1];
    cv::initUndistortRectifyMap(camera_mat,dist,R,new_cam,img_size,map_type,mapx,mapy);
}

double CV_InitUndistortRectifyMapTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 8;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Calib3d_DefaultNewCameraMatrix, accuracy) { CV_DefaultNewCameraMatrixTest test; test.safe_run(); }
TEST(Calib3d_UndistortPoints, accuracy) { CV_UndistortPointsTest test; test.safe_run(); }
TEST(Calib3d_InitUndistortRectifyMap, accuracy) { CV_InitUndistortRectifyMapTest test; test.safe_run(); }

////////////////////////////// undistort /////////////////////////////////

static void test_remap( const Mat& src, Mat& dst, const Mat& mapx, const Mat& mapy,
                        Mat* mask=0, int interpolation=CV_INTER_LINEAR )
{
    int x, y, k;
    int drows = dst.rows, dcols = dst.cols;
    int srows = src.rows, scols = src.cols;
    const uchar* sptr0 = src.ptr();
    int depth = src.depth(), cn = src.channels();
    int elem_size = (int)src.elemSize();
    int step = (int)(src.step / CV_ELEM_SIZE(depth));
    int delta;

    if( interpolation != CV_INTER_CUBIC )
    {
        delta = 0;
        scols -= 1; srows -= 1;
    }
    else
    {
        delta = 1;
        scols = MAX(scols - 3, 0);
        srows = MAX(srows - 3, 0);
    }

    int scols1 = MAX(scols - 2, 0);
    int srows1 = MAX(srows - 2, 0);

    if( mask )
        *mask = Scalar::all(0);

    for( y = 0; y < drows; y++ )
    {
        uchar* dptr = dst.ptr(y);
        const float* mx = mapx.ptr<float>(y);
        const float* my = mapy.ptr<float>(y);
        uchar* m = mask ? mask->ptr(y) : 0;

        for( x = 0; x < dcols; x++, dptr += elem_size )
        {
            float xs = mx[x];
            float ys = my[x];
            int ixs = cvFloor(xs);
            int iys = cvFloor(ys);

            if( (unsigned)(ixs - delta - 1) >= (unsigned)scols1 ||
                (unsigned)(iys - delta - 1) >= (unsigned)srows1 )
            {
                if( m )
                    m[x] = 1;
                if( (unsigned)(ixs - delta) >= (unsigned)scols ||
                    (unsigned)(iys - delta) >= (unsigned)srows )
                    continue;
            }

            xs -= ixs;
            ys -= iys;

            switch( depth )
            {
            case CV_8U:
                {
                const uchar* sptr = sptr0 + iys*step + ixs*cn;
                for( k = 0; k < cn; k++ )
                {
                    float v00 = sptr[k];
                    float v01 = sptr[cn + k];
                    float v10 = sptr[step + k];
                    float v11 = sptr[step + cn + k];

                    v00 = v00 + xs*(v01 - v00);
                    v10 = v10 + xs*(v11 - v10);
                    v00 = v00 + ys*(v10 - v00);
                    dptr[k] = (uchar)cvRound(v00);
                }
                }
                break;
            case CV_16U:
                {
                const ushort* sptr = (const ushort*)sptr0 + iys*step + ixs*cn;
                for( k = 0; k < cn; k++ )
                {
                    float v00 = sptr[k];
                    float v01 = sptr[cn + k];
                    float v10 = sptr[step + k];
                    float v11 = sptr[step + cn + k];

                    v00 = v00 + xs*(v01 - v00);
                    v10 = v10 + xs*(v11 - v10);
                    v00 = v00 + ys*(v10 - v00);
                    ((ushort*)dptr)[k] = (ushort)cvRound(v00);
                }
                }
                break;
            case CV_32F:
                {
                const float* sptr = (const float*)sptr0 + iys*step + ixs*cn;
                for( k = 0; k < cn; k++ )
                {
                    float v00 = sptr[k];
                    float v01 = sptr[cn + k];
                    float v10 = sptr[step + k];
                    float v11 = sptr[step + cn + k];

                    v00 = v00 + xs*(v01 - v00);
                    v10 = v10 + xs*(v11 - v10);
                    v00 = v00 + ys*(v10 - v00);
                    ((float*)dptr)[k] = (float)v00;
                }
                }
                break;
            default:
                assert(0);
            }
        }
    }
}

class CV_ImgWarpBaseTest : public cvtest::ArrayTest
{
public:
    CV_ImgWarpBaseTest( bool warp_matrix );

protected:
    int read_params( const cv::FileStorage& fs );
    int prepare_test_case( int test_case_idx );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high );
    void fill_array( int test_case_idx, int i, int j, Mat& arr );

    int interpolation;
    int max_interpolation;
    double spatial_scale_zoom, spatial_scale_decimate;
};


CV_ImgWarpBaseTest::CV_ImgWarpBaseTest( bool warp_matrix )
{
    test_array[INPUT].push_back(NULL);
    if( warp_matrix )
        test_array[INPUT].push_back(NULL);
    test_array[INPUT_OUTPUT].push_back(NULL);
    test_array[REF_INPUT_OUTPUT].push_back(NULL);
    max_interpolation = 5;
    interpolation = 0;
    element_wise_relative_error = false;
    spatial_scale_zoom = 0.01;
    spatial_scale_decimate = 0.005;
}


int CV_ImgWarpBaseTest::read_params( const cv::FileStorage& fs )
{
    int code = cvtest::ArrayTest::read_params( fs );
    return code;
}


void CV_ImgWarpBaseTest::get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high )
{
    cvtest::ArrayTest::get_minmax_bounds( i, j, type, low, high );
    if( CV_MAT_DEPTH(type) == CV_32F )
    {
        low = Scalar::all(-10.);
        high = Scalar::all(10);
    }
}


void CV_ImgWarpBaseTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int depth = cvtest::randInt(rng) % 3;
    int cn = cvtest::randInt(rng) % 3 + 1;
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    depth = depth == 0 ? CV_8U : depth == 1 ? CV_16U : CV_32F;
    cn += cn == 2;

    types[INPUT][0] = types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] = CV_MAKETYPE(depth, cn);
    if( test_array[INPUT].size() > 1 )
        types[INPUT][1] = cvtest::randInt(rng) & 1 ? CV_32FC1 : CV_64FC1;

    interpolation = cvtest::randInt(rng) % max_interpolation;
}


void CV_ImgWarpBaseTest::fill_array( int test_case_idx, int i, int j, Mat& arr )
{
    if( i != INPUT || j != 0 )
        cvtest::ArrayTest::fill_array( test_case_idx, i, j, arr );
}

int CV_ImgWarpBaseTest::prepare_test_case( int test_case_idx )
{
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );
    Mat& img = test_mat[INPUT][0];
    int i, j, cols = img.cols;
    int type = img.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    double scale = depth == CV_16U ? 1000. : 255.*0.5;
    double space_scale = spatial_scale_decimate;
    vector<float> buffer(img.cols*cn);

    if( code <= 0 )
        return code;

    if( test_mat[INPUT_OUTPUT][0].cols >= img.cols &&
        test_mat[INPUT_OUTPUT][0].rows >= img.rows )
        space_scale = spatial_scale_zoom;

    for( i = 0; i < img.rows; i++ )
    {
        uchar* ptr = img.ptr(i);
        switch( cn )
        {
        case 1:
            for( j = 0; j < cols; j++ )
                buffer[j] = (float)((sin((i+1)*space_scale)*sin((j+1)*space_scale)+1.)*scale);
            break;
        case 2:
            for( j = 0; j < cols; j++ )
            {
                buffer[j*2] = (float)((sin((i+1)*space_scale)+1.)*scale);
                buffer[j*2+1] = (float)((sin((i+j)*space_scale)+1.)*scale);
            }
            break;
        case 3:
            for( j = 0; j < cols; j++ )
            {
                buffer[j*3] = (float)((sin((i+1)*space_scale)+1.)*scale);
                buffer[j*3+1] = (float)((sin(j*space_scale)+1.)*scale);
                buffer[j*3+2] = (float)((sin((i+j)*space_scale)+1.)*scale);
            }
            break;
        case 4:
            for( j = 0; j < cols; j++ )
            {
                buffer[j*4] = (float)((sin((i+1)*space_scale)+1.)*scale);
                buffer[j*4+1] = (float)((sin(j*space_scale)+1.)*scale);
                buffer[j*4+2] = (float)((sin((i+j)*space_scale)+1.)*scale);
                buffer[j*4+3] = (float)((sin((i-j)*space_scale)+1.)*scale);
            }
            break;
        default:
            assert(0);
        }

        /*switch( depth )
        {
        case CV_8U:
            for( j = 0; j < cols*cn; j++ )
                ptr[j] = (uchar)cvRound(buffer[j]);
            break;
        case CV_16U:
            for( j = 0; j < cols*cn; j++ )
                ((ushort*)ptr)[j] = (ushort)cvRound(buffer[j]);
            break;
        case CV_32F:
            for( j = 0; j < cols*cn; j++ )
                ((float*)ptr)[j] = (float)buffer[j];
            break;
        default:
            assert(0);
        }*/
        cv::Mat src(1, cols*cn, CV_32F, &buffer[0]);
        cv::Mat dst(1, cols*cn, depth, ptr);
        src.convertTo(dst, dst.type());
    }

    return code;
}


class CV_UndistortTest : public CV_ImgWarpBaseTest
{
public:
    CV_UndistortTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );
    void fill_array( int test_case_idx, int i, int j, Mat& arr );

private:
    cv::Mat input0;
    cv::Mat input1;
    cv::Mat input2;
    cv::Mat input_new_cam;
    cv::Mat input_output;

    bool zero_new_cam;
    bool zero_distortion;
};


CV_UndistortTest::CV_UndistortTest() : CV_ImgWarpBaseTest( false )
{
    //spatial_scale_zoom = spatial_scale_decimate;
    test_array[INPUT].push_back(NULL);
    test_array[INPUT].push_back(NULL);
    test_array[INPUT].push_back(NULL);

    spatial_scale_decimate = spatial_scale_zoom;
}


void CV_UndistortTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int type = types[INPUT][0];
    type = CV_MAKETYPE( CV_8U, CV_MAT_CN(type) );
    types[INPUT][0] = types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] = type;
    types[INPUT][1] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;
    types[INPUT][2] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;
    sizes[INPUT][1] = cvSize(3,3);
    sizes[INPUT][2] = cvtest::randInt(rng)%2 ? cvSize(4,1) : cvSize(1,4);
    types[INPUT][3] =  types[INPUT][1];
    sizes[INPUT][3] = sizes[INPUT][1];
    interpolation = CV_INTER_LINEAR;
}


void CV_UndistortTest::fill_array( int test_case_idx, int i, int j, Mat& arr )
{
    if( i != INPUT )
        CV_ImgWarpBaseTest::fill_array( test_case_idx, i, j, arr );
}


void CV_UndistortTest::run_func()
{
    if (zero_distortion)
    {
        cv::undistort(input0,input_output,input1,cv::Mat());
    }
    else
    {
        cv::undistort(input0,input_output,input1,input2);
    }
}


double CV_UndistortTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = test_mat[INPUT][0].depth();
    return depth == CV_8U ? 16 : depth == CV_16U ? 1024 : 5e-2;
}


int CV_UndistortTest::prepare_test_case( int test_case_idx )
{
    RNG& rng = ts->get_rng();
    int code = CV_ImgWarpBaseTest::prepare_test_case( test_case_idx );

    const Mat& src = test_mat[INPUT][0];
    double k[4], a[9] = {0,0,0,0,0,0,0,0,1};
    double new_cam[9] = {0,0,0,0,0,0,0,0,1};
    double sz = MAX(src.rows, src.cols);

    Mat& _new_cam0 = test_mat[INPUT][3];
    Mat _new_cam(test_mat[INPUT][3].rows,test_mat[INPUT][3].cols,CV_64F,new_cam);
    Mat& _a0 = test_mat[INPUT][1];
    Mat _a(3,3,CV_64F,a);
    Mat& _k0 = test_mat[INPUT][2];
    Mat _k(_k0.rows,_k0.cols, CV_MAKETYPE(CV_64F,_k0.channels()),k);

    if( code <= 0 )
        return code;

    double aspect_ratio = cvtest::randReal(rng)*0.6 + 0.7;
    a[2] = (src.cols - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    a[5] = (src.rows - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    a[0] = sz/(0.9 - cvtest::randReal(rng)*0.6);
    a[4] = aspect_ratio*a[0];
    k[0] = cvtest::randReal(rng)*0.06 - 0.03;
    k[1] = cvtest::randReal(rng)*0.06 - 0.03;
    if( k[0]*k[1] > 0 )
        k[1] = -k[1];
    if( cvtest::randInt(rng)%4 != 0 )
    {
        k[2] = cvtest::randReal(rng)*0.004 - 0.002;
        k[3] = cvtest::randReal(rng)*0.004 - 0.002;
    }
    else
        k[2] = k[3] = 0;

    new_cam[0] = a[0] + (cvtest::randReal(rng) - (double)0.5)*0.2*a[0]; //10%
    new_cam[4] = a[4] + (cvtest::randReal(rng) - (double)0.5)*0.2*a[4]; //10%
    new_cam[2] = a[2] + (cvtest::randReal(rng) - (double)0.5)*0.3*test_mat[INPUT][0].rows; //15%
    new_cam[5] = a[5] + (cvtest::randReal(rng) - (double)0.5)*0.3*test_mat[INPUT][0].cols; //15%

    _a.convertTo(_a0, _a0.depth());

    zero_distortion = (cvtest::randInt(rng)%2) == 0 ? false : true;
    _k.convertTo(_k0, _k0.depth());

    zero_new_cam = (cvtest::randInt(rng)%2) == 0 ? false : true;
    _new_cam.convertTo(_new_cam0, _new_cam0.depth());

    //Testing C++ code
    //useCPlus = ((cvtest::randInt(rng) % 2)!=0);
    input0 = test_mat[INPUT][0];
    input1 = test_mat[INPUT][1];
    input2 = test_mat[INPUT][2];
    input_new_cam = test_mat[INPUT][3];

    return code;
}


void CV_UndistortTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat& output = test_mat[INPUT_OUTPUT][0];
    input_output.convertTo(output, output.type());
    Mat& src = test_mat[INPUT][0];
    Mat& dst = test_mat[REF_INPUT_OUTPUT][0];
    Mat& dst0 = test_mat[INPUT_OUTPUT][0];
    Mat mapx, mapy;
    cvtest::initUndistortMap( test_mat[INPUT][1], test_mat[INPUT][2],
                              Mat(), Mat(), dst.size(), mapx, mapy, CV_32F );
    Mat mask( dst.size(), CV_8U );
    test_remap( src, dst, mapx, mapy, &mask, interpolation );
    dst.setTo(Scalar::all(0), mask);
    dst0.setTo(Scalar::all(0), mask);
}


class CV_UndistortMapTest : public cvtest::ArrayTest
{
public:
    CV_UndistortMapTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );
    void fill_array( int test_case_idx, int i, int j, Mat& arr );

private:
    bool dualChannel;
};


CV_UndistortMapTest::CV_UndistortMapTest()
{
    test_array[INPUT].push_back(NULL);
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);

    element_wise_relative_error = false;
}


void CV_UndistortMapTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int depth = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;

    Size sz = sizes[OUTPUT][0];
    types[INPUT][0] = types[INPUT][1] = depth;
    dualChannel = cvtest::randInt(rng)%2 == 0;
    types[OUTPUT][0] = types[OUTPUT][1] =
        types[REF_OUTPUT][0] = types[REF_OUTPUT][1] = dualChannel ? CV_32FC2 : CV_32F;
    sizes[INPUT][0] = cvSize(3,3);
    sizes[INPUT][1] = cvtest::randInt(rng)%2 ? cvSize(4,1) : cvSize(1,4);

    sz.width = MAX(sz.width,16);
    sz.height = MAX(sz.height,16);
    sizes[OUTPUT][0] = sizes[OUTPUT][1] =
        sizes[REF_OUTPUT][0] = sizes[REF_OUTPUT][1] = sz;
}


void CV_UndistortMapTest::fill_array( int test_case_idx, int i, int j, Mat& arr )
{
    if( i != INPUT )
        cvtest::ArrayTest::fill_array( test_case_idx, i, j, arr );
}


void CV_UndistortMapTest::run_func()
{
    cv::Mat a = test_mat[INPUT][0], k = test_mat[INPUT][1];
    cv::Mat &mapx = test_mat[OUTPUT][0], &mapy = !dualChannel ? test_mat[OUTPUT][1] : mapx;
    cv::Size mapsz = test_mat[OUTPUT][0].size();

    cv::initUndistortRectifyMap(a, k, cv::Mat(), a,
        mapsz, dualChannel ? CV_32FC2 : CV_32FC1,
        mapx, !dualChannel ? cv::_InputOutputArray(mapy) : cv::noArray());
}


double CV_UndistortMapTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 1e-3;
}


int CV_UndistortMapTest::prepare_test_case( int test_case_idx )
{
    RNG& rng = ts->get_rng();
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );
    const Mat& mapx = test_mat[OUTPUT][0];
    double k[4], a[9] = {0,0,0,0,0,0,0,0,1};
    double sz = MAX(mapx.rows, mapx.cols);
    Mat& _a0 = test_mat[INPUT][0], &_k0 = test_mat[INPUT][1];
    Mat _a(3,3,CV_64F,a);
    Mat _k(_k0.rows,_k0.cols, CV_MAKETYPE(CV_64F,_k0.channels()),k);

    if( code <= 0 )
        return code;

    double aspect_ratio = cvtest::randReal(rng)*0.6 + 0.7;
    a[2] = (mapx.cols - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    a[5] = (mapx.rows - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    a[0] = sz/(0.9 - cvtest::randReal(rng)*0.6);
    a[4] = aspect_ratio*a[0];
    k[0] = cvtest::randReal(rng)*0.06 - 0.03;
    k[1] = cvtest::randReal(rng)*0.06 - 0.03;
    if( k[0]*k[1] > 0 )
        k[1] = -k[1];
    k[2] = cvtest::randReal(rng)*0.004 - 0.002;
    k[3] = cvtest::randReal(rng)*0.004 - 0.002;

    _a.convertTo(_a0, _a0.depth());
    _k.convertTo(_k0, _k0.depth());

    if (dualChannel)
    {
        test_mat[REF_OUTPUT][1] = Scalar::all(0);
        test_mat[OUTPUT][1] = Scalar::all(0);
    }

    return code;
}


void CV_UndistortMapTest::prepare_to_validation( int )
{
    Mat mapx, mapy;
    cvtest::initUndistortMap( test_mat[INPUT][0], test_mat[INPUT][1], Mat(), Mat(),
                              test_mat[REF_OUTPUT][0].size(), mapx, mapy, CV_32F );
    if( !dualChannel )
    {
        mapx.copyTo(test_mat[REF_OUTPUT][0]);
        mapy.copyTo(test_mat[REF_OUTPUT][1]);
    }
    else
    {
        Mat p[2] = {mapx, mapy};
        cv::merge(p, 2, test_mat[REF_OUTPUT][0]);
    }
}

TEST(Calib3d_Undistort, accuracy) { CV_UndistortTest test; test.safe_run(); }
TEST(Calib3d_InitUndistortMap, accuracy) { CV_UndistortMapTest test; test.safe_run(); }

}} // namespace
