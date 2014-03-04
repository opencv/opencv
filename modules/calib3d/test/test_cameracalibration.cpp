/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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

#include "test_precomp.hpp"
#include "opencv2/calib3d/calib3d_c.h"

#include <limits>

using namespace std;
using namespace cv;

#if 0
class CV_ProjectPointsTest : public cvtest::ArrayTest
{
public:
    CV_ProjectPointsTest();

protected:
    int read_params( CvFileStorage* fs );
    void fill_array( int test_case_idx, int i, int j, Mat& arr );
    int prepare_test_case( int test_case_idx );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );

    bool calc_jacobians;
};


CV_ProjectPointsTest::CV_ProjectPointsTest()
    : cvtest::ArrayTest( "3d-ProjectPoints", "cvProjectPoints2", "" )
{
    test_array[INPUT].push_back(NULL);  // rotation vector
    test_array[OUTPUT].push_back(NULL); // rotation matrix
    test_array[OUTPUT].push_back(NULL); // jacobian (J)
    test_array[OUTPUT].push_back(NULL); // rotation vector (backward transform result)
    test_array[OUTPUT].push_back(NULL); // inverse transform jacobian (J1)
    test_array[OUTPUT].push_back(NULL); // J*J1 (or J1*J) == I(3x3)
    test_array[REF_OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);

    element_wise_relative_error = false;
    calc_jacobians = false;
}


int CV_ProjectPointsTest::read_params( CvFileStorage* fs )
{
    int code = cvtest::ArrayTest::read_params( fs );
    return code;
}


void CV_ProjectPointsTest::get_test_array_types_and_sizes(
    int /*test_case_idx*/, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int depth = cvtest::randInt(rng) % 2 == 0 ? CV_32F : CV_64F;
    int i, code;

    code = cvtest::randInt(rng) % 3;
    types[INPUT][0] = CV_MAKETYPE(depth, 1);

    if( code == 0 )
    {
        sizes[INPUT][0] = cvSize(1,1);
        types[INPUT][0] = CV_MAKETYPE(depth, 3);
    }
    else if( code == 1 )
        sizes[INPUT][0] = cvSize(3,1);
    else
        sizes[INPUT][0] = cvSize(1,3);

    sizes[OUTPUT][0] = cvSize(3, 3);
    types[OUTPUT][0] = CV_MAKETYPE(depth, 1);

    types[OUTPUT][1] = CV_MAKETYPE(depth, 1);

    if( cvtest::randInt(rng) % 2 )
        sizes[OUTPUT][1] = cvSize(3,9);
    else
        sizes[OUTPUT][1] = cvSize(9,3);

    types[OUTPUT][2] = types[INPUT][0];
    sizes[OUTPUT][2] = sizes[INPUT][0];

    types[OUTPUT][3] = types[OUTPUT][1];
    sizes[OUTPUT][3] = cvSize(sizes[OUTPUT][1].height, sizes[OUTPUT][1].width);

    types[OUTPUT][4] = types[OUTPUT][1];
    sizes[OUTPUT][4] = cvSize(3,3);

    calc_jacobians = 1;//cvtest::randInt(rng) % 3 != 0;
    if( !calc_jacobians )
        sizes[OUTPUT][1] = sizes[OUTPUT][3] = sizes[OUTPUT][4] = cvSize(0,0);

    for( i = 0; i < 5; i++ )
    {
        types[REF_OUTPUT][i] = types[OUTPUT][i];
        sizes[REF_OUTPUT][i] = sizes[OUTPUT][i];
    }
}


double CV_ProjectPointsTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int j )
{
    return j == 4 ? 1e-2 : 1e-2;
}


void CV_ProjectPointsTest::fill_array( int /*test_case_idx*/, int /*i*/, int /*j*/, CvMat* arr )
{
    double r[3], theta0, theta1, f;
    CvMat _r = cvMat( arr->rows, arr->cols, CV_MAKETYPE(CV_64F,CV_MAT_CN(arr->type)), r );
    RNG& rng = ts->get_rng();

    r[0] = cvtest::randReal(rng)*CV_PI*2;
    r[1] = cvtest::randReal(rng)*CV_PI*2;
    r[2] = cvtest::randReal(rng)*CV_PI*2;

    theta0 = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    theta1 = fmod(theta0, CV_PI*2);

    if( theta1 > CV_PI )
        theta1 = -(CV_PI*2 - theta1);

    f = theta1/(theta0 ? theta0 : 1);
    r[0] *= f;
    r[1] *= f;
    r[2] *= f;

    cvTsConvert( &_r, arr );
}


int CV_ProjectPointsTest::prepare_test_case( int test_case_idx )
{
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );
    return code;
}


void CV_ProjectPointsTest::run_func()
{
    CvMat *v2m_jac = 0, *m2v_jac = 0;
    if( calc_jacobians )
    {
        v2m_jac = &test_mat[OUTPUT][1];
        m2v_jac = &test_mat[OUTPUT][3];
    }

    cvProjectPoints2( &test_mat[INPUT][0], &test_mat[OUTPUT][0], v2m_jac );
    cvProjectPoints2( &test_mat[OUTPUT][0], &test_mat[OUTPUT][2], m2v_jac );
}


void CV_ProjectPointsTest::prepare_to_validation( int /*test_case_idx*/ )
{
    const CvMat* vec = &test_mat[INPUT][0];
    CvMat* m = &test_mat[REF_OUTPUT][0];
    CvMat* vec2 = &test_mat[REF_OUTPUT][2];
    CvMat* v2m_jac = 0, *m2v_jac = 0;
    double theta0, theta1;

    if( calc_jacobians )
    {
        v2m_jac = &test_mat[REF_OUTPUT][1];
        m2v_jac = &test_mat[REF_OUTPUT][3];
    }


    cvTsProjectPoints( vec, m, v2m_jac );
    cvTsProjectPoints( m, vec2, m2v_jac );
    cvTsCopy( vec, vec2 );

    theta0 = cvNorm( vec2, 0, CV_L2 );
    theta1 = fmod( theta0, CV_PI*2 );

    if( theta1 > CV_PI )
        theta1 = -(CV_PI*2 - theta1);
    cvScale( vec2, vec2, theta1/(theta0 ? theta0 : 1) );

    if( calc_jacobians )
    {
        //cvInvert( v2m_jac, m2v_jac, CV_SVD );
        if( cvNorm(&test_mat[OUTPUT][3],0,CV_C) < 1000 )
        {
            cvTsGEMM( &test_mat[OUTPUT][1], &test_mat[OUTPUT][3],
                      1, 0, 0, &test_mat[OUTPUT][4],
                      v2m_jac->rows == 3 ? 0 : CV_GEMM_A_T + CV_GEMM_B_T );
        }
        else
        {
            cvTsSetIdentity( &test_mat[OUTPUT][4], cvScalarAll(1.) );
            cvTsCopy( &test_mat[REF_OUTPUT][2], &test_mat[OUTPUT][2] );
        }
        cvTsSetIdentity( &test_mat[REF_OUTPUT][4], cvScalarAll(1.) );
    }
}


CV_ProjectPointsTest ProjectPoints_test;

#endif

// --------------------------------- CV_CameraCalibrationTest --------------------------------------------

class CV_CameraCalibrationTest : public cvtest::BaseTest
{
public:
    CV_CameraCalibrationTest();
    ~CV_CameraCalibrationTest();
    void clear();
protected:
    int compare(double* val, double* refVal, int len,
                double eps, const char* paramName);
    virtual void calibrate( int imageCount, int* pointCounts,
        CvSize imageSize, CvPoint2D64f* imagePoints, CvPoint3D64f* objectPoints,
        double* distortionCoeffs, double* cameraMatrix, double* translationVectors,
        double* rotationMatrices, int flags ) = 0;
    virtual void project( int pointCount, CvPoint3D64f* objectPoints,
        double* rotationMatrix, double*  translationVector,
        double* cameraMatrix, double* distortion, CvPoint2D64f* imagePoints ) = 0;

    void run(int);
};

CV_CameraCalibrationTest::CV_CameraCalibrationTest()
{
}

CV_CameraCalibrationTest::~CV_CameraCalibrationTest()
{
    clear();
}

void CV_CameraCalibrationTest::clear()
{
    cvtest::BaseTest::clear();
}

int CV_CameraCalibrationTest::compare(double* val, double* ref_val, int len,
                                      double eps, const char* param_name )
{
    return cvtest::cmpEps2_64f( ts, val, ref_val, len, eps, param_name );
}

void CV_CameraCalibrationTest::run( int start_from )
{
    int code = cvtest::TS::OK;
    cv::String            filepath;
    cv::String            filename;

    CvSize          imageSize;
    CvSize          etalonSize;
    int             numImages;

    CvPoint2D64f*   imagePoints;
    CvPoint3D64f*   objectPoints;
    CvPoint2D64f*   reprojectPoints;

    double*       transVects;
    double*       rotMatrs;

    double*       goodTransVects;
    double*       goodRotMatrs;

    double          cameraMatrix[3*3];
    double          distortion[5]={0,0,0,0,0};

    double          goodDistortion[4];

    int*            numbers;
    FILE*           file = 0;
    FILE*           datafile = 0;
    int             i,j;
    int             currImage;
    int             currPoint;

    int             calibFlags;
    char            i_dat_file[100];
    int             numPoints;
    int numTests;
    int currTest;

    imagePoints     = 0;
    objectPoints    = 0;
    reprojectPoints = 0;
    numbers         = 0;

    transVects      = 0;
    rotMatrs        = 0;
    goodTransVects  = 0;
    goodRotMatrs    = 0;
    int progress = 0;
    int values_read = -1;

    filepath = cv::format("%scv/cameracalibration/", ts->get_data_path().c_str() );
    filename = cv::format("%sdatafiles.txt", filepath.c_str() );
    datafile = fopen( filename.c_str(), "r" );
    if( datafile == 0 )
    {
        ts->printf( cvtest::TS::LOG, "Could not open file with list of test files: %s\n", filename.c_str() );
        code = cvtest::TS::FAIL_MISSING_TEST_DATA;
        goto _exit_;
    }

    values_read = fscanf(datafile,"%d",&numTests);
    CV_Assert(values_read == 1);

    for( currTest = start_from; currTest < numTests; currTest++ )
    {
        values_read = fscanf(datafile,"%s",i_dat_file);
        CV_Assert(values_read == 1);
        filename = cv::format("%s%s", filepath.c_str(), i_dat_file);
        file = fopen(filename.c_str(),"r");

        ts->update_context( this, currTest, true );

        if( file == 0 )
        {
            ts->printf( cvtest::TS::LOG,
                "Can't open current test file: %s\n",filename.c_str());
            if( numTests == 1 )
            {
                code = cvtest::TS::FAIL_MISSING_TEST_DATA;
                goto _exit_;
            }
            continue; // if there is more than one test, just skip the test
        }

        values_read = fscanf(file,"%d %d\n",&(imageSize.width),&(imageSize.height));
        CV_Assert(values_read == 2);
        if( imageSize.width <= 0 || imageSize.height <= 0 )
        {
            ts->printf( cvtest::TS::LOG, "Image size in test file is incorrect\n" );
            code = cvtest::TS::FAIL_INVALID_TEST_DATA;
            goto _exit_;
        }

        /* Read etalon size */
        values_read = fscanf(file,"%d %d\n",&(etalonSize.width),&(etalonSize.height));
        CV_Assert(values_read == 2);
        if( etalonSize.width <= 0 || etalonSize.height <= 0 )
        {
            ts->printf( cvtest::TS::LOG, "Pattern size in test file is incorrect\n" );
            code = cvtest::TS::FAIL_INVALID_TEST_DATA;
            goto _exit_;
        }

        numPoints = etalonSize.width * etalonSize.height;

        /* Read number of images */
        values_read = fscanf(file,"%d\n",&numImages);
        CV_Assert(values_read == 1);
        if( numImages <=0 )
        {
            ts->printf( cvtest::TS::LOG, "Number of images in test file is incorrect\n");
            code = cvtest::TS::FAIL_INVALID_TEST_DATA;
            goto _exit_;
        }

        /* Need to allocate memory */
        imagePoints     = (CvPoint2D64f*)cvAlloc( numPoints *
                                                    numImages * sizeof(CvPoint2D64f));

        objectPoints    = (CvPoint3D64f*)cvAlloc( numPoints *
                                                    numImages * sizeof(CvPoint3D64f));

        reprojectPoints = (CvPoint2D64f*)cvAlloc( numPoints *
                                                    numImages * sizeof(CvPoint2D64f));

        /* Alloc memory for numbers */
        numbers = (int*)cvAlloc( numImages * sizeof(int));

        /* Fill it by numbers of points of each image*/
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            numbers[currImage] = etalonSize.width * etalonSize.height;
        }

        /* Allocate memory for translate vectors and rotmatrixs*/
        transVects     = (double*)cvAlloc(3 * 1 * numImages * sizeof(double));
        rotMatrs       = (double*)cvAlloc(3 * 3 * numImages * sizeof(double));

        goodTransVects = (double*)cvAlloc(3 * 1 * numImages * sizeof(double));
        goodRotMatrs   = (double*)cvAlloc(3 * 3 * numImages * sizeof(double));

        /* Read object points */
        i = 0;/* shift for current point */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            for( currPoint = 0; currPoint < numPoints; currPoint++ )
            {
                double x,y,z;
                values_read = fscanf(file,"%lf %lf %lf\n",&x,&y,&z);
                CV_Assert(values_read == 3);

                (objectPoints+i)->x = x;
                (objectPoints+i)->y = y;
                (objectPoints+i)->z = z;
                i++;
            }
        }

        /* Read image points */
        i = 0;/* shift for current point */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            for( currPoint = 0; currPoint < numPoints; currPoint++ )
            {
                double x,y;
                values_read = fscanf(file,"%lf %lf\n",&x,&y);
                CV_Assert(values_read == 2);

                (imagePoints+i)->x = x;
                (imagePoints+i)->y = y;
                i++;
            }
        }

        /* Read good data computed before */

        /* Focal lengths */
        double goodFcx,goodFcy;
        values_read = fscanf(file,"%lf %lf",&goodFcx,&goodFcy);
        CV_Assert(values_read == 2);

        /* Principal points */
        double goodCx,goodCy;
        values_read = fscanf(file,"%lf %lf",&goodCx,&goodCy);
        CV_Assert(values_read == 2);

        /* Read distortion */

        values_read = fscanf(file,"%lf",goodDistortion+0); CV_Assert(values_read == 1);
        values_read = fscanf(file,"%lf",goodDistortion+1); CV_Assert(values_read == 1);
        values_read = fscanf(file,"%lf",goodDistortion+2); CV_Assert(values_read == 1);
        values_read = fscanf(file,"%lf",goodDistortion+3); CV_Assert(values_read == 1);

        /* Read good Rot matrices */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            for( i = 0; i < 3; i++ )
                for( j = 0; j < 3; j++ )
                {
                    values_read = fscanf(file, "%lf", goodRotMatrs + currImage * 9 + j * 3 + i);
                    CV_Assert(values_read == 1);
                }
        }

        /* Read good Trans vectors */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            for( i = 0; i < 3; i++ )
            {
                values_read = fscanf(file, "%lf", goodTransVects + currImage * 3 + i);
                CV_Assert(values_read == 1);
            }
        }

        calibFlags = 0
                     // + CV_CALIB_FIX_PRINCIPAL_POINT
                     // + CV_CALIB_ZERO_TANGENT_DIST
                     // + CV_CALIB_FIX_ASPECT_RATIO
                     // + CV_CALIB_USE_INTRINSIC_GUESS
                     + CV_CALIB_FIX_K3
                     + CV_CALIB_FIX_K4+CV_CALIB_FIX_K5
                     + CV_CALIB_FIX_K6
                    ;
        memset( cameraMatrix, 0, 9*sizeof(cameraMatrix[0]) );
        cameraMatrix[0] = cameraMatrix[4] = 807.;
        cameraMatrix[2] = (imageSize.width - 1)*0.5;
        cameraMatrix[5] = (imageSize.height - 1)*0.5;
        cameraMatrix[8] = 1.;

        /* Now we can calibrate camera */
        calibrate(  numImages,
                    numbers,
                    imageSize,
                    imagePoints,
                    objectPoints,
                    distortion,
                    cameraMatrix,
                    transVects,
                    rotMatrs,
                    calibFlags );

        /* ---- Reproject points to the image ---- */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            int nPoints = etalonSize.width * etalonSize.height;
            project(  nPoints,
                      objectPoints + currImage * nPoints,
                      rotMatrs + currImage * 9,
                      transVects + currImage * 3,
                      cameraMatrix,
                      distortion,
                      reprojectPoints + currImage * nPoints);
        }

        /* ----- Compute reprojection error ----- */
        i = 0;
        double dx,dy;
        double rx,ry;
        double meanDx,meanDy;
        double maxDx = 0.0;
        double maxDy = 0.0;

        meanDx = 0;
        meanDy = 0;
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            for( currPoint = 0; currPoint < etalonSize.width * etalonSize.height; currPoint++ )
            {
                rx = reprojectPoints[i].x;
                ry = reprojectPoints[i].y;
                dx = rx - imagePoints[i].x;
                dy = ry - imagePoints[i].y;

                meanDx += dx;
                meanDy += dy;

                dx = fabs(dx);
                dy = fabs(dy);

                if( dx > maxDx )
                    maxDx = dx;

                if( dy > maxDy )
                    maxDy = dy;
                i++;
            }
        }

        meanDx /= numImages * etalonSize.width * etalonSize.height;
        meanDy /= numImages * etalonSize.width * etalonSize.height;

        /* ========= Compare parameters ========= */

        /* ----- Compare focal lengths ----- */
        code = compare(cameraMatrix+0,&goodFcx,1,0.1,"fx");
        if( code < 0 )
            goto _exit_;

        code = compare(cameraMatrix+4,&goodFcy,1,0.1,"fy");
        if( code < 0 )
            goto _exit_;

        /* ----- Compare principal points ----- */
        code = compare(cameraMatrix+2,&goodCx,1,0.1,"cx");
        if( code < 0 )
            goto _exit_;

        code = compare(cameraMatrix+5,&goodCy,1,0.1,"cy");
        if( code < 0 )
            goto _exit_;

        /* ----- Compare distortion ----- */
        code = compare(distortion,goodDistortion,4,0.1,"[k1,k2,p1,p2]");
        if( code < 0 )
            goto _exit_;

        /* ----- Compare rot matrixs ----- */
        code = compare(rotMatrs,goodRotMatrs, 9*numImages,0.05,"rotation matrices");
        if( code < 0 )
            goto _exit_;

        /* ----- Compare rot matrixs ----- */
        code = compare(transVects,goodTransVects, 3*numImages,0.1,"translation vectors");
        if( code < 0 )
            goto _exit_;

        if( maxDx > 1.0 )
        {
            ts->printf( cvtest::TS::LOG,
                      "Error in reprojection maxDx=%f > 1.0\n",maxDx);
            code = cvtest::TS::FAIL_BAD_ACCURACY; goto _exit_;
        }

        if( maxDy > 1.0 )
        {
            ts->printf( cvtest::TS::LOG,
                      "Error in reprojection maxDy=%f > 1.0\n",maxDy);
            code = cvtest::TS::FAIL_BAD_ACCURACY; goto _exit_;
        }

        progress = update_progress( progress, currTest, numTests, 0 );

        cvFree(&imagePoints);
        cvFree(&objectPoints);
        cvFree(&reprojectPoints);
        cvFree(&numbers);

        cvFree(&transVects);
        cvFree(&rotMatrs);
        cvFree(&goodTransVects);
        cvFree(&goodRotMatrs);

        fclose(file);
        file = 0;
    }

_exit_:

    if( file )
        fclose(file);

    if( datafile )
        fclose(datafile);

    /* Free all allocated memory */
    cvFree(&imagePoints);
    cvFree(&objectPoints);
    cvFree(&reprojectPoints);
    cvFree(&numbers);

    cvFree(&transVects);
    cvFree(&rotMatrs);
    cvFree(&goodTransVects);
    cvFree(&goodRotMatrs);

    if( code < 0 )
        ts->set_failed_test_info( code );
}

// --------------------------------- CV_CameraCalibrationTest_C --------------------------------------------

class CV_CameraCalibrationTest_C : public CV_CameraCalibrationTest
{
public:
    CV_CameraCalibrationTest_C(){}
protected:
    virtual void calibrate( int imageCount, int* pointCounts,
        CvSize imageSize, CvPoint2D64f* imagePoints, CvPoint3D64f* objectPoints,
        double* distortionCoeffs, double* cameraMatrix, double* translationVectors,
        double* rotationMatrices, int flags );
    virtual void project( int pointCount, CvPoint3D64f* objectPoints,
        double* rotationMatrix, double*  translationVector,
        double* cameraMatrix, double* distortion, CvPoint2D64f* imagePoints );
};

void CV_CameraCalibrationTest_C::calibrate( int imageCount, int* pointCounts,
        CvSize imageSize, CvPoint2D64f* imagePoints, CvPoint3D64f* objectPoints,
        double* distortionCoeffs, double* cameraMatrix, double* translationVectors,
        double* rotationMatrices, int flags )
{
    int i, total = 0;
    for( i = 0; i < imageCount; i++ )
        total += pointCounts[i];

    CvMat _objectPoints = cvMat(1, total, CV_64FC3, objectPoints);
    CvMat _imagePoints = cvMat(1, total, CV_64FC2, imagePoints);
    CvMat _pointCounts = cvMat(1, imageCount, CV_32S, pointCounts);
    CvMat _cameraMatrix = cvMat(3, 3, CV_64F, cameraMatrix);
    CvMat _distCoeffs = cvMat(4, 1, CV_64F, distortionCoeffs);
    CvMat _rotationMatrices = cvMat(imageCount, 9, CV_64F, rotationMatrices);
    CvMat _translationVectors = cvMat(imageCount, 3, CV_64F, translationVectors);

    cvCalibrateCamera2(&_objectPoints, &_imagePoints, &_pointCounts, imageSize,
                       &_cameraMatrix, &_distCoeffs, &_rotationMatrices, &_translationVectors,
                       flags);
}

void CV_CameraCalibrationTest_C::project( int pointCount, CvPoint3D64f* objectPoints,
        double* rotationMatrix, double*  translationVector,
        double* cameraMatrix, double* distortion, CvPoint2D64f* imagePoints )
{
    CvMat _objectPoints = cvMat(1, pointCount, CV_64FC3, objectPoints);
    CvMat _imagePoints = cvMat(1, pointCount, CV_64FC2, imagePoints);
    CvMat _cameraMatrix = cvMat(3, 3, CV_64F, cameraMatrix);
    CvMat _distCoeffs = cvMat(4, 1, CV_64F, distortion);
    CvMat _rotationMatrix = cvMat(3, 3, CV_64F, rotationMatrix);
    CvMat _translationVector = cvMat(1, 3, CV_64F, translationVector);

    cvProjectPoints2(&_objectPoints, &_rotationMatrix, &_translationVector, &_cameraMatrix, &_distCoeffs, &_imagePoints);
}

// --------------------------------- CV_CameraCalibrationTest_CPP --------------------------------------------

class CV_CameraCalibrationTest_CPP : public CV_CameraCalibrationTest
{
public:
    CV_CameraCalibrationTest_CPP(){}
protected:
    virtual void calibrate( int imageCount, int* pointCounts,
        CvSize imageSize, CvPoint2D64f* imagePoints, CvPoint3D64f* objectPoints,
        double* distortionCoeffs, double* cameraMatrix, double* translationVectors,
        double* rotationMatrices, int flags );
    virtual void project( int pointCount, CvPoint3D64f* objectPoints,
        double* rotationMatrix, double*  translationVector,
        double* cameraMatrix, double* distortion, CvPoint2D64f* imagePoints );
};

void CV_CameraCalibrationTest_CPP::calibrate( int imageCount, int* pointCounts,
        CvSize _imageSize, CvPoint2D64f* _imagePoints, CvPoint3D64f* _objectPoints,
        double* _distortionCoeffs, double* _cameraMatrix, double* translationVectors,
        double* rotationMatrices, int flags )
{
    vector<vector<Point3f> > objectPoints( imageCount );
    vector<vector<Point2f> > imagePoints( imageCount );
    Size imageSize = _imageSize;
    Mat cameraMatrix, distCoeffs(1,4,CV_64F,Scalar::all(0));
    vector<Mat> rvecs, tvecs;

    CvPoint3D64f* op = _objectPoints;
    CvPoint2D64f* ip = _imagePoints;
    vector<vector<Point3f> >::iterator objectPointsIt = objectPoints.begin();
    vector<vector<Point2f> >::iterator imagePointsIt = imagePoints.begin();
    for( int i = 0; i < imageCount; ++objectPointsIt, ++imagePointsIt, i++ )
    {
        int num = pointCounts[i];
        objectPointsIt->resize( num );
        imagePointsIt->resize( num );
        vector<Point3f>::iterator oIt = objectPointsIt->begin();
        vector<Point2f>::iterator iIt = imagePointsIt->begin();
        for( int j = 0; j < num; ++oIt, ++iIt, j++, op++, ip++)
        {
            oIt->x = (float)op->x, oIt->y = (float)op->y, oIt->z = (float)op->z;
            iIt->x = (float)ip->x, iIt->y = (float)ip->y;
        }
    }

    calibrateCamera( objectPoints,
                     imagePoints,
                     imageSize,
                     cameraMatrix,
                     distCoeffs,
                     rvecs,
                     tvecs,
                     flags );

    assert( cameraMatrix.type() == CV_64FC1 );
    memcpy( _cameraMatrix, cameraMatrix.data, 9*sizeof(double) );

    assert( cameraMatrix.type() == CV_64FC1 );
    memcpy( _distortionCoeffs, distCoeffs.data, 4*sizeof(double) );

    vector<Mat>::iterator rvecsIt = rvecs.begin();
    vector<Mat>::iterator tvecsIt = tvecs.begin();
    double *rm = rotationMatrices,
           *tm = translationVectors;
    assert( rvecsIt->type() == CV_64FC1 );
    assert( tvecsIt->type() == CV_64FC1 );
    for( int i = 0; i < imageCount; ++rvecsIt, ++tvecsIt, i++, rm+=9, tm+=3 )
    {
        Mat r9( 3, 3, CV_64FC1 );
        Rodrigues( *rvecsIt, r9 );
        memcpy( rm, r9.data, 9*sizeof(double) );
        memcpy( tm, tvecsIt->data, 3*sizeof(double) );
    }
}

void CV_CameraCalibrationTest_CPP::project( int pointCount, CvPoint3D64f* _objectPoints,
        double* rotationMatrix, double*  translationVector,
        double* _cameraMatrix, double* distortion, CvPoint2D64f* _imagePoints )
{
    Mat objectPoints( pointCount, 3, CV_64FC1, _objectPoints );
    Mat rmat( 3, 3, CV_64FC1, rotationMatrix ),
        rvec( 1, 3, CV_64FC1 ),
        tvec( 1, 3, CV_64FC1, translationVector );
    Mat cameraMatrix( 3, 3, CV_64FC1, _cameraMatrix );
    Mat distCoeffs( 1, 4, CV_64FC1, distortion );
    vector<Point2f> imagePoints;
    Rodrigues( rmat, rvec );

    objectPoints.convertTo( objectPoints, CV_32FC1 );
    projectPoints( objectPoints, rvec, tvec,
                   cameraMatrix, distCoeffs, imagePoints );
    vector<Point2f>::const_iterator it = imagePoints.begin();
    for( int i = 0; it != imagePoints.end(); ++it, i++ )
    {
        _imagePoints[i] = cvPoint2D64f( it->x, it->y );
    }
}


//----------------------------------------- CV_CalibrationMatrixValuesTest --------------------------------

class CV_CalibrationMatrixValuesTest : public cvtest::BaseTest
{
public:
    CV_CalibrationMatrixValuesTest() {}
protected:
    void run(int);
    virtual void calibMatrixValues( const Mat& cameraMatrix, Size imageSize,
        double apertureWidth, double apertureHeight, double& fovx, double& fovy, double& focalLength,
        Point2d& principalPoint, double& aspectRatio ) = 0;
};

void CV_CalibrationMatrixValuesTest::run(int)
{
    int code = cvtest::TS::OK;
    const double fcMinVal = 1e-5;
    const double fcMaxVal = 1000;
    const double apertureMaxVal = 0.01;

    RNG rng = ts->get_rng();

    double fx, fy, cx, cy, nx, ny;
    Mat cameraMatrix( 3, 3, CV_64FC1 );
    cameraMatrix.setTo( Scalar(0) );
    fx = cameraMatrix.at<double>(0,0) = rng.uniform( fcMinVal, fcMaxVal );
    fy = cameraMatrix.at<double>(1,1) = rng.uniform( fcMinVal, fcMaxVal );
    cx = cameraMatrix.at<double>(0,2) = rng.uniform( fcMinVal, fcMaxVal );
    cy = cameraMatrix.at<double>(1,2) = rng.uniform( fcMinVal, fcMaxVal );
    cameraMatrix.at<double>(2,2) = 1;

    Size imageSize( 600, 400 );

    double apertureWidth = (double)rng * apertureMaxVal,
           apertureHeight = (double)rng * apertureMaxVal;

    double fovx, fovy, focalLength, aspectRatio,
           goodFovx, goodFovy, goodFocalLength, goodAspectRatio;
    Point2d principalPoint, goodPrincipalPoint;


    calibMatrixValues( cameraMatrix, imageSize, apertureWidth, apertureHeight,
        fovx, fovy, focalLength, principalPoint, aspectRatio );

    // calculate calibration matrix values
    goodAspectRatio = fy / fx;

    if( apertureWidth != 0.0 && apertureHeight != 0.0 )
    {
        nx = imageSize.width / apertureWidth;
        ny = imageSize.height / apertureHeight;
    }
    else
    {
        nx = 1.0;
        ny = goodAspectRatio;
    }

    goodFovx = 2 * atan( imageSize.width / (2 * fx)) * 180.0 / CV_PI;
    goodFovy = 2 * atan( imageSize.height / (2 * fy)) * 180.0 / CV_PI;

    goodFocalLength = fx / nx;

    goodPrincipalPoint.x = cx / nx;
    goodPrincipalPoint.y = cy / ny;

    // check results
    if( fabs(fovx - goodFovx) > FLT_EPSILON )
    {
        ts->printf( cvtest::TS::LOG, "bad fovx (real=%f, good = %f\n", fovx, goodFovx );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }
    if( fabs(fovy - goodFovy) > FLT_EPSILON )
    {
        ts->printf( cvtest::TS::LOG, "bad fovy (real=%f, good = %f\n", fovy, goodFovy );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }
    if( fabs(focalLength - goodFocalLength) > FLT_EPSILON )
    {
        ts->printf( cvtest::TS::LOG, "bad focalLength (real=%f, good = %f\n", focalLength, goodFocalLength );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }
    if( fabs(aspectRatio - goodAspectRatio) > FLT_EPSILON )
    {
        ts->printf( cvtest::TS::LOG, "bad aspectRatio (real=%f, good = %f\n", aspectRatio, goodAspectRatio );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }
    if( norm( principalPoint - goodPrincipalPoint ) > FLT_EPSILON )
    {
        ts->printf( cvtest::TS::LOG, "bad principalPoint\n" );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

_exit_:
    RNG& _rng = ts->get_rng();
    _rng = rng;
    ts->set_failed_test_info( code );
}

//----------------------------------------- CV_CalibrationMatrixValuesTest_C --------------------------------

class CV_CalibrationMatrixValuesTest_C : public CV_CalibrationMatrixValuesTest
{
public:
    CV_CalibrationMatrixValuesTest_C(){}
protected:
    virtual void calibMatrixValues( const Mat& cameraMatrix, Size imageSize,
        double apertureWidth, double apertureHeight, double& fovx, double& fovy, double& focalLength,
        Point2d& principalPoint, double& aspectRatio );
};

void CV_CalibrationMatrixValuesTest_C::calibMatrixValues( const Mat& _cameraMatrix, Size imageSize,
                                               double apertureWidth, double apertureHeight,
                                               double& fovx, double& fovy, double& focalLength,
                                               Point2d& principalPoint, double& aspectRatio )
{
    CvMat cameraMatrix = _cameraMatrix;
    CvPoint2D64f pp;
    cvCalibrationMatrixValues( &cameraMatrix, imageSize, apertureWidth, apertureHeight,
        &fovx, &fovy, &focalLength, &pp, &aspectRatio );
    principalPoint.x = pp.x;
    principalPoint.y = pp.y;
}


//----------------------------------------- CV_CalibrationMatrixValuesTest_CPP --------------------------------

class CV_CalibrationMatrixValuesTest_CPP : public CV_CalibrationMatrixValuesTest
{
public:
    CV_CalibrationMatrixValuesTest_CPP() {}
protected:
    virtual void calibMatrixValues( const Mat& cameraMatrix, Size imageSize,
        double apertureWidth, double apertureHeight, double& fovx, double& fovy, double& focalLength,
        Point2d& principalPoint, double& aspectRatio );
};

void CV_CalibrationMatrixValuesTest_CPP::calibMatrixValues( const Mat& cameraMatrix, Size imageSize,
                                                         double apertureWidth, double apertureHeight,
                                                         double& fovx, double& fovy, double& focalLength,
                                                         Point2d& principalPoint, double& aspectRatio )
{
    calibrationMatrixValues( cameraMatrix, imageSize, apertureWidth, apertureHeight,
        fovx, fovy, focalLength, principalPoint, aspectRatio );
}


//----------------------------------------- CV_ProjectPointsTest --------------------------------
void calcdfdx( const vector<vector<Point2f> >& leftF, const vector<vector<Point2f> >& rightF, double eps, Mat& dfdx )
{
    const int fdim = 2;
    CV_Assert( !leftF.empty() && !rightF.empty() && !leftF[0].empty() && !rightF[0].empty() );
    CV_Assert( leftF[0].size() ==  rightF[0].size() );
    CV_Assert( fabs(eps) > std::numeric_limits<double>::epsilon() );
    int fcount = (int)leftF[0].size(), xdim = (int)leftF.size();

    dfdx.create( fcount*fdim, xdim, CV_64FC1 );

    vector<vector<Point2f> >::const_iterator arrLeftIt = leftF.begin();
    vector<vector<Point2f> >::const_iterator arrRightIt = rightF.begin();
    for( int xi = 0; xi < xdim; xi++, ++arrLeftIt, ++arrRightIt )
    {
        CV_Assert( (int)arrLeftIt->size() ==  fcount );
        CV_Assert( (int)arrRightIt->size() ==  fcount );
        vector<Point2f>::const_iterator lIt = arrLeftIt->begin();
        vector<Point2f>::const_iterator rIt = arrRightIt->begin();
        for( int fi = 0; fi < dfdx.rows; fi+=fdim, ++lIt, ++rIt )
        {
            dfdx.at<double>(fi, xi )   = 0.5 * ((double)(rIt->x - lIt->x)) / eps;
            dfdx.at<double>(fi+1, xi ) = 0.5 * ((double)(rIt->y - lIt->y)) / eps;
        }
    }
}

class CV_ProjectPointsTest : public cvtest::BaseTest
{
public:
    CV_ProjectPointsTest() {}
protected:
    void run(int);
    virtual void project( const Mat& objectPoints,
        const Mat& rvec, const Mat& tvec,
        const Mat& cameraMatrix,
        const Mat& distCoeffs,
        vector<Point2f>& imagePoints,
        Mat& dpdrot, Mat& dpdt, Mat& dpdf,
        Mat& dpdc, Mat& dpddist,
        double aspectRatio=0 ) = 0;
};

void CV_ProjectPointsTest::run(int)
{
    //typedef float matType;

    int code = cvtest::TS::OK;
    const int pointCount = 100;

    const float zMinVal = 10.0f, zMaxVal = 100.0f,
                rMinVal = -0.3f, rMaxVal = 0.3f,
                tMinVal = -2.0f, tMaxVal = 2.0f;

    const float imgPointErr = 1e-3f,
                dEps = 1e-3f;

    double err;

    Size imgSize( 600, 800 );
    Mat_<float> objPoints( pointCount, 3), rvec( 1, 3), rmat, tvec( 1, 3 ), cameraMatrix( 3, 3 ), distCoeffs( 1, 4 ),
      leftRvec, rightRvec, leftTvec, rightTvec, leftCameraMatrix, rightCameraMatrix, leftDistCoeffs, rightDistCoeffs;

    RNG rng = ts->get_rng();

    // generate data
    cameraMatrix << 300.f,  0.f,    imgSize.width/2.f,
                    0.f,    300.f,  imgSize.height/2.f,
                    0.f,    0.f,    1.f;
    distCoeffs << 0.1, 0.01, 0.001, 0.001;

    rvec(0,0) = rng.uniform( rMinVal, rMaxVal );
    rvec(0,1) = rng.uniform( rMinVal, rMaxVal );
    rvec(0,2) = rng.uniform( rMinVal, rMaxVal );
    Rodrigues( rvec, rmat );

    tvec(0,0) = rng.uniform( tMinVal, tMaxVal );
    tvec(0,1) = rng.uniform( tMinVal, tMaxVal );
    tvec(0,2) = rng.uniform( tMinVal, tMaxVal );

    for( int y = 0; y < objPoints.rows; y++ )
    {
        Mat point(1, 3, CV_32FC1, objPoints.ptr(y) );
        float z = rng.uniform( zMinVal, zMaxVal );
        point.at<float>(0,2) = z;
        point.at<float>(0,0) = (rng.uniform(2.f,(float)(imgSize.width-2)) - cameraMatrix(0,2)) / cameraMatrix(0,0) * z;
        point.at<float>(0,1) = (rng.uniform(2.f,(float)(imgSize.height-2)) - cameraMatrix(1,2)) / cameraMatrix(1,1) * z;
        point = (point - tvec) * rmat;
    }

    vector<Point2f> imgPoints;
    vector<vector<Point2f> > leftImgPoints;
    vector<vector<Point2f> > rightImgPoints;
    Mat dpdrot, dpdt, dpdf, dpdc, dpddist,
        valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist;

    project( objPoints, rvec, tvec, cameraMatrix, distCoeffs,
        imgPoints, dpdrot, dpdt, dpdf, dpdc, dpddist, 0 );

    // calculate and check image points
    assert( (int)imgPoints.size() == pointCount );
    vector<Point2f>::const_iterator it = imgPoints.begin();
    for( int i = 0; i < pointCount; i++, ++it )
    {
        Point3d p( objPoints(i,0), objPoints(i,1), objPoints(i,2) );
        double z = p.x*rmat(2,0) + p.y*rmat(2,1) + p.z*rmat(2,2) + tvec(0,2),
               x = (p.x*rmat(0,0) + p.y*rmat(0,1) + p.z*rmat(0,2) + tvec(0,0)) / z,
               y = (p.x*rmat(1,0) + p.y*rmat(1,1) + p.z*rmat(1,2) + tvec(0,1)) / z,
               r2 = x*x + y*y,
               r4 = r2*r2;
        Point2f validImgPoint;
        double a1 = 2*x*y,
               a2 = r2 + 2*x*x,
               a3 = r2 + 2*y*y,
               cdist = 1+distCoeffs(0,0)*r2+distCoeffs(0,1)*r4;
        validImgPoint.x = static_cast<float>((double)cameraMatrix(0,0)*(x*cdist + (double)distCoeffs(0,2)*a1 + (double)distCoeffs(0,3)*a2)
            + (double)cameraMatrix(0,2));
        validImgPoint.y = static_cast<float>((double)cameraMatrix(1,1)*(y*cdist + (double)distCoeffs(0,2)*a3 + distCoeffs(0,3)*a1)
            + (double)cameraMatrix(1,2));

        if( fabs(it->x - validImgPoint.x) > imgPointErr ||
            fabs(it->y - validImgPoint.y) > imgPointErr )
        {
            ts->printf( cvtest::TS::LOG, "bad image point\n" );
            code = cvtest::TS::FAIL_BAD_ACCURACY;
            goto _exit_;
        }
    }

    // check derivatives
    // 1. rotation
    leftImgPoints.resize(3);
    rightImgPoints.resize(3);
    for( int i = 0; i < 3; i++ )
    {
        rvec.copyTo( leftRvec ); leftRvec(0,i) -= dEps;
        project( objPoints, leftRvec, tvec, cameraMatrix, distCoeffs,
            leftImgPoints[i], valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist, 0 );
        rvec.copyTo( rightRvec ); rightRvec(0,i) += dEps;
        project( objPoints, rightRvec, tvec, cameraMatrix, distCoeffs,
            rightImgPoints[i], valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist, 0 );
    }
    calcdfdx( leftImgPoints, rightImgPoints, dEps, valDpdrot );
    err = norm( dpdrot, valDpdrot, NORM_INF );
    if( err > 3 )
    {
        ts->printf( cvtest::TS::LOG, "bad dpdrot: too big difference = %g\n", err );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    // 2. translation
    for( int i = 0; i < 3; i++ )
    {
        tvec.copyTo( leftTvec ); leftTvec(0,i) -= dEps;
        project( objPoints, rvec, leftTvec, cameraMatrix, distCoeffs,
            leftImgPoints[i], valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist, 0 );
        tvec.copyTo( rightTvec ); rightTvec(0,i) += dEps;
        project( objPoints, rvec, rightTvec, cameraMatrix, distCoeffs,
            rightImgPoints[i], valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist, 0 );
    }
    calcdfdx( leftImgPoints, rightImgPoints, dEps, valDpdt );
    if( norm( dpdt, valDpdt, NORM_INF ) > 0.2 )
    {
        ts->printf( cvtest::TS::LOG, "bad dpdtvec\n" );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    // 3. camera matrix
    // 3.1. focus
    leftImgPoints.resize(2);
    rightImgPoints.resize(2);
    cameraMatrix.copyTo( leftCameraMatrix ); leftCameraMatrix(0,0) -= dEps;
    project( objPoints, rvec, tvec, leftCameraMatrix, distCoeffs,
        leftImgPoints[0], valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist, 0 );
    cameraMatrix.copyTo( leftCameraMatrix ); leftCameraMatrix(1,1) -= dEps;
    project( objPoints, rvec, tvec, leftCameraMatrix, distCoeffs,
        leftImgPoints[1], valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist, 0 );
    cameraMatrix.copyTo( rightCameraMatrix ); rightCameraMatrix(0,0) += dEps;
    project( objPoints, rvec, tvec, rightCameraMatrix, distCoeffs,
        rightImgPoints[0], valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist, 0 );
    cameraMatrix.copyTo( rightCameraMatrix ); rightCameraMatrix(1,1) += dEps;
    project( objPoints, rvec, tvec, rightCameraMatrix, distCoeffs,
        rightImgPoints[1], valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist, 0 );
    calcdfdx( leftImgPoints, rightImgPoints, dEps, valDpdf );
    if ( norm( dpdf, valDpdf ) > 0.2 )
    {
        ts->printf( cvtest::TS::LOG, "bad dpdf\n" );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }
    // 3.2. principal point
    leftImgPoints.resize(2);
    rightImgPoints.resize(2);
    cameraMatrix.copyTo( leftCameraMatrix ); leftCameraMatrix(0,2) -= dEps;
    project( objPoints, rvec, tvec, leftCameraMatrix, distCoeffs,
        leftImgPoints[0], valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist, 0 );
    cameraMatrix.copyTo( leftCameraMatrix ); leftCameraMatrix(1,2) -= dEps;
    project( objPoints, rvec, tvec, leftCameraMatrix, distCoeffs,
        leftImgPoints[1], valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist, 0 );
    cameraMatrix.copyTo( rightCameraMatrix ); rightCameraMatrix(0,2) += dEps;
    project( objPoints, rvec, tvec, rightCameraMatrix, distCoeffs,
        rightImgPoints[0], valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist, 0 );
    cameraMatrix.copyTo( rightCameraMatrix ); rightCameraMatrix(1,2) += dEps;
    project( objPoints, rvec, tvec, rightCameraMatrix, distCoeffs,
        rightImgPoints[1], valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist, 0 );
    calcdfdx( leftImgPoints, rightImgPoints, dEps, valDpdc );
    if ( norm( dpdc, valDpdc ) > 0.2 )
    {
        ts->printf( cvtest::TS::LOG, "bad dpdc\n" );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    // 4. distortion
    leftImgPoints.resize(distCoeffs.cols);
    rightImgPoints.resize(distCoeffs.cols);
    for( int i = 0; i < distCoeffs.cols; i++ )
    {
        distCoeffs.copyTo( leftDistCoeffs ); leftDistCoeffs(0,i) -= dEps;
        project( objPoints, rvec, tvec, cameraMatrix, leftDistCoeffs,
            leftImgPoints[i], valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist, 0 );
        distCoeffs.copyTo( rightDistCoeffs ); rightDistCoeffs(0,i) += dEps;
        project( objPoints, rvec, tvec, cameraMatrix, rightDistCoeffs,
            rightImgPoints[i], valDpdrot, valDpdt, valDpdf, valDpdc, valDpddist, 0 );
    }
    calcdfdx( leftImgPoints, rightImgPoints, dEps, valDpddist );
    if( norm( dpddist, valDpddist ) > 0.3 )
    {
        ts->printf( cvtest::TS::LOG, "bad dpddist\n" );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

_exit_:
    RNG& _rng = ts->get_rng();
    _rng = rng;
    ts->set_failed_test_info( code );
}

//----------------------------------------- CV_ProjectPointsTest_C --------------------------------
class CV_ProjectPointsTest_C : public CV_ProjectPointsTest
{
public:
    CV_ProjectPointsTest_C() {}
protected:
    virtual void project( const Mat& objectPoints,
        const Mat& rvec, const Mat& tvec,
        const Mat& cameraMatrix,
        const Mat& distCoeffs,
        vector<Point2f>& imagePoints,
        Mat& dpdrot, Mat& dpdt, Mat& dpdf,
        Mat& dpdc, Mat& dpddist,
        double aspectRatio=0 );
};

void CV_ProjectPointsTest_C::project( const Mat& opoints, const Mat& rvec, const Mat& tvec,
                                       const Mat& cameraMatrix, const Mat& distCoeffs, vector<Point2f>& ipoints,
                                       Mat& dpdrot, Mat& dpdt, Mat& dpdf, Mat& dpdc, Mat& dpddist, double aspectRatio)
{
    int npoints = opoints.cols*opoints.rows*opoints.channels()/3;
    ipoints.resize(npoints);
    dpdrot.create(npoints*2, 3, CV_64F);
    dpdt.create(npoints*2, 3, CV_64F);
    dpdf.create(npoints*2, 2, CV_64F);
    dpdc.create(npoints*2, 2, CV_64F);
    dpddist.create(npoints*2, distCoeffs.rows + distCoeffs.cols - 1, CV_64F);
    CvMat _objectPoints = opoints, _imagePoints = Mat(ipoints);
    CvMat _rvec = rvec, _tvec = tvec, _cameraMatrix = cameraMatrix, _distCoeffs = distCoeffs;
    CvMat _dpdrot = dpdrot, _dpdt = dpdt, _dpdf = dpdf, _dpdc = dpdc, _dpddist = dpddist;

    cvProjectPoints2( &_objectPoints, &_rvec, &_tvec, &_cameraMatrix, &_distCoeffs,
                      &_imagePoints, &_dpdrot, &_dpdt, &_dpdf, &_dpdc, &_dpddist, aspectRatio );
}


//----------------------------------------- CV_ProjectPointsTest_CPP --------------------------------
class CV_ProjectPointsTest_CPP : public CV_ProjectPointsTest
{
public:
    CV_ProjectPointsTest_CPP() {}
protected:
    virtual void project( const Mat& objectPoints,
        const Mat& rvec, const Mat& tvec,
        const Mat& cameraMatrix,
        const Mat& distCoeffs,
        vector<Point2f>& imagePoints,
        Mat& dpdrot, Mat& dpdt, Mat& dpdf,
        Mat& dpdc, Mat& dpddist,
        double aspectRatio=0 );
};

void CV_ProjectPointsTest_CPP::project( const Mat& objectPoints, const Mat& rvec, const Mat& tvec,
                                       const Mat& cameraMatrix, const Mat& distCoeffs, vector<Point2f>& imagePoints,
                                       Mat& dpdrot, Mat& dpdt, Mat& dpdf, Mat& dpdc, Mat& dpddist, double aspectRatio)
{
    Mat J;
    projectPoints( objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints, J, aspectRatio);
    J.colRange(0, 3).copyTo(dpdrot);
    J.colRange(3, 6).copyTo(dpdt);
    J.colRange(6, 8).copyTo(dpdf);
    J.colRange(8, 10).copyTo(dpdc);
    J.colRange(10, J.cols).copyTo(dpddist);
}

///////////////////////////////// Stereo Calibration /////////////////////////////////////

class CV_StereoCalibrationTest : public cvtest::BaseTest
{
public:
    CV_StereoCalibrationTest();
    ~CV_StereoCalibrationTest();
    void clear();
protected:
    bool checkPandROI( int test_case_idx,
        const Mat& M, const Mat& D, const Mat& R,
        const Mat& P, Size imgsize, Rect roi );

    // covers of tested functions
    virtual double calibrateStereoCamera( const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints1,
        const vector<vector<Point2f> >& imagePoints2,
        Mat& cameraMatrix1, Mat& distCoeffs1,
        Mat& cameraMatrix2, Mat& distCoeffs2,
        Size imageSize, Mat& R, Mat& T,
        Mat& E, Mat& F, TermCriteria criteria, int flags ) = 0;
    virtual void rectify( const Mat& cameraMatrix1, const Mat& distCoeffs1,
        const Mat& cameraMatrix2, const Mat& distCoeffs2,
        Size imageSize, const Mat& R, const Mat& T,
        Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q,
        double alpha, Size newImageSize,
        Rect* validPixROI1, Rect* validPixROI2, int flags ) = 0;
    virtual bool rectifyUncalibrated( const Mat& points1,
        const Mat& points2, const Mat& F, Size imgSize,
        Mat& H1, Mat& H2, double threshold=5 ) = 0;
    virtual void triangulate( const Mat& P1, const Mat& P2,
        const Mat &points1, const Mat &points2,
        Mat &points4D ) = 0;
    virtual void correct( const Mat& F,
        const Mat &points1, const Mat &points2,
        Mat &newPoints1, Mat &newPoints2 ) = 0;

    void run(int);
};


CV_StereoCalibrationTest::CV_StereoCalibrationTest()
{
}


CV_StereoCalibrationTest::~CV_StereoCalibrationTest()
{
    clear();
}

void CV_StereoCalibrationTest::clear()
{
    cvtest::BaseTest::clear();
}

bool CV_StereoCalibrationTest::checkPandROI( int test_case_idx, const Mat& M, const Mat& D, const Mat& R,
                                            const Mat& P, Size imgsize, Rect roi )
{
    const double eps = 0.05;
    const int N = 21;
    int x, y, k;
    vector<Point2f> pts, upts;

    // step 1. check that all the original points belong to the destination image
    for( y = 0; y < N; y++ )
        for( x = 0; x < N; x++ )
            pts.push_back(Point2f((float)x*imgsize.width/(N-1), (float)y*imgsize.height/(N-1)));

    undistortPoints(Mat(pts), upts, M, D, R, P );
    for( k = 0; k < N*N; k++ )
        if( upts[k].x < -imgsize.width*eps || upts[k].x > imgsize.width*(1+eps) ||
            upts[k].y < -imgsize.height*eps || upts[k].y > imgsize.height*(1+eps) )
        {
            ts->printf(cvtest::TS::LOG, "Test #%d. The point (%g, %g) was mapped to (%g, %g) which is out of image\n",
                test_case_idx, pts[k].x, pts[k].y, upts[k].x, upts[k].y);
            return false;
        }

        // step 2. check that all the points inside ROI belong to the original source image
        Mat temp(imgsize, CV_8U), utemp, map1, map2;
        temp = Scalar::all(1);
        initUndistortRectifyMap(M, D, R, P, imgsize, CV_16SC2, map1, map2);
        remap(temp, utemp, map1, map2, INTER_LINEAR);

        if(roi.x < 0 || roi.y < 0 || roi.x + roi.width > imgsize.width || roi.y + roi.height > imgsize.height)
        {
            ts->printf(cvtest::TS::LOG, "Test #%d. The ROI=(%d, %d, %d, %d) is outside of the imge rectangle\n",
                test_case_idx, roi.x, roi.y, roi.width, roi.height);
            return false;
        }
        double s = sum(utemp(roi))[0];
        if( s > roi.area() || roi.area() - s > roi.area()*(1-eps) )
        {
            ts->printf(cvtest::TS::LOG, "Test #%d. The ratio of black pixels inside the valid ROI (~%g%%) is too large\n",
                test_case_idx, s*100./roi.area());
            return false;
        }

        return true;
}

void CV_StereoCalibrationTest::run( int )
{
    const int ntests = 1;
    const double maxReprojErr = 2;
    const double maxScanlineDistErr_c = 3;
    const double maxScanlineDistErr_uc = 4;
    FILE* f = 0;

    for(int testcase = 1; testcase <= ntests; testcase++)
    {
        cv::String filepath;
        char buf[1000];
        filepath = cv::format("%scv/stereo/case%d/stereo_calib.txt", ts->get_data_path().c_str(), testcase );
        f = fopen(filepath.c_str(), "rt");
        Size patternSize;
        vector<string> imglist;

        if( !f || !fgets(buf, sizeof(buf)-3, f) || sscanf(buf, "%d%d", &patternSize.width, &patternSize.height) != 2 )
        {
            ts->printf( cvtest::TS::LOG, "The file %s can not be opened or has invalid content\n", filepath.c_str() );
            ts->set_failed_test_info( f ? cvtest::TS::FAIL_INVALID_TEST_DATA : cvtest::TS::FAIL_MISSING_TEST_DATA );
            fclose(f);
            return;
        }

        for(;;)
        {
            if( !fgets( buf, sizeof(buf)-3, f ))
                break;
            size_t len = strlen(buf);
            while( len > 0 && isspace(buf[len-1]))
                buf[--len] = '\0';
            if( buf[0] == '#')
                continue;
            filepath = cv::format("%scv/stereo/case%d/%s", ts->get_data_path().c_str(), testcase, buf );
            imglist.push_back(string(filepath));
        }
        fclose(f);

        if( imglist.size() == 0 || imglist.size() % 2 != 0 )
        {
            ts->printf( cvtest::TS::LOG, "The number of images is 0 or an odd number in the case #%d\n", testcase );
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
            return;
        }

        int nframes = (int)(imglist.size()/2);
        int npoints = patternSize.width*patternSize.height;
        vector<vector<Point3f> > objpt(nframes);
        vector<vector<Point2f> > imgpt1(nframes);
        vector<vector<Point2f> > imgpt2(nframes);
        Size imgsize;
        int total = 0;

        for( int i = 0; i < nframes; i++ )
        {
            Mat left = imread(imglist[i*2]);
            Mat right = imread(imglist[i*2+1]);
            if(!left.data || !right.data)
            {
                ts->printf( cvtest::TS::LOG, "Can not load images %s and %s, testcase %d\n",
                    imglist[i*2].c_str(), imglist[i*2+1].c_str(), testcase );
                ts->set_failed_test_info( cvtest::TS::FAIL_MISSING_TEST_DATA );
                return;
            }
            imgsize = left.size();
            bool found1 = findChessboardCorners(left, patternSize, imgpt1[i]);
            bool found2 = findChessboardCorners(right, patternSize, imgpt2[i]);
            if(!found1 || !found2)
            {
                ts->printf( cvtest::TS::LOG, "The function could not detect boards on the images %s and %s, testcase %d\n",
                    imglist[i*2].c_str(), imglist[i*2+1].c_str(), testcase );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
            total += (int)imgpt1[i].size();
            for( int j = 0; j < npoints; j++ )
                objpt[i].push_back(Point3f((float)(j%patternSize.width), (float)(j/patternSize.width), 0.f));
        }

        // rectify (calibrated)
        Mat M1 = Mat::eye(3,3,CV_64F), M2 = Mat::eye(3,3,CV_64F), D1(5,1,CV_64F), D2(5,1,CV_64F), R, T, E, F;
        M1.at<double>(0,2) = M2.at<double>(0,2)=(imgsize.width-1)*0.5;
        M1.at<double>(1,2) = M2.at<double>(1,2)=(imgsize.height-1)*0.5;
        D1 = Scalar::all(0);
        D2 = Scalar::all(0);
        double err = calibrateStereoCamera(objpt, imgpt1, imgpt2, M1, D1, M2, D2, imgsize, R, T, E, F,
            TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 30, 1e-6),
            CV_CALIB_SAME_FOCAL_LENGTH
            //+ CV_CALIB_FIX_ASPECT_RATIO
            + CV_CALIB_FIX_PRINCIPAL_POINT
            + CV_CALIB_ZERO_TANGENT_DIST
            + CV_CALIB_FIX_K3
            + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5 //+ CV_CALIB_FIX_K6
            );
        err /= nframes*npoints;
        if( err > maxReprojErr )
        {
            ts->printf( cvtest::TS::LOG, "The average reprojection error is too big (=%g), testcase %d\n", err, testcase);
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
            return;
        }

        Mat R1, R2, P1, P2, Q;
        Rect roi1, roi2;
        rectify(M1, D1, M2, D2, imgsize, R, T, R1, R2, P1, P2, Q, 1, imgsize, &roi1, &roi2, 0);
        Mat eye33 = Mat::eye(3,3,CV_64F);
        Mat R1t = R1.t(), R2t = R2.t();

        if( norm(R1t*R1 - eye33) > 0.01 ||
            norm(R2t*R2 - eye33) > 0.01 ||
            abs(determinant(F)) > 0.01)
        {
            ts->printf( cvtest::TS::LOG, "The computed (by rectify) R1 and R2 are not orthogonal,"
                "or the computed (by calibrate) F is not singular, testcase %d\n", testcase);
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
            return;
        }

        if(!checkPandROI(testcase, M1, D1, R1, P1, imgsize, roi1))
        {
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            return;
        }

        if(!checkPandROI(testcase, M2, D2, R2, P2, imgsize, roi2))
        {
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            return;
        }

        //check that Tx after rectification is equal to distance between cameras
        double tx = fabs(P2.at<double>(0, 3) / P2.at<double>(0, 0));
        if (fabs(tx - norm(T)) > 1e-5)
        {
            ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            return;
        }

        //check that Q reprojects points before the camera
        double testPoint[4] = {0.0, 0.0, 100.0, 1.0};
        Mat reprojectedTestPoint = Q * Mat_<double>(4, 1, testPoint);
        CV_Assert(reprojectedTestPoint.type() == CV_64FC1);
        if( reprojectedTestPoint.at<double>(2) / reprojectedTestPoint.at<double>(3) < 0 )
        {
            ts->printf( cvtest::TS::LOG, "A point after rectification is reprojected behind the camera, testcase %d\n", testcase);
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }

        //check that Q reprojects the same points as reconstructed by triangulation
        const float minCoord = -300.0f;
        const float maxCoord = 300.0f;
        const float minDisparity = 0.1f;
        const float maxDisparity = 600.0f;
        const int pointsCount = 500;
        const float requiredAccuracy = 1e-3f;
        RNG& rng = ts->get_rng();

        Mat projectedPoints_1(2, pointsCount, CV_32FC1);
        Mat projectedPoints_2(2, pointsCount, CV_32FC1);
        Mat disparities(1, pointsCount, CV_32FC1);

        rng.fill(projectedPoints_1, RNG::UNIFORM, minCoord, maxCoord);
        rng.fill(disparities, RNG::UNIFORM, minDisparity, maxDisparity);
        projectedPoints_2.row(0) = projectedPoints_1.row(0) - disparities;
        Mat ys_2 = projectedPoints_2.row(1);
        projectedPoints_1.row(1).copyTo(ys_2);

        Mat points4d;
        triangulate(P1, P2, projectedPoints_1, projectedPoints_2, points4d);
        Mat homogeneousPoints4d = points4d.t();
        const int dimension = 4;
        homogeneousPoints4d = homogeneousPoints4d.reshape(dimension);
        Mat triangulatedPoints;
        convertPointsFromHomogeneous(homogeneousPoints4d, triangulatedPoints);

        Mat sparsePoints;
        sparsePoints.push_back(projectedPoints_1);
        sparsePoints.push_back(disparities);
        sparsePoints = sparsePoints.t();
        sparsePoints = sparsePoints.reshape(3);
        Mat reprojectedPoints;
        perspectiveTransform(sparsePoints, reprojectedPoints, Q);

        if (norm(triangulatedPoints - reprojectedPoints) / sqrt((double)pointsCount) > requiredAccuracy)
        {
            ts->printf( cvtest::TS::LOG, "Points reprojected with a matrix Q and points reconstructed by triangulation are different, testcase %d\n", testcase);
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }

        //check correctMatches
        const float constraintAccuracy = 1e-5f;
        Mat newPoints1, newPoints2;
        Mat points1 = projectedPoints_1.t();
        points1 = points1.reshape(2, 1);
        Mat points2 = projectedPoints_2.t();
        points2 = points2.reshape(2, 1);
        correctMatches(F, points1, points2, newPoints1, newPoints2);
        Mat newHomogeneousPoints1, newHomogeneousPoints2;
        convertPointsToHomogeneous(newPoints1, newHomogeneousPoints1);
        convertPointsToHomogeneous(newPoints2, newHomogeneousPoints2);
        newHomogeneousPoints1 = newHomogeneousPoints1.reshape(1);
        newHomogeneousPoints2 = newHomogeneousPoints2.reshape(1);
        Mat typedF;
        F.convertTo(typedF, newHomogeneousPoints1.type());
        for (int i = 0; i < newHomogeneousPoints1.rows; ++i)
        {
            Mat error = newHomogeneousPoints2.row(i) * typedF * newHomogeneousPoints1.row(i).t();
            CV_Assert(error.rows == 1 && error.cols == 1);
            if (norm(error) > constraintAccuracy)
            {
                ts->printf( cvtest::TS::LOG, "Epipolar constraint is violated after correctMatches, testcase %d\n", testcase);
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
            }
        }

        // rectifyUncalibrated
        CV_Assert( imgpt1.size() == imgpt2.size() );
        Mat _imgpt1( total, 1, CV_32FC2 ), _imgpt2( total, 1, CV_32FC2 );
        vector<vector<Point2f> >::const_iterator iit1 = imgpt1.begin();
        vector<vector<Point2f> >::const_iterator iit2 = imgpt2.begin();
        for( int pi = 0; iit1 != imgpt1.end(); ++iit1, ++iit2 )
        {
            vector<Point2f>::const_iterator pit1 = iit1->begin();
            vector<Point2f>::const_iterator pit2 = iit2->begin();
            CV_Assert( iit1->size() == iit2->size() );
            for( ; pit1 != iit1->end(); ++pit1, ++pit2, pi++ )
            {
                _imgpt1.at<Point2f>(pi,0) = Point2f( pit1->x, pit1->y );
                _imgpt2.at<Point2f>(pi,0) = Point2f( pit2->x, pit2->y );
            }
        }

        Mat _M1, _M2, _D1, _D2;
        vector<Mat> _R1, _R2, _T1, _T2;
        calibrateCamera( objpt, imgpt1, imgsize, _M1, _D1, _R1, _T1, 0 );
        calibrateCamera( objpt, imgpt2, imgsize, _M2, _D2, _R2, _T1, 0 );
        undistortPoints( _imgpt1, _imgpt1, _M1, _D1, Mat(), _M1 );
        undistortPoints( _imgpt2, _imgpt2, _M2, _D2, Mat(), _M2 );

        Mat matF, _H1, _H2;
        matF = findFundamentalMat( _imgpt1, _imgpt2 );
        rectifyUncalibrated( _imgpt1, _imgpt2, matF, imgsize, _H1, _H2 );

        Mat rectifPoints1, rectifPoints2;
        perspectiveTransform( _imgpt1, rectifPoints1, _H1 );
        perspectiveTransform( _imgpt2, rectifPoints2, _H2 );

        bool verticalStereo = abs(P2.at<double>(0,3)) < abs(P2.at<double>(1,3));
        double maxDiff_c = 0, maxDiff_uc = 0;
        for( int i = 0, k = 0; i < nframes; i++ )
        {
            vector<Point2f> temp[2];
            undistortPoints(Mat(imgpt1[i]), temp[0], M1, D1, R1, P1);
            undistortPoints(Mat(imgpt2[i]), temp[1], M2, D2, R2, P2);

            for( int j = 0; j < npoints; j++, k++ )
            {
                double diff_c = verticalStereo ? abs(temp[0][j].x - temp[1][j].x) : abs(temp[0][j].y - temp[1][j].y);
                Point2f d = rectifPoints1.at<Point2f>(k,0) - rectifPoints2.at<Point2f>(k,0);
                double diff_uc = verticalStereo ? abs(d.x) : abs(d.y);
                maxDiff_c = max(maxDiff_c, diff_c);
                maxDiff_uc = max(maxDiff_uc, diff_uc);
                if( maxDiff_c > maxScanlineDistErr_c )
                {
                    ts->printf( cvtest::TS::LOG, "The distance between %s coordinates is too big(=%g) (used calibrated stereo), testcase %d\n",
                        verticalStereo ? "x" : "y", diff_c, testcase);
                    ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
                    return;
                }
                if( maxDiff_uc > maxScanlineDistErr_uc )
                {
                    ts->printf( cvtest::TS::LOG, "The distance between %s coordinates is too big(=%g) (used uncalibrated stereo), testcase %d\n",
                        verticalStereo ? "x" : "y", diff_uc, testcase);
                    ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
                    return;
                }
            }
        }

        ts->printf( cvtest::TS::LOG, "Testcase %d. Max distance (calibrated) =%g\n"
            "Max distance (uncalibrated) =%g\n", testcase, maxDiff_c, maxDiff_uc );
    }
}

//-------------------------------- CV_StereoCalibrationTest_C ------------------------------

class CV_StereoCalibrationTest_C : public CV_StereoCalibrationTest
{
public:
    CV_StereoCalibrationTest_C() {}
protected:
    virtual double calibrateStereoCamera( const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints1,
        const vector<vector<Point2f> >& imagePoints2,
        Mat& cameraMatrix1, Mat& distCoeffs1,
        Mat& cameraMatrix2, Mat& distCoeffs2,
        Size imageSize, Mat& R, Mat& T,
        Mat& E, Mat& F, TermCriteria criteria, int flags );
    virtual void rectify( const Mat& cameraMatrix1, const Mat& distCoeffs1,
        const Mat& cameraMatrix2, const Mat& distCoeffs2,
        Size imageSize, const Mat& R, const Mat& T,
        Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q,
        double alpha, Size newImageSize,
        Rect* validPixROI1, Rect* validPixROI2, int flags );
    virtual bool rectifyUncalibrated( const Mat& points1,
        const Mat& points2, const Mat& F, Size imgSize,
        Mat& H1, Mat& H2, double threshold=5 );
    virtual void triangulate( const Mat& P1, const Mat& P2,
        const Mat &points1, const Mat &points2,
        Mat &points4D );
    virtual void correct( const Mat& F,
        const Mat &points1, const Mat &points2,
        Mat &newPoints1, Mat &newPoints2 );
};

double CV_StereoCalibrationTest_C::calibrateStereoCamera( const vector<vector<Point3f> >& objectPoints,
                 const vector<vector<Point2f> >& imagePoints1,
                 const vector<vector<Point2f> >& imagePoints2,
                 Mat& cameraMatrix1, Mat& distCoeffs1,
                 Mat& cameraMatrix2, Mat& distCoeffs2,
                 Size imageSize, Mat& R, Mat& T,
                 Mat& E, Mat& F, TermCriteria criteria, int flags )
{
    cameraMatrix1.create( 3, 3, CV_64F );
    cameraMatrix2.create( 3, 3, CV_64F);
    distCoeffs1.create( 1, 5, CV_64F);
    distCoeffs2.create( 1, 5, CV_64F);
    R.create(3, 3, CV_64F);
    T.create(3, 1, CV_64F);
    E.create(3, 3, CV_64F);
    F.create(3, 3, CV_64F);

    int  nimages = (int)objectPoints.size(), total = 0;
    for( int i = 0; i < nimages; i++ )
    {
        total += (int)objectPoints[i].size();
    }

    Mat npoints( 1, nimages, CV_32S ),
        objPt( 1, total, DataType<Point3f>::type ),
        imgPt( 1, total, DataType<Point2f>::type ),
        imgPt2( 1, total, DataType<Point2f>::type );

    Point2f* imgPtData2 = imgPt2.ptr<Point2f>();
    Point3f* objPtData = objPt.ptr<Point3f>();
    Point2f* imgPtData = imgPt.ptr<Point2f>();
    for( int i = 0, ni = 0, j = 0; i < nimages; i++, j += ni )
    {
        ni = (int)objectPoints[i].size();
        ((int*)npoints.data)[i] = ni;
        std::copy(objectPoints[i].begin(), objectPoints[i].end(), objPtData + j);
        std::copy(imagePoints1[i].begin(), imagePoints1[i].end(), imgPtData + j);
        std::copy(imagePoints2[i].begin(), imagePoints2[i].end(), imgPtData2 + j);
    }
    CvMat _objPt = objPt, _imgPt = imgPt, _imgPt2 = imgPt2, _npoints = npoints;
    CvMat _cameraMatrix1 = cameraMatrix1, _distCoeffs1 = distCoeffs1;
    CvMat _cameraMatrix2 = cameraMatrix2, _distCoeffs2 = distCoeffs2;
    CvMat matR = R, matT = T, matE = E, matF = F;

    return cvStereoCalibrate(&_objPt, &_imgPt, &_imgPt2, &_npoints, &_cameraMatrix1,
        &_distCoeffs1, &_cameraMatrix2, &_distCoeffs2, imageSize,
        &matR, &matT, &matE, &matF, flags, criteria );
}

void CV_StereoCalibrationTest_C::rectify( const Mat& cameraMatrix1, const Mat& distCoeffs1,
             const Mat& cameraMatrix2, const Mat& distCoeffs2,
             Size imageSize, const Mat& R, const Mat& T,
             Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q,
             double alpha, Size newImageSize,
             Rect* validPixROI1, Rect* validPixROI2, int flags )
{
    int rtype = CV_64F;
    R1.create(3, 3, rtype);
    R2.create(3, 3, rtype);
    P1.create(3, 4, rtype);
    P2.create(3, 4, rtype);
    Q.create(4, 4, rtype);
    CvMat _cameraMatrix1 = cameraMatrix1, _distCoeffs1 = distCoeffs1;
    CvMat _cameraMatrix2 = cameraMatrix2, _distCoeffs2 = distCoeffs2;
    CvMat matR = R, matT = T, _R1 = R1, _R2 = R2, _P1 = P1, _P2 = P2, matQ = Q;
    cvStereoRectify( &_cameraMatrix1, &_cameraMatrix2, &_distCoeffs1, &_distCoeffs2,
        imageSize, &matR, &matT, &_R1, &_R2, &_P1, &_P2, &matQ, flags,
        alpha, newImageSize, (CvRect*)validPixROI1, (CvRect*)validPixROI2);
}

bool CV_StereoCalibrationTest_C::rectifyUncalibrated( const Mat& points1,
           const Mat& points2, const Mat& F, Size imgSize, Mat& H1, Mat& H2, double threshold )
{
    H1.create(3, 3, CV_64F);
    H2.create(3, 3, CV_64F);
    CvMat _pt1 = points1, _pt2 = points2, matF, *pF=0, _H1 = H1, _H2 = H2;
    if( F.size() == Size(3, 3) )
        pF = &(matF = F);
    return cvStereoRectifyUncalibrated(&_pt1, &_pt2, pF, imgSize, &_H1, &_H2, threshold) > 0;
}

void CV_StereoCalibrationTest_C::triangulate( const Mat& P1, const Mat& P2,
        const Mat &points1, const Mat &points2,
        Mat &points4D )
{
    CvMat _P1 = P1, _P2 = P2, _points1 = points1, _points2 = points2;
    points4D.create(4, points1.cols, points1.type());
    CvMat _points4D = points4D;
    cvTriangulatePoints(&_P1, &_P2, &_points1, &_points2, &_points4D);
}

void CV_StereoCalibrationTest_C::correct( const Mat& F,
        const Mat &points1, const Mat &points2,
        Mat &newPoints1, Mat &newPoints2 )
{
    CvMat _F = F, _points1 = points1, _points2 = points2;
    newPoints1.create(1, points1.cols, points1.type());
    newPoints2.create(1, points2.cols, points2.type());
    CvMat _newPoints1 = newPoints1, _newPoints2 = newPoints2;
    cvCorrectMatches(&_F, &_points1, &_points2, &_newPoints1, &_newPoints2);
}

//-------------------------------- CV_StereoCalibrationTest_CPP ------------------------------

class CV_StereoCalibrationTest_CPP : public CV_StereoCalibrationTest
{
public:
    CV_StereoCalibrationTest_CPP() {}
protected:
    virtual double calibrateStereoCamera( const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints1,
        const vector<vector<Point2f> >& imagePoints2,
        Mat& cameraMatrix1, Mat& distCoeffs1,
        Mat& cameraMatrix2, Mat& distCoeffs2,
        Size imageSize, Mat& R, Mat& T,
        Mat& E, Mat& F, TermCriteria criteria, int flags );
    virtual void rectify( const Mat& cameraMatrix1, const Mat& distCoeffs1,
        const Mat& cameraMatrix2, const Mat& distCoeffs2,
        Size imageSize, const Mat& R, const Mat& T,
        Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q,
        double alpha, Size newImageSize,
        Rect* validPixROI1, Rect* validPixROI2, int flags );
    virtual bool rectifyUncalibrated( const Mat& points1,
        const Mat& points2, const Mat& F, Size imgSize,
        Mat& H1, Mat& H2, double threshold=5 );
    virtual void triangulate( const Mat& P1, const Mat& P2,
        const Mat &points1, const Mat &points2,
        Mat &points4D );
    virtual void correct( const Mat& F,
        const Mat &points1, const Mat &points2,
        Mat &newPoints1, Mat &newPoints2 );
};

double CV_StereoCalibrationTest_CPP::calibrateStereoCamera( const vector<vector<Point3f> >& objectPoints,
                                             const vector<vector<Point2f> >& imagePoints1,
                                             const vector<vector<Point2f> >& imagePoints2,
                                             Mat& cameraMatrix1, Mat& distCoeffs1,
                                             Mat& cameraMatrix2, Mat& distCoeffs2,
                                             Size imageSize, Mat& R, Mat& T,
                                             Mat& E, Mat& F, TermCriteria criteria, int flags )
{
    return stereoCalibrate( objectPoints, imagePoints1, imagePoints2,
                    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                    imageSize, R, T, E, F, flags, criteria );
}

void CV_StereoCalibrationTest_CPP::rectify( const Mat& cameraMatrix1, const Mat& distCoeffs1,
                                         const Mat& cameraMatrix2, const Mat& distCoeffs2,
                                         Size imageSize, const Mat& R, const Mat& T,
                                         Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q,
                                         double alpha, Size newImageSize,
                                         Rect* validPixROI1, Rect* validPixROI2, int flags )
{
    stereoRectify( cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                imageSize, R, T, R1, R2, P1, P2, Q, flags, alpha, newImageSize,validPixROI1, validPixROI2 );
}

bool CV_StereoCalibrationTest_CPP::rectifyUncalibrated( const Mat& points1,
                       const Mat& points2, const Mat& F, Size imgSize, Mat& H1, Mat& H2, double threshold )
{
    return stereoRectifyUncalibrated( points1, points2, F, imgSize, H1, H2, threshold );
}

void CV_StereoCalibrationTest_CPP::triangulate( const Mat& P1, const Mat& P2,
        const Mat &points1, const Mat &points2,
        Mat &points4D )
{
    triangulatePoints(P1, P2, points1, points2, points4D);
}

void CV_StereoCalibrationTest_CPP::correct( const Mat& F,
        const Mat &points1, const Mat &points2,
        Mat &newPoints1, Mat &newPoints2 )
{
    correctMatches(F, points1, points2, newPoints1, newPoints2);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Calib3d_CalibrateCamera_C, regression) { CV_CameraCalibrationTest_C test; test.safe_run(); }
TEST(Calib3d_CalibrateCamera_CPP, regression) { CV_CameraCalibrationTest_CPP test; test.safe_run(); }
TEST(Calib3d_CalibrationMatrixValues_C, accuracy) { CV_CalibrationMatrixValuesTest_C test; test.safe_run(); }
TEST(Calib3d_CalibrationMatrixValues_CPP, accuracy) { CV_CalibrationMatrixValuesTest_CPP test; test.safe_run(); }
TEST(Calib3d_ProjectPoints_C, accuracy) { CV_ProjectPointsTest_C  test; test.safe_run(); }
TEST(Calib3d_ProjectPoints_CPP, regression) { CV_ProjectPointsTest_CPP test; test.safe_run(); }
TEST(Calib3d_StereoCalibrate_C, regression) { CV_StereoCalibrationTest_C test; test.safe_run(); }
TEST(Calib3d_StereoCalibrate_CPP, regression) { CV_StereoCalibrationTest_CPP test; test.safe_run(); }
