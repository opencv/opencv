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
#include "opencv2/stereo.hpp"

namespace opencv_test { namespace {

// --------------------------------- CV_CameraCalibrationTest --------------------------------------------

typedef Matx33d RotMat;

class CV_CameraCalibrationTest : public cvtest::BaseTest
{
public:
    CV_CameraCalibrationTest();
    ~CV_CameraCalibrationTest();
    void clear();
protected:
    int compare(double* val, double* refVal, int len,
                double eps, const char* paramName);
    virtual void calibrate(Size imageSize,
        const std::vector<std::vector<Point2d> >& imagePoints,
        const std::vector<std::vector<Point3d> >& objectPoints,
        int iFixedPoint, Mat& distortionCoeffs, Mat& cameraMatrix, std::vector<Vec3d>& translationVectors,
        std::vector<RotMat>& rotationMatrices, std::vector<Point3d>& newObjPoints,
        std::vector<double>& stdDevs, std::vector<double>& perViewErrors,
        int flags ) = 0;
    virtual void project( const std::vector<Point3d>& objectPoints,
        const RotMat& rotationMatrix, const Vec3d& translationVector,
        const Mat& cameraMatrix, const Mat& distortion,
        std::vector<Point2d>& imagePoints ) = 0;

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

    std::vector<std::vector<Point2d> >  imagePoints;
    std::vector<std::vector<Point3d> >  objectPoints;
    std::vector<std::vector<Point2d> >  reprojectPoints;

    std::vector<Vec3d>        transVects;
    std::vector<RotMat>       rotMatrs;
    std::vector<Point3d>      newObjPoints;
    std::vector<double>       stdDevs;
    std::vector<double>       perViewErrors;

    std::vector<Vec3d>        goodTransVects;
    std::vector<RotMat>       goodRotMatrs;
    std::vector<Point3d>      goodObjPoints;
    std::vector<double>       goodPerViewErrors;
    std::vector<double>       goodStdDevs;

    Mat             cameraMatrix;
    Mat             distortion = Mat::zeros(1, 5, CV_64F);
    Mat             goodDistortion = Mat::zeros(1, 5, CV_64F);

    FILE*           file = 0;
    FILE*           datafile = 0;
    int             i,j;
    int             currImage;
    int             currPoint;
    char            i_dat_file[100];

    int progress = 0;
    int values_read = -1;

    filepath = cv::format("%scv/cameracalibration/", ts->get_data_path().c_str() );
    filename = cv::format("%sdatafiles.txt", filepath.c_str() );
    datafile = fopen( filename.c_str(), "r" );
    if( datafile == 0 )
    {
        ts->printf( cvtest::TS::LOG, "Could not open file with list of test files: %s\n", filename.c_str() );
        code = cvtest::TS::FAIL_MISSING_TEST_DATA;
        ts->set_failed_test_info( code );
        return;
    }

    int numTests = 0;
    values_read = fscanf(datafile,"%d",&numTests);
    CV_Assert(values_read == 1);

    for( int currTest = start_from; currTest < numTests; currTest++ )
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
                break;
            }
            continue; // if there is more than one test, just skip the test
        }

        Size imageSize;
        values_read = fscanf(file,"%d %d\n",&(imageSize.width),&(imageSize.height));
        CV_Assert(values_read == 2);
        if( imageSize.width <= 0 || imageSize.height <= 0 )
        {
            ts->printf( cvtest::TS::LOG, "Image size in test file is incorrect\n" );
            code = cvtest::TS::FAIL_INVALID_TEST_DATA;
            break;
        }

        /* Read etalon size */
        Size etalonSize;
        values_read = fscanf(file,"%d %d\n",&(etalonSize.width),&(etalonSize.height));
        CV_Assert(values_read == 2);
        if( etalonSize.width <= 0 || etalonSize.height <= 0 )
        {
            ts->printf( cvtest::TS::LOG, "Pattern size in test file is incorrect\n" );
            code = cvtest::TS::FAIL_INVALID_TEST_DATA;
            break;
        }

        int numPoints = etalonSize.width * etalonSize.height;

        /* Read number of images */
        int numImages = 0;
        values_read = fscanf(file,"%d\n",&numImages);
        CV_Assert(values_read == 1);
        if( numImages <=0 )
        {
            ts->printf( cvtest::TS::LOG, "Number of images in test file is incorrect\n");
            code = cvtest::TS::FAIL_INVALID_TEST_DATA;
            break;
        }

        /* Read calibration flags */
        int calibFlags = 0;
        values_read = fscanf(file,"%d\n",&calibFlags);
        CV_Assert(values_read == 1);

        /* Read index of the fixed point */
        int iFixedPoint;
        values_read = fscanf(file,"%d\n",&iFixedPoint);
        CV_Assert(values_read == 1);

        /* Need to allocate memory */
        imagePoints.resize(numImages);
        objectPoints.resize(numImages);
        reprojectPoints.resize(numImages);
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            imagePoints[currImage].resize(numPoints);
            objectPoints[currImage].resize(numPoints);
            reprojectPoints[currImage].resize(numPoints);
        }

        transVects.resize(numImages);
        rotMatrs.resize(numImages);
        newObjPoints.resize(numPoints);
        stdDevs.resize(CALIB_NINTRINSIC + 6*numImages + 3*numPoints);
        perViewErrors.resize(numImages);

        goodTransVects.resize(numImages);
        goodRotMatrs.resize(numImages);
        goodObjPoints.resize(numPoints);
        goodPerViewErrors.resize(numImages);

        int nstddev = CALIB_NINTRINSIC + 6*numImages + 3*numPoints;
        goodStdDevs.resize(nstddev);

        for( currImage = 0; currImage < numImages; currImage++ )
        {
            for( currPoint = 0; currPoint < numPoints; currPoint++ )
            {
                double x,y,z;
                values_read = fscanf(file,"%lf %lf %lf\n",&x,&y,&z);
                CV_Assert(values_read == 3);

                objectPoints[currImage][currPoint].x = x;
                objectPoints[currImage][currPoint].y = y;
                objectPoints[currImage][currPoint].z = z;
            }
        }

        /* Read image points */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            for( currPoint = 0; currPoint < numPoints; currPoint++ )
            {
                double x,y;
                values_read = fscanf(file,"%lf %lf\n",&x,&y);
                CV_Assert(values_read == 2);

                imagePoints[currImage][currPoint].x = x;
                imagePoints[currImage][currPoint].y = y;
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

        for( i = 0; i < 4; i++ )
        {
            values_read = fscanf(file,"%lf",&goodDistortion.at<double>(i)); CV_Assert(values_read == 1);
        }

        /* Read good Rot matrices */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            for( i = 0; i < 3; i++ )
                for( j = 0; j < 3; j++ )
                {
                    // Yes, load with transpose
                    values_read = fscanf(file, "%lf", &goodRotMatrs[currImage].val[j*3+i]);
                    CV_Assert(values_read == 1);
                }
        }

        /* Read good Trans vectors */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            for( i = 0; i < 3; i++ )
            {
                values_read = fscanf(file, "%lf", &goodTransVects[currImage].val[i]);
                CV_Assert(values_read == 1);
            }
        }

        bool releaseObject = iFixedPoint > 0 && iFixedPoint < numPoints - 1;
        /* Read good refined 3D object points */
        if( releaseObject )
        {
            for( i = 0; i < numPoints; i++ )
            {
                for( j = 0; j < 3; j++ )
                {
                    values_read = fscanf(file, "%lf", &goodObjPoints[i].x + j);
                    CV_Assert(values_read == 1);
                }
            }
        }

        /* Read good stdDeviations */
        for (i = 0; i < CALIB_NINTRINSIC + numImages*6; i++)
        {
            values_read = fscanf(file, "%lf", &goodStdDevs[i]);
            CV_Assert(values_read == 1);
        }
        for( ; i < nstddev; i++ )
        {
            if( releaseObject )
            {
                values_read = fscanf(file, "%lf", &goodStdDevs[i]);
                CV_Assert(values_read == 1);
            }
            else
                goodStdDevs[i] = 0.0;
        }

        cameraMatrix = Mat::zeros(3, 3, CV_64F);
        cameraMatrix.at<double>(0, 0) = cameraMatrix.at<double>(1, 1) = 807.;
        cameraMatrix.at<double>(0, 2) = (imageSize.width - 1)*0.5;
        cameraMatrix.at<double>(1, 2) = (imageSize.height - 1)*0.5;
        cameraMatrix.at<double>(2, 2) = 1.;

        /* Now we can calibrate camera */
        calibrate(  imageSize,
                    imagePoints,
                    objectPoints,
                    iFixedPoint,
                    distortion,
                    cameraMatrix,
                    transVects,
                    rotMatrs,
                    newObjPoints,
                    stdDevs,
                    perViewErrors,
                    calibFlags );

        /* ---- Reproject points to the image ---- */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            if( releaseObject )
            {
                objectPoints[currImage] = newObjPoints;
            }
            project(  objectPoints[currImage],
                      rotMatrs[currImage],
                      transVects[currImage],
                      cameraMatrix,
                      distortion,
                      reprojectPoints[currImage]);
        }

        /* ----- Compute reprojection error ----- */
        double dx,dy;
        double rx,ry;

        for( currImage = 0; currImage < numImages; currImage++ )
        {
            double imageMeanDx = 0;
            double imageMeanDy = 0;
            for( currPoint = 0; currPoint < etalonSize.width * etalonSize.height; currPoint++ )
            {
                rx = reprojectPoints[currImage][currPoint].x;
                ry = reprojectPoints[currImage][currPoint].y;
                dx = rx - imagePoints[currImage][currPoint].x;
                dy = ry - imagePoints[currImage][currPoint].y;

                imageMeanDx += dx*dx;
                imageMeanDy += dy*dy;
            }
            goodPerViewErrors[currImage] = sqrt( (imageMeanDx + imageMeanDy) /
                                           (etalonSize.width * etalonSize.height));

            //only for c-version of test (it does not provides evaluation of perViewErrors
            //and returns zeros)
            if(perViewErrors[currImage] == 0.0)
                perViewErrors[currImage] = goodPerViewErrors[currImage];
        }

        /* ========= Compare parameters ========= */
        CV_Assert(cameraMatrix.type() == CV_64F && cameraMatrix.size() == Size(3, 3));
        CV_Assert(distortion.type() == CV_64F);

        Size dsz = distortion.size();
        CV_Assert(dsz == Size(4, 1) || dsz == Size(1, 4) || dsz == Size(5, 1) || dsz == Size(1, 5));

        /*std::cout << "cameraMatrix: " << cameraMatrix << "\n";
        std::cout << "curr distCoeffs: " << distortion << "\n";
        std::cout << "good distCoeffs: " << goodDistortion << "\n";*/

        /* ----- Compare focal lengths ----- */
        code = compare(&cameraMatrix.at<double>(0, 0), &goodFcx, 1, 0.1, "fx");
        if( code < 0 )
            break;

        code = compare(&cameraMatrix.at<double>(1, 1),&goodFcy, 1, 0.1, "fy");
        if( code < 0 )
            break;

        /* ----- Compare principal points ----- */
        code = compare(&cameraMatrix.at<double>(0,2), &goodCx, 1, 0.1, "cx");
        if( code < 0 )
            break;

        code = compare(&cameraMatrix.at<double>(1,2), &goodCy, 1, 0.1, "cy");
        if( code < 0 )
            break;

        /* ----- Compare distortion ----- */
        code = compare(&distortion.at<double>(0), &goodDistortion.at<double>(0), 4, 0.1, "[k1,k2,p1,p2]");
        if( code < 0 )
            break;

        /* ----- Compare rot matrices ----- */
        CV_Assert(rotMatrs.size() == (size_t)numImages);
        CV_Assert(transVects.size() == (size_t)numImages);

        //code = compare(rotMatrs[0].val, goodRotMatrs[0].val, 9*numImages, 0.05, "rotation matrices");
        for( i = 0; i < numImages; i++ )
        {
            if( cv::norm(rotMatrs[i], goodRotMatrs[i], NORM_INF) > 0.05 )
            {
                printf("rot mats for frame #%d are very different\n", i);
                std::cout << "curr:\n" << rotMatrs[i] << std::endl;
                std::cout << "good:\n" << goodRotMatrs[i] << std::endl;

                code = TS::FAIL_BAD_ACCURACY;
                break;
            }
        }
        if( code < 0 )
            break;

        /* ----- Compare rot matrices ----- */
        code = compare(transVects[0].val, goodTransVects[0].val, 3*numImages, 0.1, "translation vectors");
        if( code < 0 )
            break;

        /* ----- Compare refined 3D object points ----- */
        if( releaseObject )
        {
            code = compare(&newObjPoints[0].x, &goodObjPoints[0].x, 3*numPoints, 0.1, "refined 3D object points");
            if( code < 0 )
                break;
        }

        /* ----- Compare per view re-projection errors ----- */
        CV_Assert(perViewErrors.size() == (size_t)numImages);
        code = compare(&perViewErrors[0], &goodPerViewErrors[0], numImages, 0.1, "per view errors vector");
        if( code < 0 )
            break;

        /* ----- Compare standard deviations of parameters ----- */
        if( stdDevs.size() < (size_t)nstddev )
            stdDevs.resize(nstddev);
        for ( i = 0; i < nstddev; i++)
        {
            if(stdDevs[i] == 0.0)
                stdDevs[i] = goodStdDevs[i];
        }
        code = compare(&stdDevs[0], &goodStdDevs[0], nstddev, .5,
                       "stdDevs vector");
        if( code < 0 )
            break;

        /*if( maxDx > 1.0 )
        {
            ts->printf( cvtest::TS::LOG,
                      "Error in reprojection maxDx=%f > 1.0\n",maxDx);
            code = cvtest::TS::FAIL_BAD_ACCURACY; break;
        }

        if( maxDy > 1.0 )
        {
            ts->printf( cvtest::TS::LOG,
                      "Error in reprojection maxDy=%f > 1.0\n",maxDy);
            code = cvtest::TS::FAIL_BAD_ACCURACY; break;
        }*/

        progress = update_progress( progress, currTest, numTests, 0 );

        fclose(file);
        file = 0;
    }

    if( file )
        fclose(file);

    if( datafile )
        fclose(datafile);

    if( code < 0 )
        ts->set_failed_test_info( code );
}

// --------------------------------- CV_CameraCalibrationTest_CPP --------------------------------------------

class CV_CameraCalibrationTest_CPP : public CV_CameraCalibrationTest
{
public:
    CV_CameraCalibrationTest_CPP(){}
protected:
    virtual void calibrate(Size imageSize,
                           const std::vector<std::vector<Point2d> >& imagePoints,
                           const std::vector<std::vector<Point3d> >& objectPoints,
                           int iFixedPoint, Mat& distortionCoeffs, Mat& cameraMatrix, std::vector<Vec3d>& translationVectors,
                           std::vector<RotMat>& rotationMatrices, std::vector<Point3d>& newObjPoints,
                           std::vector<double>& stdDevs, std::vector<double>& perViewErrors,
                           int flags );
    virtual void project( const std::vector<Point3d>& objectPoints,
                         const RotMat& rotationMatrix, const Vec3d& translationVector,
                         const Mat& cameraMatrix, const Mat& distortion,
                         std::vector<Point2d>& imagePoints );
};

void CV_CameraCalibrationTest_CPP::calibrate(Size imageSize,
    const std::vector<std::vector<Point2d> >& _imagePoints,
    const std::vector<std::vector<Point3d> >& _objectPoints,
    int iFixedPoint, Mat& _distCoeffs, Mat& _cameraMatrix, std::vector<Vec3d>& translationVectors,
    std::vector<RotMat>& rotationMatrices, std::vector<Point3d>& newObjPoints,
    std::vector<double>& stdDevs, std::vector<double>& perViewErrors,
    int flags )
{
    int pointCount = (int)_imagePoints[0].size();
    size_t i, imageCount = _imagePoints.size();
    vector<vector<Point3f> > objectPoints( imageCount );
    vector<vector<Point2f> > imagePoints( imageCount );
    Mat cameraMatrix, distCoeffs(1,4,CV_64F,Scalar::all(0));
    vector<Mat> rvecs, tvecs;
    Mat newObjMat;
    Mat stdDevsMatInt, stdDevsMatExt;
    Mat stdDevsMatObj;
    Mat perViewErrorsMat;

    for( i = 0; i < imageCount; i++ )
    {
        Mat(_imagePoints[i]).convertTo(imagePoints[i], CV_32F);
        Mat(_objectPoints[i]).convertTo(objectPoints[i], CV_32F);
    }

    size_t nstddev0 = CALIB_NINTRINSIC + imageCount*6, nstddev1 = nstddev0 + _imagePoints[0].size()*3;
    for( i = nstddev0; i < nstddev1; i++ )
    {
        stdDevs[i] = 0.0;
    }

    calibrateCameraRO( objectPoints,
                       imagePoints,
                       imageSize,
                       iFixedPoint,
                       cameraMatrix,
                       distCoeffs,
                       rvecs,
                       tvecs,
                       newObjMat,
                       stdDevsMatInt,
                       stdDevsMatExt,
                       stdDevsMatObj,
                       perViewErrorsMat,
                       flags );

    bool releaseObject = iFixedPoint > 0 && iFixedPoint < pointCount - 1;
    if( releaseObject )
    {
        newObjMat.convertTo( newObjPoints, CV_64F );
    }

    Mat stdDevMats[] = {stdDevsMatInt, stdDevsMatExt, stdDevsMatObj}, stdDevsMat;
    vconcat(stdDevMats, releaseObject ? 3 : 2, stdDevsMat);
    stdDevsMat.convertTo(stdDevs, CV_64F);

    perViewErrorsMat.convertTo(perViewErrors, CV_64F);
    cameraMatrix.convertTo(_cameraMatrix, CV_64F);
    distCoeffs.convertTo(_distCoeffs, CV_64F);

    for( i = 0; i < imageCount; i++ )
    {
        Mat r9;
        Rodrigues( rvecs[i], r9 );
        r9.convertTo(rotationMatrices[i], CV_64F);
        tvecs[i].convertTo(translationVectors[i], CV_64F);
    }
}


void CV_CameraCalibrationTest_CPP::project( const std::vector<Point3d>& objectPoints,
                         const RotMat& rotationMatrix, const Vec3d& translationVector,
                         const Mat& cameraMatrix, const Mat& distortion,
                         std::vector<Point2d>& imagePoints )
{
    projectPoints(objectPoints, rotationMatrix, translationVector, cameraMatrix, distortion, imagePoints );
    /*Mat objectPoints( pointCount, 3, CV_64FC1, _objectPoints );
    Mat rmat( 3, 3, CV_64FC1, rotationMatrix ),
        rvec( 1, 3, CV_64FC1 ),
        tvec( 1, 3, CV_64FC1, translationVector );
    Mat cameraMatrix( 3, 3, CV_64FC1, _cameraMatrix );
    Mat distCoeffs( 1, 4, CV_64FC1, distortion );
    vector<Point2f> imagePoints;
    cvtest::Rodrigues( rmat, rvec );

    objectPoints.convertTo( objectPoints, CV_32FC1 );
    projectPoints( objectPoints, rvec, tvec,
                   cameraMatrix, distCoeffs, imagePoints );
    vector<Point2f>::const_iterator it = imagePoints.begin();
    for( int i = 0; it != imagePoints.end(); ++it, i++ )
    {
        _imagePoints[i] = cvPoint2D64f( it->x, it->y );
    }*/
}

///////////////////////////////// Stereo Calibration /////////////////////////////////////

class CV_StereoCalibrationTest : public cvtest::BaseTest
{
public:
    CV_StereoCalibrationTest();
    ~CV_StereoCalibrationTest();
    void clear();
protected:
    // covers of tested functions
    virtual double calibrateStereoCamera( const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints1,
        const vector<vector<Point2f> >& imagePoints2,
        Mat& cameraMatrix1, Mat& distCoeffs1,
        Mat& cameraMatrix2, Mat& distCoeffs2,
        Size imageSize, Mat& R, Mat& T,
        Mat& E, Mat& F,
        std::vector<RotMat>& rotationMatrices, std::vector<Vec3d>& translationVectors,
        vector<double>& perViewErrors1, vector<double>& perViewErrors2,
        TermCriteria criteria, int flags ) = 0;
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
    int compare(double* val, double* refVal, int len,
                double eps, const char* paramName);

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

int CV_StereoCalibrationTest::compare(double* val, double* ref_val, int len,
                                      double eps, const char* param_name )
{
    return cvtest::cmpEps2_64f( ts, val, ref_val, len, eps, param_name );
}

void CV_StereoCalibrationTest::run( int )
{
    const int ntests = 1;
    const double maxReprojErr = 2;
    const double maxScanlineDistErr_c = 3;
    const double maxScanlineDistErr_uc = 4;
    const double maxDiffBtwRmsErrors = 1e-4;
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
            if (f)
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
            Mat left = imread(imglist[i*2], IMREAD_GRAYSCALE);
            Mat right = imread(imglist[i*2+1], IMREAD_GRAYSCALE);
            if(left.empty() || right.empty())
            {
                ts->printf( cvtest::TS::LOG, "Can not load images %s and %s, testcase %d\n",
                            imglist[i*2].c_str(), imglist[i*2+1].c_str(), testcase );
                ts->set_failed_test_info( cvtest::TS::FAIL_MISSING_TEST_DATA );
                return;
            }
            imgsize = left.size();
            bool found1 = findChessboardCorners(left, patternSize, imgpt1[i]);
            bool found2 = findChessboardCorners(right, patternSize, imgpt2[i]);
            cornerSubPix(left, imgpt1[i], Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
            cornerSubPix(right, imgpt2[i], Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
            if(!found1 || !found2)
            {
                ts->printf( cvtest::TS::LOG, "The function could not detect boards (%d x %d) on the images %s and %s, testcase %d\n",
                            patternSize.width, patternSize.height,
                            imglist[i*2].c_str(), imglist[i*2+1].c_str(), testcase );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
            total += (int)imgpt1[i].size();
            for( int j = 0; j < npoints; j++ )
                objpt[i].push_back(Point3f((float)(j%patternSize.width), (float)(j/patternSize.width), 0.f));
        }

        vector<RotMat> rotMats1(nframes);
        vector<Vec3d> transVecs1(nframes);
        vector<RotMat> rotMats2(nframes);
        vector<Vec3d> transVecs2(nframes);
        vector<double> rmsErrorPerView1(nframes);
        vector<double> rmsErrorPerView2(nframes);
        vector<double> rmsErrorPerViewFromReprojectedImgPts1(nframes);
        vector<double> rmsErrorPerViewFromReprojectedImgPts2(nframes);

        // rectify (calibrated)
        Mat M1 = Mat::eye(3,3,CV_64F), M2 = Mat::eye(3,3,CV_64F), D1(5,1,CV_64F), D2(5,1,CV_64F), R, T, E, F;
        M1.at<double>(0,2) = M2.at<double>(0,2)=(imgsize.width-1)*0.5;
        M1.at<double>(1,2) = M2.at<double>(1,2)=(imgsize.height-1)*0.5;
        D1 = Scalar::all(0);
        D2 = Scalar::all(0);
        double rmsErrorFromStereoCalib = calibrateStereoCamera(objpt, imgpt1, imgpt2, M1, D1, M2, D2, imgsize, R, T, E, F,
            rotMats1, transVecs1, rmsErrorPerView1, rmsErrorPerView2,
            TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 30, 1e-6),
            CALIB_SAME_FOCAL_LENGTH
            //+ CALIB_FIX_ASPECT_RATIO
            + CALIB_FIX_PRINCIPAL_POINT
            + CALIB_ZERO_TANGENT_DIST
            + CALIB_FIX_K3
            + CALIB_FIX_K4 + CALIB_FIX_K5 //+ CALIB_FIX_K6
            );

        /* rmsErrorFromStereoCalib /= nframes*npoints; */
        if (rmsErrorFromStereoCalib > maxReprojErr)
        {
            ts->printf(cvtest::TS::LOG, "The average reprojection error is too big (=%g), testcase %d\n",
                       rmsErrorFromStereoCalib, testcase);
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            return;
        }

        double rmsErrorFromReprojectedImgPts = 0.0f;
        if (rotMats1.empty() || transVecs1.empty())
        {
            rmsErrorPerViewFromReprojectedImgPts1 = rmsErrorPerView1;
            rmsErrorPerViewFromReprojectedImgPts2 = rmsErrorPerView2;
            rmsErrorFromReprojectedImgPts = rmsErrorFromStereoCalib;
        }
        else
        {
            size_t totalPoints = 0;
            double totalErr[2] = { 0, 0 };
            for (size_t i = 0; i < objpt.size(); ++i) {
                RotMat r1 = rotMats1[i];
                Vec3d t1 = transVecs1[i];

                RotMat r2 = Mat(R * r1);
                Mat T2t = R * t1;
                Vec3d t2 = Mat(T2t + T);

                vector<Point2f> reprojectedImgPts[2] = { vector<Point2f>(nframes),
                                                         vector<Point2f>(nframes) };
                projectPoints(objpt[i], r1, t1, M1, D1, reprojectedImgPts[0]);
                projectPoints(objpt[i], r2, t2, M2, D2, reprojectedImgPts[1]);

                double viewErr[2];
                viewErr[0] = cv::norm(imgpt1[i], reprojectedImgPts[0], cv::NORM_L2SQR);
                viewErr[1] = cv::norm(imgpt2[i], reprojectedImgPts[1], cv::NORM_L2SQR);

                size_t n = objpt[i].size();
                totalErr[0] += viewErr[0];
                totalErr[1] += viewErr[1];
                totalPoints += n;

                rmsErrorPerViewFromReprojectedImgPts1[i] = sqrt(viewErr[0] / n);
                rmsErrorPerViewFromReprojectedImgPts2[i] = sqrt(viewErr[1] / n);
            }
            rmsErrorFromReprojectedImgPts = std::sqrt((totalErr[0] + totalErr[1]) / (2 * totalPoints));
        }

        if (abs(rmsErrorFromStereoCalib - rmsErrorFromReprojectedImgPts) > maxDiffBtwRmsErrors)
        {
            ts->printf(cvtest::TS::LOG,
                       "The difference of the average reprojection error from the calibration function and from the "
                       "reprojected image points is too big (|%g - %g| = %g), testcase %d\n",
                       rmsErrorFromStereoCalib, rmsErrorFromReprojectedImgPts,
                       (rmsErrorFromStereoCalib - rmsErrorFromReprojectedImgPts), testcase);
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            return;
        }

        /* ----- Compare per view rms re-projection errors ----- */
        CV_Assert(rmsErrorPerView1.size() == (size_t)nframes);
        CV_Assert(rmsErrorPerViewFromReprojectedImgPts1.size() == (size_t)nframes);
        CV_Assert(rmsErrorPerView2.size() == (size_t)nframes);
        CV_Assert(rmsErrorPerViewFromReprojectedImgPts2.size() == (size_t)nframes);
        int code1 = compare(&rmsErrorPerView1[0], &rmsErrorPerViewFromReprojectedImgPts1[0], nframes,
                            maxDiffBtwRmsErrors, "per view errors vector");
        int code2 = compare(&rmsErrorPerView2[0], &rmsErrorPerViewFromReprojectedImgPts2[0], nframes,
                            maxDiffBtwRmsErrors, "per view errors vector");
        if (code1 < 0)
        {
            ts->printf(cvtest::TS::LOG,
                       "Some of the per view rms reprojection errors differ between calibration function and reprojected "
                       "points, for the first camera, testcase %d\n",
                       testcase);
            ts->set_failed_test_info(code1);
            return;
        }
        if (code2 < 0)
        {
            ts->printf(cvtest::TS::LOG,
                       "Some of the per view rms reprojection errors differ between calibration function and reprojected "
                       "points, for the second camera, testcase %d\n",
                       testcase);
            ts->set_failed_test_info(code2);
            return;
        }

        Mat R1, R2, P1, P2, Q;
        Rect roi1, roi2;
        rectify(M1, D1, M2, D2, imgsize, R, T, R1, R2, P1, P2, Q, 1, imgsize, &roi1, &roi2, 0);

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
        const float maxDisparity = 60.0f;
        const int pointsCount = 500;
        const float requiredAccuracy = 1e-3f;
        const float allowToFail = 0.2f; // 20%
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

        Mat diff;
        absdiff(triangulatedPoints, reprojectedPoints, diff);
        Mat mask = diff > requiredAccuracy;
        mask = mask.reshape(1);
        mask = mask.col(0) | mask.col(1) | mask.col(2);
        int numFailed = countNonZero(mask);
#if 0
        std::cout << "numFailed=" << numFailed << std::endl;
        for (int i = 0; i < triangulatedPoints.rows; i++)
        {
            if (mask.at<uchar>(i))
            {
                // failed points usually have 'w'~0 (points4d[3])
                std::cout << "i=" << i << " triangulatePoints=" << triangulatedPoints.row(i) << " reprojectedPoints=" << reprojectedPoints.row(i) << std::endl <<
                    "     points4d=" << points4d.col(i).t() << " projectedPoints_1=" << projectedPoints_1.col(i).t() << " disparities=" << disparities.col(i).t() << std::endl;
            }
        }
#endif

        if (numFailed >= allowToFail * pointsCount)
        {
            ts->printf( cvtest::TS::LOG, "Points reprojected with a matrix Q and points reconstructed by triangulation are different (tolerance=%g, failed=%d), testcase %d\n",
                requiredAccuracy, numFailed, testcase);
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
            if (cvtest::norm(error, NORM_L2) > constraintAccuracy)
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
        calibrateCamera( objpt, imgpt2, imgsize, _M2, _D2, _R2, _T2, 0 );
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
            undistortPoints(imgpt1[i], temp[0], M1, D1, R1, P1);
            undistortPoints(imgpt2[i], temp[1], M2, D2, R2, P2);

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
        Mat& E, Mat& F,
        std::vector<RotMat>& rotationMatrices, std::vector<Vec3d>& translationVectors,
        vector<double>& perViewErrors1, vector<double>& perViewErrors2,
        TermCriteria criteria, int flags );
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
                                                            Mat& E, Mat& F,
                                                            std::vector<RotMat>& rotationMatrices, std::vector<Vec3d>& translationVectors,
                                                            vector<double>& perViewErrors1, vector<double>& perViewErrors2,
                                                            TermCriteria criteria, int flags )
{
    vector<Mat> rvecs, tvecs;
    Mat perViewErrorsMat;
    double avgErr = stereoCalibrate( objectPoints, imagePoints1, imagePoints2,
                                     cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                                     imageSize, R, T, E, F,
                                     rvecs, tvecs, perViewErrorsMat,
                                     flags, criteria );

    size_t numImgs = imagePoints1.size();

    if (perViewErrors1.size() != numImgs)
    {
        perViewErrors1.resize(numImgs);
    }
    if (perViewErrors2.size() != numImgs)
    {
        perViewErrors2.resize(numImgs);
    }

    for (int i = 0; i < (int)numImgs; i++)
    {
        perViewErrors1[i] = perViewErrorsMat.at<double>(i, 0);
        perViewErrors2[i] = perViewErrorsMat.at<double>(i, 1);
    }

    if (rotationMatrices.size() != numImgs)
    {
        rotationMatrices.resize(numImgs);
    }
    if (translationVectors.size() != numImgs)
    {
        translationVectors.resize(numImgs);
    }

    for( size_t i = 0; i < numImgs; i++ )
    {
        Mat r9;
        cv::Rodrigues( rvecs[i], r9 );
        r9.convertTo(rotationMatrices[i], CV_64F);
        tvecs[i].convertTo(translationVectors[i], CV_64F);
    }
    return avgErr;
}

void CV_StereoCalibrationTest_CPP::rectify( const Mat& cameraMatrix1, const Mat& distCoeffs1,
                                         const Mat& cameraMatrix2, const Mat& distCoeffs2,
                                         Size imageSize, const Mat& R, const Mat& T,
                                         Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q,
                                         double alpha, Size newImageSize,
                                         Rect* validPixROI1, Rect* validPixROI2, int flags )
{
    stereoRectify( cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                imageSize, R, T, R1, R2, P1, P2, Q, flags, alpha, newImageSize, validPixROI1, validPixROI2 );
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

///////////////////////////////// Register Cameras ////////////////////////////////////////////////

class CV_CameraRegistrationTest : public cvtest::BaseTest
{
public:
    CV_CameraRegistrationTest();
    ~CV_CameraRegistrationTest();
    void clear();
protected:
    // covers of tested functions
    virtual double registerCameraPair( const vector<vector<Point3f> >& objectPoints1,
        const vector<vector<Point3f> >& objectPoints2,
        const vector<vector<Point2f> >& imagePoints1,
        const vector<vector<Point2f> >& imagePoints2,
        Mat& cameraMatrix1, Mat& distCoeffs1, CameraModel cameraModel1,
        Mat& cameraMatrix2, Mat& distCoeffs2, CameraModel cameraModel2,
        Mat& R, Mat& T,
        Mat& E, Mat& F,
        std::vector<RotMat>& rotationMatrices, std::vector<Vec3d>& translationVectors,
        vector<double>& perViewErrors1, vector<double>& perViewErrors2,
        TermCriteria criteria, int flags ) = 0;

    virtual double calibrateStereoCamera( const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints1,
        const vector<vector<Point2f> >& imagePoints2,
        Mat& cameraMatrix1, Mat& distCoeffs1,
        Mat& cameraMatrix2, Mat& distCoeffs2,
        Size imageSize, Mat& R, Mat& T,
        Mat& E, Mat& F,
        std::vector<RotMat>& rotationMatrices, std::vector<Vec3d>& translationVectors,
        vector<double>& perViewErrors1, vector<double>& perViewErrors2,
        TermCriteria criteria, int flags ) = 0;

    int compare(double* val, double* refVal, int len,
                double eps, const char* paramName);

    void run(int);
};


CV_CameraRegistrationTest::CV_CameraRegistrationTest()
{
}


CV_CameraRegistrationTest::~CV_CameraRegistrationTest()
{
    clear();
}

void CV_CameraRegistrationTest::clear()
{
    cvtest::BaseTest::clear();
}

int CV_CameraRegistrationTest::compare(double* val, double* ref_val, int len,
                                      double eps, const char* param_name )
{
    return cvtest::cmpEps2_64f( ts, val, ref_val, len, eps, param_name );
}

void CV_CameraRegistrationTest::run( int )
{
    const int ntests = 1;
    const double maxReprojErr = 2;
    const double maxDiffBtwRmsErrors = 1e-4;
    const double maxDiffBtwEstErrors = 1e-10;
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
            if (f)
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

        for( int i = 0; i < nframes; i++ )
        {
            Mat left = imread(imglist[i*2]);
            Mat right = imread(imglist[i*2+1]);
            if(left.empty() || right.empty())
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
                ts->printf( cvtest::TS::LOG, "The function could not detect boards (%d x %d) on the images %s and %s, testcase %d\n",
                            patternSize.width, patternSize.height,
                            imglist[i*2].c_str(), imglist[i*2+1].c_str(), testcase );
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
            for( int j = 0; j < npoints; j++ )
                objpt[i].push_back(Point3f((float)(j%patternSize.width), (float)(j/patternSize.width), 0.f));
        }

        vector<RotMat> rotMats1(nframes);
        vector<Vec3d> transVecs1(nframes);
        vector<RotMat> rotMats2(nframes);
        vector<Vec3d> transVecs2(nframes);
        vector<double> rmsErrorPerView1(nframes);
        vector<double> rmsErrorPerView2(nframes);
        vector<double> rmsErrorPerViewFromReprojectedImgPts1(nframes);
        vector<double> rmsErrorPerViewFromReprojectedImgPts2(nframes);

        Mat M1 = Mat::eye(3,3,CV_64F), M2 = Mat::eye(3,3,CV_64F), D1(5,1,CV_64F), D2(5,1,CV_64F);
        Mat R_stereo, T_stereo, R_register, T_register;
        Mat E_stereo, F_stereo, E_register, F_register;
        M1.at<double>(0,2) = M2.at<double>(0,2)=(imgsize.width-1)*0.5;
        M1.at<double>(1,2) = M2.at<double>(1,2)=(imgsize.height-1)*0.5;
        D1 = Scalar::all(0);
        D2 = Scalar::all(0
        );
        // Initialize the intrinsics
        calibrateStereoCamera(objpt, imgpt1, imgpt2, M1, D1, M2, D2, imgsize, R_stereo, T_stereo, E_stereo, F_stereo,
            rotMats1, transVecs1, rmsErrorPerView1, rmsErrorPerView2,
            TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 30, 1e-6),
            CALIB_SAME_FOCAL_LENGTH
            //+ CV_CALIB_FIX_ASPECT_RATIO
            + CALIB_FIX_PRINCIPAL_POINT
            + CALIB_ZERO_TANGENT_DIST
            + CALIB_FIX_K3
            + CALIB_FIX_K4 + CALIB_FIX_K5 //+ CV_CALIB_FIX_K6
            );

        // Use the fixed intrinsics to esimtate with two different methods
        double rmsErrorFromStereoCalibStereo = calibrateStereoCamera(objpt, imgpt1, imgpt2, M1, D1, M2, D2, imgsize, R_stereo, T_stereo, E_stereo, F_stereo,
            rotMats1, transVecs1, rmsErrorPerView1, rmsErrorPerView2,
            TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 30, 1e-6),
            CALIB_FIX_INTRINSIC
            );

        double rmsErrorFromStereoCalibRegister = registerCameraPair(objpt, objpt, imgpt1, imgpt2, M1, D1, CALIB_MODEL_PINHOLE, M2, D2, CALIB_MODEL_PINHOLE, R_register, T_register, E_register, F_register,
            rotMats2, transVecs2, rmsErrorPerView1, rmsErrorPerView2,
            TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 30, 1e-6),
            0
            );


        /* rmsErrorFromStereoCalibRegister /= nframes*npoints; */
        if (rmsErrorFromStereoCalibRegister > maxReprojErr)
        {
            ts->printf(cvtest::TS::LOG, "The average reprojection error is too big (=%g), testcase %d\n",
                       rmsErrorFromStereoCalibRegister, testcase);
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            return;
        }

        double rmsErrorFromReprojectedImgPts = 0.0f;
        if (rotMats1.empty() || transVecs1.empty())
        {
            rmsErrorPerViewFromReprojectedImgPts1 = rmsErrorPerView1;
            rmsErrorPerViewFromReprojectedImgPts2 = rmsErrorPerView2;
            rmsErrorFromReprojectedImgPts = rmsErrorFromStereoCalibRegister;
        }
        else
        {
            size_t totalPoints = 0;
            double totalErr[2] = { 0, 0 };
            for (size_t i = 0; i < objpt.size(); ++i) {
                RotMat r1 = rotMats1[i];
                Vec3d t1 = transVecs1[i];

                RotMat r2 = Mat(R_register * r1);
                Mat T2t = R_register * t1;
                Vec3d t2 = Mat(T2t + T_register);

                vector<Point2f> reprojectedImgPts[2] = { vector<Point2f>(nframes),
                                                         vector<Point2f>(nframes) };
                projectPoints(objpt[i], r1, t1, M1, D1, reprojectedImgPts[0]);
                projectPoints(objpt[i], r2, t2, M2, D2, reprojectedImgPts[1]);

                double viewErr[2];
                viewErr[0] = cv::norm(imgpt1[i], reprojectedImgPts[0], cv::NORM_L2SQR);
                viewErr[1] = cv::norm(imgpt2[i], reprojectedImgPts[1], cv::NORM_L2SQR);

                size_t n = objpt[i].size();
                totalErr[0] += viewErr[0];
                totalErr[1] += viewErr[1];
                totalPoints += n;

                rmsErrorPerViewFromReprojectedImgPts1[i] = sqrt(viewErr[0] / n);
                rmsErrorPerViewFromReprojectedImgPts2[i] = sqrt(viewErr[1] / n);
            }
            rmsErrorFromReprojectedImgPts = std::sqrt((totalErr[0] + totalErr[1]) / (2 * totalPoints));
        }

        if (abs(rmsErrorFromStereoCalibRegister - rmsErrorFromReprojectedImgPts) > maxDiffBtwRmsErrors)
        {
            ts->printf(cvtest::TS::LOG,
                       "The difference of the average reprojection error from the calibration function and from the "
                       "reprojected image points is too big (|%g - %g| = %g), testcase %d\n",
                       rmsErrorFromStereoCalibRegister, rmsErrorFromReprojectedImgPts,
                       (rmsErrorFromStereoCalibRegister - rmsErrorFromReprojectedImgPts), testcase);
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            return;
        }

        /* ----- Compare per view rms re-projection errors ----- */
        CV_Assert(rmsErrorPerView1.size() == (size_t)nframes);
        CV_Assert(rmsErrorPerViewFromReprojectedImgPts1.size() == (size_t)nframes);
        CV_Assert(rmsErrorPerView2.size() == (size_t)nframes);
        CV_Assert(rmsErrorPerViewFromReprojectedImgPts2.size() == (size_t)nframes);
        int code1 = compare(&rmsErrorPerView1[0], &rmsErrorPerViewFromReprojectedImgPts1[0], nframes,
                            maxDiffBtwRmsErrors, "per view errors vector");
        int code2 = compare(&rmsErrorPerView2[0], &rmsErrorPerViewFromReprojectedImgPts2[0], nframes,
                            maxDiffBtwRmsErrors, "per view errors vector");
        if (code1 < 0)
        {
            ts->printf(cvtest::TS::LOG,
                       "Some of the per view rms reprojection errors differ between calibration function and reprojected "
                       "points, for the first camera, testcase %d\n",
                       testcase);
            ts->set_failed_test_info(code1);
            return;
        }
        if (code2 < 0)
        {
            ts->printf(cvtest::TS::LOG,
                       "Some of the per view rms reprojection errors differ between calibration function and reprojected "
                       "points, for the second camera, testcase %d\n",
                       testcase);
            ts->set_failed_test_info(code2);
            return;
        }

        /* ----- compare the result from stereoCalibrate and registerCameras ----- */
        if (abs(rmsErrorFromStereoCalibStereo - rmsErrorFromStereoCalibRegister) > maxDiffBtwRmsErrors)
        {
            ts->printf(cvtest::TS::LOG,
                       "The difference of the average reprojection error from the register camera and stero calibration is too large (|%g - %g| = %g), testcase %d\n",
                       rmsErrorFromStereoCalibStereo, rmsErrorFromStereoCalibRegister,
                       (rmsErrorFromStereoCalibStereo - rmsErrorFromStereoCalibRegister), testcase);
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            return;
        }

        int code3 = compare(&R_stereo.at<double>(0), &R_register.at<double>(0), 9, maxDiffBtwEstErrors, "R vector");
        int code4 = compare(&T_stereo.at<double>(0), &T_register.at<double>(0), 3, maxDiffBtwEstErrors, "T vector");
        int code5 = compare(&E_stereo.at<double>(0), &E_register.at<double>(0), 9, maxDiffBtwEstErrors, "E vector");
        int code6 = compare(&F_stereo.at<double>(0), &F_register.at<double>(0), 9, maxDiffBtwEstErrors, "F vector");

        if (code3 < 0)
        {
            ts->printf(cvtest::TS::LOG,
                       "The estimated R does not match, testcase %d\n",
                       testcase);
            ts->set_failed_test_info(code3);
            return;
        }
        if (code4 < 0)
        {
            ts->printf(cvtest::TS::LOG,
                       "The estimated T does not match, testcase %d\n",
                       testcase);
            ts->set_failed_test_info(code4);
            return;
        }
        if (code5 < 0)
        {
            ts->printf(cvtest::TS::LOG,
                       "The estimated E does not match, testcase %d\n",
                       testcase);
            ts->set_failed_test_info(code5);
            return;
        }
        if (code6 < 0)
        {
            ts->printf(cvtest::TS::LOG,
                       "The estimated F does not match, testcase %d\n",
                       testcase);
            ts->set_failed_test_info(code6);
            return;
        }
    }
}

//-------------------------------- CV_CameraRegistrationTest_CPP ------------------------------

class CV_CameraRegistrationTest_CPP : public CV_CameraRegistrationTest
{
public:
    CV_CameraRegistrationTest_CPP() {}
protected:
    virtual double registerCameraPair( const vector<vector<Point3f> >& objectPoints1,
        const vector<vector<Point3f> >& objectPoints2,
        const vector<vector<Point2f> >& imagePoints1,
        const vector<vector<Point2f> >& imagePoints2,
        Mat& cameraMatrix1, Mat& distCoeffs1, CameraModel cameraModel1,
        Mat& cameraMatrix2, Mat& distCoeffs2, CameraModel cameraModel2,
        Mat& R, Mat& T,
        Mat& E, Mat& F,
        std::vector<RotMat>& rotationMatrices, std::vector<Vec3d>& translationVectors,
        vector<double>& perViewErrors1, vector<double>& perViewErrors2,
        TermCriteria criteria, int flags );

    virtual double calibrateStereoCamera( const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints1,
        const vector<vector<Point2f> >& imagePoints2,
        Mat& cameraMatrix1, Mat& distCoeffs1,
        Mat& cameraMatrix2, Mat& distCoeffs2,
        Size imageSize, Mat& R, Mat& T,
        Mat& E, Mat& F,
        std::vector<RotMat>& rotationMatrices, std::vector<Vec3d>& translationVectors,
        vector<double>& perViewErrors1, vector<double>& perViewErrors2,
        TermCriteria criteria, int flags );
};

double CV_CameraRegistrationTest_CPP::registerCameraPair( const vector<vector<Point3f> >& objectPoints1,
        const vector<vector<Point3f> >& objectPoints2,
        const vector<vector<Point2f> >& imagePoints1,
        const vector<vector<Point2f> >& imagePoints2,
        Mat& cameraMatrix1, Mat& distCoeffs1, CameraModel cameraModel1,
        Mat& cameraMatrix2, Mat& distCoeffs2, CameraModel cameraModel2,
        Mat& R, Mat& T,
        Mat& E, Mat& F,
        std::vector<RotMat>& rotationMatrices, std::vector<Vec3d>& translationVectors,
        vector<double>& perViewErrors1, vector<double>& perViewErrors2,
        TermCriteria criteria, int flags )
{

    vector<Mat> rvecs, tvecs;
    Mat perViewErrorsMat;
    double avgErr = registerCameras(objectPoints1, objectPoints2, imagePoints1, imagePoints2,
                                    cameraMatrix1, distCoeffs1, cameraModel1,
                                    cameraMatrix2, distCoeffs2, cameraModel2,
                                    R, T, E, F,
                                    rvecs, tvecs, perViewErrorsMat,
                                    flags, criteria);

    size_t numImgs = imagePoints1.size();

    if (perViewErrors1.size() != numImgs)
    {
        perViewErrors1.resize(numImgs);
    }
    if (perViewErrors2.size() != numImgs)
    {
        perViewErrors2.resize(numImgs);
    }

    for (int i = 0; i < (int)numImgs; i++)
    {
        perViewErrors1[i] = perViewErrorsMat.at<double>(i, 0);
        perViewErrors2[i] = perViewErrorsMat.at<double>(i, 1);
    }

    if (rotationMatrices.size() != numImgs)
    {
        rotationMatrices.resize(numImgs);
    }
    if (translationVectors.size() != numImgs)
    {
        translationVectors.resize(numImgs);
    }

    for( size_t i = 0; i < numImgs; i++ )
    {
        Mat r9;
        cv::Rodrigues( rvecs[i], r9 );
        r9.convertTo(rotationMatrices[i], CV_64F);
        tvecs[i].convertTo(translationVectors[i], CV_64F);
    }
    return avgErr;
}

double CV_CameraRegistrationTest_CPP::calibrateStereoCamera( const vector<vector<Point3f> >& objectPoints,
                                                            const vector<vector<Point2f> >& imagePoints1,
                                                            const vector<vector<Point2f> >& imagePoints2,
                                                            Mat& cameraMatrix1, Mat& distCoeffs1,
                                                            Mat& cameraMatrix2, Mat& distCoeffs2,
                                                            Size imageSize, Mat& R, Mat& T,
                                                            Mat& E, Mat& F,
                                                            std::vector<RotMat>& rotationMatrices, std::vector<Vec3d>& translationVectors,
                                                            vector<double>& perViewErrors1, vector<double>& perViewErrors2,
                                                            TermCriteria criteria, int flags )
{
    vector<Mat> rvecs, tvecs;
    Mat perViewErrorsMat;
    double avgErr = stereoCalibrate( objectPoints, imagePoints1, imagePoints2,
                                     cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                                     imageSize, R, T, E, F,
                                     rvecs, tvecs, perViewErrorsMat,
                                     flags, criteria );

    size_t numImgs = imagePoints1.size();

    if (perViewErrors1.size() != numImgs)
    {
        perViewErrors1.resize(numImgs);
    }
    if (perViewErrors2.size() != numImgs)
    {
        perViewErrors2.resize(numImgs);
    }

    for (int i = 0; i < (int)numImgs; i++)
    {
        perViewErrors1[i] = perViewErrorsMat.at<double>(i, 0);
        perViewErrors2[i] = perViewErrorsMat.at<double>(i, 1);
    }

    if (rotationMatrices.size() != numImgs)
    {
        rotationMatrices.resize(numImgs);
    }
    if (translationVectors.size() != numImgs)
    {
        translationVectors.resize(numImgs);
    }

    for( size_t i = 0; i < numImgs; i++ )
    {
        Mat r9;
        cv::Rodrigues( rvecs[i], r9 );
        r9.convertTo(rotationMatrices[i], CV_64F);
        tvecs[i].convertTo(translationVectors[i], CV_64F);
    }
    return avgErr;
}

TEST(Calib3d_CameraRegistration_CPP, regression) { CV_CameraRegistrationTest_CPP test; test.safe_run(); }
TEST(Calib3d_CalibrateCamera_CPP, regression) { CV_CameraCalibrationTest_CPP test; test.safe_run(); }
TEST(Calib3d_StereoCalibrate_CPP, regression) { CV_StereoCalibrationTest_CPP test; test.safe_run(); }


//-------------------------------- CV_MultiviewCalibrationTest_CPP ------------------------------

class CV_MultiviewCalibrationTest_CPP : public CV_StereoCalibrationTest
{
public:
    CV_MultiviewCalibrationTest_CPP() {}
protected:
    virtual double calibrateStereoCamera( const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints1,
        const vector<vector<Point2f> >& imagePoints2,
        Mat& cameraMatrix1, Mat& distCoeffs1,
        Mat& cameraMatrix2, Mat& distCoeffs2,
        Size imageSize, Mat& R, Mat& T,
        Mat& E, Mat& F,
        std::vector<RotMat>& rotationMatrices, std::vector<Vec3d>& translationVectors,
        vector<double>& perViewErrors1, vector<double>& perViewErrors2,
        TermCriteria criteria, int flags );
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

double CV_MultiviewCalibrationTest_CPP::calibrateStereoCamera( const vector<vector<Point3f> >& objectPoints,
                                             const vector<vector<Point2f> >& imagePoints1,
                                             const vector<vector<Point2f> >& imagePoints2,
                                             Mat& cameraMatrix1, Mat& distCoeffs1,
                                             Mat& cameraMatrix2, Mat& distCoeffs2,
                                             Size imageSize, Mat& R, Mat& T,
                                             Mat& E, Mat& F,
                                             std::vector<RotMat>& rotationMatrices, std::vector<Vec3d>& translationVectors,
                                             vector<double>& perViewErrors1, vector<double>& perViewErrors2,
                                             TermCriteria /*criteria*/, int flags )
{
    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<cv::Mat> Ks, distortions, Rs, Ts;
    cv::Mat errors_mat, output_pairs, rvecs0, tvecs0;
    int numImgs = (int)objectPoints.size();
    std::vector<std::vector<cv::Mat>> image_points_all(2, std::vector<Mat>(numImgs));
    for (int i = 0; i < numImgs; i++)
    {
        Mat img_pts1 = Mat(imagePoints1[i], false);
        Mat img_pts2 = Mat(imagePoints2[i], false);
        img_pts1.copyTo(image_points_all[0][i]);
        img_pts2.copyTo(image_points_all[1][i]);
    }
    std::vector<Size> image_sizes (2, imageSize);
    Mat visibility_mat = Mat_<uchar>::ones(2, numImgs);
    std::vector<uchar> models(2, cv::CALIB_MODEL_PINHOLE);
    std::vector<int> all_flags(2, flags);
    double rms = calibrateMultiview(objectPoints, image_points_all, image_sizes, visibility_mat, models,
                                    Ks, distortions, Rs, Ts, /*pairs*/ noArray(), rvecs, tvecs, errors_mat, all_flags);

    if (perViewErrors1.size() != (size_t)numImgs)
    {
        perViewErrors1.resize(numImgs);
    }
    if (perViewErrors2.size() != (size_t)numImgs)
    {
        perViewErrors2.resize(numImgs);
    }

    for (int i = 0; i < numImgs; i++)
    {
        perViewErrors1[i] = errors_mat.at<double>(0, i);
        perViewErrors2[i] = errors_mat.at<double>(1, i);
    }

    if (rotationMatrices.size() != (size_t)numImgs)
    {
        rotationMatrices.resize(numImgs);
    }
    if (translationVectors.size() != (size_t)numImgs)
    {
        translationVectors.resize(numImgs);
    }

    for( int i = 0; i < numImgs; i++ )
    {
        Mat r9;
        cv::Rodrigues( rvecs[i], r9 );
        r9.convertTo(rotationMatrices[i], CV_64F);
        tvecs[i].convertTo(translationVectors[i], CV_64F);
    }

    cv::Rodrigues(Rs[1], R);
    Ts[1].copyTo(T);
    distortions[0].copyTo(distCoeffs1);
    distortions[1].copyTo(distCoeffs2);
    Ks[0].copyTo(cameraMatrix1);
    Ks[1].copyTo(cameraMatrix2);
    Matx33d skewT(               0, -T.at<double>(2),   T.at<double>(1),
                   T.at<double>(2),                0,  -T.at<double>(0),
                  -T.at<double>(1),  T.at<double>(0),                 0);
    E = Mat(skewT * R);
    F = Ks[1].inv().t() * E * Ks[0].inv();
    return rms;
}

void CV_MultiviewCalibrationTest_CPP::rectify( const Mat& cameraMatrix1, const Mat& distCoeffs1,
                                         const Mat& cameraMatrix2, const Mat& distCoeffs2,
                                         Size imageSize, const Mat& R, const Mat& T,
                                         Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q,
                                         double alpha, Size newImageSize,
                                         Rect* validPixROI1, Rect* validPixROI2, int flags )
{
    stereoRectify( cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                imageSize, R, T, R1, R2, P1, P2, Q, flags, alpha, newImageSize,validPixROI1, validPixROI2 );
}

bool CV_MultiviewCalibrationTest_CPP::rectifyUncalibrated( const Mat& points1,
                       const Mat& points2, const Mat& F, Size imgSize, Mat& H1, Mat& H2, double threshold )
{
    return stereoRectifyUncalibrated( points1, points2, F, imgSize, H1, H2, threshold );
}

void CV_MultiviewCalibrationTest_CPP::triangulate( const Mat& P1, const Mat& P2,
        const Mat &points1, const Mat &points2,
        Mat &points4D )
{
    triangulatePoints(P1, P2, points1, points2, points4D);
}

void CV_MultiviewCalibrationTest_CPP::correct( const Mat& F,
        const Mat &points1, const Mat &points2,
        Mat &newPoints1, Mat &newPoints2 )
{
    correctMatches(F, points1, points2, newPoints1, newPoints2);
}

TEST(Calib3d_MultiviewCalibrate_CPP, regression) { CV_MultiviewCalibrationTest_CPP test; test.safe_run(); }

///////////////////////////////////////////////////////////////////////////////////////////////////


TEST(Calib3d_StereoCalibrate_CPP, extended)
{
    cvtest::TS* ts = cvtest::TS::ptr();
    String filepath = cv::format("%scv/stereo/case%d/", ts->get_data_path().c_str(), 1 );

    Mat left = imread(filepath+"left01.png");
    Mat right = imread(filepath+"right01.png");
    if(left.empty() || right.empty())
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_MISSING_TEST_DATA );
        return;
    }

    vector<vector<Point2f> > imgpt1(1), imgpt2(1);
    vector<vector<Point3f> > objpt(1);
    Size patternSize(9, 6), imageSize(640, 480);
    bool found1 = findChessboardCorners(left, patternSize, imgpt1[0]);
    bool found2 = findChessboardCorners(right, patternSize, imgpt2[0]);

    if(!found1 || !found2)
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        return;
    }

    for( int j = 0; j < patternSize.width*patternSize.height; j++ )
        objpt[0].push_back(Point3f((float)(j%patternSize.width), (float)(j/patternSize.width), 0.f));

    Mat K1, K2, c1, c2, R, T, E, F, err;
    int flags = 0;
    double res0 = stereoCalibrate( objpt, imgpt1, imgpt2,
                    K1, c1, K2, c2,
                    imageSize, R, T, E, F, err, flags);

    flags = CALIB_USE_EXTRINSIC_GUESS;
    double res1 = stereoCalibrate( objpt, imgpt1, imgpt2,
                    K1, c1, K2, c2,
                    imageSize, R, T, E, F, err, flags);

    EXPECT_LE(res1, res0);
    EXPECT_TRUE(err.total() == 2);
}

TEST(Calib_StereoCalibrate, regression_22421)
{
    cv::Mat K1, K2, dist1, dist2;
    std::vector<Mat> image_points1, image_points2;
    Mat desiredR, desiredT;

    std::string fname = cvtest::TS::ptr()->get_data_path() + std::string("/cv/cameracalibration/regression22421.yaml.gz");
    FileStorage fs(fname, FileStorage::Mode::READ);
    ASSERT_TRUE(fs.isOpened());
    fs["imagePoints1"] >> image_points1;
    fs["imagePoints2"] >> image_points2;
    fs["K1"] >> K1;
    fs["K2"] >> K2;
    fs["dist1"] >> dist1;
    fs["dist2"] >> dist2;
    fs["desiredR"] >> desiredR;
    fs["desiredT"] >> desiredT;

    std::vector<float> obj_points = {0, 0, 0,
                                     0.5f, 0, 0,
                                     1.f, 0, 0,
                                     1.5000001f, 0, 0,
                                     2.0f, 0, 0,
                                     0, 0.5f, 0,
                                     0.5f, 0.5f, 0,
                                     1.f, 0.5, 0,
                                     1.5000001f, 0.5f, 0,
                                     2.f, 0.5f, 0,
                                     0, 1.f, 0,
                                     0.5f, 1.f, 0,
                                     1.f, 1.f, 0,
                                     1.5000001f, 1.f, 0,
                                     2.f, 1.f, 0,
                                     0, 1.5000001f, 0,
                                     0.5f, 1.5000001f, 0,
                                     1.f, 1.5000001f, 0,
                                     1.5000001f, 1.5000001f, 0,
                                     2.f, 1.5000001f, 0};

    cv::Mat obj_points_mat(obj_points, true);
    obj_points_mat = obj_points_mat.reshape(1, int(obj_points.size()) / 3);
    std::vector<cv::Mat> grid_points(image_points1.size(), obj_points_mat);

    cv::Mat R, T;
    double error = cv::stereoCalibrate(grid_points, image_points1, image_points2,
                                       K1, dist1, K2, dist2, cv::Size(), R, T,
                                       cv::noArray(), cv::noArray(), cv::noArray(), cv::CALIB_FIX_INTRINSIC);

    EXPECT_LE(error, 0.071544);

    double rerr = cv::norm(R, desiredR, NORM_INF);
    double terr = cv::norm(T, desiredT, NORM_INF);

    EXPECT_LE(rerr, 0.0000001);
    EXPECT_LE(terr, 0.0000001);
}

}} // namespace
