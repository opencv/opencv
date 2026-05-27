// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/3d.hpp"
#include "opencv2/ts/cuda_test.hpp" // EXPECT_MAT_NEAR

namespace opencv_test { namespace {

#if 0
class CV_ProjectPointsTest : public cvtest::ArrayTest
    {
public:
    CV_ProjectPointsTest();

protected:
    int read_params( const cv::FileStorage& fs );
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


int CV_ProjectPointsTest::read_params( const cv::FileStorage& fs )
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

    theta0 = cvtest::norm( cvarrtomat(vec2), 0, NORM_L2 );
    theta1 = fmod( theta0, CV_PI*2 );

    if( theta1 > CV_PI )
        theta1 = -(CV_PI*2 - theta1);
        cvScale( vec2, vec2, theta1/(theta0 ? theta0 : 1) );

    if( calc_jacobians )
    {
        //cvInvert( v2m_jac, m2v_jac, CV_SVD );
        if( cvtest::norm(cvarrtomat(&test_mat[OUTPUT][3]), 0, NORM_INF) < 1000 )
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
    rmat = cv::Mat_<float>::zeros(3, 3);
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
    CV_Assert( (int)imgPoints.size() == pointCount );
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
    err = cvtest::norm( dpdrot, valDpdrot, NORM_INF );
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
    if( cvtest::norm( dpdt, valDpdt, NORM_INF ) > 0.2 )
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
    if ( cvtest::norm( dpdf, valDpdf, NORM_L2 ) > 0.2 )
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
    if ( cvtest::norm( dpdc, valDpdc, NORM_L2 ) > 0.2 )
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
    if( cvtest::norm( dpddist, valDpddist, NORM_L2 ) > 0.3 )
    {
        ts->printf( cvtest::TS::LOG, "bad dpddist\n" );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    _exit_:
    RNG& _rng = ts->get_rng();
    _rng = rng;
    ts->set_failed_test_info( code );
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

TEST(Calib3d_ProjectPoints_CPP, regression) { CV_ProjectPointsTest_CPP test; test.safe_run(); }

TEST(Calib3d_ProjectPoints_CPP, inputShape)
{
    Matx31d rvec = Matx31d::zeros();
    Matx31d tvec(0, 0, 1);
    Matx33d cameraMatrix = Matx33d::eye();
    const float L = 0.1f;
    {
        //3xN 1-channel
        Mat objectPoints = (Mat_<float>(3, 2) << -L,  L,
                            L,  L,
                            0,  0);
        vector<Point2f> imagePoints;
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, noArray(), imagePoints);
        EXPECT_EQ(objectPoints.cols, static_cast<int>(imagePoints.size()));
        EXPECT_NEAR(imagePoints[0].x, -L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[0].y,  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[1].x,  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[1].y,  L, std::numeric_limits<float>::epsilon());
    }
    {
        //Nx2 1-channel
        Mat objectPoints = (Mat_<float>(2, 3) << -L,  L, 0,
                            L,  L, 0);
        vector<Point2f> imagePoints;
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, noArray(), imagePoints);
        EXPECT_EQ(objectPoints.rows, static_cast<int>(imagePoints.size()));
        EXPECT_NEAR(imagePoints[0].x, -L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[0].y,  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[1].x,  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[1].y,  L, std::numeric_limits<float>::epsilon());
    }
    {
        //1xN 3-channel
        Mat objectPoints(1, 2, CV_32FC3);
        objectPoints.at<Vec3f>(0,0) = Vec3f(-L, L, 0);
        objectPoints.at<Vec3f>(0,1) = Vec3f(L, L, 0);

        vector<Point2f> imagePoints;
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, noArray(), imagePoints);
        EXPECT_EQ(objectPoints.cols, static_cast<int>(imagePoints.size()));
        EXPECT_NEAR(imagePoints[0].x, -L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[0].y,  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[1].x,  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[1].y,  L, std::numeric_limits<float>::epsilon());
    }
    {
        //Nx1 3-channel
        Mat objectPoints(2, 1, CV_32FC3);
        objectPoints.at<Vec3f>(0,0) = Vec3f(-L, L, 0);
        objectPoints.at<Vec3f>(1,0) = Vec3f(L, L, 0);

        vector<Point2f> imagePoints;
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, noArray(), imagePoints);
        EXPECT_EQ(objectPoints.rows, static_cast<int>(imagePoints.size()));
        EXPECT_NEAR(imagePoints[0].x, -L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[0].y,  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[1].x,  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[1].y,  L, std::numeric_limits<float>::epsilon());
    }
    {
        //vector<Point3f>
        vector<Point3f> objectPoints;
        objectPoints.push_back(Point3f(-L, L, 0));
        objectPoints.push_back(Point3f(L, L, 0));

        vector<Point2f> imagePoints;
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, noArray(), imagePoints);
        EXPECT_EQ(objectPoints.size(), imagePoints.size());
        EXPECT_NEAR(imagePoints[0].x, -L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[0].y,  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[1].x,  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[1].y,  L, std::numeric_limits<float>::epsilon());
    }
    {
        //vector<Point3d>
        vector<Point3d> objectPoints;
        objectPoints.push_back(Point3d(-L, L, 0));
        objectPoints.push_back(Point3d(L, L, 0));

        vector<Point2d> imagePoints;
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, noArray(), imagePoints);
        EXPECT_EQ(objectPoints.size(), imagePoints.size());
        EXPECT_NEAR(imagePoints[0].x, -L, std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(imagePoints[0].y,  L, std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(imagePoints[1].x,  L, std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(imagePoints[1].y,  L, std::numeric_limits<double>::epsilon());
    }
}

TEST(Calib3d_ProjectPoints_CPP, outputShape)
{
    Matx31d rvec = Matx31d::zeros();
    Matx31d tvec(0, 0, 1);
    Matx33d cameraMatrix = Matx33d::eye();
    const float L = 0.1f;
    {
        vector<Point3f> objectPoints;
        objectPoints.push_back(Point3f(-L,  L, 0));
        objectPoints.push_back(Point3f( L,  L, 0));
        objectPoints.push_back(Point3f( L, -L, 0));

        //Mat --> will be Nx1 2-channel
        Mat imagePoints;
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, noArray(), imagePoints);
        EXPECT_EQ(static_cast<int>(objectPoints.size()), imagePoints.rows);
        EXPECT_NEAR(imagePoints.at<Vec2f>(0,0)(0), -L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(0,0)(1),  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(1,0)(0),  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(1,0)(1),  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(2,0)(0),  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(2,0)(1), -L, std::numeric_limits<float>::epsilon());
    }
    {
        vector<Point3f> objectPoints;
        objectPoints.push_back(Point3f(-L,  L, 0));
        objectPoints.push_back(Point3f( L,  L, 0));
        objectPoints.push_back(Point3f( L, -L, 0));

        //Nx1 2-channel
        Mat imagePoints(3,1,CV_32FC2);
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, noArray(), imagePoints);
        EXPECT_EQ(static_cast<int>(objectPoints.size()), imagePoints.rows);
        EXPECT_NEAR(imagePoints.at<Vec2f>(0,0)(0), -L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(0,0)(1),  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(1,0)(0),  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(1,0)(1),  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(2,0)(0),  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(2,0)(1), -L, std::numeric_limits<float>::epsilon());
    }
    {
        vector<Point3f> objectPoints;
        objectPoints.push_back(Point3f(-L,  L, 0));
        objectPoints.push_back(Point3f( L,  L, 0));
        objectPoints.push_back(Point3f( L, -L, 0));

        //1xN 2-channel
        Mat imagePoints(1,3,CV_32FC2);
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, noArray(), imagePoints);
        EXPECT_EQ(static_cast<int>(objectPoints.size()), imagePoints.cols);
        EXPECT_NEAR(imagePoints.at<Vec2f>(0,0)(0), -L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(0,0)(1),  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(0,1)(0),  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(0,1)(1),  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(0,2)(0),  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints.at<Vec2f>(0,2)(1), -L, std::numeric_limits<float>::epsilon());
    }
    {
        vector<Point3f> objectPoints;
        objectPoints.push_back(Point3f(-L, L, 0));
        objectPoints.push_back(Point3f(L, L, 0));

        //vector<Point2f>
        vector<Point2f> imagePoints;
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, noArray(), imagePoints);
        EXPECT_EQ(objectPoints.size(), imagePoints.size());
        EXPECT_NEAR(imagePoints[0].x, -L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[0].y,  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[1].x,  L, std::numeric_limits<float>::epsilon());
        EXPECT_NEAR(imagePoints[1].y,  L, std::numeric_limits<float>::epsilon());
    }
    {
        vector<Point3d> objectPoints;
        objectPoints.push_back(Point3d(-L, L, 0));
        objectPoints.push_back(Point3d(L, L, 0));

        //vector<Point2d>
        vector<Point2d> imagePoints;
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, noArray(), imagePoints);
        EXPECT_EQ(objectPoints.size(), imagePoints.size());
        EXPECT_NEAR(imagePoints[0].x, -L, std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(imagePoints[0].y,  L, std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(imagePoints[1].x,  L, std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(imagePoints[1].y,  L, std::numeric_limits<double>::epsilon());
    }
}

#define NUM_DIST_COEFF_TILT 14

class cameraCalibrationTiltTest : public ::testing::Test {

protected:

    static const cv::Size m_imageSize;
    static const double m_pixelSize;
    static const double m_circleConfusionPixel;
    static const double m_lensFocalLength;
    static const double m_lensFNumber;
    static const double m_objectDistance;
    static const double m_planeTiltDegree;
    static const double m_pointTargetDist;
    static const int m_pointTargetNum;

    /** image distance corresponding to working distance */
    double m_imageDistance;
    /** image tilt angle corresponding to the tilt of the object plane */
    double m_imageTiltDegree;
    /** center of the field of view, near and far plane */
    std::vector<cv::Vec3d> m_fovCenter;
    /** normal of the field of view, near and far plane */
    std::vector<cv::Vec3d> m_fovNormal;
    /** points on a plane calibration target */
    std::vector<cv::Point3d> m_pointTarget;
    /** rotations for the calibration target */
    std::vector<cv::Vec3d> m_pointTargetRvec;
    /** translations for the calibration target */
    std::vector<cv::Vec3d> m_pointTargetTvec;
    /** camera matrix */
    cv::Matx33d m_cameraMatrix;
    /** distortion coefficients */
    cv::Vec<double, NUM_DIST_COEFF_TILT> m_distortionCoeff;

    /** random generator */
    cv::RNG m_rng;
    /** degree to radian conversion factor */
    const double m_toRadian;
    /** radian to degree conversion factor */
    const double m_toDegree;

    cameraCalibrationTiltTest()
    : m_toRadian(acos(-1.0)/180.0)
    , m_toDegree(180.0/acos(-1.0))
    {}

    /**
     C *hanges given distortion coefficients randomly by adding
     a uniformly distributed random variable in [-max max]
     \param coeff input
     \param max limits for the random variables
     */
    void randomDistortionCoeff(
        cv::Vec<double, NUM_DIST_COEFF_TILT>& coeff,
        const cv::Vec<double, NUM_DIST_COEFF_TILT>& max)
    {
        for (int i = 0; i < coeff.rows; ++i)
            coeff(i) += m_rng.uniform(-max(i), max(i));
    }

    void removeInvalidPoints(
        std::vector<cv::Point2d>& imagePoints,
        std::vector<cv::Point3d>& objectPoints)
    {
        // remove object and imgage points out of range
        std::vector<cv::Point2d>::iterator itImg = imagePoints.begin();
        std::vector<cv::Point3d>::iterator itObj = objectPoints.begin();
        while (itImg != imagePoints.end())
        {
            bool ok =
            itImg->x >= 0 &&
            itImg->x <= m_imageSize.width - 1.0 &&
            itImg->y >= 0 &&
            itImg->y <= m_imageSize.height - 1.0;
            if (ok)
            {
                ++itImg;
                ++itObj;
            }
            else
            {
                itImg = imagePoints.erase(itImg);
                itObj = objectPoints.erase(itObj);
            }
        }
    }

    void numericalDerivative(
        cv::Mat& jac,
        double eps,
        const std::vector<cv::Point3d>& obj,
        const cv::Vec3d& rvec,
        const cv::Vec3d& tvec,
        const cv::Matx33d& camera,
        const cv::Vec<double, NUM_DIST_COEFF_TILT>& distor)
    {
        cv::Vec3d r(rvec);
        cv::Vec3d t(tvec);
        cv::Matx33d cm(camera);
        cv::Vec<double, NUM_DIST_COEFF_TILT> dc(distor);
        double* param[10+NUM_DIST_COEFF_TILT] = {
            &r(0), &r(1), &r(2),
            &t(0), &t(1), &t(2),
            &cm(0,0), &cm(1,1), &cm(0,2), &cm(1,2),
            &dc(0), &dc(1), &dc(2), &dc(3), &dc(4), &dc(5), &dc(6),
            &dc(7), &dc(8), &dc(9), &dc(10), &dc(11), &dc(12), &dc(13)};
        std::vector<cv::Point2d> pix0, pix1;
        double invEps = .5/eps;

        for (int col = 0; col < 10+NUM_DIST_COEFF_TILT; ++col)
        {
            double save = *(param[col]);
            *(param[col]) = save + eps;
            cv::projectPoints(obj, r, t, cm, dc, pix0);
            *(param[col]) = save - eps;
            cv::projectPoints(obj, r, t, cm, dc, pix1);
            *(param[col]) = save;

            std::vector<cv::Point2d>::const_iterator it0 = pix0.begin();
            std::vector<cv::Point2d>::const_iterator it1 = pix1.begin();
            int row = 0;
            for (;it0 != pix0.end(); ++it0, ++it1)
            {
                cv::Point2d d = invEps*(*it0 - *it1);
                jac.at<double>(row, col) = d.x;
                ++row;
                jac.at<double>(row, col) = d.y;
                ++row;
            }
        }
    }

    /**
     c *omputes for a given distance of an image or object point
     the distance of the corresponding object or image point
     */
    double opticalMap(double dist) {
        return m_lensFocalLength*dist/(dist - m_lensFocalLength);
    }

    /** magnification of the optical map */
    double magnification(double dist) {
        return m_lensFocalLength/(dist - m_lensFocalLength);
    }

    void SetUp()
    {
        m_imageDistance = opticalMap(m_objectDistance);
        m_imageTiltDegree = m_toDegree * atan2(
            m_imageDistance * tan(m_toRadian * m_planeTiltDegree),
            m_objectDistance);
        // half sensor height
        double tmp = .5 * (m_imageSize.height - 1) * m_pixelSize
            * cos(m_toRadian * m_imageTiltDegree);
        // y-Value of tilted sensor
        double yImage[2] = {tmp, -tmp};
        // change in z because of the tilt
        tmp *= sin(m_toRadian * m_imageTiltDegree);
        // z-values of the sensor lower and upper corner
        double zImage[2] = {
            m_imageDistance + tmp,
            m_imageDistance - tmp};
        // circle of confusion
        double circleConfusion = m_circleConfusionPixel*m_pixelSize;
        // aperture of the lense
        double aperture = m_lensFocalLength/m_lensFNumber;
        // near and far factor on the image side
        double nearFarFactorImage[2] = {
            aperture/(aperture - circleConfusion),
            aperture/(aperture + circleConfusion)};
        // on the object side - points that determine the field of
        // view
        std::vector<cv::Vec3d> fovBottomTop(6);
        std::vector<cv::Vec3d>::iterator itFov = fovBottomTop.begin();
        for (size_t iBottomTop = 0; iBottomTop < 2; ++iBottomTop)
        {
            // mapping sensor to field of view
            *itFov = cv::Vec3d(0,yImage[iBottomTop],zImage[iBottomTop]);
            *itFov *= magnification((*itFov)(2));
            ++itFov;
            for (size_t iNearFar = 0; iNearFar < 2; ++iNearFar, ++itFov)
            {
                // scaling to the near and far distance on the
                // image side
                *itFov = cv::Vec3d(0,yImage[iBottomTop],zImage[iBottomTop]) *
                    nearFarFactorImage[iNearFar];
                // scaling to the object side
                *itFov *= magnification((*itFov)(2));
            }
        }
        m_fovCenter.resize(3);
        m_fovNormal.resize(3);
        for (size_t i = 0; i < 3; ++i)
        {
            m_fovCenter[i] = .5*(fovBottomTop[i] + fovBottomTop[i+3]);
            m_fovNormal[i] = fovBottomTop[i+3] - fovBottomTop[i];
            m_fovNormal[i] = cv::normalize(m_fovNormal[i]);
            m_fovNormal[i] = cv::Vec3d(
                m_fovNormal[i](0),
                -m_fovNormal[i](2),
                m_fovNormal[i](1));
            // one target position in each plane
            m_pointTargetTvec.push_back(m_fovCenter[i]);
            cv::Vec3d rvec = cv::Vec3d(0,0,1).cross(m_fovNormal[i]);
            rvec = cv::normalize(rvec);
            rvec *= acos(m_fovNormal[i](2));
            m_pointTargetRvec.push_back(rvec);
        }
        // calibration target
        size_t num = 2*m_pointTargetNum + 1;
        m_pointTarget.resize(num*num);
        std::vector<cv::Point3d>::iterator itTarget = m_pointTarget.begin();
        for (int iY = -m_pointTargetNum; iY <= m_pointTargetNum; ++iY)
        {
            for (int iX = -m_pointTargetNum; iX <= m_pointTargetNum; ++iX, ++itTarget)
            {
                *itTarget = cv::Point3d(iX, iY, 0) * m_pointTargetDist;
            }
        }
        // oblique target positions
        // approximate distance to the near and far plane
        double dist = std::max(
            std::abs(m_fovNormal[0].dot(m_fovCenter[0] - m_fovCenter[1])),
            std::abs(m_fovNormal[0].dot(m_fovCenter[0] - m_fovCenter[2])));
        // maximal angle such that target border "reaches" near and far plane
        double maxAngle = atan2(dist, m_pointTargetNum*m_pointTargetDist);
        std::vector<double> angle;
        angle.push_back(-maxAngle);
        angle.push_back(maxAngle);
        cv::Matx33d baseMatrix;
        cv::Rodrigues(m_pointTargetRvec.front(), baseMatrix);
        for (std::vector<double>::const_iterator itAngle = angle.begin(); itAngle != angle.end(); ++itAngle)
        {
            cv::Matx33d rmat;
            for (int i = 0; i < 2; ++i)
            {
                cv::Vec3d rvec(0,0,0);
                rvec(i) = *itAngle;
                cv::Rodrigues(rvec, rmat);
                rmat = baseMatrix*rmat;
                cv::Rodrigues(rmat, rvec);
                m_pointTargetTvec.push_back(m_fovCenter.front());
                m_pointTargetRvec.push_back(rvec);
            }
        }
        // camera matrix
        double cx = .5 * (m_imageSize.width - 1);
        double cy = .5 * (m_imageSize.height - 1);
        double f = m_imageDistance/m_pixelSize;
        m_cameraMatrix = cv::Matx33d(
            f,0,cx,
            0,f,cy,
            0,0,1);
        // distortion coefficients
        m_distortionCoeff = cv::Vec<double, NUM_DIST_COEFF_TILT>::all(0);
        // tauX
        m_distortionCoeff(12) = -m_toRadian*m_imageTiltDegree;
    }
};

/** Number of Pixel of the sensor */
const cv::Size cameraCalibrationTiltTest::m_imageSize(1600, 1200);
/** Size of a pixel in mm */
const double cameraCalibrationTiltTest::m_pixelSize(.005);
/** Diameter of the circle of confusion */
const double cameraCalibrationTiltTest::m_circleConfusionPixel(3);
/** Focal length of the lens */
const double cameraCalibrationTiltTest::m_lensFocalLength(16.4);
/** F-Number */
const double cameraCalibrationTiltTest::m_lensFNumber(8);
/** Working distance */
const double cameraCalibrationTiltTest::m_objectDistance(200);
/** Angle between optical axis and object plane normal */
const double cameraCalibrationTiltTest::m_planeTiltDegree(55);
/** the calibration target are points on a square grid with this side length */
const double cameraCalibrationTiltTest::m_pointTargetDist(5);
/** the calibration target has (2*n + 1) x (2*n + 1) points */
const int cameraCalibrationTiltTest::m_pointTargetNum(15);

TEST_F(cameraCalibrationTiltTest, projectPoints)
{
    std::vector<cv::Point2d> imagePoints;
    std::vector<cv::Point3d> objectPoints = m_pointTarget;
    cv::Vec3d rvec = m_pointTargetRvec.front();
    cv::Vec3d tvec = m_pointTargetTvec.front();

    cv::Vec<double, NUM_DIST_COEFF_TILT> coeffNoiseHalfWidth(
        .1, .1, // k1 k2
        .01, .01, // p1 p2
        .001, .001, .001, .001, // k3 k4 k5 k6
        .001, .001, .001, .001, // s1 s2 s3 s4
        .01, .01); // tauX tauY
    for (size_t numTest = 0; numTest < 10; ++numTest)
    {
        // create random distortion coefficients
        cv::Vec<double, NUM_DIST_COEFF_TILT> distortionCoeff = m_distortionCoeff;
        randomDistortionCoeff(distortionCoeff, coeffNoiseHalfWidth);

        // projection
        cv::projectPoints(
            objectPoints,
            rvec,
            tvec,
            m_cameraMatrix,
            distortionCoeff,
            imagePoints);

        // remove object and imgage points out of range
        removeInvalidPoints(imagePoints, objectPoints);

        int numPoints = (int)imagePoints.size();
        int numParams = 10 + distortionCoeff.rows;
        cv::Mat jacobian(2*numPoints, numParams, CV_64FC1);

        // projection and jacobian
        cv::projectPoints(
            objectPoints,
            rvec,
            tvec,
            m_cameraMatrix,
            distortionCoeff,
            imagePoints,
            jacobian);

        // numerical derivatives
        cv::Mat numericJacobian(2*numPoints, numParams, CV_64FC1);
        double eps = 1e-7;
        numericalDerivative(
            numericJacobian,
            eps,
            objectPoints,
            rvec,
            tvec,
            m_cameraMatrix,
            distortionCoeff);

#if 0
        for (size_t row = 0; row < 2; ++row)
        {
            std::cout << "------ Row = " << row << " ------\n";
            for (size_t i = 0; i < 10+NUM_DIST_COEFF_TILT; ++i)
            {
                std::cout << i
                    << "  jac = " << jacobian.at<double>(row,i)
                    << "  num = " << numericJacobian.at<double>(row,i)
                    << "  rel. diff = " << abs(numericJacobian.at<double>(row,i) - jacobian.at<double>(row,i))/abs(numericJacobian.at<double>(row,i))
                    << "\n";
            }
        }
#endif
        // relative difference for large values (rvec and tvec)
        cv::Mat check = abs(jacobian(cv::Range::all(), cv::Range(0,6)) - numericJacobian(cv::Range::all(), cv::Range(0,6)))/
            (1 + abs(jacobian(cv::Range::all(), cv::Range(0,6))));
        double minVal, maxVal;
        cv::minMaxIdx(check, &minVal, &maxVal);
        EXPECT_LE(maxVal, .01);
        // absolute difference for distortion and camera matrix
        EXPECT_MAT_NEAR(jacobian(cv::Range::all(), cv::Range(6,numParams)), numericJacobian(cv::Range::all(), cv::Range(6,numParams)), 1e-5);
    }
}

TEST_F(cameraCalibrationTiltTest, undistortPoints)
{
    cv::Vec<double, NUM_DIST_COEFF_TILT> coeffNoiseHalfWidth(
        .2, .1, // k1 k2
        .01, .01, // p1 p2
        .01, .01, .01, .01, // k3 k4 k5 k6
        .001, .001, .001, .001, // s1 s2 s3 s4
        .001, .001); // tauX tauY
    double step = 99;
    double toleranceBackProjection = 1e-5;

    for (size_t numTest = 0; numTest < 10; ++numTest)
    {
        cv::Vec<double, NUM_DIST_COEFF_TILT> distortionCoeff = m_distortionCoeff;
        randomDistortionCoeff(distortionCoeff, coeffNoiseHalfWidth);

        // distorted points
        std::vector<cv::Point2d> distorted;
        for (double x = 0; x <= m_imageSize.width-1; x += step)
            for (double y = 0; y <= m_imageSize.height-1; y += step)
                distorted.push_back(cv::Point2d(x,y));
        std::vector<cv::Point2d> normalizedUndistorted;

        // undistort
        cv::undistortPoints(distorted,
            normalizedUndistorted,
            m_cameraMatrix,
            distortionCoeff);

        // copy normalized points to 3D
        std::vector<cv::Point3d> objectPoints;
        for (std::vector<cv::Point2d>::const_iterator itPnt = normalizedUndistorted.begin();
            itPnt != normalizedUndistorted.end(); ++itPnt)
            objectPoints.push_back(cv::Point3d(itPnt->x, itPnt->y, 1));

        // project
        std::vector<cv::Point2d> imagePoints(objectPoints.size());
        cv::projectPoints(objectPoints,
            cv::Vec3d(0,0,0),
            cv::Vec3d(0,0,0),
            m_cameraMatrix,
            distortionCoeff,
            imagePoints);

        EXPECT_MAT_NEAR(distorted, imagePoints, toleranceBackProjection);
    }
}

TEST(Calib3d_Triangulate, accuracy)
{
    // the testcase from http://code.opencv.org/issues/4334
    {
        double P1data[] = { 250, 0, 200, 0, 0, 250, 150, 0, 0, 0, 1, 0 };
        double P2data[] = { 250, 0, 200, -250, 0, 250, 150, 0, 0, 0, 1, 0 };
        Mat P1(3, 4, CV_64F, P1data), P2(3, 4, CV_64F, P2data);

        float x1data[] = { 200.f, 0.f };
        float x2data[] = { 170.f, 1.f };
        float Xdata[] = { 0.f, -5.f, 25/3.f };
        constexpr int num_points = 5;
        Mat1f res0(num_points, 3);
        for(int i = 0; i < num_points; ++i) {
          for(int j = 0; j < 3; ++j) {
            res0(i, j) = Xdata[j];
          }
        }

        Mat res_, res;

        for(int test : {0, 1}) {
          if (test == 0) {
            // 2xN and 1xN 2-channel.
            Mat1f x1(2, num_points);
            Mat2f x2(1, num_points);
            for(int i = 0; i < 2; ++i) {
              for(int j = 0; j < num_points; ++j) {
                x1(i, j) = x1data[i];
                x2(0, j)[i] = x2data[i];
              }
            }
            triangulatePoints(P1, P2, x1, x2, res_);
          } else {
            // Array and vector.
            std::array<cv::Point2f, num_points> x1;
            x1.fill({x1data[0], x1data[1]});
            std::vector<cv::Point2f> x2(num_points, {x2data[0], x2data[1]});
            triangulatePoints(P1, P2, x1, x2, res_);
          }
          cv::transpose(res_, res_); // TODO cvtest (transpose doesn't support inplace)
          convertPointsFromHomogeneous(res_, res);
          res = res.reshape(1, num_points);

          cout << "[1]:" << endl;
          cout << "\tres0: " << res0 << endl;
          cout << "\tres: " << res << endl;

          ASSERT_LE(cvtest::norm(res, res0, NORM_INF), 1e-1);
        }
    }

    // another testcase http://code.opencv.org/issues/3461
    {
        Matx33d K1(6137.147949, 0.000000, 644.974609,
                   0.000000, 6137.147949, 573.442749,
                   0.000000, 0.000000,  1.000000);
        Matx33d K2(6137.147949,  0.000000, 644.674438,
                   0.000000, 6137.147949, 573.079834,
                   0.000000,  0.000000, 1.000000);

        Matx34d RT1(1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0);

        Matx34d RT2(0.998297, 0.0064108, -0.0579766,     143.614334,
                    -0.0065818, 0.999975, -0.00275888,   -5.160085,
                    0.0579574, 0.00313577, 0.998314,     96.066109);

        Matx34d P1 = K1*RT1;
        Matx34d P2 = K2*RT2;

        float x1data[] = { 438.f, 19.f };
        float x2data[] = { 452.363600f, 16.452225f };
        float Xdata[] = { -81.049530f, -215.702804f, 2401.645449f };
        Mat x1(2, 1, CV_32F, x1data);
        Mat x2(2, 1, CV_32F, x2data);
        Mat res0(1, 3, CV_32F, Xdata);
        Mat res_, res;

        triangulatePoints(P1, P2, x1, x2, res_);
        cv::transpose(res_, res_); // TODO cvtest (transpose doesn't support inplace)
        convertPointsFromHomogeneous(res_, res);
        res = res.reshape(1, 1);

        cout << "[2]:" << endl;
        cout << "\tres0: " << res0 << endl;
        cout << "\tres: " << res << endl;

        ASSERT_LE(cvtest::norm(res, res0, NORM_INF), 2);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(CV_RecoverPoseTest, regression_15341)
{
    // initialize test data
    const int invalid_point_count = 2;
    const float _points1_[] = {
        1537.7f, 166.8f,
        1599.1f, 179.6f,
        1288.0f, 207.5f,
        1507.1f, 193.2f,
        1742.7f, 210.0f,
        1041.6f, 271.7f,
        1591.8f, 247.2f,
        1524.0f, 261.3f,
        1330.3f, 285.0f,
        1403.1f, 284.0f,
        1506.6f, 342.9f,
        1502.8f, 347.3f,
        1344.9f, 364.9f,
        0.0f, 0.0f  // last point is initial invalid
    };

    const float _points2_[] = {
        1533.4f, 532.9f,
        1596.6f, 552.4f,
        1277.0f, 556.4f,
        1502.1f, 557.6f,
        1744.4f, 601.3f,
        1023.0f, 612.6f,
        1589.2f, 621.6f,
        1519.4f, 629.0f,
        1320.3f, 637.3f,
        1395.2f, 642.2f,
        1501.5f, 710.3f,
        1497.6f, 714.2f,
        1335.1f, 719.61f,
        1000.0f, 1000.0f  // last point is initial invalid
    };

    vector<Point2f> _points1; Mat(14, 1, CV_32FC2, (void*)_points1_).copyTo(_points1);
    vector<Point2f> _points2; Mat(14, 1, CV_32FC2, (void*)_points2_).copyTo(_points2);

    const int point_count = (int) _points1.size();
    CV_Assert(point_count == (int) _points2.size());

    // camera matrix with both focal lengths = 1, and principal point = (0, 0)
    const Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

    // camera matrix with focal lengths 0.5 and 0.6 respectively and principal point = (100, 200)
    double cameraMatrix2Data[] = { 0.5, 0, 100,
                                   0, 0.6, 200,
                                   0, 0, 1 };
    const Mat cameraMatrix2( 3, 3, CV_64F, cameraMatrix2Data );

    // zero and nonzero distortion coefficients
    double nonZeroDistCoeffsData[] = { 0.01, 0.0001, 0, 0, 1e-04, 0.2, 0.02, 0.0002 }; // k1, k2, p1, p2, k3, k4, k5, k6
    vector<Mat> distCoeffsList = {Mat::zeros(1, 5, CV_64F), Mat{1, 8, CV_64F, nonZeroDistCoeffsData}};
    const auto &zeroDistCoeffs = distCoeffsList[0];

    int Inliers = 0;

    const int ntests = 3;
    for (int testcase = 1; testcase <= ntests; ++testcase)
    {
        if (testcase == 1) // testcase with vector input data
        {
            // init temporary test data
            vector<unsigned char> mask(point_count);
            vector<Point2f> points1(_points1);
            vector<Point2f> points2(_points2);

            // Estimation of fundamental matrix using the RANSAC algorithm
            Mat E, E2, R, t;

            // Check pose when camera matrices are different.
            for (const auto &distCoeffs: distCoeffsList)
            {
                E = findEssentialMat(points1, points2, cameraMatrix, distCoeffs, cameraMatrix2, distCoeffs, RANSAC, 0.999, 1.0, mask);
                recoverPose(points1, points2, cameraMatrix, distCoeffs, cameraMatrix2, distCoeffs, E2, R, t, RANSAC, 0.999, 1.0, mask);
                EXPECT_LT(cv::norm(E, E2, NORM_INF), 1e-4) <<
                    "Two big difference between the same essential matrices computed using different functions with different cameras, testcase " << testcase;
                EXPECT_EQ(0, (int)mask[13]) << "Detecting outliers in function failed with different cameras, testcase " << testcase;
            }

            // Check pose when camera matrices are the same.
            E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, 1000/*maxIters*/, mask);
            E2 = findEssentialMat(points1, points2, cameraMatrix, zeroDistCoeffs, cameraMatrix, zeroDistCoeffs, RANSAC, 0.999, 1.0, mask);
            EXPECT_LT(cv::norm(E, E2, NORM_INF), 1e-4) <<
                "Two big difference between the same essential matrices computed using different functions with same cameras, testcase " << testcase;
            EXPECT_EQ(0, (int)mask[13]) << "Detecting outliers in function findEssentialMat failed with same cameras, testcase " << testcase;
            points2[12] = Point2f(0.0f, 0.0f); // provoke another outlier detection for recover Pose
            Inliers = recoverPose(E, points1, points2, cameraMatrix, R, t, mask);
            EXPECT_EQ(0, (int)mask[12]) << "Detecting outliers in function failed with same cameras, testcase " << testcase;
        }
        else // testcase with mat input data
        {
            Mat points1(_points1, true);
            Mat points2(_points2, true);
            Mat mask;

            if (testcase == 2)
            {
                // init temporary testdata
                mask = Mat::zeros(point_count, 1, CV_8UC1);
            }
            else // testcase == 3 - with transposed mask
            {
                mask = Mat::zeros(1, point_count, CV_8UC1);
            }

            // Estimation of fundamental matrix using the RANSAC algorithm
            Mat E, E2, R, t;

            // Check pose when camera matrices are different.
            for (const auto &distCoeffs: distCoeffsList)
            {
                E = findEssentialMat(points1, points2, cameraMatrix, distCoeffs, cameraMatrix2, distCoeffs, RANSAC, 0.999, 1.0, mask);
                recoverPose(points1, points2, cameraMatrix, distCoeffs, cameraMatrix2, distCoeffs, E2, R, t, RANSAC, 0.999, 1.0, mask);
                EXPECT_LT(cv::norm(E, E2, NORM_INF), 1e-4) <<
                    "Two big difference between the same essential matrices computed using different functions with different cameras, testcase " << testcase;
                EXPECT_EQ(0, (int)mask.at<unsigned char>(13)) << "Detecting outliers in function failed with different cameras, testcase " << testcase;
            }

            // Check pose when camera matrices are the same.
            E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, 1000/*maxIters*/, mask);
            E2 = findEssentialMat(points1, points2, cameraMatrix, zeroDistCoeffs, cameraMatrix, zeroDistCoeffs, RANSAC, 0.999, 1.0, mask);
            EXPECT_LT(cv::norm(E, E2, NORM_INF), 1e-4) <<
                "Two big difference between the same essential matrices computed using different functions with same cameras, testcase " << testcase;
            EXPECT_EQ(0, (int)mask.at<unsigned char>(13)) << "Detecting outliers in function findEssentialMat failed with same cameras, testcase " << testcase;
            points2.at<Point2f>(12) = Point2f(0.0f, 0.0f); // provoke an outlier detection
            Inliers = recoverPose(E, points1, points2, cameraMatrix, R, t, mask);
            EXPECT_EQ(0, (int)mask.at<unsigned char>(12)) << "Detecting outliers in function failed with same cameras, testcase " << testcase;
        }
        EXPECT_EQ(Inliers, point_count - invalid_point_count) <<
            "Number of inliers differs from expected number of inliers, testcase " << testcase;
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

    goodFovx = (atan2(cx, fx) + atan2(imageSize.width  - cx, fx)) * 180.0 / CV_PI;
    goodFovy = (atan2(cy, fy) + atan2(imageSize.height - cy, fy)) * 180.0 / CV_PI;

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
    if( cv::norm(principalPoint - goodPrincipalPoint) > FLT_EPSILON ) // Point2d
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

TEST(Calib3d_CalibrationMatrixValues_CPP, accuracy) { CV_CalibrationMatrixValuesTest_CPP test; test.safe_run(); }

}} // namespace
