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

using namespace cv;
using namespace std;

#define OCL_TUNING_MODE 0
#if OCL_TUNING_MODE
#define OCL_TUNING_MODE_ONLY(code) code
#else
#define OCL_TUNING_MODE_ONLY(code)
#endif

// image moments
class CV_MomentsTest : public cvtest::ArrayTest
{
public:
    CV_MomentsTest();

protected:

    enum { MOMENT_COUNT = 25 };
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    int coi;
    bool is_binary;
    bool try_umat;
};


CV_MomentsTest::CV_MomentsTest()
{
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    coi = -1;
    is_binary = false;
    OCL_TUNING_MODE_ONLY(test_case_count = 10);
    //element_wise_relative_error = false;
}


void CV_MomentsTest::get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high )
{
    cvtest::ArrayTest::get_minmax_bounds( i, j, type, low, high );
    int depth = CV_MAT_DEPTH(type);

    if( depth == CV_16U )
    {
        low = Scalar::all(0);
        high = Scalar::all(1000);
    }
    else if( depth == CV_16S )
    {
        low = Scalar::all(-1000);
        high = Scalar::all(1000);
    }
    else if( depth == CV_32F )
    {
        low = Scalar::all(-1);
        high = Scalar::all(1);
    }
}

void CV_MomentsTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int cn = (cvtest::randInt(rng) % 4) + 1;
    int depth = cvtest::randInt(rng) % 4;
    depth = depth == 0 ? CV_8U : depth == 1 ? CV_16U : depth == 2 ? CV_16S : CV_32F;

    is_binary = cvtest::randInt(rng) % 2 != 0;
    if( depth == 0 && !is_binary )
        try_umat = cvtest::randInt(rng) % 5 != 0;
    else
        try_umat = cvtest::randInt(rng) % 2 != 0;

    if( cn == 2 || try_umat )
        cn = 1;

    OCL_TUNING_MODE_ONLY(
    cn = 1;
    depth = CV_8U;
    try_umat = true;
    is_binary = false;
    sizes[INPUT][0] = Size(1024,768)
    );

    types[INPUT][0] = CV_MAKETYPE(depth, cn);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_64FC1;
    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize(MOMENT_COUNT,1);
    if(CV_MAT_DEPTH(types[INPUT][0])>=CV_32S)
        sizes[INPUT][0].width = MAX(sizes[INPUT][0].width, 3);

    coi = 0;
    cvmat_allowed = true;
    if( cn > 1 )
    {
        coi = cvtest::randInt(rng) % cn;
        cvmat_allowed = false;
    }
}


double CV_MomentsTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = test_mat[INPUT][0].depth();
    return depth != CV_32F ? FLT_EPSILON*10 : FLT_EPSILON*100;
}

int CV_MomentsTest::prepare_test_case( int test_case_idx )
{
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        int cn = test_mat[INPUT][0].channels();
        if( cn > 1 )
            cvSetImageCOI( (IplImage*)test_array[INPUT][0], coi + 1 );
    }

    return code;
}


void CV_MomentsTest::run_func()
{
    CvMoments* m = (CvMoments*)test_mat[OUTPUT][0].ptr<double>();
    double* others = (double*)(m + 1);
    if( try_umat )
    {
        UMat u;
        test_mat[INPUT][0].clone().copyTo(u);
        OCL_TUNING_MODE_ONLY(
            static double ttime = 0;
            static int ncalls = 0;
            moments(u, is_binary != 0);
            double t = (double)getTickCount());
        Moments new_m = moments(u, is_binary != 0);
        OCL_TUNING_MODE_ONLY(
            ttime += (double)getTickCount() - t;
            ncalls++;
            printf("%g\n", ttime/ncalls/u.total()));
        *m = new_m;
    }
    else
        cvMoments( test_array[INPUT][0], m, is_binary );

    others[0] = cvGetNormalizedCentralMoment( m, 2, 0 );
    others[1] = cvGetNormalizedCentralMoment( m, 1, 1 );
    others[2] = cvGetNormalizedCentralMoment( m, 0, 2 );
    others[3] = cvGetNormalizedCentralMoment( m, 3, 0 );
    others[4] = cvGetNormalizedCentralMoment( m, 2, 1 );
    others[5] = cvGetNormalizedCentralMoment( m, 1, 2 );
    others[6] = cvGetNormalizedCentralMoment( m, 0, 3 );
}


void CV_MomentsTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat& src = test_mat[INPUT][0];
    CvMoments m;
    double* mdata = test_mat[REF_OUTPUT][0].ptr<double>();
    int depth = src.depth();
    int cn = src.channels();
    int i, y, x, cols = src.cols;
    double xc = 0., yc = 0.;

    memset( &m, 0, sizeof(m));

    for( y = 0; y < src.rows; y++ )
    {
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        uchar* ptr = src.ptr(y);
        for( x = 0; x < cols; x++ )
        {
            double val;
            if( depth == CV_8U )
                val = ptr[x*cn + coi];
            else if( depth == CV_16U )
                val = ((ushort*)ptr)[x*cn + coi];
            else if( depth == CV_16S )
                val = ((short*)ptr)[x*cn + coi];
            else
                val = ((float*)ptr)[x*cn + coi];

            if( is_binary )
                val = val != 0;

            s0 += val;
            s1 += val*x;
            s2 += val*x*x;
            s3 += ((val*x)*x)*x;
        }

        m.m00 += s0;
        m.m01 += s0*y;
        m.m02 += (s0*y)*y;
        m.m03 += ((s0*y)*y)*y;

        m.m10 += s1;
        m.m11 += s1*y;
        m.m12 += (s1*y)*y;

        m.m20 += s2;
        m.m21 += s2*y;

        m.m30 += s3;
    }

    if( m.m00 != 0 )
    {
        xc = m.m10/m.m00, yc = m.m01/m.m00;
        m.inv_sqrt_m00 = 1./sqrt(fabs(m.m00));
    }

    for( y = 0; y < src.rows; y++ )
    {
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0, y1 = y - yc;
        uchar* ptr = src.ptr(y);
        for( x = 0; x < cols; x++ )
        {
            double val, x1 = x - xc;
            if( depth == CV_8U )
                val = ptr[x*cn + coi];
            else if( depth == CV_16U )
                val = ((ushort*)ptr)[x*cn + coi];
            else if( depth == CV_16S )
                val = ((short*)ptr)[x*cn + coi];
            else
                val = ((float*)ptr)[x*cn + coi];

            if( is_binary )
                val = val != 0;

            s0 += val;
            s1 += val*x1;
            s2 += val*x1*x1;
            s3 += ((val*x1)*x1)*x1;
        }

        m.mu02 += s0*y1*y1;
        m.mu03 += ((s0*y1)*y1)*y1;

        m.mu11 += s1*y1;
        m.mu12 += (s1*y1)*y1;

        m.mu20 += s2;
        m.mu21 += s2*y1;

        m.mu30 += s3;
    }

    memcpy( mdata, &m, sizeof(m));
    mdata += sizeof(m)/sizeof(m.m00);

    /* calc normalized moments */
    {
        double inv_m00 = m.inv_sqrt_m00*m.inv_sqrt_m00;
        double s2 = inv_m00*inv_m00; /* 1./(m00 ^ (2/2 + 1)) */
        double s3 = s2*m.inv_sqrt_m00; /* 1./(m00 ^ (3/2 + 1)) */

        mdata[0] = m.mu20 * s2;
        mdata[1] = m.mu11 * s2;
        mdata[2] = m.mu02 * s2;

        mdata[3] = m.mu30 * s3;
        mdata[4] = m.mu21 * s3;
        mdata[5] = m.mu12 * s3;
        mdata[6] = m.mu03 * s3;
    }

    double* a = test_mat[REF_OUTPUT][0].ptr<double>();
    double* b = test_mat[OUTPUT][0].ptr<double>();
    for( i = 0; i < MOMENT_COUNT; i++ )
    {
        if( fabs(a[i]) < 1e-3 )
            a[i] = 0;
        if( fabs(b[i]) < 1e-3 )
            b[i] = 0;
    }
}


// Hu invariants
class CV_HuMomentsTest : public cvtest::ArrayTest
{
public:
    CV_HuMomentsTest();

protected:

    enum { MOMENT_COUNT = 18, HU_MOMENT_COUNT = 7 };

    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
};


CV_HuMomentsTest::CV_HuMomentsTest()
{
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
}


void CV_HuMomentsTest::get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high )
{
    cvtest::ArrayTest::get_minmax_bounds( i, j, type, low, high );
    low = Scalar::all(-10000);
    high = Scalar::all(10000);
}


void CV_HuMomentsTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_64FC1;
    sizes[INPUT][0] = cvSize(MOMENT_COUNT,1);
    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize(HU_MOMENT_COUNT,1);
}


double CV_HuMomentsTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return FLT_EPSILON;
}



int CV_HuMomentsTest::prepare_test_case( int test_case_idx )
{
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        // ...
    }

    return code;
}


void CV_HuMomentsTest::run_func()
{
    cvGetHuMoments( (CvMoments*)test_mat[INPUT][0].data,
                    (CvHuMoments*)test_mat[OUTPUT][0].data );
}


void CV_HuMomentsTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMoments* m = (CvMoments*)test_mat[INPUT][0].data;
    CvHuMoments* hu = (CvHuMoments*)test_mat[REF_OUTPUT][0].data;

    double inv_m00 = m->inv_sqrt_m00*m->inv_sqrt_m00;
    double s2 = inv_m00*inv_m00; /* 1./(m00 ^ (2/2 + 1)) */
    double s3 = s2*m->inv_sqrt_m00; /* 1./(m00 ^ (3/2 + 1)) */

    double nu20 = m->mu20 * s2;
    double nu11 = m->mu11 * s2;
    double nu02 = m->mu02 * s2;

    double nu30 = m->mu30 * s3;
    double nu21 = m->mu21 * s3;
    double nu12 = m->mu12 * s3;
    double nu03 = m->mu03 * s3;

    #undef sqr
    #define sqr(a) ((a)*(a))

    hu->hu1 = nu20 + nu02;
    hu->hu2 = sqr(nu20 - nu02) + 4*sqr(nu11);
    hu->hu3 = sqr(nu30 - 3*nu12) + sqr(3*nu21 - nu03);
    hu->hu4 = sqr(nu30 + nu12) + sqr(nu21 + nu03);
    hu->hu5 = (nu30 - 3*nu12)*(nu30 + nu12)*(sqr(nu30 + nu12) - 3*sqr(nu21 + nu03)) +
            (3*nu21 - nu03)*(nu21 + nu03)*(3*sqr(nu30 + nu12) - sqr(nu21 + nu03));
    hu->hu6 = (nu20 - nu02)*(sqr(nu30 + nu12) - sqr(nu21 + nu03)) +
            4*nu11*(nu30 + nu12)*(nu21 + nu03);
    hu->hu7 = (3*nu21 - nu03)*(nu30 + nu12)*(sqr(nu30 + nu12) - 3*sqr(nu21 + nu03)) +
            (3*nu12 - nu30)*(nu21 + nu03)*(3*sqr(nu30 + nu12) - sqr(nu21 + nu03));
}


TEST(Imgproc_Moments, accuracy) { CV_MomentsTest test; test.safe_run(); }
TEST(Imgproc_HuMoments, accuracy) { CV_HuMomentsTest test; test.safe_run(); }

class CV_SmallContourMomentTest : public cvtest::BaseTest
{
public:
    CV_SmallContourMomentTest() {}
    ~CV_SmallContourMomentTest() {}
protected:
    void run(int)
    {
        try
        {
            vector<Point> points;
            points.push_back(Point(50, 56));
            points.push_back(Point(53, 53));
            points.push_back(Point(46, 54));
            points.push_back(Point(49, 51));

            Moments m = moments(points, false);
            double area = contourArea(points);

            CV_Assert( m.m00 == 0 && m.m01 == 0 && m.m10 == 0 && area == 0 );
        }
        catch(...)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        }
    }
};

TEST(Imgproc_ContourMoment, small) { CV_SmallContourMomentTest test; test.safe_run(); }
