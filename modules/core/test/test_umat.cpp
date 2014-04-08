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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

using namespace cvtest;
using namespace testing;
using namespace cv;

namespace cvtest {
namespace ocl {

#define UMAT_TEST_SIZES testing::Values(cv::Size(1, 1), cv::Size(1,128), cv::Size(128, 1), \
    cv::Size(128, 128), cv::Size(640, 480), cv::Size(751, 373), cv::Size(1200, 1200))

/////////////////////////////// Basic Tests ////////////////////////////////

PARAM_TEST_CASE(UMatBasicTests, int, int, Size, bool)
{
    Mat a;
    UMat ua;
    int type;
    int depth;
    int cn;
    Size size;
    bool useRoi;
    Size roi_size;
    Rect roi;

    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        size = GET_PARAM(2);
        useRoi = GET_PARAM(3);
        type = CV_MAKE_TYPE(depth, cn);
        a = randomMat(size, type, -100, 100);
        a.copyTo(ua);
        int roi_shift_x = randomInt(0, size.width-1);
        int roi_shift_y = randomInt(0, size.height-1);
        roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
        roi = Rect(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
    }
};

TEST_P(UMatBasicTests, createUMat)
{
    if(useRoi)
    {
        ua = UMat(ua, roi);
    }
    int dims = randomInt(2,6);
    int _sz[CV_MAX_DIM];
    for( int i = 0; i<dims; i++)
    {
        _sz[i] = randomInt(1,50);
    }
    int *sz = _sz;
    int new_depth = randomInt(CV_8S, CV_64F);
    int new_cn = randomInt(1,4);
    ua.create(dims, sz, CV_MAKE_TYPE(new_depth, new_cn));

    for(int i = 0; i<dims; i++)
    {
        ASSERT_EQ(ua.size[i], sz[i]);
    }
    ASSERT_EQ(ua.dims, dims);
    ASSERT_EQ(ua.type(), CV_MAKE_TYPE(new_depth, new_cn) );
    Size new_size = randomSize(1, 1000);
    ua.create(new_size, CV_MAKE_TYPE(new_depth, new_cn) );
    ASSERT_EQ( ua.size(), new_size);
    ASSERT_EQ(ua.type(), CV_MAKE_TYPE(new_depth, new_cn) );
    ASSERT_EQ( ua.dims, 2);
}

TEST_P(UMatBasicTests, swap)
{
    Mat b = randomMat(size, type, -100, 100);
    UMat ub;
    b.copyTo(ub);
    if(useRoi)
    {
        ua = UMat(ua,roi);
        ub = UMat(ub,roi);
    }
    UMat uc = ua, ud = ub;
    swap(ua,ub);
    EXPECT_MAT_NEAR(ub,uc, 0);
    EXPECT_MAT_NEAR(ud, ua, 0);
}

TEST_P(UMatBasicTests, base)
{
    if(useRoi)
    {
        ua = UMat(ua,roi);
    }
    UMat ub = ua.clone();
    EXPECT_MAT_NEAR(ub,ua,0);

    ASSERT_EQ(ua.channels(), cn);
    ASSERT_EQ(ua.depth(), depth);
    ASSERT_EQ(ua.type(), type);
    ASSERT_EQ(ua.elemSize(), a.elemSize());
    ASSERT_EQ(ua.elemSize1(), a.elemSize1());
    ASSERT_EQ(ub.empty(), ub.cols*ub.rows == 0);
    ub.release();
    ASSERT_TRUE( ub.empty() );
    if(useRoi && a.size() != ua.size())
    {
        ASSERT_EQ(ua.isSubmatrix(), true);
    }
    else
    {
        ASSERT_EQ(ua.isSubmatrix(), false);
    }

    int dims = randomInt(2,6);
    int sz[CV_MAX_DIM];
    size_t total = 1;
    for(int i = 0; i<dims; i++)
    {
        sz[i] = randomInt(1,45);
        total *= (size_t)sz[i];
    }
    int new_type = CV_MAKE_TYPE(randomInt(CV_8S,CV_64F),randomInt(1,4));
    ub = UMat(dims, sz, new_type);
    ASSERT_EQ(ub.total(), total);
}

TEST_P(UMatBasicTests, DISABLED_copyTo)
{
    UMat roi_ua;
    Mat roi_a;
    int i;
    if(useRoi)
    {
        roi_ua = UMat(ua, roi);
        roi_a = Mat(a, roi);
        roi_a.copyTo(roi_ua);
        EXPECT_MAT_NEAR(roi_a, roi_ua, 0);
        roi_ua.copyTo(roi_a);
        EXPECT_MAT_NEAR(roi_ua, roi_a, 0);
        roi_ua.copyTo(ua);
        EXPECT_MAT_NEAR(roi_ua, ua, 0);
        ua.copyTo(a);
        EXPECT_MAT_NEAR(ua, a, 0);
    }
    {
        UMat ub;
        ua.copyTo(ub);
        EXPECT_MAT_NEAR(ua, ub, 0);
    }
    {
        UMat ub;
        i = randomInt(0, ua.cols-1);
        a.col(i).copyTo(ub);
        EXPECT_MAT_NEAR(a.col(i), ub, 0);
    }
    {
        UMat ub;
        ua.col(i).copyTo(ub);
        EXPECT_MAT_NEAR(ua.col(i), ub, 0);
    }
    {
        Mat b;
        ua.col(i).copyTo(b);
        EXPECT_MAT_NEAR(ua.col(i), b, 0);
    }
    {
        UMat ub;
        i = randomInt(0, a.rows-1);
        ua.row(i).copyTo(ub);
        EXPECT_MAT_NEAR(ua.row(i), ub, 0);
    }
    {
        UMat ub;
        a.row(i).copyTo(ub);
        EXPECT_MAT_NEAR(a.row(i), ub, 0);
    }
    {
        Mat b;
        ua.row(i).copyTo(b);
        EXPECT_MAT_NEAR(ua.row(i), b, 0);
    }
}

TEST_P(UMatBasicTests, DISABLED_GetUMat)
{
    if(useRoi)
    {
        a = Mat(a, roi);
        ua = UMat(ua,roi);
    }
    {
        UMat ub;
        ub = a.getUMat(ACCESS_RW);
        EXPECT_MAT_NEAR(ub, ua, 0);
    }
    {
        Mat b;
        b = a.getUMat(ACCESS_RW).getMat(ACCESS_RW);
        EXPECT_MAT_NEAR(b, a, 0);
    }
    {
        Mat b;
        b = ua.getMat(ACCESS_RW);
        EXPECT_MAT_NEAR(b, a, 0);
    }
    {
        UMat ub;
        ub = ua.getMat(ACCESS_RW).getUMat(ACCESS_RW);
        EXPECT_MAT_NEAR(ub, ua, 0);
    }
}

INSTANTIATE_TEST_CASE_P(UMat, UMatBasicTests, Combine(testing::Values(CV_8U), testing::Values(1, 2),
    testing::Values(cv::Size(1, 1), cv::Size(1, 128), cv::Size(128, 1), cv::Size(128, 128), cv::Size(640, 480)), Bool()));

//////////////////////////////////////////////////////////////// Reshape ////////////////////////////////////////////////////////////////////////

PARAM_TEST_CASE(UMatTestReshape,  int, int, Size, bool)
{
    Mat a;
    UMat ua, ub;
    int type;
    int depth;
    int cn;
    Size size;
    bool useRoi;
    Size roi_size;
    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        size = GET_PARAM(2);
        useRoi = GET_PARAM(3);
        type = CV_MAKE_TYPE(depth, cn);
    }
};

TEST_P(UMatTestReshape, DISABLED_reshape)
{
    a = randomMat(size,type, -100, 100);
    a.copyTo(ua);
    if(useRoi)
    {
        int roi_shift_x = randomInt(0, size.width-1);
        int roi_shift_y = randomInt(0, size.height-1);
        roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
        Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
        ua = UMat(ua, roi).clone();
        a = Mat(a, roi).clone();
    }

    int nChannels = randomInt(1,4);

    if ((ua.cols*ua.channels()*ua.rows)%nChannels != 0)
    {
        EXPECT_ANY_THROW(ua.reshape(nChannels));
    }
    else
    {
        ub = ua.reshape(nChannels);
        ASSERT_EQ(ub.channels(),nChannels);
        ASSERT_EQ(ub.channels()*ub.cols*ub.rows, ua.channels()*ua.cols*ua.rows);

        EXPECT_MAT_NEAR(ua.reshape(nChannels), a.reshape(nChannels), 0);

        int new_rows = randomInt(1, INT_MAX);
        if ( ((int)ua.total()*ua.channels())%(new_rows*nChannels) != 0)
        {
            EXPECT_ANY_THROW (ua.reshape(nChannels, new_rows) );
        }
        else
        {
            EXPECT_NO_THROW ( ub = ua.reshape(nChannels, new_rows) );
            ASSERT_EQ(ub.channels(),nChannels);
            ASSERT_EQ(ub.rows, new_rows);
            ASSERT_EQ(ub.channels()*ub.cols*ub.rows, ua.channels()*ua.cols*ua.rows);

            EXPECT_MAT_NEAR(ua.reshape(nChannels,new_rows), a.reshape(nChannels,new_rows), 0);
        }

        new_rows = (int)ua.total()*ua.channels()/(nChannels*randomInt(1, size.width*size.height));
        if (new_rows == 0) new_rows = 1;
        int new_cols = (int)ua.total()*ua.channels()/(new_rows*nChannels);
        int sz[] = {new_rows, new_cols};
        if( ((int)ua.total()*ua.channels()) % (new_rows*new_cols) != 0 )
        {
            EXPECT_ANY_THROW( ua.reshape(nChannels, ua.dims, sz) );
        }
        else
        {
            EXPECT_NO_THROW ( ub = ua.reshape(nChannels, ua.dims, sz) );
            ASSERT_EQ(ub.channels(),nChannels);
            ASSERT_EQ(ub.rows, new_rows);
            ASSERT_EQ(ub.cols, new_cols);
            ASSERT_EQ(ub.channels()*ub.cols*ub.rows, ua.channels()*ua.cols*ua.rows);

            EXPECT_MAT_NEAR(ua.reshape(nChannels, ua.dims, sz), a.reshape(nChannels, a.dims, sz), 0);
        }
    }
}

INSTANTIATE_TEST_CASE_P(UMat, UMatTestReshape, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, UMAT_TEST_SIZES, Bool() ));

////////////////////////////////////////////////////////////////// ROI testing ///////////////////////////////////////////////////////////////

PARAM_TEST_CASE(UMatTestRoi, int, int, Size)
{
    Mat a, roi_a;
    UMat ua, roi_ua;
    int type;
    int depth;
    int cn;
    Size size;
    Size roi_size;
    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        size = GET_PARAM(2);
        type = CV_MAKE_TYPE(depth, cn);
    }
};

TEST_P(UMatTestRoi, createRoi)
{
    int roi_shift_x = randomInt(0, size.width-1);
    int roi_shift_y = randomInt(0, size.height-1);
    roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
    a = randomMat(size, type, -100, 100);
    Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
    roi_a = Mat(a, roi);
    a.copyTo(ua);
    roi_ua = UMat(ua, roi);

    EXPECT_MAT_NEAR(roi_a, roi_ua, 0);
}

TEST_P(UMatTestRoi, locateRoi)
{
    int roi_shift_x = randomInt(0, size.width-1);
    int roi_shift_y = randomInt(0, size.height-1);
    roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
    a = randomMat(size, type, -100, 100);
    Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
    roi_a = Mat(a, roi);
    a.copyTo(ua);
    roi_ua = UMat(ua,roi);
    Size sz, usz;
    Point p, up;
    roi_a.locateROI(sz, p);
    roi_ua.locateROI(usz, up);
    ASSERT_EQ(sz, usz);
    ASSERT_EQ(p, up);
}

TEST_P(UMatTestRoi, adjustRoi)
{
    int roi_shift_x = randomInt(0, size.width-1);
    int roi_shift_y = randomInt(0, size.height-1);
    roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
    a = randomMat(size, type, -100, 100);
    Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
    a.copyTo(ua);
    roi_ua = UMat( ua, roi);
    int adjLeft = randomInt(-(roi_ua.cols/2), (size.width-1)/2);
    int adjRight = randomInt(-(roi_ua.cols/2), (size.width-1)/2);
    int adjTop = randomInt(-(roi_ua.rows/2), (size.height-1)/2);
    int adjBot = randomInt(-(roi_ua.rows/2), (size.height-1)/2);
    roi_ua.adjustROI(adjTop, adjBot, adjLeft, adjRight);
    roi_shift_x = std::max(0, roi.x-adjLeft);
    roi_shift_y = std::max(0, roi.y-adjTop);
    Rect new_roi( roi_shift_x, roi_shift_y, std::min(roi.width+adjRight+adjLeft, size.width-roi_shift_x), std::min(roi.height+adjBot+adjTop, size.height-roi_shift_y) );
    UMat test_roi = UMat(ua, new_roi);
    EXPECT_MAT_NEAR(roi_ua, test_roi, 0);
}

INSTANTIATE_TEST_CASE_P(UMat, UMatTestRoi, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, UMAT_TEST_SIZES ));

/////////////////////////////////////////////////////////////// Size ////////////////////////////////////////////////////////////////////

PARAM_TEST_CASE(UMatTestSizeOperations, int, int, Size, bool)
{
    Mat a, b, roi_a, roi_b;
    UMat ua, ub, roi_ua, roi_ub;
    int type;
    int depth;
    int cn;
    Size size;
    Size roi_size;
    bool useRoi;
    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        size = GET_PARAM(2);
        useRoi = GET_PARAM(3);
        type = CV_MAKE_TYPE(depth, cn);
    }
};

TEST_P(UMatTestSizeOperations, copySize)
{
    Size s = randomSize(1,300);
    a = randomMat(size, type, -100, 100);
    b = randomMat(s, type, -100, 100);
    a.copyTo(ua);
    b.copyTo(ub);
    if(useRoi)
    {
        int roi_shift_x = randomInt(0, size.width-1);
        int roi_shift_y = randomInt(0, size.height-1);
        roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
        Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
        ua = UMat(ua,roi);

        roi_shift_x = randomInt(0, s.width-1);
        roi_shift_y = randomInt(0, s.height-1);
        roi_size = Size(s.width - roi_shift_x, s.height - roi_shift_y);
        roi = Rect(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
        ub = UMat(ub, roi);
    }
    ua.copySize(ub);
    ASSERT_EQ(ua.size, ub.size);
}

INSTANTIATE_TEST_CASE_P(UMat, UMatTestSizeOperations, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, UMAT_TEST_SIZES, Bool() ));

///////////////////////////////////////////////////////////////// UMat operations ////////////////////////////////////////////////////////////////////////////

PARAM_TEST_CASE(UMatTestUMatOperations, int, int, Size, bool)
{
    Mat a, b;
    UMat ua, ub;
    int type;
    int depth;
    int cn;
    Size size;
    Size roi_size;
    bool useRoi;
    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        size = GET_PARAM(2);
        useRoi = GET_PARAM(3);
        type = CV_MAKE_TYPE(depth, cn);
    }
};

TEST_P(UMatTestUMatOperations, diag)
{
    a = randomMat(size, type, -100, 100);
    a.copyTo(ua);
    Mat new_diag;
    if(useRoi)
    {
        int roi_shift_x = randomInt(0, size.width-1);
        int roi_shift_y = randomInt(0, size.height-1);
        roi_size = Size(size.width - roi_shift_x, size.height - roi_shift_y);
        Rect roi(roi_shift_x, roi_shift_y, roi_size.width, roi_size.height);
        ua = UMat(ua,roi);
        a = Mat(a, roi);
    }
    int n = randomInt(0, ua.cols-1);
    ub = ua.diag(n);
    b = a.diag(n);
    EXPECT_MAT_NEAR(b, ub, 0);
    new_diag = randomMat(Size(ua.rows, 1), type, -100, 100);
    new_diag.copyTo(ub);
    ua = cv::UMat::diag(ub);
    EXPECT_MAT_NEAR(ua.diag(), new_diag.t(), 0);
}

INSTANTIATE_TEST_CASE_P(UMat, UMatTestUMatOperations, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, UMAT_TEST_SIZES, Bool()));

///////////////////////////////////////////////////////////////// OpenCL ////////////////////////////////////////////////////////////////////////////

TEST(UMat, BufferPoolGrowing)
{
#ifdef _DEBUG
    const int ITERATIONS = 100;
#else
    const int ITERATIONS = 200;
#endif
    const Size sz(1920, 1080);
    BufferPoolController* c = cv::ocl::getOpenCLAllocator()->getBufferPoolController();
    if (c)
    {
        size_t oldMaxReservedSize = c->getMaxReservedSize();
        c->freeAllReservedBuffers();
        c->setMaxReservedSize(sz.area() * 10);
        for (int i = 0; i < ITERATIONS; i++)
        {
            UMat um(Size(sz.width + i, sz.height + i), CV_8UC1);
            UMat um2(Size(sz.width + 2 * i, sz.height + 2 * i), CV_8UC1);
        }
        c->setMaxReservedSize(oldMaxReservedSize);
        c->freeAllReservedBuffers();
    }
    else
        std::cout << "Skipped, no OpenCL" << std::endl;
}

class CV_UMatTest :
        public cvtest::BaseTest
{
public:
    CV_UMatTest() {}
    ~CV_UMatTest() {}
protected:
    void run(int);

    struct test_excep
    {
        test_excep(const string& _s=string("")) : s(_s) { }
        string s;
    };

    bool TestUMat();

    void checkDiff(const Mat& m1, const Mat& m2, const string& s)
    {
        if (cvtest::norm(m1, m2, NORM_INF) != 0)
            throw test_excep(s);
    }
    void checkDiffF(const Mat& m1, const Mat& m2, const string& s)
    {
        if (cvtest::norm(m1, m2, NORM_INF) > 1e-5)
            throw test_excep(s);
    }
};

#define STR(a) STR2(a)
#define STR2(a) #a

#define CHECK_DIFF(a, b) checkDiff(a, b, "(" #a ")  !=  (" #b ")  at l." STR(__LINE__))
#define CHECK_DIFF_FLT(a, b) checkDiffF(a, b, "(" #a ")  !=(eps)  (" #b ")  at l." STR(__LINE__))


bool CV_UMatTest::TestUMat()
{
    try
    {
        Mat a(100, 100, CV_16SC2), b, c;
        randu(a, Scalar::all(-100), Scalar::all(100));
        Rect roi(1, 3, 5, 4);
        Mat ra(a, roi), rb, rc, rc0;
        UMat ua, ura, ub, urb, uc, urc;
        a.copyTo(ua);
        ua.copyTo(b);
        CHECK_DIFF(a, b);

        ura = ua(roi);
        ura.copyTo(rb);

        CHECK_DIFF(ra, rb);

        ra += Scalar::all(1.f);
        {
            Mat temp = ura.getMat(ACCESS_RW);
            temp += Scalar::all(1.f);
        }
        ra.copyTo(rb);
        CHECK_DIFF(ra, rb);

        b = a.clone();
        ra = a(roi);
        rb = b(roi);
        randu(b, Scalar::all(-100), Scalar::all(100));
        b.copyTo(ub);
        urb = ub(roi);

        /*std::cout << "==============================================\nbefore op (CPU):\n";
        std::cout << "ra: " << ra << std::endl;
        std::cout << "rb: " << rb << std::endl;*/

        ra.copyTo(ura);
        rb.copyTo(urb);
        ra.release();
        rb.release();
        ura.copyTo(ra);
        urb.copyTo(rb);

        /*std::cout << "==============================================\nbefore op (GPU):\n";
        std::cout << "ra: " << ra << std::endl;
        std::cout << "rb: " << rb << std::endl;*/

        cv::max(ra, rb, rc);
        cv::max(ura, urb, urc);
        urc.copyTo(rc0);

        /*std::cout << "==============================================\nafter op:\n";
        std::cout << "rc: " << rc << std::endl;
        std::cout << "rc0: " << rc0 << std::endl;*/

        CHECK_DIFF(rc0, rc);

        {
            UMat tmp = rc0.getUMat(ACCESS_WRITE);
            cv::max(ura, urb, tmp);
        }
        CHECK_DIFF(rc0, rc);

        ura.copyTo(urc);
        cv::max(urc, urb, urc);
        urc.copyTo(rc0);
        CHECK_DIFF(rc0, rc);

        rc = ra ^ rb;
        cv::bitwise_xor(ura, urb, urc);
        urc.copyTo(rc0);

        /*std::cout << "==============================================\nafter op:\n";
        std::cout << "ra: " << rc0 << std::endl;
        std::cout << "rc: " << rc << std::endl;*/

        CHECK_DIFF(rc0, rc);

        rc = ra + rb;
        cv::add(ura, urb, urc);
        urc.copyTo(rc0);

        CHECK_DIFF(rc0, rc);

        cv::subtract(ra, Scalar::all(5), rc);
        cv::subtract(ura, Scalar::all(5), urc);
        urc.copyTo(rc0);

        CHECK_DIFF(rc0, rc);
    }
    catch (const test_excep& e)
    {
        ts->printf(cvtest::TS::LOG, "%s\n", e.s.c_str());
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return false;
    }
    return true;
}

void CV_UMatTest::run( int /* start_from */)
{
    printf("Use OpenCL: %s\nHave OpenCL: %s\n",
           cv::ocl::useOpenCL() ? "TRUE" : "FALSE",
           cv::ocl::haveOpenCL() ? "TRUE" : "FALSE" );

    if (!TestUMat())
        return;

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Core_UMat, base) { CV_UMatTest test; test.safe_run(); }

TEST(Core_UMat, getUMat)
{
    {
        int a[3] = { 1, 2, 3 };
        Mat m = Mat(1, 1, CV_32SC3, a);
        UMat u = m.getUMat(ACCESS_READ);
        EXPECT_NE((void*)NULL, u.u);
    }

    {
        Mat m(10, 10, CV_8UC1), ref;
        for (int y = 0; y < m.rows; ++y)
        {
            uchar * const ptr = m.ptr<uchar>(y);
            for (int x = 0; x < m.cols; ++x)
                ptr[x] = (uchar)(x + y * 2);
        }

        ref = m.clone();
        Rect r(1, 1, 8, 8);
        ref(r).setTo(17);

        {
            UMat um = m(r).getUMat(ACCESS_WRITE);
            um.setTo(17);
        }

        double err = cvtest::norm(m, ref, NORM_INF);
        if (err > 0)
        {
            std::cout << "m: " << std::endl << m << std::endl;
            std::cout << "ref: " << std::endl << ref << std::endl;
        }
        EXPECT_EQ(0., err);
    }
}

TEST(UMat, Sync)
{
    UMat um(10, 10, CV_8UC1);

    {
        Mat m = um.getMat(ACCESS_WRITE);
        m.setTo(cv::Scalar::all(17));
    }

    um.setTo(cv::Scalar::all(19));

    EXPECT_EQ(0, cvtest::norm(um.getMat(ACCESS_READ), cv::Mat(um.size(), um.type(), 19), NORM_INF));
}

TEST(UMat, setOpenCL)
{
    // save the current state
    bool useOCL = cv::ocl::useOpenCL();

    Mat m = (Mat_<uchar>(3,3)<<0,1,2,3,4,5,6,7,8);

    cv::ocl::setUseOpenCL(true);
    UMat um1;
    m.copyTo(um1);

    cv::ocl::setUseOpenCL(false);
    UMat um2;
    m.copyTo(um2);

    cv::ocl::setUseOpenCL(true);
    countNonZero(um1);
    countNonZero(um2);

    um1.copyTo(um2);
    EXPECT_MAT_NEAR(um1, um2, 0);
    EXPECT_MAT_NEAR(um1, m, 0);
    um2.copyTo(um1);
    EXPECT_MAT_NEAR(um1, m, 0);
    EXPECT_MAT_NEAR(um1, um2, 0);

    cv::ocl::setUseOpenCL(false);
    countNonZero(um1);
    countNonZero(um2);

    um1.copyTo(um2);
    EXPECT_MAT_NEAR(um1, um2, 0);
    EXPECT_MAT_NEAR(um1, m, 0);
    um2.copyTo(um1);
    EXPECT_MAT_NEAR(um1, um2, 0);
    EXPECT_MAT_NEAR(um1, m, 0);

    // reset state to the previous one
    cv::ocl::setUseOpenCL(useOCL);
}

TEST(UMat, ReadBufferRect)
{
    UMat m(1, 10000, CV_32FC2, Scalar::all(-1));
    Mat t(1, 9000, CV_32FC2, Scalar::all(-200)), t2(1, 9000, CV_32FC2, Scalar::all(-1));
    m.colRange(0, 9000).copyTo(t);

    EXPECT_MAT_NEAR(t, t2, 0);
}

// Use iGPU or OPENCV_OPENCL_DEVICE=:CPU: to catch problem
TEST(UMat, DISABLED_synchronization_map_unmap)
{
    class TestParallelLoopBody : public cv::ParallelLoopBody
    {
        UMat u_;
    public:
        TestParallelLoopBody(const UMat& u) : u_(u) { }
        void operator() (const cv::Range& range) const
        {
            printf("range: %d, %d -- begin\n", range.start, range.end);
            for (int i = 0; i < 10; i++)
            {
                printf("%d: %d map...\n", range.start, i);
                Mat m = u_.getMat(cv::ACCESS_READ);

                printf("%d: %d unmap...\n", range.start, i);
                m.release();
            }
            printf("range: %d, %d -- end\n", range.start, range.end);
        }
    };
    try
    {
        UMat u(1000, 1000, CV_32FC1);
        parallel_for_(cv::Range(0, 2), TestParallelLoopBody(u));
    }
    catch (const cv::Exception& e)
    {
        FAIL() << "Exception: " << e.what();
        ADD_FAILURE();
    }
    catch (...)
    {
        FAIL() << "Exception!";
    }
}

} } // namespace cvtest::ocl

TEST(UMat, DISABLED_bug_with_unmap)
{
    for (int i = 0; i < 20; i++)
    {
        try
        {
            Mat m = Mat(1000, 1000, CV_8UC1);
            UMat u = m.getUMat(ACCESS_READ);
            UMat dst;
            add(u, Scalar::all(0), dst); // start async operation
            u.release();
            m.release();
        }
        catch (const cv::Exception& e)
        {
            printf("i = %d... %s\n", i, e.what());
            ADD_FAILURE();
        }
        catch (...)
        {
            printf("i = %d...\n", i);
            ADD_FAILURE();
        }
    }
}

TEST(UMat, DISABLED_bug_with_unmap_in_class)
{
    class Logic
    {
    public:
        Logic() {}
        void processData(InputArray input)
        {
            Mat m = input.getMat();
            {
                Mat dst;
                m.convertTo(dst, CV_32FC1);
                // some additional CPU-based per-pixel processing into dst
                intermediateResult = dst.getUMat(ACCESS_READ);
                std::cout << "data processed..." << std::endl;
            } // problem is here: dst::~Mat()
            std::cout << "leave ProcessData()" << std::endl;
        }
        UMat getResult() const { return intermediateResult; }
    protected:
        UMat intermediateResult;
    };
    try
    {
        Mat m = Mat(1000, 1000, CV_8UC1);
        Logic l;
        l.processData(m);
        UMat result = l.getResult();
    }
    catch (const cv::Exception& e)
    {
        printf("exception... %s\n", e.what());
        ADD_FAILURE();
    }
    catch (...)
    {
        printf("exception... \n");
        ADD_FAILURE();
    }
}

TEST(UMat, Test_same_behaviour_read_and_read)
{
    bool exceptionDetected = false;
    try
    {
        UMat u(Size(10, 10), CV_8UC1);
        Mat m = u.getMat(ACCESS_READ);
        UMat dst;
        add(u, Scalar::all(1), dst);
    }
    catch (...)
    {
        exceptionDetected = true;
    }
    ASSERT_FALSE(exceptionDetected); // no data race, 2+ reads are valid
}

// VP: this test (and probably others from same_behaviour series) is not valid in my opinion.
TEST(UMat, DISABLED_Test_same_behaviour_read_and_write)
{
    bool exceptionDetected = false;
    try
    {
        UMat u(Size(10, 10), CV_8UC1);
        Mat m = u.getMat(ACCESS_READ);
        add(u, Scalar::all(1), u);
    }
    catch (...)
    {
        exceptionDetected = true;
    }
    ASSERT_TRUE(exceptionDetected); // data race
}

TEST(UMat, DISABLED_Test_same_behaviour_write_and_read)
{
    bool exceptionDetected = false;
    try
    {
        UMat u(Size(10, 10), CV_8UC1);
        Mat m = u.getMat(ACCESS_WRITE);
        UMat dst;
        add(u, Scalar::all(1), dst);
    }
    catch (...)
    {
        exceptionDetected = true;
    }
    ASSERT_TRUE(exceptionDetected); // data race
}

TEST(UMat, DISABLED_Test_same_behaviour_write_and_write)
{
    bool exceptionDetected = false;
    try
    {
        UMat u(Size(10, 10), CV_8UC1);
        Mat m = u.getMat(ACCESS_WRITE);
        add(u, Scalar::all(1), u);
    }
    catch (...)
    {
        exceptionDetected = true;
    }
    ASSERT_TRUE(exceptionDetected); // data race
}
