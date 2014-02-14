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
#include "opencv2/core/ocl.hpp"

using namespace cvtest;
using namespace testing;
using namespace cv;

#define EXPECT_MAT_NEAR(mat1, mat2, eps) \
{ \
   ASSERT_EQ(mat1.type(), mat2.type()); \
   ASSERT_EQ(mat1.size(), mat2.size()); \
   EXPECT_LE(cv::norm(mat1, mat2), eps); \
}\

////////////////////////////////////////////////////////////// Basic Tests /////////////////////////////////////////////////////////////////////

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

CORE_TEST_P(UMatBasicTests, createUMat)
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

CORE_TEST_P(UMatBasicTests, swap)
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

CORE_TEST_P(UMatBasicTests, base)
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

CORE_TEST_P(UMatBasicTests, copyTo)
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

CORE_TEST_P(UMatBasicTests, DISABLED_GetUMat)
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
    testing::Values(cv::Size(1,1), cv::Size(1,128), cv::Size(128,1), cv::Size(128, 128), cv::Size(640,480)), Bool() ) );

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

CORE_TEST_P(UMatTestReshape, reshape)
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

INSTANTIATE_TEST_CASE_P(UMat, UMatTestReshape, Combine(UMAT_TEST_DEPTH, UMAT_TEST_CHANNELS, UMAT_TEST_SIZES, Bool() ));

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

CORE_TEST_P(UMatTestRoi, createRoi)
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

CORE_TEST_P(UMatTestRoi, locateRoi)
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

CORE_TEST_P(UMatTestRoi, adjustRoi)
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
    roi_shift_x = max(0, roi.x-adjLeft);
    roi_shift_y = max(0, roi.y-adjTop);
    Rect new_roi( roi_shift_x, roi_shift_y, min(roi.width+adjRight+adjLeft, size.width-roi_shift_x), min(roi.height+adjBot+adjTop, size.height-roi_shift_y) );
    UMat test_roi = UMat(ua, new_roi);
    EXPECT_MAT_NEAR(roi_ua, test_roi, 0);
}

INSTANTIATE_TEST_CASE_P(UMat, UMatTestRoi, Combine(UMAT_TEST_DEPTH, UMAT_TEST_CHANNELS, UMAT_TEST_SIZES ));

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

CORE_TEST_P(UMatTestSizeOperations, copySize)
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

INSTANTIATE_TEST_CASE_P(UMat, UMatTestSizeOperations, Combine(UMAT_TEST_DEPTH, UMAT_TEST_CHANNELS, UMAT_TEST_SIZES, Bool() ));

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

CORE_TEST_P(UMatTestUMatOperations, diag)
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

INSTANTIATE_TEST_CASE_P(UMat, UMatTestUMatOperations, Combine(UMAT_TEST_DEPTH, UMAT_TEST_CHANNELS, UMAT_TEST_SIZES, Bool() ));

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
    {
        std::cout << "Skipped, no OpenCL" << std::endl;
    }
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
