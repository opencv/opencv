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

using namespace opencv_test;
using namespace testing;
using namespace cv;

namespace opencv_test {
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
    const int align_mask = 3;
    roi.x &= ~align_mask;
    roi.y &= ~align_mask;
    roi.width = (roi.width + align_mask) & ~align_mask;
    roi &= Rect(0, 0, ua.cols, ua.rows);

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

TEST_P(UMatBasicTests, copyTo)
{
    int i;
    if(useRoi)
    {
        UMat roi_ua;
        Mat roi_a;
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

TEST_P(UMatBasicTests, GetUMat)
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
        UMat u = a.getUMat(ACCESS_RW);
        {
            Mat b = u.getMat(ACCESS_RW);
            EXPECT_MAT_NEAR(b, a, 0);
        }
    }
    {
        Mat b;
        b = ua.getMat(ACCESS_RW);
        EXPECT_MAT_NEAR(b, a, 0);
    }
    {
        Mat m = ua.getMat(ACCESS_RW);
        {
            UMat ub = m.getUMat(ACCESS_RW);
            EXPECT_MAT_NEAR(ub, ua, 0);
        }
    }
}

INSTANTIATE_TEST_CASE_P(UMat, UMatBasicTests, Combine(testing::Values(CV_8U, CV_64F), testing::Values(1, 2),
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

TEST_P(UMatTestReshape, reshape)
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

static void check_ndim_shape(const cv::UMat &mat, int cn, int ndims, const int *sizes)
{
    EXPECT_EQ(mat.channels(), cn);
    EXPECT_EQ(mat.dims, ndims);

    if (mat.dims != ndims)
        return;

    for (int i = 0; i < ndims; i++)
        EXPECT_EQ(mat.size[i], sizes[i]);
}

TEST(UMatTestReshape, reshape_ndims_2)
{
    const cv::UMat A(8, 16, CV_8UC3);
    cv::UMat B;

    {
        int new_sizes_mask[] = { 0, 3, 4, 4 };
        int new_sizes_real[] = { 8, 3, 4, 4 };
        ASSERT_NO_THROW(B = A.reshape(1, 4, new_sizes_mask));
        check_ndim_shape(B, 1, 4, new_sizes_real);
    }
    {
        int new_sizes[] = { 16, 8 };
        ASSERT_NO_THROW(B = A.reshape(0, 2, new_sizes));
        check_ndim_shape(B, 3, 2, new_sizes);
        EXPECT_EQ(B.rows, new_sizes[0]);
        EXPECT_EQ(B.cols, new_sizes[1]);
    }
    {
        int new_sizes[] = { 2, 5, 1, 3 };
        cv::UMat A_sliced = A(cv::Range::all(), cv::Range(0, 15));
        ASSERT_ANY_THROW(A_sliced.reshape(4, 4, new_sizes));
    }
}

TEST(UMatTestReshape, reshape_ndims_4)
{
    const int sizes[] = { 2, 6, 4, 12 };
    const cv::UMat A(4, sizes, CV_8UC3);
    cv::UMat B;

    {
        int new_sizes_mask[] = { 0, 864 };
        int new_sizes_real[] = { 2, 864 };
        ASSERT_NO_THROW(B = A.reshape(1, 2, new_sizes_mask));
        check_ndim_shape(B, 1, 2, new_sizes_real);
        EXPECT_EQ(B.rows, new_sizes_real[0]);
        EXPECT_EQ(B.cols, new_sizes_real[1]);
    }
    {
        int new_sizes_mask[] = { 4, 0, 0, 2, 3 };
        int new_sizes_real[] = { 4, 6, 4, 2, 3 };
        ASSERT_NO_THROW(B = A.reshape(0, 5, new_sizes_mask));
        check_ndim_shape(B, 3, 5, new_sizes_real);
    }
    {
        int new_sizes_mask[] = { 1, 1 };
        ASSERT_ANY_THROW(A.reshape(0, 2, new_sizes_mask));
    }
    {
        int new_sizes_mask[] = { 4, 6, 3, 3, 0 };
        ASSERT_ANY_THROW(A.reshape(0, 5, new_sizes_mask));
    }
}

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

TEST(UMatTestRoi, adjustRoiOverflow)
{
    UMat m(15, 10, CV_32S);
    UMat roi(m, cv::Range(2, 10), cv::Range(3,6));
    int rowsInROI = roi.rows;
    roi.adjustROI(1, 0, 0, 0);

    ASSERT_EQ(roi.rows, rowsInROI + 1);

    roi.adjustROI(-m.rows, -m.rows, 0, 0);

    ASSERT_EQ(roi.rows, m.rows);
}

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


/////////////////////////////////////////////////////////////// getUMat -> GetMat ///////////////////////////////////////////////////////////////////

PARAM_TEST_CASE(getUMat, int, int, Size, bool)
{
    int type;
    Size size;

    virtual void SetUp()
    {
        int depth = GET_PARAM(0);
        int cn    = GET_PARAM(1);
        size      = GET_PARAM(2);
        useOpenCL = GET_PARAM(3);

        type = CV_MAKE_TYPE(depth, cn);

        isOpenCL_enabled = cv::ocl::useOpenCL();
        cv::ocl::setUseOpenCL(useOpenCL);
    }

    virtual void TearDown()
    {
        cv::ocl::setUseOpenCL(isOpenCL_enabled);
    }

    // UMat created from user allocated host memory (USE_HOST_PTR)
    void custom_ptr_test(size_t align_base, size_t align_offset)
    {
        void* pData_allocated = new unsigned char [size.area() * CV_ELEM_SIZE(type) + (align_base + align_offset)];
        void* pData = (char*)alignPtr(pData_allocated, (int)align_base) + align_offset;
        size_t step = size.width * CV_ELEM_SIZE(type);

        {
            Mat m = Mat(size, type, pData, step);
            m.setTo(cv::Scalar::all(2));

            UMat u = m.getUMat(ACCESS_RW);
            cv::add(u, cv::Scalar::all(2), u);

            Mat d = u.getMat(ACCESS_READ);

            Mat expected(m.size(), m.type(), cv::Scalar::all(4));
            double norm = cvtest::norm(d, expected, NORM_INF);

            EXPECT_EQ(0, norm);
        }

        delete[] (unsigned char*)pData_allocated;
    }

private:
    bool useOpenCL;
    bool isOpenCL_enabled;
};

TEST_P(getUMat, custom_ptr_align_4Kb)
{
    custom_ptr_test(4096, 0);
}

TEST_P(getUMat, custom_ptr_align_64b)
{
    custom_ptr_test(4096, 64);
}

TEST_P(getUMat, custom_ptr_align_none)
{
    custom_ptr_test(4096, cv::alignSize(CV_ELEM_SIZE(type), 4));
}

TEST_P(getUMat, self_allocated)
{
    Mat m = Mat(size, type);
    m.setTo(cv::Scalar::all(2));

    UMat u = m.getUMat(ACCESS_RW);
    cv::add(u, cv::Scalar::all(2), u);

    Mat d = u.getMat(ACCESS_READ);

    Mat expected(m.size(), m.type(), cv::Scalar::all(4));
    double norm = cvtest::norm(d, expected, NORM_INF);

    EXPECT_EQ(0, norm);
}

INSTANTIATE_TEST_CASE_P(UMat, getUMat, Combine(
        Values(CV_8U, CV_64F), // depth
        Values(1, 3), // channels
        Values(cv::Size(1, 1), cv::Size(255, 255), cv::Size(256, 256)), // Size
        Bool() // useOpenCL
));



///////////////////////////////////////////////////////////////// OpenCL ////////////////////////////////////////////////////////////////////////////

#ifdef HAVE_OPENCL
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
#endif

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

TEST(UMat, SyncTemp)
{
    Mat m(10, 10, CV_8UC1);

    {
        UMat um = m.getUMat(ACCESS_WRITE);

        {
            Mat m2 = um.getMat(ACCESS_WRITE);
            m2.setTo(cv::Scalar::all(17));
        }

        um.setTo(cv::Scalar::all(19));

        EXPECT_EQ(0, cvtest::norm(um.getMat(ACCESS_READ), cv::Mat(um.size(), um.type(), 19), NORM_INF));
    }
}

TEST(UMat, CopyToIfDeviceCopyIsObsolete)
{
    UMat um(7, 2, CV_8UC1);
    Mat m(um.size(), um.type());
    m.setTo(Scalar::all(0));

    {
        // make obsolete device copy of UMat
        Mat temp = um.getMat(ACCESS_WRITE);
        temp.setTo(Scalar::all(10));
    }

    m.copyTo(um);
    um.setTo(Scalar::all(17));

    EXPECT_EQ(0, cvtest::norm(um.getMat(ACCESS_READ), Mat(um.size(), um.type(), 17), NORM_INF));
}

TEST(UMat, setOpenCL)
{
#ifndef HAVE_OPENCL
    return; // test skipped
#else
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
#endif
}

TEST(UMat, ReadBufferRect)
{
    UMat m(1, 10000, CV_32FC2, Scalar::all(-1));
    Mat t(1, 9000, CV_32FC2, Scalar::all(-200)), t2(1, 9000, CV_32FC2, Scalar::all(-1));
    m.colRange(0, 9000).copyTo(t);

    EXPECT_MAT_NEAR(t, t2, 0);
}


// Use iGPU or OPENCV_OPENCL_DEVICE=:CPU: to catch problem
TEST(UMat, synchronization_map_unmap)
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
        UMat u(1000, 1000, CV_32FC1, Scalar::all(0));
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


TEST(UMat, async_unmap)
{
    for (int i = 0; i < 20; i++)
    {
        try
        {
            Mat m = Mat(1000, 1000, CV_8UC1, Scalar::all(0));
            UMat u = m.getUMat(ACCESS_READ);
            UMat dst;
            cv::add(u, Scalar::all(0), dst); // start async operation
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


TEST(UMat, unmap_in_class)
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
                intermediateResult = dst.getUMat(ACCESS_READ); // this violates lifetime of base(dst) / derived (intermediateResult) objects. Use copyTo?
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
        Mat m = Mat(1000, 1000, CV_8UC1, Scalar::all(0));
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


TEST(UMat, map_unmap_counting)
{
    if (!cv::ocl::useOpenCL())
    {
        std::cout << "OpenCL is not enabled. Skip test" << std::endl;
        return;
    }
    std::cout << "Host memory: " << cv::ocl::Device::getDefault().hostUnifiedMemory() << std::endl;
    Mat m(Size(10, 10), CV_8UC1, Scalar::all(0));
    UMat um = m.getUMat(ACCESS_RW);
    {
        Mat d1 = um.getMat(ACCESS_RW);
        Mat d2 = um.getMat(ACCESS_RW);
        d1.release();
    }
    void* h = NULL;
    EXPECT_NO_THROW(h = um.handle(ACCESS_RW));
    std::cout << "Handle: " << h << std::endl;
}


static void process_with_async_cleanup(Mat& frame)
{
    UMat blurResult;
    {
        UMat umat_buffer = frame.getUMat(ACCESS_READ);
        cv::blur(umat_buffer, blurResult, Size(3, 3));  // UMat doesn't support inplace, this call is not synchronized
    }
    Mat result;
    blurResult.copyTo(result);
    swap(result, frame);
    // umat_buffer cleanup is done asynchronously, silence warning about original 'frame' cleanup here (through 'result')
    // - release input 'frame' (as 'result')
    // - release 'umat_buffer' asynchronously and silence warning about "parent" buffer (in debug builds)
}
TEST(UMat, async_cleanup_without_call_chain_warning)
{
    Mat frame(Size(640, 480), CV_8UC1, Scalar::all(128));
    for (int i = 0; i < 10; i++)
    {
        process_with_async_cleanup(frame);
    }
}


///////////// oclCleanupCallback threadsafe check (#5062) /////////////////////

// Case 1: reuse of old src Mat in OCL pipe. Hard to catch!
OCL_TEST(UMat, DISABLED_OCL_ThreadSafe_CleanupCallback_1_VeryLongTest)
{
    if (!cv::ocl::useOpenCL())
    {
        std::cout << "OpenCL is not enabled. Skip test" << std::endl;
        return;
    }
    for (int j = 0; j < 100; j++)
    {
        const Size srcSize(320, 240);
        const int type = CV_8UC1;
        const int dtype = CV_16UC1;

        Mat src(srcSize, type, Scalar::all(0));
        Mat dst_ref(srcSize, dtype);

        // Generate reference data as additional check
        OCL_OFF(src.convertTo(dst_ref, dtype));
        cv::ocl::setUseOpenCL(true); // restore OpenCL state

        UMat dst(srcSize, dtype);

        // Use multiple iterations to increase chance of data race catching
        for(int k = 0; k < 10000; k++)
        {
            UMat tmpUMat = src.getUMat(ACCESS_RW);
            tmpUMat.convertTo(dst, dtype);
            ::cv::ocl::finish(); // force kernel to complete to start cleanup sooner
        }

        EXPECT_MAT_NEAR(dst_ref, dst, 1);
        printf(".\n"); fflush(stdout);
    }
}

// Case 2: concurrent deallocation of UMatData between UMat and Mat deallocators. Hard to catch!
OCL_TEST(UMat, DISABLED_OCL_ThreadSafe_CleanupCallback_2_VeryLongTest)
{
    if (!cv::ocl::useOpenCL())
    {
        std::cout << "OpenCL is not enabled. Skip test" << std::endl;
        return;
    }
    for (int j = 0; j < 100; j++)
    {
        const Size srcSize(320, 240);
        const int type = CV_8UC1;
        const int dtype = CV_16UC1;

        // This test is only relevant for OCL
        UMat dst(srcSize, dtype);

        // Use multiple iterations to increase chance of data race catching
        for(int k = 0; k < 10000; k++)
        {
            Mat src(srcSize, type, Scalar::all(0)); // Declare src inside loop now to catch its destruction on stack
            {
                UMat tmpUMat = src.getUMat(ACCESS_RW);
                tmpUMat.convertTo(dst, dtype);
            }
            ::cv::ocl::finish(); // force kernel to complete to start cleanup sooner
        }
        printf(".\n"); fflush(stdout);
    }
}



TEST(UMat, DISABLED_Test_same_behaviour_read_and_read)
{
    bool exceptionDetected = false;
    try
    {
        UMat u(Size(10, 10), CV_8UC1, Scalar::all(0));
        Mat m = u.getMat(ACCESS_READ);
        UMat dst;
        cv::add(u, Scalar::all(1), dst);
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
        UMat u(Size(10, 10), CV_8UC1, Scalar::all(0));
        Mat m = u.getMat(ACCESS_READ);
        cv::add(u, Scalar::all(1), u);
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
        UMat u(Size(10, 10), CV_8UC1, Scalar::all(0));
        Mat m = u.getMat(ACCESS_WRITE);
        UMat dst;
        cv::add(u, Scalar::all(1), dst);
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
        UMat u(Size(10, 10), CV_8UC1, Scalar::all(0));
        Mat m = u.getMat(ACCESS_WRITE);
        cv::add(u, Scalar::all(1), u);
    }
    catch (...)
    {
        exceptionDetected = true;
    }
    ASSERT_TRUE(exceptionDetected); // data race
}

TEST(UMat, mat_umat_sync)
{
    UMat u(10, 10, CV_8UC1, Scalar(1));
    {
        Mat m = u.getMat(ACCESS_RW).reshape(1);
        m.setTo(Scalar(255));
    }

    UMat uDiff;
    cv::compare(u, 255, uDiff, CMP_NE);
    ASSERT_EQ(0, countNonZero(uDiff));
}

TEST(UMat, testTempObjects_UMat)
{
    UMat u(10, 10, CV_8UC1, Scalar(1));
    {
        UMat u2 = u.getMat(ACCESS_RW).getUMat(ACCESS_RW);
        u2.setTo(Scalar(255));
    }

    UMat uDiff;
    cv::compare(u, 255, uDiff, CMP_NE);
    ASSERT_EQ(0, countNonZero(uDiff));
}

TEST(UMat, testTempObjects_Mat)
{
    Mat m(10, 10, CV_8UC1, Scalar(1));
    {
        Mat m2;
        ASSERT_ANY_THROW({
          // Below is unwrapped version of this invalid expression:
          // m2 = m.getUMat(ACCESS_RW).getMat(ACCESS_RW)
          UMat u = m.getUMat(ACCESS_RW);
          m2 = u.getMat(ACCESS_RW);
          u.release();
        });
    }
}

TEST(UMat, testWrongLifetime_UMat)
{
    UMat u(10, 10, CV_8UC1, Scalar(1));
    {
        UMat u2 = u.getMat(ACCESS_RW).getUMat(ACCESS_RW);
        u.release(); // base object
        u2.release(); // derived object, should show warning message
    }
}

TEST(UMat, testWrongLifetime_Mat)
{
    Mat m(10, 10, CV_8UC1, Scalar(1));
    {
        UMat u = m.getUMat(ACCESS_RW);
        Mat m2 = u.getMat(ACCESS_RW);
        m.release(); // base object
        m2.release(); // map of derived object
        u.release(); // derived object, should show warning message
    }
}

TEST(UMat, DISABLED_regression_5991)
{
    int sz[] = {2,3,2};
    UMat mat(3, sz, CV_32F, Scalar(1));
    ASSERT_NO_THROW(mat.convertTo(mat, CV_8U));
    EXPECT_EQ(sz[0], mat.size[0]);
    EXPECT_EQ(sz[1], mat.size[1]);
    EXPECT_EQ(sz[2], mat.size[2]);
    EXPECT_EQ(0, cvtest::norm(mat.getMat(ACCESS_READ), Mat(3, sz, CV_8U, Scalar(1)), NORM_INF));
}

TEST(UMat, testTempObjects_Mat_issue_8693)
{
    UMat srcUMat(3, 4, CV_32FC1);
    Mat srcMat;

    randu(srcUMat, -1.f, 1.f);
    srcUMat.copyTo(srcMat);

    reduce(srcUMat, srcUMat, 0, CV_REDUCE_SUM);
    reduce(srcMat, srcMat, 0, CV_REDUCE_SUM);

    srcUMat.convertTo(srcUMat, CV_64FC1);
    srcMat.convertTo(srcMat, CV_64FC1);

    EXPECT_EQ(0, cvtest::norm(srcUMat.getMat(ACCESS_READ), srcMat, NORM_INF));
}

TEST(UMat, resize_Mat_issue_13577)
{
    // save the current state
    bool useOCL = cv::ocl::useOpenCL();

    cv::ocl::setUseOpenCL(false);
    UMat foo(10, 10, CV_32FC1);
    cv::resize(foo, foo, cv::Size(), .5, .5);

    cv::ocl::setUseOpenCL(useOCL);  // restore state
}

TEST(UMat, exceptions_refcounts_issue_20594)
{
    if (!cv::ocl::useOpenCL())
    {
        // skip test, difficult to create exception scenario without OpenCL
        std::cout << "OpenCL is not enabled. Skip test" << std::endl;
        return;
    }

    UMat umat1(10, 10, CV_8UC1);
    EXPECT_EQ(0, umat1.u->refcount);

    // cause exception in underlying allocator
    void* const original_handle = umat1.u->handle;
    umat1.u->handle = NULL;
    try
    {
        Mat mat1 = umat1.getMat(ACCESS_RW);
    }
    catch (...)
    {
        // nothing
    }

    // check for correct refcount, and no change of intentional bad handle
    EXPECT_EQ(0, umat1.u->refcount);
    EXPECT_EQ(NULL, umat1.u->handle);

    // reset UMat to good state
    umat1.u->refcount = 0;
    umat1.u->handle = original_handle;
}

} } // namespace opencv_test::ocl
