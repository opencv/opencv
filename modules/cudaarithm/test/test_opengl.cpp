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

#if defined(HAVE_CUDA) && defined(HAVE_OPENGL)

#include "opencv2/core/cuda.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/ts/cuda_test.hpp"

using namespace cvtest;

/////////////////////////////////////////////
// Buffer

PARAM_TEST_CASE(Buffer, cv::Size, MatType)
{
    static void SetUpTestCase()
    {
        cv::namedWindow("test", cv::WINDOW_OPENGL);
    }

    static void TearDownTestCase()
    {
        cv::destroyAllWindows();
    }

    cv::Size size;
    int type;

    virtual void SetUp()
    {
        size = GET_PARAM(0);
        type = GET_PARAM(1);
    }
};

CUDA_TEST_P(Buffer, Constructor1)
{
    cv::ogl::Buffer buf(size.height, size.width, type, cv::ogl::Buffer::ARRAY_BUFFER, true);

    EXPECT_EQ(size.height, buf.rows());
    EXPECT_EQ(size.width, buf.cols());
    EXPECT_EQ(type, buf.type());
}

CUDA_TEST_P(Buffer, Constructor2)
{
    cv::ogl::Buffer buf(size, type, cv::ogl::Buffer::ARRAY_BUFFER, true);

    EXPECT_EQ(size.height, buf.rows());
    EXPECT_EQ(size.width, buf.cols());
    EXPECT_EQ(type, buf.type());
}

CUDA_TEST_P(Buffer, ConstructorFromMat)
{
    cv::Mat gold = randomMat(size, type);

    cv::ogl::Buffer buf(gold, cv::ogl::Buffer::ARRAY_BUFFER, true);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

CUDA_TEST_P(Buffer, ConstructorFromGpuMat)
{
    cv::Mat gold = randomMat(size, type);
    cv::cuda::GpuMat d_gold(gold);

    cv::ogl::Buffer buf(d_gold, cv::ogl::Buffer::ARRAY_BUFFER);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

CUDA_TEST_P(Buffer, ConstructorFromBuffer)
{
    cv::ogl::Buffer buf_gold(size, type, cv::ogl::Buffer::ARRAY_BUFFER, true);

    cv::ogl::Buffer buf(buf_gold);

    EXPECT_EQ(buf_gold.bufId(), buf.bufId());
    EXPECT_EQ(buf_gold.rows(), buf.rows());
    EXPECT_EQ(buf_gold.cols(), buf.cols());
    EXPECT_EQ(buf_gold.type(), buf.type());
}

CUDA_TEST_P(Buffer, Create)
{
    cv::ogl::Buffer buf;
    buf.create(size.height, size.width, type, cv::ogl::Buffer::ARRAY_BUFFER, true);

    EXPECT_EQ(size.height, buf.rows());
    EXPECT_EQ(size.width, buf.cols());
    EXPECT_EQ(type, buf.type());
}

CUDA_TEST_P(Buffer, CopyFromMat)
{
    cv::Mat gold = randomMat(size, type);

    cv::ogl::Buffer buf;
    buf.copyFrom(gold, cv::ogl::Buffer::ARRAY_BUFFER, true);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

CUDA_TEST_P(Buffer, CopyFromGpuMat)
{
    cv::Mat gold = randomMat(size, type);
    cv::cuda::GpuMat d_gold(gold);

    cv::ogl::Buffer buf;
    buf.copyFrom(d_gold, cv::ogl::Buffer::ARRAY_BUFFER, true);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

CUDA_TEST_P(Buffer, CopyFromBuffer)
{
    cv::Mat gold = randomMat(size, type);
    cv::ogl::Buffer buf_gold(gold, cv::ogl::Buffer::ARRAY_BUFFER, true);

    cv::ogl::Buffer buf;
    buf.copyFrom(buf_gold, cv::ogl::Buffer::ARRAY_BUFFER, true);

    EXPECT_NE(buf_gold.bufId(), buf.bufId());

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

CUDA_TEST_P(Buffer, CopyToGpuMat)
{
    cv::Mat gold = randomMat(size, type);

    cv::ogl::Buffer buf(gold, cv::ogl::Buffer::ARRAY_BUFFER, true);

    cv::cuda::GpuMat dst;
    buf.copyTo(dst);

    EXPECT_MAT_NEAR(gold, dst, 0);
}

CUDA_TEST_P(Buffer, CopyToBuffer)
{
    cv::Mat gold = randomMat(size, type);

    cv::ogl::Buffer buf(gold, cv::ogl::Buffer::ARRAY_BUFFER, true);

    cv::ogl::Buffer dst;
    buf.copyTo(dst);
    dst.setAutoRelease(true);

    EXPECT_NE(buf.bufId(), dst.bufId());

    cv::Mat bufData;
    dst.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

CUDA_TEST_P(Buffer, Clone)
{
    cv::Mat gold = randomMat(size, type);

    cv::ogl::Buffer buf(gold, cv::ogl::Buffer::ARRAY_BUFFER, true);

    cv::ogl::Buffer dst = buf.clone(cv::ogl::Buffer::ARRAY_BUFFER, true);

    EXPECT_NE(buf.bufId(), dst.bufId());

    cv::Mat bufData;
    dst.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

CUDA_TEST_P(Buffer, MapHostRead)
{
    cv::Mat gold = randomMat(size, type);

    cv::ogl::Buffer buf(gold, cv::ogl::Buffer::ARRAY_BUFFER, true);

    cv::Mat dst = buf.mapHost(cv::ogl::Buffer::READ_ONLY);

    EXPECT_MAT_NEAR(gold, dst, 0);

    buf.unmapHost();
}

CUDA_TEST_P(Buffer, MapHostWrite)
{
    cv::Mat gold = randomMat(size, type);

    cv::ogl::Buffer buf(size, type, cv::ogl::Buffer::ARRAY_BUFFER, true);

    cv::Mat dst = buf.mapHost(cv::ogl::Buffer::WRITE_ONLY);
    gold.copyTo(dst);
    buf.unmapHost();
    dst.release();

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

CUDA_TEST_P(Buffer, MapDevice)
{
    cv::Mat gold = randomMat(size, type);

    cv::ogl::Buffer buf(gold, cv::ogl::Buffer::ARRAY_BUFFER, true);

    cv::cuda::GpuMat dst = buf.mapDevice();

    EXPECT_MAT_NEAR(gold, dst, 0);

    buf.unmapDevice();
}

INSTANTIATE_TEST_CASE_P(OpenGL, Buffer, testing::Combine(DIFFERENT_SIZES, ALL_TYPES));

/////////////////////////////////////////////
// Texture2D

PARAM_TEST_CASE(Texture2D, cv::Size, MatType)
{
    static void SetUpTestCase()
    {
        cv::namedWindow("test", cv::WINDOW_OPENGL);
    }

    static void TearDownTestCase()
    {
        cv::destroyAllWindows();
    }

    cv::Size size;
    int type;
    int depth;
    int cn;
    cv::ogl::Texture2D::Format format;

    virtual void SetUp()
    {
        size = GET_PARAM(0);
        type = GET_PARAM(1);

        depth = CV_MAT_DEPTH(type);
        cn = CV_MAT_CN(type);
        format = cn == 1 ? cv::ogl::Texture2D::DEPTH_COMPONENT : cn == 3 ? cv::ogl::Texture2D::RGB : cn == 4 ? cv::ogl::Texture2D::RGBA : cv::ogl::Texture2D::NONE;
    }
};

CUDA_TEST_P(Texture2D, Constructor1)
{
    cv::ogl::Texture2D tex(size.height, size.width, format, true);

    EXPECT_EQ(size.height, tex.rows());
    EXPECT_EQ(size.width, tex.cols());
    EXPECT_EQ(format, tex.format());
}

CUDA_TEST_P(Texture2D, Constructor2)
{
    cv::ogl::Texture2D tex(size, format, true);

    EXPECT_EQ(size.height, tex.rows());
    EXPECT_EQ(size.width, tex.cols());
    EXPECT_EQ(format, tex.format());
}

CUDA_TEST_P(Texture2D, ConstructorFromMat)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);

    cv::ogl::Texture2D tex(gold, true);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);
}

CUDA_TEST_P(Texture2D, ConstructorFromGpuMat)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);
    cv::cuda::GpuMat d_gold(gold);

    cv::ogl::Texture2D tex(d_gold, true);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);
}

CUDA_TEST_P(Texture2D, ConstructorFromBuffer)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);
    cv::ogl::Buffer buf_gold(gold, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER, true);

    cv::ogl::Texture2D tex(buf_gold, true);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);
}

CUDA_TEST_P(Texture2D, ConstructorFromTexture2D)
{
    cv::ogl::Texture2D tex_gold(size, format, true);
    cv::ogl::Texture2D tex(tex_gold);

    EXPECT_EQ(tex_gold.texId(), tex.texId());
    EXPECT_EQ(tex_gold.rows(), tex.rows());
    EXPECT_EQ(tex_gold.cols(), tex.cols());
    EXPECT_EQ(tex_gold.format(), tex.format());
}

CUDA_TEST_P(Texture2D, Create)
{
    cv::ogl::Texture2D tex;
    tex.create(size.height, size.width, format, true);

    EXPECT_EQ(size.height, tex.rows());
    EXPECT_EQ(size.width, tex.cols());
    EXPECT_EQ(format, tex.format());
}

CUDA_TEST_P(Texture2D, CopyFromMat)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);

    cv::ogl::Texture2D tex;
    tex.copyFrom(gold, true);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);
}

CUDA_TEST_P(Texture2D, CopyFromGpuMat)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);
    cv::cuda::GpuMat d_gold(gold);

    cv::ogl::Texture2D tex;
    tex.copyFrom(d_gold, true);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);
}

CUDA_TEST_P(Texture2D, CopyFromBuffer)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);
    cv::ogl::Buffer buf_gold(gold, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER, true);

    cv::ogl::Texture2D tex;
    tex.copyFrom(buf_gold, true);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);
}

CUDA_TEST_P(Texture2D, CopyToGpuMat)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);

    cv::ogl::Texture2D tex(gold, true);

    cv::cuda::GpuMat dst;
    tex.copyTo(dst, depth);

    EXPECT_MAT_NEAR(gold, dst, 1e-2);
}

CUDA_TEST_P(Texture2D, CopyToBuffer)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);

    cv::ogl::Texture2D tex(gold, true);

    cv::ogl::Buffer dst;
    tex.copyTo(dst, depth, true);

    cv::Mat bufData;
    dst.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 1e-2);
}

INSTANTIATE_TEST_CASE_P(OpenGL, Texture2D, testing::Combine(DIFFERENT_SIZES, testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4)));

#endif
