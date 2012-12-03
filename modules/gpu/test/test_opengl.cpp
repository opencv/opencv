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

#if defined(HAVE_CUDA) && defined(HAVE_OPENGL)

/////////////////////////////////////////////
// GlBuffer

PARAM_TEST_CASE(GlBuffer, cv::Size, MatType)
{
    cv::Size size;
    int type;

    virtual void SetUp()
    {
        size = GET_PARAM(0);
        type = GET_PARAM(1);
    }
};

TEST_P(GlBuffer, Constructor1)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::GlBuffer buf(size.height, size.width, type);

    EXPECT_EQ(size.height, buf.rows());
    EXPECT_EQ(size.width, buf.cols());
    EXPECT_EQ(type, buf.type());

    buf.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, Constructor2)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::GlBuffer buf(size, type);

    EXPECT_EQ(size.height, buf.rows());
    EXPECT_EQ(size.width, buf.cols());
    EXPECT_EQ(type, buf.type());

    buf.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, ConstructorFromMat)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type);

    cv::GlBuffer buf(gold);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);

    buf.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, ConstructorFromGpuMat)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type);
    cv::gpu::GpuMat d_gold(gold);

    cv::GlBuffer buf(d_gold);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);

    buf.release();
    d_gold.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, ConstructorFromGlBuffer)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::GlBuffer buf_gold(size, type);
    cv::GlBuffer buf(buf_gold);

    EXPECT_EQ(buf_gold.bufId(), buf.bufId());
    EXPECT_EQ(buf_gold.rows(), buf.rows());
    EXPECT_EQ(buf_gold.cols(), buf.cols());
    EXPECT_EQ(buf_gold.type(), buf.type());

    buf.release();
    buf_gold.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, ConstructorFromGlTexture2D)
{
    const int depth = CV_MAT_DEPTH(type);
    const int cn = CV_MAT_CN(type);

    if (depth != CV_32F || cn == 2)
        return;

    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type, 0, 1.0);
    cv::GlTexture2D tex_gold(gold);

    cv::GlBuffer buf(tex_gold);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 1e-2);

    buf.release();
    tex_gold.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, Create)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::GlBuffer buf;
    buf.create(size.height, size.width, type);

    EXPECT_EQ(size.height, buf.rows());
    EXPECT_EQ(size.width, buf.cols());
    EXPECT_EQ(type, buf.type());

    buf.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, CopyFromMat)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type);

    cv::GlBuffer buf;
    buf.copyFrom(gold);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);

    buf.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, CopyFromGpuMat)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type);
    cv::gpu::GpuMat d_gold(gold);

    cv::GlBuffer buf;
    buf.copyFrom(d_gold);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);

    buf.release();
    d_gold.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, CopyFromGlBuffer)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type);
    cv::GlBuffer buf_gold(gold);

    cv::GlBuffer buf;
    buf.copyFrom(buf_gold);

    EXPECT_NE(buf_gold.bufId(), buf.bufId());

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);

    buf.release();
    buf_gold.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, CopyFromGlTexture2D)
{
    const int depth = CV_MAT_DEPTH(type);
    const int cn = CV_MAT_CN(type);

    if (depth != CV_32F || cn == 2)
        return;

    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type, 0, 1.0);
    cv::GlTexture2D tex_gold(gold);

    cv::GlBuffer buf;
    buf.copyFrom(tex_gold);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 1e-2);

    buf.release();
    tex_gold.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, CopyToGpuMat)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type);

    cv::GlBuffer buf(gold);
    cv::gpu::GpuMat dst;
    buf.copyTo(dst);

    EXPECT_MAT_NEAR(gold, dst, 0);

    dst.release();
    buf.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, CopyToGlBuffer)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type);
    cv::GlBuffer buf(gold);

    cv::GlBuffer dst;
    buf.copyTo(dst);

    EXPECT_NE(buf.bufId(), dst.bufId());

    cv::Mat bufData;
    dst.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);

    dst.release();
    buf.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, CopyToGlTexture2D)
{
    const int depth = CV_MAT_DEPTH(type);
    const int cn = CV_MAT_CN(type);

    if (depth != CV_32F || cn == 2)
        return;

    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type, 0, 1.0);
    cv::GlBuffer buf(gold);

    cv::GlTexture2D tex;
    buf.copyTo(tex);

    cv::Mat texData;
    tex.copyTo(texData);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);

    tex.release();
    buf.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, Clone)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type);
    cv::GlBuffer buf(gold);

    cv::GlBuffer dst = buf.clone();

    EXPECT_NE(buf.bufId(), dst.bufId());

    cv::Mat bufData;
    dst.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);

    dst.release();
    buf.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, MapHostRead)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type);
    cv::GlBuffer buf(gold);

    cv::Mat dst = buf.mapHost(cv::GlBuffer::READ_ONLY);

    EXPECT_MAT_NEAR(gold, dst, 0);

    buf.unmapHost();

    buf.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, MapHostWrite)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type);
    cv::GlBuffer buf(size, type);

    cv::Mat dst = buf.mapHost(cv::GlBuffer::WRITE_ONLY);
    gold.copyTo(dst);
    buf.unmapHost();
    dst.release();

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);

    buf.release();
    cv::destroyAllWindows();
}

TEST_P(GlBuffer, MapDevice)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type);
    cv::GlBuffer buf(gold);

    cv::gpu::GpuMat dst = buf.mapDevice();

    EXPECT_MAT_NEAR(gold, dst, 0);

    buf.unmapDevice();

    buf.release();
    cv::destroyAllWindows();
}

INSTANTIATE_TEST_CASE_P(OpenGL, GlBuffer, testing::Combine(DIFFERENT_SIZES, ALL_TYPES));

/////////////////////////////////////////////
// GlTexture2D

PARAM_TEST_CASE(GlTexture2D, cv::Size, MatType)
{
    cv::Size size;
    int type;
    int depth;
    int cn;
    cv::GlTexture2D::Format format;

    virtual void SetUp()
    {
        size = GET_PARAM(0);
        type = GET_PARAM(1);

        depth = CV_MAT_DEPTH(type);
        cn = CV_MAT_CN(type);
        format = cn == 1 ? cv::GlTexture2D::DEPTH_COMPONENT : cn == 3 ? cv::GlTexture2D::RGB : cn == 4 ? cv::GlTexture2D::RGBA : cv::GlTexture2D::NONE;
    }
};

TEST_P(GlTexture2D, Constructor1)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::GlTexture2D tex(size.height, size.width, format);

    EXPECT_EQ(size.height, tex.rows());
    EXPECT_EQ(size.width, tex.cols());
    EXPECT_EQ(format, tex.format());

    tex.release();
    cv::destroyAllWindows();
}

TEST_P(GlTexture2D, Constructor2)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::GlTexture2D tex(size, format);

    EXPECT_EQ(size.height, tex.rows());
    EXPECT_EQ(size.width, tex.cols());
    EXPECT_EQ(format, tex.format());

    tex.release();
    cv::destroyAllWindows();
}

TEST_P(GlTexture2D, ConstructorFromMat)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);

    cv::GlTexture2D tex(gold);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);

    tex.release();
    cv::destroyAllWindows();
}

TEST_P(GlTexture2D, ConstructorFromGpuMat)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);
    cv::gpu::GpuMat d_gold(gold);

    cv::GlTexture2D tex(d_gold);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);

    tex.release();
    d_gold.release();
    cv::destroyAllWindows();
}

TEST_P(GlTexture2D, ConstructorFromGlBuffer)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);
    cv::GlBuffer buf_gold(gold);

    cv::GlTexture2D tex(buf_gold);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);

    tex.release();
    buf_gold.release();
    cv::destroyAllWindows();
}

TEST_P(GlTexture2D, ConstructorFromGlTexture2D)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::GlTexture2D tex_gold(size, format);
    cv::GlTexture2D tex(tex_gold);

    EXPECT_EQ(tex_gold.texId(), tex.texId());
    EXPECT_EQ(tex_gold.rows(), tex.rows());
    EXPECT_EQ(tex_gold.cols(), tex.cols());
    EXPECT_EQ(tex_gold.format(), tex.format());

    tex.release();
    tex_gold.release();
    cv::destroyAllWindows();
}

TEST_P(GlTexture2D, Create)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::GlTexture2D tex;
    tex.create(size.height, size.width, format);

    EXPECT_EQ(size.height, tex.rows());
    EXPECT_EQ(size.width, tex.cols());
    EXPECT_EQ(format, tex.format());

    tex.release();
    cv::destroyAllWindows();
}

TEST_P(GlTexture2D, CopyFromMat)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);

    cv::GlTexture2D tex;
    tex.copyFrom(gold);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);

    tex.release();
    cv::destroyAllWindows();
}

TEST_P(GlTexture2D, CopyFromGpuMat)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);
    cv::gpu::GpuMat d_gold(gold);

    cv::GlTexture2D tex;
    tex.copyFrom(d_gold);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);

    tex.release();
    d_gold.release();
    cv::destroyAllWindows();
}

TEST_P(GlTexture2D, CopyFromGlBuffer)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);
    cv::GlBuffer buf_gold(gold);

    cv::GlTexture2D tex;
    tex.copyFrom(buf_gold);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);

    tex.release();
    buf_gold.release();
    cv::destroyAllWindows();
}

TEST_P(GlTexture2D, CopyToGpuMat)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);

    cv::GlTexture2D tex(gold);
    cv::gpu::GpuMat dst;
    tex.copyTo(dst, depth);

    EXPECT_MAT_NEAR(gold, dst, 1e-2);

    dst.release();
    tex.release();
    cv::destroyAllWindows();
}

TEST_P(GlTexture2D, CopyToGlBuffer)
{
    cv::namedWindow("test", cv::WINDOW_OPENGL);

    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);

    cv::GlTexture2D tex(gold);

    cv::GlBuffer dst;
    tex.copyTo(dst, depth);

    cv::Mat bufData;
    dst.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 1e-2);

    dst.release();
    tex.release();
    cv::destroyAllWindows();
}

INSTANTIATE_TEST_CASE_P(OpenGL, GlTexture2D, testing::Combine(DIFFERENT_SIZES, testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4)));

#endif
