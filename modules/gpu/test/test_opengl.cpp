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

GPU_TEST_P(GlBuffer, Constructor1)
{
    cv::GlBuffer buf(size.height, size.width, type, cv::GlBuffer::ARRAY_BUFFER, true);

    EXPECT_EQ(size.height, buf.rows());
    EXPECT_EQ(size.width, buf.cols());
    EXPECT_EQ(type, buf.type());
}

GPU_TEST_P(GlBuffer, Constructor2)
{
    cv::GlBuffer buf(size, type, cv::GlBuffer::ARRAY_BUFFER, true);

    EXPECT_EQ(size.height, buf.rows());
    EXPECT_EQ(size.width, buf.cols());
    EXPECT_EQ(type, buf.type());
}

GPU_TEST_P(GlBuffer, ConstructorFromMat)
{
    cv::Mat gold = randomMat(size, type);

    cv::GlBuffer buf(gold, cv::GlBuffer::ARRAY_BUFFER, true);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

GPU_TEST_P(GlBuffer, ConstructorFromGpuMat)
{
    cv::Mat gold = randomMat(size, type);
    cv::gpu::GpuMat d_gold(gold);

    cv::GlBuffer buf(d_gold, cv::GlBuffer::ARRAY_BUFFER);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

GPU_TEST_P(GlBuffer, ConstructorFromGlBuffer)
{
    cv::GlBuffer buf_gold(size, type, cv::GlBuffer::ARRAY_BUFFER, true);

    cv::GlBuffer buf(buf_gold);

    EXPECT_EQ(buf_gold.bufId(), buf.bufId());
    EXPECT_EQ(buf_gold.rows(), buf.rows());
    EXPECT_EQ(buf_gold.cols(), buf.cols());
    EXPECT_EQ(buf_gold.type(), buf.type());
}

GPU_TEST_P(GlBuffer, ConstructorFromGlTexture)
{
    const int depth = CV_MAT_DEPTH(type);
    const int cn = CV_MAT_CN(type);

    if (depth != CV_32F || cn == 2)
        return;

    cv::Mat gold = randomMat(size, type, 0, 1.0);
    cv::GlTexture tex_gold(gold, true);

    cv::GlBuffer buf(tex_gold, cv::GlBuffer::PIXEL_PACK_BUFFER, true);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 1e-2);
}

GPU_TEST_P(GlBuffer, Create)
{
    cv::GlBuffer buf;
    buf.create(size.height, size.width, type, cv::GlBuffer::ARRAY_BUFFER, true);

    EXPECT_EQ(size.height, buf.rows());
    EXPECT_EQ(size.width, buf.cols());
    EXPECT_EQ(type, buf.type());
}

GPU_TEST_P(GlBuffer, CopyFromMat)
{
    cv::Mat gold = randomMat(size, type);

    cv::GlBuffer buf;
    buf.copyFrom(gold, cv::GlBuffer::ARRAY_BUFFER, true);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

GPU_TEST_P(GlBuffer, CopyFromGpuMat)
{
    cv::Mat gold = randomMat(size, type);
    cv::gpu::GpuMat d_gold(gold);

    cv::GlBuffer buf;
    buf.copyFrom(d_gold, cv::GlBuffer::ARRAY_BUFFER, true);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

GPU_TEST_P(GlBuffer, CopyFromGlBuffer)
{
    cv::Mat gold = randomMat(size, type);
    cv::GlBuffer buf_gold(gold, cv::GlBuffer::ARRAY_BUFFER, true);

    cv::GlBuffer buf;
    buf.copyFrom(buf_gold, cv::GlBuffer::ARRAY_BUFFER, true);

    EXPECT_NE(buf_gold.bufId(), buf.bufId());

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

GPU_TEST_P(GlBuffer, CopyFromGlTexture)
{
    const int depth = CV_MAT_DEPTH(type);
    const int cn = CV_MAT_CN(type);

    if (depth != CV_32F || cn == 2)
        return;

    cv::Mat gold = randomMat(size, type, 0, 1.0);
    cv::GlTexture tex_gold(gold, true);

    cv::GlBuffer buf;
    buf.copyFrom(tex_gold, cv::GlBuffer::ARRAY_BUFFER, true);

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 1e-2);
}

GPU_TEST_P(GlBuffer, CopyToGpuMat)
{
    cv::Mat gold = randomMat(size, type);

    cv::GlBuffer buf(gold, cv::GlBuffer::ARRAY_BUFFER, true);

    cv::gpu::GpuMat dst;
    buf.copyTo(dst);

    EXPECT_MAT_NEAR(gold, dst, 0);
}

GPU_TEST_P(GlBuffer, CopyToGlBuffer)
{
    cv::Mat gold = randomMat(size, type);

    cv::GlBuffer buf(gold, cv::GlBuffer::ARRAY_BUFFER, true);

    cv::GlBuffer dst;
    buf.copyTo(dst, cv::GlBuffer::ARRAY_BUFFER, true);

    EXPECT_NE(buf.bufId(), dst.bufId());

    cv::Mat bufData;
    dst.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

GPU_TEST_P(GlBuffer, CopyToGlTexture)
{
    const int depth = CV_MAT_DEPTH(type);
    const int cn = CV_MAT_CN(type);

    if (depth != CV_32F || cn == 2)
        return;

    cv::Mat gold = randomMat(size, type, 0, 1.0);

    cv::GlBuffer buf(gold, cv::GlBuffer::PIXEL_PACK_BUFFER, true);

    cv::GlTexture tex;
    buf.copyTo(tex, cv::GlBuffer::PIXEL_PACK_BUFFER, true);

    cv::Mat texData;
    tex.copyTo(texData);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);
}

GPU_TEST_P(GlBuffer, Clone)
{
    cv::Mat gold = randomMat(size, type);

    cv::GlBuffer buf(gold, cv::GlBuffer::ARRAY_BUFFER, true);

    cv::GlBuffer dst = buf.clone(cv::GlBuffer::ARRAY_BUFFER, true);

    EXPECT_NE(buf.bufId(), dst.bufId());

    cv::Mat bufData;
    dst.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

GPU_TEST_P(GlBuffer, MapHostRead)
{
    cv::Mat gold = randomMat(size, type);

    cv::GlBuffer buf(gold, cv::GlBuffer::ARRAY_BUFFER, true);

    cv::Mat dst = buf.mapHost(cv::GlBuffer::READ_ONLY);

    EXPECT_MAT_NEAR(gold, dst, 0);

    buf.unmapHost();
}

GPU_TEST_P(GlBuffer, MapHostWrite)
{
    cv::Mat gold = randomMat(size, type);

    cv::GlBuffer buf(size, type, cv::GlBuffer::ARRAY_BUFFER, true);

    cv::Mat dst = buf.mapHost(cv::GlBuffer::WRITE_ONLY);
    gold.copyTo(dst);
    buf.unmapHost();
    dst.release();

    cv::Mat bufData;
    buf.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 0);
}

GPU_TEST_P(GlBuffer, MapDevice)
{
    cv::Mat gold = randomMat(size, type);

    cv::GlBuffer buf(gold, cv::GlBuffer::ARRAY_BUFFER, true);

    cv::gpu::GpuMat dst = buf.mapDevice();

    EXPECT_MAT_NEAR(gold, dst, 0);

    buf.unmapDevice();
}

INSTANTIATE_TEST_CASE_P(OpenGL, GlBuffer, testing::Combine(DIFFERENT_SIZES, ALL_TYPES));

/////////////////////////////////////////////
// GlTexture

PARAM_TEST_CASE(GlTexture, cv::Size, MatType)
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
    cv::GlTexture::Format format;

    virtual void SetUp()
    {
        size = GET_PARAM(0);
        type = GET_PARAM(1);

        depth = CV_MAT_DEPTH(type);
        cn = CV_MAT_CN(type);
        format = cn == 1 ? cv::GlTexture::DEPTH_COMPONENT : cn == 3 ? cv::GlTexture::RGB : cn == 4 ? cv::GlTexture::RGBA : cv::GlTexture::NONE;
    }
};

GPU_TEST_P(GlTexture, Constructor1)
{
    cv::GlTexture tex(size.height, size.width, format, true);

    EXPECT_EQ(size.height, tex.rows());
    EXPECT_EQ(size.width, tex.cols());
    EXPECT_EQ(format, tex.format());
}

GPU_TEST_P(GlTexture, Constructor2)
{
    cv::GlTexture tex(size, format, true);

    EXPECT_EQ(size.height, tex.rows());
    EXPECT_EQ(size.width, tex.cols());
    EXPECT_EQ(format, tex.format());
}

GPU_TEST_P(GlTexture, ConstructorFromMat)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);

    cv::GlTexture tex(gold, true);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);
}

GPU_TEST_P(GlTexture, ConstructorFromGpuMat)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);
    cv::gpu::GpuMat d_gold(gold);

    cv::GlTexture tex(d_gold, true);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);
}

GPU_TEST_P(GlTexture, ConstructorFromGlBuffer)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);
    cv::GlBuffer buf_gold(gold, cv::GlBuffer::PIXEL_UNPACK_BUFFER, true);

    cv::GlTexture tex(buf_gold, true);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);
}

GPU_TEST_P(GlTexture, ConstructorFromGlTexture)
{
    cv::GlTexture tex_gold(size, format, true);
    cv::GlTexture tex(tex_gold);

    EXPECT_EQ(tex_gold.texId(), tex.texId());
    EXPECT_EQ(tex_gold.rows(), tex.rows());
    EXPECT_EQ(tex_gold.cols(), tex.cols());
    EXPECT_EQ(tex_gold.format(), tex.format());
}

GPU_TEST_P(GlTexture, Create)
{
    cv::GlTexture tex;
    tex.create(size.height, size.width, format, true);

    EXPECT_EQ(size.height, tex.rows());
    EXPECT_EQ(size.width, tex.cols());
    EXPECT_EQ(format, tex.format());
}

GPU_TEST_P(GlTexture, CopyFromMat)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);

    cv::GlTexture tex;
    tex.copyFrom(gold, true);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);
}

GPU_TEST_P(GlTexture, CopyFromGpuMat)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);
    cv::gpu::GpuMat d_gold(gold);

    cv::GlTexture tex;
    tex.copyFrom(d_gold, true);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);
}

GPU_TEST_P(GlTexture, CopyFromGlBuffer)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);
    cv::GlBuffer buf_gold(gold, cv::GlBuffer::PIXEL_UNPACK_BUFFER, true);

    cv::GlTexture tex;
    tex.copyFrom(buf_gold, true);

    cv::Mat texData;
    tex.copyTo(texData, depth);

    EXPECT_MAT_NEAR(gold, texData, 1e-2);
}

GPU_TEST_P(GlTexture, CopyToGpuMat)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);

    cv::GlTexture tex(gold, true);

    cv::gpu::GpuMat dst;
    tex.copyTo(dst, depth);

    EXPECT_MAT_NEAR(gold, dst, 1e-2);
}

GPU_TEST_P(GlTexture, CopyToGlBuffer)
{
    cv::Mat gold = randomMat(size, type, 0, depth == CV_8U ? 255 : 1);

    cv::GlTexture tex(gold, true);

    cv::GlBuffer dst;
    tex.copyTo(dst, depth, true);

    cv::Mat bufData;
    dst.copyTo(bufData);

    EXPECT_MAT_NEAR(gold, bufData, 1e-2);
}

INSTANTIATE_TEST_CASE_P(OpenGL, GlTexture, testing::Combine(DIFFERENT_SIZES, testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4)));

#endif
