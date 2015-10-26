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

#include "precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

Mat cv::superres::arrGetMat(InputArray arr, Mat& buf)
{
    switch (arr.kind())
    {
    case _InputArray::GPU_MAT:
        arr.getGpuMat().download(buf);
        return buf;

    case _InputArray::OPENGL_BUFFER:
        arr.getOGlBuffer().copyTo(buf);
        return buf;

    case _InputArray::OPENGL_TEXTURE:
        arr.getOGlTexture2D().copyTo(buf);
        return buf;

    default:
        return arr.getMat();
    }
}

GpuMat cv::superres::arrGetGpuMat(InputArray arr, GpuMat& buf)
{
    switch (arr.kind())
    {
    case _InputArray::GPU_MAT:
        return arr.getGpuMat();

    case _InputArray::OPENGL_BUFFER:
        arr.getOGlBuffer().copyTo(buf);
        return buf;

    case _InputArray::OPENGL_TEXTURE:
        arr.getOGlTexture2D().copyTo(buf);
        return buf;

    default:
        buf.upload(arr.getMat());
        return buf;
    }
}

namespace
{
    void mat2mat(InputArray src, OutputArray dst)
    {
        src.getMat().copyTo(dst);
    }
    void arr2buf(InputArray src, OutputArray dst)
    {
        dst.getOGlBufferRef().copyFrom(src);
    }
    void arr2tex(InputArray src, OutputArray dst)
    {
        dst.getOGlTexture2D().copyFrom(src);
    }
    void mat2gpu(InputArray src, OutputArray dst)
    {
        dst.getGpuMatRef().upload(src.getMat());
    }
    void buf2arr(InputArray src, OutputArray dst)
    {
        src.getOGlBuffer().copyTo(dst);
    }
    void tex2arr(InputArray src, OutputArray dst)
    {
        src.getOGlTexture2D().copyTo(dst);
    }
    void gpu2mat(InputArray src, OutputArray dst)
    {
        GpuMat d = src.getGpuMat();
        dst.create(d.size(), d.type());
        Mat m = dst.getMat();
        d.download(m);
    }
    void gpu2gpu(InputArray src, OutputArray dst)
    {
        src.getGpuMat().copyTo(dst.getGpuMatRef());
    }
#ifdef HAVE_OPENCV_OCL
    void ocl2mat(InputArray src, OutputArray dst)
    {
        dst.getMatRef() = (Mat)ocl::getOclMatRef(src);
    }
    void mat2ocl(InputArray src, OutputArray dst)
    {
        Mat m = src.getMat();
        ocl::getOclMatRef(dst) = (ocl::oclMat)m;
    }
    void ocl2ocl(InputArray src, OutputArray dst)
    {
        ocl::getOclMatRef(src).copyTo(ocl::getOclMatRef(dst));
    }
#else
    void ocl2mat(InputArray, OutputArray)
    {
        CV_Error(CV_StsNotImplemented, "The called functionality is disabled for current build or platform");;
    }
    void mat2ocl(InputArray, OutputArray)
    {
        CV_Error(CV_StsNotImplemented, "The called functionality is disabled for current build or platform");;
    }
    void ocl2ocl(InputArray, OutputArray)
    {
        CV_Error(CV_StsNotImplemented, "The called functionality is disabled for current build or platform");
    }
#endif
}

void cv::superres::arrCopy(InputArray src, OutputArray dst)
{
    typedef void (*func_t)(InputArray src, OutputArray dst);
    static const func_t funcs[11][11] =
    {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, arr2buf, arr2tex, mat2gpu, mat2ocl},
        {0, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, arr2buf, arr2tex, mat2gpu, mat2ocl},
        {0, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, arr2buf, arr2tex, mat2gpu, mat2ocl},
        {0, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, arr2buf, arr2tex, mat2gpu, mat2ocl},
        {0, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, arr2buf, arr2tex, mat2gpu, mat2ocl},
        {0, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, arr2buf, arr2tex, mat2gpu, mat2ocl},
        {0, buf2arr, buf2arr, buf2arr, buf2arr, buf2arr, buf2arr, buf2arr, buf2arr, buf2arr, 0      },
        {0, tex2arr, tex2arr, tex2arr, tex2arr, tex2arr, tex2arr, tex2arr, tex2arr, tex2arr, 0      },
        {0, gpu2mat, gpu2mat, gpu2mat, gpu2mat, gpu2mat, gpu2mat, arr2buf, arr2tex, gpu2gpu, 0      },
        {0, ocl2mat, ocl2mat, ocl2mat, ocl2mat, ocl2mat, ocl2mat, 0,       0,       0,       ocl2ocl}
    };

    const int src_kind = src.kind() >> _InputArray::KIND_SHIFT;
    const int dst_kind = dst.kind() >> _InputArray::KIND_SHIFT;

    CV_DbgAssert( src_kind >= 0 && src_kind < 11 );
    CV_DbgAssert( dst_kind >= 0 && dst_kind < 11 );

    const func_t func = funcs[src_kind][dst_kind];
    CV_DbgAssert( func != 0 );

    func(src, dst);
}

namespace
{
    void convertToCn(InputArray src, OutputArray dst, int cn)
    {
        CV_Assert( src.channels() == 1 || src.channels() == 3 || src.channels() == 4 );
        CV_Assert( cn == 1 || cn == 3 || cn == 4 );

        static const int codes[5][5] =
        {
            {-1, -1, -1, -1, -1},
            {-1, -1, -1, COLOR_GRAY2BGR, COLOR_GRAY2BGRA},
            {-1, -1, -1, -1, -1},
            {-1, COLOR_BGR2GRAY, -1, -1, COLOR_BGR2BGRA},
            {-1, COLOR_BGRA2GRAY, -1, COLOR_BGRA2BGR, -1},
        };

        const int code = codes[src.channels()][cn];
        CV_DbgAssert( code >= 0 );

        switch (src.kind())
        {
        case _InputArray::GPU_MAT:
            #if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
                gpu::cvtColor(src.getGpuMat(), dst.getGpuMatRef(), code, cn);
            #else
                CV_Error(CV_StsNotImplemented, "The called functionality is disabled for current build or platform");
            #endif
            break;

        default:
            cvtColor(src, dst, code, cn);
            break;
        }
    }
    void convertToDepth(InputArray src, OutputArray dst, int depth)
    {
        CV_Assert( src.depth() <= CV_64F );
        CV_Assert( depth == CV_8U || depth == CV_32F );

        static const double maxVals[] =
        {
            (double)numeric_limits<uchar>::max(),
            (double)numeric_limits<schar>::max(),
            (double)numeric_limits<ushort>::max(),
            (double)numeric_limits<short>::max(),
            (double)numeric_limits<int>::max(),
            1.0,
            1.0,
        };

        const double scale = maxVals[depth] / maxVals[src.depth()];

        switch (src.kind())
        {
        case _InputArray::GPU_MAT:
            src.getGpuMat().convertTo(dst.getGpuMatRef(), depth, scale);
            break;

        default:
            src.getMat().convertTo(dst, depth, scale);
            break;
        }
    }
}

Mat cv::superres::convertToType(const Mat& src, int type, Mat& buf0, Mat& buf1)
{
    if (src.type() == type)
        return src;

    const int depth = CV_MAT_DEPTH(type);
    const int cn = CV_MAT_CN(type);

    if (src.depth() == depth)
    {
        convertToCn(src, buf0, cn);
        return buf0;
    }

    if (src.channels() == cn)
    {
        convertToDepth(src, buf1, depth);
        return buf1;
    }

    convertToCn(src, buf0, cn);
    convertToDepth(buf0, buf1, depth);
    return buf1;
}

GpuMat cv::superres::convertToType(const GpuMat& src, int type, GpuMat& buf0, GpuMat& buf1)
{
    if (src.type() == type)
        return src;

    const int depth = CV_MAT_DEPTH(type);
    const int cn = CV_MAT_CN(type);

    if (src.depth() == depth)
    {
        convertToCn(src, buf0, cn);
        return buf0;
    }

    if (src.channels() == cn)
    {
        convertToDepth(src, buf1, depth);
        return buf1;
    }

    convertToCn(src, buf0, cn);
    convertToDepth(buf0, buf1, depth);
    return buf1;
}
#ifdef HAVE_OPENCV_OCL
namespace
{
    // TODO(pengx17): remove these overloaded functions until IntputArray fully supports oclMat
    void convertToCn(const ocl::oclMat& src, ocl::oclMat& dst, int cn)
    {
        CV_Assert( src.channels() == 1 || src.channels() == 3 || src.channels() == 4 );
        CV_Assert( cn == 1 || cn == 3 || cn == 4 );

        static const int codes[5][5] =
        {
            {-1, -1, -1, -1, -1},
            {-1, -1, -1, COLOR_GRAY2BGR, COLOR_GRAY2BGRA},
            {-1, -1, -1, -1, -1},
            {-1, COLOR_BGR2GRAY, -1, -1, COLOR_BGR2BGRA},
            {-1, COLOR_BGRA2GRAY, -1, COLOR_BGRA2BGR, -1},
        };

        const int code = codes[src.channels()][cn];
        CV_DbgAssert( code >= 0 );

        ocl::cvtColor(src, dst, code, cn);
    }
    void convertToDepth(const ocl::oclMat& src, ocl::oclMat& dst, int depth)
    {
        CV_Assert( src.depth() <= CV_64F );
        CV_Assert( depth == CV_8U || depth == CV_32F );

        static const double maxVals[] =
        {
            (double)std::numeric_limits<uchar>::max(),
            (double)std::numeric_limits<schar>::max(),
            (double)std::numeric_limits<ushort>::max(),
            (double)std::numeric_limits<short>::max(),
            (double)std::numeric_limits<int>::max(),
            1.0,
            1.0,
        };
        const double scale = maxVals[depth] / maxVals[src.depth()];
        src.convertTo(dst, depth, scale);
    }
}
ocl::oclMat cv::superres::convertToType(const ocl::oclMat& src, int type, ocl::oclMat& buf0, ocl::oclMat& buf1)
{
    if (src.type() == type)
        return src;

    const int depth = CV_MAT_DEPTH(type);
    const int cn = CV_MAT_CN(type);

    if (src.depth() == depth)
    {
        convertToCn(src, buf0, cn);
        return buf0;
    }

    if (src.channels() == cn)
    {
        convertToDepth(src, buf1, depth);
        return buf1;
    }

    convertToCn(src, buf0, cn);
    convertToDepth(buf0, buf1, depth);
    return buf1;
}
#endif
