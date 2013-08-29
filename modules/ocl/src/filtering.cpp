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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Zero Lin, Zero.Lin@amd.com
//    Zhang Ying, zhangying913@gmail.com
//    Yao Wang, bitwangyaoyao@gmail.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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

using namespace cv;
using namespace cv::ocl;

//helper routines
namespace cv
{
namespace ocl
{
///////////////////////////OpenCL kernel strings///////////////////////////
extern const char *filtering_boxFilter;
extern const char *filter_sep_row;
extern const char *filter_sep_col;
extern const char *filtering_laplacian;
extern const char *filtering_morph;
}
}

namespace
{
inline int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}
}

namespace
{
inline void normalizeAnchor(int &anchor, int ksize)
{
    if (anchor < 0)
    {
        anchor = ksize >> 1;
    }

    CV_Assert(0 <= anchor && anchor < ksize);
}

inline void normalizeAnchor(Point &anchor, const Size &ksize)
{
    normalizeAnchor(anchor.x, ksize.width);
    normalizeAnchor(anchor.y, ksize.height);
}

inline void normalizeROI(Rect &roi, const Size &ksize, const Point &anchor, const Size &src_size)
{
    if (roi == Rect(0, 0, -1, -1))
    {
        roi = Rect(0, 0, src_size.width, src_size.height);
    }

    CV_Assert(ksize.height > 0 && ksize.width > 0 && ((ksize.height & 1) == 1) && ((ksize.width & 1) == 1));
    CV_Assert((anchor.x == -1 && anchor.y == -1) || (anchor.x == ksize.width >> 1 && anchor.y == ksize.height >> 1));
    CV_Assert(roi.x >= 0 && roi.y >= 0 && roi.width <= src_size.width && roi.height <= src_size.height);
}


inline void normalizeKernel(const Mat &kernel, oclMat &gpu_krnl, int type = CV_8U, int *nDivisor = 0, bool reverse = false)
{
    int scale = nDivisor && (kernel.depth() == CV_32F || kernel.depth() == CV_64F) ? 256 : 1;

    if (nDivisor)
    {
        *nDivisor = scale;
    }

    Mat temp(kernel.size(), type);
    kernel.convertTo(temp, type, scale);
    Mat cont_krnl = temp.reshape(1, 1);

    if (reverse)
    {
        int count = cont_krnl.cols >> 1;

        for (int i = 0; i < count; ++i)
        {
            std::swap(cont_krnl.at<int>(0, i), cont_krnl.at<int>(0, cont_krnl.cols - 1 - i));
        }
    }

    gpu_krnl.upload(cont_krnl);
}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Filter2D
namespace
{
class Filter2DEngine_GPU : public FilterEngine_GPU
{
public:
    Filter2DEngine_GPU(const Ptr<BaseFilter_GPU> &filter2D_) : filter2D(filter2D_) {}

    virtual void apply(const oclMat &src, oclMat &dst, Rect roi = Rect(0, 0, -1, -1))
    {
        Size src_size = src.size();

        // Delete those two clause below which exist before, However, the result is alos correct
        // dst.create(src_size, src.type());
        // dst = Scalar(0.0);

        normalizeROI(roi, filter2D->ksize, filter2D->anchor, src_size);

        oclMat srcROI = src(roi);
        oclMat dstROI = dst(roi);

        (*filter2D)(srcROI, dstROI);
    }

    Ptr<BaseFilter_GPU> filter2D;
};
}

Ptr<FilterEngine_GPU> cv::ocl::createFilter2D_GPU(const Ptr<BaseFilter_GPU> filter2D)
{
    return Ptr<FilterEngine_GPU>(new Filter2DEngine_GPU(filter2D));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Box Filter
namespace
{
typedef void (*FilterBox_t)(const oclMat & , oclMat & , Size &, const Point, const int);

class GPUBoxFilter : public BaseFilter_GPU
{
public:
    GPUBoxFilter(const Size &ksize_, const Point &anchor_, const int borderType_, FilterBox_t func_) :
        BaseFilter_GPU(ksize_, anchor_, borderType_), func(func_) {}

    virtual void operator()(const oclMat &src, oclMat &dst)
    {
        func(src, dst, ksize, anchor, borderType);
    }

    FilterBox_t func;

};
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Morphology Filter

namespace
{
typedef void (*GPUMorfFilter_t)(const oclMat & , oclMat & , oclMat & , Size &, const Point, bool rectKernel);

class MorphFilter_GPU : public BaseFilter_GPU
{
public:
    MorphFilter_GPU(const Size &ksize_, const Point &anchor_, const oclMat &kernel_, GPUMorfFilter_t func_) :
        BaseFilter_GPU(ksize_, anchor_, BORDER_CONSTANT), kernel(kernel_), func(func_), rectKernel(false) {}

    virtual void operator()(const oclMat &src, oclMat &dst)
    {
        func(src, dst, kernel, ksize, anchor, rectKernel) ;
    }

    oclMat kernel;
    GPUMorfFilter_t func;
    bool rectKernel;
};
}

/*
**We should be able to support any data types here.
**Extend this if necessary later.
**Note that the kernel need to be further refined.
*/
static void GPUErode(const oclMat &src, oclMat &dst, oclMat &mat_kernel,
                         Size &ksize, const Point anchor, bool rectKernel)
{
    //Normalize the result by default
    //float alpha = ksize.height * ksize.width;
    CV_Assert(src.clCxt == dst.clCxt);
    CV_Assert((src.cols == dst.cols) &&
              (src.rows == dst.rows));
    CV_Assert((src.oclchannels() == dst.oclchannels()));

    int srcStep = src.step1() / src.oclchannels();
    int dstStep = dst.step1() / dst.oclchannels();
    int srcOffset = src.offset /  src.elemSize();
    int dstOffset = dst.offset /  dst.elemSize();

    int srcOffset_x = srcOffset % srcStep;
    int srcOffset_y = srcOffset / srcStep;
    Context *clCxt = src.clCxt;
    String kernelName;
    size_t localThreads[3] = {16, 16, 1};
    size_t globalThreads[3] = {(src.cols + localThreads[0] - 1) / localThreads[0] *localThreads[0], (src.rows + localThreads[1] - 1) / localThreads[1] *localThreads[1], 1};

    if (src.type() == CV_8UC1)
    {
        kernelName = "morph_C1_D0";
        globalThreads[0] = ((src.cols + 3) / 4 + localThreads[0] - 1) / localThreads[0] * localThreads[0];
        CV_Assert(localThreads[0]*localThreads[1] * 8 >= (localThreads[0] * 4 + ksize.width - 1) * (localThreads[1] + ksize.height - 1));
    }
    else
    {
        kernelName = "morph";
        CV_Assert(localThreads[0]*localThreads[1] * 2 >= (localThreads[0] + ksize.width - 1) * (localThreads[1] + ksize.height - 1));
    }

    char s[64];

    switch (src.type())
    {
    case CV_8UC1:
        sprintf(s, "-D VAL=255");
        break;
    case CV_8UC3:
    case CV_8UC4:
        sprintf(s, "-D VAL=255 -D GENTYPE=uchar4");
        break;
    case CV_32FC1:
        sprintf(s, "-D VAL=FLT_MAX -D GENTYPE=float");
        break;
    case CV_32FC3:
    case CV_32FC4:
        sprintf(s, "-D VAL=FLT_MAX -D GENTYPE=float4");
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "unsupported type");
    }

    char compile_option[128];
    sprintf(compile_option, "-D RADIUSX=%d -D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D ERODE %s %s",
        anchor.x, anchor.y, (int)localThreads[0], (int)localThreads[1],
        rectKernel?"-D RECTKERNEL":"",
        s);
    std::vector< std::pair<size_t, const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&src.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&dst.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&srcOffset_x));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&srcOffset_y));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&srcStep));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dstStep));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&mat_kernel.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholecols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholerows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dstOffset));
    openCLExecuteKernel(clCxt, &filtering_morph, kernelName, globalThreads, localThreads, args, -1, -1, compile_option);
}


//! data type supported: CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4
static void GPUDilate(const oclMat &src, oclMat &dst, oclMat &mat_kernel,
                          Size &ksize, const Point anchor, bool rectKernel)
{
    //Normalize the result by default
    //float alpha = ksize.height * ksize.width;
    CV_Assert(src.clCxt == dst.clCxt);
    CV_Assert((src.cols == dst.cols) &&
              (src.rows == dst.rows));
    CV_Assert((src.oclchannels() == dst.oclchannels()));

    int srcStep = src.step1() / src.oclchannels();
    int dstStep = dst.step1() / dst.oclchannels();
    int srcOffset = src.offset /  src.elemSize();
    int dstOffset = dst.offset /  dst.elemSize();

    int srcOffset_x = srcOffset % srcStep;
    int srcOffset_y = srcOffset / srcStep;
    Context *clCxt = src.clCxt;
    String kernelName;
    size_t localThreads[3] = {16, 16, 1};
    size_t globalThreads[3] = {(src.cols + localThreads[0] - 1) / localThreads[0] *localThreads[0],
                               (src.rows + localThreads[1] - 1) / localThreads[1] *localThreads[1], 1};

    if (src.type() == CV_8UC1)
    {
        kernelName = "morph_C1_D0";
        globalThreads[0] = ((src.cols + 3) / 4 + localThreads[0] - 1) / localThreads[0] * localThreads[0];
        CV_Assert(localThreads[0]*localThreads[1] * 8 >= (localThreads[0] * 4 + ksize.width - 1) * (localThreads[1] + ksize.height - 1));
    }
    else
    {
        kernelName = "morph";
        CV_Assert(localThreads[0]*localThreads[1] * 2 >= (localThreads[0] + ksize.width - 1) * (localThreads[1] + ksize.height - 1));
    }

    char s[64];

    switch (src.type())
    {
    case CV_8UC1:
        sprintf(s, "-D VAL=0");
        break;
    case CV_8UC3:
    case CV_8UC4:
        sprintf(s, "-D VAL=0 -D GENTYPE=uchar4");
        break;
    case CV_32FC1:
        sprintf(s, "-D VAL=-FLT_MAX -D GENTYPE=float");
        break;
    case CV_32FC3:
    case CV_32FC4:
        sprintf(s, "-D VAL=-FLT_MAX -D GENTYPE=float4");
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "unsupported type");
    }

    char compile_option[128];
    sprintf(compile_option, "-D RADIUSX=%d -D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D DILATE %s %s",
        anchor.x, anchor.y, (int)localThreads[0], (int)localThreads[1],
        s, rectKernel?"-D RECTKERNEL":"");
    std::vector< std::pair<size_t, const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&src.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&dst.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&srcOffset_x));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&srcOffset_y));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&srcStep));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dstStep));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&mat_kernel.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholecols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholerows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dstOffset));
    openCLExecuteKernel(clCxt, &filtering_morph, kernelName, globalThreads, localThreads, args, -1, -1, compile_option);
}

Ptr<BaseFilter_GPU> cv::ocl::getMorphologyFilter_GPU(int op, int type, const Mat &kernel, const Size &ksize, Point anchor)
{
    static const GPUMorfFilter_t GPUMorfFilter_callers[2][5] =
    {
        {0, GPUErode, 0, GPUErode, GPUErode },
        {0, GPUDilate, 0, GPUDilate, GPUDilate}
    };

    CV_Assert(op == MORPH_ERODE || op == MORPH_DILATE);
    CV_Assert(type == CV_8UC1 || type == CV_8UC3 || type == CV_8UC4 || type == CV_32FC1 || type == CV_32FC1 || type == CV_32FC4);

    oclMat gpu_krnl;
    normalizeKernel(kernel, gpu_krnl);
    normalizeAnchor(anchor, ksize);

    bool noZero = true;
    for(int i = 0; i < kernel.rows * kernel.cols; ++i)
        if(kernel.data[i] != 1)
            noZero = false;
    MorphFilter_GPU* mfgpu=new MorphFilter_GPU(ksize, anchor, gpu_krnl, GPUMorfFilter_callers[op][CV_MAT_CN(type)]);
    if(noZero)
        mfgpu->rectKernel = true;
    return Ptr<BaseFilter_GPU>(mfgpu);
}

namespace
{
class MorphologyFilterEngine_GPU : public Filter2DEngine_GPU
{
public:
    MorphologyFilterEngine_GPU(const Ptr<BaseFilter_GPU> &filter2D_, int iters_) :
        Filter2DEngine_GPU(filter2D_), iters(iters_) {}

    virtual void apply(const oclMat &src, oclMat &dst)
    {
        Filter2DEngine_GPU::apply(src, dst);

        //if (iters > 1)
        //{
        // Size wholesize;
        // Point ofs;
        // dst.locateROI(wholesize,ofs);
        // int rows = dst.rows, cols = dst.cols;
        // dst.adjustROI(ofs.y,-ofs.y-rows+dst.wholerows,ofs.x,-ofs.x-cols+dst.wholecols);
        // dst.copyTo(morfBuf);
        // dst.adjustROI(-ofs.y,ofs.y+rows-dst.wholerows,-ofs.x,ofs.x+cols-dst.wholecols);
        // morfBuf.adjustROI(-ofs.y,ofs.y+rows-dst.wholerows,-ofs.x,ofs.x+cols-dst.wholecols);
        // //morfBuf.create(src.size(),src.type());
        // //Filter2DEngine_GPU::apply(dst, morfBuf);
        // //morfBuf.copyTo(dst);
        //}
        for (int i = 1; i < iters; ++i)
        {
            //dst.swap(morfBuf);
            Size wholesize;
            Point ofs;
            dst.locateROI(wholesize, ofs);
            int rows = dst.rows, cols = dst.cols;
            dst.adjustROI(ofs.y, -ofs.y - rows + dst.wholerows, ofs.x, -ofs.x - cols + dst.wholecols);
            dst.copyTo(morfBuf);
            dst.adjustROI(-ofs.y, ofs.y + rows - dst.wholerows, -ofs.x, ofs.x + cols - dst.wholecols);
            morfBuf.adjustROI(-ofs.y, ofs.y + rows - dst.wholerows, -ofs.x, ofs.x + cols - dst.wholecols);
            Filter2DEngine_GPU::apply(morfBuf, dst);
        }
    }

    int iters;
    oclMat morfBuf;
};
}

Ptr<FilterEngine_GPU> cv::ocl::createMorphologyFilter_GPU(int op, int type, const Mat &kernel, const Point &anchor, int iterations)
{
    CV_Assert(iterations > 0);

    Size ksize = kernel.size();

    Ptr<BaseFilter_GPU> filter2D = getMorphologyFilter_GPU(op, type, kernel, ksize, anchor);

    return Ptr<FilterEngine_GPU>(new MorphologyFilterEngine_GPU(filter2D, iterations));
}

namespace
{
void morphOp(int op, const oclMat &src, oclMat &dst, const Mat &_kernel, Point anchor, int iterations, int borderType, const Scalar &borderValue)
{
    if ((borderType != cv::BORDER_CONSTANT) || (borderValue != morphologyDefaultBorderValue()))
    {
        CV_Error(Error::StsBadArg, "unsupported border type");
    }

    Mat kernel;
    Size ksize = _kernel.data ? _kernel.size() : Size(3, 3);

    normalizeAnchor(anchor, ksize);

    if (iterations == 0 || _kernel.rows *_kernel.cols == 1)
    {
        src.copyTo(dst);
        return;
    }

    dst.create(src.size(), src.type());

    if (!_kernel.data)
    {
        kernel = getStructuringElement(MORPH_RECT, Size(1 + iterations * 2, 1 + iterations * 2));
        anchor = Point(iterations, iterations);
        iterations = 1;
    }
    else if (iterations > 1 && countNonZero(_kernel) == _kernel.rows * _kernel.cols)
    {
        anchor = Point(anchor.x * iterations, anchor.y * iterations);
        kernel = getStructuringElement(MORPH_RECT, Size(ksize.width + iterations * (ksize.width - 1),
                                       ksize.height + iterations * (ksize.height - 1)), anchor);
        iterations = 1;
    }
    else
    {
        kernel = _kernel;
    }

    Ptr<FilterEngine_GPU> f = createMorphologyFilter_GPU(op, src.type(), kernel, anchor, iterations);

    f->apply(src, dst);
}
}

void cv::ocl::erode(const oclMat &src, oclMat &dst, const Mat &kernel, Point anchor, int iterations,
                    int borderType, const Scalar &borderValue)
{
    bool allZero = true;

    for (int i = 0; i < kernel.rows * kernel.cols; ++i)
        if (kernel.data[i] != 0)
        {
            allZero = false;
        }

    if (allZero)
    {
        kernel.data[0] = 1;
    }

    morphOp(MORPH_ERODE, src, dst, kernel, anchor, iterations, borderType, borderValue);
}

void cv::ocl::dilate(const oclMat &src, oclMat &dst, const Mat &kernel, Point anchor, int iterations,
                     int borderType, const Scalar &borderValue)
{
    morphOp(MORPH_DILATE, src, dst, kernel, anchor, iterations, borderType, borderValue);
}

void cv::ocl::morphologyEx(const oclMat &src, oclMat &dst, int op, const Mat &kernel, Point anchor, int iterations,
                           int borderType, const Scalar &borderValue)
{
    oclMat temp;

    switch (op)
    {
    case MORPH_ERODE:
        erode(src, dst, kernel, anchor, iterations, borderType, borderValue);
        break;
    case MORPH_DILATE:
        dilate(src, dst, kernel, anchor, iterations, borderType, borderValue);
        break;
    case MORPH_OPEN:
        erode(src, temp, kernel, anchor, iterations, borderType, borderValue);
        dilate(temp, dst, kernel, anchor, iterations, borderType, borderValue);
        break;
    case MORPH_CLOSE:
        dilate(src, temp, kernel, anchor, iterations, borderType, borderValue);
        erode(temp, dst, kernel, anchor, iterations, borderType, borderValue);
        break;
    case MORPH_GRADIENT:
        erode(src, temp, kernel, anchor, iterations, borderType, borderValue);
        dilate(src, dst, kernel, anchor, iterations, borderType, borderValue);
        subtract(dst, temp, dst);
        break;
    case MORPH_TOPHAT:
        erode(src, dst, kernel, anchor, iterations, borderType, borderValue);
        dilate(dst, temp, kernel, anchor, iterations, borderType, borderValue);
        subtract(src, temp, dst);
        break;
    case MORPH_BLACKHAT:
        dilate(src, dst, kernel, anchor, iterations, borderType, borderValue);
        erode(dst, temp, kernel, anchor, iterations, borderType, borderValue);
        subtract(temp, src, dst);
        break;
    default:
        CV_Error(Error::StsBadArg, "unknown morphological operation");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Linear Filter

namespace
{
typedef void (*GPUFilter2D_t)(const oclMat & , oclMat & , const oclMat & , const Size &, const Point&, const int);

class LinearFilter_GPU : public BaseFilter_GPU
{
public:
    LinearFilter_GPU(const Size &ksize_, const Point &anchor_, const oclMat &kernel_, GPUFilter2D_t func_,
                     int borderType_) :
        BaseFilter_GPU(ksize_, anchor_, borderType_), kernel(kernel_), func(func_) {}

    virtual void operator()(const oclMat &src, oclMat &dst)
    {
        func(src, dst, kernel, ksize, anchor, borderType) ;
    }

    oclMat kernel;
    GPUFilter2D_t func;
};
}

static void GPUFilter2D(const oclMat &src, oclMat &dst, const oclMat &mat_kernel,
    const Size &ksize, const Point& anchor, const int borderType)
{
    CV_Assert(src.clCxt == dst.clCxt);
    CV_Assert((src.cols == dst.cols) &&
              (src.rows == dst.rows));
    CV_Assert((src.oclchannels() == dst.oclchannels()));
    CV_Assert(ksize.height > 0 && ksize.width > 0 && ((ksize.height & 1) == 1) && ((ksize.width & 1) == 1));
    CV_Assert((anchor.x == -1 && anchor.y == -1) || (anchor.x == ksize.width >> 1 && anchor.y == ksize.height >> 1));
    CV_Assert(ksize.width == ksize.height);
    Context *clCxt = src.clCxt;

    int filterWidth = ksize.width;
    bool ksize_3x3 = filterWidth == 3 && src.type() != CV_32FC4; // CV_32FC4 is not tuned up with filter2d_3x3 kernel

    String kernelName = ksize_3x3 ? "filter2D_3x3" : "filter2D";

    size_t src_offset_x = (src.offset % src.step) / src.elemSize();
    size_t src_offset_y = src.offset / src.step;

    size_t dst_offset_x = (dst.offset % dst.step) / dst.elemSize();
    size_t dst_offset_y = dst.offset / dst.step;

    int paddingPixels = filterWidth & (-2);

    size_t localThreads[3]  = {ksize_3x3 ? 256 : 16, ksize_3x3 ? 1 : 16, 1};
    size_t globalThreads[3] = {src.wholecols, src.wholerows, 1};

    int cn =  src.oclchannels();
    int src_step = (int)(src.step/src.elemSize());
    int dst_step = (int)(dst.step/src.elemSize());

    int localWidth = localThreads[0] + paddingPixels;
    int localHeight = localThreads[1] + paddingPixels;

    // 260 = divup((localThreads[0] + filterWidth * 2), 4) * 4
    // 6   = (ROWS_PER_GROUP_WHICH_IS_4 + filterWidth * 2)
    size_t localMemSize = ksize_3x3 ? 260 * 6 * src.elemSize() : (localWidth * localHeight) * src.elemSize();

    int vector_lengths[4][7] = {{4, 4, 4, 4, 4, 4, 4},
    {4, 4, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {4, 4, 4, 4, 1, 1, 4}
    };
    int cols = dst.cols + ((dst_offset_x) & (vector_lengths[cn - 1][src.depth()] - 1));

    std::vector< std::pair<size_t, const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&src.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&dst.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst_step));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&mat_kernel.data));
    args.push_back(std::make_pair(localMemSize,   (void *)NULL));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholerows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholecols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src_offset_x));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src_offset_y));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst_offset_x));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst_offset_y));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&cols));
    char btype[30];
    switch (borderType)
    {
    case 0:
        sprintf(btype, "BORDER_CONSTANT");
        break;
    case 1:
        sprintf(btype, "BORDER_REPLICATE");
        break;
    case 2:
        sprintf(btype, "BORDER_REFLECT");
        break;
    case 3:
        CV_Error(CV_StsUnsupportedFormat, "BORDER_WRAP is not supported!");
        return;
    case 4:
        sprintf(btype, "BORDER_REFLECT_101");
        break;
    }
    int type = src.depth();
    char build_options[150];
    sprintf(build_options, "-D %s -D IMG_C_%d_%d -D CN=%d -D FILTER_SIZE=%d", btype, cn, type, cn, ksize.width);
    openCLExecuteKernel(clCxt, &filtering_laplacian, kernelName, globalThreads, localThreads, args, -1, -1, build_options);
}

Ptr<BaseFilter_GPU> cv::ocl::getLinearFilter_GPU(int srcType, int dstType, const Mat &kernel, const Size &ksize,
        const Point &anchor, int borderType)
{
    static const GPUFilter2D_t GPUFilter2D_callers[] = {0, GPUFilter2D, 0, GPUFilter2D, GPUFilter2D};

    CV_Assert((srcType == CV_8UC1 || srcType == CV_8UC3 || srcType == CV_8UC4 || srcType == CV_32FC1 || srcType == CV_32FC3 || srcType == CV_32FC4) && dstType == srcType);

    oclMat gpu_krnl;
    Point norm_archor = anchor;
    normalizeKernel(kernel, gpu_krnl, CV_32FC1);
    normalizeAnchor(norm_archor, ksize);

    return Ptr<BaseFilter_GPU>(new LinearFilter_GPU(ksize, anchor, gpu_krnl, GPUFilter2D_callers[CV_MAT_CN(srcType)],
                               borderType));
}

Ptr<FilterEngine_GPU> cv::ocl::createLinearFilter_GPU(int srcType, int dstType, const Mat &kernel, const Point &anchor,
        int borderType)
{

    Size ksize = kernel.size();

    Ptr<BaseFilter_GPU> linearFilter = getLinearFilter_GPU(srcType, dstType, kernel, ksize, anchor, borderType);

    return createFilter2D_GPU(linearFilter);
}

void cv::ocl::filter2D(const oclMat &src, oclMat &dst, int ddepth, const Mat &kernel, Point anchor, int borderType)
{

    if (ddepth < 0)
    {
        ddepth = src.depth();
    }

    dst.create(src.size(), CV_MAKETYPE(ddepth, src.channels()));

    Ptr<FilterEngine_GPU> f = createLinearFilter_GPU(src.type(), dst.type(), kernel, anchor, borderType);
    f->apply(src, dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// SeparableFilter

namespace
{
class SeparableFilterEngine_GPU : public FilterEngine_GPU
{
public:
    SeparableFilterEngine_GPU(const Ptr<BaseRowFilter_GPU> &rowFilter_,
                              const Ptr<BaseColumnFilter_GPU> &columnFilter_) :
        rowFilter(rowFilter_), columnFilter(columnFilter_)
    {
        ksize = Size(rowFilter->ksize, columnFilter->ksize);
        anchor = Point(rowFilter->anchor, columnFilter->anchor);
    }

    virtual void apply(const oclMat &src, oclMat &dst, Rect roi = Rect(0, 0, -1, -1))
    {
        Size src_size = src.size();
        //int src_type = src.type();

        int cn = src.oclchannels();
        //dst.create(src_size, src_type);
        //dst = Scalar(0.0);
        //dstBuf.create(src_size, src_type);
        dstBuf.create(src_size.height + ksize.height - 1, src_size.width, CV_MAKETYPE(CV_32F, cn));
        //dstBuf = Scalar(0.0);

        normalizeROI(roi, ksize, anchor, src_size);

        srcROI = src(roi);
        dstROI = dst(roi);
        //dstBufROI = dstBuf(roi);

        (*rowFilter)(srcROI, dstBuf);
        //Mat rm(dstBufROI);
        //std::cout << "rm " << rm << endl;
        (*columnFilter)(dstBuf, dstROI);
    }

    Ptr<BaseRowFilter_GPU> rowFilter;
    Ptr<BaseColumnFilter_GPU> columnFilter;
    Size ksize;
    Point anchor;
    oclMat dstBuf;
    oclMat srcROI;
    oclMat dstROI;
    oclMat dstBufROI;
};
}

Ptr<FilterEngine_GPU> cv::ocl::createSeparableFilter_GPU(const Ptr<BaseRowFilter_GPU> &rowFilter,
        const Ptr<BaseColumnFilter_GPU> &columnFilter)
{
    return Ptr<FilterEngine_GPU>(new SeparableFilterEngine_GPU(rowFilter, columnFilter));
}

/*
**data type supported: CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4
**support four border types: BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101
*/

static void GPUFilterBox_8u_C1R(const oclMat &src, oclMat &dst,
                         Size &ksize, const Point anchor, const int borderType)
{
    //Normalize the result by default
    float alpha = ksize.height * ksize.width;

    CV_Assert(src.clCxt == dst.clCxt);
    CV_Assert((src.cols == dst.cols) &&
              (src.rows == dst.rows));
    Context *clCxt = src.clCxt;

    String kernelName = "boxFilter_C1_D0";

    char btype[30];

    switch (borderType)
    {
    case 0:
        sprintf(btype, "BORDER_CONSTANT");
        break;
    case 1:
        sprintf(btype, "BORDER_REPLICATE");
        break;
    case 2:
        sprintf(btype, "BORDER_REFLECT");
        break;
    case 3:
        CV_Error(Error::StsUnsupportedFormat, "BORDER_WRAP is not supported!");
        return;
    case 4:
        sprintf(btype, "BORDER_REFLECT_101");
        break;
    }

    char build_options[150];
    sprintf(build_options, "-D anX=%d -D anY=%d -D ksX=%d -D ksY=%d -D %s", anchor.x, anchor.y, ksize.width, ksize.height, btype);

    size_t blockSizeX = 256, blockSizeY = 1;
    size_t gSize = blockSizeX - (ksize.width - 1);
    size_t threads = (dst.offset % dst.step % 4 + dst.cols + 3) / 4;
    size_t globalSizeX = threads % gSize == 0 ? threads / gSize * blockSizeX : (threads / gSize + 1) * blockSizeX;
    size_t globalSizeY = ((dst.rows + 1) / 2) % blockSizeY == 0 ? ((dst.rows + 1) / 2) : (((dst.rows + 1) / 2) / blockSizeY + 1) * blockSizeY;

    size_t globalThreads[3] = { globalSizeX, globalSizeY, 1 };
    size_t localThreads[3]  = { blockSizeX, blockSizeY, 1 };

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), &src.data));
    args.push_back(std::make_pair(sizeof(cl_mem), &dst.data));
    args.push_back(std::make_pair(sizeof(cl_float), (void *)&alpha));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.offset));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholerows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholecols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.offset));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.step));

    openCLExecuteKernel(clCxt, &filtering_boxFilter, kernelName, globalThreads, localThreads, args, -1, -1, build_options);
}

static void GPUFilterBox_8u_C4R(const oclMat &src, oclMat &dst,
                         Size &ksize, const Point anchor, const int borderType)
{
    //Normalize the result by default
    float alpha = ksize.height * ksize.width;

    CV_Assert(src.clCxt == dst.clCxt);
    CV_Assert((src.cols == dst.cols) &&
              (src.rows == dst.rows));
    Context *clCxt = src.clCxt;

    String kernelName = "boxFilter_C4_D0";

    char btype[30];

    switch (borderType)
    {
    case 0:
        sprintf(btype, "BORDER_CONSTANT");
        break;
    case 1:
        sprintf(btype, "BORDER_REPLICATE");
        break;
    case 2:
        sprintf(btype, "BORDER_REFLECT");
        break;
    case 3:
        CV_Error(Error::StsUnsupportedFormat, "BORDER_WRAP is not supported!");
        return;
    case 4:
        sprintf(btype, "BORDER_REFLECT_101");
        break;
    }

    char build_options[150];
    sprintf(build_options, "-D anX=%d -D anY=%d -D ksX=%d -D ksY=%d -D %s", anchor.x, anchor.y, ksize.width, ksize.height, btype);

    size_t blockSizeX = 256, blockSizeY = 1;
    size_t gSize = blockSizeX - ksize.width / 2 * 2;
    size_t globalSizeX = (src.cols) % gSize == 0 ? src.cols / gSize * blockSizeX : (src.cols / gSize + 1) * blockSizeX;
    size_t rows_per_thread = 2;
    size_t globalSizeY = ((src.rows + rows_per_thread - 1) / rows_per_thread) % blockSizeY == 0 ? ((src.rows + rows_per_thread - 1) / rows_per_thread) : (((src.rows + rows_per_thread - 1) / rows_per_thread) / blockSizeY + 1) * blockSizeY;

    size_t globalThreads[3] = { globalSizeX, globalSizeY, 1};
    size_t localThreads[3]  = { blockSizeX, blockSizeY, 1};

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), &src.data));
    args.push_back(std::make_pair(sizeof(cl_mem), &dst.data));
    args.push_back(std::make_pair(sizeof(cl_float), (void *)&alpha));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.offset));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholerows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholecols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.offset));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.step));

    openCLExecuteKernel(clCxt, &filtering_boxFilter, kernelName, globalThreads, localThreads, args, -1, -1, build_options);
}

static void GPUFilterBox_32F_C1R(const oclMat &src, oclMat &dst,
                          Size &ksize, const Point anchor, const int borderType)
{
    //Normalize the result by default
    float alpha = ksize.height * ksize.width;

    CV_Assert(src.clCxt == dst.clCxt);
    CV_Assert((src.cols == dst.cols) &&
              (src.rows == dst.rows));
    Context *clCxt = src.clCxt;

    String kernelName = "boxFilter_C1_D5";

    char btype[30];

    switch (borderType)
    {
    case 0:
        sprintf(btype, "BORDER_CONSTANT");
        break;
    case 1:
        sprintf(btype, "BORDER_REPLICATE");
        break;
    case 2:
        sprintf(btype, "BORDER_REFLECT");
        break;
    case 3:
        CV_Error(Error::StsUnsupportedFormat, "BORDER_WRAP is not supported!");
        return;
    case 4:
        sprintf(btype, "BORDER_REFLECT_101");
        break;
    }

    char build_options[150];
    sprintf(build_options, "-D anX=%d -D anY=%d -D ksX=%d -D ksY=%d -D %s", anchor.x, anchor.y, ksize.width, ksize.height, btype);

    size_t blockSizeX = 256, blockSizeY = 1;
    size_t gSize = blockSizeX - ksize.width / 2 * 2;
    size_t globalSizeX = (src.cols) % gSize == 0 ? src.cols / gSize * blockSizeX : (src.cols / gSize + 1) * blockSizeX;
    size_t rows_per_thread = 2;
    size_t globalSizeY = ((src.rows + rows_per_thread - 1) / rows_per_thread) % blockSizeY == 0 ? ((src.rows + rows_per_thread - 1) / rows_per_thread) : (((src.rows + rows_per_thread - 1) / rows_per_thread) / blockSizeY + 1) * blockSizeY;


    size_t globalThreads[3] = { globalSizeX, globalSizeY, 1};
    size_t localThreads[3]  = { blockSizeX, blockSizeY, 1};

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), &src.data));
    args.push_back(std::make_pair(sizeof(cl_mem), &dst.data));
    args.push_back(std::make_pair(sizeof(cl_float), (void *)&alpha));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.offset));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholerows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholecols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.offset));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.step));

    openCLExecuteKernel(clCxt, &filtering_boxFilter, kernelName, globalThreads, localThreads, args, -1, -1, build_options);
}

static void GPUFilterBox_32F_C4R(const oclMat &src, oclMat &dst,
                          Size &ksize, const Point anchor, const int borderType)
{
    //Normalize the result by default
    float alpha = ksize.height * ksize.width;

    CV_Assert(src.clCxt == dst.clCxt);
    CV_Assert((src.cols == dst.cols) &&
              (src.rows == dst.rows));
    Context *clCxt = src.clCxt;

    String kernelName = "boxFilter_C4_D5";

    char btype[30];

    switch (borderType)
    {
    case 0:
        sprintf(btype, "BORDER_CONSTANT");
        break;
    case 1:
        sprintf(btype, "BORDER_REPLICATE");
        break;
    case 2:
        sprintf(btype, "BORDER_REFLECT");
        break;
    case 3:
        CV_Error(Error::StsUnsupportedFormat, "BORDER_WRAP is not supported!");
        return;
    case 4:
        sprintf(btype, "BORDER_REFLECT_101");
        break;
    }

    char build_options[150];
    sprintf(build_options, "-D anX=%d -D anY=%d -D ksX=%d -D ksY=%d -D %s", anchor.x, anchor.y, ksize.width, ksize.height, btype);

    size_t blockSizeX = 256, blockSizeY = 1;
    size_t gSize = blockSizeX - ksize.width / 2 * 2;
    size_t globalSizeX = (src.cols) % gSize == 0 ? src.cols / gSize * blockSizeX : (src.cols / gSize + 1) * blockSizeX;
    size_t rows_per_thread = 2;
    size_t globalSizeY = ((src.rows + rows_per_thread - 1) / rows_per_thread) % blockSizeY == 0 ? ((src.rows + rows_per_thread - 1) / rows_per_thread) : (((src.rows + rows_per_thread - 1) / rows_per_thread) / blockSizeY + 1) * blockSizeY;


    size_t globalThreads[3] = { globalSizeX, globalSizeY, 1};
    size_t localThreads[3]  = { blockSizeX, blockSizeY, 1};

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), &src.data));
    args.push_back(std::make_pair(sizeof(cl_mem), &dst.data));
    args.push_back(std::make_pair(sizeof(cl_float), (void *)&alpha));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.offset));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholerows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholecols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.offset));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.step));

    openCLExecuteKernel(clCxt, &filtering_boxFilter, kernelName, globalThreads, localThreads, args, -1, -1, build_options);
}


Ptr<BaseFilter_GPU> cv::ocl::getBoxFilter_GPU(int srcType, int dstType,
        const Size &ksize, Point anchor, int borderType)
{
    static const FilterBox_t FilterBox_callers[2][5] = {{0, GPUFilterBox_8u_C1R, 0, GPUFilterBox_8u_C4R, GPUFilterBox_8u_C4R},
        {0, GPUFilterBox_32F_C1R, 0, GPUFilterBox_32F_C4R, GPUFilterBox_32F_C4R}
    };
    //Remove this check if more data types need to be supported.
    CV_Assert((srcType == CV_8UC1 || srcType == CV_8UC3 || srcType == CV_8UC4 || srcType == CV_32FC1 ||
               srcType == CV_32FC3 || srcType == CV_32FC4) && dstType == srcType);

    normalizeAnchor(anchor, ksize);

    return Ptr<BaseFilter_GPU>(new GPUBoxFilter(ksize, anchor,
                               borderType, FilterBox_callers[(CV_MAT_DEPTH(srcType) == CV_32F)][CV_MAT_CN(srcType)]));
}

Ptr<FilterEngine_GPU> cv::ocl::createBoxFilter_GPU(int srcType, int dstType,
        const Size &ksize, const Point &anchor, int borderType)
{
    Ptr<BaseFilter_GPU> boxFilter = getBoxFilter_GPU(srcType, dstType, ksize, anchor, borderType);
    return createFilter2D_GPU(boxFilter);
}

void cv::ocl::boxFilter(const oclMat &src, oclMat &dst, int ddepth, Size ksize,
                        Point anchor, int borderType)
{
    int sdepth = src.depth(), cn = src.channels();

    if (ddepth < 0)
    {
        ddepth = sdepth;
    }

    dst.create(src.size(), CV_MAKETYPE(ddepth, cn));

    Ptr<FilterEngine_GPU> f = createBoxFilter_GPU(src.type(),
                              dst.type(), ksize, anchor, borderType);
    f->apply(src, dst);
}

namespace
{
typedef void (*gpuFilter1D_t)(const oclMat &src, const oclMat &dst, oclMat kernel, int ksize, int anchor, int bordertype);

class GpuLinearRowFilter : public BaseRowFilter_GPU
{
public:
    GpuLinearRowFilter(int ksize_, int anchor_, const oclMat &kernel_, gpuFilter1D_t func_, int bordertype_) :
        BaseRowFilter_GPU(ksize_, anchor_, bordertype_), kernel(kernel_), func(func_) {}

    virtual void operator()(const oclMat &src, oclMat &dst)
    {
        func(src, dst, kernel, ksize, anchor, bordertype);
    }

    oclMat kernel;
    gpuFilter1D_t func;
};
}

template <typename T> struct index_and_sizeof;
template <> struct index_and_sizeof<uchar>
{
    enum { index = 1 };
};
template <> struct index_and_sizeof<char>
{
    enum { index = 2 };
};
template <> struct index_and_sizeof<ushort>
{
    enum { index = 3 };
};
template <> struct index_and_sizeof<short>
{
    enum { index = 4 };
};
template <> struct index_and_sizeof<int>
{
    enum { index = 5 };
};
template <> struct index_and_sizeof<float>
{
    enum { index = 6 };
};

template <typename T>
void linearRowFilter_gpu(const oclMat &src, const oclMat &dst, oclMat mat_kernel, int ksize, int anchor, int bordertype)
{
    Context *clCxt = src.clCxt;
    int channels = src.oclchannels();

    size_t localThreads[3] = {16, 16, 1};
    String kernelName = "row_filter";

    char btype[30];

    switch (bordertype)
    {
    case 0:
        sprintf(btype, "BORDER_CONSTANT");
        break;
    case 1:
        sprintf(btype, "BORDER_REPLICATE");
        break;
    case 2:
        sprintf(btype, "BORDER_REFLECT");
        break;
    case 3:
        sprintf(btype, "BORDER_WRAP");
        break;
    case 4:
        sprintf(btype, "BORDER_REFLECT_101");
        break;
    }

    char compile_option[128];
    sprintf(compile_option, "-D RADIUSX=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d -D %s", anchor, (int)localThreads[0], (int)localThreads[1], channels, btype);

    size_t globalThreads[3];
    globalThreads[1] = (dst.rows + localThreads[1] - 1) / localThreads[1] * localThreads[1];
    globalThreads[2] = (1 + localThreads[2] - 1) / localThreads[2] * localThreads[2];

    if (src.depth() == CV_8U)
    {
        switch (channels)
        {
        case 1:
        case 3:
            globalThreads[0] = ((dst.cols + 4) / 4 + localThreads[0] - 1) / localThreads[0] * localThreads[0];
            break;
        case 2:
            globalThreads[0] = ((dst.cols + 1) / 2 + localThreads[0] - 1) / localThreads[0] * localThreads[0];
            break;
        case 4:
            globalThreads[0] = (dst.cols + localThreads[0] - 1) / localThreads[0] * localThreads[0];
            break;
        }
    }
    else
    {
        globalThreads[0] = (dst.cols + localThreads[0] - 1) / localThreads[0] * localThreads[0];
    }

    //sanity checks
    CV_Assert(clCxt == dst.clCxt);
    CV_Assert(src.cols == dst.cols);
    CV_Assert(src.oclchannels() == dst.oclchannels());
    CV_Assert(ksize == (anchor << 1) + 1);
    int src_pix_per_row, dst_pix_per_row;
    int src_offset_x, src_offset_y;//, dst_offset_in_pixel;
    src_pix_per_row = src.step / src.elemSize();
    src_offset_x = (src.offset % src.step) / src.elemSize();
    src_offset_y = src.offset / src.step;
    dst_pix_per_row = dst.step / dst.elemSize();
    //dst_offset_in_pixel = dst.offset / dst.elemSize();
    int ridusy = (dst.rows - src.rows) >> 1;
    std::vector<std::pair<size_t , const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), &src.data));
    args.push_back(std::make_pair(sizeof(cl_mem), &dst.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholecols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholerows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src_pix_per_row));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src_offset_x));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src_offset_y));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst_pix_per_row));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&ridusy));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&mat_kernel.data));

    openCLExecuteKernel(clCxt, &filter_sep_row, kernelName, globalThreads, localThreads, args, channels, src.depth(), compile_option);
}

Ptr<BaseRowFilter_GPU> cv::ocl::getLinearRowFilter_GPU(int srcType, int /*bufType*/, const Mat &rowKernel, int anchor, int bordertype)
{
    static const gpuFilter1D_t gpuFilter1D_callers[6] =
    {
        linearRowFilter_gpu<uchar>,
        linearRowFilter_gpu<char>,
        linearRowFilter_gpu<ushort>,
        linearRowFilter_gpu<short>,
        linearRowFilter_gpu<int>,
        linearRowFilter_gpu<float>
    };

    Mat temp = rowKernel.reshape(1, 1);
    oclMat mat_kernel(temp);


    int ksize = temp.cols;

    //CV_Assert(ksize < 16);

    normalizeAnchor(anchor, ksize);

    return Ptr<BaseRowFilter_GPU>(new GpuLinearRowFilter(ksize, anchor, mat_kernel,
                                  gpuFilter1D_callers[CV_MAT_DEPTH(srcType)], bordertype));
}

namespace
{
class GpuLinearColumnFilter : public BaseColumnFilter_GPU
{
public:
    GpuLinearColumnFilter(int ksize_, int anchor_, const oclMat &kernel_, gpuFilter1D_t func_, int bordertype_) :
        BaseColumnFilter_GPU(ksize_, anchor_, bordertype_), kernel(kernel_), func(func_) {}

    virtual void operator()(const oclMat &src, oclMat &dst)
    {
        func(src, dst, kernel, ksize, anchor, bordertype);
    }

    oclMat kernel;
    gpuFilter1D_t func;
};
}

template <typename T>
void linearColumnFilter_gpu(const oclMat &src, const oclMat &dst, oclMat mat_kernel, int ksize, int anchor, int bordertype)
{
    Context *clCxt = src.clCxt;
    int channels = src.oclchannels();

    size_t localThreads[3] = {16, 16, 1};
    String kernelName = "col_filter";

    char btype[30];

    switch (bordertype)
    {
    case 0:
        sprintf(btype, "BORDER_CONSTANT");
        break;
    case 1:
        sprintf(btype, "BORDER_REPLICATE");
        break;
    case 2:
        sprintf(btype, "BORDER_REFLECT");
        break;
    case 3:
        sprintf(btype, "BORDER_WRAP");
        break;
    case 4:
        sprintf(btype, "BORDER_REFLECT_101");
        break;
    }

    char compile_option[256];


    size_t globalThreads[3];
    globalThreads[1] = (dst.rows + localThreads[1] - 1) / localThreads[1] * localThreads[1];
    globalThreads[2] = (1 + localThreads[2] - 1) / localThreads[2] * localThreads[2];

    if (dst.depth() == CV_8U)
    {
        switch (channels)
        {
        case 1:
            globalThreads[0] = (dst.cols + localThreads[0] - 1) / localThreads[0] * localThreads[0];
            sprintf(compile_option, "-D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d -D %s -D GENTYPE_SRC=%s -D GENTYPE_DST=%s -D convert_to_DST=%s",
                    anchor, (int)localThreads[0], (int)localThreads[1], channels, btype, "float", "uchar", "convert_uchar_sat");
            break;
        case 2:
            globalThreads[0] = ((dst.cols + 1) / 2 + localThreads[0] - 1) / localThreads[0] * localThreads[0];
            sprintf(compile_option, "-D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d -D %s -D GENTYPE_SRC=%s -D GENTYPE_DST=%s -D convert_to_DST=%s",
                    anchor, (int)localThreads[0], (int)localThreads[1], channels, btype, "float2", "uchar2", "convert_uchar2_sat");
            break;
        case 3:
        case 4:
            globalThreads[0] = (dst.cols + localThreads[0] - 1) / localThreads[0] * localThreads[0];
            sprintf(compile_option, "-D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d -D %s -D GENTYPE_SRC=%s -D GENTYPE_DST=%s -D convert_to_DST=%s",
                    anchor, (int)localThreads[0], (int)localThreads[1], channels, btype, "float4", "uchar4", "convert_uchar4_sat");
            break;
        }
    }
    else
    {
        globalThreads[0] = (dst.cols + localThreads[0] - 1) / localThreads[0] * localThreads[0];

        switch (dst.type())
        {
        case CV_32SC1:
            sprintf(compile_option, "-D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d -D %s -D GENTYPE_SRC=%s -D GENTYPE_DST=%s -D convert_to_DST=%s",
                    anchor, (int)localThreads[0], (int)localThreads[1], channels, btype, "float", "int", "convert_int_sat");
            break;
        case CV_32SC3:
        case CV_32SC4:
            sprintf(compile_option, "-D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d -D %s -D GENTYPE_SRC=%s -D GENTYPE_DST=%s -D convert_to_DST=%s",
                    anchor, (int)localThreads[0], (int)localThreads[1], channels, btype, "float4", "int4", "convert_int4_sat");
            break;
        case CV_32FC1:
            sprintf(compile_option, "-D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d -D %s -D GENTYPE_SRC=%s -D GENTYPE_DST=%s -D convert_to_DST=%s",
                    anchor, (int)localThreads[0], (int)localThreads[1], channels, btype, "float", "float", "");
            break;
        case CV_32FC3:
        case CV_32FC4:
            sprintf(compile_option, "-D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d -D %s -D GENTYPE_SRC=%s -D GENTYPE_DST=%s -D convert_to_DST=%s",
                    anchor, (int)localThreads[0], (int)localThreads[1], channels, btype, "float4", "float4", "");
            break;
        }
    }

    //sanity checks
    CV_Assert(clCxt == dst.clCxt);
    CV_Assert(src.cols == dst.cols);
    CV_Assert(src.oclchannels() == dst.oclchannels());
    CV_Assert(ksize == (anchor << 1) + 1);
    int src_pix_per_row, dst_pix_per_row;
    //int src_offset_x, src_offset_y;
    int dst_offset_in_pixel;
    src_pix_per_row = src.step / src.elemSize();
    //src_offset_x = (src.offset % src.step) / src.elemSize();
    //src_offset_y = src.offset / src.step;
    dst_pix_per_row = dst.step / dst.elemSize();
    dst_offset_in_pixel = dst.offset / dst.elemSize();

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), &src.data));
    args.push_back(std::make_pair(sizeof(cl_mem), &dst.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholecols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.wholerows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src_pix_per_row));
    //args.push_back(std::make_pair(sizeof(cl_int),(void*)&src_offset_x));
    //args.push_back(std::make_pair(sizeof(cl_int),(void*)&src_offset_y));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst_pix_per_row));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst_offset_in_pixel));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&mat_kernel.data));

    openCLExecuteKernel(clCxt, &filter_sep_col, kernelName, globalThreads, localThreads, args, -1, -1, compile_option);
}

Ptr<BaseColumnFilter_GPU> cv::ocl::getLinearColumnFilter_GPU(int /*bufType*/, int dstType, const Mat &columnKernel, int anchor, int bordertype, double /*delta*/)
{
    static const gpuFilter1D_t gpuFilter1D_callers[6] =
    {
        linearColumnFilter_gpu<uchar>,
        linearColumnFilter_gpu<char>,
        linearColumnFilter_gpu<ushort>,
        linearColumnFilter_gpu<short>,
        linearColumnFilter_gpu<int>,
        linearColumnFilter_gpu<float>
    };
    /*
    CV_Assert(dstType == CV_8UC4 || dstType == CV_8SC4 || dstType == CV_16UC2 ||
    dstType == CV_16SC2 || dstType == CV_32SC1 || dstType == CV_32FC1);
    CV_Assert(bufType == CV_8UC4 || bufType == CV_8SC4 || bufType == CV_16UC2 ||
    bufType == CV_16SC2 || bufType == CV_32SC1 || bufType == CV_32FC1);

    Mat temp(columnKernel.size(), CV_32SC1);
    columnKernel.convertTo(temp, CV_32SC1);
    Mat cont_krnl = temp.reshape(1, 1);
    */
    Mat temp = columnKernel.reshape(1, 1);
    oclMat mat_kernel(temp);

    int ksize = temp.cols;

    //CV_Assert(ksize < 16);

    normalizeAnchor(anchor, ksize);

    return Ptr<BaseColumnFilter_GPU>(new GpuLinearColumnFilter(ksize, anchor, mat_kernel,
                                     gpuFilter1D_callers[CV_MAT_DEPTH(dstType)], bordertype));
}

Ptr<FilterEngine_GPU> cv::ocl::createSeparableLinearFilter_GPU(int srcType, int dstType,
        const Mat &rowKernel, const Mat &columnKernel, const Point &anchor, double delta, int bordertype)
{
    int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(dstType);
    int cn = CV_MAT_CN(srcType);
    int bdepth = std::max(std::max(sdepth, ddepth), CV_32F);
    int bufType = CV_MAKETYPE(bdepth, cn);

    Ptr<BaseRowFilter_GPU> rowFilter = getLinearRowFilter_GPU(srcType, bufType, rowKernel, anchor.x, bordertype);
    Ptr<BaseColumnFilter_GPU> columnFilter = getLinearColumnFilter_GPU(bufType, dstType, columnKernel, anchor.y, bordertype, delta);

    return createSeparableFilter_GPU(rowFilter, columnFilter);
}

void cv::ocl::sepFilter2D(const oclMat &src, oclMat &dst, int ddepth, const Mat &kernelX, const Mat &kernelY, Point anchor, double delta, int bordertype)
{
    if ((dst.cols != dst.wholecols) || (dst.rows != dst.wholerows)) //has roi
    {
        if ((bordertype & cv::BORDER_ISOLATED) != 0)
        {
            bordertype &= ~cv::BORDER_ISOLATED;

            if ((bordertype != cv::BORDER_CONSTANT) &&
                    (bordertype != cv::BORDER_REPLICATE))
            {
                CV_Error(Error::StsBadArg, "unsupported border type");
            }
        }
    }

    if (ddepth < 0)
    {
        ddepth = src.depth();
    }

    //CV_Assert(ddepth == src.depth());
    dst.create(src.size(), CV_MAKETYPE(ddepth, src.channels()));

    Ptr<FilterEngine_GPU> f = createSeparableLinearFilter_GPU(src.type(), dst.type(), kernelX, kernelY, anchor, delta, bordertype);
    f->apply(src, dst);
}

Ptr<FilterEngine_GPU> cv::ocl::createDerivFilter_GPU(int srcType, int dstType, int dx, int dy, int ksize, int borderType)
{
    Mat kx, ky;
    getDerivKernels(kx, ky, dx, dy, ksize, false, CV_32F);
    return createSeparableLinearFilter_GPU(srcType, dstType,
                                           kx, ky, Point(-1, -1), 0, borderType);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Deriv Filter
void cv::ocl::Sobel(const oclMat &src, oclMat &dst, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType)
{
    Mat kx, ky;
    getDerivKernels(kx, ky, dx, dy, ksize, false, CV_32F);

    if (scale != 1)
    {
        // usually the smoothing part is the slowest to compute,
        // so try to scale it instead of the faster differenciating part
        if (dx == 0)
        {
            kx *= scale;
        }
        else
        {
            ky *= scale;
        }
    }

    // Mat kx_, ky_;
    //ky.convertTo(ky_,CV_32S,1<<8);
    //kx.convertTo(kx_,CV_32S,1<<8);

    sepFilter2D(src, dst, ddepth, kx, ky, Point(-1, -1), delta, borderType);
}

void cv::ocl::Scharr(const oclMat &src, oclMat &dst, int ddepth, int dx, int dy, double scale, double delta , int bordertype)
{
    Mat kx, ky;
    getDerivKernels(kx, ky, dx, dy, -1, false, CV_32F);

    if (scale != 1)
    {
        // usually the smoothing part is the slowest to compute,
        // so try to scale it instead of the faster differenciating part
        if (dx == 0)
        {
            kx *= scale;
        }
        else
        {
            ky *= scale;
        }
    }

    // Mat kx_, ky_;
    //ky.convertTo(ky_,CV_32S,1<<8);
    //kx.convertTo(kx_,CV_32S,1<<8);

    sepFilter2D(src, dst, ddepth, kx, ky, Point(-1, -1), delta, bordertype);
}

void cv::ocl::Laplacian(const oclMat &src, oclMat &dst, int ddepth, int ksize, double scale)
{
    if (!src.clCxt->supportsFeature(Context::CL_DOUBLE) && src.type() == CV_64F)
    {
        CV_Error(Error::GpuNotSupported, "Selected device don't support double\r\n");
        return;
    }

    CV_Assert(ksize == 1 || ksize == 3);

    static const int K[2][9] =
    {
        {0, 1, 0, 1, -4, 1, 0, 1, 0},
        {2, 0, 2, 0, -8, 0, 2, 0, 2}
    };
    Mat kernel(3, 3, CV_32S, (void *)K[ksize == 3]);

    if (scale != 1)
    {
        kernel *= scale;
    }

    filter2D(src, dst, ddepth, kernel, Point(-1, -1));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Gaussian Filter

Ptr<FilterEngine_GPU> cv::ocl::createGaussianFilter_GPU(int type, Size ksize, double sigma1, double sigma2, int bordertype)
{
    int depth = CV_MAT_DEPTH(type);

    if (sigma2 <= 0)
    {
        sigma2 = sigma1;
    }

    // automatic detection of kernel size from sigma
    if (ksize.width <= 0 && sigma1 > 0)
    {
        ksize.width = cvRound(sigma1 * (depth == CV_8U ? 3 : 4) * 2 + 1) | 1;
    }

    if (ksize.height <= 0 && sigma2 > 0)
    {
        ksize.height = cvRound(sigma2 * (depth == CV_8U ? 3 : 4) * 2 + 1) | 1;
    }

    CV_Assert(ksize.width > 0 && ksize.width % 2 == 1 && ksize.height > 0 && ksize.height % 2 == 1);

    sigma1 = std::max(sigma1, 0.0);
    sigma2 = std::max(sigma2, 0.0);

    Mat kx = getGaussianKernel(ksize.width, sigma1, std::max(depth, CV_32F));
    Mat ky;

    if (ksize.height == ksize.width && std::abs(sigma1 - sigma2) < DBL_EPSILON)
    {
        ky = kx;
    }
    else
    {
        ky = getGaussianKernel(ksize.height, sigma2, std::max(depth, CV_32F));
    }

    //Mat kx_, ky_;
    //kx.convertTo(kx_,CV_32S,1<<8);
    //ky.convertTo(ky_,CV_32S,1<<8);
    return createSeparableLinearFilter_GPU(type, type, kx, ky, Point(-1, -1), 0.0, bordertype);
}

void cv::ocl::GaussianBlur(const oclMat &src, oclMat &dst, Size ksize, double sigma1, double sigma2, int bordertype)
{
    if (ksize.width == 1 && ksize.height == 1)
    {
        src.copyTo(dst);
        return;
    }

    if ((dst.cols != dst.wholecols) || (dst.rows != dst.wholerows)) //has roi
    {
        if ((bordertype & cv::BORDER_ISOLATED) != 0)
        {
            bordertype &= ~cv::BORDER_ISOLATED;

            if ((bordertype != cv::BORDER_CONSTANT) &&
                    (bordertype != cv::BORDER_REPLICATE))
            {
                CV_Error(Error::StsBadArg, "unsupported border type");
            }
        }
    }

    dst.create(src.size(), src.type());

    if (bordertype != BORDER_CONSTANT)
    {
        if (src.rows == 1)
        {
            ksize.height = 1;
        }

        if (src.cols == 1)
        {
            ksize.width = 1;
        }
    }

    Ptr<FilterEngine_GPU> f = createGaussianFilter_GPU(src.type(), ksize, sigma1, sigma2, bordertype);
    f->apply(src, dst);
}
