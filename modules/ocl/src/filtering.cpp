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
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Zero Lin, Zero.Lin@amd.com
//    Zhang Ying, zhangying913@gmail.com
//    Yao Wang, bitwangyaoyao@gmail.com
//    Harris Gasparakis, harris.gasparakis@amd.com
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
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

namespace
{
inline void normalizeAnchor(int &anchor, int ksize)
{
    if (anchor < 0)
        anchor = ksize >> 1;

    CV_Assert(0 <= anchor && anchor < ksize);
}

inline void normalizeAnchor(Point &anchor, const Size &ksize)
{
    normalizeAnchor(anchor.x, ksize.width);
    normalizeAnchor(anchor.y, ksize.height);
}

inline void normalizeROI(Rect &roi, const Size &ksize, const Point &/*anchor*/, const Size &src_size)
{
    if (roi == Rect(0, 0, -1, -1))
        roi = Rect(0, 0, src_size.width, src_size.height);

    CV_Assert(ksize.height > 0 && ksize.width > 0 && ((ksize.height & 1) == 1) && ((ksize.width & 1) == 1));
    CV_Assert(roi.x >= 0 && roi.y >= 0 && roi.width <= src_size.width && roi.height <= src_size.height);
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

        // Delete those two clause below which exist before, However, the result is also correct
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
    return makePtr<Filter2DEngine_GPU>(filter2D);
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
    MorphFilter_GPU(const Size &ksize_, const Point &anchor_, const Mat &kernel_, GPUMorfFilter_t func_) :
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

    int srcStep = src.step / src.elemSize();
    int dstStep = dst.step / dst.elemSize();
    int srcOffset = src.offset / src.elemSize();
    int dstOffset = dst.offset / dst.elemSize();

    int srcOffset_x = srcOffset % srcStep;
    int srcOffset_y = srcOffset / srcStep;
    Context *clCxt = src.clCxt;
    String kernelName;
#ifdef ANDROID
    size_t localThreads[3] = {16, 8, 1};
#else
    size_t localThreads[3] = {16, 16, 1};
#endif
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
#ifdef ANDROID
    size_t localThreads[3] = {16, 10, 1};
#else
    size_t localThreads[3] = {16, 16, 1};
#endif
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

Ptr<BaseFilter_GPU> cv::ocl::getMorphologyFilter_GPU(int op, int type, const Mat &_kernel, const Size &ksize, Point anchor)
{
    CV_Assert(op == MORPH_ERODE || op == MORPH_DILATE);
    CV_Assert(type == CV_8UC1 || type == CV_8UC3 || type == CV_8UC4 || type == CV_32FC1 || type == CV_32FC3 || type == CV_32FC4);

    normalizeAnchor(anchor, ksize);
    Mat kernel8U;
    _kernel.convertTo(kernel8U, CV_8U);
    Mat kernel = kernel8U.reshape(1, 1);

    bool noZero = true;
    for(int i = 0; i < kernel.rows * kernel.cols; ++i)
        if(kernel.at<uchar>(i) != 1)
            noZero = false;

    MorphFilter_GPU* mfgpu = new MorphFilter_GPU(ksize, anchor, kernel, op == MORPH_ERODE ? GPUErode : GPUDilate);
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

        for (int i = 1; i < iters; ++i)
        {
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

    return makePtr<MorphologyFilterEngine_GPU>(filter2D, iterations);
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
        kernel = getStructuringElement(MORPH_RECT, Size(ksize.width + (iterations - 1) * (ksize.width - 1),
                                       ksize.height + (iterations - 1) * (ksize.height - 1)), anchor);
        iterations = 1;
    }
    else
        kernel = _kernel;

    Ptr<MorphologyFilterEngine_GPU> f = createMorphologyFilter_GPU(op, src.type(), kernel, anchor, iterations)
            .staticCast<MorphologyFilterEngine_GPU>();

    f->apply(src, dst);
}
}

void cv::ocl::erode(const oclMat &src, oclMat &dst, const Mat &kernel, Point anchor, int iterations,
                    int borderType, const Scalar &borderValue)
{
    bool allZero = true;

    for (int i = 0; i < kernel.rows * kernel.cols; ++i)
        if (kernel.data[i] != 0)
            allZero = false;

    if (allZero)
        kernel.data[0] = 1;

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
typedef void (*GPUFilter2D_t)(const oclMat & , oclMat & , const Mat & , const Size &, const Point&, const int);

class LinearFilter_GPU : public BaseFilter_GPU
{
public:
    LinearFilter_GPU(const Size &ksize_, const Point &anchor_, const Mat &kernel_, GPUFilter2D_t func_,
                     int borderType_) :
        BaseFilter_GPU(ksize_, anchor_, borderType_), kernel(kernel_), func(func_) {}

    virtual void operator()(const oclMat &src, oclMat &dst)
    {
        func(src, dst, kernel, ksize, anchor, borderType) ;
    }

    Mat kernel;
    GPUFilter2D_t func;
};
}

// prepare kernel: transpose and make double rows (+align). Returns size of aligned row
// Samples:
//        a b c
// Input: d e f
//        g h i
// Output, last two zeros is the alignment:
// a d g a d g 0 0
// b e h b e h 0 0
// c f i c f i 0 0
template <typename T>
static int _prepareKernelFilter2D(std::vector<T>& data, const Mat &kernel)
{
    Mat _kernel; kernel.convertTo(_kernel, DataDepth<T>::value);
    int size_y_aligned = roundUp(kernel.rows * 2, 4);
    data.clear(); data.resize(size_y_aligned * kernel.cols, 0);
    for (int x = 0; x < kernel.cols; x++)
    {
        for (int y = 0; y < kernel.rows; y++)
        {
            data[x * size_y_aligned + y] = _kernel.at<T>(y, x);
            data[x * size_y_aligned + y + kernel.rows] = _kernel.at<T>(y, x);
        }
    }
    return size_y_aligned;
}

static void GPUFilter2D(const oclMat &src, oclMat &dst, const Mat &kernel,
    const Size &ksize, const Point& anchor, const int borderType)
{
    CV_Assert(src.clCxt == dst.clCxt);
    CV_Assert((src.cols == dst.cols) &&
              (src.rows == dst.rows));
    CV_Assert(src.oclchannels() == dst.oclchannels());

    CV_Assert(kernel.cols == ksize.width && kernel.rows == ksize.height);
    CV_Assert(kernel.channels() == 1);

    CV_Assert(anchor.x >= 0 && anchor.x < kernel.cols);
    CV_Assert(anchor.y >= 0 && anchor.y < kernel.rows);

    bool useDouble = src.depth() == CV_64F;

    std::vector<float> kernelDataFloat;
    std::vector<double> kernelDataDouble;
    int kernel_size_y2_aligned = useDouble ?
            _prepareKernelFilter2D<double>(kernelDataDouble, kernel)
            : _prepareKernelFilter2D<float>(kernelDataFloat, kernel);
    oclMat oclKernelParameter;
    if (useDouble)
    {
        oclKernelParameter.createEx(1, kernelDataDouble.size(), CV_64FC1, DEVICE_MEM_R_ONLY, DEVICE_MEM_DEFAULT);
        openCLMemcpy2D(src.clCxt, oclKernelParameter.data, kernelDataDouble.size()*sizeof(double),
                &kernelDataDouble[0], kernelDataDouble.size()*sizeof(double),
                kernelDataDouble.size()*sizeof(double), 1, clMemcpyHostToDevice);
    }
    else
    {
        oclKernelParameter.createEx(1, kernelDataFloat.size(), CV_32FC1, DEVICE_MEM_R_ONLY, DEVICE_MEM_DEFAULT);
        openCLMemcpy2D(src.clCxt, oclKernelParameter.data, kernelDataFloat.size()*sizeof(float),
                &kernelDataFloat[0], kernelDataFloat.size()*sizeof(float),
                kernelDataFloat.size()*sizeof(float), 1, clMemcpyHostToDevice);
    }

    size_t tryWorkItems = src.clCxt->getDeviceInfo().maxWorkItemSizes[0];
    do {
        size_t BLOCK_SIZE = tryWorkItems;
        while (BLOCK_SIZE > 32 && BLOCK_SIZE >= (size_t)ksize.width * 2 && BLOCK_SIZE > (size_t)src.cols * 2)
            BLOCK_SIZE /= 2;
#if 1 // TODO Mode with several blocks requires a much more VGPRs, so this optimization is not actual for the current devices
        size_t BLOCK_SIZE_Y = 1;
#else
        size_t BLOCK_SIZE_Y = 8; // TODO Check heuristic value on devices
        while (BLOCK_SIZE_Y < BLOCK_SIZE / 8 && BLOCK_SIZE_Y * src.clCxt->getDeviceInfo().maxComputeUnits * 32 < (size_t)src.rows)
            BLOCK_SIZE_Y *= 2;
#endif

        CV_Assert((size_t)ksize.width <= BLOCK_SIZE);

        bool isIsolatedBorder = (borderType & BORDER_ISOLATED) != 0;

        std::vector<std::pair<size_t , const void *> > args;

        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data));
        cl_uint stepBytes = src.step;
        args.push_back( std::make_pair( sizeof(cl_uint), (void *)&stepBytes));
        int offsetXBytes = src.offset % src.step;
        int offsetX = offsetXBytes / src.elemSize();
        CV_Assert((int)(offsetX * src.elemSize()) == offsetXBytes);
        int offsetY = src.offset / src.step;
        int endX = (offsetX + src.cols);
        int endY = (offsetY + src.rows);
        cl_int rect[4] = {offsetX, offsetY, endX, endY};
        if (!isIsolatedBorder)
        {
            rect[2] = src.wholecols;
            rect[3] = src.wholerows;
        }
        args.push_back( std::make_pair( sizeof(cl_int)*4, (void *)&rect[0]));

        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data));
        cl_uint _stepBytes = dst.step;
        args.push_back( std::make_pair( sizeof(cl_uint), (void *)&_stepBytes));
        int _offsetXBytes = dst.offset % dst.step;
        int _offsetX = _offsetXBytes / dst.elemSize();
        CV_Assert((int)(_offsetX * dst.elemSize()) == _offsetXBytes);
        int _offsetY = dst.offset / dst.step;
        int _endX = (_offsetX + dst.cols);
        int _endY = (_offsetY + dst.rows);
        cl_int _rect[4] = {_offsetX, _offsetY, _endX, _endY};
        args.push_back( std::make_pair( sizeof(cl_int)*4, (void *)&_rect[0]));

        float borderValue[4] = {0, 0, 0, 0}; // DON'T move into 'if' body
        double borderValueDouble[4] = {0, 0, 0, 0}; // DON'T move into 'if' body
        if ((borderType & ~BORDER_ISOLATED) == BORDER_CONSTANT)
        {
            if (useDouble)
                args.push_back( std::make_pair( sizeof(double) * src.oclchannels(), (void *)&borderValue[0]));
            else
                args.push_back( std::make_pair( sizeof(float) * src.oclchannels(), (void *)&borderValueDouble[0]));
        }

        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&oclKernelParameter.data));

        const char* btype = NULL;

        switch (borderType & ~BORDER_ISOLATED)
        {
        case BORDER_CONSTANT:
            btype = "BORDER_CONSTANT";
            break;
        case BORDER_REPLICATE:
            btype = "BORDER_REPLICATE";
            break;
        case BORDER_REFLECT:
            btype = "BORDER_REFLECT";
            break;
        case BORDER_WRAP:
            CV_Error(CV_StsUnsupportedFormat, "BORDER_WRAP is not supported!");
            return;
        case BORDER_REFLECT101:
            btype = "BORDER_REFLECT_101";
            break;
        }

        int requiredTop = anchor.y;
        int requiredLeft = BLOCK_SIZE; // not this: anchor.x;
        int requiredBottom = ksize.height - 1 - anchor.y;
        int requiredRight = BLOCK_SIZE; // not this: ksize.width - 1 - anchor.x;
        int h = isIsolatedBorder ? src.rows : src.wholerows;
        int w = isIsolatedBorder ? src.cols : src.wholecols;
        bool extra_extrapolation = h < requiredTop || h < requiredBottom || w < requiredLeft || w < requiredRight;

        char build_options[1024];
        sprintf(build_options, "-D LOCAL_SIZE=%d -D BLOCK_SIZE_Y=%d -D DATA_DEPTH=%d -D DATA_CHAN=%d -D USE_DOUBLE=%d "
                "-D ANCHOR_X=%d -D ANCHOR_Y=%d -D KERNEL_SIZE_X=%d -D KERNEL_SIZE_Y=%d -D KERNEL_SIZE_Y2_ALIGNED=%d "
                "-D %s -D %s -D %s",
                (int)BLOCK_SIZE, (int)BLOCK_SIZE_Y,
                src.depth(), src.oclchannels(), useDouble ? 1 : 0,
                anchor.x, anchor.y, ksize.width, ksize.height, kernel_size_y2_aligned,
                btype,
                extra_extrapolation ? "EXTRA_EXTRAPOLATION" : "NO_EXTRA_EXTRAPOLATION",
                isIsolatedBorder ? "BORDER_ISOLATED" : "NO_BORDER_ISOLATED");

        size_t lt[3] = {BLOCK_SIZE, 1, 1};
        size_t gt[3] = {divUp(dst.cols, BLOCK_SIZE - (ksize.width - 1)) * BLOCK_SIZE, divUp(dst.rows, BLOCK_SIZE_Y), 1};

        cl_kernel kernel = openCLGetKernelFromSource(src.clCxt, &filtering_filter2D, "filter2D", -1, -1, build_options);

        size_t kernelWorkGroupSize;
        openCLSafeCall(clGetKernelWorkGroupInfo(kernel, getClDeviceID(src.clCxt),
                                                CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernelWorkGroupSize, 0));
        if (lt[0] > kernelWorkGroupSize)
        {
            clReleaseKernel(kernel);
            CV_Assert(BLOCK_SIZE > kernelWorkGroupSize);
            tryWorkItems = kernelWorkGroupSize;
            continue;
        }

        openCLExecuteKernel(src.clCxt, kernel, gt, lt, args); // kernel will be released here
    } while (false);
}

Ptr<BaseFilter_GPU> cv::ocl::getLinearFilter_GPU(int /*srcType*/, int /*dstType*/, const Mat &kernel, const Size &ksize,
        const Point &anchor, int borderType)
{
    Point norm_archor = anchor;
    normalizeAnchor(norm_archor, ksize);

    return Ptr<BaseFilter_GPU>(new LinearFilter_GPU(ksize, norm_archor, kernel, GPUFilter2D,
                               borderType));
}

Ptr<FilterEngine_GPU> cv::ocl::createLinearFilter_GPU(int srcType, int dstType, const Mat &kernel, const Point &anchor,
        int borderType)
{
    Size ksize = kernel.size(); // TODO remove duplicated parameter
    Ptr<BaseFilter_GPU> linearFilter = getLinearFilter_GPU(srcType, dstType, kernel, ksize, anchor, borderType);

    return createFilter2D_GPU(linearFilter);
}

void cv::ocl::filter2D(const oclMat &src, oclMat &dst, int ddepth, const Mat &kernel, Point anchor, double delta, int borderType)
{
    CV_Assert(delta == 0);

    if (ddepth < 0)
        ddepth = src.depth();

    dst.create(src.size(), CV_MAKETYPE(ddepth, src.channels()));

    Ptr<FilterEngine_GPU> f = createLinearFilter_GPU(src.type(), dst.type(), kernel, anchor, borderType);
    f->apply(src, dst);
}

const int optimizedSepFilterLocalSize = 16;
static void sepFilter2D_SinglePass(const oclMat &src, oclMat &dst,
                                   const Mat &row_kernel, const Mat &col_kernel, int bordertype = BORDER_DEFAULT)
{
    size_t lt2[3] = {optimizedSepFilterLocalSize, optimizedSepFilterLocalSize, 1};
    size_t gt2[3] = {lt2[0]*(1 + (src.cols-1) / lt2[0]), lt2[1]*(1 + (src.rows-1) / lt2[1]), 1};

    unsigned int src_pitch = src.step;
    unsigned int dst_pitch = dst.step;

    int src_offset_x = (src.offset % src.step) / src.elemSize();
    int src_offset_y = src.offset / src.step;

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&src.data ));
    args.push_back( std::make_pair( sizeof(cl_uint) , (void *)&src_pitch ));

    args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&src_offset_x ));
    args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&src_offset_y ));

    args.push_back( std::make_pair( sizeof(cl_mem)  , (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&dst.offset ));
    args.push_back( std::make_pair( sizeof(cl_uint) , (void *)&dst_pitch ));

    args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&src.wholecols ));
    args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&src.wholerows ));

    args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&dst.cols ));
    args.push_back( std::make_pair( sizeof(cl_int)  , (void *)&dst.rows ));

    String option = cv::format("-D BLK_X=%d -D BLK_Y=%d -D RADIUSX=%d -D RADIUSY=%d",(int)lt2[0], (int)lt2[1],
        row_kernel.rows / 2, col_kernel.rows / 2 );

    option += " -D KERNEL_MATRIX_X=";
    for(int i=0; i<row_kernel.rows; i++)
        option += cv::format("0x%x,", *reinterpret_cast<const unsigned int*>( &row_kernel.at<float>(i) ) );
    option += "0x0";

    option += " -D KERNEL_MATRIX_Y=";
    for(int i=0; i<col_kernel.rows; i++)
        option += cv::format("0x%x,", *reinterpret_cast<const unsigned int*>( &col_kernel.at<float>(i) ) );
    option += "0x0";

    switch(src.type())
    {
    case CV_8UC1:
        option += " -D SRCTYPE=uchar -D CONVERT_SRCTYPE=convert_float -D WORKTYPE=float";
        break;
    case CV_32FC1:
        option += " -D SRCTYPE=float -D CONVERT_SRCTYPE= -D WORKTYPE=float";
        break;
    case CV_8UC2:
        option += " -D SRCTYPE=uchar2 -D CONVERT_SRCTYPE=convert_float2 -D WORKTYPE=float2";
        break;
    case CV_32FC2:
        option += " -D SRCTYPE=float2 -D CONVERT_SRCTYPE= -D WORKTYPE=float2";
        break;
    case CV_8UC3:
        option += " -D SRCTYPE=uchar3 -D CONVERT_SRCTYPE=convert_float3 -D WORKTYPE=float3";
        break;
    case CV_32FC3:
        option += " -D SRCTYPE=float3 -D CONVERT_SRCTYPE= -D WORKTYPE=float3";
        break;
    case CV_8UC4:
        option += " -D SRCTYPE=uchar4 -D CONVERT_SRCTYPE=convert_float4 -D WORKTYPE=float4";
        break;
    case CV_32FC4:
        option += " -D SRCTYPE=float4 -D CONVERT_SRCTYPE= -D WORKTYPE=float4";
        break;
    default:
        CV_Error(CV_StsUnsupportedFormat, "Image type is not supported!");
        break;
    }
    switch(dst.type())
    {
    case CV_8UC1:
        option += " -D DSTTYPE=uchar -D CONVERT_DSTTYPE=convert_uchar_sat";
        break;
    case CV_8UC2:
        option += " -D DSTTYPE=uchar2 -D CONVERT_DSTTYPE=convert_uchar2_sat";
        break;
    case CV_8UC3:
        option += " -D DSTTYPE=uchar3 -D CONVERT_DSTTYPE=convert_uchar3_sat";
        break;
    case CV_8UC4:
        option += " -D DSTTYPE=uchar4 -D CONVERT_DSTTYPE=convert_uchar4_sat";
        break;
    case CV_32FC1:
        option += " -D DSTTYPE=float -D CONVERT_DSTTYPE=";
        break;
    case CV_32FC2:
        option += " -D DSTTYPE=float2 -D CONVERT_DSTTYPE=";
        break;
    case CV_32FC3:
        option += " -D DSTTYPE=float3 -D CONVERT_DSTTYPE=";
        break;
    case CV_32FC4:
        option += " -D DSTTYPE=float4 -D CONVERT_DSTTYPE=";
        break;
    default:
        CV_Error(CV_StsUnsupportedFormat, "Image type is not supported!");
        break;
    }
    switch(bordertype)
    {
    case cv::BORDER_CONSTANT:
        option += " -D BORDER_CONSTANT";
        break;
    case cv::BORDER_REPLICATE:
        option += " -D BORDER_REPLICATE";
        break;
    case cv::BORDER_REFLECT:
        option += " -D BORDER_REFLECT";
        break;
    case cv::BORDER_REFLECT101:
        option += " -D BORDER_REFLECT_101";
        break;
    case cv::BORDER_WRAP:
        option += " -D BORDER_WRAP";
        break;
    default:
        CV_Error(CV_StsBadFlag, "BORDER type is not supported!");
        break;
    }

    openCLExecuteKernel(src.clCxt, &filtering_sep_filter_singlepass, "sep_filter_singlepass", gt2, lt2, args,
        -1, -1, option.c_str() );
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

        int cn = src.oclchannels();
        dstBuf.create(src_size.height + ksize.height - 1, src_size.width, CV_MAKETYPE(CV_32F, cn));

        normalizeROI(roi, ksize, anchor, src_size);

        srcROI = src(roi);
        dstROI = dst(roi);

        (*rowFilter)(srcROI, dstBuf);
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
    return makePtr<SeparableFilterEngine_GPU>(rowFilter, columnFilter);
}

namespace
{
class SingleStepSeparableFilterEngine_GPU : public FilterEngine_GPU
{
public:
    SingleStepSeparableFilterEngine_GPU( const Mat &rowKernel_, const Mat &columnKernel_, const int btype )
    {
        bordertype = btype;
        rowKernel = rowKernel_;
        columnKernel = columnKernel_;
    }

    virtual void apply(const oclMat &src, oclMat &dst, Rect roi = Rect(0, 0, -1, -1))
    {
        normalizeROI(roi, Size(rowKernel.rows, columnKernel.rows), Point(-1,-1), src.size());

        oclMat srcROI = src(roi);
        oclMat dstROI = dst(roi);

        sepFilter2D_SinglePass(src, dst, rowKernel, columnKernel, bordertype);
    }

    Mat rowKernel;
    Mat columnKernel;
    int bordertype;
};
}


static void GPUFilterBox(const oclMat &src, oclMat &dst,
                         Size &ksize, const Point anchor, const int borderType)
{
    //Normalize the result by default
    float alpha = 1.0f / (ksize.height * ksize.width);

    CV_Assert(src.clCxt == dst.clCxt);
    CV_Assert((src.cols == dst.cols) &&
              (src.rows == dst.rows));
    CV_Assert(src.oclchannels() == dst.oclchannels());

    size_t tryWorkItems = src.clCxt->getDeviceInfo().maxWorkItemSizes[0];
    do {
        size_t BLOCK_SIZE = tryWorkItems;
        while (BLOCK_SIZE > 32 && BLOCK_SIZE >= (size_t)ksize.width * 2 && BLOCK_SIZE > (size_t)src.cols * 2)
            BLOCK_SIZE /= 2;
        size_t BLOCK_SIZE_Y = 8; // TODO Check heuristic value on devices
        while (BLOCK_SIZE_Y < BLOCK_SIZE / 8 && BLOCK_SIZE_Y * src.clCxt->getDeviceInfo().maxComputeUnits * 32 < (size_t)src.rows)
            BLOCK_SIZE_Y *= 2;

        CV_Assert((size_t)ksize.width <= BLOCK_SIZE);

        bool isIsolatedBorder = (borderType & BORDER_ISOLATED) != 0;

        std::vector<std::pair<size_t , const void *> > args;

        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data));
        cl_uint stepBytes = src.step;
        args.push_back( std::make_pair( sizeof(cl_uint), (void *)&stepBytes));
        int offsetXBytes = src.offset % src.step;
        int offsetX = offsetXBytes / src.elemSize();
        CV_Assert((int)(offsetX * src.elemSize()) == offsetXBytes);
        int offsetY = src.offset / src.step;
        int endX = (offsetX + src.cols);
        int endY = (offsetY + src.rows);
        cl_int rect[4] = {offsetX, offsetY, endX, endY};
        if (!isIsolatedBorder)
        {
            rect[2] = src.wholecols;
            rect[3] = src.wholerows;
        }
        args.push_back( std::make_pair( sizeof(cl_int)*4, (void *)&rect[0]));

        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data));
        cl_uint _stepBytes = dst.step;
        args.push_back( std::make_pair( sizeof(cl_uint), (void *)&_stepBytes));
        int _offsetXBytes = dst.offset % dst.step;
        int _offsetX = _offsetXBytes / dst.elemSize();
        CV_Assert((int)(_offsetX * dst.elemSize()) == _offsetXBytes);
        int _offsetY = dst.offset / dst.step;
        int _endX = (_offsetX + dst.cols);
        int _endY = (_offsetY + dst.rows);
        cl_int _rect[4] = {_offsetX, _offsetY, _endX, _endY};
        args.push_back( std::make_pair( sizeof(cl_int)*4, (void *)&_rect[0]));

        bool useDouble = src.depth() == CV_64F;

        float borderValue[4] = {0, 0, 0, 0}; // DON'T move into 'if' body
        double borderValueDouble[4] = {0, 0, 0, 0}; // DON'T move into 'if' body
        if ((borderType & ~BORDER_ISOLATED) == BORDER_CONSTANT)
        {
            if (useDouble)
                args.push_back( std::make_pair( sizeof(double) * src.oclchannels(), (void *)&borderValue[0]));
            else
                args.push_back( std::make_pair( sizeof(float) * src.oclchannels(), (void *)&borderValueDouble[0]));
        }

        double alphaDouble = alpha; // DON'T move into 'if' body
        if (useDouble)
            args.push_back( std::make_pair( sizeof(double), (void *)&alphaDouble));
        else
            args.push_back( std::make_pair( sizeof(float), (void *)&alpha));

        const char* btype = NULL;

        switch (borderType & ~BORDER_ISOLATED)
        {
        case BORDER_CONSTANT:
            btype = "BORDER_CONSTANT";
            break;
        case BORDER_REPLICATE:
            btype = "BORDER_REPLICATE";
            break;
        case BORDER_REFLECT:
            btype = "BORDER_REFLECT";
            break;
        case BORDER_WRAP:
            CV_Error(CV_StsUnsupportedFormat, "BORDER_WRAP is not supported!");
            return;
        case BORDER_REFLECT101:
            btype = "BORDER_REFLECT_101";
            break;
        }

        int requiredTop = anchor.y;
        int requiredLeft = BLOCK_SIZE; // not this: anchor.x;
        int requiredBottom = ksize.height - 1 - anchor.y;
        int requiredRight = BLOCK_SIZE; // not this: ksize.width - 1 - anchor.x;
        int h = isIsolatedBorder ? src.rows : src.wholerows;
        int w = isIsolatedBorder ? src.cols : src.wholecols;
        bool extra_extrapolation = h < requiredTop || h < requiredBottom || w < requiredLeft || w < requiredRight;

        CV_Assert(w >= ksize.width && h >= ksize.height); // TODO Other cases are not tested well

        char build_options[1024];
        sprintf(build_options, "-D LOCAL_SIZE=%d -D BLOCK_SIZE_Y=%d -D DATA_DEPTH=%d -D DATA_CHAN=%d -D USE_DOUBLE=%d -D ANCHOR_X=%d -D ANCHOR_Y=%d -D KERNEL_SIZE_X=%d -D KERNEL_SIZE_Y=%d -D %s -D %s -D %s",
                (int)BLOCK_SIZE, (int)BLOCK_SIZE_Y,
                src.depth(), src.oclchannels(), useDouble ? 1 : 0,
                anchor.x, anchor.y, ksize.width, ksize.height,
                btype,
                extra_extrapolation ? "EXTRA_EXTRAPOLATION" : "NO_EXTRA_EXTRAPOLATION",
                isIsolatedBorder ? "BORDER_ISOLATED" : "NO_BORDER_ISOLATED");

        size_t lt[3] = {BLOCK_SIZE, 1, 1};
        size_t gt[3] = {divUp(dst.cols, BLOCK_SIZE - (ksize.width - 1)) * BLOCK_SIZE, divUp(dst.rows, BLOCK_SIZE_Y), 1};

        cl_kernel kernel = openCLGetKernelFromSource(src.clCxt, &filtering_boxFilter, "boxFilter", -1, -1, build_options);

        size_t kernelWorkGroupSize;
        openCLSafeCall(clGetKernelWorkGroupInfo(kernel, getClDeviceID(src.clCxt),
                                                CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernelWorkGroupSize, 0));
        if (lt[0] > kernelWorkGroupSize)
        {
            clReleaseKernel(kernel);
            CV_Assert(BLOCK_SIZE > kernelWorkGroupSize);
            tryWorkItems = kernelWorkGroupSize;
            continue;
        }

        openCLExecuteKernel(src.clCxt, kernel, gt, lt, args); // kernel will be released here
    } while (false);
}

Ptr<BaseFilter_GPU> cv::ocl::getBoxFilter_GPU(int /*srcType*/, int /*dstType*/,
        const Size &ksize, Point anchor, int borderType)
{
    normalizeAnchor(anchor, ksize);

    return Ptr<BaseFilter_GPU>(new GPUBoxFilter(ksize, anchor,
                               borderType, GPUFilterBox));
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
    CV_Assert(bordertype <= BORDER_REFLECT_101);
    CV_Assert(ksize == (anchor << 1) + 1);
    int channels = src.oclchannels();

#ifdef ANDROID
    size_t localThreads[3] = { 16, 10, 1 };
#else
    size_t localThreads[3] = { 16, 16, 1 };
#endif
    size_t globalThreads[3] = { dst.cols, dst.rows, 1 };

    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", "BORDER_WRAP", "BORDER_REFLECT_101" };
    std::string buildOptions = format("-D RADIUSX=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d -D %s",
            anchor, (int)localThreads[0], (int)localThreads[1], channels, borderMap[bordertype]);

    if (src.depth() == CV_8U)
    {
        switch (channels)
        {
        case 1:
            globalThreads[0] = (dst.cols + 3) >> 2;
            break;
        case 2:
            globalThreads[0] = (dst.cols + 1) >> 1;
            break;
        case 4:
            globalThreads[0] = dst.cols;
            break;
        }
    }

    int src_pix_per_row = src.step / src.elemSize();
    int src_offset_x = (src.offset % src.step) / src.elemSize();
    int src_offset_y = src.offset / src.step;
    int dst_pix_per_row = dst.step / dst.elemSize();
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

    openCLExecuteKernel(src.clCxt, &filter_sep_row, "row_filter", globalThreads, localThreads,
                        args, channels, src.depth(), buildOptions.c_str());
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

    return makePtr<GpuLinearRowFilter>(ksize, anchor, mat_kernel,
        gpuFilter1D_callers[CV_MAT_DEPTH(srcType)], bordertype);
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

#ifdef ANDROID
    size_t localThreads[3] = {16, 10, 1};
#else
    size_t localThreads[3] = {16, 16, 1};
#endif
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
    int dst_offset_in_pixel;
    src_pix_per_row = src.step / src.elemSize();
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

    Mat temp = columnKernel.reshape(1, 1);
    oclMat mat_kernel(temp);

    int ksize = temp.cols;
    normalizeAnchor(anchor, ksize);

    return makePtr<GpuLinearColumnFilter>(ksize, anchor, mat_kernel,
        gpuFilter1D_callers[CV_MAT_DEPTH(dstType)], bordertype);
}

Ptr<FilterEngine_GPU> cv::ocl::createSeparableLinearFilter_GPU(int srcType, int dstType,
        const Mat &rowKernel, const Mat &columnKernel, const Point &anchor, double delta, int bordertype, Size imgSize )
{
    int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(dstType);
    int cn = CV_MAT_CN(srcType);
    int bdepth = std::max(std::max(sdepth, ddepth), CV_32F);
    int bufType = CV_MAKETYPE(bdepth, cn);
    Context* clCxt = Context::getContext();

    //if image size is non-degenerate and large enough
    //and if filter support is reasonable to satisfy larger local memory requirements,
    //then we can use single pass routine to avoid extra runtime calls overhead
    if( clCxt && clCxt->supportsFeature(FEATURE_CL_INTEL_DEVICE) &&
        rowKernel.rows <= 21 && columnKernel.rows <= 21 &&
        (rowKernel.rows & 1) == 1 && (columnKernel.rows & 1) == 1 &&
        imgSize.width > optimizedSepFilterLocalSize + (rowKernel.rows>>1) &&
        imgSize.height > optimizedSepFilterLocalSize + (columnKernel.rows>>1) )
    {
        return Ptr<FilterEngine_GPU>(new SingleStepSeparableFilterEngine_GPU(rowKernel, columnKernel, bordertype));
    }
    else
    {
        Ptr<BaseRowFilter_GPU> rowFilter = getLinearRowFilter_GPU(srcType, bufType, rowKernel, anchor.x, bordertype);
        Ptr<BaseColumnFilter_GPU> columnFilter = getLinearColumnFilter_GPU(bufType, dstType, columnKernel, anchor.y, bordertype, delta);

        return createSeparableFilter_GPU(rowFilter, columnFilter);
    }
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
        ddepth = src.depth();

    dst.create(src.size(), CV_MAKETYPE(ddepth, src.channels()));

    Ptr<FilterEngine_GPU> f = createSeparableLinearFilter_GPU(src.type(), dst.type(), kernelX, kernelY, anchor, delta, bordertype, src.size());
    f->apply(src, dst);
}

Ptr<FilterEngine_GPU> cv::ocl::createDerivFilter_GPU(int srcType, int dstType, int dx, int dy, int ksize, int borderType, Size imgSize )
{
    Mat kx, ky;
    getDerivKernels(kx, ky, dx, dy, ksize, false, CV_32F);
    return createSeparableLinearFilter_GPU(srcType, dstType,
                                           kx, ky, Point(-1, -1), 0, borderType, imgSize);
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
            kx *= scale;
        else
            ky *= scale;
    }

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
            kx *= scale;
        else
            ky *= scale;
    }

    sepFilter2D(src, dst, ddepth, kx, ky, Point(-1, -1), delta, bordertype);
}

void cv::ocl::Laplacian(const oclMat &src, oclMat &dst, int ddepth, int ksize, double scale,
        double delta, int borderType)
{
    CV_Assert(delta == 0);

    if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.type() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    CV_Assert(ksize == 1 || ksize == 3);

    double K[2][9] =
    {
        {0, 1, 0, 1, -4, 1, 0, 1, 0},
        {2, 0, 2, 0, -8, 0, 2, 0, 2}
    };
    Mat kernel(3, 3, CV_64F, (void *)K[ksize == 3 ? 1 : 0]);

    if (scale != 1)
        kernel *= scale;

    filter2D(src, dst, ddepth, kernel, Point(-1, -1), 0, borderType);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Gaussian Filter

Ptr<FilterEngine_GPU> cv::ocl::createGaussianFilter_GPU(int type, Size ksize, double sigma1, double sigma2, int bordertype, Size imgSize)
{
    int depth = CV_MAT_DEPTH(type);

    if (sigma2 <= 0)
        sigma2 = sigma1;

    // automatic detection of kernel size from sigma
    if (ksize.width <= 0 && sigma1 > 0)
        ksize.width = cvRound(sigma1 * (depth == CV_8U ? 3 : 4) * 2 + 1) | 1;

    if (ksize.height <= 0 && sigma2 > 0)
        ksize.height = cvRound(sigma2 * (depth == CV_8U ? 3 : 4) * 2 + 1) | 1;

    CV_Assert(ksize.width > 0 && ksize.width % 2 == 1 && ksize.height > 0 && ksize.height % 2 == 1);

    sigma1 = std::max(sigma1, 0.0);
    sigma2 = std::max(sigma2, 0.0);

    Mat kx = getGaussianKernel(ksize.width, sigma1, std::max(depth, CV_32F));
    Mat ky;

    if (ksize.height == ksize.width && std::abs(sigma1 - sigma2) < DBL_EPSILON)
        ky = kx;
    else
        ky = getGaussianKernel(ksize.height, sigma2, std::max(depth, CV_32F));

    return createSeparableLinearFilter_GPU(type, type, kx, ky, Point(-1, -1), 0.0, bordertype, imgSize);
}

void cv::ocl::GaussianBlur(const oclMat &src, oclMat &dst, Size ksize, double sigma1, double sigma2, int bordertype)
{
    if (bordertype != BORDER_CONSTANT)
    {
        if (src.rows == 1)
            ksize.height = 1;

        if (src.cols == 1)
            ksize.width = 1;
    }

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

    Ptr<FilterEngine_GPU> f = createGaussianFilter_GPU(src.type(), ksize, sigma1, sigma2, bordertype, src.size());
    f->apply(src, dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Adaptive Bilateral Filter

void cv::ocl::adaptiveBilateralFilter(const oclMat& src, oclMat& dst, Size ksize, double sigmaSpace, double maxSigmaColor, Point anchor, int borderType)
{
    CV_Assert((ksize.width & 1) && (ksize.height & 1));  // ksize must be odd
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);  // source must be 8bit RGB image
    if( sigmaSpace <= 0 )
        sigmaSpace = 1;
    Mat lut(Size(ksize.width, ksize.height), CV_32FC1);
    double sigma2 = sigmaSpace * sigmaSpace;
    int idx = 0;
    int w = ksize.width / 2;
    int h = ksize.height / 2;

    int ABF_GAUSSIAN_ocl = 1;

    if(ABF_GAUSSIAN_ocl)
    {
        for(int y=-h; y<=h; y++)
            for(int x=-w; x<=w; x++)
        {
            lut.at<float>(idx++) = expf( (float)(-0.5 * (x * x + y * y)/sigma2));
        }
    }
    else
    {
        for(int y=-h; y<=h; y++)
            for(int x=-w; x<=w; x++)
        {
            lut.at<float>(idx++) = (float) (sigma2 / (sigma2 + x * x + y * y));
        }
    }

    oclMat dlut(lut);
    int depth = src.depth();
    int cn = src.oclchannels();

    normalizeAnchor(anchor, ksize);
    const static String kernelName = "adaptiveBilateralFilter";

    dst.create(src.size(), src.type());

    char btype[30];
    switch(borderType)
    {
    case BORDER_CONSTANT:
        sprintf(btype, "BORDER_CONSTANT");
        break;
    case BORDER_REPLICATE:
        sprintf(btype, "BORDER_REPLICATE");
        break;
    case BORDER_REFLECT:
        sprintf(btype, "BORDER_REFLECT");
        break;
    case BORDER_WRAP:
        sprintf(btype, "BORDER_WRAP");
        break;
    case BORDER_REFLECT101:
        sprintf(btype, "BORDER_REFLECT_101");
        break;
    default:
        CV_Error(Error::StsBadArg, "This border type is not supported");
        break;
    }

    //the following constants may be adjusted for performance concerns
    const static size_t blockSizeX = 64, blockSizeY = 1, EXTRA = ksize.height - 1;

    //Normalize the result by default
    const float alpha = ksize.height * ksize.width;

    const size_t gSize = blockSizeX - ksize.width / 2 * 2;
    const size_t globalSizeX = (src.cols) % gSize == 0 ?
        src.cols / gSize * blockSizeX :
        (src.cols / gSize + 1) * blockSizeX;
    const size_t rows_per_thread = 1 + EXTRA;
    const size_t globalSizeY = ((src.rows + rows_per_thread - 1) / rows_per_thread) % blockSizeY == 0 ?
        ((src.rows + rows_per_thread - 1) / rows_per_thread) :
        (((src.rows + rows_per_thread - 1) / rows_per_thread) / blockSizeY + 1) * blockSizeY;

    size_t globalThreads[3] = { globalSizeX, globalSizeY, 1};
    size_t localThreads[3]  = { blockSizeX, blockSizeY, 1};

    char build_options[250];

    //LDATATYPESIZE is sizeof local data store. This is to exemplify effect of LDS on kernel performance
    sprintf(build_options,
        "-D VAR_PER_CHANNEL=1 -D CALCVAR=1 -D FIXED_WEIGHT=0 -D EXTRA=%d -D MAX_VAR_VAL=%f -D ABF_GAUSSIAN=%d"
        " -D THREADS=%d -D anX=%d -D anY=%d -D ksX=%d -D ksY=%d -D %s",
        static_cast<int>(EXTRA), static_cast<float>(maxSigmaColor*maxSigmaColor), static_cast<int>(ABF_GAUSSIAN_ocl),
        static_cast<int>(blockSizeX), anchor.x, anchor.y, ksize.width, ksize.height, btype);

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
    args.push_back(std::make_pair(sizeof(cl_mem), &dlut.data));
    int lut_step = dlut.step1();
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&lut_step));

    openCLExecuteKernel(Context::getContext(), &filtering_adaptive_bilateral, kernelName,
        globalThreads, localThreads, args, cn, depth, build_options);
}
