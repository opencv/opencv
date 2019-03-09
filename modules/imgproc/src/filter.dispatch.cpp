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
#include "opencv2/core/opencl/ocl_defs.hpp"
#include "opencl_kernels_imgproc.hpp"
#include "hal_replacement.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "filter.hpp"

#include "filter.simd.hpp"
#include "filter.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content


/****************************************************************************************\
                                    Base Image Filter
\****************************************************************************************/

namespace cv {

BaseRowFilter::BaseRowFilter() { ksize = anchor = -1; }
BaseRowFilter::~BaseRowFilter() {}

BaseColumnFilter::BaseColumnFilter() { ksize = anchor = -1; }
BaseColumnFilter::~BaseColumnFilter() {}
void BaseColumnFilter::reset() {}

BaseFilter::BaseFilter() { ksize = Size(-1,-1); anchor = Point(-1,-1); }
BaseFilter::~BaseFilter() {}
void BaseFilter::reset() {}

FilterEngine::FilterEngine()
    : srcType(-1), dstType(-1), bufType(-1), maxWidth(0), wholeSize(-1, -1), dx1(0), dx2(0),
      rowBorderType(BORDER_REPLICATE), columnBorderType(BORDER_REPLICATE),
      borderElemSize(0), bufStep(0), startY(0), startY0(0), endY(0), rowCount(0), dstY(0)
{
}


FilterEngine::FilterEngine( const Ptr<BaseFilter>& _filter2D,
                            const Ptr<BaseRowFilter>& _rowFilter,
                            const Ptr<BaseColumnFilter>& _columnFilter,
                            int _srcType, int _dstType, int _bufType,
                            int _rowBorderType, int _columnBorderType,
                            const Scalar& _borderValue )
    : srcType(-1), dstType(-1), bufType(-1), maxWidth(0), wholeSize(-1, -1), dx1(0), dx2(0),
      rowBorderType(BORDER_REPLICATE), columnBorderType(BORDER_REPLICATE),
      borderElemSize(0), bufStep(0), startY(0), startY0(0), endY(0), rowCount(0), dstY(0)
{
    init(_filter2D, _rowFilter, _columnFilter, _srcType, _dstType, _bufType,
         _rowBorderType, _columnBorderType, _borderValue);
}

FilterEngine::~FilterEngine()
{
}


void FilterEngine::init( const Ptr<BaseFilter>& _filter2D,
                         const Ptr<BaseRowFilter>& _rowFilter,
                         const Ptr<BaseColumnFilter>& _columnFilter,
                         int _srcType, int _dstType, int _bufType,
                         int _rowBorderType, int _columnBorderType,
                         const Scalar& _borderValue )
{
    _srcType = CV_MAT_TYPE(_srcType);
    _bufType = CV_MAT_TYPE(_bufType);
    _dstType = CV_MAT_TYPE(_dstType);

    srcType = _srcType;
    int srcElemSize = (int)getElemSize(srcType);
    dstType = _dstType;
    bufType = _bufType;

    filter2D = _filter2D;
    rowFilter = _rowFilter;
    columnFilter = _columnFilter;

    if( _columnBorderType < 0 )
        _columnBorderType = _rowBorderType;

    rowBorderType = _rowBorderType;
    columnBorderType = _columnBorderType;

    CV_Assert( columnBorderType != BORDER_WRAP );

    if( isSeparable() )
    {
        CV_Assert( rowFilter && columnFilter );
        ksize = Size(rowFilter->ksize, columnFilter->ksize);
        anchor = Point(rowFilter->anchor, columnFilter->anchor);
    }
    else
    {
        CV_Assert( bufType == srcType );
        ksize = filter2D->ksize;
        anchor = filter2D->anchor;
    }

    CV_Assert( 0 <= anchor.x && anchor.x < ksize.width &&
               0 <= anchor.y && anchor.y < ksize.height );

    borderElemSize = srcElemSize/(CV_MAT_DEPTH(srcType) >= CV_32S ? sizeof(int) : 1);
    int borderLength = std::max(ksize.width - 1, 1);
    borderTab.resize(borderLength*borderElemSize);

    maxWidth = bufStep = 0;
    constBorderRow.clear();

    if( rowBorderType == BORDER_CONSTANT || columnBorderType == BORDER_CONSTANT )
    {
        constBorderValue.resize(srcElemSize*borderLength);
        int srcType1 = CV_MAKETYPE(CV_MAT_DEPTH(srcType), MIN(CV_MAT_CN(srcType), 4));
        scalarToRawData(_borderValue, &constBorderValue[0], srcType1,
                        borderLength*CV_MAT_CN(srcType));
    }

    wholeSize = Size(-1,-1);
}

#define VEC_ALIGN CV_MALLOC_ALIGN

int FilterEngine::start(const Size& _wholeSize, const Size& sz, const Point& ofs)
{
    CV_INSTRUMENT_REGION();

    CV_CPU_DISPATCH(FilterEngine__start, (*this, _wholeSize, sz, ofs),
        CV_CPU_DISPATCH_MODES_ALL);
}


int FilterEngine::start(const Mat& src, const Size &wsz, const Point &ofs)
{
    start( wsz, src.size(), ofs);
    return startY - ofs.y;
}

int FilterEngine::remainingInputRows() const
{
    return endY - startY - rowCount;
}

int FilterEngine::remainingOutputRows() const
{
    return roi.height - dstY;
}

int FilterEngine::proceed(const uchar* src, int srcstep, int count,
                          uchar* dst, int dststep)
{
    CV_INSTRUMENT_REGION();

    CV_Assert( wholeSize.width > 0 && wholeSize.height > 0 );

    CV_CPU_DISPATCH(FilterEngine__proceed, (*this, src, srcstep, count, dst, dststep),
        CV_CPU_DISPATCH_MODES_ALL);
}

void FilterEngine::apply(const Mat& src, Mat& dst, const Size& wsz, const Point& ofs)
{
    CV_INSTRUMENT_REGION();

    CV_CheckTypeEQ(src.type(), srcType, "");
    CV_CheckTypeEQ(dst.type(), dstType, "");

    CV_CPU_DISPATCH(FilterEngine__apply, (*this, src, dst, wsz, ofs),
        CV_CPU_DISPATCH_MODES_ALL);
}

/****************************************************************************************\
*                                 Separable linear filter                                *
\****************************************************************************************/

int getKernelType(InputArray filter_kernel, Point anchor)
{
    Mat _kernel = filter_kernel.getMat();
    CV_Assert( _kernel.channels() == 1 );
    int i, sz = _kernel.rows*_kernel.cols;

    Mat kernel;
    _kernel.convertTo(kernel, CV_64F);

    const double* coeffs = kernel.ptr<double>();
    double sum = 0;
    int type = KERNEL_SMOOTH + KERNEL_INTEGER;
    if( (_kernel.rows == 1 || _kernel.cols == 1) &&
        anchor.x*2 + 1 == _kernel.cols &&
        anchor.y*2 + 1 == _kernel.rows )
        type |= (KERNEL_SYMMETRICAL + KERNEL_ASYMMETRICAL);

    for( i = 0; i < sz; i++ )
    {
        double a = coeffs[i], b = coeffs[sz - i - 1];
        if( a != b )
            type &= ~KERNEL_SYMMETRICAL;
        if( a != -b )
            type &= ~KERNEL_ASYMMETRICAL;
        if( a < 0 )
            type &= ~KERNEL_SMOOTH;
        if( a != saturate_cast<int>(a) )
            type &= ~KERNEL_INTEGER;
        sum += a;
    }

    if( fabs(sum - 1) > FLT_EPSILON*(fabs(sum) + 1) )
        type &= ~KERNEL_SMOOTH;
    return type;
}


Ptr<BaseRowFilter> getLinearRowFilter(
        int srcType, int bufType,
        InputArray _kernel, int anchor,
        int symmetryType)
{
    CV_INSTRUMENT_REGION();

    Mat kernelMat = _kernel.getMat();
    CV_CPU_DISPATCH(getLinearRowFilter, (srcType, bufType, kernelMat, anchor, symmetryType),
        CV_CPU_DISPATCH_MODES_ALL);
}


Ptr<BaseColumnFilter> getLinearColumnFilter(
        int bufType, int dstType,
        InputArray kernel, int anchor,
        int symmetryType, double delta,
        int bits)
{
    CV_INSTRUMENT_REGION();

    Mat kernelMat = kernel.getMat();
    CV_CPU_DISPATCH(getLinearColumnFilter, (bufType, dstType, kernelMat, anchor, symmetryType, delta, bits),
        CV_CPU_DISPATCH_MODES_ALL);
}


Ptr<FilterEngine> createSeparableLinearFilter(
        int _srcType, int _dstType,
        InputArray __rowKernel, InputArray __columnKernel,
        Point _anchor, double _delta,
        int _rowBorderType, int _columnBorderType,
        const Scalar& _borderValue)
{
    Mat _rowKernel = __rowKernel.getMat(), _columnKernel = __columnKernel.getMat();
    _srcType = CV_MAT_TYPE(_srcType);
    _dstType = CV_MAT_TYPE(_dstType);
    int sdepth = CV_MAT_DEPTH(_srcType), ddepth = CV_MAT_DEPTH(_dstType);
    int cn = CV_MAT_CN(_srcType);
    CV_Assert( cn == CV_MAT_CN(_dstType) );
    int rsize = _rowKernel.rows + _rowKernel.cols - 1;
    int csize = _columnKernel.rows + _columnKernel.cols - 1;
    if( _anchor.x < 0 )
        _anchor.x = rsize/2;
    if( _anchor.y < 0 )
        _anchor.y = csize/2;
    int rtype = getKernelType(_rowKernel,
        _rowKernel.rows == 1 ? Point(_anchor.x, 0) : Point(0, _anchor.x));
    int ctype = getKernelType(_columnKernel,
        _columnKernel.rows == 1 ? Point(_anchor.y, 0) : Point(0, _anchor.y));
    Mat rowKernel, columnKernel;

    int bdepth = std::max(CV_32F,std::max(sdepth, ddepth));
    int bits = 0;

    if( sdepth == CV_8U &&
        ((rtype == KERNEL_SMOOTH+KERNEL_SYMMETRICAL &&
          ctype == KERNEL_SMOOTH+KERNEL_SYMMETRICAL &&
          ddepth == CV_8U) ||
         ((rtype & (KERNEL_SYMMETRICAL+KERNEL_ASYMMETRICAL)) &&
          (ctype & (KERNEL_SYMMETRICAL+KERNEL_ASYMMETRICAL)) &&
          (rtype & ctype & KERNEL_INTEGER) &&
          ddepth == CV_16S)) )
    {
        bdepth = CV_32S;
        bits = ddepth == CV_8U ? 8 : 0;
        _rowKernel.convertTo( rowKernel, CV_32S, 1 << bits );
        _columnKernel.convertTo( columnKernel, CV_32S, 1 << bits );
        bits *= 2;
        _delta *= (1 << bits);
    }
    else
    {
        if( _rowKernel.type() != bdepth )
            _rowKernel.convertTo( rowKernel, bdepth );
        else
            rowKernel = _rowKernel;
        if( _columnKernel.type() != bdepth )
            _columnKernel.convertTo( columnKernel, bdepth );
        else
            columnKernel = _columnKernel;
    }

    int _bufType = CV_MAKETYPE(bdepth, cn);
    Ptr<BaseRowFilter> _rowFilter = getLinearRowFilter(
        _srcType, _bufType, rowKernel, _anchor.x, rtype);
    Ptr<BaseColumnFilter> _columnFilter = getLinearColumnFilter(
        _bufType, _dstType, columnKernel, _anchor.y, ctype, _delta, bits );

    return Ptr<FilterEngine>( new FilterEngine(Ptr<BaseFilter>(), _rowFilter, _columnFilter,
        _srcType, _dstType, _bufType, _rowBorderType, _columnBorderType, _borderValue ));
}


/****************************************************************************************\
*                               Non-separable linear filter                              *
\****************************************************************************************/

void preprocess2DKernel( const Mat& kernel, std::vector<Point>& coords, std::vector<uchar>& coeffs )
{
    int i, j, k, nz = countNonZero(kernel), ktype = kernel.type();
    if(nz == 0)
        nz = 1;
    CV_Assert( ktype == CV_8U || ktype == CV_32S || ktype == CV_32F || ktype == CV_64F );
    coords.resize(nz);
    coeffs.resize(nz*getElemSize(ktype));
    uchar* _coeffs = &coeffs[0];

    for( i = k = 0; i < kernel.rows; i++ )
    {
        const uchar* krow = kernel.ptr(i);
        for( j = 0; j < kernel.cols; j++ )
        {
            if( ktype == CV_8U )
            {
                uchar val = krow[j];
                if( val == 0 )
                    continue;
                coords[k] = Point(j,i);
                _coeffs[k++] = val;
            }
            else if( ktype == CV_32S )
            {
                int val = ((const int*)krow)[j];
                if( val == 0 )
                    continue;
                coords[k] = Point(j,i);
                ((int*)_coeffs)[k++] = val;
            }
            else if( ktype == CV_32F )
            {
                float val = ((const float*)krow)[j];
                if( val == 0 )
                    continue;
                coords[k] = Point(j,i);
                ((float*)_coeffs)[k++] = val;
            }
            else
            {
                double val = ((const double*)krow)[j];
                if( val == 0 )
                    continue;
                coords[k] = Point(j,i);
                ((double*)_coeffs)[k++] = val;
            }
        }
    }
}


template<typename ST, class CastOp, class VecOp> struct Filter2D : public BaseFilter
{
    typedef typename CastOp::type1 KT;
    typedef typename CastOp::rtype DT;

    Filter2D( const Mat& _kernel, Point _anchor,
        double _delta, const CastOp& _castOp=CastOp(),
        const VecOp& _vecOp=VecOp() )
    {
        anchor = _anchor;
        ksize = _kernel.size();
        delta = saturate_cast<KT>(_delta);
        castOp0 = _castOp;
        vecOp = _vecOp;
        CV_Assert( _kernel.type() == DataType<KT>::type );
        preprocess2DKernel( _kernel, coords, coeffs );
        ptrs.resize( coords.size() );
    }

    void operator()(const uchar** src, uchar* dst, int dststep, int count, int width, int cn) CV_OVERRIDE
    {
        KT _delta = delta;
        const Point* pt = &coords[0];
        const KT* kf = (const KT*)&coeffs[0];
        const ST** kp = (const ST**)&ptrs[0];
        int i, k, nz = (int)coords.size();
        CastOp castOp = castOp0;

        width *= cn;
        for( ; count > 0; count--, dst += dststep, src++ )
        {
            DT* D = (DT*)dst;

            for( k = 0; k < nz; k++ )
                kp[k] = (const ST*)src[pt[k].y] + pt[k].x*cn;

            i = vecOp((const uchar**)kp, dst, width);
            #if CV_ENABLE_UNROLLED
            for( ; i <= width - 4; i += 4 )
            {
                KT s0 = _delta, s1 = _delta, s2 = _delta, s3 = _delta;

                for( k = 0; k < nz; k++ )
                {
                    const ST* sptr = kp[k] + i;
                    KT f = kf[k];
                    s0 += f*sptr[0];
                    s1 += f*sptr[1];
                    s2 += f*sptr[2];
                    s3 += f*sptr[3];
                }

                D[i] = castOp(s0); D[i+1] = castOp(s1);
                D[i+2] = castOp(s2); D[i+3] = castOp(s3);
            }
            #endif
            for( ; i < width; i++ )
            {
                KT s0 = _delta;
                for( k = 0; k < nz; k++ )
                    s0 += kf[k]*kp[k][i];
                D[i] = castOp(s0);
            }
        }
    }

    std::vector<Point> coords;
    std::vector<uchar> coeffs;
    std::vector<uchar*> ptrs;
    KT delta;
    CastOp castOp0;
    VecOp vecOp;
};

#ifdef HAVE_OPENCL

#define DIVUP(total, grain) (((total) + (grain) - 1) / (grain))
#define ROUNDUP(sz, n)      ((sz) + (n) - 1 - (((sz) + (n) - 1) % (n)))

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
static int _prepareKernelFilter2D(std::vector<T> & data, const Mat & kernel)
{
    Mat _kernel; kernel.convertTo(_kernel, DataDepth<T>::value);
    int size_y_aligned = ROUNDUP(kernel.rows * 2, 4);
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

static bool ocl_filter2D( InputArray _src, OutputArray _dst, int ddepth,
                   InputArray _kernel, Point anchor,
                   double delta, int borderType )
{
    int type = _src.type(), sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    ddepth = ddepth < 0 ? sdepth : ddepth;
    int dtype = CV_MAKE_TYPE(ddepth, cn), wdepth = std::max(std::max(sdepth, ddepth), CV_32F),
            wtype = CV_MAKE_TYPE(wdepth, cn);
    if (cn > 4)
        return false;

    Size ksize = _kernel.size();
    if (anchor.x < 0)
        anchor.x = ksize.width / 2;
    if (anchor.y < 0)
        anchor.y = ksize.height / 2;

    bool isolated = (borderType & BORDER_ISOLATED) != 0;
    borderType &= ~BORDER_ISOLATED;
    const cv::ocl::Device &device = cv::ocl::Device::getDefault();
    bool doubleSupport = device.doubleFPConfig() > 0;
    if (wdepth == CV_64F && !doubleSupport)
        return false;

    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT",
                                       "BORDER_WRAP", "BORDER_REFLECT_101" };

    cv::Mat kernelMat = _kernel.getMat();
    cv::Size sz = _src.size(), wholeSize;
    size_t globalsize[2] = { (size_t)sz.width, (size_t)sz.height };
    size_t localsize_general[2] = {0, 1};
    size_t* localsize = NULL;

    ocl::Kernel k;
    UMat src = _src.getUMat();
    if (!isolated)
    {
        Point ofs;
        src.locateROI(wholeSize, ofs);
    }

    size_t tryWorkItems = device.maxWorkGroupSize();
    if (device.isIntel() && 128 < tryWorkItems)
        tryWorkItems = 128;
    char cvt[2][40];

    // For smaller filter kernels, there is a special kernel that is more
    // efficient than the general one.
    UMat kernalDataUMat;
    if (device.isIntel() && (device.type() & ocl::Device::TYPE_GPU) &&
        ((ksize.width < 5 && ksize.height < 5) ||
        (ksize.width == 5 && ksize.height == 5 && cn == 1)))
    {
        kernelMat = kernelMat.reshape(0, 1);
        String kerStr = ocl::kernelToStr(kernelMat, CV_32F);
        int h = isolated ? sz.height : wholeSize.height;
        int w = isolated ? sz.width : wholeSize.width;

        if (w < ksize.width || h < ksize.height)
            return false;

        // Figure out what vector size to use for loading the pixels.
        int pxLoadNumPixels = cn != 1 || sz.width % 4 ? 1 : 4;
        int pxLoadVecSize = cn * pxLoadNumPixels;

        // Figure out how many pixels per work item to compute in X and Y
        // directions.  Too many and we run out of registers.
        int pxPerWorkItemX = 1;
        int pxPerWorkItemY = 1;
        if (cn <= 2 && ksize.width <= 4 && ksize.height <= 4)
        {
            pxPerWorkItemX = sz.width % 8 ? sz.width % 4 ? sz.width % 2 ? 1 : 2 : 4 : 8;
            pxPerWorkItemY = sz.height % 2 ? 1 : 2;
        }
        else if (cn < 4 || (ksize.width <= 4 && ksize.height <= 4))
        {
            pxPerWorkItemX = sz.width % 2 ? 1 : 2;
            pxPerWorkItemY = sz.height % 2 ? 1 : 2;
        }
        globalsize[0] = sz.width / pxPerWorkItemX;
        globalsize[1] = sz.height / pxPerWorkItemY;

        // Need some padding in the private array for pixels
        int privDataWidth = ROUNDUP(pxPerWorkItemX + ksize.width - 1, pxLoadNumPixels);

        // Make the global size a nice round number so the runtime can pick
        // from reasonable choices for the workgroup size
        const int wgRound = 256;
        globalsize[0] = ROUNDUP(globalsize[0], wgRound);

        char build_options[1024];
        sprintf(build_options, "-D cn=%d "
                "-D ANCHOR_X=%d -D ANCHOR_Y=%d -D KERNEL_SIZE_X=%d -D KERNEL_SIZE_Y=%d "
                "-D PX_LOAD_VEC_SIZE=%d -D PX_LOAD_NUM_PX=%d "
                "-D PX_PER_WI_X=%d -D PX_PER_WI_Y=%d -D PRIV_DATA_WIDTH=%d -D %s -D %s "
                "-D PX_LOAD_X_ITERATIONS=%d -D PX_LOAD_Y_ITERATIONS=%d "
                "-D srcT=%s -D srcT1=%s -D dstT=%s -D dstT1=%s -D WT=%s -D WT1=%s "
                "-D convertToWT=%s -D convertToDstT=%s %s",
                cn, anchor.x, anchor.y, ksize.width, ksize.height,
                pxLoadVecSize, pxLoadNumPixels,
                pxPerWorkItemX, pxPerWorkItemY, privDataWidth, borderMap[borderType],
                isolated ? "BORDER_ISOLATED" : "NO_BORDER_ISOLATED",
                privDataWidth / pxLoadNumPixels, pxPerWorkItemY + ksize.height - 1,
                ocl::typeToStr(type), ocl::typeToStr(sdepth), ocl::typeToStr(dtype),
                ocl::typeToStr(ddepth), ocl::typeToStr(wtype), ocl::typeToStr(wdepth),
                ocl::convertTypeStr(sdepth, wdepth, cn, cvt[0]),
                ocl::convertTypeStr(wdepth, ddepth, cn, cvt[1]), kerStr.c_str());

        if (!k.create("filter2DSmall", cv::ocl::imgproc::filter2DSmall_oclsrc, build_options))
            return false;
    }
    else
    {
        localsize = localsize_general;
        std::vector<float> kernelMatDataFloat;
        int kernel_size_y2_aligned = _prepareKernelFilter2D<float>(kernelMatDataFloat, kernelMat);
        String kerStr = ocl::kernelToStr(kernelMatDataFloat, CV_32F);

        for ( ; ; )
        {
            size_t BLOCK_SIZE = tryWorkItems;
            while (BLOCK_SIZE > 32 && BLOCK_SIZE >= (size_t)ksize.width * 2 && BLOCK_SIZE > (size_t)sz.width * 2)
                BLOCK_SIZE /= 2;

            if ((size_t)ksize.width > BLOCK_SIZE)
                return false;

            int requiredTop = anchor.y;
            int requiredLeft = (int)BLOCK_SIZE; // not this: anchor.x;
            int requiredBottom = ksize.height - 1 - anchor.y;
            int requiredRight = (int)BLOCK_SIZE; // not this: ksize.width - 1 - anchor.x;
            int h = isolated ? sz.height : wholeSize.height;
            int w = isolated ? sz.width : wholeSize.width;
            bool extra_extrapolation = h < requiredTop || h < requiredBottom || w < requiredLeft || w < requiredRight;

            if ((w < ksize.width) || (h < ksize.height))
                return false;

            String opts = format("-D LOCAL_SIZE=%d -D cn=%d "
                                 "-D ANCHOR_X=%d -D ANCHOR_Y=%d -D KERNEL_SIZE_X=%d -D KERNEL_SIZE_Y=%d "
                                 "-D KERNEL_SIZE_Y2_ALIGNED=%d -D %s -D %s -D %s%s%s "
                                 "-D srcT=%s -D srcT1=%s -D dstT=%s -D dstT1=%s -D WT=%s -D WT1=%s "
                                 "-D convertToWT=%s -D convertToDstT=%s",
                                 (int)BLOCK_SIZE, cn, anchor.x, anchor.y,
                                 ksize.width, ksize.height, kernel_size_y2_aligned, borderMap[borderType],
                                 extra_extrapolation ? "EXTRA_EXTRAPOLATION" : "NO_EXTRA_EXTRAPOLATION",
                                 isolated ? "BORDER_ISOLATED" : "NO_BORDER_ISOLATED",
                                 doubleSupport ? " -D DOUBLE_SUPPORT" : "", kerStr.c_str(),
                                 ocl::typeToStr(type), ocl::typeToStr(sdepth), ocl::typeToStr(dtype),
                                 ocl::typeToStr(ddepth), ocl::typeToStr(wtype), ocl::typeToStr(wdepth),
                                 ocl::convertTypeStr(sdepth, wdepth, cn, cvt[0]),
                                 ocl::convertTypeStr(wdepth, ddepth, cn, cvt[1]));

            localsize[0] = BLOCK_SIZE;
            globalsize[0] = DIVUP(sz.width, BLOCK_SIZE - (ksize.width - 1)) * BLOCK_SIZE;
            globalsize[1] = sz.height;

            if (!k.create("filter2D", cv::ocl::imgproc::filter2D_oclsrc, opts))
                return false;

            size_t kernelWorkGroupSize = k.workGroupSize();
            if (localsize[0] <= kernelWorkGroupSize)
                break;
            if (BLOCK_SIZE < kernelWorkGroupSize)
                return false;
            tryWorkItems = kernelWorkGroupSize;
        }
    }

    _dst.create(sz, dtype);
    UMat dst = _dst.getUMat();

    int srcOffsetX = (int)((src.offset % src.step) / src.elemSize());
    int srcOffsetY = (int)(src.offset / src.step);
    int srcEndX = (isolated ? (srcOffsetX + sz.width) : wholeSize.width);
    int srcEndY = (isolated ? (srcOffsetY + sz.height) : wholeSize.height);

    k.args(ocl::KernelArg::PtrReadOnly(src), (int)src.step, srcOffsetX, srcOffsetY,
           srcEndX, srcEndY, ocl::KernelArg::WriteOnly(dst), (float)delta);

    return k.run(2, globalsize, localsize, false);
}

const int shift_bits = 8;

static bool ocl_sepRowFilter2D(const UMat & src, UMat & buf, const Mat & kernelX, int anchor,
                               int borderType, int ddepth, bool fast8uc1, bool int_arithm)
{
    int type = src.type(), cn = CV_MAT_CN(type), sdepth = CV_MAT_DEPTH(type);
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;
    Size bufSize = buf.size();
    int buf_type = buf.type(), bdepth = CV_MAT_DEPTH(buf_type);

    if (!doubleSupport && (sdepth == CV_64F || ddepth == CV_64F))
        return false;

#ifdef __ANDROID__
    size_t localsize[2] = {16, 10};
#else
    size_t localsize[2] = {16, 16};
#endif

    size_t globalsize[2] = {DIVUP(bufSize.width, localsize[0]) * localsize[0], DIVUP(bufSize.height, localsize[1]) * localsize[1]};
    if (fast8uc1)
        globalsize[0] = DIVUP((bufSize.width + 3) >> 2, localsize[0]) * localsize[0];

    int radiusX = anchor, radiusY = (buf.rows - src.rows) >> 1;

    bool isolated = (borderType & BORDER_ISOLATED) != 0;
    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", "BORDER_WRAP", "BORDER_REFLECT_101" },
        * const btype = borderMap[borderType & ~BORDER_ISOLATED];

    bool extra_extrapolation = src.rows < (int)((-radiusY + globalsize[1]) >> 1) + 1;
    extra_extrapolation |= src.rows < radiusY;
    extra_extrapolation |= src.cols < (int)((-radiusX + globalsize[0] + 8 * localsize[0] + 3) >> 1) + 1;
    extra_extrapolation |= src.cols < radiusX;

    char cvt[40];
    cv::String build_options = cv::format("-D RADIUSX=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d -D %s -D %s -D %s"
                                          " -D srcT=%s -D dstT=%s -D convertToDstT=%s -D srcT1=%s -D dstT1=%s%s%s",
                                          radiusX, (int)localsize[0], (int)localsize[1], cn, btype,
                                          extra_extrapolation ? "EXTRA_EXTRAPOLATION" : "NO_EXTRA_EXTRAPOLATION",
                                          isolated ? "BORDER_ISOLATED" : "NO_BORDER_ISOLATED",
                                          ocl::typeToStr(type), ocl::typeToStr(buf_type),
                                          ocl::convertTypeStr(sdepth, bdepth, cn, cvt),
                                          ocl::typeToStr(sdepth), ocl::typeToStr(bdepth),
                                          doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                                          int_arithm ? " -D INTEGER_ARITHMETIC" : "");
    build_options += ocl::kernelToStr(kernelX, bdepth);

    Size srcWholeSize; Point srcOffset;
    src.locateROI(srcWholeSize, srcOffset);

    String kernelName("row_filter");
    if (fast8uc1)
        kernelName += "_C1_D0";

    ocl::Kernel k(kernelName.c_str(), cv::ocl::imgproc::filterSepRow_oclsrc,
                  build_options);
    if (k.empty())
        return false;

    if (fast8uc1)
        k.args(ocl::KernelArg::PtrReadOnly(src), (int)(src.step / src.elemSize()), srcOffset.x,
               srcOffset.y, src.cols, src.rows, srcWholeSize.width, srcWholeSize.height,
               ocl::KernelArg::PtrWriteOnly(buf), (int)(buf.step / buf.elemSize()),
               buf.cols, buf.rows, radiusY);
    else
        k.args(ocl::KernelArg::PtrReadOnly(src), (int)src.step, srcOffset.x,
               srcOffset.y, src.cols, src.rows, srcWholeSize.width, srcWholeSize.height,
               ocl::KernelArg::PtrWriteOnly(buf), (int)buf.step, buf.cols, buf.rows, radiusY);

    return k.run(2, globalsize, localsize, false);
}

static bool ocl_sepColFilter2D(const UMat & buf, UMat & dst, const Mat & kernelY, double delta, int anchor, bool int_arithm)
{
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;
    if (dst.depth() == CV_64F && !doubleSupport)
        return false;

#ifdef __ANDROID__
    size_t localsize[2] = { 16, 10 };
#else
    size_t localsize[2] = { 16, 16 };
#endif
    size_t globalsize[2] = { 0, 0 };

    int dtype = dst.type(), cn = CV_MAT_CN(dtype), ddepth = CV_MAT_DEPTH(dtype);
    Size sz = dst.size();
    int buf_type = buf.type(), bdepth = CV_MAT_DEPTH(buf_type);

    globalsize[1] = DIVUP(sz.height, localsize[1]) * localsize[1];
    globalsize[0] = DIVUP(sz.width, localsize[0]) * localsize[0];

    char cvt[40];
    cv::String build_options = cv::format("-D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d"
                                          " -D srcT=%s -D dstT=%s -D convertToDstT=%s"
                                          " -D srcT1=%s -D dstT1=%s -D SHIFT_BITS=%d%s%s",
                                          anchor, (int)localsize[0], (int)localsize[1], cn,
                                          ocl::typeToStr(buf_type), ocl::typeToStr(dtype),
                                          ocl::convertTypeStr(bdepth, ddepth, cn, cvt),
                                          ocl::typeToStr(bdepth), ocl::typeToStr(ddepth),
                                          2*shift_bits, doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                                          int_arithm ? " -D INTEGER_ARITHMETIC" : "");
    build_options += ocl::kernelToStr(kernelY, bdepth);

    ocl::Kernel k("col_filter", cv::ocl::imgproc::filterSepCol_oclsrc,
                  build_options);
    if (k.empty())
        return false;

    k.args(ocl::KernelArg::ReadOnly(buf), ocl::KernelArg::WriteOnly(dst),
           static_cast<float>(delta));

    return k.run(2, globalsize, localsize, false);
}

const int optimizedSepFilterLocalWidth  = 16;
const int optimizedSepFilterLocalHeight = 8;

static bool ocl_sepFilter2D_SinglePass(InputArray _src, OutputArray _dst,
                                       Mat row_kernel, Mat col_kernel,
                                       double delta, int borderType, int ddepth, int bdepth, bool int_arithm)
{
    Size size = _src.size(), wholeSize;
    Point origin;
    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype),
            esz = CV_ELEM_SIZE(stype), wdepth = std::max(std::max(sdepth, ddepth), bdepth),
            dtype = CV_MAKE_TYPE(ddepth, cn);
    size_t src_step = _src.step(), src_offset = _src.offset();
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    if (esz == 0 || src_step == 0
        || (src_offset % src_step) % esz != 0
        || (!doubleSupport && (sdepth == CV_64F || ddepth == CV_64F))
        || !(borderType == BORDER_CONSTANT
             || borderType == BORDER_REPLICATE
             || borderType == BORDER_REFLECT
             || borderType == BORDER_WRAP
             || borderType == BORDER_REFLECT_101))
        return false;

    size_t lt2[2] = { optimizedSepFilterLocalWidth, optimizedSepFilterLocalHeight };
    size_t gt2[2] = { lt2[0] * (1 + (size.width - 1) / lt2[0]), lt2[1]};

    char cvt[2][40];
    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", "BORDER_WRAP",
                                       "BORDER_REFLECT_101" };

    String opts = cv::format("-D BLK_X=%d -D BLK_Y=%d -D RADIUSX=%d -D RADIUSY=%d%s%s"
                             " -D srcT=%s -D convertToWT=%s -D WT=%s -D dstT=%s -D convertToDstT=%s"
                             " -D %s -D srcT1=%s -D dstT1=%s -D WT1=%s -D CN=%d -D SHIFT_BITS=%d%s",
                             (int)lt2[0], (int)lt2[1], row_kernel.cols / 2, col_kernel.cols / 2,
                             ocl::kernelToStr(row_kernel, wdepth, "KERNEL_MATRIX_X").c_str(),
                             ocl::kernelToStr(col_kernel, wdepth, "KERNEL_MATRIX_Y").c_str(),
                             ocl::typeToStr(stype), ocl::convertTypeStr(sdepth, wdepth, cn, cvt[0]),
                             ocl::typeToStr(CV_MAKE_TYPE(wdepth, cn)), ocl::typeToStr(dtype),
                             ocl::convertTypeStr(wdepth, ddepth, cn, cvt[1]), borderMap[borderType],
                             ocl::typeToStr(sdepth), ocl::typeToStr(ddepth), ocl::typeToStr(wdepth),
                             cn, 2*shift_bits, int_arithm ? " -D INTEGER_ARITHMETIC" : "");

    ocl::Kernel k("sep_filter", ocl::imgproc::filterSep_singlePass_oclsrc, opts);
    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    _dst.create(size, dtype);
    UMat dst = _dst.getUMat();

    int src_offset_x = static_cast<int>((src_offset % src_step) / esz);
    int src_offset_y = static_cast<int>(src_offset / src_step);

    src.locateROI(wholeSize, origin);

    k.args(ocl::KernelArg::PtrReadOnly(src), (int)src_step, src_offset_x, src_offset_y,
           wholeSize.height, wholeSize.width, ocl::KernelArg::WriteOnly(dst),
           static_cast<float>(delta));

    return k.run(2, gt2, lt2, false);
}

bool ocl_sepFilter2D( InputArray _src, OutputArray _dst, int ddepth,
                      InputArray _kernelX, InputArray _kernelY, Point anchor,
                      double delta, int borderType )
{
    const ocl::Device & d = ocl::Device::getDefault();
    Size imgSize = _src.size();

    int type = _src.type(), sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    if (cn > 4)
        return false;

    Mat kernelX = _kernelX.getMat().reshape(1, 1);
    if (kernelX.cols % 2 != 1)
        return false;
    Mat kernelY = _kernelY.getMat().reshape(1, 1);
    if (kernelY.cols % 2 != 1)
        return false;

    if (ddepth < 0)
        ddepth = sdepth;

    if (anchor.x < 0)
        anchor.x = kernelX.cols >> 1;
    if (anchor.y < 0)
        anchor.y = kernelY.cols >> 1;

    int rtype = getKernelType(kernelX,
        kernelX.rows == 1 ? Point(anchor.x, 0) : Point(0, anchor.x));
    int ctype = getKernelType(kernelY,
        kernelY.rows == 1 ? Point(anchor.y, 0) : Point(0, anchor.y));

    int bdepth = CV_32F;
    bool int_arithm = false;
    if( sdepth == CV_8U && ddepth == CV_8U &&
        rtype == KERNEL_SMOOTH+KERNEL_SYMMETRICAL &&
        ctype == KERNEL_SMOOTH+KERNEL_SYMMETRICAL)
    {
        if (ocl::Device::getDefault().isIntel())
        {
            for (int i=0; i<kernelX.cols; i++)
                kernelX.at<float>(0, i) = (float) cvRound(kernelX.at<float>(0, i) * (1 << shift_bits));
            if (kernelX.data != kernelY.data)
                for (int i=0; i<kernelX.cols; i++)
                    kernelY.at<float>(0, i) = (float) cvRound(kernelY.at<float>(0, i) * (1 << shift_bits));
        } else
        {
            bdepth = CV_32S;
            kernelX.convertTo( kernelX, bdepth, 1 << shift_bits );
            kernelY.convertTo( kernelY, bdepth, 1 << shift_bits );
        }
        int_arithm = true;
    }

    CV_OCL_RUN_(kernelY.cols <= 21 && kernelX.cols <= 21 &&
                imgSize.width > optimizedSepFilterLocalWidth + anchor.x &&
                imgSize.height > optimizedSepFilterLocalHeight + anchor.y &&
                (!(borderType & BORDER_ISOLATED) || _src.offset() == 0) &&
                anchor == Point(kernelX.cols >> 1, kernelY.cols >> 1) &&
                OCL_PERFORMANCE_CHECK(d.isIntel()),  // TODO FIXIT
                ocl_sepFilter2D_SinglePass(_src, _dst, kernelX, kernelY, delta,
                                           borderType & ~BORDER_ISOLATED, ddepth, bdepth, int_arithm), true)

    UMat src = _src.getUMat();
    Size srcWholeSize; Point srcOffset;
    src.locateROI(srcWholeSize, srcOffset);

    bool fast8uc1 = type == CV_8UC1 && srcOffset.x % 4 == 0 &&
            src.cols % 4 == 0 && src.step % 4 == 0;

    Size srcSize = src.size();
    Size bufSize(srcSize.width, srcSize.height + kernelY.cols - 1);
    UMat buf(bufSize, CV_MAKETYPE(bdepth, cn));
    if (!ocl_sepRowFilter2D(src, buf, kernelX, anchor.x, borderType, ddepth, fast8uc1, int_arithm))
        return false;

    _dst.create(srcSize, CV_MAKETYPE(ddepth, cn));
    UMat dst = _dst.getUMat();

    return ocl_sepColFilter2D(buf, dst, kernelY, delta, anchor.y, int_arithm);
}

#endif

Ptr<cv::BaseFilter> getLinearFilter(
        int srcType, int dstType,
        InputArray filter_kernel, Point anchor,
        double delta, int bits)
{
    CV_INSTRUMENT_REGION();

    Mat kernelMat = filter_kernel.getMat();
    CV_CPU_DISPATCH(getLinearFilter, (srcType, dstType, kernelMat, anchor, delta, bits),
        CV_CPU_DISPATCH_MODES_ALL);
}


Ptr<cv::FilterEngine> createLinearFilter(
        int _srcType, int _dstType,
        InputArray filter_kernel,
        Point _anchor, double _delta,
        int _rowBorderType, int _columnBorderType,
        const Scalar& _borderValue)
{
    Mat _kernel = filter_kernel.getMat();
    _srcType = CV_MAT_TYPE(_srcType);
    _dstType = CV_MAT_TYPE(_dstType);
    int cn = CV_MAT_CN(_srcType);
    CV_Assert( cn == CV_MAT_CN(_dstType) );

    Mat kernel = _kernel;
    int bits = 0;

    /*int sdepth = CV_MAT_DEPTH(_srcType), ddepth = CV_MAT_DEPTH(_dstType);
    int ktype = _kernel.depth() == CV_32S ? KERNEL_INTEGER : getKernelType(_kernel, _anchor);
    if( sdepth == CV_8U && (ddepth == CV_8U || ddepth == CV_16S) &&
        _kernel.rows*_kernel.cols <= (1 << 10) )
    {
        bits = (ktype & KERNEL_INTEGER) ? 0 : 11;
        _kernel.convertTo(kernel, CV_32S, 1 << bits);
    }*/

    Ptr<BaseFilter> _filter2D = getLinearFilter(_srcType, _dstType,
        kernel, _anchor, _delta, bits);

    return makePtr<FilterEngine>(_filter2D, Ptr<BaseRowFilter>(),
        Ptr<BaseColumnFilter>(), _srcType, _dstType, _srcType,
        _rowBorderType, _columnBorderType, _borderValue );
}


//================================================================
// HAL interface
//================================================================

static bool replacementFilter2D(int stype, int dtype, int kernel_type,
                                uchar * src_data, size_t src_step,
                                uchar * dst_data, size_t dst_step,
                                int width, int height,
                                int full_width, int full_height,
                                int offset_x, int offset_y,
                                uchar * kernel_data, size_t kernel_step,
                                int kernel_width, int kernel_height,
                                int anchor_x, int anchor_y,
                                double delta, int borderType, bool isSubmatrix)
{
    cvhalFilter2D* ctx;
    int res = cv_hal_filterInit(&ctx, kernel_data, kernel_step, kernel_type, kernel_width, kernel_height, width, height,
                                stype, dtype, borderType, delta, anchor_x, anchor_y, isSubmatrix, src_data == dst_data);
    if (res != CV_HAL_ERROR_OK)
        return false;
    res = cv_hal_filter(ctx, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
    bool success = (res == CV_HAL_ERROR_OK);
    res = cv_hal_filterFree(ctx);
    if (res != CV_HAL_ERROR_OK)
        return false;
    return success;
}

#ifdef HAVE_IPP
static bool ippFilter2D(int stype, int dtype, int kernel_type,
              uchar * src_data, size_t src_step,
              uchar * dst_data, size_t dst_step,
              int width, int height,
              int full_width, int full_height,
              int offset_x, int offset_y,
              uchar * kernel_data, size_t kernel_step,
              int kernel_width, int kernel_height,
              int anchor_x, int anchor_y,
              double delta, int borderType,
              bool isSubmatrix)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP();

    ::ipp::IwiSize  iwSize(width, height);
    ::ipp::IwiSize  kernelSize(kernel_width, kernel_height);
    IppDataType     type        = ippiGetDataType(CV_MAT_DEPTH(stype));
    int             channels    = CV_MAT_CN(stype);

    CV_UNUSED(isSubmatrix);

#if IPP_VERSION_X100 >= 201700 && IPP_VERSION_X100 <= 201702 // IPP bug with 1x1 kernel
    if(kernel_width == 1 && kernel_height == 1)
        return false;
#endif

#if IPP_DISABLE_FILTER2D_BIG_MASK
    // Too big difference compared to OpenCV FFT-based convolution
    if(kernel_type == CV_32FC1 && (type == ipp16s || type == ipp16u) && (kernel_width > 7 || kernel_height > 7))
        return false;

    // Poor optimization for big kernels
    if(kernel_width > 7 || kernel_height > 7)
        return false;
#endif

    if(src_data == dst_data)
        return false;

    if(stype != dtype)
        return false;

    if(kernel_type != CV_16SC1 && kernel_type != CV_32FC1)
        return false;

    // TODO: Implement offset for 8u, 16u
    if(std::fabs(delta) >= DBL_EPSILON)
        return false;

    if(!ippiCheckAnchor(anchor_x, anchor_y, kernel_width, kernel_height))
        return false;

    try
    {
        ::ipp::IwiBorderSize    iwBorderSize;
        ::ipp::IwiBorderType    iwBorderType;
        ::ipp::IwiImage         iwKernel(ippiSize(kernel_width, kernel_height), ippiGetDataType(CV_MAT_DEPTH(kernel_type)), CV_MAT_CN(kernel_type), 0, (void*)kernel_data, kernel_step);
        ::ipp::IwiImage         iwSrc(iwSize, type, channels, ::ipp::IwiBorderSize(offset_x, offset_y, full_width-offset_x-width, full_height-offset_y-height), (void*)src_data, src_step);
        ::ipp::IwiImage         iwDst(iwSize, type, channels, ::ipp::IwiBorderSize(offset_x, offset_y, full_width-offset_x-width, full_height-offset_y-height), (void*)dst_data, dst_step);

        iwBorderSize = ::ipp::iwiSizeToBorderSize(kernelSize);
        iwBorderType = ippiGetBorder(iwSrc, borderType, iwBorderSize);
        if(!iwBorderType)
            return false;

        CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilter, iwSrc, iwDst, iwKernel, ::ipp::IwiFilterParams(1, 0, ippAlgHintNone, ippRndFinancial), iwBorderType);
    }
    catch(const ::ipp::IwException& ex)
    {
        CV_UNUSED(ex);
        return false;
    }

    return true;
#else
    CV_UNUSED(stype); CV_UNUSED(dtype); CV_UNUSED(kernel_type); CV_UNUSED(src_data); CV_UNUSED(src_step);
    CV_UNUSED(dst_data); CV_UNUSED(dst_step); CV_UNUSED(width); CV_UNUSED(height); CV_UNUSED(full_width);
    CV_UNUSED(full_height); CV_UNUSED(offset_x); CV_UNUSED(offset_y); CV_UNUSED(kernel_data); CV_UNUSED(kernel_step);
    CV_UNUSED(kernel_width); CV_UNUSED(kernel_height); CV_UNUSED(anchor_x); CV_UNUSED(anchor_y); CV_UNUSED(delta);
    CV_UNUSED(borderType); CV_UNUSED(isSubmatrix);
    return false;
#endif
}
#endif

static bool dftFilter2D(int stype, int dtype, int kernel_type,
                        uchar * src_data, size_t src_step,
                        uchar * dst_data, size_t dst_step,
                        int full_width, int full_height,
                        int offset_x, int offset_y,
                        uchar * kernel_data, size_t kernel_step,
                        int kernel_width, int kernel_height,
                        int anchor_x, int anchor_y,
                        double delta, int borderType)
{
    {
        int sdepth = CV_MAT_DEPTH(stype);
        int ddepth = CV_MAT_DEPTH(dtype);
        int dft_filter_size = checkHardwareSupport(CV_CPU_SSE3) && ((sdepth == CV_8U && (ddepth == CV_8U || ddepth == CV_16S)) || (sdepth == CV_32F && ddepth == CV_32F)) ? 130 : 50;
        if (kernel_width * kernel_height < dft_filter_size)
            return false;
    }

    Point anchor = Point(anchor_x, anchor_y);
    Mat kernel = Mat(Size(kernel_width, kernel_height), kernel_type, kernel_data, kernel_step);

    Mat src(Size(full_width-offset_x, full_height-offset_y), stype, src_data, src_step);
    Mat dst(Size(full_width, full_height), dtype, dst_data, dst_step);
    Mat temp;
    int src_channels = CV_MAT_CN(stype);
    int dst_channels = CV_MAT_CN(dtype);
    int ddepth = CV_MAT_DEPTH(dtype);
    // crossCorr doesn't accept non-zero delta with multiple channels
    if (src_channels != 1 && delta != 0) {
        // The semantics of filter2D require that the delta be applied
        // as floating-point math.  So wee need an intermediate Mat
        // with a float datatype.  If the dest is already floats,
        // we just use that.
        int corrDepth = ddepth;
        if ((ddepth == CV_32F || ddepth == CV_64F) && src_data != dst_data) {
            temp = Mat(Size(full_width, full_height), dtype, dst_data, dst_step);
        } else {
            corrDepth = ddepth == CV_64F ? CV_64F : CV_32F;
            temp.create(Size(full_width, full_height), CV_MAKETYPE(corrDepth, dst_channels));
        }
        crossCorr(src, kernel, temp, src.size(),
                  CV_MAKETYPE(corrDepth, src_channels),
                  anchor, 0, borderType);
        add(temp, delta, temp);
        if (temp.data != dst_data) {
            temp.convertTo(dst, dst.type());
        }
    } else {
        if (src_data != dst_data)
            temp = Mat(Size(full_width, full_height), dtype, dst_data, dst_step);
        else
            temp.create(Size(full_width, full_height), dtype);
        crossCorr(src, kernel, temp, src.size(),
                  CV_MAKETYPE(ddepth, src_channels),
                  anchor, delta, borderType);
        if (temp.data != dst_data)
            temp.copyTo(dst);
    }
    return true;
}

static void ocvFilter2D(int stype, int dtype, int kernel_type,
                        uchar * src_data, size_t src_step,
                        uchar * dst_data, size_t dst_step,
                        int width, int height,
                        int full_width, int full_height,
                        int offset_x, int offset_y,
                        uchar * kernel_data, size_t kernel_step,
                        int kernel_width, int kernel_height,
                        int anchor_x, int anchor_y,
                        double delta, int borderType)
{
    int borderTypeValue = borderType & ~BORDER_ISOLATED;
    Mat kernel = Mat(Size(kernel_width, kernel_height), kernel_type, kernel_data, kernel_step);
    Ptr<FilterEngine> f = createLinearFilter(stype, dtype, kernel, Point(anchor_x, anchor_y), delta,
                                             borderTypeValue);
    Mat src(Size(width, height), stype, src_data, src_step);
    Mat dst(Size(width, height), dtype, dst_data, dst_step);
    f->apply(src, dst, Size(full_width, full_height), Point(offset_x, offset_y));
}

static bool replacementSepFilter(int stype, int dtype, int ktype,
                                 uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                                 int width, int height, int full_width, int full_height,
                                 int offset_x, int offset_y,
                                 uchar * kernelx_data, int kernelx_len,
                                 uchar * kernely_data, int kernely_len,
                                 int anchor_x, int anchor_y, double delta, int borderType)
{
    cvhalFilter2D *ctx;
    int res = cv_hal_sepFilterInit(&ctx, stype, dtype, ktype,
                                   kernelx_data, kernelx_len,
                                   kernely_data, kernely_len,
                                   anchor_x, anchor_y, delta, borderType);
    if (res != CV_HAL_ERROR_OK)
        return false;
    res = cv_hal_sepFilter(ctx, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y);
    bool success = (res == CV_HAL_ERROR_OK);
    res = cv_hal_sepFilterFree(ctx);
    if (res != CV_HAL_ERROR_OK)
        return false;
    return success;
}

static void ocvSepFilter(int stype, int dtype, int ktype,
                         uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                         int width, int height, int full_width, int full_height,
                         int offset_x, int offset_y,
                         uchar * kernelx_data, int kernelx_len,
                         uchar * kernely_data, int kernely_len,
                         int anchor_x, int anchor_y, double delta, int borderType)
{
    Mat kernelX(Size(kernelx_len, 1), ktype, kernelx_data);
    Mat kernelY(Size(kernely_len, 1), ktype, kernely_data);
    Ptr<FilterEngine> f = createSeparableLinearFilter(stype, dtype, kernelX, kernelY,
                                                      Point(anchor_x, anchor_y),
                                                      delta, borderType & ~BORDER_ISOLATED);
    Mat src(Size(width, height), stype, src_data, src_step);
    Mat dst(Size(width, height), dtype, dst_data, dst_step);
    f->apply(src, dst, Size(full_width, full_height), Point(offset_x, offset_y));
};

//===================================================================
//       HAL functions
//===================================================================

namespace hal {


CV_DEPRECATED Ptr<hal::Filter2D> Filter2D::create(uchar * , size_t , int ,
                                 int , int ,
                                 int , int ,
                                 int , int ,
                                 int , double ,
                                 int , int ,
                                 bool , bool ) { return Ptr<hal::Filter2D>(); }

CV_DEPRECATED Ptr<hal::SepFilter2D> SepFilter2D::create(int , int , int ,
                                    uchar * , int ,
                                    uchar * , int ,
                                    int , int ,
                                    double , int )  { return Ptr<hal::SepFilter2D>(); }


void filter2D(int stype, int dtype, int kernel_type,
              uchar * src_data, size_t src_step,
              uchar * dst_data, size_t dst_step,
              int width, int height,
              int full_width, int full_height,
              int offset_x, int offset_y,
              uchar * kernel_data, size_t kernel_step,
              int kernel_width, int kernel_height,
              int anchor_x, int anchor_y,
              double delta, int borderType,
              bool isSubmatrix)
{
    bool res;
    res = replacementFilter2D(stype, dtype, kernel_type,
                              src_data, src_step,
                              dst_data, dst_step,
                              width, height,
                              full_width, full_height,
                              offset_x, offset_y,
                              kernel_data, kernel_step,
                              kernel_width, kernel_height,
                              anchor_x, anchor_y,
                              delta, borderType, isSubmatrix);
    if (res)
        return;

    CV_IPP_RUN_FAST(ippFilter2D(stype, dtype, kernel_type,
                              src_data, src_step,
                              dst_data, dst_step,
                              width, height,
                              full_width, full_height,
                              offset_x, offset_y,
                              kernel_data, kernel_step,
                              kernel_width, kernel_height,
                              anchor_x, anchor_y,
                              delta, borderType, isSubmatrix))

    res = dftFilter2D(stype, dtype, kernel_type,
                      src_data, src_step,
                      dst_data, dst_step,
                      full_width, full_height,
                      offset_x, offset_y,
                      kernel_data, kernel_step,
                      kernel_width, kernel_height,
                      anchor_x, anchor_y,
                      delta, borderType);
    if (res)
        return;
    ocvFilter2D(stype, dtype, kernel_type,
                src_data, src_step,
                dst_data, dst_step,
                width, height,
                full_width, full_height,
                offset_x, offset_y,
                kernel_data, kernel_step,
                kernel_width, kernel_height,
                anchor_x, anchor_y,
                delta, borderType);
}

//---------------------------------------------------------------

void sepFilter2D(int stype, int dtype, int ktype,
                 uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                 int width, int height, int full_width, int full_height,
                 int offset_x, int offset_y,
                 uchar * kernelx_data, int kernelx_len,
                 uchar * kernely_data, int kernely_len,
                 int anchor_x, int anchor_y, double delta, int borderType)
{

    bool res = replacementSepFilter(stype, dtype, ktype,
                                    src_data, src_step, dst_data, dst_step,
                                    width, height, full_width, full_height,
                                    offset_x, offset_y,
                                    kernelx_data, kernelx_len,
                                    kernely_data, kernely_len,
                                    anchor_x, anchor_y, delta, borderType);
    if (res)
        return;
    ocvSepFilter(stype, dtype, ktype,
                 src_data, src_step, dst_data, dst_step,
                 width, height, full_width, full_height,
                 offset_x, offset_y,
                 kernelx_data, kernelx_len,
                 kernely_data, kernely_len,
                 anchor_x, anchor_y, delta, borderType);
}

} // namespace cv::hal::

//================================================================
//   Main interface
//================================================================

void filter2D(InputArray _src, OutputArray _dst, int ddepth,
              InputArray _kernel, Point anchor0,
              double delta, int borderType)
{
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2,
               ocl_filter2D(_src, _dst, ddepth, _kernel, anchor0, delta, borderType))

    Mat src = _src.getMat(), kernel = _kernel.getMat();

    if( ddepth < 0 )
        ddepth = src.depth();

    _dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );
    Mat dst = _dst.getMat();
    Point anchor = normalizeAnchor(anchor0, kernel.size());

    Point ofs;
    Size wsz(src.cols, src.rows);
    if( (borderType & BORDER_ISOLATED) == 0 )
        src.locateROI( wsz, ofs );

    hal::filter2D(src.type(), dst.type(), kernel.type(),
                  src.data, src.step, dst.data, dst.step,
                  dst.cols, dst.rows, wsz.width, wsz.height, ofs.x, ofs.y,
                  kernel.data, kernel.step,  kernel.cols, kernel.rows,
                  anchor.x, anchor.y,
                  delta, borderType, src.isSubmatrix());
}

void sepFilter2D(InputArray _src, OutputArray _dst, int ddepth,
                 InputArray _kernelX, InputArray _kernelY, Point anchor,
                 double delta, int borderType)
{
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2 && (size_t)_src.rows() > _kernelY.total() && (size_t)_src.cols() > _kernelX.total(),
               ocl_sepFilter2D(_src, _dst, ddepth, _kernelX, _kernelY, anchor, delta, borderType))

    Mat src = _src.getMat(), kernelX = _kernelX.getMat(), kernelY = _kernelY.getMat();

    if( ddepth < 0 )
        ddepth = src.depth();

    _dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );
    Mat dst = _dst.getMat();

    Point ofs;
    Size wsz(src.cols, src.rows);
    if( (borderType & BORDER_ISOLATED) == 0 )
        src.locateROI( wsz, ofs );

    CV_Assert( kernelX.type() == kernelY.type() &&
               (kernelX.cols == 1 || kernelX.rows == 1) &&
               (kernelY.cols == 1 || kernelY.rows == 1) );

    Mat contKernelX = kernelX.isContinuous() ? kernelX : kernelX.clone();
    Mat contKernelY = kernelY.isContinuous() ? kernelY : kernelY.clone();

    hal::sepFilter2D(src.type(), dst.type(), kernelX.type(),
                     src.data, src.step, dst.data, dst.step,
                     dst.cols, dst.rows, wsz.width, wsz.height, ofs.x, ofs.y,
                     contKernelX.data, kernelX.cols + kernelX.rows - 1,
                     contKernelY.data, kernelY.cols + kernelY.rows - 1,
                     anchor.x, anchor.y, delta, borderType & ~BORDER_ISOLATED);
}

} // namespace

CV_IMPL void
cvFilter2D( const CvArr* srcarr, CvArr* dstarr, const CvMat* _kernel, CvPoint anchor )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    cv::Mat kernel = cv::cvarrToMat(_kernel);

    CV_Assert( src.size() == dst.size() && src.channels() == dst.channels() );

    cv::filter2D( src, dst, dst.depth(), kernel, anchor, 0, cv::BORDER_REPLICATE );
}

/* End of file. */
