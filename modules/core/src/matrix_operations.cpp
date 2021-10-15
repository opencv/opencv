// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types_c.h"
#include "opencl_kernels_core.hpp"

#undef HAVE_IPP
#undef CV_IPP_RUN_FAST
#define CV_IPP_RUN_FAST(f, ...)
#undef CV_IPP_RUN
#define CV_IPP_RUN(c, f, ...)

/*************************************************************************************************\
                                        Matrix Operations
\*************************************************************************************************/

void cv::swap( Mat& a, Mat& b )
{
    std::swap(a.flags, b.flags);
    std::swap(a.dims, b.dims);
    std::swap(a.rows, b.rows);
    std::swap(a.cols, b.cols);
    std::swap(a.data, b.data);
    std::swap(a.datastart, b.datastart);
    std::swap(a.dataend, b.dataend);
    std::swap(a.datalimit, b.datalimit);
    std::swap(a.allocator, b.allocator);
    std::swap(a.u, b.u);

    std::swap(a.size.p, b.size.p);
    std::swap(a.step.p, b.step.p);
    std::swap(a.step.buf[0], b.step.buf[0]);
    std::swap(a.step.buf[1], b.step.buf[1]);

    if( a.step.p == b.step.buf )
    {
        a.step.p = a.step.buf;
        a.size.p = &a.rows;
    }

    if( b.step.p == a.step.buf )
    {
        b.step.p = b.step.buf;
        b.size.p = &b.rows;
    }
}


void cv::hconcat(const Mat* src, size_t nsrc, OutputArray _dst)
{
    CV_INSTRUMENT_REGION();

    if( nsrc == 0 || !src )
    {
        _dst.release();
        return;
    }

    int totalCols = 0, cols = 0;
    for( size_t i = 0; i < nsrc; i++ )
    {
        CV_Assert( src[i].dims <= 2 &&
                   src[i].rows == src[0].rows &&
                   src[i].type() == src[0].type());
        totalCols += src[i].cols;
    }
    _dst.create( src[0].rows, totalCols, src[0].type());
    Mat dst = _dst.getMat();
    for( size_t i = 0; i < nsrc; i++ )
    {
        Mat dpart = dst(Rect(cols, 0, src[i].cols, src[i].rows));
        src[i].copyTo(dpart);
        cols += src[i].cols;
    }
}

void cv::hconcat(InputArray src1, InputArray src2, OutputArray dst)
{
    CV_INSTRUMENT_REGION();

    Mat src[] = {src1.getMat(), src2.getMat()};
    hconcat(src, 2, dst);
}

void cv::hconcat(InputArray _src, OutputArray dst)
{
    CV_INSTRUMENT_REGION();

    std::vector<Mat> src;
    _src.getMatVector(src);
    hconcat(!src.empty() ? &src[0] : 0, src.size(), dst);
}

void cv::vconcat(const Mat* src, size_t nsrc, OutputArray _dst)
{
    CV_TRACE_FUNCTION_SKIP_NESTED()

    if( nsrc == 0 || !src )
    {
        _dst.release();
        return;
    }

    int totalRows = 0, rows = 0;
    for( size_t i = 0; i < nsrc; i++ )
    {
        CV_Assert(src[i].dims <= 2 &&
                  src[i].cols == src[0].cols &&
                  src[i].type() == src[0].type());
        totalRows += src[i].rows;
    }
    _dst.create( totalRows, src[0].cols, src[0].type());
    Mat dst = _dst.getMat();
    for( size_t i = 0; i < nsrc; i++ )
    {
        Mat dpart(dst, Rect(0, rows, src[i].cols, src[i].rows));
        src[i].copyTo(dpart);
        rows += src[i].rows;
    }
}

void cv::vconcat(InputArray src1, InputArray src2, OutputArray dst)
{
    CV_INSTRUMENT_REGION();

    Mat src[] = {src1.getMat(), src2.getMat()};
    vconcat(src, 2, dst);
}

void cv::vconcat(InputArray _src, OutputArray dst)
{
    CV_INSTRUMENT_REGION();

    std::vector<Mat> src;
    _src.getMatVector(src);
    vconcat(!src.empty() ? &src[0] : 0, src.size(), dst);
}

//////////////////////////////////////// set identity ////////////////////////////////////////////

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_setIdentity( InputOutputArray _m, const Scalar& s )
{
    int type = _m.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type), kercn = cn, rowsPerWI = 1;
    int sctype = CV_MAKE_TYPE(depth, cn == 3 ? 4 : cn);
    if (ocl::Device::getDefault().isIntel())
    {
        rowsPerWI = 4;
        if (cn == 1)
        {
            kercn = std::min(ocl::predictOptimalVectorWidth(_m), 4);
            if (kercn != 4)
                kercn = 1;
        }
    }

    ocl::Kernel k("setIdentity", ocl::core::set_identity_oclsrc,
                  format("-D T=%s -D T1=%s -D cn=%d -D ST=%s -D kercn=%d -D rowsPerWI=%d",
                         ocl::memopTypeToStr(CV_MAKE_TYPE(depth, kercn)),
                         ocl::memopTypeToStr(depth), cn,
                         ocl::memopTypeToStr(sctype),
                         kercn, rowsPerWI));
    if (k.empty())
        return false;

    UMat m = _m.getUMat();
    k.args(ocl::KernelArg::WriteOnly(m, cn, kercn),
           ocl::KernelArg::Constant(Mat(1, 1, sctype, s)));

    size_t globalsize[2] = { (size_t)m.cols * cn / kercn, ((size_t)m.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

void cv::setIdentity( InputOutputArray _m, const Scalar& s )
{
    CV_INSTRUMENT_REGION();

    CV_Assert( _m.dims() <= 2 );

    CV_OCL_RUN(_m.isUMat(),
               ocl_setIdentity(_m, s))

    Mat m = _m.getMat();
    int rows = m.rows, cols = m.cols, type = m.type();

    if( type == CV_32FC1 )
    {
        float* data = m.ptr<float>();
        float val = (float)s[0];
        size_t step = m.step/sizeof(data[0]);

        for( int i = 0; i < rows; i++, data += step )
        {
            for( int j = 0; j < cols; j++ )
                data[j] = 0;
            if( i < cols )
                data[i] = val;
        }
    }
    else if( type == CV_64FC1 )
    {
        double* data = m.ptr<double>();
        double val = s[0];
        size_t step = m.step/sizeof(data[0]);

        for( int i = 0; i < rows; i++, data += step )
        {
            for( int j = 0; j < cols; j++ )
                data[j] = j == i ? val : 0;
        }
    }
    else
    {
        m = Scalar(0);
        m.diag() = s;
    }
}


namespace cv {

UMat UMat::eye(int rows, int cols, int type, UMatUsageFlags usageFlags)
{
    return UMat::eye(Size(cols, rows), type, usageFlags);
}

UMat UMat::eye(Size size, int type, UMatUsageFlags usageFlags)
{
    UMat m(size, type, usageFlags);
    setIdentity(m);
    return m;
}

}  // namespace

//////////////////////////////////////////// trace ///////////////////////////////////////////

cv::Scalar cv::trace( InputArray _m )
{
    CV_INSTRUMENT_REGION();

    Mat m = _m.getMat();
    CV_Assert( m.dims <= 2 );
    int type = m.type();
    int nm = std::min(m.rows, m.cols);

    if( type == CV_32FC1 )
    {
        const float* ptr = m.ptr<float>();
        size_t step = m.step/sizeof(ptr[0]) + 1;
        double _s = 0;
        for( int i = 0; i < nm; i++ )
            _s += ptr[i*step];
        return _s;
    }

    if( type == CV_64FC1 )
    {
        const double* ptr = m.ptr<double>();
        size_t step = m.step/sizeof(ptr[0]) + 1;
        double _s = 0;
        for( int i = 0; i < nm; i++ )
            _s += ptr[i*step];
        return _s;
    }

    return cv::sum(m.diag());
}


////////////////////////////////////// completeSymm /////////////////////////////////////////

void cv::completeSymm( InputOutputArray _m, bool LtoR )
{
    CV_INSTRUMENT_REGION();

    Mat m = _m.getMat();
    size_t step = m.step, esz = m.elemSize();
    CV_Assert( m.dims <= 2 && m.rows == m.cols );

    int rows = m.rows;
    int j0 = 0, j1 = rows;

    uchar* data = m.ptr();
    for( int i = 0; i < rows; i++ )
    {
        if( !LtoR ) j1 = i; else j0 = i+1;
        for( int j = j0; j < j1; j++ )
            memcpy(data + (i*step + j*esz), data + (j*step + i*esz), esz);
    }
}


cv::Mat cv::Mat::cross(InputArray _m) const
{
    Mat m = _m.getMat();
    int tp = type(), d = CV_MAT_DEPTH(tp);
    CV_Assert( dims <= 2 && m.dims <= 2 && size() == m.size() && tp == m.type() &&
        ((rows == 3 && cols == 1) || (cols*channels() == 3 && rows == 1)));
    Mat result(rows, cols, tp);

    if( d == CV_32F )
    {
        const float *a = (const float*)data, *b = (const float*)m.data;
        float* c = (float*)result.data;
        size_t lda = rows > 1 ? step/sizeof(a[0]) : 1;
        size_t ldb = rows > 1 ? m.step/sizeof(b[0]) : 1;

        c[0] = a[lda] * b[ldb*2] - a[lda*2] * b[ldb];
        c[1] = a[lda*2] * b[0] - a[0] * b[ldb*2];
        c[2] = a[0] * b[ldb] - a[lda] * b[0];
    }
    else if( d == CV_64F )
    {
        const double *a = (const double*)data, *b = (const double*)m.data;
        double* c = (double*)result.data;
        size_t lda = rows > 1 ? step/sizeof(a[0]) : 1;
        size_t ldb = rows > 1 ? m.step/sizeof(b[0]) : 1;

        c[0] = a[lda] * b[ldb*2] - a[lda*2] * b[ldb];
        c[1] = a[lda*2] * b[0] - a[0] * b[ldb*2];
        c[2] = a[0] * b[ldb] - a[lda] * b[0];
    }

    return result;
}


////////////////////////////////////////// reduce ////////////////////////////////////////////

namespace cv
{

template<typename T, typename ST, class Op> static void
reduceR_( const Mat& srcmat, Mat& dstmat )
{
    typedef typename Op::rtype WT;
    Size size = srcmat.size();
    size.width *= srcmat.channels();
    AutoBuffer<WT> buffer(size.width);
    WT* buf = buffer.data();
    ST* dst = dstmat.ptr<ST>();
    const T* src = srcmat.ptr<T>();
    size_t srcstep = srcmat.step/sizeof(src[0]);
    int i;
    Op op;

    for( i = 0; i < size.width; i++ )
        buf[i] = src[i];

    for( ; --size.height; )
    {
        src += srcstep;
        i = 0;
        #if CV_ENABLE_UNROLLED
        for(; i <= size.width - 4; i += 4 )
        {
            WT s0, s1;
            s0 = op(buf[i], (WT)src[i]);
            s1 = op(buf[i+1], (WT)src[i+1]);
            buf[i] = s0; buf[i+1] = s1;

            s0 = op(buf[i+2], (WT)src[i+2]);
            s1 = op(buf[i+3], (WT)src[i+3]);
            buf[i+2] = s0; buf[i+3] = s1;
        }
        #endif
        for( ; i < size.width; i++ )
            buf[i] = op(buf[i], (WT)src[i]);
    }

    for( i = 0; i < size.width; i++ )
        dst[i] = (ST)buf[i];
}


template<typename T, typename ST, class Op> static void
reduceC_( const Mat& srcmat, Mat& dstmat )
{
    typedef typename Op::rtype WT;
    Size size = srcmat.size();
    int cn = srcmat.channels();
    size.width *= cn;
    Op op;

    for( int y = 0; y < size.height; y++ )
    {
        const T* src = srcmat.ptr<T>(y);
        ST* dst = dstmat.ptr<ST>(y);
        if( size.width == cn )
            for( int k = 0; k < cn; k++ )
                dst[k] = src[k];
        else
        {
            for( int k = 0; k < cn; k++ )
            {
                WT a0 = src[k], a1 = src[k+cn];
                int i;
                for( i = 2*cn; i <= size.width - 4*cn; i += 4*cn )
                {
                    a0 = op(a0, (WT)src[i+k]);
                    a1 = op(a1, (WT)src[i+k+cn]);
                    a0 = op(a0, (WT)src[i+k+cn*2]);
                    a1 = op(a1, (WT)src[i+k+cn*3]);
                }

                for( ; i < size.width; i += cn )
                {
                    a0 = op(a0, (WT)src[i+k]);
                }
                a0 = op(a0, a1);
                dst[k] = (ST)a0;
            }
        }
    }
}

typedef void (*ReduceFunc)( const Mat& src, Mat& dst );

}

#define reduceSumR8u32s  reduceR_<uchar, int,   OpAdd<int> >
#define reduceSumR8u32f  reduceR_<uchar, float, OpAdd<int> >
#define reduceSumR8u64f  reduceR_<uchar, double,OpAdd<int> >
#define reduceSumR16u32f reduceR_<ushort,float, OpAdd<float> >
#define reduceSumR16u64f reduceR_<ushort,double,OpAdd<double> >
#define reduceSumR16s32f reduceR_<short, float, OpAdd<float> >
#define reduceSumR16s64f reduceR_<short, double,OpAdd<double> >
#define reduceSumR32f32f reduceR_<float, float, OpAdd<float> >
#define reduceSumR32f64f reduceR_<float, double,OpAdd<double> >
#define reduceSumR64f64f reduceR_<double,double,OpAdd<double> >

#define reduceMaxR8u  reduceR_<uchar, uchar, OpMax<uchar> >
#define reduceMaxR16u reduceR_<ushort,ushort,OpMax<ushort> >
#define reduceMaxR16s reduceR_<short, short, OpMax<short> >
#define reduceMaxR32f reduceR_<float, float, OpMax<float> >
#define reduceMaxR64f reduceR_<double,double,OpMax<double> >

#define reduceMinR8u  reduceR_<uchar, uchar, OpMin<uchar> >
#define reduceMinR16u reduceR_<ushort,ushort,OpMin<ushort> >
#define reduceMinR16s reduceR_<short, short, OpMin<short> >
#define reduceMinR32f reduceR_<float, float, OpMin<float> >
#define reduceMinR64f reduceR_<double,double,OpMin<double> >

#ifdef HAVE_IPP
static inline bool ipp_reduceSumC_8u16u16s32f_64f(const cv::Mat& srcmat, cv::Mat& dstmat)
{
    int sstep = (int)srcmat.step, stype = srcmat.type(),
            ddepth = dstmat.depth();

    IppiSize roisize = { srcmat.size().width, 1 };

    typedef IppStatus (CV_STDCALL * IppiSum)(const void * pSrc, int srcStep, IppiSize roiSize, Ipp64f* pSum);
    typedef IppStatus (CV_STDCALL * IppiSumHint)(const void * pSrc, int srcStep, IppiSize roiSize, Ipp64f* pSum, IppHintAlgorithm hint);
    IppiSum ippiSum = 0;
    IppiSumHint ippiSumHint = 0;

    if(ddepth == CV_64F)
    {
        ippiSum =
            stype == CV_8UC1 ? (IppiSum)ippiSum_8u_C1R :
            stype == CV_8UC3 ? (IppiSum)ippiSum_8u_C3R :
            stype == CV_8UC4 ? (IppiSum)ippiSum_8u_C4R :
            stype == CV_16UC1 ? (IppiSum)ippiSum_16u_C1R :
            stype == CV_16UC3 ? (IppiSum)ippiSum_16u_C3R :
            stype == CV_16UC4 ? (IppiSum)ippiSum_16u_C4R :
            stype == CV_16SC1 ? (IppiSum)ippiSum_16s_C1R :
            stype == CV_16SC3 ? (IppiSum)ippiSum_16s_C3R :
            stype == CV_16SC4 ? (IppiSum)ippiSum_16s_C4R : 0;
        ippiSumHint =
            stype == CV_32FC1 ? (IppiSumHint)ippiSum_32f_C1R :
            stype == CV_32FC3 ? (IppiSumHint)ippiSum_32f_C3R :
            stype == CV_32FC4 ? (IppiSumHint)ippiSum_32f_C4R : 0;
    }

    if(ippiSum)
    {
        for(int y = 0; y < srcmat.size().height; y++)
        {
            if(CV_INSTRUMENT_FUN_IPP(ippiSum, srcmat.ptr(y), sstep, roisize, dstmat.ptr<Ipp64f>(y)) < 0)
                return false;
        }
        return true;
    }
    else if(ippiSumHint)
    {
        for(int y = 0; y < srcmat.size().height; y++)
        {
            if(CV_INSTRUMENT_FUN_IPP(ippiSumHint, srcmat.ptr(y), sstep, roisize, dstmat.ptr<Ipp64f>(y), ippAlgHintAccurate) < 0)
                return false;
        }
        return true;
    }

    return false;
}

static inline void reduceSumC_8u16u16s32f_64f(const cv::Mat& srcmat, cv::Mat& dstmat)
{
    CV_IPP_RUN_FAST(ipp_reduceSumC_8u16u16s32f_64f(srcmat, dstmat));

    cv::ReduceFunc func = 0;

    if(dstmat.depth() == CV_64F)
    {
        int sdepth = CV_MAT_DEPTH(srcmat.type());
        func =
            sdepth == CV_8U ? (cv::ReduceFunc)cv::reduceC_<uchar, double,   cv::OpAdd<double> > :
            sdepth == CV_16U ? (cv::ReduceFunc)cv::reduceC_<ushort, double,   cv::OpAdd<double> > :
            sdepth == CV_16S ? (cv::ReduceFunc)cv::reduceC_<short, double,   cv::OpAdd<double> > :
            sdepth == CV_32F ? (cv::ReduceFunc)cv::reduceC_<float, double,   cv::OpAdd<double> > : 0;
    }
    CV_Assert(func);

    func(srcmat, dstmat);
}

#endif

#define reduceSumC8u32s  reduceC_<uchar, int,   OpAdd<int> >
#define reduceSumC8u32f  reduceC_<uchar, float, OpAdd<int> >
#define reduceSumC16u32f reduceC_<ushort,float, OpAdd<float> >
#define reduceSumC16s32f reduceC_<short, float, OpAdd<float> >
#define reduceSumC32f32f reduceC_<float, float, OpAdd<float> >
#define reduceSumC64f64f reduceC_<double,double,OpAdd<double> >

#ifdef HAVE_IPP
#define reduceSumC8u64f  reduceSumC_8u16u16s32f_64f
#define reduceSumC16u64f reduceSumC_8u16u16s32f_64f
#define reduceSumC16s64f reduceSumC_8u16u16s32f_64f
#define reduceSumC32f64f reduceSumC_8u16u16s32f_64f
#else
#define reduceSumC8u64f  reduceC_<uchar, double,OpAdd<int> >
#define reduceSumC16u64f reduceC_<ushort,double,OpAdd<double> >
#define reduceSumC16s64f reduceC_<short, double,OpAdd<double> >
#define reduceSumC32f64f reduceC_<float, double,OpAdd<double> >
#endif

#ifdef HAVE_IPP
#define REDUCE_OP(favor, optype, type1, type2) \
static inline bool ipp_reduce##optype##C##favor(const cv::Mat& srcmat, cv::Mat& dstmat) \
{ \
    if((srcmat.channels() == 1)) \
    { \
        int sstep = (int)srcmat.step; \
        typedef Ipp##favor IppType; \
        IppiSize roisize = ippiSize(srcmat.size().width, 1);\
        for(int y = 0; y < srcmat.size().height; y++)\
        {\
            if(CV_INSTRUMENT_FUN_IPP(ippi##optype##_##favor##_C1R, srcmat.ptr<IppType>(y), sstep, roisize, dstmat.ptr<IppType>(y)) < 0)\
                return false;\
        }\
        return true;\
    }\
    return false; \
} \
static inline void reduce##optype##C##favor(const cv::Mat& srcmat, cv::Mat& dstmat) \
{ \
    CV_IPP_RUN_FAST(ipp_reduce##optype##C##favor(srcmat, dstmat)); \
    cv::reduceC_ < type1, type2, cv::Op##optype < type2 > >(srcmat, dstmat); \
}
#endif

#ifdef HAVE_IPP
REDUCE_OP(8u, Max, uchar, uchar)
REDUCE_OP(16u, Max, ushort, ushort)
REDUCE_OP(16s, Max, short, short)
REDUCE_OP(32f, Max, float, float)
#else
#define reduceMaxC8u  reduceC_<uchar, uchar, OpMax<uchar> >
#define reduceMaxC16u reduceC_<ushort,ushort,OpMax<ushort> >
#define reduceMaxC16s reduceC_<short, short, OpMax<short> >
#define reduceMaxC32f reduceC_<float, float, OpMax<float> >
#endif
#define reduceMaxC64f reduceC_<double,double,OpMax<double> >

#ifdef HAVE_IPP
REDUCE_OP(8u, Min, uchar, uchar)
REDUCE_OP(16u, Min, ushort, ushort)
REDUCE_OP(16s, Min, short, short)
REDUCE_OP(32f, Min, float, float)
#else
#define reduceMinC8u  reduceC_<uchar, uchar, OpMin<uchar> >
#define reduceMinC16u reduceC_<ushort,ushort,OpMin<ushort> >
#define reduceMinC16s reduceC_<short, short, OpMin<short> >
#define reduceMinC32f reduceC_<float, float, OpMin<float> >
#endif
#define reduceMinC64f reduceC_<double,double,OpMin<double> >

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_reduce(InputArray _src, OutputArray _dst,
                       int dim, int op, int op0, int stype, int dtype)
{
    const int min_opt_cols = 128, buf_cols = 32;
    int sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype),
            ddepth = CV_MAT_DEPTH(dtype), ddepth0 = ddepth;
    const ocl::Device &defDev = ocl::Device::getDefault();
    bool doubleSupport = defDev.doubleFPConfig() > 0;

    size_t wgs = defDev.maxWorkGroupSize();
    bool useOptimized = 1 == dim && _src.cols() > min_opt_cols && (wgs >= buf_cols);

    if (!doubleSupport && (sdepth == CV_64F || ddepth == CV_64F))
        return false;

    if (op == CV_REDUCE_AVG)
    {
        if (sdepth < CV_32S && ddepth < CV_32S)
            ddepth = CV_32S;
    }

    const char * const ops[4] = { "OCL_CV_REDUCE_SUM", "OCL_CV_REDUCE_AVG",
                                  "OCL_CV_REDUCE_MAX", "OCL_CV_REDUCE_MIN" };
    int wdepth = std::max(ddepth, CV_32F);
    if (useOptimized)
    {
        size_t tileHeight = (size_t)(wgs / buf_cols);
        if (defDev.isIntel())
        {
            static const size_t maxItemInGroupCount = 16;
            tileHeight = min(tileHeight, defDev.localMemSize() / buf_cols / CV_ELEM_SIZE(CV_MAKETYPE(wdepth, cn)) / maxItemInGroupCount);
        }
        char cvt[3][40];
        cv::String build_opt = format("-D OP_REDUCE_PRE -D BUF_COLS=%d -D TILE_HEIGHT=%zu -D %s -D dim=1"
                                            " -D cn=%d -D ddepth=%d"
                                            " -D srcT=%s -D bufT=%s -D dstT=%s"
                                            " -D convertToWT=%s -D convertToBufT=%s -D convertToDT=%s%s",
                                            buf_cols, tileHeight, ops[op], cn, ddepth,
                                            ocl::typeToStr(sdepth),
                                            ocl::typeToStr(ddepth),
                                            ocl::typeToStr(ddepth0),
                                            ocl::convertTypeStr(ddepth, wdepth, 1, cvt[0]),
                                            ocl::convertTypeStr(sdepth, ddepth, 1, cvt[1]),
                                            ocl::convertTypeStr(wdepth, ddepth0, 1, cvt[2]),
                                            doubleSupport ? " -D DOUBLE_SUPPORT" : "");
        ocl::Kernel k("reduce_horz_opt", ocl::core::reduce2_oclsrc, build_opt);
        if (k.empty())
            return false;
        UMat src = _src.getUMat();
        Size dsize(1, src.rows);
        _dst.create(dsize, dtype);
        UMat dst = _dst.getUMat();

        if (op0 == CV_REDUCE_AVG)
            k.args(ocl::KernelArg::ReadOnly(src),
                      ocl::KernelArg::WriteOnlyNoSize(dst), 1.0f / src.cols);
        else
            k.args(ocl::KernelArg::ReadOnly(src),
                      ocl::KernelArg::WriteOnlyNoSize(dst));

        size_t localSize[2] = { (size_t)buf_cols, (size_t)tileHeight};
        size_t globalSize[2] = { (size_t)buf_cols, (size_t)src.rows };
        return k.run(2, globalSize, localSize, false);
    }
    else
    {
        char cvt[2][40];
        cv::String build_opt = format("-D %s -D dim=%d -D cn=%d -D ddepth=%d"
                                      " -D srcT=%s -D dstT=%s -D dstT0=%s -D convertToWT=%s"
                                      " -D convertToDT=%s -D convertToDT0=%s%s",
                                      ops[op], dim, cn, ddepth, ocl::typeToStr(useOptimized ? ddepth : sdepth),
                                      ocl::typeToStr(ddepth), ocl::typeToStr(ddepth0),
                                      ocl::convertTypeStr(ddepth, wdepth, 1, cvt[0]),
                                      ocl::convertTypeStr(sdepth, ddepth, 1, cvt[0]),
                                      ocl::convertTypeStr(wdepth, ddepth0, 1, cvt[1]),
                                      doubleSupport ? " -D DOUBLE_SUPPORT" : "");

        ocl::Kernel k("reduce", ocl::core::reduce2_oclsrc, build_opt);
        if (k.empty())
            return false;

        UMat src = _src.getUMat();
        Size dsize(dim == 0 ? src.cols : 1, dim == 0 ? 1 : src.rows);
        _dst.create(dsize, dtype);
        UMat dst = _dst.getUMat();

        ocl::KernelArg srcarg = ocl::KernelArg::ReadOnly(src),
                temparg = ocl::KernelArg::WriteOnlyNoSize(dst);

        if (op0 == CV_REDUCE_AVG)
            k.args(srcarg, temparg, 1.0f / (dim == 0 ? src.rows : src.cols));
        else
            k.args(srcarg, temparg);

        size_t globalsize = std::max(dsize.width, dsize.height);
        return k.run(1, &globalsize, NULL, false);
    }
}

}

#endif

void cv::reduce(InputArray _src, OutputArray _dst, int dim, int op, int dtype)
{
    CV_INSTRUMENT_REGION();

    CV_Assert( _src.dims() <= 2 );
    int op0 = op;
    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
    if( dtype < 0 )
        dtype = _dst.fixedType() ? _dst.type() : stype;
    dtype = CV_MAKETYPE(dtype >= 0 ? dtype : stype, cn);
    int ddepth = CV_MAT_DEPTH(dtype);

    CV_Assert( cn == CV_MAT_CN(dtype) );
    CV_Assert( op == CV_REDUCE_SUM || op == CV_REDUCE_MAX ||
               op == CV_REDUCE_MIN || op == CV_REDUCE_AVG );

    CV_OCL_RUN(_dst.isUMat(),
               ocl_reduce(_src, _dst, dim, op, op0, stype, dtype))

    // Fake reference to source. Resolves issue 8693 in case of src == dst.
    UMat srcUMat;
    if (_src.isUMat())
        srcUMat = _src.getUMat();

    Mat src = _src.getMat();
    _dst.create(dim == 0 ? 1 : src.rows, dim == 0 ? src.cols : 1, dtype);
    Mat dst = _dst.getMat(), temp = dst;

    if( op == CV_REDUCE_AVG )
    {
        op = CV_REDUCE_SUM;
        if( sdepth < CV_32S && ddepth < CV_32S )
        {
            temp.create(dst.rows, dst.cols, CV_32SC(cn));
            ddepth = CV_32S;
        }
    }

    ReduceFunc func = 0;
    if( dim == 0 )
    {
        if( op == CV_REDUCE_SUM )
        {
            if(sdepth == CV_8U && ddepth == CV_32S)
                func = GET_OPTIMIZED(reduceSumR8u32s);
            else if(sdepth == CV_8U && ddepth == CV_32F)
                func = reduceSumR8u32f;
            else if(sdepth == CV_8U && ddepth == CV_64F)
                func = reduceSumR8u64f;
            else if(sdepth == CV_16U && ddepth == CV_32F)
                func = reduceSumR16u32f;
            else if(sdepth == CV_16U && ddepth == CV_64F)
                func = reduceSumR16u64f;
            else if(sdepth == CV_16S && ddepth == CV_32F)
                func = reduceSumR16s32f;
            else if(sdepth == CV_16S && ddepth == CV_64F)
                func = reduceSumR16s64f;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceSumR32f32f);
            else if(sdepth == CV_32F && ddepth == CV_64F)
                func = reduceSumR32f64f;
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceSumR64f64f;
        }
        else if(op == CV_REDUCE_MAX)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = GET_OPTIMIZED(reduceMaxR8u);
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMaxR16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMaxR16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceMaxR32f);
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMaxR64f;
        }
        else if(op == CV_REDUCE_MIN)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = GET_OPTIMIZED(reduceMinR8u);
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMinR16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMinR16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceMinR32f);
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMinR64f;
        }
    }
    else
    {
        if(op == CV_REDUCE_SUM)
        {
            if(sdepth == CV_8U && ddepth == CV_32S)
                func = GET_OPTIMIZED(reduceSumC8u32s);
            else if(sdepth == CV_8U && ddepth == CV_32F)
                func = reduceSumC8u32f;
            else if(sdepth == CV_8U && ddepth == CV_64F)
                func = reduceSumC8u64f;
            else if(sdepth == CV_16U && ddepth == CV_32F)
                func = reduceSumC16u32f;
            else if(sdepth == CV_16U && ddepth == CV_64F)
                func = reduceSumC16u64f;
            else if(sdepth == CV_16S && ddepth == CV_32F)
                func = reduceSumC16s32f;
            else if(sdepth == CV_16S && ddepth == CV_64F)
                func = reduceSumC16s64f;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceSumC32f32f);
            else if(sdepth == CV_32F && ddepth == CV_64F)
                func = reduceSumC32f64f;
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceSumC64f64f;
        }
        else if(op == CV_REDUCE_MAX)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = GET_OPTIMIZED(reduceMaxC8u);
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMaxC16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMaxC16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceMaxC32f);
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMaxC64f;
        }
        else if(op == CV_REDUCE_MIN)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = GET_OPTIMIZED(reduceMinC8u);
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMinC16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMinC16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceMinC32f);
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMinC64f;
        }
    }

    if( !func )
        CV_Error( CV_StsUnsupportedFormat,
                  "Unsupported combination of input and output array formats" );

    func( src, temp );

    if( op0 == CV_REDUCE_AVG )
        temp.convertTo(dst, dst.type(), 1./(dim == 0 ? src.rows : src.cols));
}


//////////////////////////////////////// sort ///////////////////////////////////////////

namespace cv
{

template<typename T> static void sort_( const Mat& src, Mat& dst, int flags )
{
    AutoBuffer<T> buf;
    int n, len;
    bool sortRows = (flags & 1) == CV_SORT_EVERY_ROW;
    bool inplace = src.data == dst.data;
    bool sortDescending = (flags & CV_SORT_DESCENDING) != 0;

    if( sortRows )
        n = src.rows, len = src.cols;
    else
    {
        n = src.cols, len = src.rows;
        buf.allocate(len);
    }
    T* bptr = buf.data();

    for( int i = 0; i < n; i++ )
    {
        T* ptr = bptr;
        if( sortRows )
        {
            T* dptr = dst.ptr<T>(i);
            if( !inplace )
            {
                const T* sptr = src.ptr<T>(i);
                memcpy(dptr, sptr, sizeof(T) * len);
            }
            ptr = dptr;
        }
        else
        {
            for( int j = 0; j < len; j++ )
                ptr[j] = src.ptr<T>(j)[i];
        }

        std::sort( ptr, ptr + len );
        if( sortDescending )
        {
            for( int j = 0; j < len/2; j++ )
                std::swap(ptr[j], ptr[len-1-j]);
        }

        if( !sortRows )
            for( int j = 0; j < len; j++ )
                dst.ptr<T>(j)[i] = ptr[j];
    }
}

#ifdef HAVE_IPP
typedef IppStatus (CV_STDCALL *IppSortFunc)(void  *pSrcDst, int    len, Ipp8u *pBuffer);

static IppSortFunc getSortFunc(int depth, bool sortDescending)
{
    if (!sortDescending)
        return depth == CV_8U ? (IppSortFunc)ippsSortRadixAscend_8u_I :
            depth == CV_16U ? (IppSortFunc)ippsSortRadixAscend_16u_I :
            depth == CV_16S ? (IppSortFunc)ippsSortRadixAscend_16s_I :
            depth == CV_32S ? (IppSortFunc)ippsSortRadixAscend_32s_I :
            depth == CV_32F ? (IppSortFunc)ippsSortRadixAscend_32f_I :
            depth == CV_64F ? (IppSortFunc)ippsSortRadixAscend_64f_I :
            0;
    else
        return depth == CV_8U ? (IppSortFunc)ippsSortRadixDescend_8u_I :
            depth == CV_16U ? (IppSortFunc)ippsSortRadixDescend_16u_I :
            depth == CV_16S ? (IppSortFunc)ippsSortRadixDescend_16s_I :
            depth == CV_32S ? (IppSortFunc)ippsSortRadixDescend_32s_I :
            depth == CV_32F ? (IppSortFunc)ippsSortRadixDescend_32f_I :
            depth == CV_64F ? (IppSortFunc)ippsSortRadixDescend_64f_I :
            0;
}

static bool ipp_sort(const Mat& src, Mat& dst, int flags)
{
    CV_INSTRUMENT_REGION_IPP();

    bool        sortRows        = (flags & 1) == CV_SORT_EVERY_ROW;
    bool        sortDescending  = (flags & CV_SORT_DESCENDING) != 0;
    bool        inplace         = (src.data == dst.data);
    int         depth           = src.depth();
    IppDataType type            = ippiGetDataType(depth);

    IppSortFunc ippsSortRadix_I = getSortFunc(depth, sortDescending);
    if(!ippsSortRadix_I)
        return false;

    if(sortRows)
    {
        AutoBuffer<Ipp8u> buffer;
        int               bufferSize;
        if(ippsSortRadixGetBufferSize(src.cols, type, &bufferSize) < 0)
            return false;

        buffer.allocate(bufferSize);

        if(!inplace)
            src.copyTo(dst);

        for(int i = 0; i < dst.rows; i++)
        {
            if(CV_INSTRUMENT_FUN_IPP(ippsSortRadix_I, (void*)dst.ptr(i), dst.cols, buffer.data()) < 0)
                return false;
        }
    }
    else
    {
        AutoBuffer<Ipp8u> buffer;
        int               bufferSize;
        if(ippsSortRadixGetBufferSize(src.rows, type, &bufferSize) < 0)
            return false;

        buffer.allocate(bufferSize);

        Mat  row(1, src.rows, src.type());
        Mat  srcSub;
        Mat  dstSub;
        Rect subRect(0,0,1,src.rows);

        for(int i = 0; i < src.cols; i++)
        {
            subRect.x = i;
            srcSub = Mat(src, subRect);
            dstSub = Mat(dst, subRect);
            srcSub.copyTo(row);

            if(CV_INSTRUMENT_FUN_IPP(ippsSortRadix_I, (void*)row.ptr(), dst.rows, buffer.data()) < 0)
                return false;

            row = row.reshape(1, dstSub.rows);
            row.copyTo(dstSub);
        }
    }

    return true;
}
#endif

template<typename _Tp> class LessThanIdx
{
public:
    LessThanIdx( const _Tp* _arr ) : arr(_arr) {}
    bool operator()(int a, int b) const { return arr[a] < arr[b]; }
    const _Tp* arr;
};

template<typename T> static void sortIdx_( const Mat& src, Mat& dst, int flags )
{
    AutoBuffer<T> buf;
    AutoBuffer<int> ibuf;
    bool sortRows = (flags & 1) == CV_SORT_EVERY_ROW;
    bool sortDescending = (flags & CV_SORT_DESCENDING) != 0;

    CV_Assert( src.data != dst.data );

    int n, len;
    if( sortRows )
        n = src.rows, len = src.cols;
    else
    {
        n = src.cols, len = src.rows;
        buf.allocate(len);
        ibuf.allocate(len);
    }
    T* bptr = buf.data();
    int* _iptr = ibuf.data();

    for( int i = 0; i < n; i++ )
    {
        T* ptr = bptr;
        int* iptr = _iptr;

        if( sortRows )
        {
            ptr = (T*)(src.data + src.step*i);
            iptr = dst.ptr<int>(i);
        }
        else
        {
            for( int j = 0; j < len; j++ )
                ptr[j] = src.ptr<T>(j)[i];
        }
        for( int j = 0; j < len; j++ )
            iptr[j] = j;

        std::sort( iptr, iptr + len, LessThanIdx<T>(ptr) );
        if( sortDescending )
        {
            for( int j = 0; j < len/2; j++ )
                std::swap(iptr[j], iptr[len-1-j]);
        }

        if( !sortRows )
            for( int j = 0; j < len; j++ )
                dst.ptr<int>(j)[i] = iptr[j];
    }
}

#ifdef HAVE_IPP
typedef IppStatus (CV_STDCALL *IppSortIndexFunc)(const void*  pSrc, Ipp32s srcStrideBytes, Ipp32s *pDstIndx, int len, Ipp8u *pBuffer);

static IppSortIndexFunc getSortIndexFunc(int depth, bool sortDescending)
{
    if (!sortDescending)
        return depth == CV_8U ? (IppSortIndexFunc)ippsSortRadixIndexAscend_8u :
            depth == CV_16U ? (IppSortIndexFunc)ippsSortRadixIndexAscend_16u :
            depth == CV_16S ? (IppSortIndexFunc)ippsSortRadixIndexAscend_16s :
            depth == CV_32S ? (IppSortIndexFunc)ippsSortRadixIndexAscend_32s :
            depth == CV_32F ? (IppSortIndexFunc)ippsSortRadixIndexAscend_32f :
            0;
    else
        return depth == CV_8U ? (IppSortIndexFunc)ippsSortRadixIndexDescend_8u :
            depth == CV_16U ? (IppSortIndexFunc)ippsSortRadixIndexDescend_16u :
            depth == CV_16S ? (IppSortIndexFunc)ippsSortRadixIndexDescend_16s :
            depth == CV_32S ? (IppSortIndexFunc)ippsSortRadixIndexDescend_32s :
            depth == CV_32F ? (IppSortIndexFunc)ippsSortRadixIndexDescend_32f :
            0;
}

static bool ipp_sortIdx( const Mat& src, Mat& dst, int flags )
{
    CV_INSTRUMENT_REGION_IPP();

    bool        sortRows        = (flags & 1) == SORT_EVERY_ROW;
    bool        sortDescending  = (flags & SORT_DESCENDING) != 0;
    int         depth           = src.depth();
    IppDataType type            = ippiGetDataType(depth);

    IppSortIndexFunc ippsSortRadixIndex = getSortIndexFunc(depth, sortDescending);
    if(!ippsSortRadixIndex)
        return false;

    if(sortRows)
    {
        AutoBuffer<Ipp8u> buffer;
        int               bufferSize;
        if(ippsSortRadixIndexGetBufferSize(src.cols, type, &bufferSize) < 0)
            return false;

        buffer.allocate(bufferSize);

        for(int i = 0; i < src.rows; i++)
        {
            if(CV_INSTRUMENT_FUN_IPP(ippsSortRadixIndex, (const void*)src.ptr(i), (Ipp32s)src.step[1], (Ipp32s*)dst.ptr(i), src.cols, buffer.data()) < 0)
                return false;
        }
    }
    else
    {
        Mat  dstRow(1, dst.rows, dst.type());
        Mat  dstSub;
        Rect subRect(0,0,1,src.rows);

        AutoBuffer<Ipp8u> buffer;
        int               bufferSize;
        if(ippsSortRadixIndexGetBufferSize(src.rows, type, &bufferSize) < 0)
            return false;

        buffer.allocate(bufferSize);

        Ipp32s srcStep = (Ipp32s)src.step[0];
        for(int i = 0; i < src.cols; i++)
        {
            subRect.x = i;
            dstSub = Mat(dst, subRect);

            if(CV_INSTRUMENT_FUN_IPP(ippsSortRadixIndex, (const void*)src.ptr(0, i), srcStep, (Ipp32s*)dstRow.ptr(), src.rows, buffer.data()) < 0)
                return false;

            dstRow = dstRow.reshape(1, dstSub.rows);
            dstRow.copyTo(dstSub);
        }
    }

    return true;
}
#endif

typedef void (*SortFunc)(const Mat& src, Mat& dst, int flags);
}

void cv::sort( InputArray _src, OutputArray _dst, int flags )
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat();
    CV_Assert( src.dims <= 2 && src.channels() == 1 );
    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();
    CV_IPP_RUN_FAST(ipp_sort(src, dst, flags));

    static SortFunc tab[] =
    {
        sort_<uchar>, sort_<schar>, sort_<ushort>, sort_<short>,
        sort_<int>, sort_<float>, sort_<double>, 0
    };
    SortFunc func = tab[src.depth()];
    CV_Assert( func != 0 );

    func( src, dst, flags );
}

void cv::sortIdx( InputArray _src, OutputArray _dst, int flags )
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat();
    CV_Assert( src.dims <= 2 && src.channels() == 1 );
    Mat dst = _dst.getMat();
    if( dst.data == src.data )
        _dst.release();
    _dst.create( src.size(), CV_32S );
    dst = _dst.getMat();

    CV_IPP_RUN_FAST(ipp_sortIdx(src, dst, flags));

    static SortFunc tab[] =
    {
        sortIdx_<uchar>, sortIdx_<schar>, sortIdx_<ushort>, sortIdx_<short>,
        sortIdx_<int>, sortIdx_<float>, sortIdx_<double>, 0
    };
    SortFunc func = tab[src.depth()];
    CV_Assert( func != 0 );
    func( src, dst, flags );
}
