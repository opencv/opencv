// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/core/mat.hpp"
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
    hconcat(!src.empty() ? &src[0] : nullptr, src.size(), dst);
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
    vconcat(!src.empty() ? &src[0] : nullptr, src.size(), dst);
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
    return k.run(2, globalsize, nullptr, false);
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
            std::fill(data, data + cols, 0.0);
            if (i < cols)
                data[i] = val;
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

template<typename stype, typename itype>
struct ReduceOpAdd
{
    using v_stype = stype;
    using v_itype = itype;
    static const int vlanes;
    static inline stype load(const stype *ptr, size_t step) { (void)step; return *ptr; }
    static inline itype init() { return (itype)0; }
    static inline itype reduce(const itype &val) { return val; }
    inline itype operator()(const itype &a, const stype &b) const { return a + (itype)b; }
};
template<typename stype, typename itype>
const int ReduceOpAdd<stype, itype>::vlanes = 1;

#if (CV_SIMD || CV_SIMD_SCALABLE)

template<typename T, typename VT>
static inline VT vx_load(const T *ptr, size_t step)
{
    constexpr int nlanes = VTraits<VT>::max_nlanes;
    T buf[nlanes];
    for (int i = 0; i < VTraits<VT>::vlanes(); i++)
    {
        buf[i] = *ptr;
        ptr += step;
    }
    return vx_load(buf);
}

#if CV_RVV
template<>
inline v_uint8 vx_load(const uchar *ptr, size_t step)
{
    return __riscv_vlse8_v_u8m2(ptr, step * sizeof(uchar), __riscv_vsetvlmax_e8m2());
}
template<>
inline v_uint16 vx_load(const ushort *ptr, size_t step)
{
    return __riscv_vlse16_v_u16m2(ptr, step * sizeof(ushort), __riscv_vsetvlmax_e16m2());
}
template<>
inline v_int16 vx_load(const short *ptr, size_t step)
{
    return __riscv_vlse16_v_i16m2(ptr, step * sizeof(short), __riscv_vsetvlmax_e16m2());
}
template<>
inline v_int32 vx_load(const int *ptr, size_t step)
{
    return __riscv_vlse32_v_i32m2(ptr, step * sizeof(int), __riscv_vsetvlmax_e32m2());
}
template<>
inline v_float32 vx_load(const float *ptr, size_t step)
{
    return __riscv_vlse32_v_f32m2(ptr, step * sizeof(float), __riscv_vsetvlmax_e32m2());
}
template<>
inline v_float64 vx_load(const double *ptr, size_t step)
{
    return __riscv_vlse64_v_f64m2(ptr, step * sizeof(double), __riscv_vsetvlmax_e64m2());
}
#endif

template<typename stype, typename itype>
struct ReduceVecOpAdd;

template<>
struct ReduceVecOpAdd<uchar, int>
{
    using stype = uchar;
    using itype = int;
    using v_stype = v_uint8;
    using v_itype = v_int32;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_s32(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const
    {
        v_uint16 b0, b1;
        v_expand(b, b0, b1);
        v_int16 sb = v_add(v_reinterpret_as_s16(b0), v_reinterpret_as_s16(b1));
        v_int32 sb0, sb1;
        v_expand(sb, sb0, sb1);
        return v_add(a, v_add(sb0, sb1));
    }
};
const int ReduceVecOpAdd<uchar, int>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpAdd<ushort, float>
{
    using stype = ushort;
    using itype = float;
    using v_stype = v_uint16;
    using v_itype = v_float32;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_f32(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const
    {
        v_uint32 b0, b1;
        v_expand(b, b0, b1);
        v_int32 sb = v_reinterpret_as_s32(v_add(b0, b1));
        return v_add(a, v_cvt_f32(sb));
    }
};
const int ReduceVecOpAdd<ushort, float>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpAdd<short, float>
{
    using stype = short;
    using itype = float;
    using v_stype = v_int16;
    using v_itype = v_float32;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_f32(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const
    {
        v_int32 b0, b1;
        v_expand(b, b0, b1);
        v_int32 sb = v_add(b0, b1);
        return v_add(a, v_cvt_f32(sb));
    }
};
const int ReduceVecOpAdd<short, float>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpAdd<float, float>
{
    using stype = float;
    using itype = float;
    using v_stype = v_float32;
    using v_itype = v_float32;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_f32(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_stype &a, const v_stype &b) const { return v_add(a, b); }
};
const int ReduceVecOpAdd<float, float>::vlanes = VTraits<v_stype>::vlanes();

#endif

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)

template<>
struct ReduceVecOpAdd<short, double>
{
    using stype = short;
    using itype = double;
    using v_stype = v_int16;
    using v_itype = v_float64;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_f64(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const
    {
        v_int32 b0, b1;
        v_expand(b, b0, b1);
        v_int32 sb = v_add(b0, b1);
        return v_add(a, v_add(v_cvt_f64(sb), v_cvt_f64_high(sb)));
    }
};
const int ReduceVecOpAdd<short, double>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpAdd<ushort, double>
{
    using stype = ushort;
    using itype = double;
    using v_stype = v_uint16;
    using v_itype = v_float64;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_f64(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const
    {
        v_uint32 b0, b1;
        v_expand(b, b0, b1);
        v_int32 sb = v_reinterpret_as_s32(v_add(b0, b1));
        return v_add(a, v_add(v_cvt_f64(sb), v_cvt_f64_high(sb)));
    }
};
const int ReduceVecOpAdd<ushort, double>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpAdd<float, double>
{
    using stype = float;
    using itype = double;
    using v_stype = v_float32;
    using v_itype = v_float64;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_f64(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const
    {
        return v_add(a, v_add(v_cvt_f64(b), v_cvt_f64_high(b)));
    }
};
const int ReduceVecOpAdd<float, double>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpAdd<double, double>
{
    using stype = double;
    using itype = double;
    using v_stype = v_float64;
    using v_itype = v_float64;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_f64(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const { return v_add(a, b); }
};
const int ReduceVecOpAdd<double, double>::vlanes = VTraits<v_stype>::vlanes();

#endif

template<typename stype, typename itype>
struct ReduceOpAddSqr
{
    using v_stype = stype;
    using v_itype = itype;
    static const int vlanes;
    static inline stype load(const stype *ptr, size_t step) { (void)step; return *ptr; }
    static inline itype init() { return (itype)0; }
    static inline itype reduce(const itype &val) { return val; }
    inline itype operator()(const itype &a, const stype &b) const { return a + (itype)b * (itype)b; }
};
template<typename stype, typename itype>
const int ReduceOpAddSqr<stype, itype>::vlanes = 1;

#if (CV_SIMD || CV_SIMD_SCALABLE)

template<typename stype, typename itype>
struct ReduceVecOpAddSqr;

template<>
struct ReduceVecOpAddSqr<uchar, int>
{
    using stype = uchar;
    using itype = int;
    using v_stype = v_uint8;
    using v_itype = v_int32;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_s32(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const
    {
        v_uint16 b0, b1;
        v_mul_expand(b, b, b0, b1);

        v_uint32 s00, s01;
        v_expand(b0, s00, s01);
        s00 = v_add(s00, s01);
        v_uint32 s10, s11;
        v_expand(b1, s10, s11);
        s10 = v_add(s10, s11);
        return v_add(a, v_reinterpret_as_s32(v_add(s00, s10)));
    }
};
const int ReduceVecOpAddSqr<uchar, int>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpAddSqr<ushort, float>
{
    using stype = ushort;
    using itype = float;
    using v_stype = v_uint16;
    using v_itype = v_float32;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_f32(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const
    {
        v_uint32 b0, b1;
        v_mul_expand(b, b, b0, b1);
        v_int32 sb = v_reinterpret_as_s32(v_add(b0, b1));
        return v_add(a, v_cvt_f32(sb));
    }
};
const int ReduceVecOpAddSqr<ushort, float>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpAddSqr<short, float>
{
    using stype = short;
    using itype = float;
    using v_stype = v_int16;
    using v_itype = v_float32;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_f32(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const
    {
        v_int32 b0, b1;
        v_mul_expand(b, b, b0, b1);
        v_int32 sb = v_add(b0, b1);
        return v_add(a, v_cvt_f32(sb));
    }
};
const int ReduceVecOpAddSqr<short, float>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpAddSqr<float, float>
{
    using stype = float;
    using itype = float;
    using v_stype = v_float32;
    using v_itype = v_float32;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_f32(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const { return v_add(a, v_mul(b, b)); }
};
const int ReduceVecOpAddSqr<float, float>::vlanes = VTraits<v_stype>::vlanes();

#endif

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)

template<>
struct ReduceVecOpAddSqr<short, double>
{
    using stype = short;
    using itype = double;
    using v_stype = v_int16;
    using v_itype = v_float64;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_f64(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const
    {
        v_int32 b0, b1;
        v_mul_expand(b, b, b0, b1);
        v_int32 sb = v_add(b0, b1);
        return v_add(a, v_add(v_cvt_f64(sb), v_cvt_f64_high(sb)));
    }
};
const int ReduceVecOpAddSqr<short, double>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpAddSqr<ushort, double>
{
    using stype = ushort;
    using itype = double;
    using v_stype = v_uint16;
    using v_itype = v_float64;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_f64(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const
    {
        v_uint32 b0, b1;
        v_mul_expand(b, b,  b0, b1);
        v_int32 sb = v_reinterpret_as_s32(v_add(b0, b1));
        return v_add(a, v_add(v_cvt_f64(sb), v_cvt_f64_high(sb)));
    }
};
const int ReduceVecOpAddSqr<ushort, double>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpAddSqr<float, double>
{
    using stype = float;
    using itype = double;
    using v_stype = v_float32;
    using v_itype = v_float64;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_f64(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const
    {
        v_itype b0 = v_cvt_f64(b), b1 = v_cvt_f64_high(b);
        return v_add(a, v_add(v_mul(b0, b0), v_mul(b1, b1)));
    }
};
const int ReduceVecOpAddSqr<float, double>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpAddSqr<double, double>
{
    using stype = double;
    using itype = double;
    using v_stype = v_float64;
    using v_itype = v_float64;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setzero_f64(); }
    static inline itype reduce(const v_itype &val) { return v_reduce_sum(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const { return v_add(a, v_mul(b, b)); }
};
const int ReduceVecOpAddSqr<double, double>::vlanes = VTraits<v_stype>::vlanes();

#endif

template<typename stype>
struct ReduceOpMax
{
    using v_stype = stype;
    using v_itype = stype;
    static const int vlanes;
    static inline stype load(const stype *ptr, size_t step) { (void)step; return *ptr; }
    static inline stype init() { return std::numeric_limits<stype>::lowest(); }
    static inline stype reduce(const stype &val) { return val; }
    inline stype operator()(const stype &a, const stype &b) const { return std::max(a, b); }
};
template<typename stype>
const int ReduceOpMax<stype>::vlanes = 1;

#if (CV_SIMD || CV_SIMD_SCALABLE)

template<typename stype>
struct ReduceVecOpMax;

template<>
struct ReduceVecOpMax<uchar>
{
    using stype = uchar;
    using itype = uchar;
    using v_stype = v_uint8;
    using v_itype = v_uint8;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setall_u8(std::numeric_limits<itype>::lowest()); }
    static inline itype reduce(const v_itype &val) { return v_reduce_max(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const { return v_max(a, b); }
};
const int ReduceVecOpMax<uchar>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpMax<ushort>
{
    using stype = ushort;
    using itype = ushort;
    using v_stype = v_uint16;
    using v_itype = v_uint16;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setall_u16(std::numeric_limits<itype>::lowest()); }
    static inline itype reduce(const v_itype &val) { return v_reduce_max(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const { return v_max(a, b); }
};
const int ReduceVecOpMax<ushort>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpMax<short>
{
    using stype = short;
    using itype = short;
    using v_stype = v_int16;
    using v_itype = v_int16;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setall_s16(std::numeric_limits<itype>::lowest()); }
    static inline itype reduce(const v_itype &val) { return v_reduce_max(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const { return v_max(a, b); }
};
const int ReduceVecOpMax<short>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpMax<float>
{
    using stype = float;
    using itype = float;
    using v_stype = v_float32;
    using v_itype = v_float32;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setall_f32(std::numeric_limits<itype>::lowest()); }
    static inline itype reduce(const v_itype &val) { return v_reduce_max(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const { return v_max(a, b); }
};
const int ReduceVecOpMax<float>::vlanes = VTraits<v_stype>::vlanes();

#endif

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)

template<>
struct ReduceVecOpMax<double>
{
    using stype = double;
    using itype = double;
    using v_stype = v_float64;
    using v_itype = v_float64;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setall_f64(std::numeric_limits<itype>::lowest()); }
    static inline itype reduce(const v_itype &val)
    {
        constexpr int nlanes = VTraits<v_itype>::max_nlanes;
        itype buf[nlanes];
        vx_store(buf, val);
        itype m = buf[0];
        for (int i = 1; i < VTraits<v_itype>::vlanes(); i++)
        {
            if (m < buf[i]) m = buf[i];
        }
        return m;
    }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const { return v_max(a, b); }
};
const int ReduceVecOpMax<double>::vlanes = VTraits<v_stype>::vlanes();

#endif

template<typename stype>
struct ReduceOpMin
{
    using v_stype = stype;
    using v_itype = stype;
    static const int vlanes;
    static inline stype load(const stype *ptr, size_t step) { (void)step; return *ptr; }
    static inline stype init() { return std::numeric_limits<stype>::max(); }
    static inline stype reduce(const stype &val) { return (stype)val; }
    inline stype operator()(const stype &a, const stype &b) const { return std::min(a, b); }
};
template<typename stype>
const int ReduceOpMin<stype>::vlanes = 1;

#if (CV_SIMD || CV_SIMD_SCALABLE)

template<typename stype>
struct ReduceVecOpMin;

template<>
struct ReduceVecOpMin<uchar>
{
    using stype = uchar;
    using itype = uchar;
    using v_stype = v_uint8;
    using v_itype = v_uint8;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setall_u8(std::numeric_limits<itype>::max()); }
    static inline itype reduce(const v_itype &val) { return v_reduce_min(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const { return v_min(a, b); }
};
const int ReduceVecOpMin<uchar>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpMin<ushort>
{
    using stype = ushort;
    using itype = ushort;
    using v_stype = v_uint16;
    using v_itype = v_uint16;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setall_u16(std::numeric_limits<itype>::max()); }
    static inline itype reduce(const v_itype &val) { return v_reduce_min(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const { return v_min(a, b); }
};
const int ReduceVecOpMin<ushort>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpMin<short>
{
    using stype = short;
    using itype = short;
    using v_stype = v_int16;
    using v_itype = v_int16;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setall_s16(std::numeric_limits<itype>::max()); }
    static inline itype reduce(const v_itype &val) { return v_reduce_min(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const { return v_min(a, b); }
};
const int ReduceVecOpMin<short>::vlanes = VTraits<v_stype>::vlanes();

template<>
struct ReduceVecOpMin<float>
{
    using stype = float;
    using itype = float;
    using v_stype = v_float32;
    using v_itype = v_float32;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setall_f32(std::numeric_limits<itype>::max()); }
    static inline itype reduce(const v_itype &val) { return v_reduce_min(val); }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const { return v_min(a, b); }
};
const int ReduceVecOpMin<float>::vlanes = VTraits<v_stype>::vlanes();

#endif

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)

template<>
struct ReduceVecOpMin<double>
{
    using stype = double;
    using itype = double;
    using v_stype = v_float64;
    using v_itype = v_float64;
    static const int vlanes;
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load<stype, v_stype>(ptr, step); }
    static inline v_itype init() { return vx_setall_f64(std::numeric_limits<itype>::max()); }
    static inline itype reduce(const v_itype &val)
    {
        constexpr int nlanes = VTraits<v_itype>::max_nlanes;
        itype buf[nlanes];
        vx_store(buf, val);
        itype m = buf[0];
        for (int i = 1; i < VTraits<v_itype>::vlanes(); i++)
        {
            if (m > buf[i]) m = buf[i];
        }
        return m;
    }
    inline v_itype operator()(const v_itype &a, const v_stype &b) const { return v_min(a, b); }
};
const int ReduceVecOpMin<double>::vlanes = VTraits<v_stype>::vlanes();

#endif

using ReduceOpAdd_8U32S  = ReduceOpAdd<uchar, int>;
using ReduceOpAdd_8U32F  = ReduceOpAdd<uchar, int>;
using ReduceOpAdd_8U64F  = ReduceOpAdd<uchar, int>;
using ReduceOpAdd_16U32F = ReduceOpAdd<ushort, float>;
using ReduceOpAdd_16U64F = ReduceOpAdd<ushort, double>;
using ReduceOpAdd_16S32F = ReduceOpAdd<short, float>;
using ReduceOpAdd_16S64F = ReduceOpAdd<short, double>;
using ReduceOpAdd_32F32F = ReduceOpAdd<float, float>;
using ReduceOpAdd_32F64F = ReduceOpAdd<float, double>;
using ReduceOpAdd_64F64F = ReduceOpAdd<double, double>;

using ReduceOpAddSqr_8U32S  = ReduceOpAddSqr<uchar, int>;
using ReduceOpAddSqr_8U32F  = ReduceOpAddSqr<uchar, int>;
using ReduceOpAddSqr_8U64F  = ReduceOpAddSqr<uchar, int>;
using ReduceOpAddSqr_16U32F = ReduceOpAddSqr<ushort, float>;
using ReduceOpAddSqr_16U64F = ReduceOpAddSqr<ushort, double>;
using ReduceOpAddSqr_16S32F = ReduceOpAddSqr<short, float>;
using ReduceOpAddSqr_16S64F = ReduceOpAddSqr<short, double>;
using ReduceOpAddSqr_32F32F = ReduceOpAddSqr<float, float>;
using ReduceOpAddSqr_32F64F = ReduceOpAddSqr<float, double>;
using ReduceOpAddSqr_64F64F = ReduceOpAddSqr<double, double>;

using ReduceOpMax_8U  = ReduceOpMax<uchar>;
using ReduceOpMax_16U = ReduceOpMax<ushort>;
using ReduceOpMax_16S = ReduceOpMax<short>;
using ReduceOpMax_32F = ReduceOpMax<float>;
using ReduceOpMax_64F = ReduceOpMax<double>;

using ReduceOpMin_8U  = ReduceOpMin<uchar>;
using ReduceOpMin_16U = ReduceOpMin<ushort>;
using ReduceOpMin_16S = ReduceOpMin<short>;
using ReduceOpMin_32F = ReduceOpMin<float>;
using ReduceOpMin_64F = ReduceOpMin<double>;

#if (CV_SIMD || CV_SIMD_SCALABLE)

using ReduceVecOpAdd_8U32S  = ReduceVecOpAdd<uchar, int>;
using ReduceVecOpAdd_8U32F  = ReduceVecOpAdd<uchar, int>;
using ReduceVecOpAdd_8U64F  = ReduceVecOpAdd<uchar, int>;
using ReduceVecOpAdd_16U32F = ReduceVecOpAdd<ushort, float>;
using ReduceVecOpAdd_16S32F = ReduceVecOpAdd<short, float>;
using ReduceVecOpAdd_32F32F = ReduceVecOpAdd<float, float>;
using ReduceVecOpAddSqr_8U32S  = ReduceVecOpAddSqr<uchar, int>;
using ReduceVecOpAddSqr_8U32F  = ReduceVecOpAddSqr<uchar, int>;
using ReduceVecOpAddSqr_8U64F  = ReduceVecOpAddSqr<uchar, int>;
using ReduceVecOpAddSqr_16U32F = ReduceVecOpAddSqr<ushort, float>;
using ReduceVecOpAddSqr_16S32F = ReduceVecOpAddSqr<short, float>;
using ReduceVecOpAddSqr_32F32F = ReduceVecOpAddSqr<float, float>;
using ReduceVecOpMax_8U  = ReduceVecOpMax<uchar>;
using ReduceVecOpMax_16U = ReduceVecOpMax<ushort>;
using ReduceVecOpMax_16S = ReduceVecOpMax<short>;
using ReduceVecOpMax_32F = ReduceVecOpMax<float>;
using ReduceVecOpMin_8U  = ReduceVecOpMin<uchar>;
using ReduceVecOpMin_16U = ReduceVecOpMin<ushort>;
using ReduceVecOpMin_16S = ReduceVecOpMin<short>;
using ReduceVecOpMin_32F = ReduceVecOpMin<float>;

#else

using ReduceVecOpAdd_8U32S  = ReduceOpAdd<uchar, int>;
using ReduceVecOpAdd_8U32F  = ReduceOpAdd<uchar, int>;
using ReduceVecOpAdd_8U64F  = ReduceOpAdd<uchar, int>;
using ReduceVecOpAdd_16U32F = ReduceOpAdd<ushort, float>;
using ReduceVecOpAdd_16S32F = ReduceOpAdd<short, float>;
using ReduceVecOpAdd_32F32F = ReduceOpAdd<float, float>;
using ReduceVecOpAddSqr_8U32S  = ReduceOpAddSqr<uchar, int>;
using ReduceVecOpAddSqr_8U32F  = ReduceOpAddSqr<uchar, int>;
using ReduceVecOpAddSqr_8U64F  = ReduceOpAddSqr<uchar, int>;
using ReduceVecOpAddSqr_16U32F = ReduceOpAddSqr<ushort, float>;
using ReduceVecOpAddSqr_16S32F = ReduceOpAddSqr<short, float>;
using ReduceVecOpAddSqr_32F32F = ReduceOpAddSqr<float, float>;
using ReduceVecOpMax_8U  = ReduceOpMax<uchar>;
using ReduceVecOpMax_16U = ReduceOpMax<ushort>;
using ReduceVecOpMax_16S = ReduceOpMax<short>;
using ReduceVecOpMax_32F = ReduceOpMax<float>;
using ReduceVecOpMin_8U  = ReduceOpMin<uchar>;
using ReduceVecOpMin_16U = ReduceOpMin<ushort>;
using ReduceVecOpMin_16S = ReduceOpMin<short>;
using ReduceVecOpMin_32F = ReduceOpMin<float>;

#endif

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)

using ReduceVecOpAdd_16U64F = ReduceVecOpAdd<ushort, double>;
using ReduceVecOpAdd_16S64F = ReduceVecOpAdd<short, double>;
using ReduceVecOpAdd_32F64F = ReduceVecOpAdd<float, double>;
using ReduceVecOpAdd_64F64F = ReduceVecOpAdd<double, double>;
using ReduceVecOpAddSqr_16U64F = ReduceVecOpAddSqr<ushort, double>;
using ReduceVecOpAddSqr_16S64F = ReduceVecOpAddSqr<short, double>;
using ReduceVecOpAddSqr_32F64F = ReduceVecOpAddSqr<float, double>;
using ReduceVecOpAddSqr_64F64F = ReduceVecOpAddSqr<double, double>;
using ReduceVecOpMax_64F = ReduceVecOpMax<double>;
using ReduceVecOpMin_64F = ReduceVecOpMin<double>;

#else

using ReduceVecOpAdd_16U64F = ReduceOpAdd<ushort, double>;
using ReduceVecOpAdd_16S64F = ReduceOpAdd<short, double>;
using ReduceVecOpAdd_32F64F = ReduceOpAdd<float, double>;
using ReduceVecOpAdd_64F64F = ReduceOpAdd<double, double>;
using ReduceVecOpAddSqr_16U64F = ReduceOpAddSqr<ushort, double>;
using ReduceVecOpAddSqr_16S64F = ReduceOpAddSqr<short, double>;
using ReduceVecOpAddSqr_32F64F = ReduceOpAddSqr<float, double>;
using ReduceVecOpAddSqr_64F64F = ReduceOpAddSqr<double, double>;
using ReduceVecOpMax_64F = ReduceOpMax<double>;
using ReduceVecOpMin_64F = ReduceOpMin<double>;

#endif

template<typename T, typename ST, typename WT, class Op, class OpInit>
class ReduceR_Invoker : public ParallelLoopBody
{
public:
  ReduceR_Invoker(const Mat& aSrcmat, Mat& aDstmat, Op& aOp, OpInit& aOpInit)
                 :srcmat(aSrcmat),dstmat(aDstmat),op(aOp),opInit(aOpInit),buffer(srcmat.size().width*srcmat.channels())
  {
  }
  void operator()(const Range& range) const CV_OVERRIDE
  {
    const T* src = srcmat.ptr<T>();
    const size_t srcstep = srcmat.step/sizeof(src[0]);
    WT* buf = buffer.data();
    ST* dst = dstmat.ptr<ST>();
    int i = 0;

    for( i = range.start ; i < range.end; i++ )
        buf[i] = opInit(src[i]);

    int height = srcmat.size().height;
    for( ; --height; )
    {
        src += srcstep;
        i = range.start;
        #if CV_ENABLE_UNROLLED
        for(; i <= range.end - 4; i += 4 )
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
        for( ; i < range.end; i++ )
            buf[i] = op(buf[i], (WT)src[i]);
    }

    for( i = range.start ; i < range.end; i++ )
        dst[i] = (ST)buf[i];
  }
private:
  const Mat& srcmat;
  Mat& dstmat;
  Op& op;
  OpInit& opInit;
  mutable AutoBuffer<WT> buffer;
};

template<typename T, typename ST, class Op, class OpInit = OpNop<ST> > static void
reduceR_( const Mat& srcmat, Mat& dstmat)
{
    typedef typename Op::rtype WT;
    Op op;
    OpInit opInit;

    ReduceR_Invoker<T, ST, WT, Op, OpInit> body(srcmat, dstmat, op, opInit);
    //group columns by 64 bytes for data locality
    parallel_for_(Range(0, srcmat.size().width*srcmat.channels()), body, srcmat.size().width*CV_ELEM_SIZE(srcmat.depth())/64.0);
}

template<typename T, typename ST, class Op, class VecOp>
class ReduceC_Invoker : public ParallelLoopBody
{
  using WT = typename Op::v_itype;
  using VT = typename VecOp::v_itype;
public:
  ReduceC_Invoker(const Mat& aSrcmat, Mat& aDstmat, Op& aOp, VecOp& aVop)
                 :srcmat(aSrcmat),dstmat(aDstmat),op(aOp),vop(aVop)
  {
  }
  void operator()(const Range& range) const CV_OVERRIDE
  {
    int channels = srcmat.channels();
    int width = srcmat.cols;

    const int nlanes = VecOp::vlanes;

    for (int cn = 0; cn < channels; cn++)
    {
        for (int h = range.start; h < range.end; h++)
        {
            const T *src = srcmat.ptr<T>(h)+cn;
            ST *dst = dstmat.ptr<ST>(h);
            VT vbuf = vop.init();
            int w = 0;
            for (; w <= width - nlanes; w += nlanes)
            {
                vbuf = vop(vbuf, vop.load(src+w*channels, channels));
            }
            WT wbuf = vop.reduce(vbuf);
            for (; w < width; w++)
            {
                wbuf = op(wbuf, op.load(src+w*channels, channels));
            }
            dst[cn] = (ST)op.reduce(wbuf);
        }
    }
  }
private:
  const Mat& srcmat;
  Mat& dstmat;
  Op& op;
  VecOp& vop;
};

template<typename T, typename ST, class Op, class VecOp> static void
reduceC_( const Mat& srcmat, Mat& dstmat)
{
    Op op;
    VecOp vop;

    ReduceC_Invoker<T, ST, Op, VecOp> body(srcmat, dstmat, op, vop);
    parallel_for_(Range(0, srcmat.size().height), body);
}

typedef void (*ReduceFunc)( const Mat& src, Mat& dst );

}

#define reduceSumR8u32s  reduceR_<uchar, int,   OpAdd<int>, OpNop<int> >
#define reduceSumR8u32f  reduceR_<uchar, float, OpAdd<int>, OpNop<int> >
#define reduceSumR8u64f  reduceR_<uchar, double,OpAdd<int>, OpNop<int> >
#define reduceSumR16u32f reduceR_<ushort,float, OpAdd<float> >
#define reduceSumR16u64f reduceR_<ushort,double,OpAdd<double> >
#define reduceSumR16s32f reduceR_<short, float, OpAdd<float> >
#define reduceSumR16s64f reduceR_<short, double,OpAdd<double> >
#define reduceSumR32f32f reduceR_<float, float, OpAdd<float> >
#define reduceSumR32f64f reduceR_<float, double,OpAdd<double> >
#define reduceSumR64f64f reduceR_<double,double,OpAdd<double> >

#define reduceSum2R8u32s  reduceR_<uchar, int,   OpAddSqr<int>,   OpSqr<int> >
#define reduceSum2R8u32f  reduceR_<uchar, float, OpAddSqr<int>,   OpSqr<int> >
#define reduceSum2R8u64f  reduceR_<uchar, double,OpAddSqr<int>,   OpSqr<int> >
#define reduceSum2R16u32f reduceR_<ushort,float, OpAddSqr<float>, OpSqr<float> >
#define reduceSum2R16u64f reduceR_<ushort,double,OpAddSqr<double>,OpSqr<double> >
#define reduceSum2R16s32f reduceR_<short, float, OpAddSqr<float>, OpSqr<float> >
#define reduceSum2R16s64f reduceR_<short, double,OpAddSqr<double>,OpSqr<double> >
#define reduceSum2R32f32f reduceR_<float, float, OpAddSqr<float>, OpSqr<float> >
#define reduceSum2R32f64f reduceR_<float, double,OpAddSqr<double>,OpSqr<double> >
#define reduceSum2R64f64f reduceR_<double,double,OpAddSqr<double>,OpSqr<double> >

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

#define reduceSumC8u32s  reduceC_<uchar, int,   ReduceOpAdd_8U32S,  ReduceVecOpAdd_8U32S  >
#define reduceSumC8u32f  reduceC_<uchar, float, ReduceOpAdd_8U32F,  ReduceVecOpAdd_8U32F  >
#define reduceSumC16u32f reduceC_<ushort,float, ReduceOpAdd_16U32F, ReduceVecOpAdd_16U32F >
#define reduceSumC16s32f reduceC_<short, float, ReduceOpAdd_16S32F, ReduceVecOpAdd_16S32F >
#define reduceSumC32f32f reduceC_<float, float, ReduceOpAdd_32F32F, ReduceVecOpAdd_32F32F >
#define reduceSumC64f64f reduceC_<double,double,ReduceOpAdd_64F64F, ReduceVecOpAdd_64F64F >

#define reduceSum2C8u32s  reduceC_<uchar, int,   ReduceOpAddSqr_8U32S,  ReduceVecOpAddSqr_8U32S  >
#define reduceSum2C8u32f  reduceC_<uchar, float, ReduceOpAddSqr_8U32F,  ReduceVecOpAddSqr_8U32F  >
#define reduceSum2C16u32f reduceC_<ushort,float, ReduceOpAddSqr_16U32F, ReduceVecOpAddSqr_16U32F >
#define reduceSum2C16s32f reduceC_<short, float, ReduceOpAddSqr_16S32F, ReduceVecOpAddSqr_16S32F >
#define reduceSum2C32f32f reduceC_<float, float, ReduceOpAddSqr_32F32F, ReduceVecOpAddSqr_32F32F >
#define reduceSum2C64f64f reduceC_<double,double,ReduceOpAddSqr_64F64F, ReduceVecOpAddSqr_64F64F >

#ifdef HAVE_IPP
#define reduceSumC8u64f  reduceSumC_8u16u16s32f_64f
#define reduceSumC16u64f reduceSumC_8u16u16s32f_64f
#define reduceSumC16s64f reduceSumC_8u16u16s32f_64f
#define reduceSumC32f64f reduceSumC_8u16u16s32f_64f
#else
#define reduceSumC8u64f  reduceC_<uchar, double,ReduceOpAdd_8U64F,  ReduceVecOpAdd_8U64F  >
#define reduceSumC16u64f reduceC_<ushort,double,ReduceOpAdd_16U64F, ReduceVecOpAdd_16U64F >
#define reduceSumC16s64f reduceC_<short, double,ReduceOpAdd_16S64F, ReduceVecOpAdd_16S64F >
#define reduceSumC32f64f reduceC_<float, double,ReduceOpAdd_32F64F, ReduceVecOpAdd_32F64F >

#define reduceSum2C8u64f  reduceC_<uchar, double,ReduceOpAddSqr_8U64F,  ReduceVecOpAddSqr_8U64F  >
#define reduceSum2C16u64f reduceC_<ushort,double,ReduceOpAddSqr_16U64F, ReduceVecOpAddSqr_16U64F >
#define reduceSum2C16s64f reduceC_<short, double,ReduceOpAddSqr_16S64F, ReduceVecOpAddSqr_16S64F >
#define reduceSum2C32f64f reduceC_<float, double,ReduceOpAddSqr_32F64F, ReduceVecOpAddSqr_32F64F >
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
#define reduceMaxC8u  reduceC_<uchar, uchar, ReduceOpMax_8U,  ReduceVecOpMax_8U  >
#define reduceMaxC16u reduceC_<ushort,ushort,ReduceOpMax_16U, ReduceVecOpMax_16U >
#define reduceMaxC16s reduceC_<short, short, ReduceOpMax_16S, ReduceVecOpMax_16S >
#define reduceMaxC32f reduceC_<float, float, ReduceOpMax_32F, ReduceVecOpMax_32F >
#endif
#define reduceMaxC64f reduceC_<double,double,ReduceOpMax_64F, ReduceVecOpMax_64F >

#ifdef HAVE_IPP
REDUCE_OP(8u, Min, uchar, uchar)
REDUCE_OP(16u, Min, ushort, ushort)
REDUCE_OP(16s, Min, short, short)
REDUCE_OP(32f, Min, float, float)
#else
#define reduceMinC8u  reduceC_<uchar, uchar, ReduceOpMin_8U,  ReduceVecOpMin_8U  >
#define reduceMinC16u reduceC_<ushort,ushort,ReduceOpMin_16U, ReduceVecOpMin_16U >
#define reduceMinC16s reduceC_<short, short, ReduceOpMin_16S, ReduceVecOpMin_16S >
#define reduceMinC32f reduceC_<float, float, ReduceOpMin_32F, ReduceVecOpMin_32F >
#endif
#define reduceMinC64f reduceC_<double,double,ReduceOpMin_64F, ReduceVecOpMin_64F >

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

    if (op == REDUCE_AVG)
    {
        if (sdepth < CV_32S && ddepth < CV_32S)
            ddepth = CV_32S;
    }

    const char * const ops[5] = { "OCL_CV_REDUCE_SUM", "OCL_CV_REDUCE_AVG",
                                  "OCL_CV_REDUCE_MAX", "OCL_CV_REDUCE_MIN",
                                  "OCL_CV_REDUCE_SUM2"};
    int wdepth = std::max(ddepth, CV_32F);
    if (useOptimized)
    {
        size_t tileHeight = (size_t)(wgs / buf_cols);
        if (defDev.isIntel())
        {
            static const size_t maxItemInGroupCount = 16;
            tileHeight = min(tileHeight, defDev.localMemSize() / buf_cols / CV_ELEM_SIZE(CV_MAKETYPE(wdepth, cn)) / maxItemInGroupCount);
        }
        char cvt[3][50];
        cv::String build_opt = format("-D OP_REDUCE_PRE -D BUF_COLS=%d -D TILE_HEIGHT=%zu -D %s -D dim=1"
                                            " -D cn=%d -D ddepth=%d"
                                            " -D srcT=%s -D bufT=%s -D dstT=%s"
                                            " -D convertToWT=%s -D convertToBufT=%s -D convertToDT=%s%s",
                                            buf_cols, tileHeight, ops[op], cn, ddepth,
                                            ocl::typeToStr(sdepth),
                                            ocl::typeToStr(ddepth),
                                            ocl::typeToStr(ddepth0),
                                            ocl::convertTypeStr(ddepth, wdepth, 1, cvt[0], sizeof(cvt[0])),
                                            ocl::convertTypeStr(sdepth, ddepth, 1, cvt[1], sizeof(cvt[1])),
                                            ocl::convertTypeStr(wdepth, ddepth0, 1, cvt[2], sizeof(cvt[2])),
                                            doubleSupport ? " -D DOUBLE_SUPPORT" : "");
        ocl::Kernel k("reduce_horz_opt", ocl::core::reduce2_oclsrc, build_opt);
        if (k.empty())
            return false;
        UMat src = _src.getUMat();
        Size dsize(1, src.rows);
        _dst.create(dsize, dtype);
        UMat dst = _dst.getUMat();

        if (op0 == REDUCE_AVG)
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
        char cvt[2][50];
        cv::String build_opt = format("-D %s -D dim=%d -D cn=%d -D ddepth=%d"
                                      " -D srcT=%s -D dstT=%s -D dstT0=%s -D convertToWT=%s"
                                      " -D convertToDT=%s -D convertToDT0=%s%s",
                                      ops[op], dim, cn, ddepth, ocl::typeToStr(useOptimized ? ddepth : sdepth),
                                      ocl::typeToStr(ddepth), ocl::typeToStr(ddepth0),
                                      ocl::convertTypeStr(ddepth, wdepth, 1, cvt[0], sizeof(cvt[0])),
                                      ocl::convertTypeStr(sdepth, ddepth, 1, cvt[0], sizeof(cvt[0])),
                                      ocl::convertTypeStr(wdepth, ddepth0, 1, cvt[1], sizeof(cvt[1])),
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

        if (op0 == REDUCE_AVG)
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
    CV_Assert( op == REDUCE_SUM || op == REDUCE_MAX ||
               op == REDUCE_MIN || op == REDUCE_AVG ||
               op == REDUCE_SUM2);

    CV_OCL_RUN(_dst.isUMat(),
               ocl_reduce(_src, _dst, dim, op, op0, stype, dtype))

    // Fake reference to source. Resolves issue 8693 in case of src == dst.
    UMat srcUMat;
    if (_src.isUMat())
        srcUMat = _src.getUMat();

    Mat src = _src.getMat();
    _dst.create(dim == 0 ? 1 : src.rows, dim == 0 ? src.cols : 1, dtype);
    Mat dst = _dst.getMat(), temp = dst;

    if( op == REDUCE_AVG )
    {
        op = REDUCE_SUM;
        if( sdepth < CV_32S && ddepth < CV_32S )
        {
            temp.create(dst.rows, dst.cols, CV_32SC(cn));
            ddepth = CV_32S;
        }
    }

    ReduceFunc func = 0;
    if( dim == 0 )
    {
        if( op == REDUCE_SUM )
        {
            if(sdepth == CV_8U && ddepth == CV_32S)
                func = reduceSumR8u32s;
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
                func = reduceSumR32f32f;
            else if(sdepth == CV_32F && ddepth == CV_64F)
                func = reduceSumR32f64f;
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceSumR64f64f;
        }
        else if(op == REDUCE_MAX)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = reduceMaxR8u;
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMaxR16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMaxR16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = reduceMaxR32f;
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMaxR64f;
        }
        else if(op == REDUCE_MIN)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = reduceMinR8u;
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMinR16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMinR16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = reduceMinR32f;
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMinR64f;
        }
        else if( op == REDUCE_SUM2 )
        {
            if(sdepth == CV_8U && ddepth == CV_32S)
                func = reduceSum2R8u32s;
            else if(sdepth == CV_8U && ddepth == CV_32F)
                func = reduceSum2R8u32f;
            else if(sdepth == CV_8U && ddepth == CV_64F)
                func = reduceSum2R8u64f;
            else if(sdepth == CV_16U && ddepth == CV_32F)
                func = reduceSum2R16u32f;
            else if(sdepth == CV_16U && ddepth == CV_64F)
                func = reduceSum2R16u64f;
            else if(sdepth == CV_16S && ddepth == CV_32F)
                func = reduceSum2R16s32f;
            else if(sdepth == CV_16S && ddepth == CV_64F)
                func = reduceSum2R16s64f;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = reduceSum2R32f32f;
            else if(sdepth == CV_32F && ddepth == CV_64F)
                func = reduceSum2R32f64f;
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceSum2R64f64f;
        }
    }
    else
    {
        if(op == REDUCE_SUM)
        {
            if(sdepth == CV_8U && ddepth == CV_32S)
                func = reduceSumC8u32s;
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
                func = reduceSumC32f32f;
            else if(sdepth == CV_32F && ddepth == CV_64F)
                func = reduceSumC32f64f;
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceSumC64f64f;
        }
        else if(op == REDUCE_MAX)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = reduceMaxC8u;
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMaxC16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMaxC16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = reduceMaxC32f;
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMaxC64f;
        }
        else if(op == REDUCE_MIN)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = reduceMinC8u;
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMinC16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMinC16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = reduceMinC32f;
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMinC64f;
        }
        else if(op == REDUCE_SUM2)
        {
            if(sdepth == CV_8U && ddepth == CV_32S)
                func = reduceSum2C8u32s;
            else if(sdepth == CV_8U && ddepth == CV_32F)
                func = reduceSum2C8u32f;
            else if(sdepth == CV_8U && ddepth == CV_64F)
                func = reduceSum2C8u64f;
            else if(sdepth == CV_16U && ddepth == CV_32F)
                func = reduceSum2C16u32f;
            else if(sdepth == CV_16U && ddepth == CV_64F)
                func = reduceSum2C16u64f;
            else if(sdepth == CV_16S && ddepth == CV_32F)
                func = reduceSum2C16s32f;
            else if(sdepth == CV_16S && ddepth == CV_64F)
                func = reduceSum2C16s64f;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = reduceSum2C32f32f;
            else if(sdepth == CV_32F && ddepth == CV_64F)
                func = reduceSum2C32f64f;
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceSum2C64f64f;
        }
    }

    if( !func )
        CV_Error( cv::Error::StsUnsupportedFormat,
                  "Unsupported combination of input and output array formats" );

    func( src, temp );

    if( op0 == REDUCE_AVG )
        temp.convertTo(dst, dst.type(), 1./(dim == 0 ? src.rows : src.cols));
}


//////////////////////////////////////// sort ///////////////////////////////////////////

namespace cv
{

template<typename T> static void sort_( const Mat& src, Mat& dst, int flags )
{
    AutoBuffer<T> buf;
    int n, len;
    bool sortRows = (flags & 1) == SORT_EVERY_ROW;
    bool inplace = src.data == dst.data;
    bool sortDescending = (flags & SORT_DESCENDING) != 0;

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

    bool        sortRows        = (flags & 1) == SORT_EVERY_ROW;
    bool        sortDescending  = (flags & SORT_DESCENDING) != 0;
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
    bool sortRows = (flags & 1) == SORT_EVERY_ROW;
    bool sortDescending = (flags & SORT_DESCENDING) != 0;

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

    static SortFunc tab[CV_DEPTH_MAX] =
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

    static SortFunc tab[CV_DEPTH_MAX] =
    {
        sortIdx_<uchar>, sortIdx_<schar>, sortIdx_<ushort>, sortIdx_<short>,
        sortIdx_<int>, sortIdx_<float>, sortIdx_<double>, 0
    };
    SortFunc func = tab[src.depth()];
    CV_Assert( func != 0 );
    func( src, dst, flags );
}
