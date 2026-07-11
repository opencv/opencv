// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#if (CV_SIMD || CV_SIMD_SCALABLE)

template<typename T, typename VT>
static inline VT vx_load_strided(const T *ptr, size_t step)
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
inline v_uint8 vx_load_strided(const uchar *ptr, size_t step)
{
    return __riscv_vlse8_v_u8m2(ptr, step * sizeof(uchar), __riscv_vsetvlmax_e8m2());
}
template<>
inline v_uint16 vx_load_strided(const ushort *ptr, size_t step)
{
    return __riscv_vlse16_v_u16m2(ptr, step * sizeof(ushort), __riscv_vsetvlmax_e16m2());
}
template<>
inline v_int16 vx_load_strided(const short *ptr, size_t step)
{
    return __riscv_vlse16_v_i16m2(ptr, step * sizeof(short), __riscv_vsetvlmax_e16m2());
}
template<>
inline v_int32 vx_load_strided(const int *ptr, size_t step)
{
    return __riscv_vlse32_v_i32m2(ptr, step * sizeof(int), __riscv_vsetvlmax_e32m2());
}
template<>
inline v_float32 vx_load_strided(const float *ptr, size_t step)
{
    return __riscv_vlse32_v_f32m2(ptr, step * sizeof(float), __riscv_vsetvlmax_e32m2());
}
template<>
inline v_float64 vx_load_strided(const double *ptr, size_t step)
{
    return __riscv_vlse64_v_f64m2(ptr, step * sizeof(double), __riscv_vsetvlmax_e64m2());
}
#endif
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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
    static inline v_stype load(const stype *ptr, size_t step) { return vx_load_strided<stype, v_stype>(ptr, step); }
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

using ReduceVecOpAddSqr_16U64F = ReduceVecOpAddSqr<ushort, double>;
using ReduceVecOpAddSqr_16S64F = ReduceVecOpAddSqr<short, double>;
using ReduceVecOpAddSqr_32F64F = ReduceVecOpAddSqr<float, double>;
using ReduceVecOpAddSqr_64F64F = ReduceVecOpAddSqr<double, double>;
using ReduceVecOpMax_64F = ReduceVecOpMax<double>;
using ReduceVecOpMin_64F = ReduceVecOpMin<double>;

#else

using ReduceVecOpAddSqr_16U64F = ReduceOpAddSqr<ushort, double>;
using ReduceVecOpAddSqr_16S64F = ReduceOpAddSqr<short, double>;
using ReduceVecOpAddSqr_32F64F = ReduceOpAddSqr<float, double>;
using ReduceVecOpAddSqr_64F64F = ReduceOpAddSqr<double, double>;
using ReduceVecOpMax_64F = ReduceOpMax<double>;
using ReduceVecOpMin_64F = ReduceOpMin<double>;

#endif

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

    for (int h = range.start; h < range.end; h++)
    {
        const T *srcrow = srcmat.ptr<T>(h);
        ST *dst = dstmat.ptr<ST>(h);
        for (int cn = 0; cn < channels; cn++)
        {
            const T *src = srcrow + cn;
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

template<bool isMax>
static inline uchar reduceScalarMinMax(uchar a, uchar b)
{
    return isMax ? std::max(a, b) : std::min(a, b);
}

template<bool isMax>
static void reduceColMinMax_8uFallback(const Mat& srcmat, Mat& dstmat)
{
    if (isMax)
        reduceC_<uchar, uchar, ReduceOpMax_8U, ReduceVecOpMax_8U>(srcmat, dstmat);
    else
        reduceC_<uchar, uchar, ReduceOpMin_8U, ReduceVecOpMin_8U>(srcmat, dstmat);
}

template<bool isMax>
static void reduceColMinMax_8uC1(const Mat& srcmat, Mat& dstmat)
{
#if CV_NEON || CV_AVX2 || CV_RVV
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            uchar result = isMax ? 0 : UCHAR_MAX;
            int x = 0;
#if CV_NEON
            uint8x16_t acc = vdupq_n_u8(result);
            for (; x <= cols - 16; x += 16)
            {
                uint8x16_t v = vld1q_u8(src + x);
                acc = isMax ? vmaxq_u8(acc, v) : vminq_u8(acc, v);
            }
            uchar lanes[16];
            vst1q_u8(lanes, acc);
            for (int i = 0; i < 16; i++)
                result = reduceScalarMinMax<isMax>(result, lanes[i]);
#elif CV_AVX2
            __m256i acc = _mm256_set1_epi8((char)result);
            for (; x <= cols - 32; x += 32)
            {
                __m256i v = _mm256_loadu_si256((const __m256i*)(src + x));
                acc = isMax ? _mm256_max_epu8(acc, v) : _mm256_min_epu8(acc, v);
            }
            uchar lanes[32];
            _mm256_storeu_si256((__m256i*)lanes, acc);
            for (int i = 0; i < 32; i++)
                result = reduceScalarMinMax<isMax>(result, lanes[i]);
#elif CV_RVV
            const int vlmax = __riscv_vsetvlmax_e8m8();
            vuint8m8_t acc = __riscv_vmv_v_x_u8m8(result, vlmax);
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e8m8(cols - x);
                vuint8m8_t v = __riscv_vle8_v_u8m8(src + x, vl);
                acc = isMax ? __riscv_vmaxu_tu(acc, acc, v, vl)
                            : __riscv_vminu_tu(acc, acc, v, vl);
                x += vl;
            }
            vuint8m1_t seed = __riscv_vmv_s_x_u8m1(result, __riscv_vsetvlmax_e8m1());
            vuint8m1_t reduced = isMax ? __riscv_vredmaxu(acc, seed, vlmax)
                                       : __riscv_vredminu(acc, seed, vlmax);
            result = (uchar)__riscv_vmv_x(reduced);
#endif
            for (; x < cols; x++)
                result = reduceScalarMinMax<isMax>(result, src[x]);
            dst[0] = result;
        }
    });
    v_cleanup();
#else
    reduceColMinMax_8uFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_8uC3(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    const int cols = srcmat.cols;
    const uchar initial = isMax ? 0 : UCHAR_MAX;
    const int vlmax = __riscv_vsetvlmax_e8m1();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            vuint8m1_t acc0 = __riscv_vmv_v_x_u8m1(initial, vlmax);
            vuint8m1_t acc1 = acc0, acc2 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e8m1(cols - x);
                vuint8m1x3_t v = __riscv_vlseg3e8_v_u8m1x3(src + x * 3, vl);
                vuint8m1_t v0 = __riscv_vget_v_u8m1x3_u8m1(v, 0);
                vuint8m1_t v1 = __riscv_vget_v_u8m1x3_u8m1(v, 1);
                vuint8m1_t v2 = __riscv_vget_v_u8m1x3_u8m1(v, 2);
                acc0 = isMax ? __riscv_vmaxu_tu(acc0, acc0, v0, vl)
                             : __riscv_vminu_tu(acc0, acc0, v0, vl);
                acc1 = isMax ? __riscv_vmaxu_tu(acc1, acc1, v1, vl)
                             : __riscv_vminu_tu(acc1, acc1, v1, vl);
                acc2 = isMax ? __riscv_vmaxu_tu(acc2, acc2, v2, vl)
                             : __riscv_vminu_tu(acc2, acc2, v2, vl);
                x += vl;
            }
            vuint8m1_t seed = __riscv_vmv_s_x_u8m1(initial, vlmax);
            dst[0] = (uchar)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc0, seed, vlmax)
                                                 : __riscv_vredminu(acc0, seed, vlmax));
            dst[1] = (uchar)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc1, seed, vlmax)
                                                 : __riscv_vredminu(acc1, seed, vlmax));
            dst[2] = (uchar)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc2, seed, vlmax)
                                                 : __riscv_vredminu(acc2, seed, vlmax));
        }
    });
    v_cleanup();
#elif CV_NEON
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            const uchar initial = isMax ? 0 : UCHAR_MAX;
            uint8x16x3_t acc = {{
                vdupq_n_u8(initial), vdupq_n_u8(initial), vdupq_n_u8(initial)
            }};
            int x = 0;
            for (; x <= cols - 16; x += 16)
            {
                uint8x16x3_t v = vld3q_u8(src + x * 3);
                for (int c = 0; c < 3; c++)
                    acc.val[c] = isMax ? vmaxq_u8(acc.val[c], v.val[c])
                                       : vminq_u8(acc.val[c], v.val[c]);
            }
            for (int c = 0; c < 3; c++)
            {
                uchar lanes[16];
                vst1q_u8(lanes, acc.val[c]);
                uchar result = initial;
                for (int i = 0; i < 16; i++)
                    result = reduceScalarMinMax<isMax>(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = reduceScalarMinMax<isMax>(result, src[i * 3 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
#elif CV_AVX2
    const int cols = srcmat.cols;
    const uchar initial = isMax ? 0 : UCHAR_MAX;
    const __m256i accInit = _mm256_set1_epi8((char)initial);
    const __m256i mask0 = _mm256_setr_epi8(
            0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            2, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask1 = _mm256_setr_epi8(
            1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 3, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask2 = _mm256_setr_epi8(
            2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i invalid0 = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i invalid12 = _mm256_setr_epi8(
            0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            __m256i acc0 = accInit, acc1 = accInit, acc2 = accInit;
            int x = 0;
            for (; x <= cols - 11; x += 8)
            {
                __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 3));
                __m256i v0 = _mm256_shuffle_epi8(v, mask0);
                __m256i v1 = _mm256_shuffle_epi8(v, mask1);
                __m256i v2 = _mm256_shuffle_epi8(v, mask2);
                if (!isMax)
                {
                    v0 = _mm256_or_si256(v0, invalid0);
                    v1 = _mm256_or_si256(v1, invalid12);
                    v2 = _mm256_or_si256(v2, invalid12);
                }
                acc0 = isMax ? _mm256_max_epu8(acc0, v0) : _mm256_min_epu8(acc0, v0);
                acc1 = isMax ? _mm256_max_epu8(acc1, v1) : _mm256_min_epu8(acc1, v1);
                acc2 = isMax ? _mm256_max_epu8(acc2, v2) : _mm256_min_epu8(acc2, v2);
            }

            __m256i accs[3] = {acc0, acc1, acc2};
            for (int c = 0; c < 3; c++)
            {
                uchar lanes[32];
                _mm256_storeu_si256((__m256i*)lanes, accs[c]);
                uchar result = initial;
                for (int i = 0; i < 32; i++)
                    result = reduceScalarMinMax<isMax>(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = reduceScalarMinMax<isMax>(result, src[i * 3 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
#else
    reduceColMinMax_8uFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_8uC4(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    const int cols = srcmat.cols;
    const uchar initial = isMax ? 0 : UCHAR_MAX;
    const int vlmax = __riscv_vsetvlmax_e8m1();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            vuint8m1_t acc0 = __riscv_vmv_v_x_u8m1(initial, vlmax);
            vuint8m1_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e8m1(cols - x);
                vuint8m1x4_t v = __riscv_vlseg4e8_v_u8m1x4(src + x * 4, vl);
                vuint8m1_t v0 = __riscv_vget_v_u8m1x4_u8m1(v, 0);
                vuint8m1_t v1 = __riscv_vget_v_u8m1x4_u8m1(v, 1);
                vuint8m1_t v2 = __riscv_vget_v_u8m1x4_u8m1(v, 2);
                vuint8m1_t v3 = __riscv_vget_v_u8m1x4_u8m1(v, 3);
                acc0 = isMax ? __riscv_vmaxu_tu(acc0, acc0, v0, vl)
                             : __riscv_vminu_tu(acc0, acc0, v0, vl);
                acc1 = isMax ? __riscv_vmaxu_tu(acc1, acc1, v1, vl)
                             : __riscv_vminu_tu(acc1, acc1, v1, vl);
                acc2 = isMax ? __riscv_vmaxu_tu(acc2, acc2, v2, vl)
                             : __riscv_vminu_tu(acc2, acc2, v2, vl);
                acc3 = isMax ? __riscv_vmaxu_tu(acc3, acc3, v3, vl)
                             : __riscv_vminu_tu(acc3, acc3, v3, vl);
                x += vl;
            }
            vuint8m1_t seed = __riscv_vmv_s_x_u8m1(initial, vlmax);
            dst[0] = (uchar)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc0, seed, vlmax)
                                                 : __riscv_vredminu(acc0, seed, vlmax));
            dst[1] = (uchar)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc1, seed, vlmax)
                                                 : __riscv_vredminu(acc1, seed, vlmax));
            dst[2] = (uchar)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc2, seed, vlmax)
                                                 : __riscv_vredminu(acc2, seed, vlmax));
            dst[3] = (uchar)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc3, seed, vlmax)
                                                 : __riscv_vredminu(acc3, seed, vlmax));
        }
    });
    v_cleanup();
#elif CV_NEON
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            const uchar initial = isMax ? 0 : UCHAR_MAX;
            uint8x16x4_t acc = {{
                vdupq_n_u8(initial), vdupq_n_u8(initial),
                vdupq_n_u8(initial), vdupq_n_u8(initial)
            }};
            int x = 0;
            for (; x <= cols - 16; x += 16)
            {
                uint8x16x4_t v = vld4q_u8(src + x * 4);
                for (int c = 0; c < 4; c++)
                    acc.val[c] = isMax ? vmaxq_u8(acc.val[c], v.val[c])
                                       : vminq_u8(acc.val[c], v.val[c]);
            }
            for (int c = 0; c < 4; c++)
            {
                uchar lanes[16];
                vst1q_u8(lanes, acc.val[c]);
                uchar result = initial;
                for (int i = 0; i < 16; i++)
                    result = reduceScalarMinMax<isMax>(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = reduceScalarMinMax<isMax>(result, src[i * 4 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
#elif CV_AVX2
    const int cols = srcmat.cols;
    const uchar initial = isMax ? 0 : UCHAR_MAX;
    const __m256i accInit = _mm256_set1_epi8((char)initial);
    const __m256i maskInvalid = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(0, 0, 0, 0, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i mask0 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i mask1 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(1, 5, 9, 13, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i mask2 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(2, 6, 10, 14, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i mask3 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            __m256i acc0 = accInit, acc1 = accInit, acc2 = accInit, acc3 = accInit;
            int x = 0;
            for (; x <= cols - 8; x += 8)
            {
                __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 4));
                __m256i v0 = _mm256_shuffle_epi8(v, mask0);
                __m256i v1 = _mm256_shuffle_epi8(v, mask1);
                __m256i v2 = _mm256_shuffle_epi8(v, mask2);
                __m256i v3 = _mm256_shuffle_epi8(v, mask3);
                if (!isMax)
                {
                    v0 = _mm256_or_si256(v0, maskInvalid);
                    v1 = _mm256_or_si256(v1, maskInvalid);
                    v2 = _mm256_or_si256(v2, maskInvalid);
                    v3 = _mm256_or_si256(v3, maskInvalid);
                }
                acc0 = isMax ? _mm256_max_epu8(acc0, v0) : _mm256_min_epu8(acc0, v0);
                acc1 = isMax ? _mm256_max_epu8(acc1, v1) : _mm256_min_epu8(acc1, v1);
                acc2 = isMax ? _mm256_max_epu8(acc2, v2) : _mm256_min_epu8(acc2, v2);
                acc3 = isMax ? _mm256_max_epu8(acc3, v3) : _mm256_min_epu8(acc3, v3);
            }

            __m256i accs[4] = {acc0, acc1, acc2, acc3};
            for (int c = 0; c < 4; c++)
            {
                uchar lanes[32];
                _mm256_storeu_si256((__m256i*)lanes, accs[c]);
                uchar result = initial;
                for (int i = 0; i < 32; i++)
                    result = reduceScalarMinMax<isMax>(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = reduceScalarMinMax<isMax>(result, src[i * 4 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
#else
    reduceColMinMax_8uFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_8u(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    if (cn == 1)
        reduceColMinMax_8uC1<isMax>(srcmat, dstmat);
    else if (cn == 3)
        reduceColMinMax_8uC3<isMax>(srcmat, dstmat);
    else if (cn == 4)
        reduceColMinMax_8uC4<isMax>(srcmat, dstmat);
    else
        reduceColMinMax_8uFallback<isMax>(srcmat, dstmat);
}

static void reduceColMax_8u(const Mat& srcmat, Mat& dstmat)
{
    reduceColMinMax_8u<true>(srcmat, dstmat);
}

static void reduceColMin_8u(const Mat& srcmat, Mat& dstmat)
{
    reduceColMinMax_8u<false>(srcmat, dstmat);
}

template<bool isMax>
static inline float reduceScalarMinMax(float a, float b)
{
    return isMax ? std::max(a, b) : std::min(a, b);
}

template<bool isMax>
static void reduceColMinMax_32fFallback(const Mat& srcmat, Mat& dstmat)
{
    if (isMax)
        reduceC_<float, float, ReduceOpMax_32F, ReduceVecOpMax_32F>(srcmat, dstmat);
    else
        reduceC_<float, float, ReduceOpMin_32F, ReduceVecOpMin_32F>(srcmat, dstmat);
}

template<bool isMax>
static void reduceColMinMax_32fC1(const Mat& srcmat, Mat& dstmat)
{
#if CV_NEON || CV_AVX2 || CV_RVV
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float result = src[0];
            int x = 0;
#if CV_NEON
            float32x4_t acc = vdupq_n_f32(result);
            for (; x <= cols - 4; x += 4)
            {
                float32x4_t v = vld1q_f32(src + x);
                acc = isMax ? vmaxq_f32(acc, v) : vminq_f32(acc, v);
            }
            float lanes[4];
            vst1q_f32(lanes, acc);
            for (int i = 0; i < 4; i++)
                result = reduceScalarMinMax<isMax>(result, lanes[i]);
#elif CV_AVX2
            __m256 acc = _mm256_set1_ps(result);
            for (; x <= cols - 8; x += 8)
            {
                __m256 v = _mm256_loadu_ps(src + x);
                acc = isMax ? _mm256_max_ps(acc, v) : _mm256_min_ps(acc, v);
            }
            float lanes[8];
            _mm256_storeu_ps(lanes, acc);
            for (int i = 0; i < 8; i++)
                result = reduceScalarMinMax<isMax>(result, lanes[i]);
#elif CV_RVV
            const int vlmax = __riscv_vsetvlmax_e32m8();
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(result, vlmax);
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e32m8(cols - x);
                vfloat32m8_t v = __riscv_vle32_v_f32m8(src + x, vl);
                acc = isMax ? __riscv_vfmax_tu(acc, acc, v, vl)
                            : __riscv_vfmin_tu(acc, acc, v, vl);
                x += vl;
            }
            vfloat32m1_t seed = __riscv_vfmv_s_f_f32m1(result, __riscv_vsetvlmax_e32m1());
            vfloat32m1_t reduced = isMax ? __riscv_vfredmax(acc, seed, vlmax)
                                         : __riscv_vfredmin(acc, seed, vlmax);
            result = __riscv_vfmv_f(reduced);
#endif
            for (; x < cols; x++)
                result = reduceScalarMinMax<isMax>(result, src[x]);
            dstmat.ptr<float>(y)[0] = result;
        }
    });
    v_cleanup();
#else
    reduceColMinMax_32fFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_32fC3(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    const int cols = srcmat.cols;
    const float initial = isMax ? std::numeric_limits<float>::lowest() : std::numeric_limits<float>::max();
    const int vlmax = __riscv_vsetvlmax_e32m2();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(initial, vlmax);
            vfloat32m2_t acc1 = acc0, acc2 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e32m2(cols - x);
                vfloat32m2x3_t v = __riscv_vlseg3e32_v_f32m2x3(src + x * 3, vl);
                vfloat32m2_t v0 = __riscv_vget_v_f32m2x3_f32m2(v, 0);
                vfloat32m2_t v1 = __riscv_vget_v_f32m2x3_f32m2(v, 1);
                vfloat32m2_t v2 = __riscv_vget_v_f32m2x3_f32m2(v, 2);
                acc0 = isMax ? __riscv_vfmax_tu(acc0, acc0, v0, vl)
                             : __riscv_vfmin_tu(acc0, acc0, v0, vl);
                acc1 = isMax ? __riscv_vfmax_tu(acc1, acc1, v1, vl)
                             : __riscv_vfmin_tu(acc1, acc1, v1, vl);
                acc2 = isMax ? __riscv_vfmax_tu(acc2, acc2, v2, vl)
                             : __riscv_vfmin_tu(acc2, acc2, v2, vl);
                x += vl;
            }
            vfloat32m1_t seed = __riscv_vfmv_s_f_f32m1(initial, __riscv_vsetvlmax_e32m1());
            dst[0] = __riscv_vfmv_f(isMax ? __riscv_vfredmax(acc0, seed, vlmax)
                                          : __riscv_vfredmin(acc0, seed, vlmax));
            dst[1] = __riscv_vfmv_f(isMax ? __riscv_vfredmax(acc1, seed, vlmax)
                                          : __riscv_vfredmin(acc1, seed, vlmax));
            dst[2] = __riscv_vfmv_f(isMax ? __riscv_vfredmax(acc2, seed, vlmax)
                                          : __riscv_vfredmin(acc2, seed, vlmax));
        }
    });
    v_cleanup();
#else
    reduceColMinMax_32fFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_32fC4(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    const int cols = srcmat.cols;
    const float initial = isMax ? std::numeric_limits<float>::lowest() : std::numeric_limits<float>::max();
    const int vlmax = __riscv_vsetvlmax_e32m2();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(initial, vlmax);
            vfloat32m2_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e32m2(cols - x);
                vfloat32m2x4_t v = __riscv_vlseg4e32_v_f32m2x4(src + x * 4, vl);
                vfloat32m2_t v0 = __riscv_vget_v_f32m2x4_f32m2(v, 0);
                vfloat32m2_t v1 = __riscv_vget_v_f32m2x4_f32m2(v, 1);
                vfloat32m2_t v2 = __riscv_vget_v_f32m2x4_f32m2(v, 2);
                vfloat32m2_t v3 = __riscv_vget_v_f32m2x4_f32m2(v, 3);
                acc0 = isMax ? __riscv_vfmax_tu(acc0, acc0, v0, vl)
                             : __riscv_vfmin_tu(acc0, acc0, v0, vl);
                acc1 = isMax ? __riscv_vfmax_tu(acc1, acc1, v1, vl)
                             : __riscv_vfmin_tu(acc1, acc1, v1, vl);
                acc2 = isMax ? __riscv_vfmax_tu(acc2, acc2, v2, vl)
                             : __riscv_vfmin_tu(acc2, acc2, v2, vl);
                acc3 = isMax ? __riscv_vfmax_tu(acc3, acc3, v3, vl)
                             : __riscv_vfmin_tu(acc3, acc3, v3, vl);
                x += vl;
            }
            vfloat32m1_t seed = __riscv_vfmv_s_f_f32m1(initial, __riscv_vsetvlmax_e32m1());
            dst[0] = __riscv_vfmv_f(isMax ? __riscv_vfredmax(acc0, seed, vlmax)
                                          : __riscv_vfredmin(acc0, seed, vlmax));
            dst[1] = __riscv_vfmv_f(isMax ? __riscv_vfredmax(acc1, seed, vlmax)
                                          : __riscv_vfredmin(acc1, seed, vlmax));
            dst[2] = __riscv_vfmv_f(isMax ? __riscv_vfredmax(acc2, seed, vlmax)
                                          : __riscv_vfredmin(acc2, seed, vlmax));
            dst[3] = __riscv_vfmv_f(isMax ? __riscv_vfredmax(acc3, seed, vlmax)
                                          : __riscv_vfredmin(acc3, seed, vlmax));
        }
    });
    v_cleanup();
#elif CV_NEON
    const int cols = srcmat.cols;
    const float initial = isMax ? std::numeric_limits<float>::lowest() : std::numeric_limits<float>::max();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            float32x4x4_t acc = {{
                vdupq_n_f32(initial), vdupq_n_f32(initial),
                vdupq_n_f32(initial), vdupq_n_f32(initial)
            }};
            int x = 0;
            for (; x <= cols - 4; x += 4)
            {
                float32x4x4_t v = vld4q_f32(src + x * 4);
                for (int c = 0; c < 4; c++)
                    acc.val[c] = isMax ? vmaxq_f32(acc.val[c], v.val[c])
                                       : vminq_f32(acc.val[c], v.val[c]);
            }
            for (int c = 0; c < 4; c++)
            {
                float lanes[4];
                vst1q_f32(lanes, acc.val[c]);
                float result = initial;
                for (int i = 0; i < 4; i++)
                    result = reduceScalarMinMax<isMax>(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = reduceScalarMinMax<isMax>(result, src[i * 4 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
#elif CV_AVX2
    const int cols = srcmat.cols;
    const float initial = isMax ? std::numeric_limits<float>::lowest() : std::numeric_limits<float>::max();
    const __m256 accInit = _mm256_set1_ps(initial);
    const __m256 validMask = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0));
    const __m256i idx0 = _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0);
    const __m256i idx1 = _mm256_setr_epi32(1, 5, 0, 0, 0, 0, 0, 0);
    const __m256i idx2 = _mm256_setr_epi32(2, 6, 0, 0, 0, 0, 0, 0);
    const __m256i idx3 = _mm256_setr_epi32(3, 7, 0, 0, 0, 0, 0, 0);

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            __m256 acc0 = accInit, acc1 = accInit, acc2 = accInit, acc3 = accInit;
            int x = 0;
            for (; x <= cols - 2; x += 2)
            {
                __m256 v = _mm256_loadu_ps(src + x * 4);
                __m256 v0 = _mm256_blendv_ps(accInit, _mm256_permutevar8x32_ps(v, idx0), validMask);
                __m256 v1 = _mm256_blendv_ps(accInit, _mm256_permutevar8x32_ps(v, idx1), validMask);
                __m256 v2 = _mm256_blendv_ps(accInit, _mm256_permutevar8x32_ps(v, idx2), validMask);
                __m256 v3 = _mm256_blendv_ps(accInit, _mm256_permutevar8x32_ps(v, idx3), validMask);
                acc0 = isMax ? _mm256_max_ps(acc0, v0) : _mm256_min_ps(acc0, v0);
                acc1 = isMax ? _mm256_max_ps(acc1, v1) : _mm256_min_ps(acc1, v1);
                acc2 = isMax ? _mm256_max_ps(acc2, v2) : _mm256_min_ps(acc2, v2);
                acc3 = isMax ? _mm256_max_ps(acc3, v3) : _mm256_min_ps(acc3, v3);
            }

            __m256 accs[4] = {acc0, acc1, acc2, acc3};
            for (int c = 0; c < 4; c++)
            {
                float lanes[8];
                _mm256_storeu_ps(lanes, accs[c]);
                float result = initial;
                for (int i = 0; i < 8; i++)
                    result = reduceScalarMinMax<isMax>(result, lanes[i]);
                for (int i = x; i < cols; i++)
                    result = reduceScalarMinMax<isMax>(result, src[i * 4 + c]);
                dst[c] = result;
            }
        }
    });
    v_cleanup();
#else
    reduceColMinMax_32fFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_32f(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    if (cn == 1)
        reduceColMinMax_32fC1<isMax>(srcmat, dstmat);
    else if (cn == 3)
        reduceColMinMax_32fC3<isMax>(srcmat, dstmat);
    else if (cn == 4)
        reduceColMinMax_32fC4<isMax>(srcmat, dstmat);
    else
        reduceColMinMax_32fFallback<isMax>(srcmat, dstmat);
}

static void reduceColMax_32f(const Mat& srcmat, Mat& dstmat)
{
    reduceColMinMax_32f<true>(srcmat, dstmat);
}

static void reduceColMin_32f(const Mat& srcmat, Mat& dstmat)
{
    reduceColMinMax_32f<false>(srcmat, dstmat);
}

#if CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
static inline uint32_t reduceSum2_8u_NEON(uint8x16_t v)
{
    uint16x8_t lo = vmull_u8(vget_low_u8(v), vget_low_u8(v));
    uint16x8_t hi = vmull_u8(vget_high_u8(v), vget_high_u8(v));
    return vaddvq_u32(vpaddlq_u16(lo)) + vaddvq_u32(vpaddlq_u16(hi));
}
#endif

template<typename DT>
static void reduceColSum2_8uFallback(const Mat& srcmat, Mat& dstmat)
{
    if (std::is_same<DT, int>::value)
        reduceC_<uchar, int, ReduceOpAddSqr_8U32S, ReduceVecOpAddSqr_8U32S>(srcmat, dstmat);
    else if (std::is_same<DT, float>::value)
        reduceC_<uchar, float, ReduceOpAddSqr_8U32F, ReduceVecOpAddSqr_8U32F>(srcmat, dstmat);
    else
        reduceC_<uchar, double, ReduceOpAddSqr_8U64F, ReduceVecOpAddSqr_8U64F>(srcmat, dstmat);
}

template<typename DT>
static void reduceColSum2_8uC1(const Mat& srcmat, Mat& dstmat)
{
#if (CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))) || CV_AVX2 || CV_RVV
    const int cols = srcmat.cols;

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            uint32_t result = 0;
            int x = 0;
#if CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
            for (; x <= cols - 16; x += 16)
                result += reduceSum2_8u_NEON(vld1q_u8(src + x));
#elif CV_AVX2
            __m256i acc = _mm256_setzero_si256();
            for (; x <= cols - 32; x += 32)
            {
                __m256i bytes = _mm256_loadu_si256((const __m256i*)(src + x));
                __m128i lo = _mm256_castsi256_si128(bytes);
                __m128i hi = _mm256_extracti128_si256(bytes, 1);
                __m256i lo16 = _mm256_cvtepu8_epi16(lo);
                __m256i hi16 = _mm256_cvtepu8_epi16(hi);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lo16, lo16));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(hi16, hi16));
            }
            uint32_t lanes[8];
            _mm256_storeu_si256((__m256i*)lanes, acc);
            for (int i = 0; i < 8; i++)
                result += lanes[i];
#elif CV_RVV
            vuint32m1_t acc = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e8m4(cols - x);
                vuint8m4_t v = __riscv_vle8_v_u8m4(src + x, vl);
                acc = __riscv_vwredsumu(__riscv_vwmulu(v, v, vl), acc, vl);
                x += vl;
            }
            result = (uint32_t)__riscv_vmv_x(acc);
#endif
            for (; x < cols; x++)
                result += (uint32_t)src[x] * src[x];
            dst[0] = (DT)(int32_t)result;
        }
    });
    v_cleanup();
#else
    reduceColSum2_8uFallback<DT>(srcmat, dstmat);
#endif
}

template<typename DT>
static void reduceColSum2_8uC3(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            vuint32m1_t acc0 = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
            vuint32m1_t acc1 = acc0, acc2 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e8m1(cols - x);
                vuint8m1x3_t v = __riscv_vlseg3e8_v_u8m1x3(src + x * 3, vl);
                vuint8m1_t v0 = __riscv_vget_v_u8m1x3_u8m1(v, 0);
                vuint8m1_t v1 = __riscv_vget_v_u8m1x3_u8m1(v, 1);
                vuint8m1_t v2 = __riscv_vget_v_u8m1x3_u8m1(v, 2);
                acc0 = __riscv_vwredsumu(__riscv_vwmulu(v0, v0, vl), acc0, vl);
                acc1 = __riscv_vwredsumu(__riscv_vwmulu(v1, v1, vl), acc1, vl);
                acc2 = __riscv_vwredsumu(__riscv_vwmulu(v2, v2, vl), acc2, vl);
                x += vl;
            }
            dst[0] = (DT)(int32_t)(uint32_t)__riscv_vmv_x(acc0);
            dst[1] = (DT)(int32_t)(uint32_t)__riscv_vmv_x(acc1);
            dst[2] = (DT)(int32_t)(uint32_t)__riscv_vmv_x(acc2);
        }
    });
    v_cleanup();
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            uint32_t results[3] = {0, 0, 0};
            int x = 0;
            for (; x <= cols - 16; x += 16)
            {
                uint8x16x3_t v = vld3q_u8(src + x * 3);
                for (int c = 0; c < 3; c++)
                    results[c] += reduceSum2_8u_NEON(v.val[c]);
            }
            for (int c = 0; c < 3; c++)
            {
                for (int i = x; i < cols; i++)
                {
                    uint32_t value = src[i * 3 + c];
                    results[c] += value * value;
                }
                dst[c] = (DT)(int32_t)results[c];
            }
        }
    });
    v_cleanup();
#elif CV_AVX2
    const int cols = srcmat.cols;
    const __m256i mask0 = _mm256_setr_epi8(
            0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            2, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask1 = _mm256_setr_epi8(
            1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 3, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i mask2 = _mm256_setr_epi8(
            2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            int x = 0;
            for (; x <= cols - 11; x += 8)
            {
                __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 3));
                __m256i channels[3] = {
                    _mm256_shuffle_epi8(v, mask0),
                    _mm256_shuffle_epi8(v, mask1),
                    _mm256_shuffle_epi8(v, mask2)
                };
                __m256i* accs[3] = {&acc0, &acc1, &acc2};
                for (int c = 0; c < 3; c++)
                {
                    __m128i lo = _mm256_castsi256_si128(channels[c]);
                    __m128i hi = _mm256_extracti128_si256(channels[c], 1);
                    __m256i lo16 = _mm256_cvtepu8_epi16(lo);
                    __m256i hi16 = _mm256_cvtepu8_epi16(hi);
                    *accs[c] = _mm256_add_epi32(*accs[c], _mm256_madd_epi16(lo16, lo16));
                    *accs[c] = _mm256_add_epi32(*accs[c], _mm256_madd_epi16(hi16, hi16));
                }
            }

            __m256i accs[3] = {acc0, acc1, acc2};
            for (int c = 0; c < 3; c++)
            {
                uint32_t lanes[8];
                uint32_t result = 0;
                _mm256_storeu_si256((__m256i*)lanes, accs[c]);
                for (int i = 0; i < 8; i++)
                    result += lanes[i];
                for (int i = x; i < cols; i++)
                {
                    uint32_t value = src[i * 3 + c];
                    result += value * value;
                }
                dst[c] = (DT)(int32_t)result;
            }
        }
    });
    v_cleanup();
#else
    reduceColSum2_8uFallback<DT>(srcmat, dstmat);
#endif
}

template<typename DT>
static void reduceColSum2_8uC4(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            vuint32m1_t acc0 = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
            vuint32m1_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e8m1(cols - x);
                vuint8m1x4_t v = __riscv_vlseg4e8_v_u8m1x4(src + x * 4, vl);
                vuint8m1_t v0 = __riscv_vget_v_u8m1x4_u8m1(v, 0);
                vuint8m1_t v1 = __riscv_vget_v_u8m1x4_u8m1(v, 1);
                vuint8m1_t v2 = __riscv_vget_v_u8m1x4_u8m1(v, 2);
                vuint8m1_t v3 = __riscv_vget_v_u8m1x4_u8m1(v, 3);
                acc0 = __riscv_vwredsumu(__riscv_vwmulu(v0, v0, vl), acc0, vl);
                acc1 = __riscv_vwredsumu(__riscv_vwmulu(v1, v1, vl), acc1, vl);
                acc2 = __riscv_vwredsumu(__riscv_vwmulu(v2, v2, vl), acc2, vl);
                acc3 = __riscv_vwredsumu(__riscv_vwmulu(v3, v3, vl), acc3, vl);
                x += vl;
            }
            dst[0] = (DT)(int32_t)(uint32_t)__riscv_vmv_x(acc0);
            dst[1] = (DT)(int32_t)(uint32_t)__riscv_vmv_x(acc1);
            dst[2] = (DT)(int32_t)(uint32_t)__riscv_vmv_x(acc2);
            dst[3] = (DT)(int32_t)(uint32_t)__riscv_vmv_x(acc3);
        }
    });
    v_cleanup();
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            uint32_t results[4] = {0, 0, 0, 0};
            int x = 0;
            for (; x <= cols - 16; x += 16)
            {
                uint8x16x4_t v = vld4q_u8(src + x * 4);
                for (int c = 0; c < 4; c++)
                    results[c] += reduceSum2_8u_NEON(v.val[c]);
            }
            for (int c = 0; c < 4; c++)
            {
                for (int i = x; i < cols; i++)
                {
                    uint32_t value = src[i * 4 + c];
                    results[c] += value * value;
                }
                dst[c] = (DT)(int32_t)results[c];
            }
        }
    });
    v_cleanup();
#elif CV_AVX2
    const int cols = srcmat.cols;
    const __m256i mask0 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i mask1 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(1, 5, 9, 13, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i mask2 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(2, 6, 10, 14, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));
    const __m256i mask3 = _mm256_broadcastsi128_si256(
            _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1));

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();
            int x = 0;
            for (; x <= cols - 8; x += 8)
            {
                __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 4));
                __m256i channels[4] = {
                    _mm256_shuffle_epi8(v, mask0),
                    _mm256_shuffle_epi8(v, mask1),
                    _mm256_shuffle_epi8(v, mask2),
                    _mm256_shuffle_epi8(v, mask3)
                };
                __m256i* accs[4] = {&acc0, &acc1, &acc2, &acc3};
                for (int c = 0; c < 4; c++)
                {
                    __m128i lo = _mm256_castsi256_si128(channels[c]);
                    __m128i hi = _mm256_extracti128_si256(channels[c], 1);
                    __m256i lo16 = _mm256_cvtepu8_epi16(lo);
                    __m256i hi16 = _mm256_cvtepu8_epi16(hi);
                    *accs[c] = _mm256_add_epi32(*accs[c], _mm256_madd_epi16(lo16, lo16));
                    *accs[c] = _mm256_add_epi32(*accs[c], _mm256_madd_epi16(hi16, hi16));
                }
            }

            __m256i accs[4] = {acc0, acc1, acc2, acc3};
            for (int c = 0; c < 4; c++)
            {
                uint32_t lanes[8];
                uint32_t result = 0;
                _mm256_storeu_si256((__m256i*)lanes, accs[c]);
                for (int i = 0; i < 8; i++)
                    result += lanes[i];
                for (int i = x; i < cols; i++)
                {
                    uint32_t value = src[i * 4 + c];
                    result += value * value;
                }
                dst[c] = (DT)(int32_t)result;
            }
        }
    });
    v_cleanup();
#else
    reduceColSum2_8uFallback<DT>(srcmat, dstmat);
#endif
}

template<typename DT>
static void reduceColSum2_8u(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    if (cn == 1)
        reduceColSum2_8uC1<DT>(srcmat, dstmat);
    else if (cn == 3)
        reduceColSum2_8uC3<DT>(srcmat, dstmat);
    else if (cn == 4)
        reduceColSum2_8uC4<DT>(srcmat, dstmat);
    else
        reduceColSum2_8uFallback<DT>(srcmat, dstmat);
}

static void reduceColSum2_8u32s(const Mat& srcmat, Mat& dstmat)
{
    reduceColSum2_8u<int>(srcmat, dstmat);
}

static void reduceColSum2_8u32f(const Mat& srcmat, Mat& dstmat)
{
    reduceColSum2_8u<float>(srcmat, dstmat);
}

static void reduceColSum2_8u64f(const Mat& srcmat, Mat& dstmat)
{
    reduceColSum2_8u<double>(srcmat, dstmat);
}

static void reduceColSum2_32f32fFallback(const Mat& srcmat, Mat& dstmat)
{
    reduceC_<float, float, ReduceOpAddSqr_32F32F, ReduceVecOpAddSqr_32F32F>(srcmat, dstmat);
}

static void reduceColSum2_32f32fC1(const Mat& srcmat, Mat& dstmat)
{
#if CV_NEON || CV_AVX2 || CV_RVV
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float result = 0;
            int x = 0;
#if CV_NEON
            float32x4_t acc = vdupq_n_f32(0);
            for (; x <= cols - 4; x += 4)
            {
                float32x4_t v = vld1q_f32(src + x);
                acc = vaddq_f32(acc, vmulq_f32(v, v));
            }
            float lanes[4];
            vst1q_f32(lanes, acc);
            for (int i = 0; i < 4; i++)
                result += lanes[i];
#elif CV_AVX2
            __m256 acc = _mm256_setzero_ps();
            for (; x <= cols - 8; x += 8)
            {
                __m256 v = _mm256_loadu_ps(src + x);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
            }
            float lanes[8];
            _mm256_storeu_ps(lanes, acc);
            for (int i = 0; i < 8; i++)
                result += lanes[i];
#elif CV_RVV
            const int vlmax = __riscv_vsetvlmax_e32m8();
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0, vlmax);
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e32m8(cols - x);
                vfloat32m8_t v = __riscv_vle32_v_f32m8(src + x, vl);
                acc = __riscv_vfmacc_tu(acc, v, v, vl);
                x += vl;
            }
            vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0, __riscv_vsetvlmax_e32m1());
            result = __riscv_vfmv_f(__riscv_vfredusum(acc, zero, vlmax));
#endif
            for (; x < cols; x++)
                result += src[x] * src[x];
            dstmat.ptr<float>(y)[0] = result;
        }
    });
    v_cleanup();
#else
    reduceColSum2_32f32fFallback(srcmat, dstmat);
#endif
}

static void reduceColSum2_32f32fC3(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    const int cols = srcmat.cols;
    const int vlmax = __riscv_vsetvlmax_e32m2();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0, vlmax);
            vfloat32m2_t acc1 = acc0, acc2 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e32m2(cols - x);
                vfloat32m2x3_t v = __riscv_vlseg3e32_v_f32m2x3(src + x * 3, vl);
                vfloat32m2_t v0 = __riscv_vget_v_f32m2x3_f32m2(v, 0);
                vfloat32m2_t v1 = __riscv_vget_v_f32m2x3_f32m2(v, 1);
                vfloat32m2_t v2 = __riscv_vget_v_f32m2x3_f32m2(v, 2);
                acc0 = __riscv_vfmacc_tu(acc0, v0, v0, vl);
                acc1 = __riscv_vfmacc_tu(acc1, v1, v1, vl);
                acc2 = __riscv_vfmacc_tu(acc2, v2, v2, vl);
                x += vl;
            }
            vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0, __riscv_vsetvlmax_e32m1());
            dst[0] = __riscv_vfmv_f(__riscv_vfredusum(acc0, zero, vlmax));
            dst[1] = __riscv_vfmv_f(__riscv_vfredusum(acc1, zero, vlmax));
            dst[2] = __riscv_vfmv_f(__riscv_vfredusum(acc2, zero, vlmax));
        }
    });
    v_cleanup();
#else
    reduceColSum2_32f32fFallback(srcmat, dstmat);
#endif
}

static void reduceColSum2_32f32fC4(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    const int cols = srcmat.cols;
    const int vlmax = __riscv_vsetvlmax_e32m2();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0, vlmax);
            vfloat32m2_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e32m2(cols - x);
                vfloat32m2x4_t v = __riscv_vlseg4e32_v_f32m2x4(src + x * 4, vl);
                vfloat32m2_t v0 = __riscv_vget_v_f32m2x4_f32m2(v, 0);
                vfloat32m2_t v1 = __riscv_vget_v_f32m2x4_f32m2(v, 1);
                vfloat32m2_t v2 = __riscv_vget_v_f32m2x4_f32m2(v, 2);
                vfloat32m2_t v3 = __riscv_vget_v_f32m2x4_f32m2(v, 3);
                acc0 = __riscv_vfmacc_tu(acc0, v0, v0, vl);
                acc1 = __riscv_vfmacc_tu(acc1, v1, v1, vl);
                acc2 = __riscv_vfmacc_tu(acc2, v2, v2, vl);
                acc3 = __riscv_vfmacc_tu(acc3, v3, v3, vl);
                x += vl;
            }
            vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0, __riscv_vsetvlmax_e32m1());
            dst[0] = __riscv_vfmv_f(__riscv_vfredusum(acc0, zero, vlmax));
            dst[1] = __riscv_vfmv_f(__riscv_vfredusum(acc1, zero, vlmax));
            dst[2] = __riscv_vfmv_f(__riscv_vfredusum(acc2, zero, vlmax));
            dst[3] = __riscv_vfmv_f(__riscv_vfredusum(acc3, zero, vlmax));
        }
    });
    v_cleanup();
#elif CV_NEON
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            float32x4x4_t acc = {{
                vdupq_n_f32(0), vdupq_n_f32(0),
                vdupq_n_f32(0), vdupq_n_f32(0)
            }};
            int x = 0;
            for (; x <= cols - 4; x += 4)
            {
                float32x4x4_t v = vld4q_f32(src + x * 4);
                for (int c = 0; c < 4; c++)
                    acc.val[c] = vmlaq_f32(acc.val[c], v.val[c], v.val[c]);
            }
            for (int c = 0; c < 4; c++)
            {
                float lanes[4];
                float result = 0;
                vst1q_f32(lanes, acc.val[c]);
                for (int i = 0; i < 4; i++)
                    result += lanes[i];
                for (int i = x; i < cols; i++)
                    result += src[i * 4 + c] * src[i * 4 + c];
                dst[c] = result;
            }
        }
    });
    v_cleanup();
#elif CV_AVX2
    const int cols = srcmat.cols;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 validMask = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0));
    const __m256i idx0 = _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0);
    const __m256i idx1 = _mm256_setr_epi32(1, 5, 0, 0, 0, 0, 0, 0);
    const __m256i idx2 = _mm256_setr_epi32(2, 6, 0, 0, 0, 0, 0, 0);
    const __m256i idx3 = _mm256_setr_epi32(3, 7, 0, 0, 0, 0, 0, 0);

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            __m256 acc0 = zero, acc1 = zero, acc2 = zero, acc3 = zero;
            int x = 0;
            for (; x <= cols - 2; x += 2)
            {
                __m256 v = _mm256_loadu_ps(src + x * 4);
                __m256 v0 = _mm256_blendv_ps(zero, _mm256_permutevar8x32_ps(v, idx0), validMask);
                __m256 v1 = _mm256_blendv_ps(zero, _mm256_permutevar8x32_ps(v, idx1), validMask);
                __m256 v2 = _mm256_blendv_ps(zero, _mm256_permutevar8x32_ps(v, idx2), validMask);
                __m256 v3 = _mm256_blendv_ps(zero, _mm256_permutevar8x32_ps(v, idx3), validMask);
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(v0, v0));
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(v1, v1));
                acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(v2, v2));
                acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(v3, v3));
            }

            __m256 accs[4] = {acc0, acc1, acc2, acc3};
            for (int c = 0; c < 4; c++)
            {
                float lanes[8];
                float result = 0;
                _mm256_storeu_ps(lanes, accs[c]);
                for (int i = 0; i < 8; i++)
                    result += lanes[i];
                for (int i = x; i < cols; i++)
                    result += src[i * 4 + c] * src[i * 4 + c];
                dst[c] = result;
            }
        }
    });
    v_cleanup();
#else
    reduceColSum2_32f32fFallback(srcmat, dstmat);
#endif
}

static void reduceColSum2_32f32f(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    if (cn == 1)
        reduceColSum2_32f32fC1(srcmat, dstmat);
    else if (cn == 3)
        reduceColSum2_32f32fC3(srcmat, dstmat);
    else if (cn == 4)
        reduceColSum2_32f32fC4(srcmat, dstmat);
    else
        reduceColSum2_32f32fFallback(srcmat, dstmat);
}
