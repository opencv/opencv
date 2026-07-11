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
reduceColGeneric(const Mat& srcmat, Mat& dstmat)
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
        reduceColGeneric<uchar, uchar, ReduceOpMax_8U, ReduceVecOpMax_8U>(srcmat, dstmat);
    else
        reduceColGeneric<uchar, uchar, ReduceOpMin_8U, ReduceVecOpMin_8U>(srcmat, dstmat);
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
        reduceColGeneric<float, float, ReduceOpMax_32F, ReduceVecOpMax_32F>(srcmat, dstmat);
    else
        reduceColGeneric<float, float, ReduceOpMin_32F, ReduceVecOpMin_32F>(srcmat, dstmat);
}

template<typename DT>
static void reduceColSum2_8uFallback(const Mat& srcmat, Mat& dstmat)
{
    if (std::is_same<DT, int>::value)
        reduceColGeneric<uchar, int, ReduceOpAddSqr_8U32S, ReduceVecOpAddSqr_8U32S>(srcmat, dstmat);
    else if (std::is_same<DT, float>::value)
        reduceColGeneric<uchar, float, ReduceOpAddSqr_8U32F, ReduceVecOpAddSqr_8U32F>(srcmat, dstmat);
    else
        reduceColGeneric<uchar, double, ReduceOpAddSqr_8U64F, ReduceVecOpAddSqr_8U64F>(srcmat, dstmat);
}

static void reduceColSum2_32f32fFallback(const Mat& srcmat, Mat& dstmat)
{
    reduceColGeneric<float, float, ReduceOpAddSqr_32F32F, ReduceVecOpAddSqr_32F32F>(srcmat, dstmat);
}
