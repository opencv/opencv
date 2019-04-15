// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD128
    #error "Not a standalone header"
#endif

#define OPENCV_HAL_IMPL_SSE_TYPES_DISCAST \
    void cast(v_mask8x16 disable_mask);   \
    void cast(v_mask16x8 disable_mask);   \
    void cast(v_mask32x4 disable_mask);   \
    void cast(v_mask64x2 disable_mask);

#define OPENCV_HAL_IMPL_SSE_TYPES_I(_Tvec)       \
    __m128i val;                                 \
    _Tvec() {}                                   \
    _Tvec(const __m128i& v) : val(v) {}          \
    operator __m128i() const                     \
    { return val; }                              \
    static inline _Tvec zero()                   \
    { return _mm_setzero_si128(); }              \
    template<typename Tvec>                      \
    static inline _Tvec cast(const Tvec& v)      \
    { return _mm_castsi128_non(v); }             \
    template<typename Tmvec>                     \
    static inline _Tvec fromMask(const Tmvec& a) \
    { return _Tvec(a); }                         \
    OPENCV_HAL_IMPL_SSE_TYPES_DISCAST

struct v_uint8x16
{
    OPENCV_HAL_IMPL_SSE_TYPES_I(v_uint8x16)

    typedef uchar lane_type;
    enum { nlanes = 16 };

    v_uint8x16(int v) : val(_mm_set1_epi8((char)v))
    {}
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
               uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
    : val(_mm_setr_epi8((char)v0, (char)v1, (char)v2, (char)v3,
                        (char)v4, (char)v5, (char)v6, (char)v7,
                        (char)v8, (char)v9, (char)v10, (char)v11,
                        (char)v12, (char)v13, (char)v14, (char)v15))
    {}
    uchar get0() const
    { return (uchar)_mm_cvtsi128_si32(val); }
};

struct v_int8x16
{
    OPENCV_HAL_IMPL_SSE_TYPES_I(v_int8x16)

    typedef schar lane_type;
    enum { nlanes = 16 };

    v_int8x16(int v) : val(_mm_set1_epi8((char)v))
    {}
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
              schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
    : val(_mm_setr_epi8(v0, v1, v2, v3, v4, v5, v6, v7,
                        v8, v9, v10, v11, v12, v13, v14, v15))
    {}
    schar get0() const
    { return (schar)_mm_cvtsi128_si32(val); }
};

struct v_uint16x8
{
    OPENCV_HAL_IMPL_SSE_TYPES_I(v_uint16x8)

    typedef ushort lane_type;
    enum { nlanes = 8 };

    v_uint16x8(int v) : val(_mm_set1_epi16((short)v))
    {}
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
    : val(_mm_setr_epi16((short)v0, (short)v1, (short)v2, (short)v3,
                         (short)v4, (short)v5, (short)v6, (short)v7))
    {}
    ushort get0() const
    { return (ushort)_mm_cvtsi128_si32(val); }
};

struct v_int16x8
{
    OPENCV_HAL_IMPL_SSE_TYPES_I(v_int16x8)

    typedef short lane_type;
    enum { nlanes = 8 };

    v_int16x8(int v) : val(_mm_set1_epi16((short)v))
    {}
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
    : val(_mm_setr_epi16(v0, v1, v2, v3, v4, v5, v6, v7))
    {}
    short get0() const
    { return (short)_mm_cvtsi128_si32(val); }
};

struct v_uint32x4
{
    OPENCV_HAL_IMPL_SSE_TYPES_I(v_uint32x4)

    typedef unsigned lane_type;
    enum { nlanes = 4 };

    v_uint32x4(unsigned v) : val(_mm_set1_epi32((int)v))
    {}
    v_uint32x4(unsigned v0, unsigned v1, unsigned v2, unsigned v3)
    : val(_mm_setr_epi32((int)v0, (int)v1, (int)v2, (int)v3))
    {}
    unsigned get0() const
    { return (unsigned)_mm_cvtsi128_si32(val); }
};

struct v_int32x4
{
    OPENCV_HAL_IMPL_SSE_TYPES_I(v_int32x4)

    typedef int lane_type;
    enum { nlanes = 4 };

    v_int32x4(int v) : val(_mm_set1_epi32(v))
    {}
    v_int32x4(int v0, int v1, int v2, int v3)
    : val(_mm_setr_epi32(v0, v1, v2, v3))
    {}
    int get0() const
    { return _mm_cvtsi128_si32(val); }
};

struct v_uint64x2
{
    OPENCV_HAL_IMPL_SSE_TYPES_I(v_uint64x2)

    typedef uint64 lane_type;
    enum { nlanes = 2 };

    v_uint64x2(uint64 v) : val(_mm_set1_epi64x((int64)v))
    {}
    v_uint64x2(uint64 v0, uint64 v1) : val(_mm_set_epi64x((int64)v1, (int64)v0))
    {}
    uint64 get0() const
    { return (uint64)_mm_cvtsi128_si64(val); }
};

struct v_int64x2
{
    OPENCV_HAL_IMPL_SSE_TYPES_I(v_int64x2)

    typedef int64 lane_type;
    enum { nlanes = 2 };

    v_int64x2(uint64 v) : val(_mm_set1_epi64x(v))
    {}
    v_int64x2(int64 v0, int64 v1) : val(_mm_set_epi64x(v1, v0))
    {}
    int64 get0() const
    { return _mm_cvtsi128_si64(val); }
};

struct v_float32x4
{
    __m128 val;

    typedef float lane_type;
    enum { nlanes = 4 };

    v_float32x4()
    {}
    v_float32x4(const __m128& v) : val(v)
    {}
    v_float32x4(float v) : val(_mm_set1_ps(v))
    {}
    v_float32x4(float v0, float v1, float v2, float v3)
    : val(_mm_setr_ps(v0, v1, v2, v3))
    {}

    operator __m128() const
    { return val; }

    float get0() const
    { return _mm_cvtss_f32(val); }
    static inline v_float32x4 zero()
    { return _mm_setzero_ps(); }

    template<typename Tvec>
    static inline v_float32x4 cast(const Tvec& v)
    { return _mm_castps_non(v); }

    template<typename Tmvec>
    static inline v_float32x4 fromMask(const Tmvec& a)
    { return _mm_castsi128_ps(a); }

    OPENCV_HAL_IMPL_SSE_TYPES_DISCAST
};

struct v_float64x2
{
    __m128d val;

    typedef double lane_type;
    enum { nlanes = 2 };

    v_float64x2()
    {}
    v_float64x2(const __m128d& v) : val(v)
    {}
    v_float64x2(double v) : val(_mm_set1_pd(v))
    {}
    v_float64x2(double v0, double v1) : val(_mm_setr_pd(v0, v1))
    {}

    operator __m128d() const
    { return val; }

    double get0() const
    { return _mm_cvtsd_f64(val); }
    static inline v_float64x2 zero()
    { return _mm_setzero_pd(); }

    template<typename Tvec>
    static inline v_float64x2 cast(const Tvec& v)
    { return _mm_castpd_non(v); }

    template<typename Tmvec>
    static inline v_float64x2 fromMask(const Tmvec& a)
    { return _mm_castsi128_pd(a); }

    OPENCV_HAL_IMPL_SSE_TYPES_DISCAST
};

#define OPENCV_HAL_IMPL_SSE_TYPES_MASK(_Tmvec)  \
    __m128i val;                                \
    _Tmvec() {}                                 \
    _Tmvec(const __m128i& v) : val(v) {}        \
    operator __m128i() const                    \
    { return val; }                             \
    template<typename Tvec>                     \
    static inline _Tmvec from(const Tvec& a)    \
    { return _mm_castsi128_non(a); }

struct v_mask8x16
{
    OPENCV_HAL_IMPL_SSE_TYPES_MASK(v_mask8x16)
};
struct v_mask16x8
{
    OPENCV_HAL_IMPL_SSE_TYPES_MASK(v_mask16x8)
};
struct v_mask32x4
{
    OPENCV_HAL_IMPL_SSE_TYPES_MASK(v_mask32x4)
};
struct v_mask64x2
{
    OPENCV_HAL_IMPL_SSE_TYPES_MASK(v_mask64x2)
};

//////////////// Init & Cast ///////////////

#define OPENCV_HAL_IMPL_SSE_INIT(_Tvec, _Tp, suffix)       \
    inline _Tvec v_setzero_##suffix()                      \
    { return _Tvec::zero(); }                              \
    inline _Tvec v_setall_##suffix(_Tp v)                  \
    { return _Tvec(v); }                                   \
    template<typename Tvec>                                \
    inline _Tvec v_reinterpret_as_##suffix(const Tvec& a)  \
    { return _Tvec::cast(a); }

OPENCV_HAL_IMPL_SSE_INIT(v_uint8x16,  uchar,    u8)
OPENCV_HAL_IMPL_SSE_INIT(v_int8x16,   schar,    s8)
OPENCV_HAL_IMPL_SSE_INIT(v_uint16x8,  ushort,   u16)
OPENCV_HAL_IMPL_SSE_INIT(v_int16x8,   short,    s16)
OPENCV_HAL_IMPL_SSE_INIT(v_uint32x4,  unsigned, u32)
OPENCV_HAL_IMPL_SSE_INIT(v_int32x4,   int,      s32)
OPENCV_HAL_IMPL_SSE_INIT(v_uint64x2,  uint64,   u64)
OPENCV_HAL_IMPL_SSE_INIT(v_int64x2,   int64,    s64)
OPENCV_HAL_IMPL_SSE_INIT(v_float32x4, float,    f32)
OPENCV_HAL_IMPL_SSE_INIT(v_float64x2, double,   f64)