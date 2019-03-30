// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD512
    #error "Not a standalone header"
#endif

#define OPENCV_HAL_IMPL_AVX512_TYPES_I(_Tvec)                      \
    _Tvec() {}                                                     \
    _Tvec(const __m512i& v) { val = v; }                           \
    _Tvec(const __m512&  v) { val = _mm512_castps_si512(v); }      \
    _Tvec(const __m512d& v) { val = _mm512_castpd_si512(v); }      \
    _Tvec& operator = (const __m512i& v)                           \
    { val = v; return *this; }                                     \
    _Tvec& operator = (const __m512& v)                            \
    { val = _mm512_castps_si512(v); return *this; }                \
    _Tvec& operator = (const __m512d& v)                           \
    { val = _mm512_castpd_si512(v); return *this; }                \
    operator __m512i() const { return val; }                       \
    operator __m512()  const { return _mm512_castsi512_ps(val); }  \
    operator __m512d() const { return _mm512_castsi512_pd(val); }

// note: avoid use of v_512i as return type for public intrinsics
class v_512i
{
protected:
    __m512i val;
public:
    OPENCV_HAL_IMPL_AVX512_TYPES_I(v_512i)
};

struct v_uint8x64 : public v_512i
{
    OPENCV_HAL_IMPL_AVX512_TYPES_I(v_uint8x64)

    typedef uchar lane_type;
    enum { nlanes = 64 };

    v_uint8x64(uchar v0,  uchar v1,  uchar v2,  uchar v3,
               uchar v4,  uchar v5,  uchar v6,  uchar v7,
               uchar v8,  uchar v9,  uchar v10, uchar v11,
               uchar v12, uchar v13, uchar v14, uchar v15,
               uchar v16, uchar v17, uchar v18, uchar v19,
               uchar v20, uchar v21, uchar v22, uchar v23,
               uchar v24, uchar v25, uchar v26, uchar v27,
               uchar v28, uchar v29, uchar v30, uchar v31,
               uchar v32, uchar v33, uchar v34, uchar v35,
               uchar v36, uchar v37, uchar v38, uchar v39,
               uchar v40, uchar v41, uchar v42, uchar v43,
               uchar v44, uchar v45, uchar v46, uchar v47,
               uchar v48, uchar v49, uchar v50, uchar v51,
               uchar v52, uchar v53, uchar v54, uchar v55,
               uchar v56, uchar v57, uchar v58, uchar v59,
               uchar v60, uchar v61, uchar v62, uchar v63)
    {
        val = _mm512_setr_epi8(
            (char)v0,  (char)v1,  (char)v2,  (char)v3,  (char)v4,  (char)v5,
            (char)v6 , (char)v7,  (char)v8,  (char)v9,  (char)v10, (char)v11,
            (char)v12, (char)v13, (char)v14, (char)v15, (char)v16, (char)v17,
            (char)v18, (char)v19, (char)v20, (char)v21, (char)v22, (char)v23,
            (char)v24, (char)v25, (char)v26, (char)v27, (char)v28, (char)v29,
            (char)v30, (char)v31, (char)v32, (char)v33, (char)v34, (char)v35,
            (char)v36, (char)v37, (char)v38, (char)v39, (char)v40, (char)v41,
            (char)v42, (char)v43, (char)v44, (char)v45, (char)v46, (char)v47,
            (char)v48, (char)v49, (char)v50, (char)v51, (char)v52, (char)v53,
            (char)v54, (char)v55, (char)v56, (char)v57, (char)v58, (char)v59,
            (char)v60, (char)v61, (char)v62, (char)v63
        );
    }
    uchar get0() const
    { return (uchar)_mm512_cvtsi512_si32(val); }
};

struct v_int8x64 : public v_512i
{
    OPENCV_HAL_IMPL_AVX512_TYPES_I(v_int8x64)

    typedef schar lane_type;
    enum { nlanes = 64 };

    v_int8x64(schar v0,  schar v1,  schar v2,  schar v3,
              schar v4,  schar v5,  schar v6,  schar v7,
              schar v8,  schar v9,  schar v10, schar v11,
              schar v12, schar v13, schar v14, schar v15,
              schar v16, schar v17, schar v18, schar v19,
              schar v20, schar v21, schar v22, schar v23,
              schar v24, schar v25, schar v26, schar v27,
              schar v28, schar v29, schar v30, schar v31,
              schar v32, schar v33, schar v34, schar v35,
              schar v36, schar v37, schar v38, schar v39,
              schar v40, schar v41, schar v42, schar v43,
              schar v44, schar v45, schar v46, schar v47,
              schar v48, schar v49, schar v50, schar v51,
              schar v52, schar v53, schar v54, schar v55,
              schar v56, schar v57, schar v58, schar v59,
              schar v60, schar v61, schar v62, schar v63)
    {
        val = _mm512_setr_epi8(
            v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,  v8,  v9,  v10, v11,
            v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23,
            v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35,
            v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47,
            v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59,
            v60, v61, v62, v63
        );
    }
    schar get0() const
    { return (schar)_mm512_cvtsi512_si32(val); }
};

struct v_uint16x32 : public v_512i
{
    OPENCV_HAL_IMPL_AVX512_TYPES_I(v_uint16x32)

    typedef ushort lane_type;
    enum { nlanes = 32 };

    v_uint16x32(ushort v0,  ushort v1,  ushort v2,  ushort v3,
                ushort v4,  ushort v5,  ushort v6,  ushort v7,
                ushort v8,  ushort v9,  ushort v10, ushort v11,
                ushort v12, ushort v13, ushort v14, ushort v15,
                ushort v16, ushort v17, ushort v18, ushort v19,
                ushort v20, ushort v21, ushort v22, ushort v23,
                ushort v24, ushort v25, ushort v26, ushort v27,
                ushort v28, ushort v29, ushort v30, ushort v31)
    {
        val = _mm512_setr_epi16(
            (short)v0,  (short)v1,  (short)v2,  (short)v3,  (short)v4,
            (short)v5,  (short)v6 , (short)v7,  (short)v8,  (short)v9,
            (short)v10, (short)v11, (short)v12, (short)v13, (short)v14,
            (short)v15, (short)v16, (short)v17, (short)v18, (short)v19,
            (short)v20, (short)v21, (short)v22, (short)v23, (short)v24,
            (short)v25, (short)v26, (short)v27, (short)v28, (short)v29,
            (short)v30, (short)v31
        );
    }
    ushort get0() const
    { return (ushort)_mm512_cvtsi512_si32(val); }
};

struct v_int16x32 : public v_512i
{
    OPENCV_HAL_IMPL_AVX512_TYPES_I(v_int16x32)

    typedef short lane_type;
    enum { nlanes = 32 };

    v_int16x32(short v0,  short v1,  short v2,  short v3,
               short v4,  short v5,  short v6,  short v7,
               short v8,  short v9,  short v10, short v11,
               short v12, short v13, short v14, short v15,
               short v16, short v17, short v18, short v19,
               short v20, short v21, short v22, short v23,
               short v24, short v25, short v26, short v27,
               short v28, short v29, short v30, short v31)
    {
        val = _mm512_setr_epi16(
            v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,  v8,  v9,  v10,
            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21,
            v22, v23, v24, v25, v26, v27, v28, v29, v30, v31
        );
    }
    short get0() const
    { return (short)_mm512_cvtsi512_si32(val); }
};

struct v_uint32x16 : public v_512i
{
    OPENCV_HAL_IMPL_AVX512_TYPES_I(v_uint32x16)

    typedef unsigned lane_type;
    enum { nlanes = 16 };

    v_uint32x16(unsigned v0,  unsigned v1,  unsigned v2,  unsigned v3,
                unsigned v4,  unsigned v5,  unsigned v6,  unsigned v7,
                unsigned v8,  unsigned v9,  unsigned v10, unsigned v11,
                unsigned v12, unsigned v13, unsigned v14, unsigned v15)
    {
        val = _mm512_setr_epi32(
            (int)v0,  (int)v1,  (int)v2,  (int)v3, (int)v4,  (int)v5,
            (int)v6,  (int)v7,  (int)v8,  (int)v9, (int)v10, (int)v11,
            (int)v12, (int)v13, (int)v14, (int)v15
        );
    }
    unsigned get0() const
    { return (unsigned)_mm512_cvtsi512_si32(val); }
};

struct v_int32x16 : public v_512i
{
    OPENCV_HAL_IMPL_AVX512_TYPES_I(v_int32x16)

    typedef int lane_type;
    enum { nlanes = 16 };

    v_int32x16(int v0,  int v1,  int v2,  int v3,
               int v4,  int v5,  int v6,  int v7,
               int v8,  int v9,  int v10, int v11,
               int v12, int v13, int v14, int v15)
    {
        val = _mm512_setr_epi32(
            v0, v1, v2,  v3,  v4,  v5,  v6,  v7,
            v8, v9, v10, v11, v12, v13, v14, v15
        );
    }
    int get0() const
    { return _mm512_cvtsi512_si32(val); }
};

struct v_uint64x8 : public v_512i
{
    OPENCV_HAL_IMPL_AVX512_TYPES_I(v_uint64x8)

    typedef uint64 lane_type;
    enum { nlanes = 8 };

    v_uint64x8(uint64 v0, uint64 v1, uint64 v2, uint64 v3,
               uint64 v4, uint64 v5, uint64 v6, uint64 v7)
    {
        val = _mm512_setr_epi64(
            (int64)v0, (int64)v1, (int64)v2, (int64)v3,
            (int64)v4, (int64)v5, (int64)v6, (int64)v7
        );
    }
    uint64 get0() const
    { return (uint64)_mm512_cvtsi512_si64(val); }
};

struct v_int64x8 : public v_512i
{
    OPENCV_HAL_IMPL_AVX512_TYPES_I(v_int64x8)

    typedef int64 lane_type;
    enum { nlanes = 8 };

    v_int64x8(int64 v0, int64 v1, int64 v2, int64 v3,
              int64 v4, int64 v5, int64 v6, int64 v7)
    {
        val = _mm512_setr_epi64(v0, v1, v2, v3, v4, v5, v6, v7);
    }
    int64 get0() const
    { return _mm512_cvtsi512_si64(val); }
};

class v_float32x16
{
private:
    __m512 val;
public:
    typedef float lane_type;
    enum { nlanes = 16 };

    v_float32x16()
    {}
    v_float32x16(const __m512& v)  : val(v)
    {}
    v_float32x16(const __m512d& v) : val(_mm512_castpd_ps(v))
    {}
    v_float32x16(const __m512i& v) : val(_mm512_castsi512_ps(v))
    {}

    v_float32x16& operator = (const __m512& v)
    { val = v; return *this; }
    v_float32x16& operator = (const __m512d& v)
    { val = _mm512_castpd_ps(v); return *this; }
    v_float32x16& operator = (const __m512i& v)
    { val = _mm512_castsi512_ps(v); return *this; }

    operator __m512() const
    { return val; }
    operator __m512d() const
    { return _mm512_castps_pd(val); }
    operator __m512i() const
    { return _mm512_castps_si512(val); }

    v_float32x16(float v0,  float v1,  float v2,  float v3,
                 float v4,  float v5,  float v6,  float v7,
                 float v8,  float v9,  float v10, float v11,
                 float v12, float v13, float v14, float v15)
    {
        val = _mm512_setr_ps(
            v0, v1, v2,  v3,  v4,  v5,  v6,  v7,
            v8, v9, v10, v11, v12, v13, v14, v15
        );
    }
    float get0() const
    { return _mm512_cvtss_f32(val); }
};

class v_float64x8
{
private:
    __m512d val;
public:
    typedef double lane_type;
    enum { nlanes = 8 };

    v_float64x8()
    {}
    v_float64x8(const __m512d& v) : val(v)
    {}
    v_float64x8(const __m512& v)  : val(_mm512_castps_pd(v))
    {}
    v_float64x8(const __m512i& v) : val(_mm512_castsi512_pd(v))
    {}

    v_float64x8& operator = (const __m512d& v)
    { val = v; return *this; }
    v_float64x8& operator = (const __m512& v)
    { val = _mm512_castps_pd(v); return *this; }
    v_float64x8& operator = (const __m512i& v)
    { val = _mm512_castsi512_pd(v); return *this; }

    operator __m512d() const
    { return val; }
    operator __m512()  const
    { return _mm512_castpd_ps(val); }
    operator __m512i() const
    { return _mm512_castpd_si512(val); }

    v_float64x8(double v0, double v1, double v2, double v3,
                double v4, double v5, double v6, double v7)
    {
        val = _mm512_setr_pd(v0, v1, v2, v3, v4, v5, v6, v7);
    }
    double get0() const
    { return _mm512_cvtsd_f64(val); }
};

class v_mask8x64
{
private:
    __mmask64 val;
public:
    v_mask8x64()
    {}
    v_mask8x64(const __mmask64& v) : val(v)
    {}

    v_mask8x64& operator = (const __mmask64& v)
    { val = v; return *this; }

    operator __mmask64() const
    { return val; }
};

class v_mask16x32
{
private:
    __mmask32 val;
public:
    v_mask16x32()
    {}
    v_mask16x32(const __mmask32& v) : val(v)
    {}

    v_mask16x32& operator = (const __mmask32& v)
    { val = v; return *this; }

    operator __mmask32() const
    { return val; }
};

class v_mask32x16
{
private:
    __mmask16 val;
public:
    v_mask32x16()
    {}
    v_mask32x16(const __mmask16& v) : val(v)
    {}

    v_mask32x16& operator = (const __mmask16& v)
    { val = v; return *this; }

    operator __mmask16() const
    { return val; }
};

class v_mask64x8
{
private:
    __mmask8 val;
public:
    v_mask64x8()
    {}
    v_mask64x8(const __mmask8& v) : val(v)
    {}

    v_mask64x8& operator = (const __mmask8& v)
    { val = v; return *this; }

    operator __mmask8() const
    { return val; }
};