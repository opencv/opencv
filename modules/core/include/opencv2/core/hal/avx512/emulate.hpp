// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD512
    #error "Not a standalone header"
#endif

inline int _mm512_cvtsi512_si32(const __m512i& v)
{ return _mm_cvtsi128_si32(_mm512_castsi512_si128(v)); }

inline int64 _mm512_cvtsi512_si64(const __m512i& v)
{
#if defined(__x86_64__) || defined(_M_X64)
    return _mm_cvtsi128_si64(_mm512_castsi512_si128(v));
#else
    int a = _mm512_cvtsi512_si32(v);
    int b = _mm512_cvtsi512_si32(_mm512_srli_epi64(v, 32));
    return (int64)(a | ((uint64)b << 32));
#endif
}

// todo: check MVC and ICC

inline __m512i _mm512_setr_epi8(
    char v0,  char v1,  char v2,  char v3,
    char v4,  char v5,  char v6,  char v7,
    char v8,  char v9,  char v10, char v11,
    char v12, char v13, char v14, char v15,
    char v16, char v17, char v18, char v19,
    char v20, char v21, char v22, char v23,
    char v24, char v25, char v26, char v27,
    char v28, char v29, char v30, char v31,
    char v32, char v33, char v34, char v35,
    char v36, char v37, char v38, char v39,
    char v40, char v41, char v42, char v43,
    char v44, char v45, char v46, char v47,
    char v48, char v49, char v50, char v51,
    char v52, char v53, char v54, char v55,
    char v56, char v57, char v58, char v59,
    char v60, char v61, char v62, char v63
)
{
    return __extension__(__m512i)(__v64qi) {
        v0,  v1,  v2,  v3,  v4,  v5,
        v6 , v7,  v8,  v9,  v10, v11,
        v12, v13, v14, v15, v16, v17,
        v18, v19, v20, v21, v22, v23,
        v24, v25, v26, v27, v28, v29,
        v30, v31, v32, v33, v34, v35,
        v36, v37, v38, v39, v40, v41,
        v42, v43, v44, v45, v46, v47,
        v48, v49, v50, v51, v52, v53,
        v54, v55, v56, v57, v58, v59,
        v60, v61, v62, v63
    };
}

inline __m512i _mm512_setr_epi16(
    short v0,  short v1,  short v2,  short v3,
    short v4,  short v5,  short v6,  short v7,
    short v8,  short v9,  short v10, short v11,
    short v12, short v13, short v14, short v15,
    short v16, short v17, short v18, short v19,
    short v20, short v21, short v22, short v23,
    short v24, short v25, short v26, short v27,
    short v28, short v29, short v30, short v31
)
{
    return __extension__(__m512i)(__v32hi) {
        v0,  v1,  v2,  v3,  v4,  v5,
        v6 , v7,  v8,  v9,  v10, v11,
        v12, v13, v14, v15, v16, v17,
        v18, v19, v20, v21, v22, v23,
        v24, v25, v26, v27, v28, v29,
        v30, v31
    };
}