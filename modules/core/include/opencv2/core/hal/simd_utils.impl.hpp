// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This header is not standalone. Don't include directly, use "intrin.hpp" instead.
#ifdef OPENCV_HAL_INTRIN_HPP  // defined in intrin.hpp


#if CV_SIMD128 || CV_SIMD128_CPP

template<typename _T> struct Type2Vec128_Traits;
#define CV_INTRIN_DEF_TYPE2VEC128_TRAITS(type_, vec_type_) \
    template<> struct Type2Vec128_Traits<type_> \
    { \
        typedef vec_type_ vec_type; \
    }

CV_INTRIN_DEF_TYPE2VEC128_TRAITS(uchar, v_uint8x16);
CV_INTRIN_DEF_TYPE2VEC128_TRAITS(schar, v_int8x16);
CV_INTRIN_DEF_TYPE2VEC128_TRAITS(ushort, v_uint16x8);
CV_INTRIN_DEF_TYPE2VEC128_TRAITS(short, v_int16x8);
CV_INTRIN_DEF_TYPE2VEC128_TRAITS(unsigned, v_uint32x4);
CV_INTRIN_DEF_TYPE2VEC128_TRAITS(int, v_int32x4);
CV_INTRIN_DEF_TYPE2VEC128_TRAITS(float, v_float32x4);
CV_INTRIN_DEF_TYPE2VEC128_TRAITS(uint64, v_uint64x2);
CV_INTRIN_DEF_TYPE2VEC128_TRAITS(int64, v_int64x2);
#if CV_SIMD128_64F
CV_INTRIN_DEF_TYPE2VEC128_TRAITS(double, v_float64x2);
#endif

template<typename _T> static inline
typename Type2Vec128_Traits<_T>::vec_type v_setall(const _T& a);

template<> inline Type2Vec128_Traits< uchar>::vec_type v_setall< uchar>(const  uchar& a) { return v_setall_u8(a); }
template<> inline Type2Vec128_Traits< schar>::vec_type v_setall< schar>(const  schar& a) { return v_setall_s8(a); }
template<> inline Type2Vec128_Traits<ushort>::vec_type v_setall<ushort>(const ushort& a) { return v_setall_u16(a); }
template<> inline Type2Vec128_Traits< short>::vec_type v_setall< short>(const  short& a) { return v_setall_s16(a); }
template<> inline Type2Vec128_Traits<  uint>::vec_type v_setall<  uint>(const   uint& a) { return v_setall_u32(a); }
template<> inline Type2Vec128_Traits<   int>::vec_type v_setall<   int>(const    int& a) { return v_setall_s32(a); }
template<> inline Type2Vec128_Traits<uint64>::vec_type v_setall<uint64>(const uint64& a) { return v_setall_u64(a); }
template<> inline Type2Vec128_Traits< int64>::vec_type v_setall< int64>(const  int64& a) { return v_setall_s64(a); }
template<> inline Type2Vec128_Traits< float>::vec_type v_setall< float>(const  float& a) { return v_setall_f32(a); }
#if CV_SIMD128_64F
template<> inline Type2Vec128_Traits<double>::vec_type v_setall<double>(const double& a) { return v_setall_f64(a); }
#endif

#endif  // SIMD128


#if CV_SIMD256

template<typename _T> struct Type2Vec256_Traits;
#define CV_INTRIN_DEF_TYPE2VEC256_TRAITS(type_, vec_type_) \
    template<> struct Type2Vec256_Traits<type_> \
    { \
        typedef vec_type_ vec_type; \
    }

CV_INTRIN_DEF_TYPE2VEC256_TRAITS(uchar, v_uint8x32);
CV_INTRIN_DEF_TYPE2VEC256_TRAITS(schar, v_int8x32);
CV_INTRIN_DEF_TYPE2VEC256_TRAITS(ushort, v_uint16x16);
CV_INTRIN_DEF_TYPE2VEC256_TRAITS(short, v_int16x16);
CV_INTRIN_DEF_TYPE2VEC256_TRAITS(unsigned, v_uint32x8);
CV_INTRIN_DEF_TYPE2VEC256_TRAITS(int, v_int32x8);
CV_INTRIN_DEF_TYPE2VEC256_TRAITS(float, v_float32x8);
CV_INTRIN_DEF_TYPE2VEC256_TRAITS(uint64, v_uint64x4);
CV_INTRIN_DEF_TYPE2VEC256_TRAITS(int64, v_int64x4);
#if CV_SIMD256_64F
CV_INTRIN_DEF_TYPE2VEC256_TRAITS(double, v_float64x4);
#endif

template<typename _T> static inline
typename Type2Vec256_Traits<_T>::vec_type v256_setall(const _T& a);

template<> inline Type2Vec256_Traits< uchar>::vec_type v256_setall< uchar>(const  uchar& a) { return v256_setall_u8(a); }
template<> inline Type2Vec256_Traits< schar>::vec_type v256_setall< schar>(const  schar& a) { return v256_setall_s8(a); }
template<> inline Type2Vec256_Traits<ushort>::vec_type v256_setall<ushort>(const ushort& a) { return v256_setall_u16(a); }
template<> inline Type2Vec256_Traits< short>::vec_type v256_setall< short>(const  short& a) { return v256_setall_s16(a); }
template<> inline Type2Vec256_Traits<  uint>::vec_type v256_setall<  uint>(const   uint& a) { return v256_setall_u32(a); }
template<> inline Type2Vec256_Traits<   int>::vec_type v256_setall<   int>(const    int& a) { return v256_setall_s32(a); }
template<> inline Type2Vec256_Traits<uint64>::vec_type v256_setall<uint64>(const uint64& a) { return v256_setall_u64(a); }
template<> inline Type2Vec256_Traits< int64>::vec_type v256_setall< int64>(const  int64& a) { return v256_setall_s64(a); }
template<> inline Type2Vec256_Traits< float>::vec_type v256_setall< float>(const  float& a) { return v256_setall_f32(a); }
#if CV_SIMD256_64F
template<> inline Type2Vec256_Traits<double>::vec_type v256_setall<double>(const double& a) { return v256_setall_f64(a); }
#endif

#endif  // SIMD256


#if CV_SIMD512

template<typename _T> struct Type2Vec512_Traits;
#define CV_INTRIN_DEF_TYPE2VEC512_TRAITS(type_, vec_type_) \
    template<> struct Type2Vec512_Traits<type_> \
    { \
        typedef vec_type_ vec_type; \
    }

CV_INTRIN_DEF_TYPE2VEC512_TRAITS(uchar, v_uint8x64);
CV_INTRIN_DEF_TYPE2VEC512_TRAITS(schar, v_int8x64);
CV_INTRIN_DEF_TYPE2VEC512_TRAITS(ushort, v_uint16x32);
CV_INTRIN_DEF_TYPE2VEC512_TRAITS(short, v_int16x32);
CV_INTRIN_DEF_TYPE2VEC512_TRAITS(unsigned, v_uint32x16);
CV_INTRIN_DEF_TYPE2VEC512_TRAITS(int, v_int32x16);
CV_INTRIN_DEF_TYPE2VEC512_TRAITS(float, v_float32x16);
CV_INTRIN_DEF_TYPE2VEC512_TRAITS(uint64, v_uint64x8);
CV_INTRIN_DEF_TYPE2VEC512_TRAITS(int64, v_int64x8);
#if CV_SIMD512_64F
CV_INTRIN_DEF_TYPE2VEC512_TRAITS(double, v_float64x8);
#endif

template<typename _T> static inline
typename Type2Vec512_Traits<_T>::vec_type v512_setall(const _T& a);

template<> inline Type2Vec512_Traits< uchar>::vec_type v512_setall< uchar>(const  uchar& a) { return v512_setall_u8(a); }
template<> inline Type2Vec512_Traits< schar>::vec_type v512_setall< schar>(const  schar& a) { return v512_setall_s8(a); }
template<> inline Type2Vec512_Traits<ushort>::vec_type v512_setall<ushort>(const ushort& a) { return v512_setall_u16(a); }
template<> inline Type2Vec512_Traits< short>::vec_type v512_setall< short>(const  short& a) { return v512_setall_s16(a); }
template<> inline Type2Vec512_Traits<  uint>::vec_type v512_setall<  uint>(const   uint& a) { return v512_setall_u32(a); }
template<> inline Type2Vec512_Traits<   int>::vec_type v512_setall<   int>(const    int& a) { return v512_setall_s32(a); }
template<> inline Type2Vec512_Traits<uint64>::vec_type v512_setall<uint64>(const uint64& a) { return v512_setall_u64(a); }
template<> inline Type2Vec512_Traits< int64>::vec_type v512_setall< int64>(const  int64& a) { return v512_setall_s64(a); }
template<> inline Type2Vec512_Traits< float>::vec_type v512_setall< float>(const  float& a) { return v512_setall_f32(a); }
#if CV_SIMD512_64F
template<> inline Type2Vec512_Traits<double>::vec_type v512_setall<double>(const double& a) { return v512_setall_f64(a); }
#endif

#endif  // SIMD512


#if CV_SIMD_WIDTH == 16
template<typename _T> static inline
typename Type2Vec128_Traits<_T>::vec_type vx_setall(const _T& a) { return v_setall(a); }
#elif CV_SIMD_WIDTH == 32
template<typename _T> static inline
typename Type2Vec256_Traits<_T>::vec_type vx_setall(const _T& a) { return v256_setall(a); }
#elif CV_SIMD_WIDTH == 64
template<typename _T> static inline
typename Type2Vec512_Traits<_T>::vec_type vx_setall(const _T& a) { return v512_setall(a); }
#else
#error "Build configuration error, unsupported CV_SIMD_WIDTH"
#endif


#endif  // OPENCV_HAL_INTRIN_HPP
