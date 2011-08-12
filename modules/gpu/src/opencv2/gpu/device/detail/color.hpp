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
// any express or bpied warranties, including, but not limited to, the bpied
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

#ifndef __OPENCV_GPU_COLOR_DETAIL_HPP__
#define __OPENCV_GPU_COLOR_DETAIL_HPP__

#include "../vec_traits.hpp"
#include "../saturate_cast.hpp"
#include "../limits.hpp"
#include "../functional.hpp"

#ifndef CV_DESCALE
    #define CV_DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))
#endif

namespace cv { namespace gpu { namespace device
{
    namespace detail
    {
        template<typename T> struct ColorChannel
        {
            typedef float worktype_f;
            static __device__ __forceinline__ T max() { return numeric_limits<T>::max(); }
            static __device__ __forceinline__ T half() { return (T)(max()/2 + 1); }
        };
        template<> struct ColorChannel<float>
        {
            typedef float worktype_f;
            static __device__ __forceinline__ float max() { return 1.f; }
            static __device__ __forceinline__ float half() { return 0.5f; }
        };

        template <typename T> static __device__ __forceinline__ void setAlpha(typename TypeVec<T, 3>::vec_type& vec, T val)
        {
        }
        template <typename T> static __device__ __forceinline__ void setAlpha(typename TypeVec<T, 4>::vec_type& vec, T val)
        {
            vec.w = val;
        }
        template <typename T> static __device__ __forceinline__ T getAlpha(const typename TypeVec<T, 3>::vec_type& vec)
        {
            return ColorChannel<T>::max();
        }
        template <typename T> static __device__ __forceinline__ T getAlpha(const typename TypeVec<T, 4>::vec_type& vec)
        {
            return vec.w;
        }

        enum
        {
            yuv_shift  = 14,
            xyz_shift  = 12,
            R2Y        = 4899,
            G2Y        = 9617,
            B2Y        = 1868,
            BLOCK_SIZE = 256
        };
    }

////////////////// Various 3/4-channel to 3/4-channel RGB transformations /////////////////

    namespace detail
    {
        template <typename T, typename D, int bidx> struct RGB2RGB : public unary_function<T, D>
        {
            __device__ D operator()(const T& src) const
            {
                D dst;
                dst.x = (&src.x)[bidx];
                dst.y = src.y;
                dst.z = (&src.x)[bidx^2];
                setAlpha(dst, getAlpha<typename VecTraits<T>::elem_type>(src));
                return dst;
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_RGB2RGB_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef detail::RGB2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx>(); \
        } \
    };

/////////// Transforming 16-bit (565 or 555) RGB to/from 24/32-bit (888[8]) RGB //////////

    namespace detail
    {
        template <int GREEN_BITS, int bidx> struct RGB2RGB5x5Converter;
        template<int bidx> struct RGB2RGB5x5Converter<6, bidx> 
        {
            template <typename T> static __device__ __forceinline__ ushort cvt(const T& src)
            {
                return (ushort)(((&src.x)[bidx] >> 3) | ((src.y & ~3) << 3) | (((&src.x)[bidx^2] & ~7) << 8));
            }
        };
        template<int bidx> struct RGB2RGB5x5Converter<5, bidx> 
        {
            static __device__ __forceinline__ ushort cvt(const uchar3& src)
            {
                return (ushort)(((&src.x)[bidx] >> 3) | ((src.y & ~7) << 2) | (((&src.x)[bidx^2] & ~7) << 7));
            }
            static __device__ __forceinline__ ushort cvt(const uchar4& src)
            {
                return (ushort)(((&src.x)[bidx] >> 3) | ((src.y & ~7) << 2) | (((&src.x)[bidx^2] & ~7) << 7) | (src.w ? 0x8000 : 0));
            }
        };

        template<typename T, int bidx, int GREEN_BITS> struct RGB2RGB5x5 : public unary_function<T, ushort>
        {
            __device__ __forceinline__ ushort operator()(const T& src) const
            {
                return RGB2RGB5x5Converter<GREEN_BITS, bidx>::cvt(src);
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_RGB2RGB5x5_TRAITS(name, scn, bidx, green_bits) \
    struct name ## _traits \
    { \
        typedef detail::RGB2RGB5x5<uchar ## scn, bidx, green_bits> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB2RGB5x5<uchar ## scn, bidx, green_bits>(); \
        } \
    };

    namespace detail
    {
        template <int GREEN_BITS, int bidx> struct RGB5x52RGBConverter;    
        template <int bidx> struct RGB5x52RGBConverter<5, bidx>
        {
            template <typename D> static __device__ void cvt(uint src, D& dst)
            {            
                (&dst.x)[bidx] = (uchar)(src << 3);
                dst.y = (uchar)((src >> 2) & ~7);
                (&dst.x)[bidx ^ 2] = (uchar)((src >> 7) & ~7);
                setAlpha(dst, (uchar)(src & 0x8000 ? 255 : 0));
            }
        };
        template <int bidx> struct RGB5x52RGBConverter<6, bidx>
        {
            template <typename D>
            static __device__ void cvt(uint src, D& dst)
            {            
                (&dst.x)[bidx] = (uchar)(src << 3);
                dst.y = (uchar)((src >> 3) & ~3);
                (&dst.x)[bidx ^ 2] = (uchar)((src >> 8) & ~7);
                setAlpha(dst, (uchar)(255));
            }
        };

        template <typename D, int bidx, int GREEN_BITS> struct RGB5x52RGB : public unary_function<ushort, D>
        {
            __device__ __forceinline__ D operator()(ushort src) const
            {
                D dst;
                RGB5x52RGBConverter<GREEN_BITS, bidx>::cvt((uint)src, dst);
                return dst;
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_RGB5x52RGB_TRAITS(name, dcn, bidx, green_bits) \
    struct name ## _traits \
    { \
        typedef detail::RGB5x52RGB<uchar ## dcn, bidx, green_bits> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB5x52RGB<uchar ## dcn, bidx, green_bits>(); \
        } \
    };

///////////////////////////////// Grayscale to Color ////////////////////////////////

    namespace detail
    {
        template <typename T, typename D> struct Gray2RGB : public unary_function<T, D>
        {
            __device__ __forceinline__ D operator()(const T& src) const
            {
                D dst;
                dst.z = dst.y = dst.x = src;            
                setAlpha(dst, ColorChannel<typename VecTraits<T>::elem_type>::max());
                return dst;
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_GRAY2RGB_TRAITS(name, dcn) \
    template <typename T> struct name ## _traits \
    { \
        typedef detail::Gray2RGB<T, typename TypeVec<T, dcn>::vec_type> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::Gray2RGB<T, typename TypeVec<T, dcn>::vec_type>(); \
        } \
    };

    namespace detail
    {
        template <int GREEN_BITS> struct Gray2RGB5x5Converter;
        template<> struct Gray2RGB5x5Converter<6> 
        {
            static __device__ __forceinline__ ushort cvt(uint t)
            {
                return (ushort)((t >> 3) | ((t & ~3) << 3) | ((t & ~7) << 8));
            }
        };
        template<> struct Gray2RGB5x5Converter<5> 
        {
            static __device__ __forceinline__ ushort cvt(uint t)
            {
                t >>= 3;
                return (ushort)(t | (t << 5) | (t << 10));
            }
        };

        template<int GREEN_BITS> struct Gray2RGB5x5 : public unary_function<uchar, ushort>
        {
            __device__ __forceinline__ ushort operator()(uchar src) const
            {
                return Gray2RGB5x5Converter<GREEN_BITS>::cvt((uint)src);
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_GRAY2RGB5x5_TRAITS(name, green_bits) \
    struct name ## _traits \
    { \
        typedef detail::Gray2RGB5x5<green_bits> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::Gray2RGB5x5<green_bits>(); \
        } \
    };

///////////////////////////////// Color to Grayscale ////////////////////////////////

    namespace detail
    {
        template <int GREEN_BITS> struct RGB5x52GrayConverter;
        template <> struct RGB5x52GrayConverter<6> 
        {
            static __device__ __forceinline__ uchar cvt(uint t)
            {
                return (uchar)CV_DESCALE(((t << 3) & 0xf8) * B2Y + ((t >> 3) & 0xfc) * G2Y + ((t >> 8) & 0xf8) * R2Y, yuv_shift);
            }
        };
        template <> struct RGB5x52GrayConverter<5> 
        {
            static __device__ __forceinline__ uchar cvt(uint t)
            {
                return (uchar)CV_DESCALE(((t << 3) & 0xf8) * B2Y + ((t >> 2) & 0xf8) * G2Y + ((t >> 7) & 0xf8) * R2Y, yuv_shift);
            }
        };   

        template<int GREEN_BITS> struct RGB5x52Gray : public unary_function<ushort, uchar>
        {
            __device__ __forceinline__ uchar operator()(ushort src) const
            {
                return RGB5x52GrayConverter<GREEN_BITS>::cvt((uint)src);
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_RGB5x52GRAY_TRAITS(name, green_bits) \
    struct name ## _traits \
    { \
        typedef detail::RGB5x52Gray<green_bits> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB5x52Gray<green_bits>(); \
        } \
    };

    namespace detail
    {
        template <int bidx, typename T> static __device__ __forceinline__ T RGB2GrayConvert(const T* src)
        {
            return (T)CV_DESCALE((unsigned)(src[bidx] * B2Y + src[1] * G2Y + src[bidx^2] * R2Y), yuv_shift);
        }
        template <int bidx> static __device__ __forceinline__ float RGB2GrayConvert(const float* src)
        {
            return src[bidx] * 0.114f + src[1] * 0.587f + src[bidx^2] * 0.299f;
        }

        template <typename T, typename D, int bidx> struct RGB2Gray : public unary_function<T, D>
        {
            __device__ __forceinline__ D operator()(const T& src) const
            {
                return RGB2GrayConvert<bidx>(&src.x);
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_RGB2GRAY_TRAITS(name, scn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef detail::RGB2Gray<typename TypeVec<T, scn>::vec_type, T, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB2Gray<typename TypeVec<T, scn>::vec_type, T, bidx>(); \
        } \
    };

///////////////////////////////////// RGB <-> YUV //////////////////////////////////////

    namespace detail
    {
        __constant__ float c_RGB2YUVCoeffs_f[5] = { 0.114f, 0.587f, 0.299f, 0.492f, 0.877f };
        __constant__ int   c_RGB2YUVCoeffs_i[5] = { B2Y, G2Y, R2Y, 8061, 14369 };

        template <int bidx, typename T, typename D> static __device__ void RGB2YUVConvert(const T* src, D& dst)
        {
            const int delta = ColorChannel<T>::half() * (1 << yuv_shift);

            const int Y = CV_DESCALE(src[0] * c_RGB2YUVCoeffs_i[bidx^2] + src[1] * c_RGB2YUVCoeffs_i[1] + src[2] * c_RGB2YUVCoeffs_i[bidx], yuv_shift);
            const int Cr = CV_DESCALE((src[bidx^2] - Y) * c_RGB2YUVCoeffs_i[3] + delta, yuv_shift);
            const int Cb = CV_DESCALE((src[bidx] - Y) * c_RGB2YUVCoeffs_i[4] + delta, yuv_shift);

            dst.x = saturate_cast<T>(Y);
            dst.y = saturate_cast<T>(Cr);
            dst.z = saturate_cast<T>(Cb);
        }
        template <int bidx, typename D> static __device__ __forceinline__ void RGB2YUVConvert(const float* src, D& dst)
        {
            dst.x = src[0] * c_RGB2YUVCoeffs_f[bidx^2] + src[1] * c_RGB2YUVCoeffs_f[1] + src[2] * c_RGB2YUVCoeffs_f[bidx];
            dst.y = (src[bidx^2] - dst.x) * c_RGB2YUVCoeffs_f[3] + ColorChannel<float>::half();
            dst.z = (src[bidx] - dst.x) * c_RGB2YUVCoeffs_f[4] + ColorChannel<float>::half();
        }

        template <typename T, typename D, int bidx> struct RGB2YUV : public unary_function<T, D>
        {
            __device__ __forceinline__ D operator ()(const T& src) const
            {
                D dst;
                RGB2YUVConvert<bidx>(&src.x, dst);
                return dst;
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_RGB2YUV_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef detail::RGB2YUV<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB2YUV<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx>(); \
        } \
    };

    namespace detail
    {
        __constant__ float c_YUV2RGBCoeffs_f[5] = { 2.032f, -0.395f, -0.581f, 1.140f };
        __constant__ int   c_YUV2RGBCoeffs_i[5] = { 33292, -6472, -9519, 18678 }; 

        template <int bidx, typename T, typename D> static __device__ void YUV2RGBConvert(const T& src, D* dst)
        {
            const int b = src.x + CV_DESCALE((src.z - ColorChannel<D>::half()) * c_YUV2RGBCoeffs_i[3], yuv_shift);
            const int g = src.x + CV_DESCALE((src.z - ColorChannel<D>::half()) * c_YUV2RGBCoeffs_i[2] + (src.y - ColorChannel<D>::half()) * c_YUV2RGBCoeffs_i[1], yuv_shift);
            const int r = src.x + CV_DESCALE((src.y - ColorChannel<D>::half()) * c_YUV2RGBCoeffs_i[0], yuv_shift);

            dst[bidx] = saturate_cast<D>(b);
            dst[1] = saturate_cast<D>(g);
            dst[bidx^2] = saturate_cast<D>(r);
        }
        template <int bidx, typename T> static __device__ __forceinline__ void YUV2RGBConvert(const T& src, float* dst)
        {
            dst[bidx] = src.x + (src.z - ColorChannel<float>::half()) * c_YUV2RGBCoeffs_f[3];
            dst[1] = src.x + (src.z - ColorChannel<float>::half()) * c_YUV2RGBCoeffs_f[2] + (src.y - ColorChannel<float>::half()) * c_YUV2RGBCoeffs_f[1];
            dst[bidx^2] = src.x + (src.y - ColorChannel<float>::half()) * c_YUV2RGBCoeffs_f[0];
        }

        template <typename T, typename D, int bidx> struct YUV2RGB : public unary_function<T, D>
        {
            __device__ __forceinline__ D operator ()(const T& src) const
            {
                D dst;

                YUV2RGBConvert<bidx>(src, &dst.x);
                setAlpha(dst, ColorChannel<typename VecTraits<T>::elem_type>::max());

                return dst;
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_YUV2RGB_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef detail::YUV2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::YUV2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx>(); \
        } \
    };

///////////////////////////////////// RGB <-> YCrCb //////////////////////////////////////
    
    namespace detail
    {
        __constant__ float c_RGB2YCrCbCoeffs_f[5] = {0.299f, 0.587f, 0.114f, 0.713f, 0.564f};
        __constant__ int   c_RGB2YCrCbCoeffs_i[5] = {R2Y, G2Y, B2Y, 11682, 9241};
        
        template <int bidx, typename T, typename D> static __device__ void RGB2YCrCbConvert(const T* src, D& dst)
        {
            const int delta = ColorChannel<T>::half() * (1 << yuv_shift);

            const int Y = CV_DESCALE(src[0] * c_RGB2YCrCbCoeffs_i[bidx^2] + src[1] * c_RGB2YCrCbCoeffs_i[1] + src[2] * c_RGB2YCrCbCoeffs_i[bidx], yuv_shift);
            const int Cr = CV_DESCALE((src[bidx^2] - Y) * c_RGB2YCrCbCoeffs_i[3] + delta, yuv_shift);
            const int Cb = CV_DESCALE((src[bidx] - Y) * c_RGB2YCrCbCoeffs_i[4] + delta, yuv_shift);

            dst.x = saturate_cast<T>(Y);
            dst.y = saturate_cast<T>(Cr);
            dst.z = saturate_cast<T>(Cb);
        }
        template <int bidx, typename D> static __device__ __forceinline__ void RGB2YCrCbConvert(const float* src, D& dst)
        {
            dst.x = src[0] * c_RGB2YCrCbCoeffs_f[bidx^2] + src[1] * c_RGB2YCrCbCoeffs_f[1] + src[2] * c_RGB2YCrCbCoeffs_f[bidx];
            dst.y = (src[bidx^2] - dst.x) * c_RGB2YCrCbCoeffs_f[3] + ColorChannel<float>::half();
            dst.z = (src[bidx] - dst.x) * c_RGB2YCrCbCoeffs_f[4] + ColorChannel<float>::half();
        }

        template <typename T, typename D, int bidx> struct RGB2YCrCb : public unary_function<T, D>
        {
            __device__ __forceinline__ D operator ()(const T& src) const
            {
                D dst;
                RGB2YCrCbConvert<bidx>(&src.x, dst);
                return dst;
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_RGB2YCrCb_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef detail::RGB2YCrCb<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB2YCrCb<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx>(); \
        } \
    };

    namespace detail
    {
        __constant__ float c_YCrCb2RGBCoeffs_f[5] = {1.403f, -0.714f, -0.344f, 1.773f};
        __constant__ int   c_YCrCb2RGBCoeffs_i[5] = {22987, -11698, -5636, 29049};

        template <int bidx, typename T, typename D> static __device__ void YCrCb2RGBConvert(const T& src, D* dst)
        {
            const int b = src.x + CV_DESCALE((src.z - ColorChannel<D>::half()) * c_YCrCb2RGBCoeffs_i[3], yuv_shift);
            const int g = src.x + CV_DESCALE((src.z - ColorChannel<D>::half()) * c_YCrCb2RGBCoeffs_i[2] + (src.y - ColorChannel<D>::half()) * c_YCrCb2RGBCoeffs_i[1], yuv_shift);
            const int r = src.x + CV_DESCALE((src.y - ColorChannel<D>::half()) * c_YCrCb2RGBCoeffs_i[0], yuv_shift);

            dst[bidx] = saturate_cast<D>(b);
            dst[1] = saturate_cast<D>(g);
            dst[bidx^2] = saturate_cast<D>(r);
        }
        template <int bidx, typename T> __device__ __forceinline__ void YCrCb2RGBConvert(const T& src, float* dst)
        {
            dst[bidx] = src.x + (src.z - ColorChannel<float>::half()) * c_YCrCb2RGBCoeffs_f[3];
            dst[1] = src.x + (src.z - ColorChannel<float>::half()) * c_YCrCb2RGBCoeffs_f[2] + (src.y - ColorChannel<float>::half()) * c_YCrCb2RGBCoeffs_f[1];
            dst[bidx^2] = src.x + (src.y - ColorChannel<float>::half()) * c_YCrCb2RGBCoeffs_f[0];
        }

        template <typename T, typename D, int bidx> struct YCrCb2RGB : public unary_function<T, D>
        {
            __device__ __forceinline__ D operator ()(const T& src) const
            {
                D dst;

                YCrCb2RGBConvert<bidx>(src, &dst.x);
                setAlpha(dst, ColorChannel<typename VecTraits<T>::elem_type>::max());

                return dst;
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_YCrCb2RGB_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef detail::YCrCb2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::YCrCb2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx>(); \
        } \
    };

////////////////////////////////////// RGB <-> XYZ ///////////////////////////////////////

    namespace detail
    {
        __constant__ float c_RGB2XYZ_D65f[9] = { 0.412453f, 0.357580f, 0.180423f, 0.212671f, 0.715160f, 0.072169f, 0.019334f, 0.119193f, 0.950227f };
        __constant__ int   c_RGB2XYZ_D65i[9] = { 1689, 1465, 739, 871, 2929, 296, 79, 488, 3892 };

        template <int bidx, typename T, typename D> static __device__ __forceinline__ void RGB2XYZConvert(const T* src, D& dst)
        {
            dst.x = saturate_cast<T>(CV_DESCALE(src[bidx^2] * c_RGB2XYZ_D65i[0] + src[1] * c_RGB2XYZ_D65i[1] + src[bidx] * c_RGB2XYZ_D65i[2], xyz_shift));
            dst.y = saturate_cast<T>(CV_DESCALE(src[bidx^2] * c_RGB2XYZ_D65i[3] + src[1] * c_RGB2XYZ_D65i[4] + src[bidx] * c_RGB2XYZ_D65i[5], xyz_shift));
            dst.z = saturate_cast<T>(CV_DESCALE(src[bidx^2] * c_RGB2XYZ_D65i[6] + src[1] * c_RGB2XYZ_D65i[7] + src[bidx] * c_RGB2XYZ_D65i[8], xyz_shift));
        }
        template <int bidx, typename D> static __device__ __forceinline__ void RGB2XYZConvert(const float* src, D& dst)
        {
            dst.x = src[bidx^2] * c_RGB2XYZ_D65f[0] + src[1] * c_RGB2XYZ_D65f[1] + src[bidx] * c_RGB2XYZ_D65f[2];
            dst.y = src[bidx^2] * c_RGB2XYZ_D65f[3] + src[1] * c_RGB2XYZ_D65f[4] + src[bidx] * c_RGB2XYZ_D65f[5];
            dst.z = src[bidx^2] * c_RGB2XYZ_D65f[6] + src[1] * c_RGB2XYZ_D65f[7] + src[bidx] * c_RGB2XYZ_D65f[8];
        }

        template <typename T, typename D, int bidx> struct RGB2XYZ : public unary_function<T, D>
        {
            __device__ __forceinline__ D operator()(const T& src) const
            {
                D dst;
                RGB2XYZConvert<bidx>(&src.x, dst);
                return dst;
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_RGB2XYZ_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef detail::RGB2XYZ<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB2XYZ<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx>(); \
        } \
    };

    namespace detail
    {
        __constant__ float c_XYZ2sRGB_D65f[9] = { 3.240479f, -1.53715f, -0.498535f, -0.969256f, 1.875991f, 0.041556f, 0.055648f, -0.204043f, 1.057311f };
        __constant__ int   c_XYZ2sRGB_D65i[9] = { 13273, -6296, -2042, -3970, 7684, 170, 228, -836, 4331 };

        template <int bidx, typename T, typename D> static __device__ __forceinline__ void XYZ2RGBConvert(const T& src, D* dst)
        {
            dst[bidx^2] = saturate_cast<D>(CV_DESCALE(src.x * c_XYZ2sRGB_D65i[0] + src.y * c_XYZ2sRGB_D65i[1] + src.z * c_XYZ2sRGB_D65i[2], xyz_shift));
	        dst[1]      = saturate_cast<D>(CV_DESCALE(src.x * c_XYZ2sRGB_D65i[3] + src.y * c_XYZ2sRGB_D65i[4] + src.z * c_XYZ2sRGB_D65i[5], xyz_shift));
	        dst[bidx]   = saturate_cast<D>(CV_DESCALE(src.x * c_XYZ2sRGB_D65i[6] + src.y * c_XYZ2sRGB_D65i[7] + src.z * c_XYZ2sRGB_D65i[8], xyz_shift));
        }
        template <int bidx, typename T> static __device__ __forceinline__ void XYZ2RGBConvert(const T& src, float* dst)
        {
            dst[bidx^2] = src.x * c_XYZ2sRGB_D65f[0] + src.y * c_XYZ2sRGB_D65f[1] + src.z * c_XYZ2sRGB_D65f[2];
	        dst[1]      = src.x * c_XYZ2sRGB_D65f[3] + src.y * c_XYZ2sRGB_D65f[4] + src.z * c_XYZ2sRGB_D65f[5];
	        dst[bidx]   = src.x * c_XYZ2sRGB_D65f[6] + src.y * c_XYZ2sRGB_D65f[7] + src.z * c_XYZ2sRGB_D65f[8];
        }

        template <typename T, typename D, int bidx> struct XYZ2RGB : public unary_function<T, D>
        {
            __device__ __forceinline__ D operator()(const T& src) const
            {
                D dst;
                XYZ2RGBConvert<bidx>(src, &dst.x);
                setAlpha(dst, ColorChannel<typename VecTraits<T>::elem_type>::max());
                return dst;
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_XYZ2RGB_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef detail::XYZ2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::XYZ2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx>(); \
        } \
    };

////////////////////////////////////// RGB <-> HSV ///////////////////////////////////////

    namespace detail
    {
        __constant__ int c_HsvDivTable   [256] = {0, 1044480, 522240, 348160, 261120, 208896, 174080, 149211, 130560, 116053, 104448, 94953, 87040, 80345, 74606, 69632, 65280, 61440, 58027, 54973, 52224, 49737, 47476, 45412, 43520, 41779, 40172, 38684, 37303, 36017, 34816, 33693, 32640, 31651, 30720, 29842, 29013, 28229, 27486, 26782, 26112, 25475, 24869, 24290, 23738, 23211, 22706, 22223, 21760, 21316, 20890, 20480, 20086, 19707, 19342, 18991, 18651, 18324, 18008, 17703, 17408, 17123, 16846, 16579, 16320, 16069, 15825, 15589, 15360, 15137, 14921, 14711, 14507, 14308, 14115, 13926, 13743, 13565, 13391, 13221, 13056, 12895, 12738, 12584, 12434, 12288, 12145, 12006, 11869, 11736, 11605, 11478, 11353, 11231, 11111, 10995, 10880, 10768, 10658, 10550, 10445, 10341, 10240, 10141, 10043, 9947, 9854, 9761, 9671, 9582, 9495, 9410, 9326, 9243, 9162, 9082, 9004, 8927, 8852, 8777, 8704, 8632, 8561, 8492, 8423, 8356, 8290, 8224, 8160, 8097, 8034, 7973, 7913, 7853, 7795, 7737, 7680, 7624, 7569, 7514, 7461, 7408, 7355, 7304, 7253, 7203, 7154, 7105, 7057, 7010, 6963, 6917, 6872, 6827, 6782, 6739, 6695, 6653, 6611, 6569, 6528, 6487, 6447, 6408, 6369, 6330, 6292, 6254, 6217, 6180, 6144, 6108, 6073, 6037, 6003, 5968, 5935, 5901, 5868, 5835, 5803, 5771, 5739, 5708, 5677, 5646, 5615, 5585, 5556, 5526, 5497, 5468, 5440, 5412, 5384, 5356, 5329, 5302, 5275, 5249, 5222, 5196, 5171, 5145, 5120, 5095, 5070, 5046, 5022, 4998, 4974, 4950, 4927, 4904, 4881, 4858, 4836, 4813, 4791, 4769, 4748, 4726, 4705, 4684, 4663, 4642, 4622, 4601, 4581, 4561, 4541, 4522, 4502, 4483, 4464, 4445, 4426, 4407, 4389, 4370, 4352, 4334, 4316, 4298, 4281, 4263, 4246, 4229, 4212, 4195, 4178, 4161, 4145, 4128, 4112, 4096};
        __constant__ int c_HsvDivTable180[256] = {0, 122880, 61440, 40960, 30720, 24576, 20480, 17554, 15360, 13653, 12288, 11171, 10240, 9452, 8777, 8192, 7680, 7228, 6827, 6467, 6144, 5851, 5585, 5343, 5120, 4915, 4726, 4551, 4389, 4237, 4096, 3964, 3840, 3724, 3614, 3511, 3413, 3321, 3234, 3151, 3072, 2997, 2926, 2858, 2793, 2731, 2671, 2614, 2560, 2508, 2458, 2409, 2363, 2318, 2276, 2234, 2194, 2156, 2119, 2083, 2048, 2014, 1982, 1950, 1920, 1890, 1862, 1834, 1807, 1781, 1755, 1731, 1707, 1683, 1661, 1638, 1617, 1596, 1575, 1555, 1536, 1517, 1499, 1480, 1463, 1446, 1429, 1412, 1396, 1381, 1365, 1350, 1336, 1321, 1307, 1293, 1280, 1267, 1254, 1241, 1229, 1217, 1205, 1193, 1182, 1170, 1159, 1148, 1138, 1127, 1117, 1107, 1097, 1087, 1078, 1069, 1059, 1050, 1041, 1033, 1024, 1016, 1007, 999, 991, 983, 975, 968, 960, 953, 945, 938, 931, 924, 917, 910, 904, 897, 890, 884, 878, 871, 865, 859, 853, 847, 842, 836, 830, 825, 819, 814, 808, 803, 798, 793, 788, 783, 778, 773, 768, 763, 759, 754, 749, 745, 740, 736, 731, 727, 723, 719, 714, 710, 706, 702, 698, 694, 690, 686, 683, 679, 675, 671, 668, 664, 661, 657, 654, 650, 647, 643, 640, 637, 633, 630, 627, 624, 621, 617, 614, 611, 608, 605, 602, 599, 597, 594, 591, 588, 585, 582, 580, 577, 574, 572, 569, 566, 564, 561, 559, 556, 554, 551, 549, 546, 544, 541, 539, 537, 534, 532, 530, 527, 525, 523, 521, 518, 516, 514, 512, 510, 508, 506, 504, 502, 500, 497, 495, 493, 492, 490, 488, 486, 484, 482};
        __constant__ int c_HsvDivTable256[256] = {0, 174763, 87381, 58254, 43691, 34953, 29127, 24966, 21845, 19418, 17476, 15888, 14564, 13443, 12483, 11651, 10923, 10280, 9709, 9198, 8738, 8322, 7944, 7598, 7282, 6991, 6722, 6473, 6242, 6026, 5825, 5638, 5461, 5296, 5140, 4993, 4855, 4723, 4599, 4481, 4369, 4263, 4161, 4064, 3972, 3884, 3799, 3718, 3641, 3567, 3495, 3427, 3361, 3297, 3236, 3178, 3121, 3066, 3013, 2962, 2913, 2865, 2819, 2774, 2731, 2689, 2648, 2608, 2570, 2533, 2497, 2461, 2427, 2394, 2362, 2330, 2300, 2270, 2241, 2212, 2185, 2158, 2131, 2106, 2081, 2056, 2032, 2009, 1986, 1964, 1942, 1920, 1900, 1879, 1859, 1840, 1820, 1802, 1783, 1765, 1748, 1730, 1713, 1697, 1680, 1664, 1649, 1633, 1618, 1603, 1589, 1574, 1560, 1547, 1533, 1520, 1507, 1494, 1481, 1469, 1456, 1444, 1432, 1421, 1409, 1398, 1387, 1376, 1365, 1355, 1344, 1334, 1324, 1314, 1304, 1295, 1285, 1276, 1266, 1257, 1248, 1239, 1231, 1222, 1214, 1205, 1197, 1189, 1181, 1173, 1165, 1157, 1150, 1142, 1135, 1128, 1120, 1113, 1106, 1099, 1092, 1085, 1079, 1072, 1066, 1059, 1053, 1046, 1040, 1034, 1028, 1022, 1016, 1010, 1004, 999, 993, 987, 982, 976, 971, 966, 960, 955, 950, 945, 940, 935, 930, 925, 920, 915, 910, 906, 901, 896, 892, 887, 883, 878, 874, 869, 865, 861, 857, 853, 848, 844, 840, 836, 832, 828, 824, 820, 817, 813, 809, 805, 802, 798, 794, 791, 787, 784, 780, 777, 773, 770, 767, 763, 760, 757, 753, 750, 747, 744, 741, 737, 734, 731, 728, 725, 722, 719, 716, 713, 710, 708, 705, 702, 699, 696, 694, 691, 688, 685};
        
        template <int bidx, int hr, typename D> static __device__ void RGB2HSVConvert(const uchar* src, D& dst)
        {
            const int hsv_shift = 12;
            const int* hdiv_table = hr == 180 ? c_HsvDivTable180 : c_HsvDivTable256;

            int b = src[bidx], g = src[1], r = src[bidx^2];
            int h, s, v = b;
            int vmin = b, diff;
            int vr, vg;

            v = ::max(v, g);
            v = ::max(v, r);
            vmin = ::min(vmin, g);
            vmin = ::min(vmin, r);

            diff = v - vmin;
            vr = v == r ? -1 : 0;
            vg = v == g ? -1 : 0;

            s = (diff * c_HsvDivTable[v] + (1 << (hsv_shift-1))) >> hsv_shift;
            h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
            h = (h * hdiv_table[diff] + (1 << (hsv_shift-1))) >> hsv_shift;
            h += h < 0 ? hr : 0;

            dst.x = saturate_cast<uchar>(h);
            dst.y = (uchar)s;
            dst.z = (uchar)v;
        }
        template <int bidx, int hr, typename D> static __device__ void RGB2HSVConvert(const float* src, D& dst)
        {
            const float hscale = hr * (1.f / 360.f);

            float b = src[bidx], g = src[1], r = src[bidx^2];
            float h, s, v;

            float vmin, diff;

            v = vmin = r;
            v = fmax(v, g);
            v = fmax(v, b);
            vmin = fmin(vmin, g);
            vmin = fmin(vmin, b);

            diff = v - vmin;
            s = diff / (float)(::fabs(v) + numeric_limits<float>::epsilon());
            diff = (float)(60. / (diff + numeric_limits<float>::epsilon()));

            if (v == r)
                h = (g - b) * diff;
            else if (v == g)
                h = (b - r) * diff + 120.f;
            else
                h = (r - g) * diff + 240.f;

            if (h < 0) h += 360.f;

            dst.x = h * hscale;
            dst.y = s;
            dst.z = v;
        }

        template <typename T, typename D, int bidx, int hr> struct RGB2HSV : public unary_function<T, D>
        {
            __device__ __forceinline__ D operator()(const T& src) const
            {
                D dst;
                RGB2HSVConvert<bidx, hr>(&src.x, dst);
                return dst;
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_RGB2HSV_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef detail::RGB2HSV<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 180> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB2HSV<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 180>(); \
        } \
    }; \
    template <typename T> struct name ## _full_traits \
    { \
        typedef detail::RGB2HSV<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 256> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB2HSV<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 256>(); \
        } \
    }; \
    template <> struct name ## _traits<float> \
    { \
        typedef detail::RGB2HSV<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB2HSV<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360>(); \
        } \
    }; \
    template <> struct name ## _full_traits<float> \
    { \
        typedef detail::RGB2HSV<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB2HSV<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360>(); \
        } \
    };

    namespace detail
    {
        __constant__ int c_HsvSectorData[6][3] = { {1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0} };

        template <int bidx, int HR, typename T> static __device__ void HSV2RGBConvert(const T& src, float* dst)
        {
            const float hscale = 6.f / HR;
            
            float h = src.x, s = src.y, v = src.z;
            float b, g, r;

            if( s == 0 )
                b = g = r = v;
            else
            {
                float tab[4];
                int sector;
                h *= hscale;
                if( h < 0 )
                    do h += 6; while( h < 0 );
                else if( h >= 6 )
                    do h -= 6; while( h >= 6 );
                sector = __float2int_rd(h);
                h -= sector;

                tab[0] = v;
                tab[1] = v*(1.f - s);
                tab[2] = v*(1.f - s*h);
                tab[3] = v*(1.f - s*(1.f - h));

                b = tab[c_HsvSectorData[sector][0]];
                g = tab[c_HsvSectorData[sector][1]];
                r = tab[c_HsvSectorData[sector][2]];
            }

            dst[bidx] = b;
            dst[1] = g;
            dst[bidx^2] = r;
        }
        template <int bidx, int HR, typename T> static __device__ void HSV2RGBConvert(const T& src, uchar* dst)
        {
            float3 buf;

            buf.x = src.x;
            buf.y = src.y * (1.f/255.f);
            buf.z = src.z * (1.f/255.f);

            HSV2RGBConvert<bidx, HR>(buf, &buf.x);

            dst[0] = saturate_cast<uchar>(buf.x * 255.f);
            dst[1] = saturate_cast<uchar>(buf.y * 255.f);
            dst[2] = saturate_cast<uchar>(buf.z * 255.f);
        }

        template <typename T, typename D, int bidx, int HR> struct HSV2RGB : public unary_function<T, D>
        {
            __device__ __forceinline__ D operator()(const T& src) const
            {
                D dst;
                HSV2RGBConvert<bidx, HR>(src, &dst.x);
                setAlpha(dst, ColorChannel<typename VecTraits<T>::elem_type>::max());
                return dst;
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_HSV2RGB_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef detail::HSV2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 180> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::HSV2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 180>(); \
        } \
    }; \
    template <typename T> struct name ## _full_traits \
    { \
        typedef detail::HSV2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 255> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::HSV2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 255>(); \
        } \
    }; \
    template <> struct name ## _traits<float> \
    { \
        typedef detail::HSV2RGB<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::HSV2RGB<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360>(); \
        } \
    }; \
    template <> struct name ## _full_traits<float> \
    { \
        typedef detail::HSV2RGB<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::HSV2RGB<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360>(); \
        } \
    };

/////////////////////////////////////// RGB <-> HLS ////////////////////////////////////////

    namespace detail
    {
        template <int bidx, int hr, typename D> static __device__ void RGB2HLSConvert(const float* src, D& dst)
        {
            const float hscale = hr * (1.f / 360.f);

            float b = src[bidx], g = src[1], r = src[bidx^2];
            float h = 0.f, s = 0.f, l;
            float vmin, vmax, diff;

            vmax = vmin = r;
            vmax = fmax(vmax, g);
            vmax = fmax(vmax, b);
            vmin = fmin(vmin, g);
            vmin = fmin(vmin, b);

            diff = vmax - vmin;
            l = (vmax + vmin) * 0.5f;

            if (diff > numeric_limits<float>::epsilon())
            {
                s = l < 0.5f ? diff / (vmax + vmin) : diff / (2.0f - vmax - vmin);
                diff = 60.f / diff;

                if (vmax == r)
                    h = (g - b)*diff;
                else if (vmax == g)
                    h = (b - r)*diff + 120.f;
                else
                    h = (r - g)*diff + 240.f;

                if (h < 0.f) h += 360.f;
            }

            dst.x = h * hscale;
            dst.y = l;
            dst.z = s;
        }
        template <int bidx, int hr, typename D> static __device__ void RGB2HLSConvert(const uchar* src, D& dst)
        {
            float3 buf;

            buf.x = src[0]*(1.f/255.f);
            buf.y = src[1]*(1.f/255.f);
            buf.z = src[2]*(1.f/255.f);

            RGB2HLSConvert<bidx, hr>(&buf.x, buf);

            dst.x = saturate_cast<uchar>(buf.x);
            dst.y = saturate_cast<uchar>(buf.y*255.f);
            dst.z = saturate_cast<uchar>(buf.z*255.f);
        }

        template <typename T, typename D, int bidx, int hr> struct RGB2HLS : public unary_function<T, D>
        {
            __device__ __forceinline__ D operator()(const T& src) const
            {
                D dst;
                RGB2HLSConvert<bidx, hr>(&src.x, dst);
                return dst;
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_RGB2HLS_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef detail::RGB2HLS<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 180> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB2HLS<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 180>(); \
        } \
    }; \
    template <typename T> struct name ## _full_traits \
    { \
        typedef detail::RGB2HLS<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 256> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB2HLS<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 256>(); \
        } \
    }; \
    template <> struct name ## _traits<float> \
    { \
        typedef detail::RGB2HLS<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB2HLS<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360>(); \
        } \
    }; \
    template <> struct name ## _full_traits<float> \
    { \
        typedef detail::RGB2HLS<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::RGB2HLS<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360>(); \
        } \
    };

    namespace detail
    {
        __constant__ int c_HlsSectorData[6][3] = { {1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0} };

        template <int bidx, int HR, typename T> static __device__ void HLS2RGBConvert(const T& src, float* dst)
        {
            const float hscale = 6.0f / HR;

            float h = src.x, l = src.y, s = src.z;
            float b, g, r;

            if (s == 0)
                b = g = r = l;
            else
            {
                float tab[4];
                int sector;

                float p2 = l <= 0.5f ? l * (1 + s) : l + s - l * s;
                float p1 = 2 * l - p2;

                h *= hscale;

                if( h < 0 )
                    do h += 6; while( h < 0 );
                else if( h >= 6 )
                    do h -= 6; while( h >= 6 );

                sector = __float2int_rd(h);
                h -= sector;

                tab[0] = p2;
                tab[1] = p1;
                tab[2] = p1 + (p2 - p1) * (1 - h);
                tab[3] = p1 + (p2 - p1) * h;

                b = tab[c_HlsSectorData[sector][0]];
                g = tab[c_HlsSectorData[sector][1]];
                r = tab[c_HlsSectorData[sector][2]];
            }

            dst[bidx] = b;
            dst[1] = g;
            dst[bidx^2] = r;
        }
        template <int bidx, int HR, typename T> static __device__ void HLS2RGBConvert(const T& src, uchar* dst)
        {
            float3 buf;

            buf.x = src.x;
            buf.y = src.y*(1.f/255.f);
            buf.z = src.z*(1.f/255.f);

            HLS2RGBConvert<bidx, HR>(buf, &buf.x);

            dst[0] = saturate_cast<uchar>(buf.x*255.f);
            dst[1] = saturate_cast<uchar>(buf.y*255.f);
            dst[2] = saturate_cast<uchar>(buf.z*255.f);
        }

        template <typename T, typename D, int bidx, int HR> struct HLS2RGB : public unary_function<T, D>
        {
            __device__ __forceinline__ D operator()(const T& src) const
            {
                D dst;
                HLS2RGBConvert<bidx, HR>(src, &dst.x);
                setAlpha(dst, ColorChannel<typename VecTraits<T>::elem_type>::max());
                return dst;
            }
        };
    }

#define OPENCV_GPU_IMPLEMENT_HLS2RGB_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef detail::HLS2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 180> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::HLS2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 180>(); \
        } \
    }; \
    template <typename T> struct name ## _full_traits \
    { \
        typedef detail::HLS2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 255> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::HLS2RGB<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type, bidx, 255>(); \
        } \
    }; \
    template <> struct name ## _traits<float> \
    { \
        typedef detail::HLS2RGB<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::HLS2RGB<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360>(); \
        } \
    }; \
    template <> struct name ## _full_traits<float> \
    { \
        typedef detail::HLS2RGB<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return detail::HLS2RGB<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type, bidx, 360>(); \
        } \
    };
}}}

#endif // __OPENCV_GPU_COLOR_DETAIL_HPP__
