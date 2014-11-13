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

#ifndef __OPENCV_CUDA_COLOR_DETAIL_HPP__
#define __OPENCV_CUDA_COLOR_DETAIL_HPP__

#include "../common.hpp"
#include "../vec_traits.hpp"
#include "../saturate_cast.hpp"
#include "../limits.hpp"
#include "../functional.hpp"

//! @cond IGNORED

namespace cv { namespace cuda { namespace device
{
    #ifndef CV_DESCALE
        #define CV_DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))
    #endif

    namespace color_detail
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

    namespace color_detail
    {
        template <typename T, int scn, int dcn, int bidx> struct RGB2RGB
            : unary_function<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type>
        {
            __device__ typename TypeVec<T, dcn>::vec_type operator()(const typename TypeVec<T, scn>::vec_type& src) const
            {
                typename TypeVec<T, dcn>::vec_type dst;

                dst.x = (&src.x)[bidx];
                dst.y = src.y;
                dst.z = (&src.x)[bidx^2];
                setAlpha(dst, getAlpha<T>(src));

                return dst;
            }

            __host__ __device__ __forceinline__ RGB2RGB() {}
            __host__ __device__ __forceinline__ RGB2RGB(const RGB2RGB&) {}
        };

        template <> struct RGB2RGB<uchar, 4, 4, 2> : unary_function<uint, uint>
        {
            __device__ uint operator()(uint src) const
            {
                uint dst = 0;

                dst |= (0xffu & (src >> 16));
                dst |= (0xffu & (src >> 8)) << 8;
                dst |= (0xffu & (src)) << 16;
                dst |= (0xffu & (src >> 24)) << 24;

                return dst;
            }

            __host__ __device__ __forceinline__ RGB2RGB() {}
            __host__ __device__ __forceinline__ RGB2RGB(const RGB2RGB&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_RGB2RGB_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2RGB<T, scn, dcn, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

/////////// Transforming 16-bit (565 or 555) RGB to/from 24/32-bit (888[8]) RGB //////////

    namespace color_detail
    {
        template <int green_bits, int bidx> struct RGB2RGB5x5Converter;
        template<int bidx> struct RGB2RGB5x5Converter<6, bidx>
        {
            static __device__ __forceinline__ ushort cvt(const uchar3& src)
            {
                return (ushort)(((&src.x)[bidx] >> 3) | ((src.y & ~3) << 3) | (((&src.x)[bidx^2] & ~7) << 8));
            }

            static __device__ __forceinline__ ushort cvt(uint src)
            {
                uint b = 0xffu & (src >> (bidx * 8));
                uint g = 0xffu & (src >> 8);
                uint r = 0xffu & (src >> ((bidx ^ 2) * 8));
                return (ushort)((b >> 3) | ((g & ~3) << 3) | ((r & ~7) << 8));
            }
        };

        template<int bidx> struct RGB2RGB5x5Converter<5, bidx>
        {
            static __device__ __forceinline__ ushort cvt(const uchar3& src)
            {
                return (ushort)(((&src.x)[bidx] >> 3) | ((src.y & ~7) << 2) | (((&src.x)[bidx^2] & ~7) << 7));
            }

            static __device__ __forceinline__ ushort cvt(uint src)
            {
                uint b = 0xffu & (src >> (bidx * 8));
                uint g = 0xffu & (src >> 8);
                uint r = 0xffu & (src >> ((bidx ^ 2) * 8));
                uint a = 0xffu & (src >> 24);
                return (ushort)((b >> 3) | ((g & ~7) << 2) | ((r & ~7) << 7) | (a * 0x8000));
            }
        };

        template<int scn, int bidx, int green_bits> struct RGB2RGB5x5;

        template<int bidx, int green_bits> struct RGB2RGB5x5<3, bidx,green_bits> : unary_function<uchar3, ushort>
        {
            __device__ __forceinline__ ushort operator()(const uchar3& src) const
            {
                return RGB2RGB5x5Converter<green_bits, bidx>::cvt(src);
            }

            __host__ __device__ __forceinline__ RGB2RGB5x5() {}
            __host__ __device__ __forceinline__ RGB2RGB5x5(const RGB2RGB5x5&) {}
        };

        template<int bidx, int green_bits> struct RGB2RGB5x5<4, bidx,green_bits> : unary_function<uint, ushort>
        {
            __device__ __forceinline__ ushort operator()(uint src) const
            {
                return RGB2RGB5x5Converter<green_bits, bidx>::cvt(src);
            }

            __host__ __device__ __forceinline__ RGB2RGB5x5() {}
            __host__ __device__ __forceinline__ RGB2RGB5x5(const RGB2RGB5x5&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_RGB2RGB5x5_TRAITS(name, scn, bidx, green_bits) \
    struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2RGB5x5<scn, bidx, green_bits> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

    namespace color_detail
    {
        template <int green_bits, int bidx> struct RGB5x52RGBConverter;

        template <int bidx> struct RGB5x52RGBConverter<5, bidx>
        {
            static __device__ __forceinline__ void cvt(uint src, uchar3& dst)
            {
                (&dst.x)[bidx] = src << 3;
                dst.y = (src >> 2) & ~7;
                (&dst.x)[bidx ^ 2] = (src >> 7) & ~7;
            }

            static __device__ __forceinline__ void cvt(uint src, uint& dst)
            {
                dst = 0;

                dst |= (0xffu & (src << 3)) << (bidx * 8);
                dst |= (0xffu & ((src >> 2) & ~7)) << 8;
                dst |= (0xffu & ((src >> 7) & ~7)) << ((bidx ^ 2) * 8);
                dst |= ((src & 0x8000) * 0xffu) << 24;
            }
        };

        template <int bidx> struct RGB5x52RGBConverter<6, bidx>
        {
            static __device__ __forceinline__ void cvt(uint src, uchar3& dst)
            {
                (&dst.x)[bidx] = src << 3;
                dst.y = (src >> 3) & ~3;
                (&dst.x)[bidx ^ 2] = (src >> 8) & ~7;
            }

            static __device__ __forceinline__ void cvt(uint src, uint& dst)
            {
                dst = 0xffu << 24;

                dst |= (0xffu & (src << 3)) << (bidx * 8);
                dst |= (0xffu &((src >> 3) & ~3)) << 8;
                dst |= (0xffu & ((src >> 8) & ~7)) << ((bidx ^ 2) * 8);
            }
        };

        template <int dcn, int bidx, int green_bits> struct RGB5x52RGB;

        template <int bidx, int green_bits> struct RGB5x52RGB<3, bidx, green_bits> : unary_function<ushort, uchar3>
        {
            __device__ __forceinline__ uchar3 operator()(ushort src) const
            {
                uchar3 dst;
                RGB5x52RGBConverter<green_bits, bidx>::cvt(src, dst);
                return dst;
            }
            __host__ __device__ __forceinline__ RGB5x52RGB() {}
            __host__ __device__ __forceinline__ RGB5x52RGB(const RGB5x52RGB&) {}

        };

        template <int bidx, int green_bits> struct RGB5x52RGB<4, bidx, green_bits> : unary_function<ushort, uint>
        {
            __device__ __forceinline__ uint operator()(ushort src) const
            {
                uint dst;
                RGB5x52RGBConverter<green_bits, bidx>::cvt(src, dst);
                return dst;
            }
            __host__ __device__ __forceinline__ RGB5x52RGB() {}
            __host__ __device__ __forceinline__ RGB5x52RGB(const RGB5x52RGB&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_RGB5x52RGB_TRAITS(name, dcn, bidx, green_bits) \
    struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::RGB5x52RGB<dcn, bidx, green_bits> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

///////////////////////////////// Grayscale to Color ////////////////////////////////

    namespace color_detail
    {
        template <typename T, int dcn> struct Gray2RGB : unary_function<T, typename TypeVec<T, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<T, dcn>::vec_type operator()(T src) const
            {
                typename TypeVec<T, dcn>::vec_type dst;

                dst.z = dst.y = dst.x = src;
                setAlpha(dst, ColorChannel<T>::max());

                return dst;
            }
            __host__ __device__ __forceinline__ Gray2RGB() {}
            __host__ __device__ __forceinline__ Gray2RGB(const Gray2RGB&) {}
        };

        template <> struct Gray2RGB<uchar, 4> : unary_function<uchar, uint>
        {
            __device__ __forceinline__ uint operator()(uint src) const
            {
                uint dst = 0xffu << 24;

                dst |= src;
                dst |= src << 8;
                dst |= src << 16;

                return dst;
            }
            __host__ __device__ __forceinline__ Gray2RGB() {}
            __host__ __device__ __forceinline__ Gray2RGB(const Gray2RGB&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_GRAY2RGB_TRAITS(name, dcn) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::Gray2RGB<T, dcn> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

    namespace color_detail
    {
        template <int green_bits> struct Gray2RGB5x5Converter;
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

        template<int green_bits> struct Gray2RGB5x5 : unary_function<uchar, ushort>
        {
            __device__ __forceinline__ ushort operator()(uint src) const
            {
                return Gray2RGB5x5Converter<green_bits>::cvt(src);
            }

            __host__ __device__ __forceinline__ Gray2RGB5x5() {}
            __host__ __device__ __forceinline__ Gray2RGB5x5(const Gray2RGB5x5&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_GRAY2RGB5x5_TRAITS(name, green_bits) \
    struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::Gray2RGB5x5<green_bits> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

///////////////////////////////// Color to Grayscale ////////////////////////////////

    namespace color_detail
    {
        template <int green_bits> struct RGB5x52GrayConverter;
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

        template<int green_bits> struct RGB5x52Gray : unary_function<ushort, uchar>
        {
            __device__ __forceinline__ uchar operator()(uint src) const
            {
                return RGB5x52GrayConverter<green_bits>::cvt(src);
            }
            __host__ __device__ __forceinline__ RGB5x52Gray() {}
            __host__ __device__ __forceinline__ RGB5x52Gray(const RGB5x52Gray&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_RGB5x52GRAY_TRAITS(name, green_bits) \
    struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::RGB5x52Gray<green_bits> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

    namespace color_detail
    {
        template <int bidx, typename T> static __device__ __forceinline__ T RGB2GrayConvert(const T* src)
        {
            return (T)CV_DESCALE((unsigned)(src[bidx] * B2Y + src[1] * G2Y + src[bidx^2] * R2Y), yuv_shift);
        }

        template <int bidx> static __device__ __forceinline__ uchar RGB2GrayConvert(uint src)
        {
            uint b = 0xffu & (src >> (bidx * 8));
            uint g = 0xffu & (src >> 8);
            uint r = 0xffu & (src >> ((bidx ^ 2) * 8));
            return CV_DESCALE((uint)(b * B2Y + g * G2Y + r * R2Y), yuv_shift);
        }

        template <int bidx> static __device__ __forceinline__ float RGB2GrayConvert(const float* src)
        {
            return src[bidx] * 0.114f + src[1] * 0.587f + src[bidx^2] * 0.299f;
        }

        template <typename T, int scn, int bidx> struct RGB2Gray : unary_function<typename TypeVec<T, scn>::vec_type, T>
        {
            __device__ __forceinline__ T operator()(const typename TypeVec<T, scn>::vec_type& src) const
            {
                return RGB2GrayConvert<bidx>(&src.x);
            }
            __host__ __device__ __forceinline__ RGB2Gray() {}
            __host__ __device__ __forceinline__ RGB2Gray(const RGB2Gray&) {}
        };

        template <int bidx> struct RGB2Gray<uchar, 4, bidx> : unary_function<uint, uchar>
        {
            __device__ __forceinline__ uchar operator()(uint src) const
            {
                return RGB2GrayConvert<bidx>(src);
            }
            __host__ __device__ __forceinline__ RGB2Gray() {}
            __host__ __device__ __forceinline__ RGB2Gray(const RGB2Gray&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_RGB2GRAY_TRAITS(name, scn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2Gray<T, scn, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

///////////////////////////////////// RGB <-> YUV //////////////////////////////////////

    namespace color_detail
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

        template <typename T, int scn, int dcn, int bidx> struct RGB2YUV
            : unary_function<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<T, dcn>::vec_type operator ()(const typename TypeVec<T, scn>::vec_type& src) const
            {
                typename TypeVec<T, dcn>::vec_type dst;
                RGB2YUVConvert<bidx>(&src.x, dst);
                return dst;
            }
            __host__ __device__ __forceinline__ RGB2YUV() {}
            __host__ __device__ __forceinline__ RGB2YUV(const RGB2YUV&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_RGB2YUV_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2YUV<T, scn, dcn, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

    namespace color_detail
    {
        __constant__ float c_YUV2RGBCoeffs_f[5] = { 2.032f, -0.395f, -0.581f, 1.140f };
        __constant__ int   c_YUV2RGBCoeffs_i[5] = { 33292, -6472, -9519, 18678 };

        template <int bidx, typename T, typename D> static __device__ void YUV2RGBConvert(const T& src, D* dst)
        {
            const int b = src.x + CV_DESCALE((src.z - ColorChannel<D>::half()) * c_YUV2RGBCoeffs_i[3], yuv_shift);

            const int g = src.x + CV_DESCALE((src.z - ColorChannel<D>::half()) * c_YUV2RGBCoeffs_i[2]
                                             + (src.y - ColorChannel<D>::half()) * c_YUV2RGBCoeffs_i[1], yuv_shift);

            const int r = src.x + CV_DESCALE((src.y - ColorChannel<D>::half()) * c_YUV2RGBCoeffs_i[0], yuv_shift);

            dst[bidx] = saturate_cast<D>(b);
            dst[1] = saturate_cast<D>(g);
            dst[bidx^2] = saturate_cast<D>(r);
        }

        template <int bidx> static __device__ uint YUV2RGBConvert(uint src)
        {
            const int x = 0xff & (src);
            const int y = 0xff & (src >> 8);
            const int z = 0xff & (src >> 16);

            const int b = x + CV_DESCALE((z - ColorChannel<uchar>::half()) * c_YUV2RGBCoeffs_i[3], yuv_shift);

            const int g = x + CV_DESCALE((z - ColorChannel<uchar>::half()) * c_YUV2RGBCoeffs_i[2]
                                         + (y - ColorChannel<uchar>::half()) * c_YUV2RGBCoeffs_i[1], yuv_shift);

            const int r = x + CV_DESCALE((y - ColorChannel<uchar>::half()) * c_YUV2RGBCoeffs_i[0], yuv_shift);

            uint dst = 0xffu << 24;

            dst |= saturate_cast<uchar>(b) << (bidx * 8);
            dst |= saturate_cast<uchar>(g) << 8;
            dst |= saturate_cast<uchar>(r) << ((bidx ^ 2) * 8);

            return dst;
        }

        template <int bidx, typename T> static __device__ __forceinline__ void YUV2RGBConvert(const T& src, float* dst)
        {
            dst[bidx] = src.x + (src.z - ColorChannel<float>::half()) * c_YUV2RGBCoeffs_f[3];

            dst[1] = src.x + (src.z - ColorChannel<float>::half()) * c_YUV2RGBCoeffs_f[2]
                     + (src.y - ColorChannel<float>::half()) * c_YUV2RGBCoeffs_f[1];

            dst[bidx^2] = src.x + (src.y - ColorChannel<float>::half()) * c_YUV2RGBCoeffs_f[0];
        }

        template <typename T, int scn, int dcn, int bidx> struct YUV2RGB
            : unary_function<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<T, dcn>::vec_type operator ()(const typename TypeVec<T, scn>::vec_type& src) const
            {
                typename TypeVec<T, dcn>::vec_type dst;

                YUV2RGBConvert<bidx>(src, &dst.x);
                setAlpha(dst, ColorChannel<T>::max());

                return dst;
            }
            __host__ __device__ __forceinline__ YUV2RGB() {}
            __host__ __device__ __forceinline__ YUV2RGB(const YUV2RGB&) {}
        };

        template <int bidx> struct YUV2RGB<uchar, 4, 4, bidx> : unary_function<uint, uint>
        {
            __device__ __forceinline__ uint operator ()(uint src) const
            {
                return YUV2RGBConvert<bidx>(src);
            }
            __host__ __device__ __forceinline__ YUV2RGB() {}
            __host__ __device__ __forceinline__ YUV2RGB(const YUV2RGB&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_YUV2RGB_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::YUV2RGB<T, scn, dcn, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

///////////////////////////////////// RGB <-> YCrCb //////////////////////////////////////

    namespace color_detail
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

        template <int bidx> static __device__ uint RGB2YCrCbConvert(uint src)
        {
            const int delta = ColorChannel<uchar>::half() * (1 << yuv_shift);

            const int Y = CV_DESCALE((0xffu & src) * c_RGB2YCrCbCoeffs_i[bidx^2] + (0xffu & (src >> 8)) * c_RGB2YCrCbCoeffs_i[1] + (0xffu & (src >> 16)) * c_RGB2YCrCbCoeffs_i[bidx], yuv_shift);
            const int Cr = CV_DESCALE(((0xffu & (src >> ((bidx ^ 2) * 8))) - Y) * c_RGB2YCrCbCoeffs_i[3] + delta, yuv_shift);
            const int Cb = CV_DESCALE(((0xffu & (src >> (bidx * 8))) - Y) * c_RGB2YCrCbCoeffs_i[4] + delta, yuv_shift);

            uint dst = 0;

            dst |= saturate_cast<uchar>(Y);
            dst |= saturate_cast<uchar>(Cr) << 8;
            dst |= saturate_cast<uchar>(Cb) << 16;

            return dst;
        }

        template <int bidx, typename D> static __device__ __forceinline__ void RGB2YCrCbConvert(const float* src, D& dst)
        {
            dst.x = src[0] * c_RGB2YCrCbCoeffs_f[bidx^2] + src[1] * c_RGB2YCrCbCoeffs_f[1] + src[2] * c_RGB2YCrCbCoeffs_f[bidx];
            dst.y = (src[bidx^2] - dst.x) * c_RGB2YCrCbCoeffs_f[3] + ColorChannel<float>::half();
            dst.z = (src[bidx] - dst.x) * c_RGB2YCrCbCoeffs_f[4] + ColorChannel<float>::half();
        }

        template <typename T, int scn, int dcn, int bidx> struct RGB2YCrCb
            : unary_function<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<T, dcn>::vec_type operator ()(const typename TypeVec<T, scn>::vec_type& src) const
            {
                typename TypeVec<T, dcn>::vec_type dst;
                RGB2YCrCbConvert<bidx>(&src.x, dst);
                return dst;
            }
            __host__ __device__ __forceinline__ RGB2YCrCb() {}
            __host__ __device__ __forceinline__ RGB2YCrCb(const RGB2YCrCb&) {}
        };

        template <int bidx> struct RGB2YCrCb<uchar, 4, 4, bidx> : unary_function<uint, uint>
        {
            __device__ __forceinline__ uint operator ()(uint src) const
            {
                return RGB2YCrCbConvert<bidx>(src);
            }

            __host__ __device__ __forceinline__ RGB2YCrCb() {}
            __host__ __device__ __forceinline__ RGB2YCrCb(const RGB2YCrCb&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_RGB2YCrCb_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2YCrCb<T, scn, dcn, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

    namespace color_detail
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

        template <int bidx> static __device__ uint YCrCb2RGBConvert(uint src)
        {
            const int x = 0xff & (src);
            const int y = 0xff & (src >> 8);
            const int z = 0xff & (src >> 16);

            const int b = x + CV_DESCALE((z - ColorChannel<uchar>::half()) * c_YCrCb2RGBCoeffs_i[3], yuv_shift);
            const int g = x + CV_DESCALE((z - ColorChannel<uchar>::half()) * c_YCrCb2RGBCoeffs_i[2] + (y - ColorChannel<uchar>::half()) * c_YCrCb2RGBCoeffs_i[1], yuv_shift);
            const int r = x + CV_DESCALE((y - ColorChannel<uchar>::half()) * c_YCrCb2RGBCoeffs_i[0], yuv_shift);

            uint dst = 0xffu << 24;

            dst |= saturate_cast<uchar>(b) << (bidx * 8);
            dst |= saturate_cast<uchar>(g) << 8;
            dst |= saturate_cast<uchar>(r) << ((bidx ^ 2) * 8);

            return dst;
        }

        template <int bidx, typename T> __device__ __forceinline__ void YCrCb2RGBConvert(const T& src, float* dst)
        {
            dst[bidx] = src.x + (src.z - ColorChannel<float>::half()) * c_YCrCb2RGBCoeffs_f[3];
            dst[1] = src.x + (src.z - ColorChannel<float>::half()) * c_YCrCb2RGBCoeffs_f[2] + (src.y - ColorChannel<float>::half()) * c_YCrCb2RGBCoeffs_f[1];
            dst[bidx^2] = src.x + (src.y - ColorChannel<float>::half()) * c_YCrCb2RGBCoeffs_f[0];
        }

        template <typename T, int scn, int dcn, int bidx> struct YCrCb2RGB
            : unary_function<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<T, dcn>::vec_type operator ()(const typename TypeVec<T, scn>::vec_type& src) const
            {
                typename TypeVec<T, dcn>::vec_type dst;

                YCrCb2RGBConvert<bidx>(src, &dst.x);
                setAlpha(dst, ColorChannel<T>::max());

                return dst;
            }
            __host__ __device__ __forceinline__ YCrCb2RGB() {}
            __host__ __device__ __forceinline__ YCrCb2RGB(const YCrCb2RGB&) {}
        };

        template <int bidx> struct YCrCb2RGB<uchar, 4, 4, bidx> : unary_function<uint, uint>
        {
            __device__ __forceinline__ uint operator ()(uint src) const
            {
                return YCrCb2RGBConvert<bidx>(src);
            }
            __host__ __device__ __forceinline__ YCrCb2RGB() {}
            __host__ __device__ __forceinline__ YCrCb2RGB(const YCrCb2RGB&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_YCrCb2RGB_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::YCrCb2RGB<T, scn, dcn, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

////////////////////////////////////// RGB <-> XYZ ///////////////////////////////////////

    namespace color_detail
    {
        __constant__ float c_RGB2XYZ_D65f[9] = { 0.412453f, 0.357580f, 0.180423f, 0.212671f, 0.715160f, 0.072169f, 0.019334f, 0.119193f, 0.950227f };
        __constant__ int   c_RGB2XYZ_D65i[9] = { 1689, 1465, 739, 871, 2929, 296, 79, 488, 3892 };

        template <int bidx, typename T, typename D> static __device__ __forceinline__ void RGB2XYZConvert(const T* src, D& dst)
        {
            dst.z = saturate_cast<T>(CV_DESCALE(src[bidx^2] * c_RGB2XYZ_D65i[6] + src[1] * c_RGB2XYZ_D65i[7] + src[bidx] * c_RGB2XYZ_D65i[8], xyz_shift));
            dst.x = saturate_cast<T>(CV_DESCALE(src[bidx^2] * c_RGB2XYZ_D65i[0] + src[1] * c_RGB2XYZ_D65i[1] + src[bidx] * c_RGB2XYZ_D65i[2], xyz_shift));
            dst.y = saturate_cast<T>(CV_DESCALE(src[bidx^2] * c_RGB2XYZ_D65i[3] + src[1] * c_RGB2XYZ_D65i[4] + src[bidx] * c_RGB2XYZ_D65i[5], xyz_shift));
        }

        template <int bidx> static __device__ __forceinline__ uint RGB2XYZConvert(uint src)
        {
            const uint b = 0xffu & (src >> (bidx * 8));
            const uint g = 0xffu & (src >> 8);
            const uint r = 0xffu & (src >> ((bidx ^ 2) * 8));

            const uint x = saturate_cast<uchar>(CV_DESCALE(r * c_RGB2XYZ_D65i[0] + g * c_RGB2XYZ_D65i[1] + b * c_RGB2XYZ_D65i[2], xyz_shift));
            const uint y = saturate_cast<uchar>(CV_DESCALE(r * c_RGB2XYZ_D65i[3] + g * c_RGB2XYZ_D65i[4] + b * c_RGB2XYZ_D65i[5], xyz_shift));
            const uint z = saturate_cast<uchar>(CV_DESCALE(r * c_RGB2XYZ_D65i[6] + g * c_RGB2XYZ_D65i[7] + b * c_RGB2XYZ_D65i[8], xyz_shift));

            uint dst = 0;

            dst |= x;
            dst |= y << 8;
            dst |= z << 16;

            return dst;
        }

        template <int bidx, typename D> static __device__ __forceinline__ void RGB2XYZConvert(const float* src, D& dst)
        {
            dst.x = src[bidx^2] * c_RGB2XYZ_D65f[0] + src[1] * c_RGB2XYZ_D65f[1] + src[bidx] * c_RGB2XYZ_D65f[2];
            dst.y = src[bidx^2] * c_RGB2XYZ_D65f[3] + src[1] * c_RGB2XYZ_D65f[4] + src[bidx] * c_RGB2XYZ_D65f[5];
            dst.z = src[bidx^2] * c_RGB2XYZ_D65f[6] + src[1] * c_RGB2XYZ_D65f[7] + src[bidx] * c_RGB2XYZ_D65f[8];
        }

        template <typename T, int scn, int dcn, int bidx> struct RGB2XYZ
            : unary_function<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<T, dcn>::vec_type operator()(const typename TypeVec<T, scn>::vec_type& src) const
            {
                typename TypeVec<T, dcn>::vec_type dst;

                RGB2XYZConvert<bidx>(&src.x, dst);

                return dst;
            }
            __host__ __device__ __forceinline__ RGB2XYZ() {}
            __host__ __device__ __forceinline__ RGB2XYZ(const RGB2XYZ&) {}
        };

        template <int bidx> struct RGB2XYZ<uchar, 4, 4, bidx> : unary_function<uint, uint>
        {
            __device__ __forceinline__ uint operator()(uint src) const
            {
                return RGB2XYZConvert<bidx>(src);
            }
            __host__ __device__ __forceinline__ RGB2XYZ() {}
            __host__ __device__ __forceinline__ RGB2XYZ(const RGB2XYZ&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_RGB2XYZ_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2XYZ<T, scn, dcn, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

    namespace color_detail
    {
        __constant__ float c_XYZ2sRGB_D65f[9] = { 3.240479f, -1.53715f, -0.498535f, -0.969256f, 1.875991f, 0.041556f, 0.055648f, -0.204043f, 1.057311f };
        __constant__ int   c_XYZ2sRGB_D65i[9] = { 13273, -6296, -2042, -3970, 7684, 170, 228, -836, 4331 };

        template <int bidx, typename T, typename D> static __device__ __forceinline__ void XYZ2RGBConvert(const T& src, D* dst)
        {
            dst[bidx^2] = saturate_cast<D>(CV_DESCALE(src.x * c_XYZ2sRGB_D65i[0] + src.y * c_XYZ2sRGB_D65i[1] + src.z * c_XYZ2sRGB_D65i[2], xyz_shift));
            dst[1]      = saturate_cast<D>(CV_DESCALE(src.x * c_XYZ2sRGB_D65i[3] + src.y * c_XYZ2sRGB_D65i[4] + src.z * c_XYZ2sRGB_D65i[5], xyz_shift));
            dst[bidx]   = saturate_cast<D>(CV_DESCALE(src.x * c_XYZ2sRGB_D65i[6] + src.y * c_XYZ2sRGB_D65i[7] + src.z * c_XYZ2sRGB_D65i[8], xyz_shift));
        }

        template <int bidx> static __device__ __forceinline__ uint XYZ2RGBConvert(uint src)
        {
            const int x = 0xff & src;
            const int y = 0xff & (src >> 8);
            const int z = 0xff & (src >> 16);

            const uint r = saturate_cast<uchar>(CV_DESCALE(x * c_XYZ2sRGB_D65i[0] + y * c_XYZ2sRGB_D65i[1] + z * c_XYZ2sRGB_D65i[2], xyz_shift));
            const uint g = saturate_cast<uchar>(CV_DESCALE(x * c_XYZ2sRGB_D65i[3] + y * c_XYZ2sRGB_D65i[4] + z * c_XYZ2sRGB_D65i[5], xyz_shift));
            const uint b = saturate_cast<uchar>(CV_DESCALE(x * c_XYZ2sRGB_D65i[6] + y * c_XYZ2sRGB_D65i[7] + z * c_XYZ2sRGB_D65i[8], xyz_shift));

            uint dst = 0xffu << 24;

            dst |= b << (bidx * 8);
            dst |= g << 8;
            dst |= r << ((bidx ^ 2) * 8);

            return dst;
        }

        template <int bidx, typename T> static __device__ __forceinline__ void XYZ2RGBConvert(const T& src, float* dst)
        {
            dst[bidx^2] = src.x * c_XYZ2sRGB_D65f[0] + src.y * c_XYZ2sRGB_D65f[1] + src.z * c_XYZ2sRGB_D65f[2];
            dst[1]      = src.x * c_XYZ2sRGB_D65f[3] + src.y * c_XYZ2sRGB_D65f[4] + src.z * c_XYZ2sRGB_D65f[5];
            dst[bidx]   = src.x * c_XYZ2sRGB_D65f[6] + src.y * c_XYZ2sRGB_D65f[7] + src.z * c_XYZ2sRGB_D65f[8];
        }

        template <typename T, int scn, int dcn, int bidx> struct XYZ2RGB
            : unary_function<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<T, dcn>::vec_type operator()(const typename TypeVec<T, scn>::vec_type& src) const
            {
                typename TypeVec<T, dcn>::vec_type dst;

                XYZ2RGBConvert<bidx>(src, &dst.x);
                setAlpha(dst, ColorChannel<T>::max());

                return dst;
            }
            __host__ __device__ __forceinline__ XYZ2RGB() {}
            __host__ __device__ __forceinline__ XYZ2RGB(const XYZ2RGB&) {}
        };

        template <int bidx> struct XYZ2RGB<uchar, 4, 4, bidx> : unary_function<uint, uint>
        {
            __device__ __forceinline__ uint operator()(uint src) const
            {
                return XYZ2RGBConvert<bidx>(src);
            }
            __host__ __device__ __forceinline__ XYZ2RGB() {}
            __host__ __device__ __forceinline__ XYZ2RGB(const XYZ2RGB&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_XYZ2RGB_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::XYZ2RGB<T, scn, dcn, bidx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

////////////////////////////////////// RGB <-> HSV ///////////////////////////////////////

    namespace color_detail
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
            vr = (v == r) * -1;
            vg = (v == g) * -1;

            s = (diff * c_HsvDivTable[v] + (1 << (hsv_shift-1))) >> hsv_shift;
            h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
            h = (h * hdiv_table[diff] + (1 << (hsv_shift-1))) >> hsv_shift;
            h += (h < 0) * hr;

            dst.x = saturate_cast<uchar>(h);
            dst.y = (uchar)s;
            dst.z = (uchar)v;
        }

        template <int bidx, int hr> static __device__ uint RGB2HSVConvert(uint src)
        {
            const int hsv_shift = 12;
            const int* hdiv_table = hr == 180 ? c_HsvDivTable180 : c_HsvDivTable256;

            const int b = 0xff & (src >> (bidx * 8));
            const int g = 0xff & (src >> 8);
            const int r = 0xff & (src >> ((bidx ^ 2) * 8));

            int h, s, v = b;
            int vmin = b, diff;
            int vr, vg;

            v = ::max(v, g);
            v = ::max(v, r);
            vmin = ::min(vmin, g);
            vmin = ::min(vmin, r);

            diff = v - vmin;
            vr = (v == r) * -1;
            vg = (v == g) * -1;

            s = (diff * c_HsvDivTable[v] + (1 << (hsv_shift-1))) >> hsv_shift;
            h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
            h = (h * hdiv_table[diff] + (1 << (hsv_shift-1))) >> hsv_shift;
            h += (h < 0) * hr;

            uint dst = 0;

            dst |= saturate_cast<uchar>(h);
            dst |= (0xffu & s) << 8;
            dst |= (0xffu & v) << 16;

            return dst;
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

            h  = (v == r) * (g - b) * diff;
            h += (v != r && v == g) * ((b - r) * diff + 120.f);
            h += (v != r && v != g) * ((r - g) * diff + 240.f);
            h += (h < 0) * 360.f;

            dst.x = h * hscale;
            dst.y = s;
            dst.z = v;
        }

        template <typename T, int scn, int dcn, int bidx, int hr> struct RGB2HSV
            : unary_function<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<T, dcn>::vec_type operator()(const typename TypeVec<T, scn>::vec_type& src) const
            {
                typename TypeVec<T, dcn>::vec_type dst;

                RGB2HSVConvert<bidx, hr>(&src.x, dst);

                return dst;
            }
            __host__ __device__ __forceinline__ RGB2HSV() {}
            __host__ __device__ __forceinline__ RGB2HSV(const RGB2HSV&) {}
        };

        template <int bidx, int hr> struct RGB2HSV<uchar, 4, 4, bidx, hr> : unary_function<uint, uint>
        {
            __device__ __forceinline__ uint operator()(uint src) const
            {
                return RGB2HSVConvert<bidx, hr>(src);
            }
            __host__ __device__ __forceinline__ RGB2HSV() {}
            __host__ __device__ __forceinline__ RGB2HSV(const RGB2HSV&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_RGB2HSV_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2HSV<T, scn, dcn, bidx, 180> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    }; \
    template <typename T> struct name ## _full_traits \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2HSV<T, scn, dcn, bidx, 256> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    }; \
    template <> struct name ## _traits<float> \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2HSV<float, scn, dcn, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    }; \
    template <> struct name ## _full_traits<float> \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2HSV<float, scn, dcn, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

    namespace color_detail
    {
        __constant__ int c_HsvSectorData[6][3] = { {1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0} };

        template <int bidx, int hr, typename T> static __device__ void HSV2RGBConvert(const T& src, float* dst)
        {
            const float hscale = 6.f / hr;

            float h = src.x, s = src.y, v = src.z;
            float b = v, g = v, r = v;

            if (s != 0)
            {
                h *= hscale;

                if( h < 0 )
                    do h += 6; while( h < 0 );
                else if( h >= 6 )
                    do h -= 6; while( h >= 6 );

                int sector = __float2int_rd(h);
                h -= sector;

                if ( (unsigned)sector >= 6u )
                {
                    sector = 0;
                    h = 0.f;
                }

                float tab[4];
                tab[0] = v;
                tab[1] = v * (1.f - s);
                tab[2] = v * (1.f - s * h);
                tab[3] = v * (1.f - s * (1.f - h));

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
            buf.y = src.y * (1.f / 255.f);
            buf.z = src.z * (1.f / 255.f);

            HSV2RGBConvert<bidx, HR>(buf, &buf.x);

            dst[0] = saturate_cast<uchar>(buf.x * 255.f);
            dst[1] = saturate_cast<uchar>(buf.y * 255.f);
            dst[2] = saturate_cast<uchar>(buf.z * 255.f);
        }

        template <int bidx, int hr> static __device__ uint HSV2RGBConvert(uint src)
        {
            float3 buf;

            buf.x = src & 0xff;
            buf.y = ((src >> 8) & 0xff) * (1.f/255.f);
            buf.z = ((src >> 16) & 0xff) * (1.f/255.f);

            HSV2RGBConvert<bidx, hr>(buf, &buf.x);

            uint dst = 0xffu << 24;

            dst |= saturate_cast<uchar>(buf.x * 255.f);
            dst |= saturate_cast<uchar>(buf.y * 255.f) << 8;
            dst |= saturate_cast<uchar>(buf.z * 255.f) << 16;

            return dst;
        }

        template <typename T, int scn, int dcn, int bidx, int hr> struct HSV2RGB
            : unary_function<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<T, dcn>::vec_type operator()(const typename TypeVec<T, scn>::vec_type& src) const
            {
                typename TypeVec<T, dcn>::vec_type dst;

                HSV2RGBConvert<bidx, hr>(src, &dst.x);
                setAlpha(dst, ColorChannel<T>::max());

                return dst;
            }
            __host__ __device__ __forceinline__ HSV2RGB() {}
            __host__ __device__ __forceinline__ HSV2RGB(const HSV2RGB&) {}
        };

        template <int bidx, int hr> struct HSV2RGB<uchar, 4, 4, bidx, hr> : unary_function<uint, uint>
        {
            __device__ __forceinline__ uint operator()(uint src) const
            {
                return HSV2RGBConvert<bidx, hr>(src);
            }
            __host__ __device__ __forceinline__ HSV2RGB() {}
            __host__ __device__ __forceinline__ HSV2RGB(const HSV2RGB&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_HSV2RGB_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::HSV2RGB<T, scn, dcn, bidx, 180> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    }; \
    template <typename T> struct name ## _full_traits \
    { \
        typedef ::cv::cuda::device::color_detail::HSV2RGB<T, scn, dcn, bidx, 255> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    }; \
    template <> struct name ## _traits<float> \
    { \
        typedef ::cv::cuda::device::color_detail::HSV2RGB<float, scn, dcn, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    }; \
    template <> struct name ## _full_traits<float> \
    { \
        typedef ::cv::cuda::device::color_detail::HSV2RGB<float, scn, dcn, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

/////////////////////////////////////// RGB <-> HLS ////////////////////////////////////////

    namespace color_detail
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
                s = (l < 0.5f) * diff / (vmax + vmin);
                s += (l >= 0.5f) * diff / (2.0f - vmax - vmin);

                diff = 60.f / diff;

                h  = (vmax == r) * (g - b) * diff;
                h += (vmax != r && vmax == g) * ((b - r) * diff + 120.f);
                h += (vmax != r && vmax != g) * ((r - g) * diff + 240.f);
                h += (h < 0.f) * 360.f;
            }

            dst.x = h * hscale;
            dst.y = l;
            dst.z = s;
        }

        template <int bidx, int hr, typename D> static __device__ void RGB2HLSConvert(const uchar* src, D& dst)
        {
            float3 buf;

            buf.x = src[0] * (1.f / 255.f);
            buf.y = src[1] * (1.f / 255.f);
            buf.z = src[2] * (1.f / 255.f);

            RGB2HLSConvert<bidx, hr>(&buf.x, buf);

            dst.x = saturate_cast<uchar>(buf.x);
            dst.y = saturate_cast<uchar>(buf.y*255.f);
            dst.z = saturate_cast<uchar>(buf.z*255.f);
        }

        template <int bidx, int hr> static __device__ uint RGB2HLSConvert(uint src)
        {
            float3 buf;

            buf.x = (0xff & src) * (1.f / 255.f);
            buf.y = (0xff & (src >> 8)) * (1.f / 255.f);
            buf.z = (0xff & (src >> 16)) * (1.f / 255.f);

            RGB2HLSConvert<bidx, hr>(&buf.x, buf);

            uint dst = 0xffu << 24;

            dst |= saturate_cast<uchar>(buf.x);
            dst |= saturate_cast<uchar>(buf.y * 255.f) << 8;
            dst |= saturate_cast<uchar>(buf.z * 255.f) << 16;

            return dst;
        }

        template <typename T, int scn, int dcn, int bidx, int hr> struct RGB2HLS
            : unary_function<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<T, dcn>::vec_type operator()(const typename TypeVec<T, scn>::vec_type& src) const
            {
                typename TypeVec<T, dcn>::vec_type dst;

                RGB2HLSConvert<bidx, hr>(&src.x, dst);

                return dst;
            }
            __host__ __device__ __forceinline__ RGB2HLS() {}
            __host__ __device__ __forceinline__ RGB2HLS(const RGB2HLS&) {}
        };

        template <int bidx, int hr> struct RGB2HLS<uchar, 4, 4, bidx, hr> : unary_function<uint, uint>
        {
            __device__ __forceinline__ uint operator()(uint src) const
            {
                return RGB2HLSConvert<bidx, hr>(src);
            }
            __host__ __device__ __forceinline__ RGB2HLS() {}
            __host__ __device__ __forceinline__ RGB2HLS(const RGB2HLS&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_RGB2HLS_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2HLS<T, scn, dcn, bidx, 180> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    }; \
    template <typename T> struct name ## _full_traits \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2HLS<T, scn, dcn, bidx, 256> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    }; \
    template <> struct name ## _traits<float> \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2HLS<float, scn, dcn, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    }; \
    template <> struct name ## _full_traits<float> \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2HLS<float, scn, dcn, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

    namespace color_detail
    {
        __constant__ int c_HlsSectorData[6][3] = { {1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0} };

        template <int bidx, int hr, typename T> static __device__ void HLS2RGBConvert(const T& src, float* dst)
        {
            const float hscale = 6.0f / hr;

            float h = src.x, l = src.y, s = src.z;
            float b = l, g = l, r = l;

            if (s != 0)
            {
                float p2  = (l <= 0.5f) * l * (1 + s);
                      p2 += (l > 0.5f) * (l + s - l * s);
                float p1 = 2 * l - p2;

                h *= hscale;

                if( h < 0 )
                    do h += 6; while( h < 0 );
                else if( h >= 6 )
                    do h -= 6; while( h >= 6 );

                int sector;
                sector = __float2int_rd(h);

                h -= sector;

                float tab[4];
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

        template <int bidx, int hr, typename T> static __device__ void HLS2RGBConvert(const T& src, uchar* dst)
        {
            float3 buf;

            buf.x = src.x;
            buf.y = src.y * (1.f / 255.f);
            buf.z = src.z * (1.f / 255.f);

            HLS2RGBConvert<bidx, hr>(buf, &buf.x);

            dst[0] = saturate_cast<uchar>(buf.x * 255.f);
            dst[1] = saturate_cast<uchar>(buf.y * 255.f);
            dst[2] = saturate_cast<uchar>(buf.z * 255.f);
        }

        template <int bidx, int hr> static __device__ uint HLS2RGBConvert(uint src)
        {
            float3 buf;

            buf.x = 0xff & src;
            buf.y = (0xff & (src >> 8)) * (1.f / 255.f);
            buf.z = (0xff & (src >> 16)) * (1.f / 255.f);

            HLS2RGBConvert<bidx, hr>(buf, &buf.x);

            uint dst = 0xffu << 24;

            dst |= saturate_cast<uchar>(buf.x * 255.f);
            dst |= saturate_cast<uchar>(buf.y * 255.f) << 8;
            dst |= saturate_cast<uchar>(buf.z * 255.f) << 16;

            return dst;
        }

        template <typename T, int scn, int dcn, int bidx, int hr> struct HLS2RGB
            : unary_function<typename TypeVec<T, scn>::vec_type, typename TypeVec<T, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<T, dcn>::vec_type operator()(const typename TypeVec<T, scn>::vec_type& src) const
            {
                typename TypeVec<T, dcn>::vec_type dst;

                HLS2RGBConvert<bidx, hr>(src, &dst.x);
                setAlpha(dst, ColorChannel<T>::max());

                return dst;
            }
            __host__ __device__ __forceinline__ HLS2RGB() {}
            __host__ __device__ __forceinline__ HLS2RGB(const HLS2RGB&) {}
        };

        template <int bidx, int hr> struct HLS2RGB<uchar, 4, 4, bidx, hr> : unary_function<uint, uint>
        {
            __device__ __forceinline__ uint operator()(uint src) const
            {
                return HLS2RGBConvert<bidx, hr>(src);
            }
            __host__ __device__ __forceinline__ HLS2RGB() {}
            __host__ __device__ __forceinline__ HLS2RGB(const HLS2RGB&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_HLS2RGB_TRAITS(name, scn, dcn, bidx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::HLS2RGB<T, scn, dcn, bidx, 180> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    }; \
    template <typename T> struct name ## _full_traits \
    { \
        typedef ::cv::cuda::device::color_detail::HLS2RGB<T, scn, dcn, bidx, 255> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    }; \
    template <> struct name ## _traits<float> \
    { \
        typedef ::cv::cuda::device::color_detail::HLS2RGB<float, scn, dcn, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    }; \
    template <> struct name ## _full_traits<float> \
    { \
        typedef ::cv::cuda::device::color_detail::HLS2RGB<float, scn, dcn, bidx, 360> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

///////////////////////////////////// RGB <-> Lab /////////////////////////////////////

    namespace color_detail
    {
        enum
        {
            LAB_CBRT_TAB_SIZE = 1024,
            GAMMA_TAB_SIZE = 1024,
            lab_shift = xyz_shift,
            gamma_shift = 3,
            lab_shift2 = (lab_shift + gamma_shift),
            LAB_CBRT_TAB_SIZE_B = (256 * 3 / 2 * (1 << gamma_shift))
        };

        __constant__ ushort c_sRGBGammaTab_b[] = {0,1,1,2,2,3,4,4,5,6,6,7,8,8,9,10,11,11,12,13,14,15,16,17,19,20,21,22,24,25,26,28,29,31,33,34,36,38,40,41,43,45,47,49,51,54,56,58,60,63,65,68,70,73,75,78,81,83,86,89,92,95,98,101,105,108,111,115,118,121,125,129,132,136,140,144,147,151,155,160,164,168,172,176,181,185,190,194,199,204,209,213,218,223,228,233,239,244,249,255,260,265,271,277,282,288,294,300,306,312,318,324,331,337,343,350,356,363,370,376,383,390,397,404,411,418,426,433,440,448,455,463,471,478,486,494,502,510,518,527,535,543,552,560,569,578,586,595,604,613,622,631,641,650,659,669,678,688,698,707,717,727,737,747,757,768,778,788,799,809,820,831,842,852,863,875,886,897,908,920,931,943,954,966,978,990,1002,1014,1026,1038,1050,1063,1075,1088,1101,1113,1126,1139,1152,1165,1178,1192,1205,1218,1232,1245,1259,1273,1287,1301,1315,1329,1343,1357,1372,1386,1401,1415,1430,1445,1460,1475,1490,1505,1521,1536,1551,1567,1583,1598,1614,1630,1646,1662,1678,1695,1711,1728,1744,1761,1778,1794,1811,1828,1846,1863,1880,1897,1915,1933,1950,1968,1986,2004,2022,2040};

        __device__ __forceinline__ int LabCbrt_b(int i)
        {
            float x = i * (1.f / (255.f * (1 << gamma_shift)));
            return (1 << lab_shift2) * (x < 0.008856f ? x * 7.787f + 0.13793103448275862f : ::cbrtf(x));
        }

        template <bool srgb, int blueIdx, typename T, typename D>
        __device__ __forceinline__ void RGB2LabConvert_b(const T& src, D& dst)
        {
            const int Lscale = (116 * 255 + 50) / 100;
            const int Lshift = -((16 * 255 * (1 << lab_shift2) + 50) / 100);

            int B = blueIdx == 0 ? src.x : src.z;
            int G = src.y;
            int R = blueIdx == 0 ? src.z : src.x;

            if (srgb)
            {
                B = c_sRGBGammaTab_b[B];
                G = c_sRGBGammaTab_b[G];
                R = c_sRGBGammaTab_b[R];
            }
            else
            {
                B <<= 3;
                G <<= 3;
                R <<= 3;
            }

            int fX = LabCbrt_b(CV_DESCALE(B * 778 + G * 1541 + R * 1777, lab_shift));
            int fY = LabCbrt_b(CV_DESCALE(B * 296 + G * 2929 + R * 871, lab_shift));
            int fZ = LabCbrt_b(CV_DESCALE(B * 3575 + G * 448 + R * 73, lab_shift));

            int L = CV_DESCALE(Lscale * fY + Lshift, lab_shift2);
            int a = CV_DESCALE(500 * (fX - fY) + 128 * (1 << lab_shift2), lab_shift2);
            int b = CV_DESCALE(200 * (fY - fZ) + 128 * (1 << lab_shift2), lab_shift2);

            dst.x = saturate_cast<uchar>(L);
            dst.y = saturate_cast<uchar>(a);
            dst.z = saturate_cast<uchar>(b);
        }

        __device__ __forceinline__ float splineInterpolate(float x, const float* tab, int n)
        {
            int ix = ::min(::max(int(x), 0), n-1);
            x -= ix;
            tab += ix * 4;
            return ((tab[3] * x + tab[2]) * x + tab[1]) * x + tab[0];
        }

        __constant__ float c_sRGBGammaTab[] = {0,7.55853e-05,0.,-7.51331e-13,7.55853e-05,7.55853e-05,-2.25399e-12,3.75665e-12,0.000151171,7.55853e-05,9.01597e-12,-6.99932e-12,0.000226756,7.55853e-05,-1.1982e-11,2.41277e-12,0.000302341,7.55853e-05,-4.74369e-12,1.19001e-11,0.000377927,7.55853e-05,3.09568e-11,-2.09095e-11,0.000453512,7.55853e-05,-3.17718e-11,1.35303e-11,0.000529097,7.55853e-05,8.81905e-12,-4.10782e-12,0.000604683,7.55853e-05,-3.50439e-12,2.90097e-12,0.000680268,7.55853e-05,5.19852e-12,-7.49607e-12,0.000755853,7.55853e-05,-1.72897e-11,2.70833e-11,0.000831439,7.55854e-05,6.39602e-11,-4.26295e-11,0.000907024,7.55854e-05,-6.39282e-11,2.70193e-11,0.000982609,7.55853e-05,1.71298e-11,-7.24017e-12,0.00105819,7.55853e-05,-4.59077e-12,1.94137e-12,0.00113378,7.55853e-05,1.23333e-12,-5.25291e-13,0.00120937,7.55853e-05,-3.42545e-13,1.59799e-13,0.00128495,7.55853e-05,1.36852e-13,-1.13904e-13,0.00136054,7.55853e-05,-2.04861e-13,2.95818e-13,0.00143612,7.55853e-05,6.82594e-13,-1.06937e-12,0.00151171,7.55853e-05,-2.52551e-12,3.98166e-12,0.00158729,7.55853e-05,9.41946e-12,-1.48573e-11,0.00166288,7.55853e-05,-3.51523e-11,5.54474e-11,0.00173846,7.55854e-05,1.3119e-10,-9.0517e-11,0.00181405,7.55854e-05,-1.40361e-10,7.37899e-11,0.00188963,7.55853e-05,8.10085e-11,-8.82272e-11,0.00196522,7.55852e-05,-1.83673e-10,1.62704e-10,0.0020408,7.55853e-05,3.04438e-10,-2.13341e-10,0.00211639,7.55853e-05,-3.35586e-10,2.25e-10,0.00219197,7.55853e-05,3.39414e-10,-2.20997e-10,0.00226756,7.55853e-05,-3.23576e-10,1.93326e-10,0.00234315,7.55853e-05,2.564e-10,-8.66446e-11,0.00241873,7.55855e-05,-3.53328e-12,-7.9578e-11,0.00249432,7.55853e-05,-2.42267e-10,1.72126e-10,0.0025699,7.55853e-05,2.74111e-10,-1.43265e-10,0.00264549,7.55854e-05,-1.55683e-10,-6.47292e-11,0.00272107,7.55849e-05,-3.4987e-10,8.67842e-10,0.00279666,7.55868e-05,2.25366e-09,-3.8723e-09,0.00287224,7.55797e-05,-9.36325e-09,1.5087e-08,0.00294783,7.56063e-05,3.58978e-08,-5.69415e-08,0.00302341,7.55072e-05,-1.34927e-07,2.13144e-07,0.003099,7.58768e-05,5.04507e-07,1.38713e-07,0.00317552,7.7302e-05,9.20646e-07,-1.55186e-07,0.00325359,7.86777e-05,4.55087e-07,4.26813e-08,0.00333276,7.97159e-05,5.83131e-07,-1.06495e-08,0.00341305,8.08502e-05,5.51182e-07,3.87467e-09,0.00349446,8.19642e-05,5.62806e-07,-1.92586e-10,0.00357698,8.30892e-05,5.62228e-07,1.0866e-09,0.00366063,8.4217e-05,5.65488e-07,5.02818e-10,0.00374542,8.53494e-05,5.66997e-07,8.60211e-10,0.00383133,8.6486e-05,5.69577e-07,7.13044e-10,0.00391839,8.76273e-05,5.71716e-07,4.78527e-10,0.00400659,8.87722e-05,5.73152e-07,1.09818e-09,0.00409594,8.99218e-05,5.76447e-07,2.50964e-10,0.00418644,9.10754e-05,5.772e-07,1.15762e-09,0.00427809,9.22333e-05,5.80672e-07,2.40865e-10,0.0043709,9.33954e-05,5.81395e-07,1.13854e-09,0.00446488,9.45616e-05,5.84811e-07,3.27267e-10,0.00456003,9.57322e-05,5.85792e-07,8.1197e-10,0.00465635,9.69062e-05,5.88228e-07,6.15823e-10,0.00475384,9.80845e-05,5.90076e-07,9.15747e-10,0.00485252,9.92674e-05,5.92823e-07,3.778e-10,0.00495238,0.000100454,5.93956e-07,8.32623e-10,0.00505343,0.000101645,5.96454e-07,4.82695e-10,0.00515567,0.000102839,5.97902e-07,9.61904e-10,0.00525911,0.000104038,6.00788e-07,3.26281e-10,0.00536375,0.00010524,6.01767e-07,9.926e-10,0.00546959,0.000106447,6.04745e-07,3.59933e-10,0.00557664,0.000107657,6.05824e-07,8.2728e-10,0.0056849,0.000108871,6.08306e-07,5.21898e-10,0.00579438,0.00011009,6.09872e-07,8.10492e-10,0.00590508,0.000111312,6.12303e-07,4.27046e-10,0.00601701,0.000112538,6.13585e-07,7.40878e-10,0.00613016,0.000113767,6.15807e-07,8.00469e-10,0.00624454,0.000115001,6.18209e-07,2.48178e-10,0.00636016,0.000116238,6.18953e-07,1.00073e-09,0.00647702,0.000117479,6.21955e-07,4.05654e-10,0.00659512,0.000118724,6.23172e-07,6.36192e-10,0.00671447,0.000119973,6.25081e-07,7.74927e-10,0.00683507,0.000121225,6.27406e-07,4.54975e-10,0.00695692,0.000122481,6.28771e-07,6.64841e-10,0.00708003,0.000123741,6.30765e-07,6.10972e-10,0.00720441,0.000125004,6.32598e-07,6.16543e-10,0.00733004,0.000126271,6.34448e-07,6.48204e-10,0.00745695,0.000127542,6.36392e-07,5.15835e-10,0.00758513,0.000128816,6.3794e-07,5.48103e-10,0.00771458,0.000130094,6.39584e-07,1.01706e-09,0.00784532,0.000131376,6.42635e-07,4.0283e-11,0.00797734,0.000132661,6.42756e-07,6.84471e-10,0.00811064,0.000133949,6.4481e-07,9.47144e-10,0.00824524,0.000135241,6.47651e-07,1.83472e-10,0.00838112,0.000136537,6.48201e-07,1.11296e-09,0.00851831,0.000137837,6.5154e-07,2.13163e-11,0.0086568,0.00013914,6.51604e-07,6.64462e-10,0.00879659,0.000140445,6.53598e-07,1.04613e-09,0.00893769,0.000141756,6.56736e-07,-1.92377e-10,0.0090801,0.000143069,6.56159e-07,1.58601e-09,0.00922383,0.000144386,6.60917e-07,-5.63754e-10,0.00936888,0.000145706,6.59226e-07,1.60033e-09,0.00951524,0.000147029,6.64027e-07,-2.49543e-10,0.00966294,0.000148356,6.63278e-07,1.26043e-09,0.00981196,0.000149687,6.67059e-07,-1.35572e-10,0.00996231,0.00015102,6.66653e-07,1.14458e-09,0.010114,0.000152357,6.70086e-07,2.13864e-10,0.010267,0.000153698,6.70728e-07,7.93856e-10,0.0104214,0.000155042,6.73109e-07,3.36077e-10,0.0105771,0.000156389,6.74118e-07,6.55765e-10,0.0107342,0.000157739,6.76085e-07,7.66211e-10,0.0108926,0.000159094,6.78384e-07,4.66116e-12,0.0110524,0.000160451,6.78398e-07,1.07775e-09,0.0112135,0.000161811,6.81631e-07,3.41023e-10,0.011376,0.000163175,6.82654e-07,3.5205e-10,0.0115398,0.000164541,6.8371e-07,1.04473e-09,0.0117051,0.000165912,6.86844e-07,1.25757e-10,0.0118717,0.000167286,6.87222e-07,3.14818e-10,0.0120396,0.000168661,6.88166e-07,1.40886e-09,0.012209,0.000170042,6.92393e-07,-3.62244e-10,0.0123797,0.000171425,6.91306e-07,9.71397e-10,0.0125518,0.000172811,6.9422e-07,2.02003e-10,0.0127253,0.0001742,6.94826e-07,1.01448e-09,0.0129002,0.000175593,6.97869e-07,3.96653e-10,0.0130765,0.00017699,6.99059e-07,1.92927e-10,0.0132542,0.000178388,6.99638e-07,6.94305e-10,0.0134333,0.00017979,7.01721e-07,7.55108e-10,0.0136138,0.000181195,7.03986e-07,1.05918e-11,0.0137957,0.000182603,7.04018e-07,1.06513e-09,0.013979,0.000184015,7.07214e-07,3.85512e-10,0.0141637,0.00018543,7.0837e-07,1.86769e-10,0.0143499,0.000186848,7.0893e-07,7.30116e-10,0.0145374,0.000188268,7.11121e-07,6.17983e-10,0.0147264,0.000189692,7.12975e-07,5.23282e-10,0.0149168,0.000191119,7.14545e-07,8.28398e-11,0.0151087,0.000192549,7.14793e-07,1.0081e-09,0.0153019,0.000193981,7.17817e-07,5.41244e-10,0.0154966,0.000195418,7.19441e-07,-3.7907e-10,0.0156928,0.000196856,7.18304e-07,1.90641e-09,0.0158903,0.000198298,7.24023e-07,-7.27387e-10,0.0160893,0.000199744,7.21841e-07,1.00317e-09,0.0162898,0.000201191,7.24851e-07,4.39949e-10,0.0164917,0.000202642,7.2617e-07,9.6234e-10,0.0166951,0.000204097,7.29057e-07,-5.64019e-10,0.0168999,0.000205554,7.27365e-07,1.29374e-09,0.0171062,0.000207012,7.31247e-07,9.77025e-10,0.017314,0.000208478,7.34178e-07,-1.47651e-09,0.0175232,0.000209942,7.29748e-07,3.06636e-09,0.0177338,0.00021141,7.38947e-07,-1.47573e-09,0.017946,0.000212884,7.3452e-07,9.7386e-10,0.0181596,0.000214356,7.37442e-07,1.30562e-09,0.0183747,0.000215835,7.41358e-07,-6.08376e-10,0.0185913,0.000217315,7.39533e-07,1.12785e-09,0.0188093,0.000218798,7.42917e-07,-1.77711e-10,0.0190289,0.000220283,7.42384e-07,1.44562e-09,0.0192499,0.000221772,7.46721e-07,-1.68825e-11,0.0194724,0.000223266,7.4667e-07,4.84533e-10,0.0196964,0.000224761,7.48124e-07,-5.85298e-11,0.0199219,0.000226257,7.47948e-07,1.61217e-09,0.0201489,0.000227757,7.52785e-07,-8.02136e-10,0.0203775,0.00022926,7.50378e-07,1.59637e-09,0.0206075,0.000230766,7.55167e-07,4.47168e-12,0.020839,0.000232276,7.55181e-07,2.48387e-10,0.021072,0.000233787,7.55926e-07,8.6474e-10,0.0213066,0.000235302,7.5852e-07,1.78299e-11,0.0215426,0.000236819,7.58573e-07,9.26567e-10,0.0217802,0.000238339,7.61353e-07,1.34529e-12,0.0220193,0.000239862,7.61357e-07,9.30659e-10,0.0222599,0.000241387,7.64149e-07,1.34529e-12,0.0225021,0.000242915,7.64153e-07,9.26567e-10,0.0227458,0.000244447,7.66933e-07,1.76215e-11,0.022991,0.00024598,7.66986e-07,8.65536e-10,0.0232377,0.000247517,7.69582e-07,2.45677e-10,0.023486,0.000249057,7.70319e-07,1.44193e-11,0.0237358,0.000250598,7.70363e-07,1.55918e-09,0.0239872,0.000252143,7.7504e-07,-6.63173e-10,0.0242401,0.000253691,7.73051e-07,1.09357e-09,0.0244946,0.000255241,7.76331e-07,1.41919e-11,0.0247506,0.000256793,7.76374e-07,7.12248e-10,0.0250082,0.000258348,7.78511e-07,8.62049e-10,0.0252673,0.000259908,7.81097e-07,-4.35061e-10,0.025528,0.000261469,7.79792e-07,8.7825e-10,0.0257902,0.000263031,7.82426e-07,6.47181e-10,0.0260541,0.000264598,7.84368e-07,2.58448e-10,0.0263194,0.000266167,7.85143e-07,1.81558e-10,0.0265864,0.000267738,7.85688e-07,8.78041e-10,0.0268549,0.000269312,7.88322e-07,3.15102e-11,0.027125,0.000270889,7.88417e-07,8.58525e-10,0.0273967,0.000272468,7.90992e-07,2.59812e-10,0.02767,0.000274051,7.91772e-07,-3.5224e-11,0.0279448,0.000275634,7.91666e-07,1.74377e-09,0.0282212,0.000277223,7.96897e-07,-1.35196e-09,0.0284992,0.000278813,7.92841e-07,1.80141e-09,0.0287788,0.000280404,7.98246e-07,-2.65629e-10,0.0290601,0.000281999,7.97449e-07,1.12374e-09,0.0293428,0.000283598,8.0082e-07,-5.04106e-10,0.0296272,0.000285198,7.99308e-07,8.92764e-10,0.0299132,0.000286799,8.01986e-07,6.58379e-10,0.0302008,0.000288405,8.03961e-07,1.98971e-10,0.0304901,0.000290014,8.04558e-07,4.08382e-10,0.0307809,0.000291624,8.05783e-07,3.01839e-11,0.0310733,0.000293236,8.05874e-07,1.33343e-09,0.0313673,0.000294851,8.09874e-07,2.2419e-10,0.031663,0.000296472,8.10547e-07,-3.67606e-10,0.0319603,0.000298092,8.09444e-07,1.24624e-09,0.0322592,0.000299714,8.13182e-07,-8.92025e-10,0.0325597,0.000301338,8.10506e-07,2.32183e-09,0.0328619,0.000302966,8.17472e-07,-9.44719e-10,0.0331657,0.000304598,8.14638e-07,1.45703e-09,0.0334711,0.000306232,8.19009e-07,-1.15805e-09,0.0337781,0.000307866,8.15535e-07,3.17507e-09,0.0340868,0.000309507,8.2506e-07,-4.09161e-09,0.0343971,0.000311145,8.12785e-07,5.74079e-09,0.0347091,0.000312788,8.30007e-07,-3.97034e-09,0.0350227,0.000314436,8.18096e-07,2.68985e-09,0.035338,0.00031608,8.26166e-07,6.61676e-10,0.0356549,0.000317734,8.28151e-07,-1.61123e-09,0.0359734,0.000319386,8.23317e-07,2.05786e-09,0.0362936,0.000321038,8.29491e-07,8.30388e-10,0.0366155,0.0003227,8.31982e-07,-1.65424e-09,0.036939,0.000324359,8.27019e-07,2.06129e-09,0.0372642,0.000326019,8.33203e-07,8.59719e-10,0.0375911,0.000327688,8.35782e-07,-1.77488e-09,0.0379196,0.000329354,8.30458e-07,2.51464e-09,0.0382498,0.000331023,8.38002e-07,-8.33135e-10,0.0385817,0.000332696,8.35502e-07,8.17825e-10,0.0389152,0.00033437,8.37956e-07,1.28718e-09,0.0392504,0.00033605,8.41817e-07,-2.2413e-09,0.0395873,0.000337727,8.35093e-07,3.95265e-09,0.0399258,0.000339409,8.46951e-07,-2.39332e-09,0.0402661,0.000341095,8.39771e-07,1.89533e-09,0.040608,0.000342781,8.45457e-07,-1.46271e-09,0.0409517,0.000344467,8.41069e-07,3.95554e-09,0.041297,0.000346161,8.52936e-07,-3.18369e-09,0.041644,0.000347857,8.43385e-07,1.32873e-09,0.0419927,0.000349548,8.47371e-07,1.59402e-09,0.0423431,0.000351248,8.52153e-07,-2.54336e-10,0.0426952,0.000352951,8.5139e-07,-5.76676e-10,0.043049,0.000354652,8.4966e-07,2.56114e-09,0.0434045,0.000356359,8.57343e-07,-2.21744e-09,0.0437617,0.000358067,8.50691e-07,2.58344e-09,0.0441206,0.000359776,8.58441e-07,-6.65826e-10,0.0444813,0.000361491,8.56444e-07,7.99218e-11,0.0448436,0.000363204,8.56684e-07,3.46063e-10,0.0452077,0.000364919,8.57722e-07,2.26116e-09,0.0455734,0.000366641,8.64505e-07,-1.94005e-09,0.045941,0.000368364,8.58685e-07,1.77384e-09,0.0463102,0.000370087,8.64007e-07,-1.43005e-09,0.0466811,0.000371811,8.59717e-07,3.94634e-09,0.0470538,0.000373542,8.71556e-07,-3.17946e-09,0.0474282,0.000375276,8.62017e-07,1.32104e-09,0.0478043,0.000377003,8.6598e-07,1.62045e-09,0.0481822,0.00037874,8.70842e-07,-3.52297e-10,0.0485618,0.000380481,8.69785e-07,-2.11211e-10,0.0489432,0.00038222,8.69151e-07,1.19716e-09,0.0493263,0.000383962,8.72743e-07,-8.52026e-10,0.0497111,0.000385705,8.70187e-07,2.21092e-09,0.0500977,0.000387452,8.76819e-07,-5.41339e-10,0.050486,0.000389204,8.75195e-07,-4.5361e-11,0.0508761,0.000390954,8.75059e-07,7.22669e-10,0.0512679,0.000392706,8.77227e-07,8.79936e-10,0.0516615,0.000394463,8.79867e-07,-5.17048e-10,0.0520568,0.000396222,8.78316e-07,1.18833e-09,0.0524539,0.000397982,8.81881e-07,-5.11022e-10,0.0528528,0.000399744,8.80348e-07,8.55683e-10,0.0532534,0.000401507,8.82915e-07,8.13562e-10,0.0536558,0.000403276,8.85356e-07,-3.84603e-10,0.05406,0.000405045,8.84202e-07,7.24962e-10,0.0544659,0.000406816,8.86377e-07,1.20986e-09,0.0548736,0.000408592,8.90006e-07,-1.83896e-09,0.0552831,0.000410367,8.84489e-07,2.42071e-09,0.0556944,0.000412143,8.91751e-07,-3.93413e-10,0.0561074,0.000413925,8.90571e-07,-8.46967e-10,0.0565222,0.000415704,8.8803e-07,3.78122e-09,0.0569388,0.000417491,8.99374e-07,-3.1021e-09,0.0573572,0.000419281,8.90068e-07,1.17658e-09,0.0577774,0.000421064,8.93597e-07,2.12117e-09,0.0581993,0.000422858,8.99961e-07,-2.21068e-09,0.0586231,0.000424651,8.93329e-07,2.9961e-09,0.0590486,0.000426447,9.02317e-07,-2.32311e-09,0.059476,0.000428244,8.95348e-07,2.57122e-09,0.0599051,0.000430043,9.03062e-07,-5.11098e-10,0.0603361,0.000431847,9.01528e-07,-5.27166e-10,0.0607688,0.000433649,8.99947e-07,2.61984e-09,0.0612034,0.000435457,9.07806e-07,-2.50141e-09,0.0616397,0.000437265,9.00302e-07,3.66045e-09,0.0620779,0.000439076,9.11283e-07,-4.68977e-09,0.0625179,0.000440885,8.97214e-07,7.64783e-09,0.0629597,0.000442702,9.20158e-07,-7.27499e-09,0.0634033,0.000444521,8.98333e-07,6.55113e-09,0.0638487,0.000446337,9.17986e-07,-4.02844e-09,0.0642959,0.000448161,9.05901e-07,2.11196e-09,0.064745,0.000449979,9.12236e-07,3.03125e-09,0.0651959,0.000451813,9.2133e-07,-6.78648e-09,0.0656486,0.000453635,9.00971e-07,9.21375e-09,0.0661032,0.000455464,9.28612e-07,-7.71684e-09,0.0665596,0.000457299,9.05462e-07,6.7522e-09,0.0670178,0.00045913,9.25718e-07,-4.3907e-09,0.0674778,0.000460968,9.12546e-07,3.36e-09,0.0679397,0.000462803,9.22626e-07,-1.59876e-09,0.0684034,0.000464644,9.1783e-07,3.0351e-09,0.068869,0.000466488,9.26935e-07,-3.09101e-09,0.0693364,0.000468333,9.17662e-07,1.8785e-09,0.0698057,0.000470174,9.23298e-07,3.02733e-09,0.0702768,0.00047203,9.3238e-07,-6.53722e-09,0.0707497,0.000473875,9.12768e-07,8.22054e-09,0.0712245,0.000475725,9.37429e-07,-3.99325e-09,0.0717012,0.000477588,9.2545e-07,3.01839e-10,0.0721797,0.00047944,9.26355e-07,2.78597e-09,0.0726601,0.000481301,9.34713e-07,-3.99507e-09,0.0731423,0.000483158,9.22728e-07,5.7435e-09,0.0736264,0.000485021,9.39958e-07,-4.07776e-09,0.0741123,0.000486888,9.27725e-07,3.11695e-09,0.0746002,0.000488753,9.37076e-07,-9.39394e-10,0.0750898,0.000490625,9.34258e-07,6.4055e-10,0.0755814,0.000492495,9.3618e-07,-1.62265e-09,0.0760748,0.000494363,9.31312e-07,5.84995e-09,0.0765701,0.000496243,9.48861e-07,-6.87601e-09,0.0770673,0.00049812,9.28233e-07,6.75296e-09,0.0775664,0.000499997,9.48492e-07,-5.23467e-09,0.0780673,0.000501878,9.32788e-07,6.73523e-09,0.0785701,0.000503764,9.52994e-07,-6.80514e-09,0.0790748,0.000505649,9.32578e-07,5.5842e-09,0.0795814,0.000507531,9.49331e-07,-6.30583e-10,0.0800899,0.000509428,9.47439e-07,-3.0618e-09,0.0806003,0.000511314,9.38254e-07,5.4273e-09,0.0811125,0.000513206,9.54536e-07,-3.74627e-09,0.0816267,0.000515104,9.43297e-07,2.10713e-09,0.0821427,0.000516997,9.49618e-07,2.76839e-09,0.0826607,0.000518905,9.57924e-07,-5.73006e-09,0.0831805,0.000520803,9.40733e-07,5.25072e-09,0.0837023,0.0005227,9.56486e-07,-3.71718e-10,0.084226,0.000524612,9.5537e-07,-3.76404e-09,0.0847515,0.000526512,9.44078e-07,7.97735e-09,0.085279,0.000528424,9.6801e-07,-5.79367e-09,0.0858084,0.000530343,9.50629e-07,2.96268e-10,0.0863397,0.000532245,9.51518e-07,4.6086e-09,0.0868729,0.000534162,9.65344e-07,-3.82947e-09,0.087408,0.000536081,9.53856e-07,3.25861e-09,0.087945,0.000537998,9.63631e-07,-1.7543e-09,0.088484,0.00053992,9.58368e-07,3.75849e-09,0.0890249,0.000541848,9.69644e-07,-5.82891e-09,0.0895677,0.00054377,9.52157e-07,4.65593e-09,0.0901124,0.000545688,9.66125e-07,2.10643e-09,0.0906591,0.000547627,9.72444e-07,-5.63099e-09,0.0912077,0.000549555,9.55551e-07,5.51627e-09,0.0917582,0.000551483,9.721e-07,-1.53292e-09,0.0923106,0.000553422,9.67501e-07,6.15311e-10,0.092865,0.000555359,9.69347e-07,-9.28291e-10,0.0934213,0.000557295,9.66562e-07,3.09774e-09,0.0939796,0.000559237,9.75856e-07,-4.01186e-09,0.0945398,0.000561177,9.6382e-07,5.49892e-09,0.095102,0.000563121,9.80317e-07,-3.08258e-09,0.0956661,0.000565073,9.71069e-07,-6.19176e-10,0.0962321,0.000567013,9.69212e-07,5.55932e-09,0.0968001,0.000568968,9.8589e-07,-6.71704e-09,0.09737,0.00057092,9.65738e-07,6.40762e-09,0.0979419,0.00057287,9.84961e-07,-4.0122e-09,0.0985158,0.000574828,9.72925e-07,2.19059e-09,0.0990916,0.000576781,9.79496e-07,2.70048e-09,0.0996693,0.000578748,9.87598e-07,-5.54193e-09,0.100249,0.000580706,9.70972e-07,4.56597e-09,0.100831,0.000582662,9.8467e-07,2.17923e-09,0.101414,0.000584638,9.91208e-07,-5.83232e-09,0.102,0.000586603,9.73711e-07,6.24884e-09,0.102588,0.000588569,9.92457e-07,-4.26178e-09,0.103177,0.000590541,9.79672e-07,3.34781e-09,0.103769,0.00059251,9.89715e-07,-1.67904e-09,0.104362,0.000594485,9.84678e-07,3.36839e-09,0.104958,0.000596464,9.94783e-07,-4.34397e-09,0.105555,0.000598441,9.81751e-07,6.55696e-09,0.106155,0.000600424,1.00142e-06,-6.98272e-09,0.106756,0.000602406,9.80474e-07,6.4728e-09,0.107359,0.000604386,9.99893e-07,-4.00742e-09,0.107965,0.000606374,9.8787e-07,2.10654e-09,0.108572,0.000608356,9.9419e-07,3.0318e-09,0.109181,0.000610353,1.00329e-06,-6.7832e-09,0.109793,0.00061234,9.82936e-07,9.1998e-09,0.110406,0.000614333,1.01054e-06,-7.6642e-09,0.111021,0.000616331,9.87543e-07,6.55579e-09,0.111639,0.000618326,1.00721e-06,-3.65791e-09,0.112258,0.000620329,9.96236e-07,6.25467e-10,0.112879,0.000622324,9.98113e-07,1.15593e-09,0.113503,0.000624323,1.00158e-06,2.20158e-09,0.114128,0.000626333,1.00819e-06,-2.51191e-09,0.114755,0.000628342,1.00065e-06,3.95517e-10,0.115385,0.000630345,1.00184e-06,9.29807e-10,0.116016,0.000632351,1.00463e-06,3.33599e-09,0.116649,0.00063437,1.01463e-06,-6.82329e-09,0.117285,0.000636379,9.94163e-07,9.05595e-09,0.117922,0.000638395,1.02133e-06,-7.04862e-09,0.118562,0.000640416,1.00019e-06,4.23737e-09,0.119203,0.000642429,1.0129e-06,-2.45033e-09,0.119847,0.000644448,1.00555e-06,5.56395e-09,0.120492,0.000646475,1.02224e-06,-4.9043e-09,0.121139,0.000648505,1.00753e-06,-8.47952e-10,0.121789,0.000650518,1.00498e-06,8.29622e-09,0.122441,0.000652553,1.02987e-06,-9.98538e-09,0.123094,0.000654582,9.99914e-07,9.2936e-09,0.12375,0.00065661,1.02779e-06,-4.83707e-09,0.124407,0.000658651,1.01328e-06,2.60411e-09,0.125067,0.000660685,1.0211e-06,-5.57945e-09,0.125729,0.000662711,1.00436e-06,1.22631e-08,0.126392,0.000664756,1.04115e-06,-1.36704e-08,0.127058,0.000666798,1.00014e-06,1.26161e-08,0.127726,0.000668836,1.03798e-06,-6.99155e-09,0.128396,0.000670891,1.01701e-06,4.48836e-10,0.129068,0.000672926,1.01836e-06,5.19606e-09,0.129742,0.000674978,1.03394e-06,-6.3319e-09,0.130418,0.000677027,1.01495e-06,5.2305e-09,0.131096,0.000679073,1.03064e-06,3.11123e-10,0.131776,0.000681135,1.03157e-06,-6.47511e-09,0.132458,0.000683179,1.01215e-06,1.06882e-08,0.133142,0.000685235,1.04421e-06,-6.47519e-09,0.133829,0.000687304,1.02479e-06,3.11237e-10,0.134517,0.000689355,1.02572e-06,5.23035e-09,0.135207,0.000691422,1.04141e-06,-6.3316e-09,0.1359,0.000693486,1.02242e-06,5.19484e-09,0.136594,0.000695546,1.038e-06,4.53497e-10,0.137291,0.000697623,1.03936e-06,-7.00891e-09,0.137989,0.000699681,1.01834e-06,1.2681e-08,0.13869,0.000701756,1.05638e-06,-1.39128e-08,0.139393,0.000703827,1.01464e-06,1.31679e-08,0.140098,0.000705896,1.05414e-06,-8.95659e-09,0.140805,0.000707977,1.02727e-06,7.75742e-09,0.141514,0.000710055,1.05055e-06,-7.17182e-09,0.142225,0.000712135,1.02903e-06,6.02862e-09,0.142938,0.000714211,1.04712e-06,-2.04163e-09,0.143653,0.000716299,1.04099e-06,2.13792e-09,0.144371,0.000718387,1.04741e-06,-6.51009e-09,0.14509,0.000720462,1.02787e-06,9.00123e-09,0.145812,0.000722545,1.05488e-06,3.07523e-10,0.146535,0.000724656,1.0558e-06,-1.02312e-08,0.147261,0.000726737,1.02511e-06,1.0815e-08,0.147989,0.000728819,1.05755e-06,-3.22681e-09,0.148719,0.000730925,1.04787e-06,2.09244e-09,0.14945,0.000733027,1.05415e-06,-5.143e-09,0.150185,0.00073512,1.03872e-06,3.57844e-09,0.150921,0.000737208,1.04946e-06,5.73027e-09,0.151659,0.000739324,1.06665e-06,-1.15983e-08,0.152399,0.000741423,1.03185e-06,1.08605e-08,0.153142,0.000743519,1.06443e-06,-2.04106e-09,0.153886,0.000745642,1.05831e-06,-2.69642e-09,0.154633,0.00074775,1.05022e-06,-2.07425e-09,0.155382,0.000749844,1.044e-06,1.09934e-08,0.156133,0.000751965,1.07698e-06,-1.20972e-08,0.156886,0.000754083,1.04069e-06,7.59288e-09,0.157641,0.000756187,1.06347e-06,-3.37305e-09,0.158398,0.000758304,1.05335e-06,5.89921e-09,0.159158,0.000760428,1.07104e-06,-5.32248e-09,0.159919,0.000762554,1.05508e-06,4.8927e-10,0.160683,0.000764666,1.05654e-06,3.36547e-09,0.161448,0.000766789,1.06664e-06,9.50081e-10,0.162216,0.000768925,1.06949e-06,-7.16568e-09,0.162986,0.000771043,1.04799e-06,1.28114e-08,0.163758,0.000773177,1.08643e-06,-1.42774e-08,0.164533,0.000775307,1.0436e-06,1.44956e-08,0.165309,0.000777438,1.08708e-06,-1.39025e-08,0.166087,0.00077957,1.04538e-06,1.13118e-08,0.166868,0.000781695,1.07931e-06,-1.54224e-09,0.167651,0.000783849,1.07468e-06,-5.14312e-09,0.168436,0.000785983,1.05925e-06,7.21381e-09,0.169223,0.000788123,1.0809e-06,-8.81096e-09,0.170012,0.000790259,1.05446e-06,1.31289e-08,0.170803,0.000792407,1.09385e-06,-1.39022e-08,0.171597,0.000794553,1.05214e-06,1.26775e-08,0.172392,0.000796695,1.09018e-06,-7.00557e-09,0.17319,0.000798855,1.06916e-06,4.43796e-10,0.17399,0.000800994,1.07049e-06,5.23031e-09,0.174792,0.000803151,1.08618e-06,-6.46397e-09,0.175596,0.000805304,1.06679e-06,5.72444e-09,0.176403,0.000807455,1.08396e-06,-1.53254e-09,0.177211,0.000809618,1.07937e-06,4.05673e-10,0.178022,0.000811778,1.08058e-06,-9.01916e-11,0.178835,0.000813939,1.08031e-06,-4.49821e-11,0.17965,0.000816099,1.08018e-06,2.70234e-10,0.180467,0.00081826,1.08099e-06,-1.03603e-09,0.181286,0.000820419,1.07788e-06,3.87392e-09,0.182108,0.000822587,1.0895e-06,4.41522e-10,0.182932,0.000824767,1.09083e-06,-5.63997e-09,0.183758,0.000826932,1.07391e-06,7.21707e-09,0.184586,0.000829101,1.09556e-06,-8.32718e-09,0.185416,0.000831267,1.07058e-06,1.11907e-08,0.186248,0.000833442,1.10415e-06,-6.63336e-09,0.187083,0.00083563,1.08425e-06,4.41484e-10,0.187919,0.0008378,1.08557e-06,4.86754e-09,0.188758,0.000839986,1.10017e-06,-5.01041e-09,0.189599,0.000842171,1.08514e-06,2.72811e-10,0.190443,0.000844342,1.08596e-06,3.91916e-09,0.191288,0.000846526,1.09772e-06,-1.04819e-09,0.192136,0.000848718,1.09457e-06,2.73531e-10,0.192985,0.000850908,1.0954e-06,-4.58916e-11,0.193837,0.000853099,1.09526e-06,-9.01158e-11,0.194692,0.000855289,1.09499e-06,4.06506e-10,0.195548,0.00085748,1.09621e-06,-1.53595e-09,0.196407,0.000859668,1.0916e-06,5.73717e-09,0.197267,0.000861869,1.10881e-06,-6.51164e-09,0.19813,0.000864067,1.08928e-06,5.40831e-09,0.198995,0.000866261,1.1055e-06,-2.20401e-10,0.199863,0.000868472,1.10484e-06,-4.52652e-09,0.200732,0.000870668,1.09126e-06,3.42508e-09,0.201604,0.000872861,1.10153e-06,5.72762e-09,0.202478,0.000875081,1.11872e-06,-1.14344e-08,0.203354,0.000877284,1.08441e-06,1.02076e-08,0.204233,0.000879484,1.11504e-06,4.06355e-10,0.205113,0.000881715,1.11626e-06,-1.18329e-08,0.205996,0.000883912,1.08076e-06,1.71227e-08,0.206881,0.000886125,1.13213e-06,-1.19546e-08,0.207768,0.000888353,1.09626e-06,8.93465e-10,0.208658,0.000890548,1.09894e-06,8.38062e-09,0.209549,0.000892771,1.12408e-06,-4.61353e-09,0.210443,0.000895006,1.11024e-06,-4.82756e-09,0.211339,0.000897212,1.09576e-06,9.02245e-09,0.212238,0.00089943,1.12283e-06,-1.45997e-09,0.213138,0.000901672,1.11845e-06,-3.18255e-09,0.214041,0.000903899,1.1089e-06,-7.11073e-10,0.214946,0.000906115,1.10677e-06,6.02692e-09,0.215853,0.000908346,1.12485e-06,-8.49548e-09,0.216763,0.00091057,1.09936e-06,1.30537e-08,0.217675,0.000912808,1.13852e-06,-1.3917e-08,0.218588,0.000915044,1.09677e-06,1.28121e-08,0.219505,0.000917276,1.13521e-06,-7.5288e-09,0.220423,0.000919523,1.11262e-06,2.40205e-09,0.221344,0.000921756,1.11983e-06,-2.07941e-09,0.222267,0.000923989,1.11359e-06,5.91551e-09,0.223192,0.000926234,1.13134e-06,-6.68149e-09,0.224119,0.000928477,1.11129e-06,5.90929e-09,0.225049,0.000930717,1.12902e-06,-2.05436e-09,0.22598,0.000932969,1.12286e-06,2.30807e-09,0.226915,0.000935222,1.12978e-06,-7.17796e-09,0.227851,0.00093746,1.10825e-06,1.15028e-08,0.228789,0.000939711,1.14276e-06,-9.03083e-09,0.22973,0.000941969,1.11566e-06,9.71932e-09,0.230673,0.00094423,1.14482e-06,-1.49452e-08,0.231619,0.000946474,1.09998e-06,2.02591e-08,0.232566,0.000948735,1.16076e-06,-2.13879e-08,0.233516,0.000950993,1.0966e-06,2.05888e-08,0.234468,0.000953247,1.15837e-06,-1.62642e-08,0.235423,0.000955515,1.10957e-06,1.46658e-08,0.236379,0.000957779,1.15357e-06,-1.25966e-08,0.237338,0.000960048,1.11578e-06,5.91793e-09,0.238299,0.000962297,1.13353e-06,3.82602e-09,0.239263,0.000964576,1.14501e-06,-6.3208e-09,0.240229,0.000966847,1.12605e-06,6.55613e-09,0.241197,0.000969119,1.14572e-06,-5.00268e-09,0.242167,0.000971395,1.13071e-06,-1.44659e-09,0.243139,0.000973652,1.12637e-06,1.07891e-08,0.244114,0.000975937,1.15874e-06,-1.19073e-08,0.245091,0.000978219,1.12302e-06,7.03782e-09,0.246071,0.000980486,1.14413e-06,-1.34276e-09,0.247052,0.00098277,1.1401e-06,-1.66669e-09,0.248036,0.000985046,1.1351e-06,8.00935e-09,0.249022,0.00098734,1.15913e-06,-1.54694e-08,0.250011,0.000989612,1.11272e-06,2.4066e-08,0.251002,0.000991909,1.18492e-06,-2.11901e-08,0.251995,0.000994215,1.12135e-06,1.08973e-09,0.25299,0.000996461,1.12462e-06,1.68311e-08,0.253988,0.000998761,1.17511e-06,-8.8094e-09,0.254987,0.00100109,1.14868e-06,-1.13958e-08,0.25599,0.00100335,1.1145e-06,2.45902e-08,0.256994,0.00100565,1.18827e-06,-2.73603e-08,0.258001,0.00100795,1.10618e-06,2.52464e-08,0.25901,0.00101023,1.18192e-06,-1.40207e-08,0.260021,0.00101256,1.13986e-06,1.03387e-09,0.261035,0.00101484,1.14296e-06,9.8853e-09,0.262051,0.00101715,1.17262e-06,-1.07726e-08,0.263069,0.00101947,1.1403e-06,3.40272e-09,0.26409,0.00102176,1.15051e-06,-2.83827e-09,0.265113,0.00102405,1.142e-06,7.95039e-09,0.266138,0.00102636,1.16585e-06,8.39047e-10,0.267166,0.00102869,1.16836e-06,-1.13066e-08,0.268196,0.00103099,1.13444e-06,1.4585e-08,0.269228,0.00103331,1.1782e-06,-1.72314e-08,0.270262,0.00103561,1.1265e-06,2.45382e-08,0.271299,0.00103794,1.20012e-06,-2.13166e-08,0.272338,0.00104028,1.13617e-06,1.12364e-09,0.273379,0.00104255,1.13954e-06,1.68221e-08,0.274423,0.00104488,1.19001e-06,-8.80736e-09,0.275469,0.00104723,1.16358e-06,-1.13948e-08,0.276518,0.00104953,1.1294e-06,2.45839e-08,0.277568,0.00105186,1.20315e-06,-2.73361e-08,0.278621,0.00105418,1.12114e-06,2.51559e-08,0.279677,0.0010565,1.19661e-06,-1.36832e-08,0.280734,0.00105885,1.15556e-06,-2.25706e-10,0.281794,0.00106116,1.15488e-06,1.45862e-08,0.282857,0.00106352,1.19864e-06,-2.83167e-08,0.283921,0.00106583,1.11369e-06,3.90759e-08,0.284988,0.00106817,1.23092e-06,-3.85801e-08,0.286058,0.00107052,1.11518e-06,2.58375e-08,0.287129,0.00107283,1.19269e-06,-5.16498e-09,0.288203,0.0010752,1.1772e-06,-5.17768e-09,0.28928,0.00107754,1.16167e-06,-3.92671e-09,0.290358,0.00107985,1.14988e-06,2.08846e-08,0.29144,0.00108221,1.21254e-06,-2.00072e-08,0.292523,0.00108458,1.15252e-06,-4.60659e-10,0.293609,0.00108688,1.15114e-06,2.18499e-08,0.294697,0.00108925,1.21669e-06,-2.73343e-08,0.295787,0.0010916,1.13468e-06,2.78826e-08,0.29688,0.00109395,1.21833e-06,-2.45915e-08,0.297975,0.00109632,1.14456e-06,1.08787e-08,0.299073,0.00109864,1.17719e-06,1.08788e-08,0.300172,0.00110102,1.20983e-06,-2.45915e-08,0.301275,0.00110337,1.13605e-06,2.78828e-08,0.302379,0.00110573,1.2197e-06,-2.73348e-08,0.303486,0.00110808,1.1377e-06,2.18518e-08,0.304595,0.00111042,1.20325e-06,-4.67556e-10,0.305707,0.00111283,1.20185e-06,-1.99816e-08,0.306821,0.00111517,1.14191e-06,2.07891e-08,0.307937,0.00111752,1.20427e-06,-3.57026e-09,0.309056,0.00111992,1.19356e-06,-6.50797e-09,0.310177,0.00112228,1.17404e-06,-2.00165e-10,0.3113,0.00112463,1.17344e-06,7.30874e-09,0.312426,0.001127,1.19536e-06,7.67424e-10,0.313554,0.00112939,1.19767e-06,-1.03784e-08,0.314685,0.00113176,1.16653e-06,1.09437e-08,0.315818,0.00113412,1.19936e-06,-3.59406e-09,0.316953,0.00113651,1.18858e-06,3.43251e-09,0.318091,0.0011389,1.19888e-06,-1.0136e-08,0.319231,0.00114127,1.16847e-06,7.30915e-09,0.320374,0.00114363,1.1904e-06,1.07018e-08,0.321518,0.00114604,1.2225e-06,-2.03137e-08,0.322666,0.00114842,1.16156e-06,1.09484e-08,0.323815,0.00115078,1.19441e-06,6.32224e-09,0.324967,0.00115319,1.21337e-06,-6.43509e-09,0.326122,0.00115559,1.19407e-06,-1.03842e-08,0.327278,0.00115795,1.16291e-06,1.81697e-08,0.328438,0.00116033,1.21742e-06,-2.6901e-09,0.329599,0.00116276,1.20935e-06,-7.40939e-09,0.330763,0.00116515,1.18713e-06,2.52533e-09,0.331929,0.00116754,1.1947e-06,-2.69191e-09,0.333098,0.00116992,1.18663e-06,8.24218e-09,0.334269,0.00117232,1.21135e-06,-4.74377e-10,0.335443,0.00117474,1.20993e-06,-6.34471e-09,0.336619,0.00117714,1.1909e-06,-3.94922e-09,0.337797,0.00117951,1.17905e-06,2.21417e-08,0.338978,0.00118193,1.24547e-06,-2.50128e-08,0.340161,0.00118435,1.17043e-06,1.8305e-08,0.341346,0.00118674,1.22535e-06,-1.84048e-08,0.342534,0.00118914,1.17013e-06,2.55121e-08,0.343725,0.00119156,1.24667e-06,-2.40389e-08,0.344917,0.00119398,1.17455e-06,1.10389e-08,0.346113,0.00119636,1.20767e-06,9.68574e-09,0.34731,0.0011988,1.23673e-06,-1.99797e-08,0.34851,0.00120122,1.17679e-06,1.06284e-08,0.349713,0.0012036,1.20867e-06,7.26868e-09,0.350917,0.00120604,1.23048e-06,-9.90072e-09,0.352125,0.00120847,1.20078e-06,2.53177e-09,0.353334,0.00121088,1.20837e-06,-2.26199e-10,0.354546,0.0012133,1.20769e-06,-1.62705e-09,0.355761,0.00121571,1.20281e-06,6.73435e-09,0.356978,0.00121813,1.22302e-06,4.49207e-09,0.358197,0.00122059,1.23649e-06,-2.47027e-08,0.359419,0.00122299,1.16238e-06,3.47142e-08,0.360643,0.00122542,1.26653e-06,-2.47472e-08,0.36187,0.00122788,1.19229e-06,4.66965e-09,0.363099,0.00123028,1.20629e-06,6.06872e-09,0.36433,0.00123271,1.2245e-06,8.57729e-10,0.365564,0.00123516,1.22707e-06,-9.49952e-09,0.366801,0.00123759,1.19858e-06,7.33792e-09,0.36804,0.00124001,1.22059e-06,9.95025e-09,0.369281,0.00124248,1.25044e-06,-1.73366e-08,0.370525,0.00124493,1.19843e-06,-2.08464e-10,0.371771,0.00124732,1.1978e-06,1.81704e-08,0.373019,0.00124977,1.25232e-06,-1.28683e-08,0.37427,0.00125224,1.21371e-06,3.50042e-09,0.375524,0.00125468,1.22421e-06,-1.1335e-09,0.37678,0.00125712,1.22081e-06,1.03345e-09,0.378038,0.00125957,1.22391e-06,-3.00023e-09,0.379299,0.00126201,1.21491e-06,1.09676e-08,0.380562,0.00126447,1.24781e-06,-1.10676e-08,0.381828,0.00126693,1.21461e-06,3.50042e-09,0.383096,0.00126937,1.22511e-06,-2.93403e-09,0.384366,0.00127181,1.21631e-06,8.23574e-09,0.385639,0.00127427,1.24102e-06,-2.06607e-10,0.386915,0.00127675,1.2404e-06,-7.40935e-09,0.388193,0.00127921,1.21817e-06,4.1761e-11,0.389473,0.00128165,1.21829e-06,7.24223e-09,0.390756,0.0012841,1.24002e-06,7.91564e-10,0.392042,0.00128659,1.2424e-06,-1.04086e-08,0.393329,0.00128904,1.21117e-06,1.10405e-08,0.39462,0.0012915,1.24429e-06,-3.951e-09,0.395912,0.00129397,1.23244e-06,4.7634e-09,0.397208,0.00129645,1.24673e-06,-1.51025e-08,0.398505,0.0012989,1.20142e-06,2.58443e-08,0.399805,0.00130138,1.27895e-06,-2.86702e-08,0.401108,0.00130385,1.19294e-06,2.92318e-08,0.402413,0.00130632,1.28064e-06,-2.86524e-08,0.403721,0.0013088,1.19468e-06,2.57731e-08,0.405031,0.00131127,1.272e-06,-1.48355e-08,0.406343,0.00131377,1.2275e-06,3.76652e-09,0.407658,0.00131623,1.23879e-06,-2.30784e-10,0.408976,0.00131871,1.2381e-06,-2.84331e-09,0.410296,0.00132118,1.22957e-06,1.16041e-08,0.411618,0.00132367,1.26438e-06,-1.37708e-08,0.412943,0.00132616,1.22307e-06,1.36768e-08,0.41427,0.00132865,1.2641e-06,-1.1134e-08,0.4156,0.00133114,1.2307e-06,1.05714e-09,0.416933,0.00133361,1.23387e-06,6.90538e-09,0.418267,0.00133609,1.25459e-06,1.12372e-09,0.419605,0.00133861,1.25796e-06,-1.14002e-08,0.420945,0.00134109,1.22376e-06,1.46747e-08,0.422287,0.00134358,1.26778e-06,-1.7496e-08,0.423632,0.00134606,1.21529e-06,2.5507e-08,0.424979,0.00134857,1.29182e-06,-2.49272e-08,0.426329,0.00135108,1.21703e-06,1.45972e-08,0.427681,0.00135356,1.26083e-06,-3.65935e-09,0.429036,0.00135607,1.24985e-06,4.00178e-11,0.430393,0.00135857,1.24997e-06,3.49917e-09,0.431753,0.00136108,1.26047e-06,-1.40366e-08,0.433116,0.00136356,1.21836e-06,2.28448e-08,0.43448,0.00136606,1.28689e-06,-1.77378e-08,0.435848,0.00136858,1.23368e-06,1.83043e-08,0.437218,0.0013711,1.28859e-06,-2.56769e-08,0.43859,0.0013736,1.21156e-06,2.47987e-08,0.439965,0.0013761,1.28595e-06,-1.39133e-08,0.441342,0.00137863,1.24421e-06,1.05202e-09,0.442722,0.00138112,1.24737e-06,9.70507e-09,0.444104,0.00138365,1.27649e-06,-1.00698e-08,0.445489,0.00138617,1.24628e-06,7.72123e-10,0.446877,0.00138867,1.24859e-06,6.98132e-09,0.448267,0.00139118,1.26954e-06,1.10477e-09,0.449659,0.00139373,1.27285e-06,-1.14003e-08,0.451054,0.00139624,1.23865e-06,1.4694e-08,0.452452,0.00139876,1.28273e-06,-1.75734e-08,0.453852,0.00140127,1.23001e-06,2.5797e-08,0.455254,0.00140381,1.3074e-06,-2.60097e-08,0.456659,0.00140635,1.22937e-06,1.86371e-08,0.458067,0.00140886,1.28529e-06,-1.8736e-08,0.459477,0.00141137,1.22908e-06,2.65048e-08,0.46089,0.00141391,1.30859e-06,-2.76784e-08,0.462305,0.00141645,1.22556e-06,2.46043e-08,0.463722,0.00141897,1.29937e-06,-1.11341e-08,0.465143,0.00142154,1.26597e-06,-9.87033e-09,0.466565,0.00142404,1.23636e-06,2.08131e-08,0.467991,0.00142657,1.2988e-06,-1.37773e-08,0.469419,0.00142913,1.25746e-06,4.49378e-09,0.470849,0.00143166,1.27094e-06,-4.19781e-09,0.472282,0.00143419,1.25835e-06,1.22975e-08,0.473717,0.00143674,1.29524e-06,-1.51902e-08,0.475155,0.00143929,1.24967e-06,1.86608e-08,0.476596,0.00144184,1.30566e-06,-2.96506e-08,0.478039,0.00144436,1.2167e-06,4.03368e-08,0.479485,0.00144692,1.33771e-06,-4.22896e-08,0.480933,0.00144947,1.21085e-06,3.94148e-08,0.482384,0.00145201,1.32909e-06,-2.59626e-08,0.483837,0.00145459,1.2512e-06,4.83124e-09,0.485293,0.0014571,1.2657e-06,6.63757e-09,0.486751,0.00145966,1.28561e-06,-1.57911e-09,0.488212,0.00146222,1.28087e-06,-3.21468e-10,0.489676,0.00146478,1.27991e-06,2.86517e-09,0.491142,0.00146735,1.2885e-06,-1.11392e-08,0.49261,0.00146989,1.25508e-06,1.18893e-08,0.494081,0.00147244,1.29075e-06,-6.61574e-09,0.495555,0.001475,1.27091e-06,1.45736e-08,0.497031,0.00147759,1.31463e-06,-2.18759e-08,0.49851,0.00148015,1.249e-06,1.33252e-08,0.499992,0.00148269,1.28897e-06,-1.62277e-09,0.501476,0.00148526,1.28411e-06,-6.83421e-09,0.502962,0.00148781,1.2636e-06,2.89596e-08,0.504451,0.00149042,1.35048e-06,-4.93997e-08,0.505943,0.00149298,1.20228e-06,4.94299e-08,0.507437,0.00149553,1.35057e-06,-2.91107e-08,0.508934,0.00149814,1.26324e-06,7.40848e-09,0.510434,0.00150069,1.28547e-06,-5.23187e-10,0.511936,0.00150326,1.2839e-06,-5.31585e-09,0.51344,0.00150581,1.26795e-06,2.17866e-08,0.514947,0.00150841,1.33331e-06,-2.22257e-08,0.516457,0.00151101,1.26663e-06,7.51178e-09,0.517969,0.00151357,1.28917e-06,-7.82128e-09,0.519484,0.00151613,1.2657e-06,2.37733e-08,0.521002,0.00151873,1.33702e-06,-2.76674e-08,0.522522,0.00152132,1.25402e-06,2.72917e-08,0.524044,0.00152391,1.3359e-06,-2.18949e-08,0.525569,0.00152652,1.27021e-06,6.83372e-10,0.527097,0.00152906,1.27226e-06,1.91613e-08,0.528628,0.00153166,1.32974e-06,-1.77241e-08,0.53016,0.00153427,1.27657e-06,-7.86963e-09,0.531696,0.0015368,1.25296e-06,4.92027e-08,0.533234,0.00153945,1.40057e-06,-6.9732e-08,0.534775,0.00154204,1.19138e-06,5.09114e-08,0.536318,0.00154458,1.34411e-06,-1.4704e-08,0.537864,0.00154722,1.3e-06,7.9048e-09,0.539413,0.00154984,1.32371e-06,-1.69152e-08,0.540964,0.00155244,1.27297e-06,1.51355e-10,0.542517,0.00155499,1.27342e-06,1.63099e-08,0.544074,0.00155758,1.32235e-06,-5.78647e-09,0.545633,0.00156021,1.30499e-06,6.83599e-09,0.547194,0.00156284,1.3255e-06,-2.15575e-08,0.548758,0.00156543,1.26083e-06,1.97892e-08,0.550325,0.00156801,1.32019e-06,2.00525e-09,0.551894,0.00157065,1.32621e-06,-2.78103e-08,0.553466,0.00157322,1.24278e-06,4.96314e-08,0.555041,0.00157586,1.39167e-06,-5.1506e-08,0.556618,0.00157849,1.23716e-06,3.71835e-08,0.558198,0.00158107,1.34871e-06,-3.76233e-08,0.55978,0.00158366,1.23584e-06,5.37052e-08,0.561365,0.00158629,1.39695e-06,-5.79884e-08,0.562953,0.00158891,1.22299e-06,5.90392e-08,0.564543,0.00159153,1.4001e-06,-5.89592e-08,0.566136,0.00159416,1.22323e-06,5.7588e-08,0.567731,0.00159678,1.39599e-06,-5.21835e-08,0.569329,0.00159941,1.23944e-06,3.19369e-08,0.57093,0.00160199,1.33525e-06,-1.59594e-08,0.572533,0.00160461,1.28737e-06,3.19006e-08,0.574139,0.00160728,1.38307e-06,-5.20383e-08,0.575748,0.00160989,1.22696e-06,5.70431e-08,0.577359,0.00161251,1.39809e-06,-5.69247e-08,0.578973,0.00161514,1.22731e-06,5.14463e-08,0.580589,0.00161775,1.38165e-06,-2.9651e-08,0.582208,0.00162042,1.2927e-06,7.55339e-09,0.58383,0.00162303,1.31536e-06,-5.62636e-10,0.585455,0.00162566,1.31367e-06,-5.30281e-09,0.587081,0.00162827,1.29776e-06,2.17738e-08,0.588711,0.00163093,1.36309e-06,-2.21875e-08,0.590343,0.00163359,1.29652e-06,7.37164e-09,0.591978,0.00163621,1.31864e-06,-7.29907e-09,0.593616,0.00163882,1.29674e-06,2.18247e-08,0.595256,0.00164148,1.36221e-06,-2.03952e-08,0.596899,0.00164414,1.30103e-06,1.51241e-10,0.598544,0.00164675,1.30148e-06,1.97902e-08,0.600192,0.00164941,1.36085e-06,-1.97074e-08,0.601843,0.00165207,1.30173e-06,-5.65175e-10,0.603496,0.00165467,1.30004e-06,2.1968e-08,0.605152,0.00165734,1.36594e-06,-2.77024e-08,0.606811,0.00165999,1.28283e-06,2.92369e-08,0.608472,0.00166264,1.37054e-06,-2.96407e-08,0.610136,0.00166529,1.28162e-06,2.97215e-08,0.611803,0.00166795,1.37079e-06,-2.96408e-08,0.613472,0.0016706,1.28186e-06,2.92371e-08,0.615144,0.00167325,1.36957e-06,-2.77031e-08,0.616819,0.00167591,1.28647e-06,2.19708e-08,0.618496,0.00167855,1.35238e-06,-5.75407e-10,0.620176,0.00168125,1.35065e-06,-1.9669e-08,0.621858,0.00168389,1.29164e-06,1.96468e-08,0.623544,0.00168653,1.35058e-06,6.86403e-10,0.625232,0.00168924,1.35264e-06,-2.23924e-08,0.626922,0.00169187,1.28547e-06,2.92788e-08,0.628615,0.00169453,1.3733e-06,-3.51181e-08,0.630311,0.00169717,1.26795e-06,5.15889e-08,0.63201,0.00169987,1.42272e-06,-5.2028e-08,0.633711,0.00170255,1.26663e-06,3.73139e-08,0.635415,0.0017052,1.37857e-06,-3.76227e-08,0.637121,0.00170784,1.2657e-06,5.35722e-08,0.63883,0.00171054,1.42642e-06,-5.74567e-08,0.640542,0.00171322,1.25405e-06,5.70456e-08,0.642257,0.0017159,1.42519e-06,-5.15163e-08,0.643974,0.00171859,1.27064e-06,2.98103e-08,0.645694,0.00172122,1.36007e-06,-8.12016e-09,0.647417,0.00172392,1.33571e-06,2.67039e-09,0.649142,0.0017266,1.34372e-06,-2.56152e-09,0.65087,0.00172928,1.33604e-06,7.57571e-09,0.6526,0.00173197,1.35876e-06,-2.77413e-08,0.654334,0.00173461,1.27554e-06,4.3785e-08,0.65607,0.00173729,1.40689e-06,-2.81896e-08,0.657808,0.00174002,1.32233e-06,9.36893e-09,0.65955,0.00174269,1.35043e-06,-9.28617e-09,0.661294,0.00174536,1.32257e-06,2.77757e-08,0.66304,0.00174809,1.4059e-06,-4.2212e-08,0.66479,0.00175078,1.27926e-06,2.1863e-08,0.666542,0.0017534,1.34485e-06,1.43648e-08,0.668297,0.00175613,1.38795e-06,-1.97177e-08,0.670054,0.00175885,1.3288e-06,4.90115e-09,0.671814,0.00176152,1.3435e-06,1.13232e-10,0.673577,0.00176421,1.34384e-06,-5.3542e-09,0.675343,0.00176688,1.32778e-06,2.13035e-08,0.677111,0.0017696,1.39169e-06,-2.02553e-08,0.678882,0.00177232,1.33092e-06,1.13005e-10,0.680656,0.00177499,1.33126e-06,1.98031e-08,0.682432,0.00177771,1.39067e-06,-1.97211e-08,0.684211,0.00178043,1.33151e-06,-5.2349e-10,0.685993,0.00178309,1.32994e-06,2.18151e-08,0.687777,0.00178582,1.39538e-06,-2.71325e-08,0.689564,0.00178853,1.31398e-06,2.71101e-08,0.691354,0.00179124,1.39531e-06,-2.17035e-08,0.693147,0.00179396,1.3302e-06,9.92865e-11,0.694942,0.00179662,1.3305e-06,2.13063e-08,0.69674,0.00179935,1.39442e-06,-2.57198e-08,0.698541,0.00180206,1.31726e-06,2.19682e-08,0.700344,0.00180476,1.38317e-06,-2.54852e-09,0.70215,0.00180752,1.37552e-06,-1.17741e-08,0.703959,0.00181023,1.3402e-06,-9.95999e-09,0.705771,0.00181288,1.31032e-06,5.16141e-08,0.707585,0.00181566,1.46516e-06,-7.72869e-08,0.709402,0.00181836,1.2333e-06,7.87197e-08,0.711222,0.00182106,1.46946e-06,-5.87781e-08,0.713044,0.00182382,1.29312e-06,3.71834e-08,0.714869,0.00182652,1.40467e-06,-3.03511e-08,0.716697,0.00182924,1.31362e-06,2.46161e-08,0.718528,0.00183194,1.38747e-06,-8.5087e-09,0.720361,0.00183469,1.36194e-06,9.41892e-09,0.722197,0.00183744,1.3902e-06,-2.91671e-08,0.724036,0.00184014,1.3027e-06,4.76448e-08,0.725878,0.00184288,1.44563e-06,-4.22028e-08,0.727722,0.00184565,1.31902e-06,1.95682e-09,0.729569,0.00184829,1.3249e-06,3.43754e-08,0.731419,0.00185104,1.42802e-06,-2.0249e-08,0.733271,0.00185384,1.36727e-06,-1.29838e-08,0.735126,0.00185654,1.32832e-06,1.25794e-08,0.736984,0.00185923,1.36606e-06,2.22711e-08,0.738845,0.00186203,1.43287e-06,-4.20594e-08,0.740708,0.00186477,1.3067e-06,2.67571e-08,0.742574,0.00186746,1.38697e-06,-5.36424e-09,0.744443,0.00187022,1.37087e-06,-5.30023e-09,0.746315,0.00187295,1.35497e-06,2.65653e-08,0.748189,0.00187574,1.43467e-06,-4.13564e-08,0.750066,0.00187848,1.3106e-06,1.9651e-08,0.751946,0.00188116,1.36955e-06,2.23572e-08,0.753828,0.00188397,1.43663e-06,-4.9475e-08,0.755714,0.00188669,1.2882e-06,5.63335e-08,0.757602,0.00188944,1.4572e-06,-5.66499e-08,0.759493,0.00189218,1.28725e-06,5.10567e-08,0.761386,0.00189491,1.44042e-06,-2.83677e-08,0.763283,0.00189771,1.35532e-06,2.80962e-09,0.765182,0.00190042,1.36375e-06,1.71293e-08,0.767083,0.0019032,1.41513e-06,-1.17221e-08,0.768988,0.001906,1.37997e-06,-2.98453e-08,0.770895,0.00190867,1.29043e-06,7.14987e-08,0.772805,0.00191146,1.50493e-06,-7.73354e-08,0.774718,0.00191424,1.27292e-06,5.90292e-08,0.776634,0.00191697,1.45001e-06,-3.9572e-08,0.778552,0.00191975,1.33129e-06,3.9654e-08,0.780473,0.00192253,1.45026e-06,-5.94395e-08,0.782397,0.00192525,1.27194e-06,7.88945e-08,0.784324,0.00192803,1.50862e-06,-7.73249e-08,0.786253,0.00193082,1.27665e-06,5.15913e-08,0.788185,0.00193352,1.43142e-06,-9.83099e-09,0.79012,0.00193636,1.40193e-06,-1.22672e-08,0.792058,0.00193912,1.36513e-06,-7.05275e-10,0.793999,0.00194185,1.36301e-06,1.50883e-08,0.795942,0.00194462,1.40828e-06,-4.33147e-11,0.797888,0.00194744,1.40815e-06,-1.49151e-08,0.799837,0.00195021,1.3634e-06,9.93244e-11,0.801788,0.00195294,1.3637e-06,1.45179e-08,0.803743,0.00195571,1.40725e-06,1.43363e-09,0.8057,0.00195853,1.41155e-06,-2.02525e-08,0.80766,0.00196129,1.35079e-06,1.99718e-08,0.809622,0.00196405,1.41071e-06,-3.01649e-11,0.811588,0.00196687,1.41062e-06,-1.9851e-08,0.813556,0.00196964,1.35107e-06,1.98296e-08,0.815527,0.0019724,1.41056e-06,1.37485e-10,0.817501,0.00197522,1.41097e-06,-2.03796e-08,0.819477,0.00197798,1.34983e-06,2.17763e-08,0.821457,0.00198074,1.41516e-06,-7.12085e-09,0.823439,0.00198355,1.3938e-06,6.70707e-09,0.825424,0.00198636,1.41392e-06,-1.97074e-08,0.827412,0.00198913,1.35479e-06,1.25179e-08,0.829402,0.00199188,1.39235e-06,2.92405e-08,0.831396,0.00199475,1.48007e-06,-6.98755e-08,0.833392,0.0019975,1.27044e-06,7.14477e-08,0.835391,0.00200026,1.48479e-06,-3.71014e-08,0.837392,0.00200311,1.37348e-06,1.73533e-08,0.839397,0.00200591,1.42554e-06,-3.23118e-08,0.841404,0.00200867,1.32861e-06,5.2289e-08,0.843414,0.00201148,1.48547e-06,-5.76348e-08,0.845427,0.00201428,1.31257e-06,5.9041e-08,0.847443,0.00201708,1.48969e-06,-5.93197e-08,0.849461,0.00201988,1.31173e-06,5.90289e-08,0.851482,0.00202268,1.48882e-06,-5.75864e-08,0.853507,0.00202549,1.31606e-06,5.21075e-08,0.855533,0.00202828,1.47238e-06,-3.16344e-08,0.857563,0.00203113,1.37748e-06,1.48257e-08,0.859596,0.00203393,1.42196e-06,-2.76684e-08,0.861631,0.00203669,1.33895e-06,3.62433e-08,0.863669,0.00203947,1.44768e-06,1.90463e-09,0.86571,0.00204237,1.45339e-06,-4.38617e-08,0.867754,0.00204515,1.32181e-06,5.43328e-08,0.8698,0.00204796,1.48481e-06,-5.42603e-08,0.87185,0.00205076,1.32203e-06,4.34989e-08,0.873902,0.00205354,1.45252e-06,-5.26029e-10,0.875957,0.00205644,1.45095e-06,-4.13949e-08,0.878015,0.00205922,1.32676e-06,4.68962e-08,0.880075,0.00206201,1.46745e-06,-2.69807e-08,0.882139,0.00206487,1.38651e-06,1.42181e-09,0.884205,0.00206764,1.39077e-06,2.12935e-08,0.886274,0.00207049,1.45465e-06,-2.69912e-08,0.888346,0.00207332,1.37368e-06,2.70664e-08,0.890421,0.00207615,1.45488e-06,-2.16698e-08,0.892498,0.00207899,1.38987e-06,8.14756e-12,0.894579,0.00208177,1.38989e-06,2.16371e-08,0.896662,0.00208462,1.45481e-06,-2.6952e-08,0.898748,0.00208744,1.37395e-06,2.65663e-08,0.900837,0.00209027,1.45365e-06,-1.97084e-08,0.902928,0.00209312,1.39452e-06,-7.33731e-09,0.905023,0.00209589,1.37251e-06,4.90578e-08,0.90712,0.00209878,1.51968e-06,-6.96845e-08,0.90922,0.00210161,1.31063e-06,5.08664e-08,0.911323,0.00210438,1.46323e-06,-1.45717e-08,0.913429,0.00210727,1.41952e-06,7.42038e-09,0.915538,0.00211013,1.44178e-06,-1.51097e-08,0.917649,0.00211297,1.39645e-06,-6.58618e-09,0.919764,0.00211574,1.37669e-06,4.14545e-08,0.921881,0.00211862,1.50105e-06,-4.00222e-08,0.924001,0.0021215,1.38099e-06,-5.7518e-10,0.926124,0.00212426,1.37926e-06,4.23229e-08,0.92825,0.00212714,1.50623e-06,-4.9507e-08,0.930378,0.00213001,1.35771e-06,3.64958e-08,0.93251,0.00213283,1.4672e-06,-3.68713e-08,0.934644,0.00213566,1.35658e-06,5.13848e-08,0.936781,0.00213852,1.51074e-06,-4.94585e-08,0.938921,0.0021414,1.36236e-06,2.72399e-08,0.941064,0.0021442,1.44408e-06,1.0372e-10,0.943209,0.00214709,1.44439e-06,-2.76547e-08,0.945358,0.0021499,1.36143e-06,5.09106e-08,0.947509,0.00215277,1.51416e-06,-5.67784e-08,0.949663,0.00215563,1.34382e-06,5.69935e-08,0.95182,0.00215849,1.5148e-06,-5.19861e-08,0.95398,0.00216136,1.35885e-06,3.17417e-08,0.956143,0.00216418,1.45407e-06,-1.53758e-08,0.958309,0.00216704,1.40794e-06,2.97615e-08,0.960477,0.00216994,1.49723e-06,-4.40657e-08,0.962649,0.00217281,1.36503e-06,2.72919e-08,0.964823,0.00217562,1.44691e-06,-5.49729e-09,0.967,0.0021785,1.43041e-06,-5.30273e-09,0.96918,0.00218134,1.41451e-06,2.67084e-08,0.971363,0.00218425,1.49463e-06,-4.19265e-08,0.973548,0.00218711,1.36885e-06,2.17881e-08,0.975737,0.00218992,1.43422e-06,1.43789e-08,0.977928,0.00219283,1.47735e-06,-1.96989e-08,0.980122,0.00219572,1.41826e-06,4.81221e-09,0.98232,0.00219857,1.43269e-06,4.50048e-10,0.98452,0.00220144,1.43404e-06,-6.61237e-09,0.986722,0.00220429,1.41421e-06,2.59993e-08,0.988928,0.0022072,1.4922e-06,-3.77803e-08,0.991137,0.00221007,1.37886e-06,5.9127e-09,0.993348,0.00221284,1.3966e-06,1.33339e-07,0.995563,0.00221604,1.79662e-06,-5.98872e-07,0.99778,0.00222015,0.,0.};

        template <bool srgb, int blueIdx, typename T, typename D>
        __device__ __forceinline__ void RGB2LabConvert_f(const T& src, D& dst)
        {
            const float _1_3 = 1.0f / 3.0f;
            const float _a = 16.0f / 116.0f;

            float B = blueIdx == 0 ? src.x : src.z;
            float G = src.y;
            float R = blueIdx == 0 ? src.z : src.x;

            if (srgb)
            {
                B = splineInterpolate(B * GAMMA_TAB_SIZE, c_sRGBGammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G * GAMMA_TAB_SIZE, c_sRGBGammaTab, GAMMA_TAB_SIZE);
                R = splineInterpolate(R * GAMMA_TAB_SIZE, c_sRGBGammaTab, GAMMA_TAB_SIZE);
            }

            float X = B * 0.189828f + G * 0.376219f + R * 0.433953f;
            float Y = B * 0.072169f + G * 0.715160f + R * 0.212671f;
            float Z = B * 0.872766f + G * 0.109477f + R * 0.017758f;

            float FX = X > 0.008856f ? ::powf(X, _1_3) : (7.787f * X + _a);
            float FY = Y > 0.008856f ? ::powf(Y, _1_3) : (7.787f * Y + _a);
            float FZ = Z > 0.008856f ? ::powf(Z, _1_3) : (7.787f * Z + _a);

            float L = Y > 0.008856f ? (116.f * FY - 16.f) : (903.3f * Y);
            float a = 500.f * (FX - FY);
            float b = 200.f * (FY - FZ);

            dst.x = L;
            dst.y = a;
            dst.z = b;
        }

        template <typename T, int scn, int dcn, bool srgb, int blueIdx> struct RGB2Lab;
        template <int scn, int dcn, bool srgb, int blueIdx>
        struct RGB2Lab<uchar, scn, dcn, srgb, blueIdx>
            : unary_function<typename TypeVec<uchar, scn>::vec_type, typename TypeVec<uchar, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<uchar, dcn>::vec_type operator ()(const typename TypeVec<uchar, scn>::vec_type& src) const
            {
                typename TypeVec<uchar, dcn>::vec_type dst;

                RGB2LabConvert_b<srgb, blueIdx>(src, dst);

                return dst;
            }
            __host__ __device__ __forceinline__ RGB2Lab() {}
            __host__ __device__ __forceinline__ RGB2Lab(const RGB2Lab&) {}
        };
        template <int scn, int dcn, bool srgb, int blueIdx>
        struct RGB2Lab<float, scn, dcn, srgb, blueIdx>
            : unary_function<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<float, dcn>::vec_type operator ()(const typename TypeVec<float, scn>::vec_type& src) const
            {
                typename TypeVec<float, dcn>::vec_type dst;

                RGB2LabConvert_f<srgb, blueIdx>(src, dst);

                return dst;
            }
            __host__ __device__ __forceinline__ RGB2Lab() {}
            __host__ __device__ __forceinline__ RGB2Lab(const RGB2Lab&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_RGB2Lab_TRAITS(name, scn, dcn, srgb, blueIdx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2Lab<T, scn, dcn, srgb, blueIdx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

    namespace color_detail
    {
        __constant__ float c_sRGBInvGammaTab[] = {0,0.0126255,0.,-8.33961e-06,0.0126172,0.0126005,-2.50188e-05,4.1698e-05,0.0252344,0.0126756,0.000100075,-0.000158451,0.0378516,0.0124004,-0.000375277,-0.000207393,0.0496693,0.0110276,-0.000997456,0.00016837,0.0598678,0.00953783,-0.000492346,2.07235e-05,0.068934,0.00861531,-0.000430176,3.62876e-05,0.0771554,0.00786382,-0.000321313,1.87625e-05,0.0847167,0.00727748,-0.000265025,1.53594e-05,0.0917445,0.00679351,-0.000218947,1.10545e-05,0.0983301,0.00638877,-0.000185784,8.66984e-06,0.104542,0.00604322,-0.000159774,6.82996e-06,0.110432,0.00574416,-0.000139284,5.51008e-06,0.116042,0.00548212,-0.000122754,4.52322e-06,0.121406,0.00525018,-0.000109184,3.75557e-06,0.126551,0.00504308,-9.79177e-05,3.17134e-06,0.131499,0.00485676,-8.84037e-05,2.68469e-06,0.13627,0.004688,-8.03496e-05,2.31725e-06,0.14088,0.00453426,-7.33978e-05,2.00868e-06,0.145343,0.00439349,-6.73718e-05,1.74775e-06,0.149671,0.00426399,-6.21286e-05,1.53547e-06,0.153875,0.00414434,-5.75222e-05,1.364e-06,0.157963,0.00403338,-5.34301e-05,1.20416e-06,0.161944,0.00393014,-4.98177e-05,1.09114e-06,0.165825,0.00383377,-4.65443e-05,9.57987e-07,0.169613,0.00374356,-4.36703e-05,8.88359e-07,0.173314,0.00365888,-4.10052e-05,7.7849e-07,0.176933,0.00357921,-3.86697e-05,7.36254e-07,0.180474,0.00350408,-3.6461e-05,6.42534e-07,0.183942,0.00343308,-3.45334e-05,6.12614e-07,0.187342,0.00336586,-3.26955e-05,5.42894e-07,0.190675,0.00330209,-3.10669e-05,5.08967e-07,0.193947,0.00324149,-2.954e-05,4.75977e-07,0.197159,0.00318383,-2.8112e-05,4.18343e-07,0.200315,0.00312887,-2.6857e-05,4.13651e-07,0.203418,0.00307639,-2.5616e-05,3.70847e-07,0.206469,0.00302627,-2.45035e-05,3.3813e-07,0.209471,0.00297828,-2.34891e-05,3.32999e-07,0.212426,0.0029323,-2.24901e-05,2.96826e-07,0.215336,0.00288821,-2.15996e-05,2.82736e-07,0.218203,0.00284586,-2.07514e-05,2.70961e-07,0.221029,0.00280517,-1.99385e-05,2.42744e-07,0.223814,0.00276602,-1.92103e-05,2.33277e-07,0.226561,0.0027283,-1.85105e-05,2.2486e-07,0.229271,0.00269195,-1.78359e-05,2.08383e-07,0.231945,0.00265691,-1.72108e-05,1.93305e-07,0.234585,0.00262307,-1.66308e-05,1.80687e-07,0.237192,0.00259035,-1.60888e-05,1.86632e-07,0.239766,0.00255873,-1.55289e-05,1.60569e-07,0.24231,0.00252815,-1.50472e-05,1.54566e-07,0.244823,0.00249852,-1.45835e-05,1.59939e-07,0.247307,0.00246983,-1.41037e-05,1.29549e-07,0.249763,0.00244202,-1.3715e-05,1.41429e-07,0.252191,0.00241501,-1.32907e-05,1.39198e-07,0.254593,0.00238885,-1.28731e-05,1.06444e-07,0.256969,0.00236342,-1.25538e-05,1.2048e-07,0.25932,0.00233867,-1.21924e-05,1.26892e-07,0.261647,0.00231467,-1.18117e-05,8.72084e-08,0.26395,0.00229131,-1.15501e-05,1.20323e-07,0.26623,0.00226857,-1.11891e-05,8.71514e-08,0.268487,0.00224645,-1.09276e-05,9.73165e-08,0.270723,0.00222489,-1.06357e-05,8.98259e-08,0.272937,0.00220389,-1.03662e-05,7.98218e-08,0.275131,0.00218339,-1.01267e-05,9.75254e-08,0.277304,0.00216343,-9.83416e-06,6.65195e-08,0.279458,0.00214396,-9.63461e-06,8.34313e-08,0.281592,0.00212494,-9.38431e-06,7.65919e-08,0.283708,0.00210641,-9.15454e-06,5.7236e-08,0.285805,0.00208827,-8.98283e-06,8.18939e-08,0.287885,0.00207055,-8.73715e-06,6.2224e-08,0.289946,0.00205326,-8.55047e-06,5.66388e-08,0.291991,0.00203633,-8.38056e-06,6.88491e-08,0.294019,0.00201978,-8.17401e-06,5.53955e-08,0.296031,0.00200359,-8.00782e-06,6.71971e-08,0.298027,0.00198778,-7.80623e-06,3.34439e-08,0.300007,0.00197227,-7.7059e-06,6.7248e-08,0.301971,0.00195706,-7.50416e-06,5.51915e-08,0.303921,0.00194221,-7.33858e-06,3.98124e-08,0.305856,0.00192766,-7.21915e-06,5.37795e-08,0.307776,0.00191338,-7.05781e-06,4.30919e-08,0.309683,0.00189939,-6.92853e-06,4.20744e-08,0.311575,0.00188566,-6.80231e-06,5.68321e-08,0.313454,0.00187223,-6.63181e-06,2.86195e-08,0.31532,0.00185905,-6.54595e-06,3.73075e-08,0.317172,0.00184607,-6.43403e-06,6.05684e-08,0.319012,0.00183338,-6.25233e-06,1.84426e-08,0.320839,0.00182094,-6.197e-06,4.44757e-08,0.322654,0.00180867,-6.06357e-06,4.20729e-08,0.324456,0.00179667,-5.93735e-06,2.56511e-08,0.326247,0.00178488,-5.8604e-06,3.41368e-08,0.328026,0.00177326,-5.75799e-06,4.64177e-08,0.329794,0.00176188,-5.61874e-06,1.86107e-08,0.33155,0.0017507,-5.5629e-06,2.81511e-08,0.333295,0.00173966,-5.47845e-06,4.75987e-08,0.335029,0.00172884,-5.33565e-06,1.98726e-08,0.336753,0.00171823,-5.27604e-06,2.19226e-08,0.338466,0.00170775,-5.21027e-06,4.14483e-08,0.340169,0.00169745,-5.08592e-06,2.09017e-08,0.341861,0.00168734,-5.02322e-06,2.39561e-08,0.343543,0.00167737,-4.95135e-06,3.22852e-08,0.345216,0.00166756,-4.85449e-06,2.57173e-08,0.346878,0.00165793,-4.77734e-06,1.38569e-08,0.348532,0.00164841,-4.73577e-06,3.80634e-08,0.350175,0.00163906,-4.62158e-06,1.27043e-08,0.35181,0.00162985,-4.58347e-06,3.03279e-08,0.353435,0.00162078,-4.49249e-06,1.49961e-08,0.355051,0.00161184,-4.4475e-06,2.88977e-08,0.356659,0.00160303,-4.3608e-06,1.84241e-08,0.358257,0.00159436,-4.30553e-06,1.6616e-08,0.359848,0.0015858,-4.25568e-06,3.43218e-08,0.361429,0.00157739,-4.15272e-06,-4.89172e-09,0.363002,0.00156907,-4.16739e-06,4.48498e-08,0.364567,0.00156087,-4.03284e-06,4.30676e-09,0.366124,0.00155282,-4.01992e-06,2.73303e-08,0.367673,0.00154486,-3.93793e-06,5.58036e-09,0.369214,0.001537,-3.92119e-06,3.97554e-08,0.370747,0.00152928,-3.80193e-06,-1.55904e-08,0.372272,0.00152163,-3.8487e-06,5.24081e-08,0.37379,0.00151409,-3.69147e-06,-1.52272e-08,0.375301,0.00150666,-3.73715e-06,3.83028e-08,0.376804,0.0014993,-3.62225e-06,1.10278e-08,0.378299,0.00149209,-3.58916e-06,6.99326e-09,0.379788,0.00148493,-3.56818e-06,2.06038e-08,0.381269,0.00147786,-3.50637e-06,2.98009e-08,0.382744,0.00147093,-3.41697e-06,-2.05978e-08,0.384211,0.00146404,-3.47876e-06,5.25899e-08,0.385672,0.00145724,-3.32099e-06,-1.09471e-08,0.387126,0.00145056,-3.35383e-06,2.10009e-08,0.388573,0.00144392,-3.29083e-06,1.63501e-08,0.390014,0.00143739,-3.24178e-06,3.00641e-09,0.391448,0.00143091,-3.23276e-06,3.12282e-08,0.392875,0.00142454,-3.13908e-06,-8.70932e-09,0.394297,0.00141824,-3.16521e-06,3.34114e-08,0.395712,0.00141201,-3.06497e-06,-5.72754e-09,0.397121,0.00140586,-3.08215e-06,1.9301e-08,0.398524,0.00139975,-3.02425e-06,1.7931e-08,0.39992,0.00139376,-2.97046e-06,-1.61822e-09,0.401311,0.00138781,-2.97531e-06,1.83442e-08,0.402696,0.00138192,-2.92028e-06,1.76485e-08,0.404075,0.00137613,-2.86733e-06,4.68617e-10,0.405448,0.00137039,-2.86593e-06,1.02794e-08,0.406816,0.00136469,-2.83509e-06,1.80179e-08,0.408178,0.00135908,-2.78104e-06,7.05594e-09,0.409534,0.00135354,-2.75987e-06,1.33633e-08,0.410885,0.00134806,-2.71978e-06,-9.04568e-10,0.41223,0.00134261,-2.72249e-06,2.0057e-08,0.41357,0.00133723,-2.66232e-06,1.00841e-08,0.414905,0.00133194,-2.63207e-06,-7.88835e-10,0.416234,0.00132667,-2.63444e-06,2.28734e-08,0.417558,0.00132147,-2.56582e-06,-1.29785e-09,0.418877,0.00131633,-2.56971e-06,1.21205e-08,0.420191,0.00131123,-2.53335e-06,1.24202e-08,0.421499,0.0013062,-2.49609e-06,-2.19681e-09,0.422803,0.0013012,-2.50268e-06,2.61696e-08,0.424102,0.00129628,-2.42417e-06,-1.30747e-08,0.425396,0.00129139,-2.46339e-06,2.6129e-08,0.426685,0.00128654,-2.38501e-06,-2.03454e-09,0.427969,0.00128176,-2.39111e-06,1.18115e-08,0.429248,0.00127702,-2.35567e-06,1.43932e-08,0.430523,0.00127235,-2.31249e-06,-9.77965e-09,0.431793,0.00126769,-2.34183e-06,2.47253e-08,0.433058,0.00126308,-2.26766e-06,2.85278e-10,0.434319,0.00125855,-2.2668e-06,3.93614e-09,0.435575,0.00125403,-2.25499e-06,1.37722e-08,0.436827,0.00124956,-2.21368e-06,5.79803e-10,0.438074,0.00124513,-2.21194e-06,1.37112e-08,0.439317,0.00124075,-2.1708e-06,4.17973e-09,0.440556,0.00123642,-2.15826e-06,-6.27703e-10,0.44179,0.0012321,-2.16015e-06,2.81332e-08,0.44302,0.00122787,-2.07575e-06,-2.24985e-08,0.444246,0.00122365,-2.14324e-06,3.20586e-08,0.445467,0.00121946,-2.04707e-06,-1.6329e-08,0.446685,0.00121532,-2.09605e-06,3.32573e-08,0.447898,0.00121122,-1.99628e-06,-2.72927e-08,0.449107,0.00120715,-2.07816e-06,4.6111e-08,0.450312,0.00120313,-1.93983e-06,-3.79416e-08,0.451514,0.00119914,-2.05365e-06,4.60507e-08,0.452711,0.00119517,-1.9155e-06,-2.7052e-08,0.453904,0.00119126,-1.99666e-06,3.23551e-08,0.455093,0.00118736,-1.89959e-06,-1.29613e-08,0.456279,0.00118352,-1.93848e-06,1.94905e-08,0.45746,0.0011797,-1.88e-06,-5.39588e-09,0.458638,0.00117593,-1.89619e-06,2.09282e-09,0.459812,0.00117214,-1.88991e-06,2.68267e-08,0.460982,0.00116844,-1.80943e-06,-1.99925e-08,0.462149,0.00116476,-1.86941e-06,2.3341e-08,0.463312,0.00116109,-1.79939e-06,-1.37674e-08,0.464471,0.00115745,-1.84069e-06,3.17287e-08,0.465627,0.00115387,-1.7455e-06,-2.37407e-08,0.466779,0.00115031,-1.81673e-06,3.34315e-08,0.467927,0.00114677,-1.71643e-06,-2.05786e-08,0.469073,0.00114328,-1.77817e-06,1.90802e-08,0.470214,0.00113978,-1.72093e-06,3.86247e-09,0.471352,0.00113635,-1.70934e-06,-4.72759e-09,0.472487,0.00113292,-1.72352e-06,1.50478e-08,0.473618,0.00112951,-1.67838e-06,4.14108e-09,0.474746,0.00112617,-1.66595e-06,-1.80986e-09,0.47587,0.00112283,-1.67138e-06,3.09816e-09,0.476991,0.0011195,-1.66209e-06,1.92198e-08,0.478109,0.00111623,-1.60443e-06,-2.03726e-08,0.479224,0.00111296,-1.66555e-06,3.2468e-08,0.480335,0.00110973,-1.56814e-06,-2.00922e-08,0.481443,0.00110653,-1.62842e-06,1.80983e-08,0.482548,0.00110333,-1.57413e-06,7.30362e-09,0.48365,0.0011002,-1.55221e-06,-1.75107e-08,0.484749,0.00109705,-1.60475e-06,3.29373e-08,0.485844,0.00109393,-1.50594e-06,-2.48315e-08,0.486937,0.00109085,-1.58043e-06,3.65865e-08,0.488026,0.0010878,-1.47067e-06,-3.21078e-08,0.489112,0.00108476,-1.56699e-06,3.22397e-08,0.490195,0.00108172,-1.47027e-06,-7.44391e-09,0.491276,0.00107876,-1.49261e-06,-2.46428e-09,0.492353,0.00107577,-1.5e-06,1.73011e-08,0.493427,0.00107282,-1.4481e-06,-7.13552e-09,0.494499,0.0010699,-1.4695e-06,1.1241e-08,0.495567,0.001067,-1.43578e-06,-8.02637e-09,0.496633,0.0010641,-1.45986e-06,2.08645e-08,0.497695,0.00106124,-1.39726e-06,-1.58271e-08,0.498755,0.0010584,-1.44475e-06,1.26415e-08,0.499812,0.00105555,-1.40682e-06,2.48655e-08,0.500866,0.00105281,-1.33222e-06,-5.24988e-08,0.501918,0.00104999,-1.48972e-06,6.59206e-08,0.502966,0.00104721,-1.29196e-06,-3.237e-08,0.504012,0.00104453,-1.38907e-06,3.95479e-09,0.505055,0.00104176,-1.3772e-06,1.65509e-08,0.506096,0.00103905,-1.32755e-06,-1.05539e-08,0.507133,0.00103637,-1.35921e-06,2.56648e-08,0.508168,0.00103373,-1.28222e-06,-3.25007e-08,0.509201,0.00103106,-1.37972e-06,4.47336e-08,0.51023,0.00102844,-1.24552e-06,-2.72245e-08,0.511258,0.00102587,-1.32719e-06,4.55952e-09,0.512282,0.00102323,-1.31352e-06,8.98645e-09,0.513304,0.00102063,-1.28656e-06,1.90992e-08,0.514323,0.00101811,-1.22926e-06,-2.57786e-08,0.51534,0.00101557,-1.30659e-06,2.44104e-08,0.516355,0.00101303,-1.23336e-06,-1.22581e-08,0.517366,0.00101053,-1.27014e-06,2.4622e-08,0.518376,0.00100806,-1.19627e-06,-2.66253e-08,0.519383,0.00100559,-1.27615e-06,2.22744e-08,0.520387,0.00100311,-1.20932e-06,-2.8679e-09,0.521389,0.00100068,-1.21793e-06,-1.08029e-08,0.522388,0.000998211,-1.25034e-06,4.60795e-08,0.523385,0.000995849,-1.1121e-06,-5.4306e-08,0.52438,0.000993462,-1.27502e-06,5.19354e-08,0.525372,0.000991067,-1.11921e-06,-3.42262e-08,0.526362,0.000988726,-1.22189e-06,2.53646e-08,0.52735,0.000986359,-1.14579e-06,-7.62782e-09,0.528335,0.000984044,-1.16868e-06,5.14668e-09,0.529318,0.000981722,-1.15324e-06,-1.29589e-08,0.530298,0.000979377,-1.19211e-06,4.66888e-08,0.531276,0.000977133,-1.05205e-06,-5.45868e-08,0.532252,0.000974865,-1.21581e-06,5.24495e-08,0.533226,0.000972591,-1.05846e-06,-3.60019e-08,0.534198,0.000970366,-1.16647e-06,3.19537e-08,0.535167,0.000968129,-1.07061e-06,-3.2208e-08,0.536134,0.000965891,-1.16723e-06,3.72738e-08,0.537099,0.000963668,-1.05541e-06,2.32205e-09,0.538061,0.000961564,-1.04844e-06,-4.65618e-08,0.539022,0.000959328,-1.18813e-06,6.47159e-08,0.53998,0.000957146,-9.93979e-07,-3.3488e-08,0.540936,0.000955057,-1.09444e-06,9.63166e-09,0.54189,0.000952897,-1.06555e-06,-5.03871e-09,0.542842,0.000950751,-1.08066e-06,1.05232e-08,0.543792,0.000948621,-1.04909e-06,2.25503e-08,0.544739,0.000946591,-9.81444e-07,-4.11195e-08,0.545685,0.000944504,-1.1048e-06,2.27182e-08,0.546628,0.000942363,-1.03665e-06,9.85146e-09,0.54757,0.000940319,-1.00709e-06,-2.51938e-09,0.548509,0.000938297,-1.01465e-06,2.25858e-10,0.549446,0.000936269,-1.01397e-06,1.61598e-09,0.550381,0.000934246,-1.00913e-06,-6.68983e-09,0.551315,0.000932207,-1.0292e-06,2.51434e-08,0.552246,0.000930224,-9.53765e-07,-3.42793e-08,0.553175,0.000928214,-1.0566e-06,5.23688e-08,0.554102,0.000926258,-8.99497e-07,-5.59865e-08,0.555028,0.000924291,-1.06746e-06,5.23679e-08,0.555951,0.000922313,-9.10352e-07,-3.42763e-08,0.556872,0.00092039,-1.01318e-06,2.51326e-08,0.557792,0.000918439,-9.37783e-07,-6.64954e-09,0.558709,0.000916543,-9.57732e-07,1.46554e-09,0.559625,0.000914632,-9.53335e-07,7.87281e-10,0.560538,0.000912728,-9.50973e-07,-4.61466e-09,0.56145,0.000910812,-9.64817e-07,1.76713e-08,0.56236,0.000908935,-9.11804e-07,-6.46564e-09,0.563268,0.000907092,-9.312e-07,8.19121e-09,0.564174,0.000905255,-9.06627e-07,-2.62992e-08,0.565078,0.000903362,-9.85524e-07,3.74007e-08,0.565981,0.000901504,-8.73322e-07,-4.0942e-09,0.566882,0.000899745,-8.85605e-07,-2.1024e-08,0.56778,0.00089791,-9.48677e-07,2.85854e-08,0.568677,0.000896099,-8.62921e-07,-3.3713e-08,0.569573,0.000894272,-9.64059e-07,4.6662e-08,0.570466,0.000892484,-8.24073e-07,-3.37258e-08,0.571358,0.000890734,-9.25251e-07,2.86365e-08,0.572247,0.00088897,-8.39341e-07,-2.12155e-08,0.573135,0.000887227,-9.02988e-07,-3.37913e-09,0.574022,0.000885411,-9.13125e-07,3.47319e-08,0.574906,0.000883689,-8.08929e-07,-1.63394e-08,0.575789,0.000882022,-8.57947e-07,-2.8979e-08,0.57667,0.00088022,-9.44885e-07,7.26509e-08,0.57755,0.000878548,-7.26932e-07,-8.28106e-08,0.578427,0.000876845,-9.75364e-07,7.97774e-08,0.579303,0.000875134,-7.36032e-07,-5.74849e-08,0.580178,0.00087349,-9.08486e-07,3.09529e-08,0.58105,0.000871765,-8.15628e-07,-6.72206e-09,0.581921,0.000870114,-8.35794e-07,-4.06451e-09,0.582791,0.00086843,-8.47987e-07,2.29799e-08,0.583658,0.000866803,-7.79048e-07,-2.82503e-08,0.584524,0.00086516,-8.63799e-07,3.04167e-08,0.585388,0.000863524,-7.72548e-07,-3.38119e-08,0.586251,0.000861877,-8.73984e-07,4.52264e-08,0.587112,0.000860265,-7.38305e-07,-2.78842e-08,0.587972,0.000858705,-8.21958e-07,6.70567e-09,0.58883,0.000857081,-8.01841e-07,1.06161e-09,0.589686,0.000855481,-7.98656e-07,-1.09521e-08,0.590541,0.00085385,-8.31512e-07,4.27468e-08,0.591394,0.000852316,-7.03272e-07,-4.08257e-08,0.592245,0.000850787,-8.25749e-07,1.34677e-09,0.593095,0.000849139,-8.21709e-07,3.54387e-08,0.593944,0.000847602,-7.15393e-07,-2.38924e-08,0.59479,0.0008461,-7.8707e-07,5.26143e-10,0.595636,0.000844527,-7.85491e-07,2.17879e-08,0.596479,0.000843021,-7.20127e-07,-2.80733e-08,0.597322,0.000841497,-8.04347e-07,3.09005e-08,0.598162,0.000839981,-7.11646e-07,-3.5924e-08,0.599002,0.00083845,-8.19418e-07,5.3191e-08,0.599839,0.000836971,-6.59845e-07,-5.76307e-08,0.600676,0.000835478,-8.32737e-07,5.81227e-08,0.60151,0.000833987,-6.58369e-07,-5.56507e-08,0.602344,0.000832503,-8.25321e-07,4.52706e-08,0.603175,0.000830988,-6.89509e-07,-6.22236e-09,0.604006,0.000829591,-7.08176e-07,-2.03811e-08,0.604834,0.000828113,-7.6932e-07,2.8142e-08,0.605662,0.000826659,-6.84894e-07,-3.25822e-08,0.606488,0.000825191,-7.8264e-07,4.25823e-08,0.607312,0.000823754,-6.54893e-07,-1.85376e-08,0.608135,0.000822389,-7.10506e-07,-2.80365e-08,0.608957,0.000820883,-7.94616e-07,7.1079e-08,0.609777,0.000819507,-5.81379e-07,-7.74655e-08,0.610596,0.000818112,-8.13775e-07,5.9969e-08,0.611413,0.000816665,-6.33868e-07,-4.32013e-08,0.612229,0.000815267,-7.63472e-07,5.32313e-08,0.613044,0.0008139,-6.03778e-07,-5.05148e-08,0.613857,0.000812541,-7.55323e-07,2.96187e-08,0.614669,0.000811119,-6.66466e-07,-8.35545e-09,0.615479,0.000809761,-6.91533e-07,3.80301e-09,0.616288,0.00080839,-6.80124e-07,-6.85666e-09,0.617096,0.000807009,-7.00694e-07,2.36237e-08,0.617903,0.000805678,-6.29822e-07,-2.80336e-08,0.618708,0.000804334,-7.13923e-07,2.8906e-08,0.619511,0.000802993,-6.27205e-07,-2.79859e-08,0.620314,0.000801655,-7.11163e-07,2.34329e-08,0.621114,0.000800303,-6.40864e-07,-6.14108e-09,0.621914,0.000799003,-6.59287e-07,1.13151e-09,0.622712,0.000797688,-6.55893e-07,1.61507e-09,0.62351,0.000796381,-6.51048e-07,-7.59186e-09,0.624305,0.000795056,-6.73823e-07,2.87524e-08,0.6251,0.000793794,-5.87566e-07,-4.7813e-08,0.625893,0.000792476,-7.31005e-07,4.32901e-08,0.626685,0.000791144,-6.01135e-07,-6.13814e-09,0.627475,0.000789923,-6.19549e-07,-1.87376e-08,0.628264,0.000788628,-6.75762e-07,2.14837e-08,0.629052,0.000787341,-6.11311e-07,-7.59265e-09,0.629839,0.000786095,-6.34089e-07,8.88692e-09,0.630625,0.000784854,-6.07428e-07,-2.7955e-08,0.631409,0.000783555,-6.91293e-07,4.33285e-08,0.632192,0.000782302,-5.61307e-07,-2.61497e-08,0.632973,0.000781101,-6.39757e-07,1.6658e-09,0.633754,0.000779827,-6.34759e-07,1.94866e-08,0.634533,0.000778616,-5.76299e-07,-2.00076e-08,0.635311,0.000777403,-6.36322e-07,9.39091e-10,0.636088,0.000776133,-6.33505e-07,1.62512e-08,0.636863,0.000774915,-5.84751e-07,-6.33937e-09,0.637638,0.000773726,-6.03769e-07,9.10609e-09,0.638411,0.000772546,-5.76451e-07,-3.00849e-08,0.639183,0.000771303,-6.66706e-07,5.1629e-08,0.639953,0.000770125,-5.11819e-07,-5.7222e-08,0.640723,0.000768929,-6.83485e-07,5.80497e-08,0.641491,0.000767736,-5.09336e-07,-5.57674e-08,0.642259,0.000766551,-6.76638e-07,4.58105e-08,0.643024,0.000765335,-5.39206e-07,-8.26541e-09,0.643789,0.000764231,-5.64002e-07,-1.27488e-08,0.644553,0.000763065,-6.02249e-07,-3.44168e-10,0.645315,0.00076186,-6.03281e-07,1.41254e-08,0.646077,0.000760695,-5.60905e-07,3.44727e-09,0.646837,0.000759584,-5.50563e-07,-2.79144e-08,0.647596,0.000758399,-6.34307e-07,4.86057e-08,0.648354,0.000757276,-4.88489e-07,-4.72989e-08,0.64911,0.000756158,-6.30386e-07,2.13807e-08,0.649866,0.000754961,-5.66244e-07,2.13808e-08,0.65062,0.000753893,-5.02102e-07,-4.7299e-08,0.651374,0.000752746,-6.43999e-07,4.86059e-08,0.652126,0.000751604,-4.98181e-07,-2.79154e-08,0.652877,0.000750524,-5.81927e-07,3.45089e-09,0.653627,0.000749371,-5.71575e-07,1.41119e-08,0.654376,0.00074827,-5.29239e-07,-2.93748e-10,0.655123,0.00074721,-5.3012e-07,-1.29368e-08,0.65587,0.000746111,-5.68931e-07,-7.56355e-09,0.656616,0.000744951,-5.91621e-07,4.3191e-08,0.65736,0.000743897,-4.62048e-07,-4.59911e-08,0.658103,0.000742835,-6.00022e-07,2.15642e-08,0.658846,0.0007417,-5.35329e-07,1.93389e-08,0.659587,0.000740687,-4.77312e-07,-3.93152e-08,0.660327,0.000739615,-5.95258e-07,1.87126e-08,0.661066,0.00073848,-5.3912e-07,2.40695e-08,0.661804,0.000737474,-4.66912e-07,-5.53859e-08,0.662541,0.000736374,-6.33069e-07,7.82648e-08,0.663277,0.000735343,-3.98275e-07,-7.88593e-08,0.664012,0.00073431,-6.34853e-07,5.83585e-08,0.664745,0.000733215,-4.59777e-07,-3.53656e-08,0.665478,0.000732189,-5.65874e-07,2.34994e-08,0.66621,0.000731128,-4.95376e-07,9.72743e-10,0.66694,0.00073014,-4.92458e-07,-2.73903e-08,0.66767,0.000729073,-5.74629e-07,4.89839e-08,0.668398,0.000728071,-4.27677e-07,-4.93359e-08,0.669126,0.000727068,-5.75685e-07,2.91504e-08,0.669853,0.000726004,-4.88234e-07,-7.66109e-09,0.670578,0.000725004,-5.11217e-07,1.49392e-09,0.671303,0.000723986,-5.06735e-07,1.68533e-09,0.672026,0.000722978,-5.01679e-07,-8.23525e-09,0.672749,0.00072195,-5.26385e-07,3.12556e-08,0.67347,0.000720991,-4.32618e-07,-5.71825e-08,0.674191,0.000719954,-6.04166e-07,7.8265e-08,0.67491,0.00071898,-3.69371e-07,-7.70634e-08,0.675628,0.00071801,-6.00561e-07,5.11747e-08,0.676346,0.000716963,-4.47037e-07,-8.42615e-09,0.677062,0.000716044,-4.72315e-07,-1.747e-08,0.677778,0.000715046,-5.24725e-07,1.87015e-08,0.678493,0.000714053,-4.68621e-07,2.26856e-09,0.679206,0.000713123,-4.61815e-07,-2.77758e-08,0.679919,0.000712116,-5.45142e-07,4.92298e-08,0.68063,0.000711173,-3.97453e-07,-4.99339e-08,0.681341,0.000710228,-5.47255e-07,3.12967e-08,0.682051,0.000709228,-4.53365e-07,-1.56481e-08,0.68276,0.000708274,-5.00309e-07,3.12958e-08,0.683467,0.000707367,-4.06422e-07,-4.99303e-08,0.684174,0.000706405,-5.56213e-07,4.9216e-08,0.68488,0.00070544,-4.08565e-07,-2.77245e-08,0.685585,0.00070454,-4.91738e-07,2.07748e-09,0.686289,0.000703562,-4.85506e-07,1.94146e-08,0.686992,0.00070265,-4.27262e-07,-2.01314e-08,0.687695,0.000701735,-4.87656e-07,1.50616e-09,0.688396,0.000700764,-4.83137e-07,1.41067e-08,0.689096,0.00069984,-4.40817e-07,1.67168e-09,0.689795,0.000698963,-4.35802e-07,-2.07934e-08,0.690494,0.000698029,-4.98182e-07,2.18972e-08,0.691192,0.000697099,-4.32491e-07,-7.19092e-09,0.691888,0.000696212,-4.54064e-07,6.86642e-09,0.692584,0.000695325,-4.33464e-07,-2.02747e-08,0.693279,0.000694397,-4.94288e-07,1.46279e-08,0.693973,0.000693452,-4.50405e-07,2.13678e-08,0.694666,0.000692616,-3.86301e-07,-4.04945e-08,0.695358,0.000691721,-5.07785e-07,2.14009e-08,0.696049,0.00069077,-4.43582e-07,1.44955e-08,0.69674,0.000689926,-4.00096e-07,-1.97783e-08,0.697429,0.000689067,-4.5943e-07,5.01296e-09,0.698118,0.000688163,-4.44392e-07,-2.73521e-10,0.698805,0.000687273,-4.45212e-07,-3.91893e-09,0.699492,0.000686371,-4.56969e-07,1.59493e-08,0.700178,0.000685505,-4.09121e-07,-2.73351e-10,0.700863,0.000684686,-4.09941e-07,-1.4856e-08,0.701548,0.000683822,-4.54509e-07,9.25979e-11,0.702231,0.000682913,-4.54231e-07,1.44855e-08,0.702913,0.000682048,-4.10775e-07,1.56992e-09,0.703595,0.000681231,-4.06065e-07,-2.07652e-08,0.704276,0.000680357,-4.68361e-07,2.18864e-08,0.704956,0.000679486,-4.02701e-07,-7.17595e-09,0.705635,0.000678659,-4.24229e-07,6.81748e-09,0.706313,0.000677831,-4.03777e-07,-2.0094e-08,0.70699,0.000676963,-4.64059e-07,1.39538e-08,0.707667,0.000676077,-4.22197e-07,2.38835e-08,0.708343,0.000675304,-3.50547e-07,-4.98831e-08,0.709018,0.000674453,-5.00196e-07,5.64395e-08,0.709692,0.000673622,-3.30878e-07,-5.66657e-08,0.710365,0.00067279,-5.00875e-07,5.1014e-08,0.711037,0.000671942,-3.47833e-07,-2.81809e-08,0.711709,0.000671161,-4.32376e-07,2.10513e-09,0.712379,0.000670303,-4.2606e-07,1.97604e-08,0.713049,0.00066951,-3.66779e-07,-2.15422e-08,0.713718,0.000668712,-4.31406e-07,6.8038e-09,0.714387,0.000667869,-4.10994e-07,-5.67295e-09,0.715054,0.00066703,-4.28013e-07,1.5888e-08,0.715721,0.000666222,-3.80349e-07,1.72576e-09,0.716387,0.000665467,-3.75172e-07,-2.27911e-08,0.717052,0.000664648,-4.43545e-07,2.9834e-08,0.717716,0.00066385,-3.54043e-07,-3.69401e-08,0.718379,0.000663031,-4.64864e-07,5.83219e-08,0.719042,0.000662277,-2.89898e-07,-7.71382e-08,0.719704,0.000661465,-5.21313e-07,7.14171e-08,0.720365,0.000660637,-3.07061e-07,-2.97161e-08,0.721025,0.000659934,-3.96209e-07,-1.21575e-08,0.721685,0.000659105,-4.32682e-07,1.87412e-08,0.722343,0.000658296,-3.76458e-07,-3.2029e-09,0.723001,0.000657533,-3.86067e-07,-5.9296e-09,0.723659,0.000656743,-4.03856e-07,2.69213e-08,0.724315,0.000656016,-3.23092e-07,-4.21511e-08,0.724971,0.000655244,-4.49545e-07,2.24737e-08,0.725625,0.000654412,-3.82124e-07,1.18611e-08,0.726279,0.000653683,-3.46541e-07,-1.03132e-08,0.726933,0.000652959,-3.7748e-07,-3.02128e-08,0.727585,0.000652114,-4.68119e-07,7.15597e-08,0.728237,0.000651392,-2.5344e-07,-7.72119e-08,0.728888,0.000650654,-4.85075e-07,5.8474e-08,0.729538,0.000649859,-3.09654e-07,-3.74746e-08,0.730188,0.000649127,-4.22077e-07,3.18197e-08,0.730837,0.000648379,-3.26618e-07,-3.01997e-08,0.731485,0.000647635,-4.17217e-07,2.93747e-08,0.732132,0.000646888,-3.29093e-07,-2.76943e-08,0.732778,0.000646147,-4.12176e-07,2.17979e-08,0.733424,0.000645388,-3.46783e-07,1.07292e-10,0.734069,0.000644695,-3.46461e-07,-2.22271e-08,0.734713,0.000643935,-4.13142e-07,2.91963e-08,0.735357,0.000643197,-3.25553e-07,-3.49536e-08,0.736,0.000642441,-4.30414e-07,5.10133e-08,0.736642,0.000641733,-2.77374e-07,-4.98904e-08,0.737283,0.000641028,-4.27045e-07,2.93392e-08,0.737924,0.000640262,-3.39028e-07,-7.86156e-09,0.738564,0.000639561,-3.62612e-07,2.10703e-09,0.739203,0.000638842,-3.56291e-07,-5.6653e-10,0.739842,0.000638128,-3.57991e-07,1.59086e-10,0.740479,0.000637412,-3.57513e-07,-6.98321e-11,0.741116,0.000636697,-3.57723e-07,1.20214e-10,0.741753,0.000635982,-3.57362e-07,-4.10987e-10,0.742388,0.000635266,-3.58595e-07,1.5237e-09,0.743023,0.000634553,-3.54024e-07,-5.68376e-09,0.743657,0.000633828,-3.71075e-07,2.12113e-08,0.744291,0.00063315,-3.07441e-07,-1.95569e-08,0.744924,0.000632476,-3.66112e-07,-2.58816e-09,0.745556,0.000631736,-3.73877e-07,2.99096e-08,0.746187,0.000631078,-2.84148e-07,-5.74454e-08,0.746818,0.000630337,-4.56484e-07,8.06629e-08,0.747448,0.000629666,-2.14496e-07,-8.63922e-08,0.748077,0.000628978,-4.73672e-07,8.60918e-08,0.748706,0.000628289,-2.15397e-07,-7.91613e-08,0.749334,0.000627621,-4.5288e-07,5.17393e-08,0.749961,0.00062687,-2.97663e-07,-8.58662e-09,0.750588,0.000626249,-3.23422e-07,-1.73928e-08,0.751214,0.00062555,-3.75601e-07,1.85532e-08,0.751839,0.000624855,-3.19941e-07,2.78479e-09,0.752463,0.000624223,-3.11587e-07,-2.96923e-08,0.753087,0.000623511,-4.00664e-07,5.63799e-08,0.75371,0.000622879,-2.31524e-07,-7.66179e-08,0.754333,0.000622186,-4.61378e-07,7.12778e-08,0.754955,0.000621477,-2.47545e-07,-2.96794e-08,0.755576,0.000620893,-3.36583e-07,-1.21648e-08,0.756196,0.000620183,-3.73077e-07,1.87339e-08,0.756816,0.000619493,-3.16875e-07,-3.16622e-09,0.757435,0.00061885,-3.26374e-07,-6.0691e-09,0.758054,0.000618179,-3.44581e-07,2.74426e-08,0.758672,0.000617572,-2.62254e-07,-4.40968e-08,0.759289,0.000616915,-3.94544e-07,2.97352e-08,0.759906,0.000616215,-3.05338e-07,-1.52393e-08,0.760522,0.000615559,-3.51056e-07,3.12221e-08,0.761137,0.000614951,-2.5739e-07,-5.00443e-08,0.761751,0.000614286,-4.07523e-07,4.9746e-08,0.762365,0.00061362,-2.58285e-07,-2.97303e-08,0.762979,0.000613014,-3.47476e-07,9.57079e-09,0.763591,0.000612348,-3.18764e-07,-8.55287e-09,0.764203,0.000611685,-3.44422e-07,2.46407e-08,0.764815,0.00061107,-2.705e-07,-3.04053e-08,0.765426,0.000610437,-3.61716e-07,3.73759e-08,0.766036,0.000609826,-2.49589e-07,-5.94935e-08,0.766645,0.000609149,-4.28069e-07,8.13889e-08,0.767254,0.000608537,-1.83902e-07,-8.72483e-08,0.767862,0.000607907,-4.45647e-07,8.87901e-08,0.76847,0.000607282,-1.79277e-07,-8.90983e-08,0.769077,0.000606656,-4.46572e-07,8.87892e-08,0.769683,0.000606029,-1.80204e-07,-8.72446e-08,0.770289,0.000605407,-4.41938e-07,8.13752e-08,0.770894,0.000604768,-1.97812e-07,-5.94423e-08,0.771498,0.000604194,-3.76139e-07,3.71848e-08,0.772102,0.000603553,-2.64585e-07,-2.96922e-08,0.772705,0.000602935,-3.53661e-07,2.19793e-08,0.773308,0.000602293,-2.87723e-07,1.37955e-09,0.77391,0.000601722,-2.83585e-07,-2.74976e-08,0.774512,0.000601072,-3.66077e-07,4.9006e-08,0.775112,0.000600487,-2.19059e-07,-4.93171e-08,0.775712,0.000599901,-3.67011e-07,2.90531e-08,0.776312,0.000599254,-2.79851e-07,-7.29081e-09,0.776911,0.000598673,-3.01724e-07,1.10077e-10,0.777509,0.00059807,-3.01393e-07,6.85053e-09,0.778107,0.000597487,-2.80842e-07,-2.75123e-08,0.778704,0.000596843,-3.63379e-07,4.35939e-08,0.779301,0.000596247,-2.32597e-07,-2.7654e-08,0.779897,0.000595699,-3.15559e-07,7.41741e-09,0.780492,0.00059509,-2.93307e-07,-2.01562e-09,0.781087,0.000594497,-2.99354e-07,6.45059e-10,0.781681,0.000593901,-2.97418e-07,-5.64635e-10,0.782275,0.000593304,-2.99112e-07,1.61347e-09,0.782868,0.000592711,-2.94272e-07,-5.88926e-09,0.78346,0.000592105,-3.1194e-07,2.19436e-08,0.784052,0.000591546,-2.46109e-07,-2.22805e-08,0.784643,0.000590987,-3.1295e-07,7.57368e-09,0.785234,0.000590384,-2.90229e-07,-8.01428e-09,0.785824,0.00058978,-3.14272e-07,2.44834e-08,0.786414,0.000589225,-2.40822e-07,-3.03148e-08,0.787003,0.000588652,-3.31766e-07,3.7171e-08,0.787591,0.0005881,-2.20253e-07,-5.87646e-08,0.788179,0.000587483,-3.96547e-07,7.86782e-08,0.788766,0.000586926,-1.60512e-07,-7.71342e-08,0.789353,0.000586374,-3.91915e-07,5.10444e-08,0.789939,0.000585743,-2.38782e-07,-7.83422e-09,0.790524,0.000585242,-2.62284e-07,-1.97076e-08,0.791109,0.000584658,-3.21407e-07,2.70598e-08,0.791693,0.000584097,-2.40228e-07,-2.89269e-08,0.792277,0.000583529,-3.27008e-07,2.90431e-08,0.792861,0.000582963,-2.39879e-07,-2.76409e-08,0.793443,0.0005824,-3.22802e-07,2.1916e-08,0.794025,0.00058182,-2.57054e-07,-4.18368e-10,0.794607,0.000581305,-2.58309e-07,-2.02425e-08,0.795188,0.000580727,-3.19036e-07,2.17838e-08,0.795768,0.000580155,-2.53685e-07,-7.28814e-09,0.796348,0.000579625,-2.75549e-07,7.36871e-09,0.796928,0.000579096,-2.53443e-07,-2.21867e-08,0.797506,0.000578523,-3.20003e-07,2.17736e-08,0.798085,0.000577948,-2.54683e-07,-5.30296e-09,0.798662,0.000577423,-2.70592e-07,-5.61698e-10,0.799239,0.00057688,-2.72277e-07,7.54977e-09,0.799816,0.000576358,-2.49627e-07,-2.96374e-08,0.800392,0.00057577,-3.38539e-07,5.1395e-08,0.800968,0.000575247,-1.84354e-07,-5.67335e-08,0.801543,0.000574708,-3.54555e-07,5.63297e-08,0.802117,0.000574168,-1.85566e-07,-4.93759e-08,0.802691,0.000573649,-3.33693e-07,2.19646e-08,0.803264,0.000573047,-2.678e-07,2.1122e-08,0.803837,0.000572575,-2.04433e-07,-4.68482e-08,0.804409,0.000572026,-3.44978e-07,4.70613e-08,0.804981,0.000571477,-2.03794e-07,-2.21877e-08,0.805552,0.000571003,-2.70357e-07,-1.79153e-08,0.806123,0.000570408,-3.24103e-07,3.42443e-08,0.806693,0.000569863,-2.2137e-07,1.47556e-10,0.807263,0.000569421,-2.20928e-07,-3.48345e-08,0.807832,0.000568874,-3.25431e-07,1.99812e-08,0.808401,0.000568283,-2.65487e-07,1.45143e-08,0.808969,0.000567796,-2.21945e-07,-1.84338e-08,0.809536,0.000567297,-2.77246e-07,-3.83608e-10,0.810103,0.000566741,-2.78397e-07,1.99683e-08,0.81067,0.000566244,-2.18492e-07,-1.98848e-08,0.811236,0.000565747,-2.78146e-07,-3.38976e-11,0.811801,0.000565191,-2.78248e-07,2.00204e-08,0.812366,0.000564695,-2.18187e-07,-2.04429e-08,0.812931,0.000564197,-2.79516e-07,2.1467e-09,0.813495,0.000563644,-2.73076e-07,1.18561e-08,0.814058,0.000563134,-2.37507e-07,1.00334e-08,0.814621,0.000562689,-2.07407e-07,-5.19898e-08,0.815183,0.000562118,-3.63376e-07,7.87163e-08,0.815745,0.000561627,-1.27227e-07,-8.40616e-08,0.816306,0.000561121,-3.79412e-07,7.87163e-08,0.816867,0.000560598,-1.43263e-07,-5.19898e-08,0.817428,0.000560156,-2.99233e-07,1.00335e-08,0.817988,0.000559587,-2.69132e-07,1.18559e-08,0.818547,0.000559085,-2.33564e-07,2.14764e-09,0.819106,0.000558624,-2.27122e-07,-2.04464e-08,0.819664,0.000558108,-2.88461e-07,2.00334e-08,0.820222,0.000557591,-2.28361e-07,-8.24277e-11,0.820779,0.000557135,-2.28608e-07,-1.97037e-08,0.821336,0.000556618,-2.87719e-07,1.92925e-08,0.821893,0.000556101,-2.29841e-07,2.13831e-09,0.822448,0.000555647,-2.23427e-07,-2.78458e-08,0.823004,0.000555117,-3.06964e-07,4.96402e-08,0.823559,0.000554652,-1.58043e-07,-5.15058e-08,0.824113,0.000554181,-3.12561e-07,3.71737e-08,0.824667,0.000553668,-2.0104e-07,-3.75844e-08,0.82522,0.000553153,-3.13793e-07,5.35592e-08,0.825773,0.000552686,-1.53115e-07,-5.74431e-08,0.826326,0.000552207,-3.25444e-07,5.7004e-08,0.826878,0.000551728,-1.54433e-07,-5.13635e-08,0.827429,0.000551265,-3.08523e-07,2.92406e-08,0.82798,0.000550735,-2.20801e-07,-5.99424e-09,0.828531,0.000550276,-2.38784e-07,-5.26363e-09,0.829081,0.000549782,-2.54575e-07,2.70488e-08,0.82963,0.000549354,-1.73429e-07,-4.33268e-08,0.83018,0.000548878,-3.03409e-07,2.7049e-08,0.830728,0.000548352,-2.22262e-07,-5.26461e-09,0.831276,0.000547892,-2.38056e-07,-5.99057e-09,0.831824,0.000547397,-2.56027e-07,2.92269e-08,0.832371,0.000546973,-1.68347e-07,-5.13125e-08,0.832918,0.000546482,-3.22284e-07,5.68139e-08,0.833464,0.000546008,-1.51843e-07,-5.67336e-08,0.83401,0.000545534,-3.22043e-07,5.09113e-08,0.834555,0.000545043,-1.6931e-07,-2.77022e-08,0.8351,0.000544621,-2.52416e-07,2.92924e-10,0.835644,0.000544117,-2.51537e-07,2.65305e-08,0.836188,0.000543694,-1.71946e-07,-4.68105e-08,0.836732,0.00054321,-3.12377e-07,4.15021e-08,0.837275,0.000542709,-1.87871e-07,1.13355e-11,0.837817,0.000542334,-1.87837e-07,-4.15474e-08,0.838359,0.000541833,-3.12479e-07,4.69691e-08,0.838901,0.000541349,-1.71572e-07,-2.71196e-08,0.839442,0.000540925,-2.52931e-07,1.90462e-09,0.839983,0.000540425,-2.47217e-07,1.95011e-08,0.840523,0.000539989,-1.88713e-07,-2.03045e-08,0.841063,0.00053955,-2.49627e-07,2.11216e-09,0.841602,0.000539057,-2.4329e-07,1.18558e-08,0.842141,0.000538606,-2.07723e-07,1.00691e-08,0.842679,0.000538221,-1.77516e-07,-5.21324e-08,0.843217,0.00053771,-3.33913e-07,7.92513e-08,0.843755,0.00053728,-9.6159e-08,-8.60587e-08,0.844292,0.000536829,-3.54335e-07,8.61696e-08,0.844828,0.000536379,-9.58263e-08,-7.98057e-08,0.845364,0.000535948,-3.35243e-07,5.42394e-08,0.8459,0.00053544,-1.72525e-07,-1.79426e-08,0.846435,0.000535041,-2.26353e-07,1.75308e-08,0.84697,0.000534641,-1.73761e-07,-5.21806e-08,0.847505,0.000534137,-3.30302e-07,7.19824e-08,0.848038,0.000533692,-1.14355e-07,-5.69349e-08,0.848572,0.000533293,-2.8516e-07,3.65479e-08,0.849105,0.000532832,-1.75516e-07,-2.96519e-08,0.849638,0.000532392,-2.64472e-07,2.2455e-08,0.85017,0.000531931,-1.97107e-07,-5.63451e-10,0.850702,0.000531535,-1.98797e-07,-2.02011e-08,0.851233,0.000531077,-2.59401e-07,2.17634e-08,0.851764,0.000530623,-1.94111e-07,-7.24794e-09,0.852294,0.000530213,-2.15854e-07,7.22832e-09,0.852824,0.000529803,-1.94169e-07,-2.16653e-08,0.853354,0.00052935,-2.59165e-07,1.98283e-08,0.853883,0.000528891,-1.9968e-07,1.95678e-09,0.854412,0.000528497,-1.9381e-07,-2.76554e-08,0.85494,0.000528027,-2.76776e-07,4.90603e-08,0.855468,0.00052762,-1.29596e-07,-4.93764e-08,0.855995,0.000527213,-2.77725e-07,2.92361e-08,0.856522,0.000526745,-1.90016e-07,-7.96341e-09,0.857049,0.000526341,-2.13907e-07,2.61752e-09,0.857575,0.000525922,-2.06054e-07,-2.50665e-09,0.8581,0.000525502,-2.13574e-07,7.40906e-09,0.858626,0.000525097,-1.91347e-07,-2.71296e-08,0.859151,0.000524633,-2.72736e-07,4.15048e-08,0.859675,0.000524212,-1.48221e-07,-1.96802e-08,0.860199,0.000523856,-2.07262e-07,-2.23886e-08,0.860723,0.000523375,-2.74428e-07,4.96299e-08,0.861246,0.000522975,-1.25538e-07,-5.69216e-08,0.861769,0.000522553,-2.96303e-07,5.88473e-08,0.862291,0.000522137,-1.19761e-07,-5.92584e-08,0.862813,0.00052172,-2.97536e-07,5.8977e-08,0.863334,0.000521301,-1.20605e-07,-5.74403e-08,0.863855,0.000520888,-2.92926e-07,5.15751e-08,0.864376,0.000520457,-1.38201e-07,-2.96506e-08,0.864896,0.000520091,-2.27153e-07,7.42277e-09,0.865416,0.000519659,-2.04885e-07,-4.05057e-11,0.865936,0.00051925,-2.05006e-07,-7.26074e-09,0.866455,0.000518818,-2.26788e-07,2.90835e-08,0.866973,0.000518451,-1.39538e-07,-4.94686e-08,0.867492,0.000518024,-2.87944e-07,4.95814e-08,0.868009,0.000517597,-1.39199e-07,-2.96479e-08,0.868527,0.000517229,-2.28143e-07,9.40539e-09,0.869044,0.000516801,-1.99927e-07,-7.9737e-09,0.86956,0.000516378,-2.23848e-07,2.24894e-08,0.870077,0.000515997,-1.5638e-07,-2.23793e-08,0.870592,0.000515617,-2.23517e-07,7.42302e-09,0.871108,0.000515193,-2.01248e-07,-7.31283e-09,0.871623,0.000514768,-2.23187e-07,2.18283e-08,0.872137,0.000514387,-1.57702e-07,-2.03959e-08,0.872652,0.000514011,-2.1889e-07,1.50711e-10,0.873165,0.000513573,-2.18437e-07,1.97931e-08,0.873679,0.000513196,-1.59058e-07,-1.97183e-08,0.874192,0.000512819,-2.18213e-07,-5.24324e-10,0.874704,0.000512381,-2.19786e-07,2.18156e-08,0.875217,0.000512007,-1.54339e-07,-2.71336e-08,0.875728,0.000511616,-2.3574e-07,2.71141e-08,0.87624,0.000511226,-1.54398e-07,-2.17182e-08,0.876751,0.000510852,-2.19552e-07,1.54131e-10,0.877262,0.000510414,-2.1909e-07,2.11017e-08,0.877772,0.000510039,-1.55785e-07,-2.49562e-08,0.878282,0.000509652,-2.30654e-07,1.91183e-08,0.878791,0.000509248,-1.73299e-07,8.08751e-09,0.8793,0.000508926,-1.49036e-07,-5.14684e-08,0.879809,0.000508474,-3.03441e-07,7.85766e-08,0.880317,0.000508103,-6.77112e-08,-8.40242e-08,0.880825,0.000507715,-3.19784e-07,7.87063e-08,0.881333,0.000507312,-8.36649e-08,-5.19871e-08,0.88184,0.000506988,-2.39626e-07,1.00327e-08,0.882346,0.000506539,-2.09528e-07,1.18562e-08,0.882853,0.000506156,-1.73959e-07,2.14703e-09,0.883359,0.000505814,-1.67518e-07,-2.04444e-08,0.883864,0.000505418,-2.28851e-07,2.00258e-08,0.88437,0.00050502,-1.68774e-07,-5.42855e-11,0.884874,0.000504682,-1.68937e-07,-1.98087e-08,0.885379,0.000504285,-2.28363e-07,1.96842e-08,0.885883,0.000503887,-1.6931e-07,6.76342e-10,0.886387,0.000503551,-1.67281e-07,-2.23896e-08,0.88689,0.000503149,-2.3445e-07,2.92774e-08,0.887393,0.000502768,-1.46618e-07,-3.51152e-08,0.887896,0.00050237,-2.51963e-07,5.15787e-08,0.888398,0.00050202,-9.72271e-08,-5.19903e-08,0.8889,0.00050167,-2.53198e-07,3.71732e-08,0.889401,0.000501275,-1.41678e-07,-3.70978e-08,0.889902,0.00050088,-2.52972e-07,5.16132e-08,0.890403,0.000500529,-9.81321e-08,-5.01459e-08,0.890903,0.000500183,-2.4857e-07,2.9761e-08,0.891403,0.000499775,-1.59287e-07,-9.29351e-09,0.891903,0.000499428,-1.87167e-07,7.41301e-09,0.892402,0.000499076,-1.64928e-07,-2.03585e-08,0.892901,0.000498685,-2.26004e-07,1.44165e-08,0.893399,0.000498276,-1.82754e-07,2.22974e-08,0.893898,0.000497978,-1.15862e-07,-4.40013e-08,0.894395,0.000497614,-2.47866e-07,3.44985e-08,0.894893,0.000497222,-1.44371e-07,-3.43882e-08,0.89539,0.00049683,-2.47535e-07,4.34497e-08,0.895886,0.000496465,-1.17186e-07,-2.02012e-08,0.896383,0.00049617,-1.7779e-07,-2.22497e-08,0.896879,0.000495748,-2.44539e-07,4.95952e-08,0.897374,0.000495408,-9.57532e-08,-5.69217e-08,0.89787,0.000495045,-2.66518e-07,5.88823e-08,0.898364,0.000494689,-8.98713e-08,-5.93983e-08,0.898859,0.000494331,-2.68066e-07,5.95017e-08,0.899353,0.000493973,-8.95613e-08,-5.9399e-08,0.899847,0.000493616,-2.67758e-07,5.8885e-08,0.90034,0.000493257,-9.11033e-08,-5.69317e-08,0.900833,0.000492904,-2.61898e-07,4.96326e-08,0.901326,0.000492529,-1.13001e-07,-2.23893e-08,0.901819,0.000492236,-1.80169e-07,-1.968e-08,0.902311,0.000491817,-2.39209e-07,4.15047e-08,0.902802,0.000491463,-1.14694e-07,-2.71296e-08,0.903293,0.000491152,-1.96083e-07,7.409e-09,0.903784,0.000490782,-1.73856e-07,-2.50645e-09,0.904275,0.000490427,-1.81376e-07,2.61679e-09,0.904765,0.000490072,-1.73525e-07,-7.96072e-09,0.905255,0.000489701,-1.97407e-07,2.92261e-08,0.905745,0.000489394,-1.09729e-07,-4.93389e-08,0.906234,0.000489027,-2.57746e-07,4.89204e-08,0.906723,0.000488658,-1.10985e-07,-2.71333e-08,0.907211,0.000488354,-1.92385e-07,8.30861e-12,0.907699,0.00048797,-1.9236e-07,2.71001e-08,0.908187,0.000487666,-1.1106e-07,-4.88041e-08,0.908675,0.000487298,-2.57472e-07,4.89069e-08,0.909162,0.000486929,-1.10751e-07,-2.76143e-08,0.909649,0.000486625,-1.93594e-07,1.9457e-09,0.910135,0.000486244,-1.87757e-07,1.98315e-08,0.910621,0.000485928,-1.28262e-07,-2.16671e-08,0.911107,0.000485606,-1.93264e-07,7.23216e-09,0.911592,0.000485241,-1.71567e-07,-7.26152e-09,0.912077,0.000484877,-1.93352e-07,2.18139e-08,0.912562,0.000484555,-1.2791e-07,-2.03895e-08,0.913047,0.000484238,-1.89078e-07,1.39494e-10,0.913531,0.000483861,-1.8866e-07,1.98315e-08,0.914014,0.000483543,-1.29165e-07,-1.98609e-08,0.914498,0.000483225,-1.88748e-07,7.39912e-12,0.914981,0.000482847,-1.88726e-07,1.98313e-08,0.915463,0.000482529,-1.29232e-07,-1.9728e-08,0.915946,0.000482212,-1.88416e-07,-5.24035e-10,0.916428,0.000481833,-1.89988e-07,2.18241e-08,0.916909,0.000481519,-1.24516e-07,-2.71679e-08,0.917391,0.000481188,-2.06019e-07,2.72427e-08,0.917872,0.000480858,-1.24291e-07,-2.21985e-08,0.918353,0.000480543,-1.90886e-07,1.94644e-09,0.918833,0.000480167,-1.85047e-07,1.44127e-08,0.919313,0.00047984,-1.41809e-07,7.39438e-12,0.919793,0.000479556,-1.41787e-07,-1.44423e-08,0.920272,0.000479229,-1.85114e-07,-1.84291e-09,0.920751,0.000478854,-1.90642e-07,2.18139e-08,0.92123,0.000478538,-1.25201e-07,-2.58081e-08,0.921708,0.00047821,-2.02625e-07,2.18139e-08,0.922186,0.00047787,-1.37183e-07,-1.84291e-09,0.922664,0.00047759,-1.42712e-07,-1.44423e-08,0.923141,0.000477262,-1.86039e-07,7.34701e-12,0.923618,0.00047689,-1.86017e-07,1.44129e-08,0.924095,0.000476561,-1.42778e-07,1.94572e-09,0.924572,0.000476281,-1.36941e-07,-2.21958e-08,0.925048,0.000475941,-2.03528e-07,2.72327e-08,0.925523,0.000475615,-1.2183e-07,-2.71304e-08,0.925999,0.00047529,-2.03221e-07,2.16843e-08,0.926474,0.000474949,-1.38168e-07,-2.16005e-12,0.926949,0.000474672,-1.38175e-07,-2.16756e-08,0.927423,0.000474331,-2.03202e-07,2.71001e-08,0.927897,0.000474006,-1.21902e-07,-2.71201e-08,0.928371,0.000473681,-2.03262e-07,2.17757e-08,0.928845,0.00047334,-1.37935e-07,-3.78028e-10,0.929318,0.000473063,-1.39069e-07,-2.02636e-08,0.929791,0.000472724,-1.9986e-07,2.18276e-08,0.930263,0.000472389,-1.34377e-07,-7.44231e-09,0.930736,0.000472098,-1.56704e-07,7.94165e-09,0.931208,0.000471809,-1.32879e-07,-2.43243e-08,0.931679,0.00047147,-2.05851e-07,2.97508e-08,0.932151,0.000471148,-1.16599e-07,-3.50742e-08,0.932622,0.000470809,-2.21822e-07,5.09414e-08,0.933092,0.000470518,-6.89976e-08,-4.94821e-08,0.933563,0.000470232,-2.17444e-07,2.77775e-08,0.934033,0.00046988,-1.34111e-07,-2.02351e-09,0.934502,0.000469606,-1.40182e-07,-1.96835e-08,0.934972,0.000469267,-1.99232e-07,2.11529e-08,0.935441,0.000468932,-1.35774e-07,-5.32332e-09,0.93591,0.000468644,-1.51743e-07,1.40413e-10,0.936378,0.000468341,-1.51322e-07,4.76166e-09,0.936846,0.000468053,-1.37037e-07,-1.9187e-08,0.937314,0.000467721,-1.94598e-07,1.23819e-08,0.937782,0.000467369,-1.57453e-07,2.92642e-08,0.938249,0.000467142,-6.96601e-08,-6.98342e-08,0.938716,0.000466793,-2.79163e-07,7.12586e-08,0.939183,0.000466449,-6.53869e-08,-3.63863e-08,0.939649,0.000466209,-1.74546e-07,1.46818e-08,0.940115,0.000465904,-1.305e-07,-2.2341e-08,0.940581,0.000465576,-1.97523e-07,1.50774e-08,0.941046,0.000465226,-1.52291e-07,2.16359e-08,0.941511,0.000464986,-8.73832e-08,-4.20162e-08,0.941976,0.000464685,-2.13432e-07,2.72198e-08,0.942441,0.00046434,-1.31773e-07,-7.2581e-09,0.942905,0.000464055,-1.53547e-07,1.81263e-09,0.943369,0.000463753,-1.48109e-07,7.58386e-12,0.943832,0.000463457,-1.48086e-07,-1.84298e-09,0.944296,0.000463155,-1.53615e-07,7.36433e-09,0.944759,0.00046287,-1.31522e-07,-2.76143e-08,0.945221,0.000462524,-2.14365e-07,4.34883e-08,0.945684,0.000462226,-8.39003e-08,-2.71297e-08,0.946146,0.000461977,-1.65289e-07,5.42595e-09,0.946608,0.000461662,-1.49012e-07,5.42593e-09,0.947069,0.000461381,-1.32734e-07,-2.71297e-08,0.94753,0.000461034,-2.14123e-07,4.34881e-08,0.947991,0.000460736,-8.36585e-08,-2.76134e-08,0.948452,0.000460486,-1.66499e-07,7.36083e-09,0.948912,0.000460175,-1.44416e-07,-1.82993e-09,0.949372,0.000459881,-1.49906e-07,-4.11073e-11,0.949832,0.000459581,-1.50029e-07,1.99434e-09,0.950291,0.000459287,-1.44046e-07,-7.93627e-09,0.950751,0.000458975,-1.67855e-07,2.97507e-08,0.951209,0.000458728,-7.86029e-08,-5.1462e-08,0.951668,0.000458417,-2.32989e-07,5.6888e-08,0.952126,0.000458121,-6.2325e-08,-5.68806e-08,0.952584,0.000457826,-2.32967e-07,5.14251e-08,0.953042,0.000457514,-7.86914e-08,-2.96107e-08,0.953499,0.000457268,-1.67523e-07,7.41296e-09,0.953956,0.000456955,-1.45285e-07,-4.11262e-11,0.954413,0.000456665,-1.45408e-07,-7.24847e-09,0.95487,0.000456352,-1.67153e-07,2.9035e-08,0.955326,0.000456105,-8.00484e-08,-4.92869e-08,0.955782,0.000455797,-2.27909e-07,4.89032e-08,0.956238,0.000455488,-8.11994e-08,-2.71166e-08,0.956693,0.000455244,-1.62549e-07,-4.13678e-11,0.957148,0.000454919,-1.62673e-07,2.72821e-08,0.957603,0.000454675,-8.0827e-08,-4.94824e-08,0.958057,0.000454365,-2.29274e-07,5.14382e-08,0.958512,0.000454061,-7.49597e-08,-3.7061e-08,0.958965,0.0004538,-1.86143e-07,3.72013e-08,0.959419,0.000453539,-7.45389e-08,-5.21396e-08,0.959873,0.000453234,-2.30958e-07,5.21476e-08,0.960326,0.000452928,-7.45146e-08,-3.72416e-08,0.960778,0.000452667,-1.8624e-07,3.72143e-08,0.961231,0.000452407,-7.45967e-08,-5.20109e-08,0.961683,0.000452101,-2.30629e-07,5.16199e-08,0.962135,0.000451795,-7.57696e-08,-3.52595e-08,0.962587,0.000451538,-1.81548e-07,2.98133e-08,0.963038,0.000451264,-9.2108e-08,-2.43892e-08,0.963489,0.000451007,-1.65276e-07,8.13892e-09,0.96394,0.000450701,-1.40859e-07,-8.16647e-09,0.964391,0.000450394,-1.65358e-07,2.45269e-08,0.964841,0.000450137,-9.17775e-08,-3.03367e-08,0.965291,0.000449863,-1.82787e-07,3.7215e-08,0.965741,0.000449609,-7.11424e-08,-5.89188e-08,0.96619,0.00044929,-2.47899e-07,7.92509e-08,0.966639,0.000449032,-1.01462e-08,-7.92707e-08,0.967088,0.000448773,-2.47958e-07,5.90181e-08,0.967537,0.000448455,-7.0904e-08,-3.75925e-08,0.967985,0.0004482,-1.83681e-07,3.17471e-08,0.968433,0.000447928,-8.84401e-08,-2.97913e-08,0.968881,0.000447662,-1.77814e-07,2.78133e-08,0.969329,0.000447389,-9.4374e-08,-2.18572e-08,0.969776,0.000447135,-1.59946e-07,1.10134e-11,0.970223,0.000446815,-1.59913e-07,2.18132e-08,0.97067,0.000446561,-9.44732e-08,-2.76591e-08,0.971116,0.000446289,-1.7745e-07,2.92185e-08,0.971562,0.000446022,-8.97948e-08,-2.96104e-08,0.972008,0.000445753,-1.78626e-07,2.96185e-08,0.972454,0.000445485,-8.97706e-08,-2.92588e-08,0.972899,0.000445218,-1.77547e-07,2.78123e-08,0.973344,0.000444946,-9.41103e-08,-2.23856e-08,0.973789,0.000444691,-1.61267e-07,2.12559e-09,0.974233,0.000444374,-1.5489e-07,1.38833e-08,0.974678,0.000444106,-1.13241e-07,1.94591e-09,0.975122,0.000443886,-1.07403e-07,-2.16669e-08,0.975565,0.000443606,-1.72404e-07,2.5117e-08,0.976009,0.000443336,-9.70526e-08,-1.91963e-08,0.976452,0.000443085,-1.54642e-07,-7.93627e-09,0.976895,0.000442752,-1.7845e-07,5.09414e-08,0.977338,0.000442548,-2.56262e-08,-7.66201e-08,0.97778,0.000442266,-2.55486e-07,7.67249e-08,0.978222,0.000441986,-2.53118e-08,-5.14655e-08,0.978664,0.000441781,-1.79708e-07,9.92773e-09,0.979106,0.000441451,-1.49925e-07,1.17546e-08,0.979547,0.000441186,-1.14661e-07,2.65868e-09,0.979988,0.000440965,-1.06685e-07,-2.23893e-08,0.980429,0.000440684,-1.73853e-07,2.72939e-08,0.980869,0.000440419,-9.19716e-08,-2.71816e-08,0.98131,0.000440153,-1.73516e-07,2.18278e-08,0.98175,0.000439872,-1.08033e-07,-5.24833e-10,0.982189,0.000439654,-1.09607e-07,-1.97284e-08,0.982629,0.000439376,-1.68793e-07,1.98339e-08,0.983068,0.000439097,-1.09291e-07,-2.62901e-12,0.983507,0.000438879,-1.09299e-07,-1.98234e-08,0.983946,0.000438601,-1.68769e-07,1.96916e-08,0.984384,0.000438322,-1.09694e-07,6.6157e-10,0.984823,0.000438105,-1.0771e-07,-2.23379e-08,0.985261,0.000437823,-1.74723e-07,2.90855e-08,0.985698,0.00043756,-8.74669e-08,-3.43992e-08,0.986136,0.000437282,-1.90665e-07,4.89068e-08,0.986573,0.000437048,-4.39442e-08,-4.20188e-08,0.98701,0.000436834,-1.7e-07,-4.11073e-11,0.987446,0.000436494,-1.70124e-07,4.21832e-08,0.987883,0.00043628,-4.35742e-08,-4.94824e-08,0.988319,0.000436044,-1.92021e-07,3.6537e-08,0.988755,0.00043577,-8.24102e-08,-3.70611e-08,0.989191,0.000435494,-1.93593e-07,5.21026e-08,0.989626,0.000435263,-3.72855e-08,-5.21402e-08,0.990061,0.000435032,-1.93706e-07,3.7249e-08,0.990496,0.000434756,-8.19592e-08,-3.72512e-08,0.990931,0.000434481,-1.93713e-07,5.21511e-08,0.991365,0.00043425,-3.72595e-08,-5.21439e-08,0.991799,0.000434019,-1.93691e-07,3.72152e-08,0.992233,0.000433743,-8.20456e-08,-3.71123e-08,0.992667,0.000433468,-1.93382e-07,5.16292e-08,0.9931,0.000433236,-3.84947e-08,-5.01953e-08,0.993533,0.000433008,-1.89081e-07,2.99427e-08,0.993966,0.00043272,-9.92525e-08,-9.9708e-09,0.994399,0.000432491,-1.29165e-07,9.94051e-09,0.994831,0.000432263,-9.93434e-08,-2.97912e-08,0.995263,0.000431975,-1.88717e-07,4.96198e-08,0.995695,0.000431746,-3.98578e-08,-4.94785e-08,0.996127,0.000431518,-1.88293e-07,2.9085e-08,0.996558,0.000431229,-1.01038e-07,-7.25675e-09,0.996989,0.000431005,-1.22809e-07,-5.79945e-11,0.99742,0.000430759,-1.22983e-07,7.48873e-09,0.997851,0.000430536,-1.00516e-07,-2.98969e-08,0.998281,0.000430245,-1.90207e-07,5.24942e-08,0.998711,0.000430022,-3.27246e-08,-6.08706e-08,0.999141,0.000429774,-2.15336e-07,7.17788e-08,0.999571,0.000429392,0.,0.};

        template <bool srgb, int blueIdx, typename T, typename D>
        __device__ __forceinline__ void Lab2RGBConvert_f(const T& src, D& dst)
        {
            const float lThresh = 0.008856f * 903.3f;
            const float fThresh = 7.787f * 0.008856f + 16.0f / 116.0f;

            float Y, fy;

            if (src.x <= lThresh)
            {
                Y = src.x / 903.3f;
                fy = 7.787f * Y + 16.0f / 116.0f;
            }
            else
            {
                fy = (src.x + 16.0f) / 116.0f;
                Y = fy * fy * fy;
            }

            float X = src.y / 500.0f + fy;
            float Z = fy - src.z / 200.0f;

            if (X <= fThresh)
                X = (X - 16.0f / 116.0f) / 7.787f;
            else
                X = X * X * X;

            if (Z <= fThresh)
                Z = (Z - 16.0f / 116.0f) / 7.787f;
            else
                Z = Z * Z * Z;

            float B = 0.052891f * X - 0.204043f * Y + 1.151152f * Z;
            float G = -0.921235f * X + 1.875991f * Y + 0.045244f * Z;
            float R = 3.079933f * X - 1.537150f * Y - 0.542782f * Z;

            if (srgb)
            {
                B = splineInterpolate(B * GAMMA_TAB_SIZE, c_sRGBInvGammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G * GAMMA_TAB_SIZE, c_sRGBInvGammaTab, GAMMA_TAB_SIZE);
                R = splineInterpolate(R * GAMMA_TAB_SIZE, c_sRGBInvGammaTab, GAMMA_TAB_SIZE);
            }

            dst.x = blueIdx == 0 ? B : R;
            dst.y = G;
            dst.z = blueIdx == 0 ? R : B;
            setAlpha(dst, ColorChannel<float>::max());
        }

        template <bool srgb, int blueIdx, typename T, typename D>
        __device__ __forceinline__ void Lab2RGBConvert_b(const T& src, D& dst)
        {
            float3 srcf, dstf;

            srcf.x = src.x * (100.f / 255.f);
            srcf.y = src.y - 128;
            srcf.z = src.z - 128;

            Lab2RGBConvert_f<srgb, blueIdx>(srcf, dstf);

            dst.x = saturate_cast<uchar>(dstf.x * 255.f);
            dst.y = saturate_cast<uchar>(dstf.y * 255.f);
            dst.z = saturate_cast<uchar>(dstf.z * 255.f);
            setAlpha(dst, ColorChannel<uchar>::max());
        }

        template <typename T, int scn, int dcn, bool srgb, int blueIdx> struct Lab2RGB;
        template <int scn, int dcn, bool srgb, int blueIdx>
        struct Lab2RGB<uchar, scn, dcn, srgb, blueIdx>
            : unary_function<typename TypeVec<uchar, scn>::vec_type, typename TypeVec<uchar, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<uchar, dcn>::vec_type operator ()(const typename TypeVec<uchar, scn>::vec_type& src) const
            {
                typename TypeVec<uchar, dcn>::vec_type dst;

                Lab2RGBConvert_b<srgb, blueIdx>(src, dst);

                return dst;
            }
            __host__ __device__ __forceinline__ Lab2RGB() {}
            __host__ __device__ __forceinline__ Lab2RGB(const Lab2RGB&) {}
        };
        template <int scn, int dcn, bool srgb, int blueIdx>
        struct Lab2RGB<float, scn, dcn, srgb, blueIdx>
            : unary_function<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<float, dcn>::vec_type operator ()(const typename TypeVec<float, scn>::vec_type& src) const
            {
                typename TypeVec<float, dcn>::vec_type dst;

                Lab2RGBConvert_f<srgb, blueIdx>(src, dst);

                return dst;
            }
            __host__ __device__ __forceinline__ Lab2RGB() {}
            __host__ __device__ __forceinline__ Lab2RGB(const Lab2RGB&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_Lab2RGB_TRAITS(name, scn, dcn, srgb, blueIdx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::Lab2RGB<T, scn, dcn, srgb, blueIdx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

///////////////////////////////////// RGB <-> Luv /////////////////////////////////////

    namespace color_detail
    {
        __constant__ float c_LabCbrtTab[] = {0.137931,0.0114066,0.,1.18859e-07,0.149338,0.011407,3.56578e-07,-5.79396e-07,0.160745,0.0114059,-1.38161e-06,2.16892e-06,0.172151,0.0114097,5.12516e-06,-8.0814e-06,0.183558,0.0113957,-1.9119e-05,3.01567e-05,0.194965,0.0114479,7.13509e-05,-0.000112545,0.206371,0.011253,-0.000266285,-0.000106493,0.217252,0.0104009,-0.000585765,7.32149e-05,0.22714,0.00944906,-0.00036612,1.21917e-05,0.236235,0.0087534,-0.000329545,2.01753e-05,0.244679,0.00815483,-0.000269019,1.24435e-05,0.252577,0.00765412,-0.000231689,1.05618e-05,0.26001,0.00722243,-0.000200003,8.26662e-06,0.267041,0.00684723,-0.000175203,6.76746e-06,0.27372,0.00651712,-0.000154901,5.61192e-06,0.280088,0.00622416,-0.000138065,4.67009e-06,0.286179,0.00596204,-0.000124055,3.99012e-06,0.292021,0.0057259,-0.000112085,3.36032e-06,0.297638,0.00551181,-0.000102004,2.95338e-06,0.30305,0.00531666,-9.31435e-05,2.52875e-06,0.308277,0.00513796,-8.55572e-05,2.22022e-06,0.313331,0.00497351,-7.88966e-05,1.97163e-06,0.318228,0.00482163,-7.29817e-05,1.7248e-06,0.322978,0.00468084,-6.78073e-05,1.55998e-06,0.327593,0.0045499,-6.31274e-05,1.36343e-06,0.332081,0.00442774,-5.90371e-05,1.27136e-06,0.336451,0.00431348,-5.5223e-05,1.09111e-06,0.34071,0.00420631,-5.19496e-05,1.0399e-06,0.344866,0.00410553,-4.88299e-05,9.18347e-07,0.348923,0.00401062,-4.60749e-05,8.29942e-07,0.352889,0.00392096,-4.35851e-05,7.98478e-07,0.356767,0.00383619,-4.11896e-05,6.84917e-07,0.360562,0.00375586,-3.91349e-05,6.63976e-07,0.36428,0.00367959,-3.7143e-05,5.93086e-07,0.367923,0.00360708,-3.53637e-05,5.6976e-07,0.371495,0.00353806,-3.36544e-05,4.95533e-07,0.375,0.00347224,-3.21678e-05,4.87951e-07,0.378441,0.00340937,-3.0704e-05,4.4349e-07,0.38182,0.00334929,-2.93735e-05,4.20297e-07,0.38514,0.0032918,-2.81126e-05,3.7872e-07,0.388404,0.00323671,-2.69764e-05,3.596e-07,0.391614,0.00318384,-2.58976e-05,3.5845e-07,0.394772,0.00313312,-2.48223e-05,2.92765e-07,0.397881,0.00308435,-2.3944e-05,3.18232e-07,0.400942,0.00303742,-2.29893e-05,2.82046e-07,0.403957,0.00299229,-2.21432e-05,2.52315e-07,0.406927,0.00294876,-2.13862e-05,2.58416e-07,0.409855,0.00290676,-2.0611e-05,2.33939e-07,0.412741,0.00286624,-1.99092e-05,2.36342e-07,0.415587,0.00282713,-1.92001e-05,1.916e-07,0.418396,0.00278931,-1.86253e-05,2.1915e-07,0.421167,0.00275271,-1.79679e-05,1.83498e-07,0.423901,0.00271733,-1.74174e-05,1.79343e-07,0.426602,0.00268303,-1.68794e-05,1.72013e-07,0.429268,0.00264979,-1.63633e-05,1.75686e-07,0.431901,0.00261759,-1.58363e-05,1.3852e-07,0.434503,0.00258633,-1.54207e-05,1.64304e-07,0.437074,0.00255598,-1.49278e-05,1.28136e-07,0.439616,0.00252651,-1.45434e-05,1.57618e-07,0.442128,0.0024979,-1.40705e-05,1.0566e-07,0.444612,0.00247007,-1.37535e-05,1.34998e-07,0.447068,0.00244297,-1.33485e-05,1.29207e-07,0.449498,0.00241666,-1.29609e-05,9.32347e-08,0.451902,0.00239102,-1.26812e-05,1.23703e-07,0.45428,0.00236603,-1.23101e-05,9.74072e-08,0.456634,0.0023417,-1.20179e-05,1.12518e-07,0.458964,0.002318,-1.16803e-05,7.83681e-08,0.46127,0.00229488,-1.14452e-05,1.10452e-07,0.463554,0.00227232,-1.11139e-05,7.58719e-08,0.465815,0.00225032,-1.08863e-05,9.2699e-08,0.468055,0.00222882,-1.06082e-05,8.97738e-08,0.470273,0.00220788,-1.03388e-05,5.4845e-08,0.47247,0.00218736,-1.01743e-05,1.0808e-07,0.474648,0.00216734,-9.85007e-06,4.9277e-08,0.476805,0.00214779,-9.70224e-06,8.22408e-08,0.478943,0.00212863,-9.45551e-06,6.87942e-08,0.481063,0.00210993,-9.24913e-06,5.98144e-08,0.483163,0.00209161,-9.06969e-06,7.93789e-08,0.485246,0.00207371,-8.83155e-06,3.99032e-08,0.487311,0.00205616,-8.71184e-06,8.88325e-08,0.489358,0.002039,-8.44534e-06,2.20004e-08,0.491389,0.00202218,-8.37934e-06,9.13872e-08,0.493403,0.0020057,-8.10518e-06,2.96829e-08,0.495401,0.00198957,-8.01613e-06,5.81028e-08,0.497382,0.00197372,-7.84183e-06,6.5731e-08,0.499348,0.00195823,-7.64463e-06,3.66019e-08,0.501299,0.00194305,-7.53483e-06,2.62811e-08,0.503234,0.00192806,-7.45598e-06,9.66907e-08,0.505155,0.00191344,-7.16591e-06,4.18928e-09,0.507061,0.00189912,-7.15334e-06,6.53665e-08,0.508953,0.00188501,-6.95724e-06,3.23686e-08,0.510831,0.00187119,-6.86014e-06,4.35774e-08,0.512696,0.0018576,-6.72941e-06,3.17406e-08,0.514547,0.00184424,-6.63418e-06,6.78785e-08,0.516384,0.00183117,-6.43055e-06,-5.23126e-09,0.518209,0.0018183,-6.44624e-06,7.22562e-08,0.520021,0.00180562,-6.22947e-06,1.42292e-08,0.52182,0.0017932,-6.18679e-06,4.9641e-08,0.523607,0.00178098,-6.03786e-06,2.56259e-08,0.525382,0.00176898,-5.96099e-06,2.66696e-08,0.527145,0.00175714,-5.88098e-06,4.65094e-08,0.528897,0.00174552,-5.74145e-06,2.57114e-08,0.530637,0.00173411,-5.66431e-06,2.94588e-08,0.532365,0.00172287,-5.57594e-06,3.52667e-08,0.534082,0.00171182,-5.47014e-06,8.28868e-09,0.535789,0.00170091,-5.44527e-06,5.07871e-08,0.537484,0.00169017,-5.29291e-06,2.69817e-08,0.539169,0.00167967,-5.21197e-06,2.01009e-08,0.540844,0.0016693,-5.15166e-06,1.18237e-08,0.542508,0.00165903,-5.11619e-06,5.18135e-08,0.544162,0.00164896,-4.96075e-06,1.9341e-08,0.545806,0.00163909,-4.90273e-06,-9.96867e-09,0.54744,0.00162926,-4.93263e-06,8.01382e-08,0.549064,0.00161963,-4.69222e-06,-1.25601e-08,0.550679,0.00161021,-4.7299e-06,2.97067e-08,0.552285,0.00160084,-4.64078e-06,1.29426e-08,0.553881,0.0015916,-4.60195e-06,3.77327e-08,0.555468,0.00158251,-4.48875e-06,1.49412e-08,0.557046,0.00157357,-4.44393e-06,2.17118e-08,0.558615,0.00156475,-4.3788e-06,1.74206e-08,0.560176,0.00155605,-4.32653e-06,2.78152e-08,0.561727,0.00154748,-4.24309e-06,-9.47239e-09,0.563271,0.00153896,-4.27151e-06,6.9679e-08,0.564805,0.00153063,-4.06247e-06,-3.08246e-08,0.566332,0.00152241,-4.15494e-06,5.36188e-08,0.56785,0.00151426,-3.99409e-06,-4.83594e-09,0.56936,0.00150626,-4.00859e-06,2.53293e-08,0.570863,0.00149832,-3.93261e-06,2.27286e-08,0.572357,0.00149052,-3.86442e-06,2.96541e-09,0.573844,0.0014828,-3.85552e-06,2.50147e-08,0.575323,0.00147516,-3.78048e-06,1.61842e-08,0.576794,0.00146765,-3.73193e-06,2.94582e-08,0.578258,0.00146028,-3.64355e-06,-1.48076e-08,0.579715,0.00145295,-3.68798e-06,2.97724e-08,0.581164,0.00144566,-3.59866e-06,1.49272e-08,0.582606,0.00143851,-3.55388e-06,2.97285e-08,0.584041,0.00143149,-3.46469e-06,-1.46323e-08,0.585469,0.00142451,-3.50859e-06,2.88004e-08,0.58689,0.00141758,-3.42219e-06,1.864e-08,0.588304,0.00141079,-3.36627e-06,1.58482e-08,0.589712,0.00140411,-3.31872e-06,-2.24279e-08,0.591112,0.00139741,-3.38601e-06,7.38639e-08,0.592507,0.00139085,-3.16441e-06,-3.46088e-08,0.593894,0.00138442,-3.26824e-06,4.96675e-09,0.595275,0.0013779,-3.25334e-06,7.4346e-08,0.59665,0.00137162,-3.0303e-06,-6.39319e-08,0.598019,0.00136536,-3.2221e-06,6.21725e-08,0.599381,0.00135911,-3.03558e-06,-5.94423e-09,0.600737,0.00135302,-3.05341e-06,2.12091e-08,0.602087,0.00134697,-2.98979e-06,-1.92876e-08,0.603431,0.00134094,-3.04765e-06,5.5941e-08,0.604769,0.00133501,-2.87983e-06,-2.56622e-08,0.606101,0.00132917,-2.95681e-06,4.67078e-08,0.607427,0.0013234,-2.81669e-06,-4.19592e-08,0.608748,0.00131764,-2.94257e-06,6.15243e-08,0.610062,0.00131194,-2.75799e-06,-2.53244e-08,0.611372,0.00130635,-2.83397e-06,3.97739e-08,0.612675,0.0013008,-2.71465e-06,-1.45618e-08,0.613973,0.00129533,-2.75833e-06,1.84733e-08,0.615266,0.00128986,-2.70291e-06,2.73606e-10,0.616553,0.00128446,-2.70209e-06,4.00367e-08,0.617835,0.00127918,-2.58198e-06,-4.12113e-08,0.619111,0.00127389,-2.70561e-06,6.52039e-08,0.620383,0.00126867,-2.51e-06,-4.07901e-08,0.621649,0.00126353,-2.63237e-06,3.83516e-08,0.62291,0.00125838,-2.51732e-06,6.59315e-09,0.624166,0.00125337,-2.49754e-06,-5.11939e-09,0.625416,0.00124836,-2.5129e-06,1.38846e-08,0.626662,0.00124337,-2.47124e-06,9.18514e-09,0.627903,0.00123846,-2.44369e-06,8.97952e-09,0.629139,0.0012336,-2.41675e-06,1.45012e-08,0.63037,0.00122881,-2.37325e-06,-7.37949e-09,0.631597,0.00122404,-2.39538e-06,1.50169e-08,0.632818,0.00121929,-2.35033e-06,6.91648e-09,0.634035,0.00121461,-2.32958e-06,1.69219e-08,0.635248,0.00121,-2.27882e-06,-1.49997e-08,0.636455,0.0012054,-2.32382e-06,4.30769e-08,0.637659,0.00120088,-2.19459e-06,-3.80986e-08,0.638857,0.00119638,-2.30888e-06,4.97134e-08,0.640051,0.00119191,-2.15974e-06,-4.15463e-08,0.641241,0.00118747,-2.28438e-06,5.68667e-08,0.642426,0.00118307,-2.11378e-06,-7.10641e-09,0.643607,0.00117882,-2.1351e-06,-2.8441e-08,0.644784,0.00117446,-2.22042e-06,6.12658e-08,0.645956,0.00117021,-2.03663e-06,-3.78083e-08,0.647124,0.00116602,-2.15005e-06,3.03627e-08,0.648288,0.00116181,-2.05896e-06,-2.40379e-08,0.649448,0.00115762,-2.13108e-06,6.57887e-08,0.650603,0.00115356,-1.93371e-06,-6.03028e-08,0.651755,0.00114951,-2.11462e-06,5.62134e-08,0.652902,0.00114545,-1.94598e-06,-4.53417e-08,0.654046,0.00114142,-2.082e-06,6.55489e-08,0.655185,0.00113745,-1.88536e-06,-3.80396e-08,0.656321,0.00113357,-1.99948e-06,2.70049e-08,0.657452,0.00112965,-1.91846e-06,-1.03755e-08,0.65858,0.00112578,-1.94959e-06,1.44973e-08,0.659704,0.00112192,-1.9061e-06,1.1991e-08,0.660824,0.00111815,-1.87012e-06,-2.85634e-09,0.66194,0.0011144,-1.87869e-06,-5.65782e-10,0.663053,0.00111064,-1.88039e-06,5.11947e-09,0.664162,0.0011069,-1.86503e-06,3.96924e-08,0.665267,0.00110328,-1.74595e-06,-4.46795e-08,0.666368,0.00109966,-1.87999e-06,1.98161e-08,0.667466,0.00109596,-1.82054e-06,2.502e-08,0.66856,0.00109239,-1.74548e-06,-6.86593e-10,0.669651,0.0010889,-1.74754e-06,-2.22739e-08,0.670738,0.00108534,-1.81437e-06,3.01776e-08,0.671821,0.0010818,-1.72383e-06,2.07732e-08,0.672902,0.00107841,-1.66151e-06,-5.36658e-08,0.673978,0.00107493,-1.82251e-06,7.46802e-08,0.675051,0.00107151,-1.59847e-06,-6.62411e-08,0.676121,0.00106811,-1.79719e-06,7.10748e-08,0.677188,0.00106473,-1.58397e-06,-3.92441e-08,0.678251,0.00106145,-1.7017e-06,2.62973e-08,0.679311,0.00105812,-1.62281e-06,-6.34035e-09,0.680367,0.00105486,-1.64183e-06,-9.36249e-10,0.68142,0.00105157,-1.64464e-06,1.00854e-08,0.68247,0.00104831,-1.61438e-06,2.01995e-08,0.683517,0.00104514,-1.55378e-06,-3.1279e-08,0.68456,0.00104194,-1.64762e-06,4.53114e-08,0.685601,0.00103878,-1.51169e-06,-3.07573e-08,0.686638,0.00103567,-1.60396e-06,1.81133e-08,0.687672,0.00103251,-1.54962e-06,1.79085e-08,0.688703,0.00102947,-1.49589e-06,-3.01428e-08,0.689731,0.00102639,-1.58632e-06,4.30583e-08,0.690756,0.00102334,-1.45715e-06,-2.28814e-08,0.691778,0.00102036,-1.52579e-06,-1.11373e-08,0.692797,0.00101727,-1.5592e-06,6.74305e-08,0.693812,0.00101436,-1.35691e-06,-7.97709e-08,0.694825,0.0010114,-1.59622e-06,7.28391e-08,0.695835,0.00100843,-1.37771e-06,-3.27715e-08,0.696842,0.00100558,-1.47602e-06,-1.35807e-09,0.697846,0.00100262,-1.48009e-06,3.82037e-08,0.698847,0.000999775,-1.36548e-06,-3.22474e-08,0.699846,0.000996948,-1.46223e-06,3.11809e-08,0.700841,0.000994117,-1.36868e-06,-3.28714e-08,0.701834,0.000991281,-1.4673e-06,4.07001e-08,0.702824,0.000988468,-1.3452e-06,-1.07197e-08,0.703811,0.000985746,-1.37736e-06,2.17866e-09,0.704795,0.000982998,-1.37082e-06,2.00521e-09,0.705777,0.000980262,-1.3648e-06,-1.01996e-08,0.706756,0.000977502,-1.3954e-06,3.87931e-08,0.707732,0.000974827,-1.27902e-06,-2.57632e-08,0.708706,0.000972192,-1.35631e-06,4.65513e-09,0.709676,0.000969493,-1.34235e-06,7.14257e-09,0.710645,0.00096683,-1.32092e-06,2.63791e-08,0.71161,0.000964267,-1.24178e-06,-5.30543e-08,0.712573,0.000961625,-1.40095e-06,6.66289e-08,0.713533,0.000959023,-1.20106e-06,-3.46474e-08,0.714491,0.000956517,-1.305e-06,1.23559e-08,0.715446,0.000953944,-1.26793e-06,-1.47763e-08,0.716399,0.000951364,-1.31226e-06,4.67494e-08,0.717349,0.000948879,-1.17201e-06,-5.3012e-08,0.718297,0.000946376,-1.33105e-06,4.60894e-08,0.719242,0.000943852,-1.19278e-06,-1.21366e-08,0.720185,0.00094143,-1.22919e-06,2.45673e-09,0.721125,0.000938979,-1.22182e-06,2.30966e-09,0.722063,0.000936543,-1.21489e-06,-1.16954e-08,0.722998,0.000934078,-1.24998e-06,4.44718e-08,0.723931,0.000931711,-1.11656e-06,-4.69823e-08,0.724861,0.000929337,-1.25751e-06,2.4248e-08,0.725789,0.000926895,-1.18477e-06,9.5949e-09,0.726715,0.000924554,-1.15598e-06,-3.02286e-09,0.727638,0.000922233,-1.16505e-06,2.49649e-09,0.72856,0.00091991,-1.15756e-06,-6.96321e-09,0.729478,0.000917575,-1.17845e-06,2.53564e-08,0.730395,0.000915294,-1.10238e-06,-3.48578e-08,0.731309,0.000912984,-1.20695e-06,5.44704e-08,0.732221,0.000910734,-1.04354e-06,-6.38144e-08,0.73313,0.000908455,-1.23499e-06,8.15781e-08,0.734038,0.00090623,-9.90253e-07,-8.3684e-08,0.734943,0.000903999,-1.2413e-06,7.43441e-08,0.735846,0.000901739,-1.01827e-06,-3.48787e-08,0.736746,0.000899598,-1.12291e-06,5.56596e-09,0.737645,0.000897369,-1.10621e-06,1.26148e-08,0.738541,0.000895194,-1.06837e-06,3.57935e-09,0.739435,0.000893068,-1.05763e-06,-2.69322e-08,0.740327,0.000890872,-1.13842e-06,4.45448e-08,0.741217,0.000888729,-1.00479e-06,-3.20376e-08,0.742105,0.000886623,-1.1009e-06,2.40011e-08,0.74299,0.000884493,-1.0289e-06,-4.36209e-09,0.743874,0.000882422,-1.04199e-06,-6.55268e-09,0.744755,0.000880319,-1.06164e-06,3.05728e-08,0.745634,0.000878287,-9.69926e-07,-5.61338e-08,0.746512,0.000876179,-1.13833e-06,7.4753e-08,0.747387,0.000874127,-9.14068e-07,-6.40644e-08,0.74826,0.000872106,-1.10626e-06,6.22955e-08,0.749131,0.000870081,-9.19375e-07,-6.59083e-08,0.75,0.000868044,-1.1171e-06,8.21284e-08,0.750867,0.000866056,-8.70714e-07,-8.37915e-08,0.751732,0.000864064,-1.12209e-06,7.42237e-08,0.752595,0.000862042,-8.99418e-07,-3.42894e-08,0.753456,0.00086014,-1.00229e-06,3.32955e-09,0.754315,0.000858146,-9.92297e-07,2.09712e-08,0.755173,0.000856224,-9.29384e-07,-2.76096e-08,0.756028,0.000854282,-1.01221e-06,2.98627e-08,0.756881,0.000852348,-9.22625e-07,-3.22365e-08,0.757733,0.000850406,-1.01933e-06,3.94786e-08,0.758582,0.000848485,-9.00898e-07,-6.46833e-09,0.75943,0.000846664,-9.20303e-07,-1.36052e-08,0.760275,0.000844783,-9.61119e-07,1.28447e-09,0.761119,0.000842864,-9.57266e-07,8.4674e-09,0.761961,0.000840975,-9.31864e-07,2.44506e-08,0.762801,0.000839185,-8.58512e-07,-4.6665e-08,0.763639,0.000837328,-9.98507e-07,4.30001e-08,0.764476,0.00083546,-8.69507e-07,-6.12609e-09,0.76531,0.000833703,-8.87885e-07,-1.84959e-08,0.766143,0.000831871,-9.43372e-07,2.05052e-08,0.766974,0.000830046,-8.81857e-07,-3.92026e-09,0.767803,0.000828271,-8.93618e-07,-4.82426e-09,0.768631,0.000826469,-9.0809e-07,2.32172e-08,0.769456,0.000824722,-8.38439e-07,-2.84401e-08,0.77028,0.00082296,-9.23759e-07,3.09386e-08,0.771102,0.000821205,-8.30943e-07,-3.57099e-08,0.771922,0.000819436,-9.38073e-07,5.22963e-08,0.772741,0.000817717,-7.81184e-07,-5.42658e-08,0.773558,0.000815992,-9.43981e-07,4.55579e-08,0.774373,0.000814241,-8.07308e-07,-8.75656e-09,0.775186,0.0008126,-8.33578e-07,-1.05315e-08,0.775998,0.000810901,-8.65172e-07,-8.72188e-09,0.776808,0.000809145,-8.91338e-07,4.54191e-08,0.777616,0.000807498,-7.5508e-07,-5.37454e-08,0.778423,0.000805827,-9.16317e-07,5.03532e-08,0.779228,0.000804145,-7.65257e-07,-2.84584e-08,0.780031,0.000802529,-8.50632e-07,3.87579e-09,0.780833,0.00080084,-8.39005e-07,1.29552e-08,0.781633,0.0007992,-8.00139e-07,3.90804e-09,0.782432,0.000797612,-7.88415e-07,-2.85874e-08,0.783228,0.000795949,-8.74177e-07,5.0837e-08,0.784023,0.000794353,-7.21666e-07,-5.55513e-08,0.784817,0.000792743,-8.8832e-07,5.21587e-08,0.785609,0.000791123,-7.31844e-07,-3.38744e-08,0.786399,0.000789558,-8.33467e-07,2.37342e-08,0.787188,0.000787962,-7.62264e-07,-1.45775e-09,0.787975,0.000786433,-7.66638e-07,-1.79034e-08,0.788761,0.000784846,-8.20348e-07,1.34665e-08,0.789545,0.000783246,-7.79948e-07,2.3642e-08,0.790327,0.000781757,-7.09022e-07,-4.84297e-08,0.791108,0.000780194,-8.54311e-07,5.08674e-08,0.791888,0.000778638,-7.01709e-07,-3.58303e-08,0.792666,0.000777127,-8.092e-07,3.28493e-08,0.793442,0.000775607,-7.10652e-07,-3.59624e-08,0.794217,0.000774078,-8.1854e-07,5.13959e-08,0.79499,0.000772595,-6.64352e-07,-5.04121e-08,0.795762,0.000771115,-8.15588e-07,3.10431e-08,0.796532,0.000769577,-7.22459e-07,-1.41557e-08,0.797301,0.00076809,-7.64926e-07,2.55795e-08,0.798069,0.000766636,-6.88187e-07,-2.85578e-08,0.798835,0.000765174,-7.73861e-07,2.90472e-08,0.799599,0.000763714,-6.86719e-07,-2.80262e-08,0.800362,0.000762256,-7.70798e-07,2.34531e-08,0.801123,0.000760785,-7.00438e-07,-6.18144e-09,0.801884,0.000759366,-7.18983e-07,1.27263e-09,0.802642,0.000757931,-7.15165e-07,1.09101e-09,0.803399,0.000756504,-7.11892e-07,-5.63675e-09,0.804155,0.000755064,-7.28802e-07,2.14559e-08,0.80491,0.00075367,-6.64434e-07,-2.05821e-08,0.805663,0.00075228,-7.26181e-07,1.26812e-09,0.806414,0.000750831,-7.22377e-07,1.55097e-08,0.807164,0.000749433,-6.75848e-07,-3.70216e-09,0.807913,0.00074807,-6.86954e-07,-7.0105e-10,0.80866,0.000746694,-6.89057e-07,6.5063e-09,0.809406,0.000745336,-6.69538e-07,-2.53242e-08,0.810151,0.000743921,-7.45511e-07,3.51858e-08,0.810894,0.000742535,-6.39953e-07,3.79034e-09,0.811636,0.000741267,-6.28582e-07,-5.03471e-08,0.812377,0.000739858,-7.79624e-07,7.83886e-08,0.813116,0.000738534,-5.44458e-07,-8.43935e-08,0.813854,0.000737192,-7.97638e-07,8.03714e-08,0.81459,0.000735838,-5.56524e-07,-5.82784e-08,0.815325,0.00073455,-7.31359e-07,3.35329e-08,0.816059,0.000733188,-6.3076e-07,-1.62486e-08,0.816792,0.000731878,-6.79506e-07,3.14614e-08,0.817523,0.000730613,-5.85122e-07,-4.99925e-08,0.818253,0.000729293,-7.35099e-07,4.92994e-08,0.818982,0.000727971,-5.87201e-07,-2.79959e-08,0.819709,0.000726712,-6.71189e-07,3.07959e-09,0.820435,0.000725379,-6.6195e-07,1.56777e-08,0.82116,0.000724102,-6.14917e-07,-6.18564e-09,0.821883,0.000722854,-6.33474e-07,9.06488e-09,0.822606,0.000721614,-6.06279e-07,-3.00739e-08,0.823327,0.000720311,-6.96501e-07,5.16262e-08,0.824046,0.000719073,-5.41623e-07,-5.72214e-08,0.824765,0.000717818,-7.13287e-07,5.80503e-08,0.825482,0.000716566,-5.39136e-07,-5.57703e-08,0.826198,0.00071532,-7.06447e-07,4.58215e-08,0.826912,0.000714045,-5.68983e-07,-8.30636e-09,0.827626,0.000712882,-5.93902e-07,-1.25961e-08,0.828338,0.000711656,-6.3169e-07,-9.13985e-10,0.829049,0.00071039,-6.34432e-07,1.62519e-08,0.829759,0.00070917,-5.85676e-07,-4.48904e-09,0.830468,0.000707985,-5.99143e-07,1.70418e-09,0.831175,0.000706792,-5.9403e-07,-2.32768e-09,0.831881,0.000705597,-6.01014e-07,7.60648e-09,0.832586,0.000704418,-5.78194e-07,-2.80982e-08,0.83329,0.000703177,-6.62489e-07,4.51817e-08,0.833993,0.000701988,-5.26944e-07,-3.34192e-08,0.834694,0.000700834,-6.27201e-07,2.88904e-08,0.835394,0.000699666,-5.4053e-07,-2.25378e-08,0.836093,0.000698517,-6.08143e-07,1.65589e-09,0.836791,0.000697306,-6.03176e-07,1.59142e-08,0.837488,0.000696147,-5.55433e-07,-5.70801e-09,0.838184,0.000695019,-5.72557e-07,6.91792e-09,0.838878,0.000693895,-5.51803e-07,-2.19637e-08,0.839571,0.000692725,-6.17694e-07,2.13321e-08,0.840263,0.000691554,-5.53698e-07,-3.75996e-09,0.840954,0.000690435,-5.64978e-07,-6.29219e-09,0.841644,0.000689287,-5.83855e-07,2.89287e-08,0.842333,0.000688206,-4.97068e-07,-4.98181e-08,0.843021,0.000687062,-6.46523e-07,5.11344e-08,0.843707,0.000685922,-4.9312e-07,-3.55102e-08,0.844393,0.00068483,-5.9965e-07,3.13019e-08,0.845077,0.000683724,-5.05745e-07,-3.00925e-08,0.84576,0.000682622,-5.96022e-07,2.94636e-08,0.846442,0.000681519,-5.07631e-07,-2.81572e-08,0.847123,0.000680419,-5.92103e-07,2.35606e-08,0.847803,0.000679306,-5.21421e-07,-6.48045e-09,0.848482,0.000678243,-5.40863e-07,2.36124e-09,0.849159,0.000677169,-5.33779e-07,-2.96461e-09,0.849836,0.000676092,-5.42673e-07,9.49728e-09,0.850512,0.000675035,-5.14181e-07,-3.50245e-08,0.851186,0.000673902,-6.19254e-07,7.09959e-08,0.851859,0.000672876,-4.06267e-07,-7.01453e-08,0.852532,0.000671853,-6.16703e-07,3.07714e-08,0.853203,0.000670712,-5.24388e-07,6.66423e-09,0.853873,0.000669684,-5.04396e-07,2.17629e-09,0.854542,0.000668681,-4.97867e-07,-1.53693e-08,0.855211,0.000667639,-5.43975e-07,-3.03752e-10,0.855878,0.000666551,-5.44886e-07,1.65844e-08,0.856544,0.000665511,-4.95133e-07,-6.42907e-09,0.857209,0.000664501,-5.1442e-07,9.13195e-09,0.857873,0.0006635,-4.87024e-07,-3.00987e-08,0.858536,0.000662435,-5.7732e-07,5.16584e-08,0.859198,0.000661436,-4.22345e-07,-5.73255e-08,0.859859,0.000660419,-5.94322e-07,5.84343e-08,0.860518,0.000659406,-4.19019e-07,-5.72022e-08,0.861177,0.000658396,-5.90626e-07,5.11653e-08,0.861835,0.000657368,-4.3713e-07,-2.82495e-08,0.862492,0.000656409,-5.21878e-07,2.22788e-09,0.863148,0.000655372,-5.15195e-07,1.9338e-08,0.863803,0.0006544,-4.5718e-07,-1.99754e-08,0.864457,0.000653425,-5.17107e-07,9.59024e-10,0.86511,0.000652394,-5.1423e-07,1.61393e-08,0.865762,0.000651414,-4.65812e-07,-5.91149e-09,0.866413,0.000650465,-4.83546e-07,7.50665e-09,0.867063,0.00064952,-4.61026e-07,-2.4115e-08,0.867712,0.000648526,-5.33371e-07,2.93486e-08,0.86836,0.000647547,-4.45325e-07,-3.36748e-08,0.869007,0.000646555,-5.4635e-07,4.57461e-08,0.869653,0.0006456,-4.09112e-07,-3.01002e-08,0.870298,0.000644691,-4.99412e-07,1.50501e-08,0.870942,0.000643738,-4.54262e-07,-3.01002e-08,0.871585,0.000642739,-5.44563e-07,4.57461e-08,0.872228,0.000641787,-4.07324e-07,-3.36748e-08,0.872869,0.000640871,-5.08349e-07,2.93486e-08,0.873509,0.000639943,-4.20303e-07,-2.4115e-08,0.874149,0.00063903,-4.92648e-07,7.50655e-09,0.874787,0.000638067,-4.70128e-07,-5.91126e-09,0.875425,0.000637109,-4.87862e-07,1.61385e-08,0.876062,0.000636182,-4.39447e-07,9.61961e-10,0.876697,0.000635306,-4.36561e-07,-1.99863e-08,0.877332,0.000634373,-4.9652e-07,1.93785e-08,0.877966,0.000633438,-4.38384e-07,2.07697e-09,0.878599,0.000632567,-4.32153e-07,-2.76864e-08,0.879231,0.00063162,-5.15212e-07,4.90641e-08,0.879862,0.000630737,-3.6802e-07,-4.93606e-08,0.880493,0.000629852,-5.16102e-07,2.9169e-08,0.881122,0.000628908,-4.28595e-07,-7.71083e-09,0.881751,0.000628027,-4.51727e-07,1.6744e-09,0.882378,0.000627129,-4.46704e-07,1.01317e-09,0.883005,0.000626239,-4.43665e-07,-5.72703e-09,0.883631,0.000625334,-4.60846e-07,2.1895e-08,0.884255,0.000624478,-3.95161e-07,-2.22481e-08,0.88488,0.000623621,-4.61905e-07,7.4928e-09,0.885503,0.00062272,-4.39427e-07,-7.72306e-09,0.886125,0.000621818,-4.62596e-07,2.33995e-08,0.886746,0.000620963,-3.92398e-07,-2.62704e-08,0.887367,0.000620099,-4.71209e-07,2.20775e-08,0.887987,0.000619223,-4.04976e-07,-2.43496e-09,0.888605,0.000618406,-4.12281e-07,-1.23377e-08,0.889223,0.000617544,-4.49294e-07,-7.81876e-09,0.88984,0.000616622,-4.72751e-07,4.36128e-08,0.890457,0.000615807,-3.41912e-07,-4.7423e-08,0.891072,0.000614981,-4.84181e-07,2.68698e-08,0.891687,0.000614093,-4.03572e-07,-4.51384e-10,0.8923,0.000613285,-4.04926e-07,-2.50643e-08,0.892913,0.0006124,-4.80119e-07,4.11038e-08,0.893525,0.000611563,-3.56808e-07,-2.01414e-08,0.894136,0.000610789,-4.17232e-07,-2.01426e-08,0.894747,0.000609894,-4.7766e-07,4.11073e-08,0.895356,0.000609062,-3.54338e-07,-2.50773e-08,0.895965,0.000608278,-4.2957e-07,-4.02954e-10,0.896573,0.000607418,-4.30779e-07,2.66891e-08,0.89718,0.000606636,-3.50711e-07,-4.67489e-08,0.897786,0.000605795,-4.90958e-07,4.10972e-08,0.898391,0.000604936,-3.67666e-07,1.56948e-09,0.898996,0.000604205,-3.62958e-07,-4.73751e-08,0.8996,0.000603337,-5.05083e-07,6.87214e-08,0.900202,0.000602533,-2.98919e-07,-4.86966e-08,0.900805,0.000601789,-4.45009e-07,6.85589e-09,0.901406,0.00060092,-4.24441e-07,2.1273e-08,0.902007,0.000600135,-3.60622e-07,-3.23434e-08,0.902606,0.000599317,-4.57652e-07,4.84959e-08,0.903205,0.000598547,-3.12164e-07,-4.24309e-08,0.903803,0.000597795,-4.39457e-07,2.01844e-09,0.904401,0.000596922,-4.33402e-07,3.43571e-08,0.904997,0.000596159,-3.30331e-07,-2.02374e-08,0.905593,0.000595437,-3.91043e-07,-1.30123e-08,0.906188,0.000594616,-4.3008e-07,1.26819e-08,0.906782,0.000593794,-3.92034e-07,2.18894e-08,0.907376,0.000593076,-3.26366e-07,-4.06349e-08,0.907968,0.000592301,-4.4827e-07,2.1441e-08,0.90856,0.000591469,-3.83947e-07,1.44754e-08,0.909151,0.000590744,-3.40521e-07,-1.97379e-08,0.909742,0.000590004,-3.99735e-07,4.87161e-09,0.910331,0.000589219,-3.8512e-07,2.51532e-10,0.91092,0.00058845,-3.84366e-07,-5.87776e-09,0.911508,0.000587663,-4.01999e-07,2.32595e-08,0.912096,0.000586929,-3.3222e-07,-2.75554e-08,0.912682,0.000586182,-4.14887e-07,2.73573e-08,0.913268,0.000585434,-3.32815e-07,-2.22692e-08,0.913853,0.000584702,-3.99622e-07,2.11486e-09,0.914437,0.000583909,-3.93278e-07,1.38098e-08,0.915021,0.000583164,-3.51848e-07,2.25042e-09,0.915604,0.000582467,-3.45097e-07,-2.28115e-08,0.916186,0.000581708,-4.13531e-07,2.93911e-08,0.916767,0.000580969,-3.25358e-07,-3.51481e-08,0.917348,0.000580213,-4.30803e-07,5.15967e-08,0.917928,0.000579506,-2.76012e-07,-5.20296e-08,0.918507,0.000578798,-4.32101e-07,3.73124e-08,0.919085,0.000578046,-3.20164e-07,-3.76154e-08,0.919663,0.000577293,-4.3301e-07,5.35447e-08,0.92024,0.000576587,-2.72376e-07,-5.7354e-08,0.920816,0.000575871,-4.44438e-07,5.66621e-08,0.921391,0.000575152,-2.74452e-07,-5.00851e-08,0.921966,0.000574453,-4.24707e-07,2.4469e-08,0.92254,0.000573677,-3.513e-07,1.18138e-08,0.923114,0.000573009,-3.15859e-07,-1.21195e-08,0.923686,0.000572341,-3.52217e-07,-2.29403e-08,0.924258,0.000571568,-4.21038e-07,4.4276e-08,0.924829,0.000570859,-2.8821e-07,-3.49546e-08,0.9254,0.000570178,-3.93074e-07,3.59377e-08,0.92597,0.000569499,-2.85261e-07,-4.91915e-08,0.926539,0.000568781,-4.32835e-07,4.16189e-08,0.927107,0.00056804,-3.07979e-07,1.92523e-09,0.927675,0.00056743,-3.02203e-07,-4.93198e-08,0.928242,0.000566678,-4.50162e-07,7.61447e-08,0.928809,0.000566006,-2.21728e-07,-7.6445e-08,0.929374,0.000565333,-4.51063e-07,5.08216e-08,0.929939,0.000564583,-2.98599e-07,-7.63212e-09,0.930503,0.000563963,-3.21495e-07,-2.02931e-08,0.931067,0.000563259,-3.82374e-07,2.92001e-08,0.93163,0.000562582,-2.94774e-07,-3.69025e-08,0.932192,0.000561882,-4.05482e-07,5.88053e-08,0.932754,0.000561247,-2.29066e-07,-7.91094e-08,0.933315,0.000560552,-4.66394e-07,7.88184e-08,0.933875,0.000559856,-2.29939e-07,-5.73501e-08,0.934434,0.000559224,-4.01989e-07,3.13727e-08,0.934993,0.000558514,-3.07871e-07,-8.53611e-09,0.935551,0.000557873,-3.33479e-07,2.77175e-09,0.936109,0.000557214,-3.25164e-07,-2.55091e-09,0.936666,0.000556556,-3.32817e-07,7.43188e-09,0.937222,0.000555913,-3.10521e-07,-2.71766e-08,0.937778,0.00055521,-3.92051e-07,4.167e-08,0.938333,0.000554551,-2.67041e-07,-2.02941e-08,0.938887,0.000553956,-3.27923e-07,-2.00984e-08,0.93944,0.00055324,-3.88218e-07,4.10828e-08,0.939993,0.000552587,-2.6497e-07,-2.50237e-08,0.940546,0.000551982,-3.40041e-07,-5.92583e-10,0.941097,0.0005513,-3.41819e-07,2.7394e-08,0.941648,0.000550698,-2.59637e-07,-4.93788e-08,0.942199,0.000550031,-4.07773e-07,5.09119e-08,0.942748,0.000549368,-2.55038e-07,-3.50595e-08,0.943297,0.000548753,-3.60216e-07,2.97214e-08,0.943846,0.000548122,-2.71052e-07,-2.42215e-08,0.944394,0.000547507,-3.43716e-07,7.55985e-09,0.944941,0.000546842,-3.21037e-07,-6.01796e-09,0.945487,0.000546182,-3.3909e-07,1.65119e-08,0.946033,0.000545553,-2.89555e-07,-4.2498e-10,0.946578,0.000544973,-2.9083e-07,-1.4812e-08,0.947123,0.000544347,-3.35266e-07,6.83068e-11,0.947667,0.000543676,-3.35061e-07,1.45388e-08,0.94821,0.00054305,-2.91444e-07,1.38123e-09,0.948753,0.000542471,-2.87301e-07,-2.00637e-08,0.949295,0.000541836,-3.47492e-07,1.92688e-08,0.949837,0.000541199,-2.89685e-07,2.59298e-09,0.950378,0.000540628,-2.81906e-07,-2.96407e-08,0.950918,0.000539975,-3.70829e-07,5.63652e-08,0.951458,0.000539402,-2.01733e-07,-7.66107e-08,0.951997,0.000538769,-4.31565e-07,7.12638e-08,0.952535,0.00053812,-2.17774e-07,-2.96305e-08,0.953073,0.000537595,-3.06665e-07,-1.23464e-08,0.95361,0.000536945,-3.43704e-07,1.94114e-08,0.954147,0.000536316,-2.8547e-07,-5.69451e-09,0.954683,0.000535728,-3.02554e-07,3.36666e-09,0.955219,0.000535133,-2.92454e-07,-7.77208e-09,0.955753,0.000534525,-3.1577e-07,2.77216e-08,0.956288,0.000533976,-2.32605e-07,-4.35097e-08,0.956821,0.00053338,-3.63134e-07,2.7108e-08,0.957354,0.000532735,-2.8181e-07,-5.31772e-09,0.957887,0.000532156,-2.97764e-07,-5.83718e-09,0.958419,0.000531543,-3.15275e-07,2.86664e-08,0.95895,0.000530998,-2.29276e-07,-4.9224e-08,0.959481,0.000530392,-3.76948e-07,4.90201e-08,0.960011,0.000529785,-2.29887e-07,-2.76471e-08,0.96054,0.000529243,-3.12829e-07,1.96385e-09,0.961069,0.000528623,-3.06937e-07,1.97917e-08,0.961598,0.000528068,-2.47562e-07,-2.15261e-08,0.962125,0.000527508,-3.1214e-07,6.70795e-09,0.962653,0.000526904,-2.92016e-07,-5.30573e-09,0.963179,0.000526304,-3.07934e-07,1.4515e-08,0.963705,0.000525732,-2.64389e-07,6.85048e-09,0.964231,0.000525224,-2.43837e-07,-4.19169e-08,0.964756,0.00052461,-3.69588e-07,4.1608e-08,0.96528,0.000523996,-2.44764e-07,-5.30598e-09,0.965804,0.000523491,-2.60682e-07,-2.03841e-08,0.966327,0.000522908,-3.21834e-07,2.72378e-08,0.966849,0.000522346,-2.40121e-07,-2.89625e-08,0.967371,0.000521779,-3.27008e-07,2.90075e-08,0.967893,0.000521212,-2.39986e-07,-2.74629e-08,0.968414,0.00052065,-3.22374e-07,2.12396e-08,0.968934,0.000520069,-2.58656e-07,2.10922e-09,0.969454,0.000519558,-2.52328e-07,-2.96765e-08,0.969973,0.000518964,-3.41357e-07,5.6992e-08,0.970492,0.000518452,-1.70382e-07,-7.90821e-08,0.97101,0.000517874,-4.07628e-07,8.05224e-08,0.971528,0.000517301,-1.66061e-07,-6.41937e-08,0.972045,0.000516776,-3.58642e-07,5.70429e-08,0.972561,0.00051623,-1.87513e-07,-4.47686e-08,0.973077,0.00051572,-3.21819e-07,2.82237e-09,0.973593,0.000515085,-3.13352e-07,3.34792e-08,0.974108,0.000514559,-2.12914e-07,-1.75298e-08,0.974622,0.000514081,-2.65503e-07,-2.29648e-08,0.975136,0.000513481,-3.34398e-07,4.97843e-08,0.975649,0.000512961,-1.85045e-07,-5.6963e-08,0.976162,0.00051242,-3.55934e-07,5.88585e-08,0.976674,0.000511885,-1.79359e-07,-5.92616e-08,0.977185,0.000511348,-3.57143e-07,5.89785e-08,0.977696,0.000510811,-1.80208e-07,-5.74433e-08,0.978207,0.000510278,-3.52538e-07,5.15854e-08,0.978717,0.000509728,-1.97781e-07,-2.9689e-08,0.979226,0.000509243,-2.86848e-07,7.56591e-09,0.979735,0.000508692,-2.64151e-07,-5.74649e-10,0.980244,0.000508162,-2.65875e-07,-5.26732e-09,0.980752,0.000507615,-2.81677e-07,2.16439e-08,0.981259,0.000507116,-2.16745e-07,-2.17037e-08,0.981766,0.000506618,-2.81856e-07,5.56636e-09,0.982272,0.000506071,-2.65157e-07,-5.61689e-10,0.982778,0.000505539,-2.66842e-07,-3.31963e-09,0.983283,0.000504995,-2.76801e-07,1.38402e-08,0.983788,0.000504483,-2.3528e-07,7.56339e-09,0.984292,0.000504035,-2.1259e-07,-4.40938e-08,0.984796,0.000503478,-3.44871e-07,4.96026e-08,0.985299,0.000502937,-1.96064e-07,-3.51071e-08,0.985802,0.000502439,-3.01385e-07,3.12212e-08,0.986304,0.00050193,-2.07721e-07,-3.0173e-08,0.986806,0.000501424,-2.9824e-07,2.9866e-08,0.987307,0.000500917,-2.08642e-07,-2.96865e-08,0.987808,0.000500411,-2.97702e-07,2.92753e-08,0.988308,0.000499903,-2.09876e-07,-2.78101e-08,0.988807,0.0004994,-2.93306e-07,2.23604e-08,0.989307,0.000498881,-2.26225e-07,-2.02681e-09,0.989805,0.000498422,-2.32305e-07,-1.42531e-08,0.990303,0.000497915,-2.75065e-07,-5.65232e-10,0.990801,0.000497363,-2.76761e-07,1.65141e-08,0.991298,0.000496859,-2.27218e-07,-5.88639e-09,0.991795,0.000496387,-2.44878e-07,7.0315e-09,0.992291,0.000495918,-2.23783e-07,-2.22396e-08,0.992787,0.000495404,-2.90502e-07,2.23224e-08,0.993282,0.00049489,-2.23535e-07,-7.44543e-09,0.993776,0.000494421,-2.45871e-07,7.45924e-09,0.994271,0.000493951,-2.23493e-07,-2.23915e-08,0.994764,0.000493437,-2.90668e-07,2.25021e-08,0.995257,0.000492923,-2.23161e-07,-8.01218e-09,0.99575,0.000492453,-2.47198e-07,9.54669e-09,0.996242,0.000491987,-2.18558e-07,-3.01746e-08,0.996734,0.000491459,-3.09082e-07,5.1547e-08,0.997225,0.000490996,-1.54441e-07,-5.68039e-08,0.997716,0.000490517,-3.24853e-07,5.64594e-08,0.998206,0.000490036,-1.55474e-07,-4.98245e-08,0.998696,0.000489576,-3.04948e-07,2.36292e-08,0.999186,0.000489037,-2.3406e-07,1.49121e-08,0.999674,0.000488613,-1.89324e-07,-2.3673e-08,1.00016,0.000488164,-2.60343e-07,2.01754e-08,1.00065,0.000487704,-1.99816e-07,-5.70288e-08,1.00114,0.000487133,-3.70903e-07,8.87303e-08,1.00162,0.000486657,-1.04712e-07,-5.94737e-08,1.00211,0.000486269,-2.83133e-07,2.99553e-08,1.0026,0.000485793,-1.93267e-07,-6.03474e-08,1.00308,0.000485225,-3.74309e-07,9.2225e-08,1.00357,0.000484754,-9.76345e-08,-7.0134e-08,1.00405,0.000484348,-3.08036e-07,6.91016e-08,1.00454,0.000483939,-1.00731e-07,-8.70633e-08,1.00502,0.000483476,-3.61921e-07,4.07328e-08,1.0055,0.000482875,-2.39723e-07,4.33413e-08,1.00599,0.000482525,-1.09699e-07,-9.48886e-08,1.00647,0.000482021,-3.94365e-07,9.77947e-08,1.00695,0.000481526,-1.00981e-07,-5.78713e-08,1.00743,0.00048115,-2.74595e-07,1.44814e-08,1.00791,0.000480645,-2.31151e-07,-5.42665e-11,1.00839,0.000480182,-2.31314e-07,-1.42643e-08,1.00887,0.000479677,-2.74106e-07,5.71115e-08,1.00935,0.0004793,-1.02772e-07,-9.49724e-08,1.00983,0.000478809,-3.87689e-07,8.43596e-08,1.01031,0.000478287,-1.3461e-07,-4.04755e-09,1.01079,0.000478006,-1.46753e-07,-6.81694e-08,1.01127,0.000477508,-3.51261e-07,3.83067e-08,1.01174,0.00047692,-2.36341e-07,3.41521e-08,1.01222,0.00047655,-1.33885e-07,-5.57058e-08,1.0127,0.000476115,-3.01002e-07,6.94616e-08,1.01317,0.000475721,-9.26174e-08,-1.02931e-07,1.01365,0.000475227,-4.01412e-07,1.03846e-07,1.01412,0.000474736,-8.98751e-08,-7.40321e-08,1.0146,0.000474334,-3.11971e-07,7.30735e-08,1.01507,0.00047393,-9.27508e-08,-9.90527e-08,1.01554,0.000473447,-3.89909e-07,8.47188e-08,1.01602,0.000472921,-1.35753e-07,-1.40381e-09,1.01649,0.000472645,-1.39964e-07,-7.91035e-08,1.01696,0.000472128,-3.77275e-07,7.93993e-08,1.01744,0.000471612,-1.39077e-07,-7.52607e-11,1.01791,0.000471334,-1.39302e-07,-7.90983e-08,1.01838,0.000470818,-3.76597e-07,7.80499e-08,1.01885,0.000470299,-1.42448e-07,5.31733e-09,1.01932,0.00047003,-1.26496e-07,-9.93193e-08,1.01979,0.000469479,-4.24453e-07,1.53541e-07,1.02026,0.00046909,3.617e-08,-1.57217e-07,1.02073,0.000468691,-4.35482e-07,1.177e-07,1.02119,0.000468173,-8.23808e-08,-7.51659e-08,1.02166,0.000467783,-3.07878e-07,6.37538e-08,1.02213,0.000467358,-1.16617e-07,-6.064e-08,1.0226,0.000466943,-2.98537e-07,5.9597e-08,1.02306,0.000466525,-1.19746e-07,-5.85386e-08,1.02353,0.00046611,-2.95362e-07,5.53482e-08,1.024,0.000465685,-1.29317e-07,-4.36449e-08,1.02446,0.000465296,-2.60252e-07,2.20268e-11,1.02493,0.000464775,-2.60186e-07,4.35568e-08,1.02539,0.000464386,-1.29516e-07,-5.50398e-08,1.02586,0.000463961,-2.94635e-07,5.73932e-08,1.02632,0.000463544,-1.22456e-07,-5.53236e-08,1.02678,0.000463133,-2.88426e-07,4.46921e-08,1.02725,0.000462691,-1.5435e-07,-4.23534e-09,1.02771,0.000462369,-1.67056e-07,-2.77507e-08,1.02817,0.000461952,-2.50308e-07,-3.97101e-09,1.02863,0.000461439,-2.62221e-07,4.36348e-08,1.02909,0.000461046,-1.31317e-07,-5.13589e-08,1.02955,0.000460629,-2.85394e-07,4.25913e-08,1.03001,0.000460186,-1.5762e-07,2.0285e-10,1.03047,0.000459871,-1.57011e-07,-4.34027e-08,1.03093,0.000459427,-2.87219e-07,5.41987e-08,1.03139,0.000459015,-1.24623e-07,-5.4183e-08,1.03185,0.000458604,-2.87172e-07,4.33239e-08,1.03231,0.000458159,-1.572e-07,9.65817e-11,1.03277,0.000457845,-1.56911e-07,-4.37103e-08,1.03323,0.0004574,-2.88041e-07,5.55351e-08,1.03368,0.000456991,-1.21436e-07,-5.9221e-08,1.03414,0.00045657,-2.99099e-07,6.21394e-08,1.0346,0.000456158,-1.1268e-07,-7.01275e-08,1.03505,0.000455723,-3.23063e-07,9.91614e-08,1.03551,0.000455374,-2.55788e-08,-8.80996e-08,1.03596,0.000455058,-2.89878e-07,1.48184e-08,1.03642,0.000454523,-2.45422e-07,2.88258e-08,1.03687,0.000454119,-1.58945e-07,-1.09125e-08,1.03733,0.000453768,-1.91682e-07,1.48241e-08,1.03778,0.000453429,-1.4721e-07,-4.83838e-08,1.03823,0.00045299,-2.92361e-07,5.95019e-08,1.03869,0.000452584,-1.13856e-07,-7.04146e-08,1.03914,0.000452145,-3.25099e-07,1.02947e-07,1.03959,0.000451803,-1.62583e-08,-1.02955e-07,1.04004,0.000451462,-3.25123e-07,7.04544e-08,1.04049,0.000451023,-1.1376e-07,-5.96534e-08,1.04094,0.000450616,-2.9272e-07,4.89499e-08,1.04139,0.000450178,-1.45871e-07,-1.69369e-08,1.04184,0.000449835,-1.96681e-07,1.87977e-08,1.04229,0.000449498,-1.40288e-07,-5.82539e-08,1.04274,0.000449043,-3.1505e-07,9.50087e-08,1.04319,0.000448698,-3.00238e-08,-8.33623e-08,1.04364,0.000448388,-2.80111e-07,2.20363e-11,1.04409,0.000447828,-2.80045e-07,8.32742e-08,1.04454,0.000447517,-3.02221e-08,-9.47002e-08,1.04498,0.000447173,-3.14323e-07,5.7108e-08,1.04543,0.000446716,-1.42999e-07,-1.45225e-08,1.04588,0.000446386,-1.86566e-07,9.82022e-10,1.04632,0.000446016,-1.8362e-07,1.05944e-08,1.04677,0.00044568,-1.51837e-07,-4.33597e-08,1.04721,0.000445247,-2.81916e-07,4.36352e-08,1.04766,0.000444814,-1.51011e-07,-1.19717e-08,1.0481,0.000444476,-1.86926e-07,4.25158e-09,1.04855,0.000444115,-1.74171e-07,-5.03461e-09,1.04899,0.000443751,-1.89275e-07,1.58868e-08,1.04944,0.00044342,-1.41614e-07,-5.85127e-08,1.04988,0.000442961,-3.17152e-07,9.89548e-08,1.05032,0.000442624,-2.0288e-08,-9.88878e-08,1.05076,0.000442287,-3.16951e-07,5.81779e-08,1.05121,0.000441827,-1.42418e-07,-1.46144e-08,1.05165,0.000441499,-1.86261e-07,2.79892e-10,1.05209,0.000441127,-1.85421e-07,1.34949e-08,1.05253,0.000440797,-1.44937e-07,-5.42594e-08,1.05297,0.000440344,-3.07715e-07,8.43335e-08,1.05341,0.000439982,-5.47146e-08,-4.46558e-08,1.05385,0.000439738,-1.88682e-07,-2.49193e-08,1.05429,0.000439286,-2.6344e-07,2.5124e-08,1.05473,0.000438835,-1.88068e-07,4.36328e-08,1.05517,0.000438589,-5.71699e-08,-8.04459e-08,1.05561,0.000438234,-2.98508e-07,3.97324e-08,1.05605,0.000437756,-1.79311e-07,4.07258e-08,1.05648,0.000437519,-5.71332e-08,-8.34263e-08,1.05692,0.000437155,-3.07412e-07,5.45608e-08,1.05736,0.000436704,-1.4373e-07,-1.56078e-08,1.05779,0.000436369,-1.90553e-07,7.87043e-09,1.05823,0.000436012,-1.66942e-07,-1.58739e-08,1.05867,0.00043563,-2.14563e-07,5.56251e-08,1.0591,0.000435368,-4.76881e-08,-8.74172e-08,1.05954,0.000435011,-3.0994e-07,5.56251e-08,1.05997,0.000434558,-1.43064e-07,-1.58739e-08,1.06041,0.000434224,-1.90686e-07,7.87042e-09,1.06084,0.000433866,-1.67075e-07,-1.56078e-08,1.06127,0.000433485,-2.13898e-07,5.45609e-08,1.06171,0.000433221,-5.02157e-08,-8.34263e-08,1.06214,0.00043287,-3.00495e-07,4.07258e-08,1.06257,0.000432391,-1.78317e-07,3.97325e-08,1.063,0.000432154,-5.91198e-08,-8.04464e-08,1.06344,0.000431794,-3.00459e-07,4.36347e-08,1.06387,0.000431324,-1.69555e-07,2.5117e-08,1.0643,0.000431061,-9.42041e-08,-2.48934e-08,1.06473,0.000430798,-1.68884e-07,-4.47527e-08,1.06516,0.000430326,-3.03142e-07,8.46951e-08,1.06559,0.000429973,-4.90573e-08,-5.56089e-08,1.06602,0.000429708,-2.15884e-07,1.85314e-08,1.06645,0.000429332,-1.6029e-07,-1.85166e-08,1.06688,0.000428956,-2.1584e-07,5.5535e-08,1.06731,0.000428691,-4.92347e-08,-8.44142e-08,1.06774,0.000428339,-3.02477e-07,4.37032e-08,1.06816,0.000427865,-1.71368e-07,2.88107e-08,1.06859,0.000427609,-8.49356e-08,-3.97367e-08,1.06902,0.00042732,-2.04146e-07,1.09267e-08,1.06945,0.000426945,-1.71365e-07,-3.97023e-09,1.06987,0.00042659,-1.83276e-07,4.9542e-09,1.0703,0.000426238,-1.68414e-07,-1.58466e-08,1.07073,0.000425854,-2.15953e-07,5.84321e-08,1.07115,0.000425597,-4.0657e-08,-9.86725e-08,1.07158,0.00042522,-3.36674e-07,9.78392e-08,1.072,0.00042484,-4.31568e-08,-5.42658e-08,1.07243,0.000424591,-2.05954e-07,1.45377e-11,1.07285,0.000424179,-2.0591e-07,5.42076e-08,1.07328,0.00042393,-4.32877e-08,-9.76357e-08,1.0737,0.00042355,-3.36195e-07,9.79165e-08,1.07412,0.000423172,-4.24451e-08,-5.56118e-08,1.07455,0.00042292,-2.09281e-07,5.32143e-09,1.07497,0.000422518,-1.93316e-07,3.43261e-08,1.07539,0.000422234,-9.0338e-08,-2.34165e-08,1.07581,0.000421983,-1.60588e-07,-5.98692e-08,1.07623,0.000421482,-3.40195e-07,1.43684e-07,1.07666,0.000421233,9.08574e-08,-1.5724e-07,1.07708,0.000420943,-3.80862e-07,1.27647e-07,1.0775,0.000420564,2.0791e-09,-1.1493e-07,1.07792,0.000420223,-3.4271e-07,9.36534e-08,1.07834,0.000419819,-6.17499e-08,-2.12653e-08,1.07876,0.000419632,-1.25546e-07,-8.59219e-09,1.07918,0.000419355,-1.51322e-07,-6.35752e-08,1.0796,0.000418861,-3.42048e-07,1.43684e-07,1.08002,0.000418608,8.90034e-08,-1.53532e-07,1.08043,0.000418326,-3.71593e-07,1.12817e-07,1.08085,0.000417921,-3.31414e-08,-5.93184e-08,1.08127,0.000417677,-2.11097e-07,5.24697e-09,1.08169,0.00041727,-1.95356e-07,3.83305e-08,1.0821,0.000416995,-8.03642e-08,-3.93597e-08,1.08252,0.000416716,-1.98443e-07,-1.0094e-10,1.08294,0.000416319,-1.98746e-07,3.97635e-08,1.08335,0.00041604,-7.94557e-08,-3.97437e-08,1.08377,0.000415762,-1.98687e-07,1.94215e-12,1.08419,0.000415365,-1.98681e-07,3.97359e-08,1.0846,0.000415087,-7.94732e-08,-3.97362e-08,1.08502,0.000414809,-1.98682e-07,-4.31063e-13,1.08543,0.000414411,-1.98683e-07,3.97379e-08,1.08584,0.000414133,-7.94694e-08,-3.97418e-08,1.08626,0.000413855,-1.98695e-07,2.00563e-11,1.08667,0.000413458,-1.98635e-07,3.96616e-08,1.08709,0.000413179,-7.965e-08,-3.9457e-08,1.0875,0.000412902,-1.98021e-07,-1.04281e-09,1.08791,0.000412502,-2.01149e-07,4.36282e-08,1.08832,0.000412231,-7.02648e-08,-5.42608e-08,1.08874,0.000411928,-2.33047e-07,5.42057e-08,1.08915,0.000411624,-7.04301e-08,-4.33527e-08,1.08956,0.000411353,-2.00488e-07,-4.07378e-12,1.08997,0.000410952,-2.005e-07,4.3369e-08,1.09038,0.000410681,-7.03934e-08,-5.42627e-08,1.09079,0.000410378,-2.33182e-07,5.44726e-08,1.0912,0.000410075,-6.97637e-08,-4.44186e-08,1.09161,0.000409802,-2.03019e-07,3.99235e-09,1.09202,0.000409408,-1.91042e-07,2.84491e-08,1.09243,0.000409111,-1.05695e-07,1.42043e-09,1.09284,0.000408904,-1.01434e-07,-3.41308e-08,1.09325,0.000408599,-2.03826e-07,1.58937e-08,1.09366,0.000408239,-1.56145e-07,-2.94438e-08,1.09406,0.000407838,-2.44476e-07,1.01881e-07,1.09447,0.000407655,6.11676e-08,-1.39663e-07,1.09488,0.000407358,-3.57822e-07,9.91432e-08,1.09529,0.00040694,-6.03921e-08,-1.84912e-08,1.09569,0.000406764,-1.15866e-07,-2.51785e-08,1.0961,0.000406457,-1.91401e-07,-4.03115e-12,1.09651,0.000406074,-1.91413e-07,2.51947e-08,1.09691,0.000405767,-1.15829e-07,1.84346e-08,1.09732,0.00040559,-6.05254e-08,-9.89332e-08,1.09772,0.000405172,-3.57325e-07,1.3888e-07,1.09813,0.000404874,5.93136e-08,-9.8957e-08,1.09853,0.000404696,-2.37557e-07,1.853e-08,1.09894,0.000404277,-1.81968e-07,2.48372e-08,1.09934,0.000403987,-1.07456e-07,1.33047e-09,1.09975,0.000403776,-1.03465e-07,-3.01591e-08,1.10015,0.000403479,-1.93942e-07,9.66054e-11,1.10055,0.000403091,-1.93652e-07,2.97727e-08,1.10096,0.000402793,-1.04334e-07,2.19273e-11,1.10136,0.000402585,-1.04268e-07,-2.98604e-08,1.10176,0.000402287,-1.93849e-07,2.10325e-10,1.10216,0.0004019,-1.93218e-07,2.90191e-08,1.10256,0.0004016,-1.06161e-07,2.92264e-09,1.10297,0.000401397,-9.73931e-08,-4.07096e-08,1.10337,0.00040108,-2.19522e-07,4.07067e-08,1.10377,0.000400763,-9.7402e-08,-2.90783e-09,1.10417,0.000400559,-1.06126e-07,-2.90754e-08,1.10457,0.00040026,-1.93352e-07,9.00021e-14,1.10497,0.000399873,-1.93351e-07,2.9075e-08,1.10537,0.000399574,-1.06126e-07,2.90902e-09,1.10577,0.00039937,-9.73992e-08,-4.07111e-08,1.10617,0.000399053,-2.19533e-07,4.07262e-08,1.10657,0.000398736,-9.73541e-08,-2.98424e-09,1.10697,0.000398533,-1.06307e-07,-2.87892e-08,1.10736,0.000398234,-1.92674e-07,-1.06824e-09,1.10776,0.000397845,-1.95879e-07,3.30622e-08,1.10816,0.000397552,-9.66926e-08,-1.19712e-08,1.10856,0.000397323,-1.32606e-07,1.48225e-08,1.10895,0.000397102,-8.81387e-08,-4.73187e-08,1.10935,0.000396784,-2.30095e-07,5.52429e-08,1.10975,0.00039649,-6.4366e-08,-5.44437e-08,1.11014,0.000396198,-2.27697e-07,4.33226e-08,1.11054,0.000395872,-9.77293e-08,3.62656e-10,1.11094,0.000395678,-9.66414e-08,-4.47732e-08,1.11133,0.00039535,-2.30961e-07,5.95208e-08,1.11173,0.000395067,-5.23985e-08,-7.41008e-08,1.11212,0.00039474,-2.74701e-07,1.17673e-07,1.11252,0.000394543,7.83181e-08,-1.58172e-07,1.11291,0.000394225,-3.96199e-07,1.57389e-07,1.1133,0.000393905,7.59679e-08,-1.13756e-07,1.1137,0.000393716,-2.653e-07,5.92165e-08,1.11409,0.000393363,-8.76507e-08,-3.90074e-09,1.11449,0.000393176,-9.93529e-08,-4.36136e-08,1.11488,0.000392846,-2.30194e-07,5.91457e-08,1.11527,0.000392563,-5.27564e-08,-7.376e-08,1.11566,0.000392237,-2.74037e-07,1.16685e-07,1.11606,0.000392039,7.60189e-08,-1.54562e-07,1.11645,0.000391727,-3.87667e-07,1.43935e-07,1.11684,0.000391384,4.4137e-08,-6.35487e-08,1.11723,0.000391281,-1.46509e-07,-8.94896e-09,1.11762,0.000390961,-1.73356e-07,-1.98647e-08,1.11801,0.000390555,-2.3295e-07,8.8408e-08,1.1184,0.000390354,3.22736e-08,-9.53486e-08,1.11879,0.000390133,-2.53772e-07,5.45677e-08,1.11918,0.000389789,-9.0069e-08,-3.71296e-09,1.11957,0.000389598,-1.01208e-07,-3.97159e-08,1.11996,0.000389276,-2.20355e-07,4.33671e-08,1.12035,0.000388966,-9.02542e-08,-1.45431e-08,1.12074,0.000388741,-1.33883e-07,1.48052e-08,1.12113,0.000388518,-8.94678e-08,-4.46778e-08,1.12152,0.000388205,-2.23501e-07,4.46966e-08,1.12191,0.000387892,-8.94114e-08,-1.48992e-08,1.12229,0.000387669,-1.34109e-07,1.49003e-08,1.12268,0.000387445,-8.94082e-08,-4.47019e-08,1.12307,0.000387132,-2.23514e-07,4.4698e-08,1.12345,0.000386819,-8.942e-08,-1.48806e-08,1.12384,0.000386596,-1.34062e-07,1.48245e-08,1.12423,0.000386372,-8.95885e-08,-4.44172e-08,1.12461,0.00038606,-2.2284e-07,4.36351e-08,1.125,0.000385745,-9.19348e-08,-1.09139e-08,1.12539,0.000385528,-1.24677e-07,2.05584e-11,1.12577,0.000385279,-1.24615e-07,1.08317e-08,1.12616,0.000385062,-9.21198e-08,-4.33473e-08,1.12654,0.000384748,-2.22162e-07,4.33481e-08,1.12693,0.000384434,-9.21174e-08,-1.08356e-08,1.12731,0.000384217,-1.24624e-07,-5.50907e-12,1.12769,0.000383968,-1.24641e-07,1.08577e-08,1.12808,0.000383751,-9.20679e-08,-4.34252e-08,1.12846,0.000383437,-2.22343e-07,4.36337e-08,1.12884,0.000383123,-9.14422e-08,-1.19005e-08,1.12923,0.000382904,-1.27144e-07,3.96813e-09,1.12961,0.000382662,-1.15239e-07,-3.97207e-09,1.12999,0.000382419,-1.27155e-07,1.19201e-08,1.13038,0.000382201,-9.1395e-08,-4.37085e-08,1.13076,0.000381887,-2.2252e-07,4.37046e-08,1.13114,0.000381573,-9.14068e-08,-1.19005e-08,1.13152,0.000381355,-1.27108e-07,3.89734e-09,1.1319,0.000381112,-1.15416e-07,-3.68887e-09,1.13228,0.00038087,-1.26483e-07,1.08582e-08,1.13266,0.00038065,-9.39083e-08,-3.97438e-08,1.13304,0.000380343,-2.1314e-07,2.89076e-08,1.13342,0.000380003,-1.26417e-07,4.33225e-08,1.1338,0.00037988,3.55072e-09,-8.29883e-08,1.13418,0.000379638,-2.45414e-07,5.0212e-08,1.13456,0.000379298,-9.47781e-08,1.34964e-09,1.13494,0.000379113,-9.07292e-08,-5.56105e-08,1.13532,0.000378764,-2.57561e-07,1.01883e-07,1.1357,0.000378555,4.80889e-08,-1.13504e-07,1.13608,0.000378311,-2.92423e-07,1.13713e-07,1.13646,0.000378067,4.87176e-08,-1.02931e-07,1.13683,0.000377856,-2.60076e-07,5.95923e-08,1.13721,0.000377514,-8.12988e-08,-1.62288e-08,1.13759,0.000377303,-1.29985e-07,5.32278e-09,1.13797,0.000377059,-1.14017e-07,-5.06237e-09,1.13834,0.000376816,-1.29204e-07,1.49267e-08,1.13872,0.000376602,-8.44237e-08,-5.46444e-08,1.1391,0.000376269,-2.48357e-07,8.44417e-08,1.13947,0.000376026,4.96815e-09,-4.47039e-08,1.13985,0.000375902,-1.29143e-07,-2.48355e-08,1.14023,0.000375569,-2.0365e-07,2.48368e-08,1.1406,0.000375236,-1.2914e-07,4.46977e-08,1.14098,0.000375112,4.95341e-09,-8.44184e-08,1.14135,0.000374869,-2.48302e-07,5.45572e-08,1.14173,0.000374536,-8.463e-08,-1.46013e-08,1.1421,0.000374323,-1.28434e-07,3.8478e-09,1.14247,0.000374077,-1.1689e-07,-7.89941e-10,1.14285,0.000373841,-1.1926e-07,-6.88042e-10,1.14322,0.0003736,-1.21324e-07,3.54213e-09,1.1436,0.000373368,-1.10698e-07,-1.34805e-08,1.14397,0.000373107,-1.51139e-07,5.03798e-08,1.14434,0.000372767,0.,0.};

        template <bool srgb, int blueIdx, typename T, typename D>
        __device__ __forceinline__ void RGB2LuvConvert_f(const T& src, D& dst)
        {
            const float _d = 1.f / (0.950456f + 15 + 1.088754f * 3);
            const float _un = 13 * (4 * 0.950456f * _d);
            const float _vn = 13 * (9 * _d);

            float B = blueIdx == 0 ? src.x : src.z;
            float G = src.y;
            float R = blueIdx == 0 ? src.z : src.x;

            if (srgb)
            {
                B = splineInterpolate(B * GAMMA_TAB_SIZE, c_sRGBGammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G * GAMMA_TAB_SIZE, c_sRGBGammaTab, GAMMA_TAB_SIZE);
                R = splineInterpolate(R * GAMMA_TAB_SIZE, c_sRGBGammaTab, GAMMA_TAB_SIZE);
            }

            float X = R * 0.412453f + G * 0.357580f + B * 0.180423f;
            float Y = R * 0.212671f + G * 0.715160f + B * 0.072169f;
            float Z = R * 0.019334f + G * 0.119193f + B * 0.950227f;

            float L = splineInterpolate(Y * (LAB_CBRT_TAB_SIZE / 1.5f), c_LabCbrtTab, LAB_CBRT_TAB_SIZE);
            L = 116.f * L - 16.f;

            const float d = (4 * 13) / ::fmaxf(X + 15 * Y + 3 * Z, numeric_limits<float>::epsilon());
            float u = L * (X * d - _un);
            float v = L * ((9 * 0.25f) * Y * d - _vn);

            dst.x = L;
            dst.y = u;
            dst.z = v;
        }

        template <bool srgb, int blueIdx, typename T, typename D>
        __device__ __forceinline__ void RGB2LuvConvert_b(const T& src, D& dst)
        {
            float3 srcf, dstf;

            srcf.x = src.x * (1.f / 255.f);
            srcf.y = src.y * (1.f / 255.f);
            srcf.z = src.z * (1.f / 255.f);

            RGB2LuvConvert_f<srgb, blueIdx>(srcf, dstf);

            dst.x = saturate_cast<uchar>(dstf.x * 2.55f);
            dst.y = saturate_cast<uchar>(dstf.y * 0.72033898305084743f + 96.525423728813564f);
            dst.z = saturate_cast<uchar>(dstf.z * 0.9732824427480916f + 136.259541984732824f);
        }

        template <typename T, int scn, int dcn, bool srgb, int blueIdx> struct RGB2Luv;
        template <int scn, int dcn, bool srgb, int blueIdx>
        struct RGB2Luv<uchar, scn, dcn, srgb, blueIdx>
            : unary_function<typename TypeVec<uchar, scn>::vec_type, typename TypeVec<uchar, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<uchar, dcn>::vec_type operator ()(const typename TypeVec<uchar, scn>::vec_type& src) const
            {
                typename TypeVec<uchar, dcn>::vec_type dst;

                RGB2LuvConvert_b<srgb, blueIdx>(src, dst);

                return dst;
            }
            __host__ __device__ __forceinline__ RGB2Luv() {}
            __host__ __device__ __forceinline__ RGB2Luv(const RGB2Luv&) {}
        };
        template <int scn, int dcn, bool srgb, int blueIdx>
        struct RGB2Luv<float, scn, dcn, srgb, blueIdx>
            : unary_function<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<float, dcn>::vec_type operator ()(const typename TypeVec<float, scn>::vec_type& src) const
            {
                typename TypeVec<float, dcn>::vec_type dst;

                RGB2LuvConvert_f<srgb, blueIdx>(src, dst);

                return dst;
            }
            __host__ __device__ __forceinline__ RGB2Luv() {}
            __host__ __device__ __forceinline__ RGB2Luv(const RGB2Luv&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_RGB2Luv_TRAITS(name, scn, dcn, srgb, blueIdx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::RGB2Luv<T, scn, dcn, srgb, blueIdx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

    namespace color_detail
    {
        template <bool srgb, int blueIdx, typename T, typename D>
        __device__ __forceinline__ void Luv2RGBConvert_f(const T& src, D& dst)
        {
            const float _d = 1.f / (0.950456f + 15 + 1.088754f * 3);
            const float _un = 4 * 0.950456f * _d;
            const float _vn = 9 * _d;

            float L = src.x;
            float u = src.y;
            float v = src.z;

            float Y = (L + 16.f) * (1.f / 116.f);
            Y = Y * Y * Y;

            float d = (1.f / 13.f) / L;
            u = u * d + _un;
            v = v * d + _vn;

            float iv = 1.f / v;
            float X = 2.25f * u * Y * iv;
            float Z = (12 - 3 * u - 20 * v) * Y * 0.25f * iv;

            float B = 0.055648f * X - 0.204043f * Y + 1.057311f * Z;
            float G = -0.969256f * X + 1.875991f * Y + 0.041556f * Z;
            float R = 3.240479f * X - 1.537150f * Y - 0.498535f * Z;

            if (srgb)
            {
                B = splineInterpolate(B * GAMMA_TAB_SIZE, c_sRGBInvGammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G * GAMMA_TAB_SIZE, c_sRGBInvGammaTab, GAMMA_TAB_SIZE);
                R = splineInterpolate(R * GAMMA_TAB_SIZE, c_sRGBInvGammaTab, GAMMA_TAB_SIZE);
            }

            dst.x = blueIdx == 0 ? B : R;
            dst.y = G;
            dst.z = blueIdx == 0 ? R : B;
            setAlpha(dst, ColorChannel<float>::max());
        }

        template <bool srgb, int blueIdx, typename T, typename D>
        __device__ __forceinline__ void Luv2RGBConvert_b(const T& src, D& dst)
        {
            float3 srcf, dstf;

            srcf.x = src.x * (100.f / 255.f);
            srcf.y = src.y * 1.388235294117647f - 134.f;
            srcf.z = src.z * 1.027450980392157f - 140.f;

            Luv2RGBConvert_f<srgb, blueIdx>(srcf, dstf);

            dst.x = saturate_cast<uchar>(dstf.x * 255.f);
            dst.y = saturate_cast<uchar>(dstf.y * 255.f);
            dst.z = saturate_cast<uchar>(dstf.z * 255.f);
            setAlpha(dst, ColorChannel<uchar>::max());
        }

        template <typename T, int scn, int dcn, bool srgb, int blueIdx> struct Luv2RGB;
        template <int scn, int dcn, bool srgb, int blueIdx>
        struct Luv2RGB<uchar, scn, dcn, srgb, blueIdx>
            : unary_function<typename TypeVec<uchar, scn>::vec_type, typename TypeVec<uchar, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<uchar, dcn>::vec_type operator ()(const typename TypeVec<uchar, scn>::vec_type& src) const
            {
                typename TypeVec<uchar, dcn>::vec_type dst;

                Luv2RGBConvert_b<srgb, blueIdx>(src, dst);

                return dst;
            }
            __host__ __device__ __forceinline__ Luv2RGB() {}
            __host__ __device__ __forceinline__ Luv2RGB(const Luv2RGB&) {}
        };
        template <int scn, int dcn, bool srgb, int blueIdx>
        struct Luv2RGB<float, scn, dcn, srgb, blueIdx>
            : unary_function<typename TypeVec<float, scn>::vec_type, typename TypeVec<float, dcn>::vec_type>
        {
            __device__ __forceinline__ typename TypeVec<float, dcn>::vec_type operator ()(const typename TypeVec<float, scn>::vec_type& src) const
            {
                typename TypeVec<float, dcn>::vec_type dst;

                Luv2RGBConvert_f<srgb, blueIdx>(src, dst);

                return dst;
            }
            __host__ __device__ __forceinline__ Luv2RGB() {}
            __host__ __device__ __forceinline__ Luv2RGB(const Luv2RGB&) {}
        };
    }

#define OPENCV_CUDA_IMPLEMENT_Luv2RGB_TRAITS(name, scn, dcn, srgb, blueIdx) \
    template <typename T> struct name ## _traits \
    { \
        typedef ::cv::cuda::device::color_detail::Luv2RGB<T, scn, dcn, srgb, blueIdx> functor_type; \
        static __host__ __device__ __forceinline__ functor_type create_functor() \
        { \
            return functor_type(); \
        } \
    };

    #undef CV_DESCALE

}}} // namespace cv { namespace cuda { namespace cudev

//! @endcond

#endif // __OPENCV_CUDA_COLOR_DETAIL_HPP__
