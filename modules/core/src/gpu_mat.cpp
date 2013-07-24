/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include "precomp.hpp"

using namespace cv;
using namespace cv::gpu;

/////////////////////////// matrix operations /////////////////////////

#ifdef HAVE_CUDA

// CUDA implementation

#include "cuda/matrix_operations.hpp"

namespace
{
    template <typename T> void cudaSet_(GpuMat& src, Scalar s, cudaStream_t stream)
    {
        Scalar_<T> sf = s;
        cudev::set<T>(PtrStepSz<T>(src), sf.val, src.channels(), stream);
    }

    void cudaSet(GpuMat& src, Scalar s, cudaStream_t stream)
    {
        typedef void (*func_t)(GpuMat& src, Scalar s, cudaStream_t stream);
        static const func_t funcs[] =
        {
            cudaSet_<uchar>,
            cudaSet_<schar>,
            cudaSet_<ushort>,
            cudaSet_<short>,
            cudaSet_<int>,
            cudaSet_<float>,
            cudaSet_<double>
        };

        funcs[src.depth()](src, s, stream);
    }

    template <typename T> void cudaSet_(GpuMat& src, Scalar s, PtrStepSzb mask, cudaStream_t stream)
    {
        Scalar_<T> sf = s;
        cudev::set<T>(PtrStepSz<T>(src), sf.val, mask, src.channels(), stream);
    }

    void cudaSet(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream)
    {
        typedef void (*func_t)(GpuMat& src, Scalar s, PtrStepSzb mask, cudaStream_t stream);
        static const func_t funcs[] =
        {
            cudaSet_<uchar>,
            cudaSet_<schar>,
            cudaSet_<ushort>,
            cudaSet_<short>,
            cudaSet_<int>,
            cudaSet_<float>,
            cudaSet_<double>
        };

        funcs[src.depth()](src, s, mask, stream);
    }

    void cudaCopyWithMask(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t stream)
    {
        cudev::copyWithMask(src.reshape(1), dst.reshape(1), src.elemSize1(), src.channels(), mask.reshape(1), mask.channels() != 1, stream);
    }

    void cudaConvert(const GpuMat& src, GpuMat& dst, cudaStream_t stream)
    {
        cudev::convert(src.reshape(1), src.depth(), dst.reshape(1), dst.depth(), 1.0, 0.0, stream);
    }

    void cudaConvert(const GpuMat& src, GpuMat& dst, double alpha, double beta, cudaStream_t stream)
    {
        cudev::convert(src.reshape(1), src.depth(), dst.reshape(1), dst.depth(), alpha, beta, stream);
    }
}

// NPP implementation

namespace
{
    //////////////////////////////////////////////////////////////////////////
    // Convert

    template<int SDEPTH, int DDEPTH> struct NppConvertFunc
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, dst_t* pDst, int nDstStep, NppiSize oSizeROI);
    };
    template<int DDEPTH> struct NppConvertFunc<CV_32F, DDEPTH>
    {
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        typedef NppStatus (*func_ptr)(const Npp32f* pSrc, int nSrcStep, dst_t* pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);
    };

    template<int SDEPTH, int DDEPTH, typename NppConvertFunc<SDEPTH, DDEPTH>::func_ptr func> struct NppCvt
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        static void call(const GpuMat& src, GpuMat& dst, cudaStream_t stream)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), dst.ptr<dst_t>(), static_cast<int>(dst.step), sz) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int DDEPTH, typename NppConvertFunc<CV_32F, DDEPTH>::func_ptr func> struct NppCvt<CV_32F, DDEPTH, func>
    {
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        static void call(const GpuMat& src, GpuMat& dst, cudaStream_t stream)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<Npp32f>(), static_cast<int>(src.step), dst.ptr<dst_t>(), static_cast<int>(dst.step), sz, NPP_RND_NEAR) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    //////////////////////////////////////////////////////////////////////////
    // Set

    template<int SDEPTH, int SCN> struct NppSetFunc
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(const src_t values[], src_t* pSrc, int nSrcStep, NppiSize oSizeROI);
    };
    template<int SDEPTH> struct NppSetFunc<SDEPTH, 1>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(src_t val, src_t* pSrc, int nSrcStep, NppiSize oSizeROI);
    };
    template<int SCN> struct NppSetFunc<CV_8S, SCN>
    {
        typedef NppStatus (*func_ptr)(Npp8s values[], Npp8s* pSrc, int nSrcStep, NppiSize oSizeROI);
    };
    template<> struct NppSetFunc<CV_8S, 1>
    {
        typedef NppStatus (*func_ptr)(Npp8s val, Npp8s* pSrc, int nSrcStep, NppiSize oSizeROI);
    };

    template<int SDEPTH, int SCN, typename NppSetFunc<SDEPTH, SCN>::func_ptr func> struct NppSet
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void call(GpuMat& src, Scalar s, cudaStream_t stream)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Scalar_<src_t> nppS = s;

            NppStreamHandler h(stream);

            nppSafeCall( func(nppS.val, src.ptr<src_t>(), static_cast<int>(src.step), sz) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename NppSetFunc<SDEPTH, 1>::func_ptr func> struct NppSet<SDEPTH, 1, func>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void call(GpuMat& src, Scalar s, cudaStream_t stream)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Scalar_<src_t> nppS = s;

            NppStreamHandler h(stream);

            nppSafeCall( func(nppS[0], src.ptr<src_t>(), static_cast<int>(src.step), sz) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    template<int SDEPTH, int SCN> struct NppSetMaskFunc
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(const src_t values[], src_t* pSrc, int nSrcStep, NppiSize oSizeROI, const Npp8u* pMask, int nMaskStep);
    };
    template<int SDEPTH> struct NppSetMaskFunc<SDEPTH, 1>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(src_t val, src_t* pSrc, int nSrcStep, NppiSize oSizeROI, const Npp8u* pMask, int nMaskStep);
    };

    template<int SDEPTH, int SCN, typename NppSetMaskFunc<SDEPTH, SCN>::func_ptr func> struct NppSetMask
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void call(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Scalar_<src_t> nppS = s;

            NppStreamHandler h(stream);

            nppSafeCall( func(nppS.val, src.ptr<src_t>(), static_cast<int>(src.step), sz, mask.ptr<Npp8u>(), static_cast<int>(mask.step)) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename NppSetMaskFunc<SDEPTH, 1>::func_ptr func> struct NppSetMask<SDEPTH, 1, func>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void call(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Scalar_<src_t> nppS = s;

            NppStreamHandler h(stream);

            nppSafeCall( func(nppS[0], src.ptr<src_t>(), static_cast<int>(src.step), sz, mask.ptr<Npp8u>(), static_cast<int>(mask.step)) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    //////////////////////////////////////////////////////////////////////////
    // CopyMasked

    template<int SDEPTH> struct NppCopyWithMaskFunc
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, src_t* pDst, int nDstStep, NppiSize oSizeROI, const Npp8u* pMask, int nMaskStep);
    };

    template<int SDEPTH, typename NppCopyWithMaskFunc<SDEPTH>::func_ptr func> struct NppCopyWithMask
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void call(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t stream)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), dst.ptr<src_t>(), static_cast<int>(dst.step), sz, mask.ptr<Npp8u>(), static_cast<int>(mask.step)) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
}

// Dispatcher

namespace
{
    void copyWithMask(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t stream = 0)
    {
        CV_DbgAssert( src.size() == dst.size() && src.type() == dst.type() );

        CV_Assert( src.depth() <= CV_64F && src.channels() <= 4 );
        CV_Assert( src.size() == mask.size() && mask.depth() == CV_8U && (mask.channels() == 1 || mask.channels() == src.channels()) );

        if (src.depth() == CV_64F)
        {
            CV_Assert( deviceSupports(NATIVE_DOUBLE) );
        }

        typedef void (*func_t)(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t stream);
        static const func_t funcs[7][4] =
        {
            /*  8U */ {NppCopyWithMask<CV_8U , nppiCopy_8u_C1MR >::call, cudaCopyWithMask, NppCopyWithMask<CV_8U , nppiCopy_8u_C3MR >::call, NppCopyWithMask<CV_8U , nppiCopy_8u_C4MR >::call},
            /*  8S */ {cudaCopyWithMask                                , cudaCopyWithMask, cudaCopyWithMask                                , cudaCopyWithMask                                },
            /* 16U */ {NppCopyWithMask<CV_16U, nppiCopy_16u_C1MR>::call, cudaCopyWithMask, NppCopyWithMask<CV_16U, nppiCopy_16u_C3MR>::call, NppCopyWithMask<CV_16U, nppiCopy_16u_C4MR>::call},
            /* 16S */ {NppCopyWithMask<CV_16S, nppiCopy_16s_C1MR>::call, cudaCopyWithMask, NppCopyWithMask<CV_16S, nppiCopy_16s_C3MR>::call, NppCopyWithMask<CV_16S, nppiCopy_16s_C4MR>::call},
            /* 32S */ {NppCopyWithMask<CV_32S, nppiCopy_32s_C1MR>::call, cudaCopyWithMask, NppCopyWithMask<CV_32S, nppiCopy_32s_C3MR>::call, NppCopyWithMask<CV_32S, nppiCopy_32s_C4MR>::call},
            /* 32F */ {NppCopyWithMask<CV_32F, nppiCopy_32f_C1MR>::call, cudaCopyWithMask, NppCopyWithMask<CV_32F, nppiCopy_32f_C3MR>::call, NppCopyWithMask<CV_32F, nppiCopy_32f_C4MR>::call},
            /* 64F */ {cudaCopyWithMask                                , cudaCopyWithMask, cudaCopyWithMask                                , cudaCopyWithMask                                }
        };

        const func_t func = mask.channels() == src.channels() ? funcs[src.depth()][src.channels() - 1] : cudaCopyWithMask;

        func(src, dst, mask, stream);
    }

    void convert(const GpuMat& src, GpuMat& dst, cudaStream_t stream = 0)
    {
        CV_DbgAssert( src.size() == dst.size() && src.channels() == dst.channels() );

        CV_Assert( src.depth() <= CV_64F && src.channels() <= 4 );
        CV_Assert( dst.depth() <= CV_64F );

        if (src.depth() == CV_64F || dst.depth() == CV_64F)
        {
            CV_Assert( deviceSupports(NATIVE_DOUBLE) );
        }

        typedef void (*func_t)(const GpuMat& src, GpuMat& dst, cudaStream_t stream);
        static const func_t funcs[7][7][4] =
        {
            {
                /*  8U ->  8U */ {0, 0, 0, 0},
                /*  8U ->  8S */ {cudaConvert                                       , cudaConvert, cudaConvert, cudaConvert                                       },
                /*  8U -> 16U */ {NppCvt<CV_8U, CV_16U, nppiConvert_8u16u_C1R>::call, cudaConvert, cudaConvert, NppCvt<CV_8U, CV_16U, nppiConvert_8u16u_C4R>::call},
                /*  8U -> 16S */ {NppCvt<CV_8U, CV_16S, nppiConvert_8u16s_C1R>::call, cudaConvert, cudaConvert, NppCvt<CV_8U, CV_16S, nppiConvert_8u16s_C4R>::call},
                /*  8U -> 32S */ {cudaConvert                                       , cudaConvert, cudaConvert, cudaConvert                                       },
                /*  8U -> 32F */ {NppCvt<CV_8U, CV_32F, nppiConvert_8u32f_C1R>::call, cudaConvert, cudaConvert, cudaConvert                                       },
                /*  8U -> 64F */ {cudaConvert                                       , cudaConvert, cudaConvert, cudaConvert                                       }
            },
            {
                /*  8S ->  8U */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /*  8S ->  8S */ {0,0,0,0},
                /*  8S -> 16U */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /*  8S -> 16S */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /*  8S -> 32S */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /*  8S -> 32F */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /*  8S -> 64F */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert}
            },
            {
                /* 16U ->  8U */ {NppCvt<CV_16U, CV_8U , nppiConvert_16u8u_C1R >::call, cudaConvert, cudaConvert, NppCvt<CV_16U, CV_8U, nppiConvert_16u8u_C4R>::call},
                /* 16U ->  8S */ {cudaConvert                                         , cudaConvert, cudaConvert, cudaConvert                                       },
                /* 16U -> 16U */ {0,0,0,0},
                /* 16U -> 16S */ {cudaConvert                                         , cudaConvert, cudaConvert, cudaConvert                                       },
                /* 16U -> 32S */ {NppCvt<CV_16U, CV_32S, nppiConvert_16u32s_C1R>::call, cudaConvert, cudaConvert, cudaConvert                                       },
                /* 16U -> 32F */ {NppCvt<CV_16U, CV_32F, nppiConvert_16u32f_C1R>::call, cudaConvert, cudaConvert, cudaConvert                                       },
                /* 16U -> 64F */ {cudaConvert                                         , cudaConvert, cudaConvert, cudaConvert                                       }
            },
            {
                /* 16S ->  8U */ {NppCvt<CV_16S, CV_8U , nppiConvert_16s8u_C1R >::call, cudaConvert, cudaConvert, NppCvt<CV_16S, CV_8U, nppiConvert_16s8u_C4R>::call},
                /* 16S ->  8S */ {cudaConvert                                         , cudaConvert, cudaConvert, cudaConvert                                       },
                /* 16S -> 16U */ {cudaConvert                                         , cudaConvert, cudaConvert, cudaConvert                                       },
                /* 16S -> 16S */ {0,0,0,0},
                /* 16S -> 32S */ {NppCvt<CV_16S, CV_32S, nppiConvert_16s32s_C1R>::call, cudaConvert, cudaConvert, cudaConvert                                       },
                /* 16S -> 32F */ {NppCvt<CV_16S, CV_32F, nppiConvert_16s32f_C1R>::call, cudaConvert, cudaConvert, cudaConvert                                       },
                /* 16S -> 64F */ {cudaConvert                                         , cudaConvert, cudaConvert, cudaConvert                                       }
            },
            {
                /* 32S ->  8U */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /* 32S ->  8S */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /* 32S -> 16U */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /* 32S -> 16S */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /* 32S -> 32S */ {0,0,0,0},
                /* 32S -> 32F */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /* 32S -> 64F */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert}
            },
            {
                /* 32F ->  8U */ {NppCvt<CV_32F, CV_8U , nppiConvert_32f8u_C1R >::call, cudaConvert, cudaConvert, cudaConvert},
                /* 32F ->  8S */ {cudaConvert                                         , cudaConvert, cudaConvert, cudaConvert},
                /* 32F -> 16U */ {NppCvt<CV_32F, CV_16U, nppiConvert_32f16u_C1R>::call, cudaConvert, cudaConvert, cudaConvert},
                /* 32F -> 16S */ {NppCvt<CV_32F, CV_16S, nppiConvert_32f16s_C1R>::call, cudaConvert, cudaConvert, cudaConvert},
                /* 32F -> 32S */ {cudaConvert                                         , cudaConvert, cudaConvert, cudaConvert},
                /* 32F -> 32F */ {0,0,0,0},
                /* 32F -> 64F */ {cudaConvert                                         , cudaConvert, cudaConvert, cudaConvert}
            },
            {
                /* 64F ->  8U */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /* 64F ->  8S */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /* 64F -> 16U */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /* 64F -> 16S */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /* 64F -> 32S */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /* 64F -> 32F */ {cudaConvert, cudaConvert, cudaConvert, cudaConvert},
                /* 64F -> 64F */ {0,0,0,0}
            }
        };

        const bool aligned = isAligned(src.data, 16) && isAligned(dst.data, 16);
        if (!aligned)
        {
            cudaConvert(src, dst, stream);
            return;
        }

        const func_t func = funcs[src.depth()][dst.depth()][src.channels() - 1];
        CV_DbgAssert( func != 0 );

        func(src, dst, stream);
    }

    void convert(const GpuMat& src, GpuMat& dst, double alpha, double beta, cudaStream_t stream = 0)
    {
        CV_DbgAssert( src.size() == dst.size() && src.channels() == dst.channels() );

        CV_Assert( src.depth() <= CV_64F && src.channels() <= 4 );
        CV_Assert( dst.depth() <= CV_64F );

        if (src.depth() == CV_64F || dst.depth() == CV_64F)
        {
            CV_Assert( deviceSupports(NATIVE_DOUBLE) );
        }

        cudaConvert(src, dst, alpha, beta, stream);
    }

    void set(GpuMat& m, Scalar s, cudaStream_t stream = 0)
    {
        if (s[0] == 0.0 && s[1] == 0.0 && s[2] == 0.0 && s[3] == 0.0)
        {
            if (stream)
                cudaSafeCall( cudaMemset2DAsync(m.data, m.step, 0, m.cols * m.elemSize(), m.rows, stream) );
            else
                cudaSafeCall( cudaMemset2D(m.data, m.step, 0, m.cols * m.elemSize(), m.rows) );
            return;
        }

        if (m.depth() == CV_8U)
        {
            int cn = m.channels();

            if (cn == 1 || (cn == 2 && s[0] == s[1]) || (cn == 3 && s[0] == s[1] && s[0] == s[2]) || (cn == 4 && s[0] == s[1] && s[0] == s[2] && s[0] == s[3]))
            {
                int val = saturate_cast<uchar>(s[0]);
                if (stream)
                    cudaSafeCall( cudaMemset2DAsync(m.data, m.step, val, m.cols * m.elemSize(), m.rows, stream) );
                else
                    cudaSafeCall( cudaMemset2D(m.data, m.step, val, m.cols * m.elemSize(), m.rows) );
                return;
            }
        }

        typedef void (*func_t)(GpuMat& src, Scalar s, cudaStream_t stream);
        static const func_t funcs[7][4] =
        {
            {NppSet<CV_8U , 1, nppiSet_8u_C1R >::call, cudaSet                                 , cudaSet                               , NppSet<CV_8U , 4, nppiSet_8u_C4R >::call},
            {NppSet<CV_8S , 1, nppiSet_8s_C1R >::call, NppSet<CV_8S , 2, nppiSet_8s_C2R >::call, NppSet<CV_8S, 3, nppiSet_8s_C3R>::call, NppSet<CV_8S , 4, nppiSet_8s_C4R >::call},
            {NppSet<CV_16U, 1, nppiSet_16u_C1R>::call, NppSet<CV_16U, 2, nppiSet_16u_C2R>::call, cudaSet                               , NppSet<CV_16U, 4, nppiSet_16u_C4R>::call},
            {NppSet<CV_16S, 1, nppiSet_16s_C1R>::call, NppSet<CV_16S, 2, nppiSet_16s_C2R>::call, cudaSet                               , NppSet<CV_16S, 4, nppiSet_16s_C4R>::call},
            {NppSet<CV_32S, 1, nppiSet_32s_C1R>::call, cudaSet                                 , cudaSet                               , NppSet<CV_32S, 4, nppiSet_32s_C4R>::call},
            {NppSet<CV_32F, 1, nppiSet_32f_C1R>::call, cudaSet                                 , cudaSet                               , NppSet<CV_32F, 4, nppiSet_32f_C4R>::call},
            {cudaSet                                 , cudaSet                                 , cudaSet                               , cudaSet                                 }
        };

        CV_Assert( m.depth() <= CV_64F && m.channels() <= 4 );

        if (m.depth() == CV_64F)
        {
            CV_Assert( deviceSupports(NATIVE_DOUBLE) );
        }

        funcs[m.depth()][m.channels() - 1](m, s, stream);
    }

    void set(GpuMat& m, Scalar s, const GpuMat& mask, cudaStream_t stream = 0)
    {
        CV_DbgAssert( !mask.empty() );

        CV_Assert( m.depth() <= CV_64F && m.channels() <= 4 );

        if (m.depth() == CV_64F)
        {
            CV_Assert( deviceSupports(NATIVE_DOUBLE) );
        }

        typedef void (*func_t)(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream);
        static const func_t funcs[7][4] =
        {
            {NppSetMask<CV_8U , 1, nppiSet_8u_C1MR >::call, cudaSet, cudaSet, NppSetMask<CV_8U , 4, nppiSet_8u_C4MR >::call},
            {cudaSet                                      , cudaSet, cudaSet, cudaSet                                      },
            {NppSetMask<CV_16U, 1, nppiSet_16u_C1MR>::call, cudaSet, cudaSet, NppSetMask<CV_16U, 4, nppiSet_16u_C4MR>::call},
            {NppSetMask<CV_16S, 1, nppiSet_16s_C1MR>::call, cudaSet, cudaSet, NppSetMask<CV_16S, 4, nppiSet_16s_C4MR>::call},
            {NppSetMask<CV_32S, 1, nppiSet_32s_C1MR>::call, cudaSet, cudaSet, NppSetMask<CV_32S, 4, nppiSet_32s_C4MR>::call},
            {NppSetMask<CV_32F, 1, nppiSet_32f_C1MR>::call, cudaSet, cudaSet, NppSetMask<CV_32F, 4, nppiSet_32f_C4MR>::call},
            {cudaSet                                      , cudaSet, cudaSet, cudaSet                                      }
        };

        funcs[m.depth()][m.channels() - 1](m, s, mask, stream);
    }
}

#endif // HAVE_CUDA

cv::gpu::GpuMat::GpuMat(int rows_, int cols_, int type_, void* data_, size_t step_) :
    flags(Mat::MAGIC_VAL + (type_ & Mat::TYPE_MASK)), rows(rows_), cols(cols_),
    step(step_), data((uchar*)data_), refcount(0),
    datastart((uchar*)data_), dataend((uchar*)data_)
{
    size_t minstep = cols * elemSize();

    if (step == Mat::AUTO_STEP)
    {
        step = minstep;
        flags |= Mat::CONTINUOUS_FLAG;
    }
    else
    {
        if (rows == 1)
            step = minstep;

        CV_DbgAssert( step >= minstep );

        flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
    }

    dataend += step * (rows - 1) + minstep;
}

cv::gpu::GpuMat::GpuMat(Size size_, int type_, void* data_, size_t step_) :
    flags(Mat::MAGIC_VAL + (type_ & Mat::TYPE_MASK)), rows(size_.height), cols(size_.width),
    step(step_), data((uchar*)data_), refcount(0),
    datastart((uchar*)data_), dataend((uchar*)data_)
{
    size_t minstep = cols * elemSize();

    if (step == Mat::AUTO_STEP)
    {
        step = minstep;
        flags |= Mat::CONTINUOUS_FLAG;
    }
    else
    {
        if (rows == 1)
            step = minstep;

        CV_DbgAssert( step >= minstep );

        flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
    }
    dataend += step * (rows - 1) + minstep;
}

cv::gpu::GpuMat::GpuMat(const GpuMat& m, Range rowRange_, Range colRange_)
{
    flags = m.flags;
    step = m.step; refcount = m.refcount;
    data = m.data; datastart = m.datastart; dataend = m.dataend;

    if (rowRange_ == Range::all())
    {
        rows = m.rows;
    }
    else
    {
        CV_Assert( 0 <= rowRange_.start && rowRange_.start <= rowRange_.end && rowRange_.end <= m.rows );

        rows = rowRange_.size();
        data += step*rowRange_.start;
    }

    if (colRange_ == Range::all())
    {
        cols = m.cols;
    }
    else
    {
        CV_Assert( 0 <= colRange_.start && colRange_.start <= colRange_.end && colRange_.end <= m.cols );

        cols = colRange_.size();
        data += colRange_.start*elemSize();
        flags &= cols < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
    }

    if (rows == 1)
        flags |= Mat::CONTINUOUS_FLAG;

    if (refcount)
        CV_XADD(refcount, 1);

    if (rows <= 0 || cols <= 0)
        rows = cols = 0;
}

cv::gpu::GpuMat::GpuMat(const GpuMat& m, Rect roi) :
    flags(m.flags), rows(roi.height), cols(roi.width),
    step(m.step), data(m.data + roi.y*step), refcount(m.refcount),
    datastart(m.datastart), dataend(m.dataend)
{
    flags &= roi.width < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
    data += roi.x * elemSize();

    CV_Assert( 0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows );

    if (refcount)
        CV_XADD(refcount, 1);

    if (rows <= 0 || cols <= 0)
        rows = cols = 0;
}

void cv::gpu::GpuMat::create(int _rows, int _cols, int _type)
{
#ifndef HAVE_CUDA
    (void) _rows;
    (void) _cols;
    (void) _type;
    throw_no_cuda();
#else
    _type &= Mat::TYPE_MASK;

    if (rows == _rows && cols == _cols && type() == _type && data)
        return;

    if (data)
        release();

    CV_DbgAssert( _rows >= 0 && _cols >= 0 );

    if (_rows > 0 && _cols > 0)
    {
        flags = Mat::MAGIC_VAL + _type;
        rows = _rows;
        cols = _cols;

        size_t esz = elemSize();

        void* devPtr;

        if (rows > 1 && cols > 1)
        {
            cudaSafeCall( cudaMallocPitch(&devPtr, &step, esz * cols, rows) );
        }
        else
        {
            // Single row or single column must be continuous
            cudaSafeCall( cudaMalloc(&devPtr, esz * cols * rows) );
            step = esz * cols;
        }

        if (esz * cols == step)
            flags |= Mat::CONTINUOUS_FLAG;

        int64 _nettosize = static_cast<int64>(step) * rows;
        size_t nettosize = static_cast<size_t>(_nettosize);

        datastart = data = static_cast<uchar*>(devPtr);
        dataend = data + nettosize;

        refcount = static_cast<int*>(fastMalloc(sizeof(*refcount)));
        *refcount = 1;
    }
#endif
}

void cv::gpu::GpuMat::release()
{
#ifdef HAVE_CUDA
    if (refcount && CV_XADD(refcount, -1) == 1)
    {
        cudaFree(datastart);
        fastFree(refcount);
    }

    data = datastart = dataend = 0;
    step = rows = cols = 0;
    refcount = 0;
#endif
}

void cv::gpu::GpuMat::upload(InputArray arr)
{
#ifndef HAVE_CUDA
    (void) arr;
    throw_no_cuda();
#else
    Mat mat = arr.getMat();

    CV_DbgAssert( !mat.empty() );

    create(mat.size(), mat.type());

    cudaSafeCall( cudaMemcpy2D(data, step, mat.data, mat.step, cols * elemSize(), rows, cudaMemcpyHostToDevice) );
#endif
}

void cv::gpu::GpuMat::upload(InputArray arr, Stream& _stream)
{
#ifndef HAVE_CUDA
    (void) arr;
    (void) _stream;
    throw_no_cuda();
#else
    Mat mat = arr.getMat();

    CV_DbgAssert( !mat.empty() );

    create(mat.size(), mat.type());

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    cudaSafeCall( cudaMemcpy2DAsync(data, step, mat.data, mat.step, cols * elemSize(), rows, cudaMemcpyHostToDevice, stream) );
#endif
}

void cv::gpu::GpuMat::download(OutputArray _dst) const
{
#ifndef HAVE_CUDA
    (void) _dst;
    throw_no_cuda();
#else
    CV_DbgAssert( !empty() );

    _dst.create(size(), type());
    Mat dst = _dst.getMat();

    cudaSafeCall( cudaMemcpy2D(dst.data, dst.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToHost) );
#endif
}

void cv::gpu::GpuMat::download(OutputArray _dst, Stream& _stream) const
{
#ifndef HAVE_CUDA
    (void) _dst;
    (void) _stream;
    throw_no_cuda();
#else
    CV_DbgAssert( !empty() );

    _dst.create(size(), type());
    Mat dst = _dst.getMat();

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    cudaSafeCall( cudaMemcpy2DAsync(dst.data, dst.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToHost, stream) );
#endif
}

void cv::gpu::GpuMat::copyTo(OutputArray _dst) const
{
#ifndef HAVE_CUDA
    (void) _dst;
    throw_no_cuda();
#else
    CV_DbgAssert( !empty() );

    _dst.create(size(), type());
    GpuMat dst = _dst.getGpuMat();

    cudaSafeCall( cudaMemcpy2D(dst.data, dst.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToDevice) );
#endif
}

void cv::gpu::GpuMat::copyTo(OutputArray _dst, Stream& _stream) const
{
#ifndef HAVE_CUDA
    (void) _dst;
    (void) _stream;
    throw_no_cuda();
#else
    CV_DbgAssert( !empty() );

    _dst.create(size(), type());
    GpuMat dst = _dst.getGpuMat();

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    cudaSafeCall( cudaMemcpy2DAsync(dst.data, dst.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToDevice, stream) );
#endif
}

void cv::gpu::GpuMat::copyTo(OutputArray _dst, InputArray _mask, Stream& _stream) const
{
#ifndef HAVE_CUDA
    (void) _dst;
    (void) _mask;
    (void) _stream;
    throw_no_cuda();
#else
    CV_DbgAssert( !empty() );

    _dst.create(size(), type());
    GpuMat dst = _dst.getGpuMat();

    GpuMat mask = _mask.getGpuMat();

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    ::copyWithMask(*this, dst, mask, stream);
#endif
}

GpuMat& cv::gpu::GpuMat::setTo(Scalar s, Stream& _stream)
{
#ifndef HAVE_CUDA
    (void) s;
    (void) _stream;
    throw_no_cuda();
#else
    CV_DbgAssert( !empty() );

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    ::set(*this, s, stream);
#endif

    return *this;
}

GpuMat& cv::gpu::GpuMat::setTo(Scalar s, InputArray _mask, Stream& _stream)
{
#ifndef HAVE_CUDA
    (void) s;
    (void) _mask;
    (void) _stream;
    throw_no_cuda();
#else
    CV_DbgAssert( !empty() );

    GpuMat mask = _mask.getGpuMat();

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    ::set(*this, s, mask, stream);
#endif

    return *this;
}

void cv::gpu::GpuMat::convertTo(OutputArray _dst, int rtype, Stream& _stream) const
{
#ifndef HAVE_CUDA
    (void) _dst;
    (void) rtype;
    (void) _stream;
    throw_no_cuda();
#else
    if (rtype < 0)
        rtype = type();
    else
        rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());

    const int sdepth = depth();
    const int ddepth = CV_MAT_DEPTH(rtype);
    if (sdepth == ddepth)
    {
        if (_stream)
            copyTo(_dst, _stream);
        else
            copyTo(_dst);

        return;
    }

    GpuMat src = *this;

    _dst.create(size(), rtype);
    GpuMat dst = _dst.getGpuMat();

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    ::convert(src, dst, stream);
#endif
}

void cv::gpu::GpuMat::convertTo(OutputArray _dst, int rtype, double alpha, double beta, Stream& _stream) const
{
#ifndef HAVE_CUDA
    (void) _dst;
    (void) rtype;
    (void) alpha;
    (void) beta;
    (void) _stream;
    throw_no_cuda();
#else
    if (rtype < 0)
        rtype = type();
    else
        rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());

    GpuMat src = *this;

    _dst.create(size(), rtype);
    GpuMat dst = _dst.getGpuMat();

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    ::convert(src, dst, alpha, beta, stream);
#endif
}

GpuMat cv::gpu::GpuMat::reshape(int new_cn, int new_rows) const
{
    GpuMat hdr = *this;

    int cn = channels();
    if (new_cn == 0)
        new_cn = cn;

    int total_width = cols * cn;

    if ((new_cn > total_width || total_width % new_cn != 0) && new_rows == 0)
        new_rows = rows * total_width / new_cn;

    if (new_rows != 0 && new_rows != rows)
    {
        int total_size = total_width * rows;

        if (!isContinuous())
            CV_Error(cv::Error::BadStep, "The matrix is not continuous, thus its number of rows can not be changed");

        if ((unsigned)new_rows > (unsigned)total_size)
            CV_Error(cv::Error::StsOutOfRange, "Bad new number of rows");

        total_width = total_size / new_rows;

        if (total_width * new_rows != total_size)
            CV_Error(cv::Error::StsBadArg, "The total number of matrix elements is not divisible by the new number of rows");

        hdr.rows = new_rows;
        hdr.step = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;

    if (new_width * new_cn != total_width)
        CV_Error(cv::Error::BadNumChannels, "The total width is not divisible by the new number of channels");

    hdr.cols = new_width;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn - 1) << CV_CN_SHIFT);

    return hdr;
}

void cv::gpu::GpuMat::locateROI(Size& wholeSize, Point& ofs) const
{
    CV_DbgAssert( step > 0 );

    size_t esz = elemSize();
    ptrdiff_t delta1 = data - datastart;
    ptrdiff_t delta2 = dataend - datastart;

    if (delta1 == 0)
    {
        ofs.x = ofs.y = 0;
    }
    else
    {
        ofs.y = static_cast<int>(delta1 / step);
        ofs.x = static_cast<int>((delta1 - step * ofs.y) / esz);

        CV_DbgAssert( data == datastart + ofs.y * step + ofs.x * esz );
    }

    size_t minstep = (ofs.x + cols) * esz;

    wholeSize.height = std::max(static_cast<int>((delta2 - minstep) / step + 1), ofs.y + rows);
    wholeSize.width = std::max(static_cast<int>((delta2 - step * (wholeSize.height - 1)) / esz), ofs.x + cols);
}

GpuMat& cv::gpu::GpuMat::adjustROI(int dtop, int dbottom, int dleft, int dright)
{
    Size wholeSize;
    Point ofs;
    locateROI(wholeSize, ofs);

    size_t esz = elemSize();

    int row1 = std::max(ofs.y - dtop, 0);
    int row2 = std::min(ofs.y + rows + dbottom, wholeSize.height);

    int col1 = std::max(ofs.x - dleft, 0);
    int col2 = std::min(ofs.x + cols + dright, wholeSize.width);

    data += (row1 - ofs.y) * step + (col1 - ofs.x) * esz;
    rows = row2 - row1;
    cols = col2 - col1;

    if (esz * cols == step || rows == 1)
        flags |= Mat::CONTINUOUS_FLAG;
    else
        flags &= ~Mat::CONTINUOUS_FLAG;

    return *this;
}

namespace
{
    template <class ObjType>
    void createContinuousImpl(int rows, int cols, int type, ObjType& obj)
    {
        const int area = rows * cols;

        if (obj.empty() || obj.type() != type || !obj.isContinuous() || obj.size().area() < area)
            obj.create(1, area, type);

        obj = obj.reshape(obj.channels(), rows);
    }
}

void cv::gpu::createContinuous(int rows, int cols, int type, OutputArray arr)
{
    switch (arr.kind())
    {
    case _InputArray::MAT:
        ::createContinuousImpl(rows, cols, type, arr.getMatRef());
        break;

    case _InputArray::GPU_MAT:
        ::createContinuousImpl(rows, cols, type, arr.getGpuMatRef());
        break;

    case _InputArray::CUDA_MEM:
        ::createContinuousImpl(rows, cols, type, arr.getCudaMemRef());
        break;

    default:
        arr.create(rows, cols, type);
    }
}

namespace
{
    template <class ObjType>
    void ensureSizeIsEnoughImpl(int rows, int cols, int type, ObjType& obj)
    {
        if (obj.empty() || obj.type() != type || obj.data != obj.datastart)
        {
            obj.create(rows, cols, type);
        }
        else
        {
            const size_t esz = obj.elemSize();
            const ptrdiff_t delta2 = obj.dataend - obj.datastart;

            const size_t minstep = obj.cols * esz;

            Size wholeSize;
            wholeSize.height = std::max(static_cast<int>((delta2 - minstep) / static_cast<size_t>(obj.step) + 1), obj.rows);
            wholeSize.width = std::max(static_cast<int>((delta2 - static_cast<size_t>(obj.step) * (wholeSize.height - 1)) / esz), obj.cols);

            if (wholeSize.height < rows || wholeSize.width < cols)
            {
                obj.create(rows, cols, type);
            }
            else
            {
                obj.cols = cols;
                obj.rows = rows;
            }
        }
    }
}

void cv::gpu::ensureSizeIsEnough(int rows, int cols, int type, OutputArray arr)
{
    switch (arr.kind())
    {
    case _InputArray::MAT:
        ::ensureSizeIsEnoughImpl(rows, cols, type, arr.getMatRef());
        break;

    case _InputArray::GPU_MAT:
        ::ensureSizeIsEnoughImpl(rows, cols, type, arr.getGpuMatRef());
        break;

    case _InputArray::CUDA_MEM:
        ::ensureSizeIsEnoughImpl(rows, cols, type, arr.getCudaMemRef());
        break;

    default:
        arr.create(rows, cols, type);
    }
}

GpuMat cv::gpu::allocMatFromBuf(int rows, int cols, int type, GpuMat& mat)
{
    if (!mat.empty() && mat.type() == type && mat.rows >= rows && mat.cols >= cols)
        return mat(Rect(0, 0, cols, rows));

    return mat = GpuMat(rows, cols, type);
}
