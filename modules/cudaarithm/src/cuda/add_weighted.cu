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

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudev.hpp"

using namespace cv::cudev;

namespace
{
    template <typename T1, typename T2, typename D, typename S> struct AddWeightedOp : binary_function<T1, T2, D>
    {
        S alpha;
        S beta;
        S gamma;

        __device__ __forceinline__ D operator ()(T1 a, T2 b) const
        {
            return saturate_cast<D>(a * alpha + b * beta + gamma);
        }
    };

    template <typename ScalarDepth> struct TransformPolicy : DefaultTransformPolicy
    {
    };
    template <> struct TransformPolicy<double> : DefaultTransformPolicy
    {
        enum {
            shift = 1
        };
    };

    template <typename T1, typename T2, typename D>
    void addWeightedImpl(const GpuMat& src1, double alpha, const GpuMat& src2, double beta, double gamma, GpuMat& dst, Stream& stream)
    {
        typedef typename LargerType<T1, T2>::type larger_type1;
        typedef typename LargerType<larger_type1, D>::type larger_type2;
        typedef typename LargerType<larger_type2, float>::type scalar_type;

        AddWeightedOp<T1, T2, D, scalar_type> op;
        op.alpha = static_cast<scalar_type>(alpha);
        op.beta = static_cast<scalar_type>(beta);
        op.gamma = static_cast<scalar_type>(gamma);

        gridTransformBinary_< TransformPolicy<scalar_type> >(globPtr<T1>(src1), globPtr<T2>(src2), globPtr<D>(dst), op, stream);
    }
}

void cv::cuda::addWeighted(InputArray _src1, double alpha, InputArray _src2, double beta, double gamma, OutputArray _dst, int ddepth, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src1, double alpha, const GpuMat& src2, double beta, double gamma, GpuMat& dst, Stream& stream);
    static const func_t funcs[7][7][7] =
    {
        {
            {
                addWeightedImpl<uchar, uchar, uchar >,
                addWeightedImpl<uchar, uchar, schar >,
                addWeightedImpl<uchar, uchar, ushort>,
                addWeightedImpl<uchar, uchar, short >,
                addWeightedImpl<uchar, uchar, int   >,
                addWeightedImpl<uchar, uchar, float >,
                addWeightedImpl<uchar, uchar, double>
            },
            {
                addWeightedImpl<uchar, schar, uchar >,
                addWeightedImpl<uchar, schar, schar >,
                addWeightedImpl<uchar, schar, ushort>,
                addWeightedImpl<uchar, schar, short >,
                addWeightedImpl<uchar, schar, int   >,
                addWeightedImpl<uchar, schar, float >,
                addWeightedImpl<uchar, schar, double>
            },
            {
                addWeightedImpl<uchar, ushort, uchar >,
                addWeightedImpl<uchar, ushort, schar >,
                addWeightedImpl<uchar, ushort, ushort>,
                addWeightedImpl<uchar, ushort, short >,
                addWeightedImpl<uchar, ushort, int   >,
                addWeightedImpl<uchar, ushort, float >,
                addWeightedImpl<uchar, ushort, double>
            },
            {
                addWeightedImpl<uchar, short, uchar >,
                addWeightedImpl<uchar, short, schar >,
                addWeightedImpl<uchar, short, ushort>,
                addWeightedImpl<uchar, short, short >,
                addWeightedImpl<uchar, short, int   >,
                addWeightedImpl<uchar, short, float >,
                addWeightedImpl<uchar, short, double>
            },
            {
                addWeightedImpl<uchar, int, uchar >,
                addWeightedImpl<uchar, int, schar >,
                addWeightedImpl<uchar, int, ushort>,
                addWeightedImpl<uchar, int, short >,
                addWeightedImpl<uchar, int, int   >,
                addWeightedImpl<uchar, int, float >,
                addWeightedImpl<uchar, int, double>
            },
            {
                addWeightedImpl<uchar, float, uchar >,
                addWeightedImpl<uchar, float, schar >,
                addWeightedImpl<uchar, float, ushort>,
                addWeightedImpl<uchar, float, short >,
                addWeightedImpl<uchar, float, int   >,
                addWeightedImpl<uchar, float, float >,
                addWeightedImpl<uchar, float, double>
            },
            {
                addWeightedImpl<uchar, double, uchar >,
                addWeightedImpl<uchar, double, schar >,
                addWeightedImpl<uchar, double, ushort>,
                addWeightedImpl<uchar, double, short >,
                addWeightedImpl<uchar, double, int   >,
                addWeightedImpl<uchar, double, float >,
                addWeightedImpl<uchar, double, double>
            }
        },
        {
            {
                0/*addWeightedImpl<schar, uchar, uchar >*/,
                0/*addWeightedImpl<schar, uchar, schar >*/,
                0/*addWeightedImpl<schar, uchar, ushort>*/,
                0/*addWeightedImpl<schar, uchar, short >*/,
                0/*addWeightedImpl<schar, uchar, int   >*/,
                0/*addWeightedImpl<schar, uchar, float >*/,
                0/*addWeightedImpl<schar, uchar, double>*/
            },
            {
                addWeightedImpl<schar, schar, uchar >,
                addWeightedImpl<schar, schar, schar >,
                addWeightedImpl<schar, schar, ushort>,
                addWeightedImpl<schar, schar, short >,
                addWeightedImpl<schar, schar, int   >,
                addWeightedImpl<schar, schar, float >,
                addWeightedImpl<schar, schar, double>
            },
            {
                addWeightedImpl<schar, ushort, uchar >,
                addWeightedImpl<schar, ushort, schar >,
                addWeightedImpl<schar, ushort, ushort>,
                addWeightedImpl<schar, ushort, short >,
                addWeightedImpl<schar, ushort, int   >,
                addWeightedImpl<schar, ushort, float >,
                addWeightedImpl<schar, ushort, double>
            },
            {
                addWeightedImpl<schar, short, uchar >,
                addWeightedImpl<schar, short, schar >,
                addWeightedImpl<schar, short, ushort>,
                addWeightedImpl<schar, short, short >,
                addWeightedImpl<schar, short, int   >,
                addWeightedImpl<schar, short, float >,
                addWeightedImpl<schar, short, double>
            },
            {
                addWeightedImpl<schar, int, uchar >,
                addWeightedImpl<schar, int, schar >,
                addWeightedImpl<schar, int, ushort>,
                addWeightedImpl<schar, int, short >,
                addWeightedImpl<schar, int, int   >,
                addWeightedImpl<schar, int, float >,
                addWeightedImpl<schar, int, double>
            },
            {
                addWeightedImpl<schar, float, uchar >,
                addWeightedImpl<schar, float, schar >,
                addWeightedImpl<schar, float, ushort>,
                addWeightedImpl<schar, float, short >,
                addWeightedImpl<schar, float, int   >,
                addWeightedImpl<schar, float, float >,
                addWeightedImpl<schar, float, double>
            },
            {
                addWeightedImpl<schar, double, uchar >,
                addWeightedImpl<schar, double, schar >,
                addWeightedImpl<schar, double, ushort>,
                addWeightedImpl<schar, double, short >,
                addWeightedImpl<schar, double, int   >,
                addWeightedImpl<schar, double, float >,
                addWeightedImpl<schar, double, double>
            }
        },
        {
            {
                0/*addWeightedImpl<ushort, uchar, uchar >*/,
                0/*addWeightedImpl<ushort, uchar, schar >*/,
                0/*addWeightedImpl<ushort, uchar, ushort>*/,
                0/*addWeightedImpl<ushort, uchar, short >*/,
                0/*addWeightedImpl<ushort, uchar, int   >*/,
                0/*addWeightedImpl<ushort, uchar, float >*/,
                0/*addWeightedImpl<ushort, uchar, double>*/
            },
            {
                0/*addWeightedImpl<ushort, schar, uchar >*/,
                0/*addWeightedImpl<ushort, schar, schar >*/,
                0/*addWeightedImpl<ushort, schar, ushort>*/,
                0/*addWeightedImpl<ushort, schar, short >*/,
                0/*addWeightedImpl<ushort, schar, int   >*/,
                0/*addWeightedImpl<ushort, schar, float >*/,
                0/*addWeightedImpl<ushort, schar, double>*/
            },
            {
                addWeightedImpl<ushort, ushort, uchar >,
                addWeightedImpl<ushort, ushort, schar >,
                addWeightedImpl<ushort, ushort, ushort>,
                addWeightedImpl<ushort, ushort, short >,
                addWeightedImpl<ushort, ushort, int   >,
                addWeightedImpl<ushort, ushort, float >,
                addWeightedImpl<ushort, ushort, double>
            },
            {
                addWeightedImpl<ushort, short, uchar >,
                addWeightedImpl<ushort, short, schar >,
                addWeightedImpl<ushort, short, ushort>,
                addWeightedImpl<ushort, short, short >,
                addWeightedImpl<ushort, short, int   >,
                addWeightedImpl<ushort, short, float >,
                addWeightedImpl<ushort, short, double>
            },
            {
                addWeightedImpl<ushort, int, uchar >,
                addWeightedImpl<ushort, int, schar >,
                addWeightedImpl<ushort, int, ushort>,
                addWeightedImpl<ushort, int, short >,
                addWeightedImpl<ushort, int, int   >,
                addWeightedImpl<ushort, int, float >,
                addWeightedImpl<ushort, int, double>
            },
            {
                addWeightedImpl<ushort, float, uchar >,
                addWeightedImpl<ushort, float, schar >,
                addWeightedImpl<ushort, float, ushort>,
                addWeightedImpl<ushort, float, short >,
                addWeightedImpl<ushort, float, int   >,
                addWeightedImpl<ushort, float, float >,
                addWeightedImpl<ushort, float, double>
            },
            {
                addWeightedImpl<ushort, double, uchar >,
                addWeightedImpl<ushort, double, schar >,
                addWeightedImpl<ushort, double, ushort>,
                addWeightedImpl<ushort, double, short >,
                addWeightedImpl<ushort, double, int   >,
                addWeightedImpl<ushort, double, float >,
                addWeightedImpl<ushort, double, double>
            }
        },
        {
            {
                0/*addWeightedImpl<short, uchar, uchar >*/,
                0/*addWeightedImpl<short, uchar, schar >*/,
                0/*addWeightedImpl<short, uchar, ushort>*/,
                0/*addWeightedImpl<short, uchar, short >*/,
                0/*addWeightedImpl<short, uchar, int   >*/,
                0/*addWeightedImpl<short, uchar, float >*/,
                0/*addWeightedImpl<short, uchar, double>*/
            },
            {
                0/*addWeightedImpl<short, schar, uchar >*/,
                0/*addWeightedImpl<short, schar, schar >*/,
                0/*addWeightedImpl<short, schar, ushort>*/,
                0/*addWeightedImpl<short, schar, short >*/,
                0/*addWeightedImpl<short, schar, int   >*/,
                0/*addWeightedImpl<short, schar, float >*/,
                0/*addWeightedImpl<short, schar, double>*/
            },
            {
                0/*addWeightedImpl<short, ushort, uchar >*/,
                0/*addWeightedImpl<short, ushort, schar >*/,
                0/*addWeightedImpl<short, ushort, ushort>*/,
                0/*addWeightedImpl<short, ushort, short >*/,
                0/*addWeightedImpl<short, ushort, int   >*/,
                0/*addWeightedImpl<short, ushort, float >*/,
                0/*addWeightedImpl<short, ushort, double>*/
            },
            {
                addWeightedImpl<short, short, uchar >,
                addWeightedImpl<short, short, schar >,
                addWeightedImpl<short, short, ushort>,
                addWeightedImpl<short, short, short >,
                addWeightedImpl<short, short, int   >,
                addWeightedImpl<short, short, float >,
                addWeightedImpl<short, short, double>
            },
            {
                addWeightedImpl<short, int, uchar >,
                addWeightedImpl<short, int, schar >,
                addWeightedImpl<short, int, ushort>,
                addWeightedImpl<short, int, short >,
                addWeightedImpl<short, int, int   >,
                addWeightedImpl<short, int, float >,
                addWeightedImpl<short, int, double>
            },
            {
                addWeightedImpl<short, float, uchar >,
                addWeightedImpl<short, float, schar >,
                addWeightedImpl<short, float, ushort>,
                addWeightedImpl<short, float, short >,
                addWeightedImpl<short, float, int   >,
                addWeightedImpl<short, float, float >,
                addWeightedImpl<short, float, double>
            },
            {
                addWeightedImpl<short, double, uchar >,
                addWeightedImpl<short, double, schar >,
                addWeightedImpl<short, double, ushort>,
                addWeightedImpl<short, double, short >,
                addWeightedImpl<short, double, int   >,
                addWeightedImpl<short, double, float >,
                addWeightedImpl<short, double, double>
            }
        },
        {
            {
                0/*addWeightedImpl<int, uchar, uchar >*/,
                0/*addWeightedImpl<int, uchar, schar >*/,
                0/*addWeightedImpl<int, uchar, ushort>*/,
                0/*addWeightedImpl<int, uchar, short >*/,
                0/*addWeightedImpl<int, uchar, int   >*/,
                0/*addWeightedImpl<int, uchar, float >*/,
                0/*addWeightedImpl<int, uchar, double>*/
            },
            {
                0/*addWeightedImpl<int, schar, uchar >*/,
                0/*addWeightedImpl<int, schar, schar >*/,
                0/*addWeightedImpl<int, schar, ushort>*/,
                0/*addWeightedImpl<int, schar, short >*/,
                0/*addWeightedImpl<int, schar, int   >*/,
                0/*addWeightedImpl<int, schar, float >*/,
                0/*addWeightedImpl<int, schar, double>*/
            },
            {
                0/*addWeightedImpl<int, ushort, uchar >*/,
                0/*addWeightedImpl<int, ushort, schar >*/,
                0/*addWeightedImpl<int, ushort, ushort>*/,
                0/*addWeightedImpl<int, ushort, short >*/,
                0/*addWeightedImpl<int, ushort, int   >*/,
                0/*addWeightedImpl<int, ushort, float >*/,
                0/*addWeightedImpl<int, ushort, double>*/
            },
            {
                0/*addWeightedImpl<int, short, uchar >*/,
                0/*addWeightedImpl<int, short, schar >*/,
                0/*addWeightedImpl<int, short, ushort>*/,
                0/*addWeightedImpl<int, short, short >*/,
                0/*addWeightedImpl<int, short, int   >*/,
                0/*addWeightedImpl<int, short, float >*/,
                0/*addWeightedImpl<int, short, double>*/
            },
            {
                addWeightedImpl<int, int, uchar >,
                addWeightedImpl<int, int, schar >,
                addWeightedImpl<int, int, ushort>,
                addWeightedImpl<int, int, short >,
                addWeightedImpl<int, int, int   >,
                addWeightedImpl<int, int, float >,
                addWeightedImpl<int, int, double>
            },
            {
                addWeightedImpl<int, float, uchar >,
                addWeightedImpl<int, float, schar >,
                addWeightedImpl<int, float, ushort>,
                addWeightedImpl<int, float, short >,
                addWeightedImpl<int, float, int   >,
                addWeightedImpl<int, float, float >,
                addWeightedImpl<int, float, double>
            },
            {
                addWeightedImpl<int, double, uchar >,
                addWeightedImpl<int, double, schar >,
                addWeightedImpl<int, double, ushort>,
                addWeightedImpl<int, double, short >,
                addWeightedImpl<int, double, int   >,
                addWeightedImpl<int, double, float >,
                addWeightedImpl<int, double, double>
            }
        },
        {
            {
                0/*addWeightedImpl<float, uchar, uchar >*/,
                0/*addWeightedImpl<float, uchar, schar >*/,
                0/*addWeightedImpl<float, uchar, ushort>*/,
                0/*addWeightedImpl<float, uchar, short >*/,
                0/*addWeightedImpl<float, uchar, int   >*/,
                0/*addWeightedImpl<float, uchar, float >*/,
                0/*addWeightedImpl<float, uchar, double>*/
            },
            {
                0/*addWeightedImpl<float, schar, uchar >*/,
                0/*addWeightedImpl<float, schar, schar >*/,
                0/*addWeightedImpl<float, schar, ushort>*/,
                0/*addWeightedImpl<float, schar, short >*/,
                0/*addWeightedImpl<float, schar, int   >*/,
                0/*addWeightedImpl<float, schar, float >*/,
                0/*addWeightedImpl<float, schar, double>*/
            },
            {
                0/*addWeightedImpl<float, ushort, uchar >*/,
                0/*addWeightedImpl<float, ushort, schar >*/,
                0/*addWeightedImpl<float, ushort, ushort>*/,
                0/*addWeightedImpl<float, ushort, short >*/,
                0/*addWeightedImpl<float, ushort, int   >*/,
                0/*addWeightedImpl<float, ushort, float >*/,
                0/*addWeightedImpl<float, ushort, double>*/
            },
            {
                0/*addWeightedImpl<float, short, uchar >*/,
                0/*addWeightedImpl<float, short, schar >*/,
                0/*addWeightedImpl<float, short, ushort>*/,
                0/*addWeightedImpl<float, short, short >*/,
                0/*addWeightedImpl<float, short, int   >*/,
                0/*addWeightedImpl<float, short, float >*/,
                0/*addWeightedImpl<float, short, double>*/
            },
            {
                0/*addWeightedImpl<float, int, uchar >*/,
                0/*addWeightedImpl<float, int, schar >*/,
                0/*addWeightedImpl<float, int, ushort>*/,
                0/*addWeightedImpl<float, int, short >*/,
                0/*addWeightedImpl<float, int, int   >*/,
                0/*addWeightedImpl<float, int, float >*/,
                0/*addWeightedImpl<float, int, double>*/
            },
            {
                addWeightedImpl<float, float, uchar >,
                addWeightedImpl<float, float, schar >,
                addWeightedImpl<float, float, ushort>,
                addWeightedImpl<float, float, short >,
                addWeightedImpl<float, float, int   >,
                addWeightedImpl<float, float, float >,
                addWeightedImpl<float, float, double>
            },
            {
                addWeightedImpl<float, double, uchar >,
                addWeightedImpl<float, double, schar >,
                addWeightedImpl<float, double, ushort>,
                addWeightedImpl<float, double, short >,
                addWeightedImpl<float, double, int   >,
                addWeightedImpl<float, double, float >,
                addWeightedImpl<float, double, double>
            }
        },
        {
            {
                0/*addWeightedImpl<double, uchar, uchar >*/,
                0/*addWeightedImpl<double, uchar, schar >*/,
                0/*addWeightedImpl<double, uchar, ushort>*/,
                0/*addWeightedImpl<double, uchar, short >*/,
                0/*addWeightedImpl<double, uchar, int   >*/,
                0/*addWeightedImpl<double, uchar, float >*/,
                0/*addWeightedImpl<double, uchar, double>*/
            },
            {
                0/*addWeightedImpl<double, schar, uchar >*/,
                0/*addWeightedImpl<double, schar, schar >*/,
                0/*addWeightedImpl<double, schar, ushort>*/,
                0/*addWeightedImpl<double, schar, short >*/,
                0/*addWeightedImpl<double, schar, int   >*/,
                0/*addWeightedImpl<double, schar, float >*/,
                0/*addWeightedImpl<double, schar, double>*/
            },
            {
                0/*addWeightedImpl<double, ushort, uchar >*/,
                0/*addWeightedImpl<double, ushort, schar >*/,
                0/*addWeightedImpl<double, ushort, ushort>*/,
                0/*addWeightedImpl<double, ushort, short >*/,
                0/*addWeightedImpl<double, ushort, int   >*/,
                0/*addWeightedImpl<double, ushort, float >*/,
                0/*addWeightedImpl<double, ushort, double>*/
            },
            {
                0/*addWeightedImpl<double, short, uchar >*/,
                0/*addWeightedImpl<double, short, schar >*/,
                0/*addWeightedImpl<double, short, ushort>*/,
                0/*addWeightedImpl<double, short, short >*/,
                0/*addWeightedImpl<double, short, int   >*/,
                0/*addWeightedImpl<double, short, float >*/,
                0/*addWeightedImpl<double, short, double>*/
            },
            {
                0/*addWeightedImpl<double, int, uchar >*/,
                0/*addWeightedImpl<double, int, schar >*/,
                0/*addWeightedImpl<double, int, ushort>*/,
                0/*addWeightedImpl<double, int, short >*/,
                0/*addWeightedImpl<double, int, int   >*/,
                0/*addWeightedImpl<double, int, float >*/,
                0/*addWeightedImpl<double, int, double>*/
            },
            {
                0/*addWeightedImpl<double, float, uchar >*/,
                0/*addWeightedImpl<double, float, schar >*/,
                0/*addWeightedImpl<double, float, ushort>*/,
                0/*addWeightedImpl<double, float, short >*/,
                0/*addWeightedImpl<double, float, int   >*/,
                0/*addWeightedImpl<double, float, float >*/,
                0/*addWeightedImpl<double, float, double>*/
            },
            {
                addWeightedImpl<double, double, uchar >,
                addWeightedImpl<double, double, schar >,
                addWeightedImpl<double, double, ushort>,
                addWeightedImpl<double, double, short >,
                addWeightedImpl<double, double, int   >,
                addWeightedImpl<double, double, float >,
                addWeightedImpl<double, double, double>
            }
        }
    };

    GpuMat src1 = _src1.getGpuMat();
    GpuMat src2 = _src2.getGpuMat();

    int sdepth1 = src1.depth();
    int sdepth2 = src2.depth();

    ddepth = ddepth >= 0 ? CV_MAT_DEPTH(ddepth) : std::max(sdepth1, sdepth2);
    const int cn = src1.channels();

    CV_DbgAssert( src2.size() == src1.size() && src2.channels() == cn );
    CV_DbgAssert( sdepth1 <= CV_64F && sdepth2 <= CV_64F && ddepth <= CV_64F );

    _dst.create(src1.size(), CV_MAKE_TYPE(ddepth, cn));
    GpuMat dst = _dst.getGpuMat();

    GpuMat src1_ = src1.reshape(1);
    GpuMat src2_ = src2.reshape(1);
    GpuMat dst_ = dst.reshape(1);

    if (sdepth1 > sdepth2)
    {
        src1_.swap(src2_);
        std::swap(alpha, beta);
        std::swap(sdepth1, sdepth2);
    }

    const func_t func = funcs[sdepth1][sdepth2][ddepth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, alpha, src2_, beta, gamma, dst_, stream);
}

#endif
