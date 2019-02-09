// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_CORE_HPP
#define OPENCV_GAPI_CORE_HPP

#include <utility> // std::tuple

#include <opencv2/imgproc.hpp>

#include "opencv2/gapi/gmat.hpp"
#include "opencv2/gapi/gscalar.hpp"
#include "opencv2/gapi/gkernel.hpp"

/** \defgroup gapi_core G-API core (basic) functionality
@{
    @defgroup gapi_math Graph API: Math operations
    @defgroup gapi_pixelwise Graph API: Pixelwise operations
    @defgroup gapi_matrixop Graph API: Operations on matrices
    @defgroup gapi_transform Graph API: Geometric, depth and LUT-like image transformations
@}
 */
namespace cv { namespace gapi {
namespace core {
    using GMat2 = std::tuple<GMat,GMat>;
    using GMat3 = std::tuple<GMat,GMat,GMat>; // FIXME: how to avoid this?
    using GMat4 = std::tuple<GMat,GMat,GMat,GMat>;
    using GMatScalar = std::tuple<GMat, GScalar>;

    G_TYPED_KERNEL(GAdd, <GMat(GMat, GMat, int)>, "org.opencv.core.math.add") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc b, int ddepth) {
            if (ddepth == -1)
            {
                // OpenCV: When the input arrays in add/subtract/multiply/divide
                // functions have different depths, the output array depth must be
                // explicitly specified!
                // See artim_op() @ arithm.cpp
                GAPI_Assert(a.chan == b.chan);
                GAPI_Assert(a.depth == b.depth);
                return a;
            }
            return a.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GAddC, <GMat(GMat, GScalar, int)>, "org.opencv.core.math.addC") {
        static GMatDesc outMeta(GMatDesc a, GScalarDesc, int ddepth) {
            return a.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GSub, <GMat(GMat, GMat, int)>, "org.opencv.core.math.sub") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc b, int ddepth) {
            if (ddepth == -1)
            {
                // This macro should select a larger data depth from a and b
                // considering the number of channels in the same
                // FIXME!!! Clarify if it is valid for sub()
                GAPI_Assert(a.chan == b.chan);
                ddepth = std::max(a.depth, b.depth);
            }
            return a.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GSubC, <GMat(GMat, GScalar, int)>, "org.opencv.core.math.subC") {
        static GMatDesc outMeta(GMatDesc a, GScalarDesc, int ddepth) {
            return a.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GSubRC,<GMat(GScalar, GMat, int)>, "org.opencv.core.math.subRC") {
        static GMatDesc outMeta(GScalarDesc, GMatDesc b, int ddepth) {
            return b.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GMul, <GMat(GMat, GMat, double, int)>, "org.opencv.core.math.mul") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc, double, int ddepth) {
            return a.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GMulCOld, <GMat(GMat, double, int)>, "org.opencv.core.math.mulCOld") {
        static GMatDesc outMeta(GMatDesc a, double, int ddepth) {
            return a.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GMulC, <GMat(GMat, GScalar, int)>, "org.opencv.core.math.mulC"){
        static GMatDesc outMeta(GMatDesc a, GScalarDesc, int ddepth) {
            return a.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GMulS, <GMat(GMat, GScalar)>, "org.opencv.core.math.muls") {
        static GMatDesc outMeta(GMatDesc a, GScalarDesc) {
            return a;
        }
    }; // FIXME: Merge with MulC

    G_TYPED_KERNEL(GDiv, <GMat(GMat, GMat, double, int)>, "org.opencv.core.math.div") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc b, double, int ddepth) {
            if (ddepth == -1)
            {
                GAPI_Assert(a.depth == b.depth);
                return b;
            }
            return a.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GDivC, <GMat(GMat, GScalar, double, int)>, "org.opencv.core.math.divC") {
        static GMatDesc outMeta(GMatDesc a, GScalarDesc, double, int ddepth) {
            return a.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GDivRC, <GMat(GScalar, GMat, double, int)>, "org.opencv.core.math.divRC") {
        static GMatDesc outMeta(GScalarDesc, GMatDesc b, double, int ddepth) {
            return b.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GMean, <GScalar(GMat)>, "org.opencv.core.math.mean") {
        static GScalarDesc outMeta(GMatDesc) {
            return empty_scalar_desc();
        }
    };

    G_TYPED_KERNEL_M(GPolarToCart, <GMat2(GMat, GMat, bool)>, "org.opencv.core.math.polarToCart") {
        static std::tuple<GMatDesc, GMatDesc> outMeta(GMatDesc, GMatDesc a, bool) {
            return std::make_tuple(a, a);
        }
    };

    G_TYPED_KERNEL_M(GCartToPolar, <GMat2(GMat, GMat, bool)>, "org.opencv.core.math.cartToPolar") {
        static std::tuple<GMatDesc, GMatDesc> outMeta(GMatDesc x, GMatDesc, bool) {
            return std::make_tuple(x, x);
        }
    };

    G_TYPED_KERNEL(GPhase, <GMat(GMat, GMat, bool)>, "org.opencv.core.math.phase") {
        static GMatDesc outMeta(const GMatDesc &inx, const GMatDesc &, bool) {
            return inx;
        }
    };

    G_TYPED_KERNEL(GMask, <GMat(GMat,GMat)>, "org.opencv.core.pixelwise.mask") {
        static GMatDesc outMeta(GMatDesc in, GMatDesc) {
            return in;
        }
    };

    G_TYPED_KERNEL(GCmpGT, <GMat(GMat, GMat)>, "org.opencv.core.pixelwise.compare.cmpGT") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc) {
            return a.withDepth(CV_8U);
        }
    };

    G_TYPED_KERNEL(GCmpGE, <GMat(GMat, GMat)>, "org.opencv.core.pixelwise.compare.cmpGE") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc) {
            return a.withDepth(CV_8U);
        }
    };

    G_TYPED_KERNEL(GCmpLE, <GMat(GMat, GMat)>, "org.opencv.core.pixelwise.compare.cmpLE") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc) {
            return a.withDepth(CV_8U);
        }
    };

    G_TYPED_KERNEL(GCmpLT, <GMat(GMat, GMat)>, "org.opencv.core.pixelwise.compare.cmpLT") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc) {
            return a.withDepth(CV_8U);
        }
    };

    G_TYPED_KERNEL(GCmpEQ, <GMat(GMat, GMat)>, "org.opencv.core.pixelwise.compare.cmpEQ") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc) {
            return a.withDepth(CV_8U);
        }
    };

    G_TYPED_KERNEL(GCmpNE, <GMat(GMat, GMat)>, "org.opencv.core.pixelwise.compare.cmpNE") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc) {
            return a.withDepth(CV_8U);
        }
    };

    G_TYPED_KERNEL(GCmpGTScalar, <GMat(GMat, GScalar)>, "org.opencv.core.pixelwise.compare.cmpGTScalar"){
        static GMatDesc outMeta(GMatDesc a, GScalarDesc) {
            return a.withDepth(CV_8U);
        }
    };

    G_TYPED_KERNEL(GCmpGEScalar, <GMat(GMat, GScalar)>, "org.opencv.core.pixelwise.compare.cmpGEScalar"){
        static GMatDesc outMeta(GMatDesc a, GScalarDesc) {
            return a.withDepth(CV_8U);
        }
    };

    G_TYPED_KERNEL(GCmpLEScalar, <GMat(GMat, GScalar)>, "org.opencv.core.pixelwise.compare.cmpLEScalar"){
        static GMatDesc outMeta(GMatDesc a, GScalarDesc) {
            return a.withDepth(CV_8U);
        }
    };

    G_TYPED_KERNEL(GCmpLTScalar, <GMat(GMat, GScalar)>, "org.opencv.core.pixelwise.compare.cmpLTScalar"){
    static GMatDesc outMeta(GMatDesc a, GScalarDesc) {
            return a.withDepth(CV_8U);
        }
    };

    G_TYPED_KERNEL(GCmpEQScalar, <GMat(GMat, GScalar)>, "org.opencv.core.pixelwise.compare.cmpEQScalar"){
        static GMatDesc outMeta(GMatDesc a, GScalarDesc) {
            return a.withDepth(CV_8U);
        }
    };

    G_TYPED_KERNEL(GCmpNEScalar, <GMat(GMat, GScalar)>, "org.opencv.core.pixelwise.compare.cmpNEScalar"){
        static GMatDesc outMeta(GMatDesc a, GScalarDesc) {
            return a.withDepth(CV_8U);
        }
    };

    G_TYPED_KERNEL(GAnd, <GMat(GMat, GMat)>, "org.opencv.core.pixelwise.bitwise_and") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc) {
            return a;
        }
    };

    G_TYPED_KERNEL(GAndS, <GMat(GMat, GScalar)>, "org.opencv.core.pixelwise.bitwise_andS") {
        static GMatDesc outMeta(GMatDesc a, GScalarDesc) {
            return a;
        }
    };

    G_TYPED_KERNEL(GOr, <GMat(GMat, GMat)>, "org.opencv.core.pixelwise.bitwise_or") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc) {
            return a;
        }
    };

    G_TYPED_KERNEL(GOrS, <GMat(GMat, GScalar)>, "org.opencv.core.pixelwise.bitwise_orS") {
        static GMatDesc outMeta(GMatDesc a, GScalarDesc) {
            return a;
        }
    };

    G_TYPED_KERNEL(GXor, <GMat(GMat, GMat)>, "org.opencv.core.pixelwise.bitwise_xor") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc) {
            return a;
        }
    };

    G_TYPED_KERNEL(GXorS, <GMat(GMat, GScalar)>, "org.opencv.core.pixelwise.bitwise_xorS") {
        static GMatDesc outMeta(GMatDesc a, GScalarDesc) {
            return a;
        }
    };

    G_TYPED_KERNEL(GNot, <GMat(GMat)>, "org.opencv.core.pixelwise.bitwise_not") {
        static GMatDesc outMeta(GMatDesc a) {
            return a;
        }
    };

    G_TYPED_KERNEL(GSelect, <GMat(GMat, GMat, GMat)>, "org.opencv.core.pixelwise.select") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc, GMatDesc) {
            return a;
        }
    };

    G_TYPED_KERNEL(GMin, <GMat(GMat, GMat)>, "org.opencv.core.matrixop.min") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc) {
            return a;
        }
    };

    G_TYPED_KERNEL(GMax, <GMat(GMat, GMat)>, "org.opencv.core.matrixop.max") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc) {
            return a;
        }
    };

    G_TYPED_KERNEL(GAbsDiff, <GMat(GMat, GMat)>, "org.opencv.core.matrixop.absdiff") {
        static GMatDesc outMeta(GMatDesc a, GMatDesc) {
            return a;
        }
    };

    G_TYPED_KERNEL(GAbsDiffC, <GMat(GMat, GScalar)>, "org.opencv.core.matrixop.absdiffC") {
        static GMatDesc outMeta(GMatDesc a, GScalarDesc) {
            return a;
        }
    };

    G_TYPED_KERNEL(GSum, <GScalar(GMat)>, "org.opencv.core.matrixop.sum") {
        static GScalarDesc outMeta(GMatDesc) {
            return empty_scalar_desc();
        }
    };

    G_TYPED_KERNEL(GAddW, <GMat(GMat, double, GMat, double, double, int)>, "org.opencv.core.matrixop.addweighted") {
        static GMatDesc outMeta(GMatDesc a, double, GMatDesc b, double, double, int ddepth) {
            if (ddepth == -1)
            {
                // OpenCV: When the input arrays in add/subtract/multiply/divide
                // functions have different depths, the output array depth must be
                // explicitly specified!
                // See artim_op() @ arithm.cpp
                GAPI_Assert(a.chan == b.chan);
                GAPI_Assert(a.depth == b.depth);
                return a;
            }
            return a.withDepth(ddepth);
        }
    };

    G_TYPED_KERNEL(GNormL1, <GScalar(GMat)>, "org.opencv.core.matrixop.norml1") {
        static GScalarDesc outMeta(GMatDesc) {
            return empty_scalar_desc();
        }
    };

    G_TYPED_KERNEL(GNormL2, <GScalar(GMat)>, "org.opencv.core.matrixop.norml2") {
        static GScalarDesc outMeta(GMatDesc) {
            return empty_scalar_desc();
        }
    };

    G_TYPED_KERNEL(GNormInf, <GScalar(GMat)>, "org.opencv.core.matrixop.norminf") {
        static GScalarDesc outMeta(GMatDesc) {
            return empty_scalar_desc();
        }
    };

    G_TYPED_KERNEL_M(GIntegral, <GMat2(GMat, int, int)>, "org.opencv.core.matrixop.integral") {
        static std::tuple<GMatDesc, GMatDesc> outMeta(GMatDesc in, int sd, int sqd) {
            return std::make_tuple(in.withSizeDelta(1,1).withDepth(sd),
                                   in.withSizeDelta(1,1).withDepth(sqd));
        }
    };

    G_TYPED_KERNEL(GThreshold, <GMat(GMat, GScalar, GScalar, int)>, "org.opencv.core.matrixop.threshold") {
        static GMatDesc outMeta(GMatDesc in, GScalarDesc, GScalarDesc, int) {
            return in;
        }
    };


    G_TYPED_KERNEL_M(GThresholdOT, <GMatScalar(GMat, GScalar, int)>, "org.opencv.core.matrixop.thresholdOT") {
        static std::tuple<GMatDesc,GScalarDesc> outMeta(GMatDesc in, GScalarDesc, int) {
            return std::make_tuple(in, empty_scalar_desc());
        }
    };

    G_TYPED_KERNEL(GInRange, <GMat(GMat, GScalar, GScalar)>, "org.opencv.core.matrixop.inrange") {
        static GMatDesc outMeta(GMatDesc in, GScalarDesc, GScalarDesc) {
            return in.withType(CV_8U, 1);
        }
    };

    G_TYPED_KERNEL_M(GSplit3, <GMat3(GMat)>, "org.opencv.core.transform.split3") {
        static std::tuple<GMatDesc, GMatDesc, GMatDesc> outMeta(GMatDesc in) {
            const auto out_depth = in.depth;
            const auto out_desc  = in.withType(out_depth, 1);
            return std::make_tuple(out_desc, out_desc, out_desc);
        }
    };

    G_TYPED_KERNEL_M(GSplit4, <GMat4(GMat)>,"org.opencv.core.transform.split4") {
        static std::tuple<GMatDesc, GMatDesc, GMatDesc, GMatDesc> outMeta(GMatDesc in) {
            const auto out_depth = in.depth;
            const auto out_desc = in.withType(out_depth, 1);
            return std::make_tuple(out_desc, out_desc, out_desc, out_desc);
        }
    };

    G_TYPED_KERNEL(GResize, <GMat(GMat,Size,double,double,int)>, "org.opencv.core.transform.resize") {
        static GMatDesc outMeta(GMatDesc in, Size sz, double fx, double fy, int) {
            if (sz.width != 0 && sz.height != 0)
            {
                return in.withSize(sz);
            }
            else
            {
                GAPI_Assert(fx != 0. && fy != 0.);
                return in.withSize
                    (Size(static_cast<int>(std::round(in.size.width  * fx)),
                          static_cast<int>(std::round(in.size.height * fy))));
            }
        }
    };

    G_TYPED_KERNEL(GMerge3, <GMat(GMat,GMat,GMat)>, "org.opencv.core.transform.merge3") {
        static GMatDesc outMeta(GMatDesc in, GMatDesc, GMatDesc) {
            // Preserve depth and add channel component
            return in.withType(in.depth, 3);
        }
    };

    G_TYPED_KERNEL(GMerge4, <GMat(GMat,GMat,GMat,GMat)>, "org.opencv.core.transform.merge4") {
        static GMatDesc outMeta(GMatDesc in, GMatDesc, GMatDesc, GMatDesc) {
            // Preserve depth and add channel component
            return in.withType(in.depth, 4);
        }
    };

    G_TYPED_KERNEL(GRemap, <GMat(GMat, Mat, Mat, int, int, Scalar)>, "org.opencv.core.transform.remap") {
        static GMatDesc outMeta(GMatDesc in, Mat m1, Mat, int, int, Scalar) {
            return in.withSize(m1.size());
        }
    };

    G_TYPED_KERNEL(GFlip, <GMat(GMat, int)>, "org.opencv.core.transform.flip") {
        static GMatDesc outMeta(GMatDesc in, int) {
            return in;
        }
    };

    G_TYPED_KERNEL(GCrop, <GMat(GMat, Rect)>, "org.opencv.core.transform.crop") {
        static GMatDesc outMeta(GMatDesc in, Rect rc) {
            return in.withSize(Size(rc.width, rc.height));
        }
    };

    G_TYPED_KERNEL(GConcatHor, <GMat(GMat, GMat)>, "org.opencv.imgproc.transform.concatHor") {
        static GMatDesc outMeta(GMatDesc l, GMatDesc r) {
            return l.withSizeDelta(+r.size.width, 0);
        }
    };

    G_TYPED_KERNEL(GConcatVert, <GMat(GMat, GMat)>, "org.opencv.imgproc.transform.concatVert") {
        static GMatDesc outMeta(GMatDesc t, GMatDesc b) {
            return t.withSizeDelta(0, +b.size.height);
        }
    };

    G_TYPED_KERNEL(GLUT, <GMat(GMat, Mat)>, "org.opencv.core.transform.LUT") {
        static GMatDesc outMeta(GMatDesc in, Mat) {
            return in;
        }
    };

    G_TYPED_KERNEL(GConvertTo, <GMat(GMat, int, double, double)>, "org.opencv.core.transform.convertTo") {
        static GMatDesc outMeta(GMatDesc in, int rdepth, double, double) {
            return rdepth < 0 ? in : in.withDepth(rdepth);
        }
    };

    G_TYPED_KERNEL(GSqrt, <GMat(GMat)>, "org.opencv.core.math.sqrt") {
        static GMatDesc outMeta(GMatDesc in) {
            return in;
        }
    };

    G_TYPED_KERNEL(GNormalize, <GMat(GMat, double, double, int, int)>, "org.opencv.core.normalize") {
        static GMatDesc outMeta(GMatDesc in, double, double, int, int ddepth) {
            // unlike opencv doesn't have a mask as a parameter
            return (ddepth < 0 ? in : in.withDepth(ddepth));
        }
    };
}

//! @addtogroup gapi_math
//! @{

/** @brief Calculates the per-element sum of two matrices.

The function add calculates sum of two matrices of the same size and the same number of channels:
\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\f]

The function can be replaced with matrix expressions:
    \f[\texttt{dst} =  \texttt{src1} + \texttt{src2}\f]

The input matrices and the output matrix can all have the same or different depths. For example, you
can add a 16-bit unsigned matrix to a 8-bit signed matrix and store the sum as a 32-bit
floating-point matrix. Depth of the output matrix is determined by the ddepth parameter.
If src1.depth() == src2.depth(), ddepth can be set to the default -1. In this case, the output matrix will have
the same depth as the input matrices.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.math.add"
@param src1 first input matrix.
@param src2 second input matrix.
@param ddepth optional depth of the output matrix.
@sa sub, addWeighted
*/
GAPI_EXPORTS GMat add(const GMat& src1, const GMat& src2, int ddepth = -1);

/** @brief Calculates the per-element sum of matrix and given scalar.

The function addC adds a given scalar value to each element of given matrix.
The function can be replaced with matrix expressions:

    \f[\texttt{dst} =  \texttt{src1} + \texttt{c}\f]

Depth of the output matrix is determined by the ddepth parameter.
If ddepth is set to default -1, the depth of output matrix will be the same as the depth of input matrix.
The matrices can be single or multi channel. Output matrix must have the same size and number of channels as the input matrix.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.math.addC"
@param src1 first input matrix.
@param c scalar value to be added.
@param ddepth optional depth of the output matrix.
@sa sub, addWeighted
*/
GAPI_EXPORTS GMat addC(const GMat& src1, const GScalar& c, int ddepth = -1);
//! @overload
GAPI_EXPORTS GMat addC(const GScalar& c, const GMat& src1, int ddepth = -1);

/** @brief Calculates the per-element difference between two matrices.

The function sub calculates difference between two matrices, when both matrices have the same size and the same number of
channels:
    \f[\texttt{dst}(I) =   \texttt{src1}(I) -  \texttt{src2}(I)\f]

The function can be replaced with matrix expressions:
\f[\texttt{dst} =   \texttt{src1} -  \texttt{src2}\f]

The input matrices and the output matrix can all have the same or different depths. For example, you
can subtract two 8-bit unsigned matrices store the result as a 16-bit signed matrix.
Depth of the output matrix is determined by the ddepth parameter.
If src1.depth() == src2.depth(), ddepth can be set to the default -1. In this case, the output matrix will have
the same depth as the input matrices. The matrices can be single or multi channel.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.math.sub"
@param src1 first input matrix.
@param src2 second input matrix.
@param ddepth optional depth of the output matrix.
@sa  add, addC
  */
GAPI_EXPORTS GMat sub(const GMat& src1, const GMat& src2, int ddepth = -1);

/** @brief Calculates the per-element difference between matrix and given scalar.

The function can be replaced with matrix expressions:
    \f[\texttt{dst} =  \texttt{src} - \texttt{c}\f]

Depth of the output matrix is determined by the ddepth parameter.
If ddepth is set to default -1, the depth of output matrix will be the same as the depth of input matrix.
The matrices can be single or multi channel. Output matrix must have the same size as src.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.math.subC"
@param src first input matrix.
@param c scalar value to subtracted.
@param ddepth optional depth of the output matrix.
@sa  add, addC, subRC
  */
GAPI_EXPORTS GMat subC(const GMat& src, const GScalar& c, int ddepth = -1);

/** @brief Calculates the per-element difference between given scalar and the matrix.

The function can be replaced with matrix expressions:
    \f[\texttt{dst} =  \texttt{val} - \texttt{src}\f]

Depth of the output matrix is determined by the ddepth parameter.
If ddepth is set to default -1, the depth of output matrix will be the same as the depth of input matrix.
The matrices can be single or multi channel. Output matrix must have the same size as src.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.math.subRC"
@param c scalar value to subtract from.
@param src input matrix to be subtracted.
@param ddepth optional depth of the output matrix.
@sa  add, addC, subC
  */
GAPI_EXPORTS GMat subRC(const GScalar& c, const GMat& src, int ddepth = -1);

/** @brief Calculates the per-element scaled product of two matrices.

The function mul calculates the per-element product of two matrices:

\f[\texttt{dst} (I)= \texttt{saturate} ( \texttt{scale} \cdot \texttt{src1} (I)  \cdot \texttt{src2} (I))\f]

If src1.depth() == src2.depth(), ddepth can be set to the default -1. In this case, the output matrix will have
the same depth as the input matrices. The matrices can be single or multi channel.
Output matrix must have the same size as input matrices.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.math.mul"
@param src1 first input matrix.
@param src2 second input matrix of the same size and the same depth as src1.
@param scale optional scale factor.
@param ddepth optional depth of the output matrix.
@sa add, sub, div, addWeighted
*/
GAPI_EXPORTS GMat mul(const GMat& src1, const GMat& src2, double scale = 1.0, int ddepth = -1);

/** @brief Multiplies matrix by scalar.

The function mulC multiplies each element of matrix src by given scalar value:

\f[\texttt{dst} (I)= \texttt{saturate} (  \texttt{src1} (I)  \cdot \texttt{multiplier} )\f]

The matrices can be single or multi channel. Output matrix must have the same size as src.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.math.mulC"
@param src input matrix.
@param multiplier factor to be multiplied.
@param ddepth optional depth of the output matrix. If -1, the depth of output matrix will be the same as input matrix depth.
@sa add, sub, div, addWeighted
*/
GAPI_EXPORTS GMat mulC(const GMat& src, double multiplier, int ddepth = -1);
//! @overload
GAPI_EXPORTS GMat mulC(const GMat& src, const GScalar& multiplier, int ddepth = -1);   // FIXME: merge with mulc
//! @overload
GAPI_EXPORTS GMat mulC(const GScalar& multiplier, const GMat& src, int ddepth = -1);   // FIXME: merge with mulc

/** @brief Performs per-element division of two matrices.

The function divides one matrix by another:
\f[\texttt{dst(I) = saturate(src1(I)*scale/src2(I))}\f]

When src2(I) is zero, dst(I) will also be zero. Different channels of
multi-channel matrices are processed independently.
The matrices can be single or multi channel. Output matrix must have the same size and depth as src.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.math.div"
@param src1 first input matrix.
@param src2 second input matrix of the same size and depth as src1.
@param scale scalar factor.
@param ddepth optional depth of the output matrix; you can only pass -1 when src1.depth() == src2.depth().
@sa  mul, add, sub
*/
GAPI_EXPORTS GMat div(const GMat& src1, const GMat& src2, double scale, int ddepth = -1);

/** @brief Divides matrix by scalar.

The function divC divides each element of matrix src by given scalar value:

\f[\texttt{dst(I) = saturate(src(I)*scale/divisor)}\f]

When divisor is zero, dst(I) will also be zero. Different channels of
multi-channel matrices are processed independently.
The matrices can be single or multi channel. Output matrix must have the same size and depth as src.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.math.divC"
@param src input matrix.
@param divisor number to be divided by.
@param ddepth optional depth of the output matrix. If -1, the depth of output matrix will be the same as input matrix depth.
@param scale scale factor.
@sa add, sub, div, addWeighted
*/
GAPI_EXPORTS GMat divC(const GMat& src, const GScalar& divisor, double scale, int ddepth = -1);

/** @brief Divides scalar by matrix.

The function divRC divides given scalar by each element of matrix src and keep the division result in new matrix of the same size and type as src:

\f[\texttt{dst(I) = saturate(divident*scale/src(I))}\f]

When src(I) is zero, dst(I) will also be zero. Different channels of
multi-channel matrices are processed independently.
The matrices can be single or multi channel. Output matrix must have the same size and depth as src.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.math.divRC"
@param src input matrix.
@param divident number to be divided.
@param ddepth optional depth of the output matrix. If -1, the depth of output matrix will be the same as input matrix depth.
@param scale scale factor
@sa add, sub, div, addWeighted
*/
GAPI_EXPORTS GMat divRC(const GScalar& divident, const GMat& src, double scale, int ddepth = -1);

/** @brief Applies a mask to a matrix.

The function mask set value from given matrix if the corresponding pixel value in mask matrix set to true,
and set the matrix value to 0 overwise.

Supported src matrix data types are @ref CV_8UC1, @ref CV_16SC1, @ref CV_16UC1. Supported mask data type is @ref CV_8UC1.

@note Function textual ID is "org.opencv.core.math.mask"
@param src input matrix.
@param mask input mask matrix.
*/
GAPI_EXPORTS GMat mask(const GMat& src, const GMat& mask);

/** @brief Calculates an average (mean) of matrix elements.

The function mean calculates the mean value M of matrix elements,
independently for each channel, and return it.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.math.mean"
@param src input matrix.
*/
GAPI_EXPORTS GScalar mean(const GMat& src);

/** @brief Calculates x and y coordinates of 2D vectors from their magnitude and angle.

The function polarToCart calculates the Cartesian coordinates of each 2D
vector represented by the corresponding elements of magnitude and angle:
\f[\begin{array}{l} \texttt{x} (I) =  \texttt{magnitude} (I) \cos ( \texttt{angle} (I)) \\ \texttt{y} (I) =  \texttt{magnitude} (I) \sin ( \texttt{angle} (I)) \\ \end{array}\f]

The relative accuracy of the estimated coordinates is about 1e-6.

First output is a matrix of x-coordinates of 2D vectors.
Second output is a matrix of y-coordinates of 2D vectors.
Both output must have the same size and depth as input matrices.

@note Function textual ID is "org.opencv.core.math.polarToCart"

@param magnitude input floating-point @ref CV_32FC1 matrix (1xN) of magnitudes of 2D vectors;
@param angle input floating-point @ref CV_32FC1 matrix (1xN) of angles of 2D vectors.
@param angleInDegrees when true, the input angles are measured in
degrees, otherwise, they are measured in radians.
@sa cartToPolar, exp, log, pow, sqrt
*/
GAPI_EXPORTS std::tuple<GMat, GMat> polarToCart(const GMat& magnitude, const GMat& angle,
                                              bool angleInDegrees = false);

/** @brief Calculates the magnitude and angle of 2D vectors.

The function cartToPolar calculates either the magnitude, angle, or both
for every 2D vector (x(I),y(I)):
\f[\begin{array}{l} \texttt{magnitude} (I)= \sqrt{\texttt{x}(I)^2+\texttt{y}(I)^2} , \\ \texttt{angle} (I)= \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))[ \cdot180 / \pi ] \end{array}\f]

The angles are calculated with accuracy about 0.3 degrees. For the point
(0,0), the angle is set to 0.

First output is a matrix of magnitudes of the same size and depth as input x.
Second output is a matrix of angles that has the same size and depth as
x; the angles are measured in radians (from 0 to 2\*Pi) or in degrees (0 to 360 degrees).

@note Function textual ID is "org.opencv.core.math.cartToPolar"

@param x matrix of @ref CV_32FC1 x-coordinates.
@param y array of @ref CV_32FC1 y-coordinates.
@param angleInDegrees a flag, indicating whether the angles are measured
in radians (which is by default), or in degrees.
@sa polarToCart
*/
GAPI_EXPORTS std::tuple<GMat, GMat> cartToPolar(const GMat& x, const GMat& y,
                                              bool angleInDegrees = false);

/** @brief Calculates the rotation angle of 2D vectors.

The function cv::phase calculates the rotation angle of each 2D vector that
is formed from the corresponding elements of x and y :
\f[\texttt{angle} (I) =  \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))\f]

The angle estimation accuracy is about 0.3 degrees. When x(I)=y(I)=0 ,
the corresponding angle(I) is set to 0.
@param x input floating-point array of x-coordinates of 2D vectors.
@param y input array of y-coordinates of 2D vectors; it must have the
same size and the same type as x.
@param angleInDegrees when true, the function calculates the angle in
degrees, otherwise, they are measured in radians.
@return array of vector angles; it has the same size and same type as x.
*/
GAPI_EXPORTS GMat phase(const GMat& x, const GMat &y, bool angleInDegrees = false);

/** @brief Calculates a square root of array elements.

The function cv::gapi::sqrt calculates a square root of each input array element.
In case of multi-channel arrays, each channel is processed
independently. The accuracy is approximately the same as of the built-in
std::sqrt .
@param src input floating-point array.
@return output array of the same size and type as src.
*/
GAPI_EXPORTS GMat sqrt(const GMat &src);

//! @} gapi_math
//!
//! @addtogroup gapi_pixelwise
//! @{

/** @brief Performs the per-element comparison of two matrices checking if elements from first matrix are greater compare to elements in second.

The function compares elements of two matrices src1 and src2 of the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  > \texttt{src2} (I)\f]

When the comparison result is true, the corresponding element of output
array is set to 255. The comparison operations can be replaced with the
equivalent matrix expressions:
\f[\texttt{dst} =   \texttt{src1} > \texttt{src2}\f]

Output matrix of depth @ref CV_8U must have the same size and the same number of channels as
    the input matrices/matrix.

Supported input matrix data types are @ref CV_8UC1, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.pixelwise.compare.cmpGT"
@param src1 first input matrix.
@param src2 second input matrix/scalar of the same depth as first input matrix.
@sa min, max, threshold, cmpLE, cmpGE, cmpLS
*/
GAPI_EXPORTS GMat cmpGT(const GMat& src1, const GMat& src2);
/** @overload
@note Function textual ID is "org.opencv.core.pixelwise.compare.cmpGTScalar"
*/
GAPI_EXPORTS GMat cmpGT(const GMat& src1, const GScalar& src2);

/** @brief Performs the per-element comparison of two matrices checking if elements from first matrix are less than elements in second.

The function compares elements of two matrices src1 and src2 of the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  < \texttt{src2} (I)\f]

When the comparison result is true, the corresponding element of output
array is set to 255. The comparison operations can be replaced with the
equivalent matrix expressions:
    \f[\texttt{dst} =   \texttt{src1} < \texttt{src2}\f]

Output matrix of depth @ref CV_8U must have the same size and the same number of channels as
    the input matrices/matrix.

Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.pixelwise.compare.cmpLT"
@param src1 first input matrix.
@param src2 second input matrix/scalar of the same depth as first input matrix.
@sa min, max, threshold, cmpLE, cmpGE, cmpGT
*/
GAPI_EXPORTS GMat cmpLT(const GMat& src1, const GMat& src2);
/** @overload
@note Function textual ID is "org.opencv.core.pixelwise.compare.cmpLTScalar"
*/
GAPI_EXPORTS GMat cmpLT(const GMat& src1, const GScalar& src2);

/** @brief Performs the per-element comparison of two matrices checking if elements from first matrix are greater or equal compare to elements in second.

The function compares elements of two matrices src1 and src2 of the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  >= \texttt{src2} (I)\f]

When the comparison result is true, the corresponding element of output
array is set to 255. The comparison operations can be replaced with the
equivalent matrix expressions:
    \f[\texttt{dst} =   \texttt{src1} >= \texttt{src2}\f]

Output matrix of depth @ref CV_8U must have the same size and the same number of channels as
    the input matrices.

Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.pixelwise.compare.cmpGE"
@param src1 first input matrix.
@param src2 second input matrix/scalar of the same depth as first input matrix.
@sa min, max, threshold, cmpLE, cmpGT, cmpLS
*/
GAPI_EXPORTS GMat cmpGE(const GMat& src1, const GMat& src2);
/** @overload
@note Function textual ID is "org.opencv.core.pixelwise.compare.cmpLGEcalar"
*/
GAPI_EXPORTS GMat cmpGE(const GMat& src1, const GScalar& src2);

/** @brief Performs the per-element comparison of two matrices checking if elements from first matrix are less or equal compare to elements in second.

The function compares elements of two matrices src1 and src2 of the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  <=  \texttt{src2} (I)\f]

When the comparison result is true, the corresponding element of output
array is set to 255. The comparison operations can be replaced with the
equivalent matrix expressions:
    \f[\texttt{dst} =   \texttt{src1} <= \texttt{src2}\f]

Output matrix of depth @ref CV_8U must have the same size and the same number of channels as
    the input matrices.

Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.pixelwise.compare.cmpLE"
@param src1 first input matrix.
@param src2 second input matrix/scalar of the same depth as first input matrix.
@sa min, max, threshold, cmpGT, cmpGE, cmpLS
*/
GAPI_EXPORTS GMat cmpLE(const GMat& src1, const GMat& src2);
/** @overload
@note Function textual ID is "org.opencv.core.pixelwise.compare.cmpLEScalar"
*/
GAPI_EXPORTS GMat cmpLE(const GMat& src1, const GScalar& src2);

/** @brief Performs the per-element comparison of two matrices checking if elements from first matrix are equal to elements in second.

The function compares elements of two matrices src1 and src2 of the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  ==  \texttt{src2} (I)\f]

When the comparison result is true, the corresponding element of output
array is set to 255. The comparison operations can be replaced with the
equivalent matrix expressions:
    \f[\texttt{dst} =   \texttt{src1} == \texttt{src2}\f]

Output matrix of depth @ref CV_8U must have the same size and the same number of channels as
    the input matrices.

Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.pixelwise.compare.cmpEQ"
@param src1 first input matrix.
@param src2 second input matrix/scalar of the same depth as first input matrix.
@sa min, max, threshold, cmpNE
*/
GAPI_EXPORTS GMat cmpEQ(const GMat& src1, const GMat& src2);
/** @overload
@note Function textual ID is "org.opencv.core.pixelwise.compare.cmpEQScalar"
*/
GAPI_EXPORTS GMat cmpEQ(const GMat& src1, const GScalar& src2);

/** @brief Performs the per-element comparison of two matrices checking if elements from first matrix are not equal to elements in second.

The function compares elements of two matrices src1 and src2 of the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  !=  \texttt{src2} (I)\f]

When the comparison result is true, the corresponding element of output
array is set to 255. The comparison operations can be replaced with the
equivalent matrix expressions:
    \f[\texttt{dst} =   \texttt{src1} != \texttt{src2}\f]

Output matrix of depth @ref CV_8U must have the same size and the same number of channels as
    the input matrices.

Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.pixelwise.compare.cmpNE"
@param src1 first input matrix.
@param src2 second input matrix/scalar of the same depth as first input matrix.
@sa min, max, threshold, cmpEQ
*/
GAPI_EXPORTS GMat cmpNE(const GMat& src1, const GMat& src2);
/** @overload
@note Function textual ID is "org.opencv.core.pixelwise.compare.cmpNEScalar"
*/
GAPI_EXPORTS GMat cmpNE(const GMat& src1, const GScalar& src2);

/** @brief computes bitwise conjunction of the two matrixes (src1 & src2)
Calculates the per-element bit-wise logical conjunction of two matrices of the same size.

In case of floating-point matrices, their machine-specific bit
representations (usually IEEE754-compliant) are used for the operation.
In case of multi-channel matrices, each channel is processed
independently. Output matrix must have the same size and depth as the input
matrices.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.pixelwise.bitwise_and"

@param src1 first input matrix.
@param src2 second input matrix.
*/
GAPI_EXPORTS GMat bitwise_and(const GMat& src1, const GMat& src2);
/** @overload
@note Function textual ID is "org.opencv.core.pixelwise.compare.bitwise_andS"
@param src1 first input matrix.
@param src2 scalar, which will be per-lemenetly conjuncted with elements of src1.
*/
GAPI_EXPORTS GMat bitwise_and(const GMat& src1, const GScalar& src2);

/** @brief computes bitwise disjunction of the two matrixes (src1 | src2)
Calculates the per-element bit-wise logical disjunction of two matrices of the same size.

In case of floating-point matrices, their machine-specific bit
representations (usually IEEE754-compliant) are used for the operation.
In case of multi-channel matrices, each channel is processed
independently. Output matrix must have the same size and depth as the input
matrices.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.pixelwise.bitwise_or"

@param src1 first input matrix.
@param src2 second input matrix.
*/
GAPI_EXPORTS GMat bitwise_or(const GMat& src1, const GMat& src2);
/** @overload
@note Function textual ID is "org.opencv.core.pixelwise.compare.bitwise_orS"
@param src1 first input matrix.
@param src2 scalar, which will be per-lemenetly disjuncted with elements of src1.
*/
GAPI_EXPORTS GMat bitwise_or(const GMat& src1, const GScalar& src2);


/** @brief computes bitwise logical "exclusive or" of the two matrixes (src1 ^ src2)
Calculates the per-element bit-wise logical "exclusive or" of two matrices of the same size.

In case of floating-point matrices, their machine-specific bit
representations (usually IEEE754-compliant) are used for the operation.
In case of multi-channel matrices, each channel is processed
independently. Output matrix must have the same size and depth as the input
matrices.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.pixelwise.bitwise_xor"

@param src1 first input matrix.
@param src2 second input matrix.
*/
GAPI_EXPORTS GMat bitwise_xor(const GMat& src1, const GMat& src2);
/** @overload
@note Function textual ID is "org.opencv.core.pixelwise.compare.bitwise_xorS"
@param src1 first input matrix.
@param src2 scalar, for which per-lemenet "logical or" operation on elements of src1 will be performed.
*/
GAPI_EXPORTS GMat bitwise_xor(const GMat& src1, const GScalar& src2);


/** @brief Inverts every bit of an array.
The function bitwise_not calculates per-element bit-wise inversion of the input
matrix:
\f[\texttt{dst} (I) =  \neg \texttt{src} (I)\f]

In case of floating-point matrices, their machine-specific bit
representations (usually IEEE754-compliant) are used for the operation.
In case of multi-channel matrices, each channel is processed
independently. Output matrix must have the same size and depth as the input
matrix.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.pixelwise.bitwise_not"

@param src input matrix.
*/
GAPI_EXPORTS GMat bitwise_not(const GMat& src);

/** @brief Select values from either first or second of input matrices by given mask.
The function set to the output matrix either the value from the first input matrix if corresponding value of mask matrix is 255,
 or value from the second input matrix (if value of mask matrix set to 0).

Input mask matrix must be of @ref CV_8UC1 type, two other inout matrices and output matrix should be of the same type. The size should
be the same for all input and output matrices.
Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.pixelwise.select"

@param src1 first input matrix.
@param src2 second input matrix.
@param mask mask input matrix.
*/
GAPI_EXPORTS GMat select(const GMat& src1, const GMat& src2, const GMat& mask);

//! @} gapi_pixelwise


//! @addtogroup gapi_matrixop
//! @{
/** @brief Calculates per-element minimum of two matrices.

The function min calculates the per-element minimum of two matrices of the same size, number of channels and depth:
\f[\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{src2} (I))\f]
    where I is a multi-dimensional index of matrix elements. In case of
    multi-channel matrices, each channel is processed independently.
Output matrix must be of the same size and depth as src1.

Supported input matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.matrixop.min"
@param src1 first input matrix.
@param src2 second input matrix of the same size and depth as src1.
@sa max, compareEqual, compareLess, compareLessEqual
*/
GAPI_EXPORTS GMat min(const GMat& src1, const GMat& src2);

/** @brief Calculates per-element maximum of two matrices.

The function max calculates the per-element maximum of two matrices of the same size, number of channels and depth:
\f[\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{src2} (I))\f]
    where I is a multi-dimensional index of matrix elements. In case of
    multi-channel matrices, each channel is processed independently.
Output matrix must be of the same size and depth as src1.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.matrixop.max"
@param src1 first input matrix.
@param src2 second input matrix of the same size and depth as src1.
@sa min, compare, compareEqual, compareGreater, compareGreaterEqual
*/
GAPI_EXPORTS GMat max(const GMat& src1, const GMat& src2);

/** @brief Calculates the per-element absolute difference between two matrices.

The function absDiff calculates absolute difference between two matrices of the same size and depth:
    \f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2}(I)|)\f]
    where I is a multi-dimensional index of matrix elements. In case of
    multi-channel matrices, each channel is processed independently.
Output matrix must have the same size and depth as input matrices.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.matrixop.absdiff"
@param src1 first input matrix.
@param src2 second input matrix.
@sa abs
*/
GAPI_EXPORTS GMat absDiff(const GMat& src1, const GMat& src2);

/** @brief Calculates absolute value of matrix elements.

The function abs calculates absolute difference between matrix elements and given scalar value:
    \f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{matC}(I)|)\f]
    where matC is constructed from given scalar c and has the same sizes and depth as input matrix src.

Output matrix must be of the same size and depth as src.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.matrixop.absdiffC"
@param src input matrix.
@param c scalar to be subtracted.
@sa min, max
*/
GAPI_EXPORTS GMat absDiffC(const GMat& src, const GScalar& c);

/** @brief Calculates sum of all matrix elements.

The function sum calculates sum of all matrix elements, independently for each channel.

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.matrixop.sum"
@param src input matrix.
@sa min, max
*/
GAPI_EXPORTS GScalar sum(const GMat& src);

/** @brief Calculates the weighted sum of two matrices.

The function addWeighted calculates the weighted sum of two matrices as follows:
\f[\texttt{dst} (I)= \texttt{saturate} ( \texttt{src1} (I)* \texttt{alpha} +  \texttt{src2} (I)* \texttt{beta} +  \texttt{gamma} )\f]
where I is a multi-dimensional index of array elements. In case of multi-channel matrices, each
channel is processed independently.

The function can be replaced with a matrix expression:
    \f[\texttt{dst}(I) =  \texttt{alpha} * \texttt{src1}(I) - \texttt{beta} * \texttt{src2}(I) + \texttt{gamma} \f]

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.matrixop.addweighted"
@param src1 first input matrix.
@param alpha weight of the first matrix elements.
@param src2 second input matrix of the same size and channel number as src1.
@param beta weight of the second matrix elements.
@param gamma scalar added to each sum.
@param ddepth optional depth of the output matrix.
@sa  add, sub
*/
GAPI_EXPORTS GMat addWeighted(const GMat& src1, double alpha, const GMat& src2, double beta, double gamma, int ddepth = -1);

/** @brief Calculates the  absolute L1 norm of a matrix.

This version of normL1 calculates the absolute L1 norm of src.

As example for one array consider the function \f$r(x)= \begin{pmatrix} x \\ 1-x \end{pmatrix}, x \in [-1;1]\f$.
The \f$ L_{1} \f$ norm for the sample value \f$r(-1) = \begin{pmatrix} -1 \\ 2 \end{pmatrix}\f$
is calculated as follows
\f{align*}
    \| r(-1) \|_{L_1} &= |-1| + |2| = 3 \\
\f}
and for \f$r(0.5) = \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix}\f$ the calculation is
\f{align*}
    \| r(0.5) \|_{L_1} &= |0.5| + |0.5| = 1 \\
\f}

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.matrixop.norml1"
@param src input matrix.
@sa normL2, normInf
*/
GAPI_EXPORTS GScalar normL1(const GMat& src);

/** @brief Calculates the absolute L2 norm of a matrix.

This version of normL2 calculates the absolute L2 norm of src.

As example for one array consider the function \f$r(x)= \begin{pmatrix} x \\ 1-x \end{pmatrix}, x \in [-1;1]\f$.
The \f$ L_{2} \f$  norm for the sample value \f$r(-1) = \begin{pmatrix} -1 \\ 2 \end{pmatrix}\f$
is calculated as follows
\f{align*}
    \| r(-1) \|_{L_2} &= \sqrt{(-1)^{2} + (2)^{2}} = \sqrt{5} \\
\f}
and for \f$r(0.5) = \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix}\f$ the calculation is
\f{align*}
    \| r(0.5) \|_{L_2} &= \sqrt{(0.5)^{2} + (0.5)^{2}} = \sqrt{0.5} \\
\f}

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.
@note Function textual ID is "org.opencv.core.matrixop.norml2"
@param src input matrix.
@sa normL1, normInf
*/
GAPI_EXPORTS GScalar normL2(const GMat& src);

/** @brief Calculates the absolute infinite norm of a matrix.

This version of normInf calculates the absolute infinite norm of src.

As example for one array consider the function \f$r(x)= \begin{pmatrix} x \\ 1-x \end{pmatrix}, x \in [-1;1]\f$.
The \f$ L_{\infty} \f$ norm for the sample value \f$r(-1) = \begin{pmatrix} -1 \\ 2 \end{pmatrix}\f$
is calculated as follows
\f{align*}
    \| r(-1) \|_{L_\infty} &= \max(|-1|,|2|) = 2
\f}
and for \f$r(0.5) = \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix}\f$ the calculation is
\f{align*}
    \| r(0.5) \|_{L_\infty} &= \max(|0.5|,|0.5|) = 0.5.
\f}

Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.core.matrixop.norminf"
@param src input matrix.
@sa normL1, normL2
*/
GAPI_EXPORTS GScalar normInf(const GMat& src);

/** @brief Calculates the integral of an image.

The function calculates one or more integral images for the source image as follows:

\f[\texttt{sum} (X,Y) =  \sum _{x<X,y<Y}  \texttt{image} (x,y)\f]

\f[\texttt{sqsum} (X,Y) =  \sum _{x<X,y<Y}  \texttt{image} (x,y)^2\f]

The function return integral image as \f$(W+1)\times (H+1)\f$ , 32-bit integer or floating-point (32f or 64f) and
 integral image for squared pixel values; it is \f$(W+1)\times (H+)\f$, double-precision floating-point (64f) array.

@note Function textual ID is "org.opencv.core.matrixop.integral"

@param src input image.
@param sdepth desired depth of the integral and the tilted integral images, CV_32S, CV_32F, or
CV_64F.
@param sqdepth desired depth of the integral image of squared pixel values, CV_32F or CV_64F.
 */
GAPI_EXPORTS std::tuple<GMat, GMat> integral(const GMat& src, int sdepth = -1, int sqdepth = -1);

/** @brief Applies a fixed-level threshold to each matrix element.

The function applies fixed-level thresholding to a single- or multiple-channel matrix.
The function is typically used to get a bi-level (binary) image out of a grayscale image ( cmp functions could be also used for
this purpose) or for removing a noise, that is, filtering out pixels with too small or too large
values. There are several depths of thresholding supported by the function. They are determined by
depth parameter.

Also, the special values cv::THRESH_OTSU or cv::THRESH_TRIANGLE may be combined with one of the
above values. In these cases, the function determines the optimal threshold value using the Otsu's
or Triangle algorithm and uses it instead of the specified thresh . The function returns the
computed threshold value in addititon to thresholded matrix.
The Otsu's and Triangle methods are implemented only for 8-bit matrices.

Input image should be single channel only in case of cv::THRESH_OTSU or cv::THRESH_TRIANGLE flags.
Output matrix must be of the same size and depth as src.

@note Function textual ID is "org.opencv.core.matrixop.threshold"

@param src input matrix (@ref CV_8UC1, @ref CV_8UC3, or @ref CV_32FC1).
@param thresh threshold value.
@param maxval maximum value to use with the cv::THRESH_BINARY and cv::THRESH_BINARY_INV thresholding
depths.
@param depth thresholding depth (see the cv::ThresholdTypes).

@sa min, max, cmpGT, cmpLE, cmpGE, cmpLS
 */
GAPI_EXPORTS GMat threshold(const GMat& src, const GScalar& thresh, const GScalar& maxval, int depth);
/** @overload
This function appicable for all threshold depths except CV_THRESH_OTSU and CV_THRESH_TRIANGLE
@note Function textual ID is "org.opencv.core.matrixop.thresholdOT"
*/
GAPI_EXPORTS std::tuple<GMat, GScalar> threshold(const GMat& src, const GScalar& maxval, int depth);

/** @brief Applies a range-level threshold to each matrix element.

The function applies range-level thresholding to a single- or multiple-channel matrix.
It sets output pixel value to OxFF if the corresponding pixel value of input matrix is in specified range,or 0 otherwise.

Input and output matrices must be CV_8UC1.

@note Function textual ID is "org.opencv.core.matrixop.inRange"

@param src input matrix (CV_8UC1).
@param threshLow lower boundary value.
@param threshUp upper boundary value.

@sa threshold
 */
GAPI_EXPORTS GMat inRange(const GMat& src, const GScalar& threshLow, const GScalar& threshUp);

//! @} gapi_matrixop

//! @addtogroup gapi_transform
//! @{
/** @brief Resizes an image.

The function resizes the image src down to or up to the specified size.

Output image size will have the size dsize (when dsize is non-zero) or the size computed from
src.size(), fx, and fy; the depth of output is the same as of src.

If you want to resize src so that it fits the pre-created dst,
you may call the function as follows:
@code
    // explicitly specify dsize=dst.size(); fx and fy will be computed from that.
    resize(src, dst, dst.size(), 0, 0, interpolation);
@endcode
If you want to decimate the image by factor of 2 in each direction, you can call the function this
way:
@code
    // specify fx and fy and let the function compute the destination image size.
    resize(src, dst, Size(), 0.5, 0.5, interpolation);
@endcode
To shrink an image, it will generally look best with cv::INTER_AREA interpolation, whereas to
enlarge an image, it will generally look best with cv::INTER_CUBIC (slow) or cv::INTER_LINEAR
(faster but still looks OK).

@note Function textual ID is "org.opencv.core.transform.resize"

@param src input image.
@param dsize output image size; if it equals zero, it is computed as:
 \f[\texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}\f]
 Either dsize or both fx and fy must be non-zero.
@param fx scale factor along the horizontal axis; when it equals 0, it is computed as
\f[\texttt{(double)dsize.width/src.cols}\f]
@param fy scale factor along the vertical axis; when it equals 0, it is computed as
\f[\texttt{(double)dsize.height/src.rows}\f]
@param interpolation interpolation method, see cv::InterpolationFlags

@sa  warpAffine, warpPerspective, remap
 */
GAPI_EXPORTS GMat resize(const GMat& src, const Size& dsize, double fx = 0, double fy = 0, int interpolation = INTER_LINEAR);

/** @brief Creates one 3-channel (4-channel) matrix out of 3(4) single-channel ones.

The function merges several matrices to make a single multi-channel matrix. That is, each
element of the output matrix will be a concatenation of the elements of the input matrices, where
elements of i-th input matrix are treated as mv[i].channels()-element vectors.
Input matrix must be of @ref CV_8UC3 (@ref CV_8UC4) type.

The function split3/split4 does the reverse operation.

@note Function textual ID for merge3 is "org.opencv.core.transform.merge3"
@note Function textual ID for merge4 is "org.opencv.core.transform.merge4"

@param src1 first input matrix to be merged
@param src2 second input matrix to be merged
@param src3 third input matrix to be merged
@param src4 fourth input matrix to be merged
@sa  split4, split3
*/
GAPI_EXPORTS GMat merge4(const GMat& src1, const GMat& src2, const GMat& src3, const GMat& src4);
GAPI_EXPORTS GMat merge3(const GMat& src1, const GMat& src2, const GMat& src3);

/** @brief Divides a 3-channel (4-channel) matrix into 3(4) single-channel matrices.

The function splits a 3-channel (4-channel) matrix into 3(4) single-channel matrices:
\f[\texttt{mv} [c](I) =  \texttt{src} (I)_c\f]

All output matrices must be in @ref CV_8UC1.

@note Function textual for split3 ID is "org.opencv.core.transform.split3"
@note Function textual for split4 ID is "org.opencv.core.transform.split4"

@param src input @ref CV_8UC4 (@ref CV_8UC3) matrix.
@sa merge3, merge4
*/
GAPI_EXPORTS std::tuple<GMat, GMat, GMat,GMat> split4(const GMat& src);
GAPI_EXPORTS std::tuple<GMat, GMat, GMat> split3(const GMat& src);

/** @brief Applies a generic geometrical transformation to an image.

The function remap transforms the source image using the specified map:

\f[\texttt{dst} (x,y) =  \texttt{src} (map_x(x,y),map_y(x,y))\f]

where values of pixels with non-integer coordinates are computed using one of available
interpolation methods. \f$map_x\f$ and \f$map_y\f$ can be encoded as separate floating-point maps
in \f$map_1\f$ and \f$map_2\f$ respectively, or interleaved floating-point maps of \f$(x,y)\f$ in
\f$map_1\f$, or fixed-point maps created by using convertMaps. The reason you might want to
convert from floating to fixed-point representations of a map is that they can yield much faster
(\~2x) remapping operations. In the converted case, \f$map_1\f$ contains pairs (cvFloor(x),
cvFloor(y)) and \f$map_2\f$ contains indices in a table of interpolation coefficients.
Output image must be of the same size and depth as input one.

@note Function textual ID is "org.opencv.core.transform.remap"

@param src Source image.
@param map1 The first map of either (x,y) points or just x values having the type CV_16SC2,
CV_32FC1, or CV_32FC2.
@param map2 The second map of y values having the type CV_16UC1, CV_32FC1, or none (empty map
if map1 is (x,y) points), respectively.
@param interpolation Interpolation method (see cv::InterpolationFlags). The method INTER_AREA is
not supported by this function.
@param borderMode Pixel extrapolation method (see cv::BorderTypes). When
borderMode=BORDER_TRANSPARENT, it means that the pixels in the destination image that
corresponds to the "outliers" in the source image are not modified by the function.
@param borderValue Value used in case of a constant border. By default, it is 0.
@note
Due to current implementation limitations the size of an input and output images should be less than 32767x32767.
 */
GAPI_EXPORTS GMat remap(const GMat& src, const Mat& map1, const Mat& map2,
                      int interpolation, int borderMode = BORDER_CONSTANT,
                      const Scalar& borderValue = Scalar());

/** @brief Flips a 2D matrix around vertical, horizontal, or both axes.

The function flips the matrix in one of three different ways (row
and column indices are 0-based):
\f[\texttt{dst} _{ij} =
\left\{
\begin{array}{l l}
\texttt{src} _{\texttt{src.rows}-i-1,j} & if\;  \texttt{flipCode} = 0 \\
\texttt{src} _{i, \texttt{src.cols} -j-1} & if\;  \texttt{flipCode} > 0 \\
\texttt{src} _{ \texttt{src.rows} -i-1, \texttt{src.cols} -j-1} & if\; \texttt{flipCode} < 0 \\
\end{array}
\right.\f]
The example scenarios of using the function are the following:
*   Vertical flipping of the image (flipCode == 0) to switch between
    top-left and bottom-left image origin. This is a typical operation
    in video processing on Microsoft Windows\* OS.
*   Horizontal flipping of the image with the subsequent horizontal
    shift and absolute difference calculation to check for a
    vertical-axis symmetry (flipCode \> 0).
*   Simultaneous horizontal and vertical flipping of the image with
    the subsequent shift and absolute difference calculation to check
    for a central symmetry (flipCode \< 0).
*   Reversing the order of point arrays (flipCode \> 0 or
    flipCode == 0).
Output image must be of the same depth as input one, size should be correct for given flipCode.

@note Function textual ID is "org.opencv.core.transform.flip"

@param src input matrix.
@param flipCode a flag to specify how to flip the array; 0 means
flipping around the x-axis and positive value (for example, 1) means
flipping around y-axis. Negative value (for example, -1) means flipping
around both axes.
@sa remap
*/
GAPI_EXPORTS GMat flip(const GMat& src, int flipCode);

/** @brief Crops a 2D matrix.

The function crops the matrix by given cv::Rect.

Output matrix must be of the same depth as input one, size is specified by given rect size.

@note Function textual ID is "org.opencv.core.transform.crop"

@param src input matrix.
@param rect a rect to crop a matrix to
@sa resize
*/
GAPI_EXPORTS GMat crop(const GMat& src, const Rect& rect);

/** @brief Applies horizontal concatenation to given matrices.

The function horizontally concatenates two GMat matrices (with the same number of rows).
@code{.cpp}
    GMat A = { 1, 4,
               2, 5,
               3, 6 };
    GMat B = { 7, 10,
               8, 11,
               9, 12 };

    GMat C = gapi::concatHor(A, B);
    //C:
    //[1, 4, 7, 10;
    // 2, 5, 8, 11;
    // 3, 6, 9, 12]
@endcode
Output matrix must the same number of rows and depth as the src1 and src2, and the sum of cols of the src1 and src2.
Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.imgproc.transform.concatHor"

@param src1 first input matrix to be considered for horizontal concatenation.
@param src2 second input matrix to be considered for horizontal concatenation.
@sa concatVert
*/
GAPI_EXPORTS GMat concatHor(const GMat& src1, const GMat& src2);

/** @overload
The function horizontally concatenates given number of GMat matrices (with the same number of columns).
Output matrix must the same number of columns and depth as the input matrices, and the sum of rows of input matrices.

@param v vector of input matrices to be concatenated horizontally.
*/
GAPI_EXPORTS GMat concatHor(const std::vector<GMat> &v);

/** @brief Applies vertical concatenation to given matrices.

The function vertically concatenates two GMat matrices (with the same number of cols).
 @code{.cpp}
    GMat A = { 1, 7,
               2, 8,
               3, 9 };
    GMat B = { 4, 10,
               5, 11,
               6, 12 };

    GMat C = gapi::concatVert(A, B);
    //C:
    //[1, 7;
    // 2, 8;
    // 3, 9;
    // 4, 10;
    // 5, 11;
    // 6, 12]
 @endcode

Output matrix must the same number of cols and depth as the src1 and src2, and the sum of rows of the src1 and src2.
Supported matrix data types are @ref CV_8UC1, @ref CV_8UC3, @ref CV_16UC1, @ref CV_16SC1, @ref CV_32FC1.

@note Function textual ID is "org.opencv.imgproc.transform.concatVert"

@param src1 first input matrix to be considered for vertical concatenation.
@param src2 second input matrix to be considered for vertical concatenation.
@sa concatHor
*/
GAPI_EXPORTS GMat concatVert(const GMat& src1, const GMat& src2);

/** @overload
The function vertically concatenates given number of GMat matrices (with the same number of columns).
Output matrix must the same number of columns and depth as the input matrices, and the sum of rows of input matrices.

@param v vector of input matrices to be concatenated vertically.
*/
GAPI_EXPORTS GMat concatVert(const std::vector<GMat> &v);


/** @brief Performs a look-up table transform of a matrix.

The function LUT fills the output matrix with values from the look-up table. Indices of the entries
are taken from the input matrix. That is, the function processes each element of src as follows:
\f[\texttt{dst} (I)  \leftarrow \texttt{lut(src(I))}\f]

Supported matrix data types are @ref CV_8UC1.
Output is a matrix of the same size and number of channels as src, and the same depth as lut.

@note Function textual ID is "org.opencv.core.transform.LUT"

@param src input matrix of 8-bit elements.
@param lut look-up table of 256 elements; in case of multi-channel input array, the table should
either have a single channel (in this case the same table is used for all channels) or the same
number of channels as in the input matrix.
*/
GAPI_EXPORTS GMat LUT(const GMat& src, const Mat& lut);

/** @brief Performs a 3D look-up table transform of a multi-channel matrix.

The function LUT3D fills the output matrix with values from the look-up table. Indices of the entries
are taken from the input matrix. Interpolation is applied for mapping 0-255 range values to 0-16 range of 3DLUT table.
The function processes each element of src as follows:
@code{.cpp}
    dst[i][j][k] = lut3D[~src_r][~src_g][~src_b];
@endcode
where ~ means approximation.
Output is a matrix of of @ref CV_8UC3.

@note Function textual ID is "org.opencv.core.transform.LUT3D"

@param src input matrix of @ref CV_8UC3.
@param lut3D look-up table 17x17x17 3-channel elements.
@param interpolation The depth of interpoolation to be used.
*/
GAPI_EXPORTS GMat LUT3D(const GMat& src, const GMat& lut3D, int interpolation = INTER_NEAREST);

/** @brief Converts a matrix to another data depth with optional scaling.

The method converts source pixel values to the target data depth. saturate_cast\<\> is applied at
the end to avoid possible overflows:

\f[m(x,y) = saturate \_ cast<rType>( \alpha (*this)(x,y) +  \beta )\f]
Output matrix must be of the same size as input one.

@note Function textual ID is "org.opencv.core.transform.convertTo"
@param src input matrix to be converted from.
@param rdepth desired output matrix depth or, rather, the depth since the number of channels are the
same as the input has; if rdepth is negative, the output matrix will have the same depth as the input.
@param alpha optional scale factor.
@param beta optional delta added to the scaled values.
 */
GAPI_EXPORTS GMat convertTo(const GMat& src, int rdepth, double alpha=1, double beta=0);

/** @brief Normalizes the norm or value range of an array.

The function normalizes scale and shift the input array elements so that
\f[\| \texttt{dst} \| _{L_p}= \texttt{alpha}\f]
(where p=Inf, 1 or 2) when normType=NORM_INF, NORM_L1, or NORM_L2, respectively; or so that
\f[\min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}\f]
when normType=NORM_MINMAX (for dense arrays only).

@note Function textual ID is "org.opencv.core.normalize"

@param src input array.
@param alpha norm value to normalize to or the lower range boundary in case of the range
normalization.
@param beta upper range boundary in case of the range normalization; it is not used for the norm
normalization.
@param norm_type normalization type (see cv::NormTypes).
@param ddepth when negative, the output array has the same type as src; otherwise, it has the same
number of channels as src and the depth =ddepth.
@sa norm, Mat::convertTo
*/
GAPI_EXPORTS GMat normalize(const GMat& src, double alpha, double beta,
                            int norm_type, int ddepth = -1);
//! @} gapi_transform

} //namespace gapi
} //namespace cv

#endif //OPENCV_GAPI_CORE_HPP
