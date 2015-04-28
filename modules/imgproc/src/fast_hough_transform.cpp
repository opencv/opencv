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
// Copyright (C) 2015, Smart Engines Ltd, all rights reserved.
// Copyright (C) 2015, Institute for Information Transmission Problems of the Russian Academy of Sciences (Kharkevich Institute), all rights reserved.
// Copyright (C) 2015, Dmitry Nikolaev, Simon Karpenko, Michail Aliev, Elena Kuznetsova, all rights reserved.
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
#include "fast_hough_transform.hpp"

namespace cv {

#if defined(_WIN32) && !defined(int32_t)
    typedef __int32 int32_t;
#endif

template<typename T, int D, HoughOp Op>
struct HoughOperator { };
#define SPECIALIZE_HOUGHOP(TOp, body)                                         \
    template<typename T, int D>                                               \
    struct HoughOperator<T, D, TOp> {                                         \
        static void operate(T *pDst, T *pSrc0, T* pSrc1, int len) {           \
            Mat dst (Size(1, len), D, pDst);                                  \
            Mat src0(Size(1, len), D, pSrc0);                                 \
            Mat src1(Size(1, len), D, pSrc1);                                 \
            body;                                                             \
        }                                                                     \
    };
SPECIALIZE_HOUGHOP(FHT_ADD, add(src0, src1, dst));
SPECIALIZE_HOUGHOP(FHT_MIN, min(src0, src1, dst));
SPECIALIZE_HOUGHOP(FHT_MAX, max(src0, src1, dst));
SPECIALIZE_HOUGHOP(FHT_AVE, addWeighted(src0, 0.5, src1, 0.5, 0.0, dst));
#undef SPECIALIZE_HOUGHOP

//----------------------fht----------------------------------------------------

template <typename T, int D, HoughOp OP>
void fhtCore(Mat     &img0,
             Mat     &img1,
             int32_t  y0,
             int32_t  h,
             bool     isPositiveShift,
             int      level,
             double   aspl)
{
    if (level <= 0)
        return;

    CV_Assert(h > 0);
    if (h == 1)
    {
        if ((aspl != 0.0) && (level == 1))
        {
            int w = img0.cols;
            uchar* pLine0 = img0.data + img0.step * y0;
            uchar* pLine1 = img1.data + img1.step * y0;
            int dLine = cvRound(y0 * aspl);
            dLine = dLine % w;
            dLine = dLine * img1.elemSize();
            int wLine = img0.cols * img0.elemSize();
            memcpy(pLine0, pLine1 + wLine - dLine, dLine);
            memcpy(pLine0 + dLine, pLine1, wLine - dLine);
        }
        else
        {
            memcpy(img0.data + img0.step * y0,
                   img1.data + img1.step * y0,
                   img0.cols * img0.elemSize());
        }
        return;
    }
    const int32_t k = h >> 1;
    fhtCore<T, D, OP>(img1, img0, y0, k,
                      isPositiveShift, level - 1, aspl);
    fhtCore<T, D, OP>(img1, img0, y0 + k, h - k,
                      isPositiveShift, level - 1, aspl);

    int au = 2 * k - 2;
    int ad = 2 * h - 2 * k - 2;
    int b = h - 1;
    int d = 2 * h - 2;
    int w = img0.cols;
    int wm = (h / w + 1) * w;

    for (int32_t s = 0; s < h; s++)
    {
        int su = (s * au + b) / d;
        int sd = (s * ad + b) / d;
        int rd = isPositiveShift ? sd - s : s - sd;
        rd = (rd + wm) % w;
        uchar *pLine0 = img0.data + img0.step * (y0 + s);
        uchar *pLineU = img1.data + img1.step * (y0 + su);
        uchar *pLineD = img1.data + img1.step * (y0 + k + sd);
        int w0 = img0.channels() * rd;
        int w1 = img0.channels() * (w - rd);

        if ((aspl != 0.0) && (level == 1))
        {
            int dU = cvRound((y0 + su) * aspl);
            dU = dU % w;
            dU *= img0.channels();
            int dD = cvRound((y0 + k + sd) * aspl);
            dD = dD % w;
            dD *= img0.channels();
            int wB = w * img0.channels();

            int dX = dD - dU;
            if (w0 >= dX)
            {
                if (w0 >= dD)
                {
                    HoughOperator<T, D, OP>::operate((T *)pLine0 + dU,
                                               (T *)pLineU,
                                               (T *)pLineD + (w0 - dX),
                                               w1 + dX);
                    HoughOperator<T, D, OP>::operate((T *)pLine0 + (w1 + dD),
                                               (T *)pLineU + (w1 + dX),
                                               (T *)pLineD,
                                               w0 - dD);
                    HoughOperator<T, D, OP>::operate((T *)pLine0,
                                               (T *)pLineU + (wB - dU),
                                               (T *)pLineD + (w0 - dD),
                                               dU);
                }
                else
                {
                    HoughOperator<T, D, OP>::operate((T *)pLine0 + dU,
                                               (T *)pLineU,
                                               (T *)pLineD + (w0 - dX),
                                               wB - dU);
                    HoughOperator<T, D, OP>::operate((T *)pLine0,
                                               (T *)pLineU + (wB - dU),
                                               (T *)pLineD + (w0 + wB - dD),
                                               dD - w0);
                    HoughOperator<T, D, OP>::operate((T *)pLine0 + (dD - w0),
                                               (T *)pLineU + (w1 + dX),
                                               (T *)pLineD,
                                               w0 - dX);
                }
            }
            else
            {
                HoughOperator<T, D, OP>::operate((T *)pLine0 + dU,
                                           (T *)pLineU,
                                           (T *)pLineD + (wB - (dX - w0)),
                                           dX - w0);
                HoughOperator<T, D, OP>::operate((T *)pLine0 + (dD - w0),
                                           (T *)pLineU + (dX - w0),
                                           (T *)pLineD,
                                           wB - (dX - w0) - dU);
                HoughOperator<T, D, OP>::operate((T *)pLine0,
                                           (T *)pLineU + (wB - dU),
                                           (T *)pLineD + (wB - (dX - w0) - dU),
                                           dU);
            }
        }
        else
        {
            HoughOperator<T, D, OP>::operate((T *)pLine0,
                                        (T *)pLineU,
                                        (T *)pLineD + w0,
                                        w1);
            HoughOperator<T, D, OP>::operate((T *)pLine0 + w1,
                                        (T *)pLineU + w1,
                                        (T *)pLineD,
                                        w0);
        }
    }
}

template <typename T, int D, HoughOp Op>
void fhtVoT(Mat    &img0,
            Mat    &img1,
            bool    isPositiveShift,
            double  aspl)
{
    int level = 0;
    for (int thres = 1; img0.rows > thres; thres <<= 1)
        level++;

    fhtCore<T, D, Op>(img0, img1, 0, img0.rows, isPositiveShift, level, aspl);
}

template <typename T, int D>
void fhtVo(Mat    &img0,
           Mat    &img1,
           bool    isPositiveShift,
           int     operation,
           double  aspl)
{
    switch (operation)
    {
    case FHT_ADD:
        fhtVoT<T, D, FHT_ADD>(img0, img1, isPositiveShift, aspl);
        break;
    case FHT_AVE:
        fhtVoT<T, D, FHT_AVE>(img0, img1, isPositiveShift, aspl);
        break;
    case FHT_MAX:
        fhtVoT<T, D, FHT_MAX>(img0, img1, isPositiveShift, aspl);
        break;
    case FHT_MIN:
        fhtVoT<T, D, FHT_MIN>(img0, img1, isPositiveShift, aspl);
        break;
    default:
        CV_Error_(CV_StsNotImplemented, ("Unknown operation %d", operation));
        break;
    }
}

static void fhtVo(Mat    &img0,
                  Mat    &img1,
                  bool    isPositiveShift,
                  int     operation,
                  double  aspl)
{
    int const depth = img0.depth();
    switch (depth)
    {
    case CV_8U:
        fhtVo<uchar, CV_8UC1>(img0, img1, isPositiveShift, operation, aspl);
        break;
    case CV_8S:
        fhtVo<schar, CV_8SC1>(img0, img1, isPositiveShift, operation, aspl);
        break;
    case CV_16U:
        fhtVo<ushort, CV_16UC1>(img0, img1, isPositiveShift, operation, aspl);
        break;
    case CV_16S:
        fhtVo<short, CV_16SC1>(img0, img1, isPositiveShift, operation, aspl);
        break;
    case CV_32S:
        fhtVo<int, CV_32SC1>(img0, img1, isPositiveShift, operation, aspl);
        break;
    case CV_32F:
        fhtVo<float, CV_32FC1>(img0, img1, isPositiveShift, operation, aspl);
        break;
    case CV_64F:
        fhtVo<double, CV_64FC1>(img0, img1, isPositiveShift, operation, aspl);
        break;
    default:
        CV_Error_(CV_StsNotImplemented, ("Unknown depth %d", depth));
        break;
    }
}

static void FHT(Mat       &dst,
                const Mat &src,
                int        operation,
                bool       isVertical,
                bool       isClockwise,
                double     aspl)
{
    CV_Assert(dst.cols > 0 && dst.rows > 0);
    CV_Assert(src.channels() == dst.channels());
    if (isVertical)
        CV_Assert(src.cols == dst.cols && src.rows == dst.rows);
    else
        CV_Assert(src.cols == dst.rows && src.rows == dst.cols);

    int level = 0;
    for (int thres = 1; dst.rows > thres; thres <<= 1)
        level++;

    Mat tmp;
    src.convertTo(tmp, dst.type());
    if (!isVertical)
        transpose(tmp, tmp);
    tmp.copyTo(dst);

    fhtVo(dst, tmp,
          isVertical ? isClockwise : !isClockwise,
          operation, aspl);
}

static void calculateFHTQuadrant(Mat       &dst,
                                 const Mat &src,
                                 int        operation,
                                 int        quadrant)
{
    bool bVert = true;
    bool bClock = true;
    double aspl = 0.0;
    switch (quadrant)
    {
    case ARO_315_0:
        bVert = true;
        bClock = false;
        break;
    case ARO_0_45:
        bVert = true;
        bClock = true;
        break;
    case ARO_45_90:
        bVert = false;
        bClock = false;
        break;
    case ARO_90_135:
        bVert = false;
        bClock = true;
        break;
    case ARO_CTR_VER:
        bVert = true;
        bClock = false;
        aspl = 0.5;
        break;
    case ARO_CTR_HOR:
        bVert = false;
        bClock = true;
        aspl = 0.5;
        break;
    default:
        CV_Error_(CV_StsNotImplemented, ("Unknown quadrant %d", quadrant));
    }

  FHT(dst, src, operation, bVert, bClock, aspl);
}

static void createDstFhtMat(OutputArray dst,
                            InputArray  src,
                            int         depth,
                            int         angleRange)
{
    int const rows = src.size().height;
    int const cols = src.size().width;
    int const channels = src.channels();

    int wd = cols + rows;
    int ht = 0;

    switch (angleRange)
    {
    case ARO_315_0:
    case ARO_0_45:
    case ARO_CTR_VER:
        ht = rows;
        break;
    case ARO_45_90:
    case ARO_90_135:
    case ARO_CTR_HOR:
        ht = cols;
        break;
    case ARO_315_45:
        ht = 2 * rows - 1;
        break;
    case ARO_45_135:
        ht = 2 * cols - 1;
        break;
    case ARO_315_135:
        ht = 2 * (cols + rows) - 3;
        break;
    default:
        CV_Error_(CV_StsNotImplemented, ("Unknown angleRange %d", angleRange));
    }

    dst.create(ht, wd, CV_MAKETYPE(depth, channels));
}

static void createFHTSrc(Mat       &srcFull,
                         const Mat &src,
                         int        angleRange)
{
    bool verticalTiling = false;
    switch (angleRange)
    {
    case ARO_315_0:
    case ARO_0_45:
    case ARO_CTR_VER:
    case ARO_315_45:
        verticalTiling = false;
        break;
    case ARO_45_90:
    case ARO_90_135:
    case ARO_CTR_HOR:
    case ARO_45_135:
        verticalTiling = true;
        break;
    default:
        CV_Error_(CV_StsNotImplemented, ("Unknown angleRange %d", angleRange));
    }

    int wd = verticalTiling ? src.cols : src.cols + src.rows;
    int ht = verticalTiling ? src.cols + src.rows : src.rows;
    srcFull = Mat(ht, wd, src.type());

    Mat imgReg;
    if (verticalTiling)
        imgReg = Mat(srcFull, Rect(0, src.rows, src.cols, src.cols));
    else
        imgReg = Mat(srcFull, Rect(src.cols, 0, src.rows, src.rows));
    imgReg = Mat::zeros(imgReg.size(), imgReg.type());

    imgReg = Mat(srcFull, Rect(0, 0, src.cols, src.rows));
    src.copyTo(imgReg);
}

static void setFHTDstRegion(Mat       &dstRegion,
                            const Mat &dst,
                            const Mat &src,
                            int        quadrant,
                            int        angleRange)
{
    int base = -1;
    switch (angleRange)
    {
    case ARO_315_0:
    case ARO_315_45:
    case ARO_315_135:
        base = 0;
        break;
    case ARO_0_45:
        base = 1;
        break;
    case ARO_45_90:
    case ARO_45_135:
        base = 2;
        break;
    case ARO_90_135:
        base = 3;
        break;
    default:
        CV_Error_(CV_StsNotImplemented, ("Unknown angleRange %d", angleRange));
    }

    int quad = -1;
    switch (quadrant)
    {
    case ARO_315_0:
        quad = 0;
        break;
    case ARO_0_45:
        quad = 1;
        break;
    case ARO_45_90:
        quad = 2;
        break;
    case ARO_90_135:
        quad = 3;
        break;
    default:
        CV_Error_(CV_StsNotImplemented, ("Unknown quadrant %d", quadrant));
    }

    if (quad < base)
        quad += 4;

    int shift = 0;
    for (int i = base; i < quad; i++)
        shift += (i & 2) ? src.cols - 1 : src.rows - 1;
    const int ht = (quad & 2) ? src.cols : src.rows;

    dstRegion = Mat(dst, Rect(0, shift, src.rows + src.cols, ht));
}



static void rotateLineRightCyclic(uchar *pLine,
                                  uchar *pBuf,
                                  int    len,
                                  int    shift)
{
  shift = shift % len;
  shift = (shift + len) % len;
  memcpy(pBuf, pLine, len);
  memcpy(pLine + shift, pBuf, len - shift);
  if (shift > 0)
    memcpy(pLine, pBuf + len - shift, shift);
}

static void skewQuadrant(Mat         &quad,
                         const Mat   &src,
                         uchar       *pBuf,
                         int          quadrant)
{
    CV_Assert(pBuf);

    const int wd = src.cols;
    const int ht = src.rows;

    double start = 0.;
    double step = .5;
    switch (quadrant)
    {
    case ARO_315_0:
    case ARO_CTR_VER:
        step = -.5;
        start = ht - 0.5;
        break;
    case ARO_0_45:
        step = -.5;
        start = ht * .5;
        break;
    case ARO_45_90:
        break;
    case ARO_90_135:
    case ARO_CTR_HOR:
        start = wd * .5 - 0.5;
        break;
    default:
        CV_Error_(CV_StsNotImplemented, ("Unknown quadrant %d", quadrant));
    }

    const int pixlen = quad.elemSize();
    const int len = quad.cols * pixlen;
    for (int y = 0; y < quad.rows; y++)
    {
        uchar *pLine = quad.ptr(y);
        int shift = static_cast<int>(start + step * y) * pixlen;
        rotateLineRightCyclic(pLine, pBuf, len, shift);
    }
}

void FastHoughTransform(InputArray  src,
                        OutputArray dst,
                        int         dstMatDepth,
                        int         angleRange,
                        int         operation,
                        int         makeSkew)
{
    Mat srcMat = src.getMat();
    if (!srcMat.isContinuous())
        srcMat = srcMat.clone();
    CV_Assert(srcMat.cols > 0 && srcMat.rows > 0);

    createDstFhtMat(dst, src, dstMatDepth, angleRange);
    Mat dstMat = dst.getMat();

    Mat imgRegDst;
    const int len = dstMat.cols * dstMat.elemSize();
    CV_Assert(len > 0);
    std::vector<uchar> buf_(len);
    uchar *buf(&buf_[0]);

    if (angleRange == ARO_315_135)
    {
        {
            Mat imgSrc;
            createFHTSrc(imgSrc, srcMat, ARO_315_45);

            setFHTDstRegion(imgRegDst, dstMat, srcMat, ARO_315_0, angleRange);
            calculateFHTQuadrant(imgRegDst, imgSrc, operation, ARO_315_0);
            flip(imgRegDst, imgRegDst, 0);
            if (HDO_DESKEW == makeSkew)
                skewQuadrant(imgRegDst, imgSrc, buf, ARO_315_0);

            setFHTDstRegion(imgRegDst, dstMat, srcMat, ARO_0_45, angleRange);
            calculateFHTQuadrant(imgRegDst, imgSrc, operation, ARO_0_45);
            if (HDO_DESKEW == makeSkew)
                skewQuadrant(imgRegDst, imgSrc, buf, ARO_0_45);
        }
        {
            Mat imgSrc;
            createFHTSrc(imgSrc, srcMat, ARO_45_135);

            setFHTDstRegion(imgRegDst, dstMat, srcMat, ARO_45_90, angleRange);
            calculateFHTQuadrant(imgRegDst, imgSrc, operation, ARO_45_90);
            flip(imgRegDst, imgRegDst, 0);
            if (HDO_DESKEW == makeSkew)
                skewQuadrant(imgRegDst, imgSrc, buf, ARO_45_90);

            setFHTDstRegion(imgRegDst, dstMat, srcMat, ARO_90_135, angleRange);
            calculateFHTQuadrant(imgRegDst, imgSrc, operation, ARO_90_135);
            if (HDO_DESKEW == makeSkew)
                skewQuadrant(imgRegDst, imgSrc, buf, ARO_90_135);
        }
        return;
    }

    Mat imgSrc;
    createFHTSrc(imgSrc, srcMat, angleRange);

    switch (angleRange)
    {
    case ARO_315_0:
        calculateFHTQuadrant(dstMat, imgSrc, operation, angleRange);
        flip(dstMat, dstMat, 0);
        if (HDO_DESKEW == makeSkew)
            skewQuadrant(dstMat, imgSrc, buf, angleRange);
        return;
    case ARO_0_45:
        calculateFHTQuadrant(dstMat, imgSrc, operation, angleRange);
        if (HDO_DESKEW == makeSkew)
            skewQuadrant(dstMat, imgSrc, buf, angleRange);
        return;
    case ARO_45_90:
        calculateFHTQuadrant(dstMat, imgSrc, operation, angleRange);
        flip(dstMat, dstMat, 0);
        if (HDO_DESKEW == makeSkew)
            skewQuadrant(dstMat, imgSrc, buf, angleRange);
        return;
    case ARO_90_135:
        calculateFHTQuadrant(dstMat, imgSrc, operation, angleRange);
        if (HDO_DESKEW == makeSkew)
            skewQuadrant(dstMat, imgSrc, buf, angleRange);
        return;
    case ARO_315_45:
        setFHTDstRegion(imgRegDst, dstMat, srcMat, ARO_315_0, angleRange);
        calculateFHTQuadrant(imgRegDst, imgSrc, operation, ARO_315_0);
        flip(imgRegDst, imgRegDst, 0);
        if (HDO_DESKEW == makeSkew)
            skewQuadrant(imgRegDst, imgSrc, buf, ARO_315_0);

        setFHTDstRegion(imgRegDst, dstMat, srcMat, ARO_0_45, angleRange);
        calculateFHTQuadrant(imgRegDst, imgSrc, operation, ARO_0_45);
        if (HDO_DESKEW == makeSkew)
            skewQuadrant(imgRegDst, imgSrc, buf, ARO_0_45);
        return;
    case ARO_45_135:
        setFHTDstRegion(imgRegDst, dstMat, srcMat, ARO_45_90, angleRange);
        calculateFHTQuadrant(imgRegDst, imgSrc, operation, ARO_45_90);
        flip(imgRegDst, imgRegDst, 0);
        if (HDO_DESKEW == makeSkew)
            skewQuadrant(imgRegDst, imgSrc, buf, ARO_45_90);

        setFHTDstRegion(imgRegDst, dstMat, srcMat, ARO_90_135, angleRange);
        calculateFHTQuadrant(imgRegDst, imgSrc, operation, ARO_90_135);
        if (HDO_DESKEW == makeSkew)
            skewQuadrant(imgRegDst, imgSrc, buf, ARO_90_135);
        return;
    case ARO_CTR_VER:
        calculateFHTQuadrant(dstMat, imgSrc, operation, angleRange);
        flip(dstMat, dstMat, 0);
        if (HDO_DESKEW == makeSkew)
            skewQuadrant(dstMat, imgSrc, buf, angleRange);
        return;
    case ARO_CTR_HOR:
        calculateFHTQuadrant(dstMat, imgSrc, operation, angleRange);
        if (HDO_DESKEW == makeSkew)
            skewQuadrant(dstMat, imgSrc, buf, angleRange);
        return;
    default:
        CV_Error_(CV_StsNotImplemented, ("Unknown angleRange %d", angleRange));
    }
}

//-----------------------------------------------------------------------------

//----------------------fht point2line-----------------------------------------
struct LineSegment {
    Point u, v;
    LineSegment(const Point _u, const Point _v) : u(_u), v(_v) { }
};

static void getRawPoint(Point       &rawHoughPoint,
                        int         &quadRawPoint,
                        const Point &givenHoughPoint,
                        const Mat   &srcImgInfo,
                        int          angleRange,
                        int          makeSkew)
{
    int base = -1;
    switch (angleRange)
    {
    case ARO_315_0:
    case ARO_315_45:
    case ARO_CTR_VER:
    case ARO_315_135:
        base = 0;
        break;
    case ARO_0_45:
        base = 1;
        break;
    case ARO_45_90:
    case ARO_45_135:
        base = 2;
        break;
    case ARO_90_135:
    case ARO_CTR_HOR:
        base = 3;
        break;
    default:
        CV_Error_(CV_StsNotImplemented, ("Unknown angleRange %d", angleRange));
    }

    int const cols = srcImgInfo.cols;
    int const rows = srcImgInfo.rows;

    rawHoughPoint.y = givenHoughPoint.y;
    int quad = 0, qsize = 0;
    for (quad = base; quad < 4; quad++)
    {
        qsize = (quad & 2) ? cols - 1 : rows - 1;
        if (rawHoughPoint.y <= qsize)
            break;
        rawHoughPoint.y -= qsize;
    }
    if (quad >= 4)
        CV_Error(CV_StsInternal, "");

    quadRawPoint = quad;

    double skewShift = 0.0;
    if (makeSkew == HDO_DESKEW)
    {
        switch (quad)
        {
        case 0:
            skewShift = rows - (rawHoughPoint.y + 1) * 0.5;
            break;
        case 1:
            skewShift = (rows - rawHoughPoint.y) * 0.5;
            break;
        case 2:
            skewShift = rawHoughPoint.y * 0.5;
            break;
        default:
            skewShift = 0.5 * (cols + rawHoughPoint.y - 1);
            break;
        }
    }

    rawHoughPoint.x = givenHoughPoint.x - static_cast<int>(skewShift);
    if (rawHoughPoint.x < 0)
        rawHoughPoint.x = rows + cols + rawHoughPoint.x;
}

static bool checkRawPoint(const Point &rawHoughPoint,
                          int          quadRawPoint,
                          const Mat   &srcImgInfo)
{
    int const cols = srcImgInfo.cols;
    int const rows = srcImgInfo.rows;

    switch (quadRawPoint)
    {
    case 0:
        //down left triangle on FHT
        if (rawHoughPoint.x - cols <= rawHoughPoint.y &&
            rawHoughPoint.x - cols >= 0)
        return false;
        break;
    case 1:
        //up right triangle on FHT
        if (rawHoughPoint.x - cols >= rawHoughPoint.y)
            return false;
        break;
    case 2:
        //up right triangle on up-down FHT image
        if (rawHoughPoint.x - rows >= cols - 1 - rawHoughPoint.y)
        return false;
        break;
    default:
        //down left triangle on up-down FHT image
        if (rawHoughPoint.x - rows <= cols - 1 - rawHoughPoint.y &&
            rawHoughPoint.x - rows >= 0)
        return false;
        break;
    }
    return true;
}

static void shiftLineSegment(LineSegment &segment,
                             const Point &shift)
{
    segment.u.x += shift.x;
    segment.v.x += shift.x;
    segment.u.y += shift.y;
    segment.v.y += shift.y;
}

static void lineFactors(double      &a,
                        double      &b,
                        double      &c,
                        const Point &point1,
                        const Point &point2)
{
    CV_Assert(point1.x != point2.x || point1.y != point2.y);

    Point vectorSegment(point2.x - point1.x, point2.y - point1.y);
    a = - vectorSegment.y;
    b = vectorSegment.x;
    c = - (a * point1.x + b * point1.y);
}

static void crossSegments(Point             &point,
                          const LineSegment &line1,
                          const LineSegment &line2)
{
    double a1 = 0.0, b1 = 0.0, c1 = 0.0;
    double a2 = 0.0, b2 = 0.0, c2 = 0.0;
    lineFactors(a1, b1, c1, line1.u, line1.v);
    lineFactors(a2, b2, c2, line2.u, line2.v);

    double uLine1onLine2 = a2 * line1.u.x + b2 * line1.u.y + c2;
    double vLine1onLine2 = a2 * line1.v.x + b2 * line1.v.y + c2;

    double ULine2onLine1 = a1 * line2.u.x + b1 * line2.u.y + c1;
    double VLine2onLine1 = a1 * line2.v.x + b1 * line2.v.y + c1;

    CV_Assert(ULine2onLine1 != 0 || VLine2onLine1 != 0 ||
              uLine1onLine2 != 0 || vLine1onLine2 != 0);
    CV_Assert(ULine2onLine1 * VLine2onLine1 <= 0 &&
              uLine1onLine2 * vLine1onLine2 <= 0);

    static const double double_eps = 1e-10;
    CV_Assert(std::abs(uLine1onLine2 - vLine1onLine2) >= double_eps);

    double mul = uLine1onLine2 / (uLine1onLine2 - vLine1onLine2);
    point.x =  cvRound(line1.u.x + mul * (line1.v.x - line1.u.x));
    point.y =  cvRound(line1.u.y + mul * (line1.v.y - line1.u.y));
}

void HoughPoint2Line(OutputArray  line,
                     const Point &houghPoint,
                     const Mat   &srcImgInfo,
                     int          angleRange,
                     int          makeSkew,
                     int          rules)
{
    int const cols = srcImgInfo.cols;
    int const rows = srcImgInfo.rows;

    CV_Assert(houghPoint.y >= 0);
    CV_Assert(houghPoint.x < cols + rows);

    int quad = 0;
    Point rawPoint(0, 0);
    getRawPoint(rawPoint, quad, houghPoint, srcImgInfo, angleRange, makeSkew);
    bool ret = checkRawPoint(rawPoint, quad, srcImgInfo);
    if (!(rules & RO_IGNORE_BORDERS))
    {
        CV_Assert(ret);
    }

    LineSegment dstLine(Point(0, 0), Point(0, 0));
    switch (quad)
    {
    case 0:
        dstLine.v.y = rows - 1;
        dstLine.u.x = rawPoint.x;
        dstLine.v.x = dstLine.u.x + rows - rawPoint.y - 1;
        break;
    case 1:
        dstLine.v.y = rows - 1;
        dstLine.u.x = rawPoint.x;
        dstLine.v.x = dstLine.u.x - rawPoint.y;
        break;
    case 2:
        dstLine.v.x = cols - 1;
        dstLine.u.y = rawPoint.x;
        dstLine.v.y = dstLine.u.y - cols + rawPoint.y + 1;
        break;
    default:
        dstLine.v.x = cols - 1;
        dstLine.u.y = rawPoint.x;
        dstLine.v.y = dstLine.u.y + rawPoint.y;
        break;
    }

    if (angleRange == ARO_CTR_VER)
    {
        int shift = cvRound(0.5 * dstLine.u.y);
        shift = shift % (cols + rows);
        dstLine.u.x = dstLine.u.x - shift;

        shift = cvRound(0.5 * dstLine.v.y);
        shift = shift % (cols + rows);
        dstLine.v.x = dstLine.v.x - shift;
    }
    else if (angleRange == ARO_CTR_HOR)
    {
        int shift = cvRound(0.5 * dstLine.u.x);
        shift = shift % (cols + rows);
        dstLine.u.y = dstLine.u.y - shift;

        shift = cvRound(0.5 * dstLine.v.x);
        shift = shift % (cols + rows);
        dstLine.v.y = dstLine.v.y - shift;
    }

    if (!ret)
    {
        Vec4i pts(dstLine.v.x, dstLine.v.y, dstLine.u.x, dstLine.u.y);
        Mat(pts).copyTo(line);
        return;
    }

    if (rules & RO_IGNORE_BORDERS)
    {
        switch (quad)
        {
        case 0:
            if (dstLine.v.x > cols + rows - 1)
            {
                Point shiftVector(-(cols + rows), 0);
                shiftLineSegment(dstLine, shiftVector);
            }
            break;
        case 3:
            if (dstLine.v.y > rows + cols - 1)
            {
                Point shiftVector(0, -(cols + rows));
                shiftLineSegment(dstLine, shiftVector);
            }
            break;
        default:
            break;
        }

        Vec4i pts(dstLine.v.x, dstLine.v.y, dstLine.u.x, dstLine.u.y);
        Mat(pts).copyTo(line);
        return;
    }

    Point minIntersectPoint(0, 0);

    Point minLeftUpSrcPoint(0, 0);
    Point minRightUpSrcPoint(cols - 1, 0);
    Point minLeftDownSrcPoint(0, rows - 1);
    Point minRightDownSrcPoint(cols - 1, rows - 1);

    switch (quad)
    {
    case 0:
        if (dstLine.v.x > cols + rows - 1)
        {
            LineSegment minRightToDstLine(Point(cols + rows, 0),
                                          Point(cols + rows, rows - 1) );
            crossSegments(minIntersectPoint, dstLine, minRightToDstLine);
            dstLine.u.y = minIntersectPoint.y;
            dstLine.u.x = 0;
            dstLine.v.x = dstLine.v.x - (cols + rows);
        }
        if (dstLine.v.x > cols  - 1)
        {
            LineSegment minRightSrcLine(minRightUpSrcPoint,
                                        minRightDownSrcPoint);
            crossSegments(minIntersectPoint, dstLine, minRightSrcLine);
            dstLine.v.y = minIntersectPoint.y;
            dstLine.v.x = cols - 1;
        }
        break;
    case 1:
        if (dstLine.v.x < 0)
        {
            LineSegment minLeftSrcLine(minLeftUpSrcPoint, minLeftDownSrcPoint);
            crossSegments(minIntersectPoint, dstLine, minLeftSrcLine);
            dstLine.v.y = minIntersectPoint.y;
            dstLine.v.x = 0;
        }
        if (dstLine.u.x > cols - 1)
        {
            LineSegment minRightSrcLine(minRightUpSrcPoint,
                                        minRightDownSrcPoint);
            crossSegments(minIntersectPoint, dstLine, minRightSrcLine);
            dstLine.u.y = minIntersectPoint.y;
            dstLine.u.x = cols - 1;
        }
        break;
    case 2:
        if (dstLine.v.y < 0)
        {
            LineSegment minTopSrcLine(minLeftUpSrcPoint, minRightUpSrcPoint);
            crossSegments(minIntersectPoint, dstLine, minTopSrcLine);
            dstLine.v.x = minIntersectPoint.x;
            dstLine.v.y = 0;
        }
        if (dstLine.u.y > rows - 1)
        {
            LineSegment minBottomSrcLine(minLeftDownSrcPoint,
                                         minRightDownSrcPoint);
            crossSegments(minIntersectPoint, dstLine, minBottomSrcLine );
            dstLine.u.x = minIntersectPoint.x;
            dstLine.u.y = rows - 1;
        }
        break;
    default:
        if (dstLine.v.y > rows + cols - 1)
        {
            LineSegment minDownToDstLine(Point(0, rows + cols),
                                         Point(cols - 1, rows + cols));
            crossSegments(minIntersectPoint, dstLine, minDownToDstLine);
            dstLine.u.x = minIntersectPoint.x;
            dstLine.u.y = 0;
            dstLine.v.y = dstLine.v.y - (rows + cols);
        }
        if (dstLine.v.y > rows - 1)
        {
            LineSegment minBottomSrcLine(minLeftDownSrcPoint,
                                         minRightDownSrcPoint);
            crossSegments(minIntersectPoint, dstLine, minBottomSrcLine);
            dstLine.v.x = minIntersectPoint.x;
            dstLine.v.y = rows - 1;
        }
        break;
    }

    Vec4i pts(dstLine.v.x, dstLine.v.y, dstLine.u.x, dstLine.u.y);
    Mat(pts).copyTo(line);
}

//-----------------------------------------------------------------------------


} // namespace cv
