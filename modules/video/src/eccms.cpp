// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

/****************************************************************************************\
*                    Image Alignment (ECC algorithm, pyramidal version)                  *
\****************************************************************************************/

namespace cv {
typedef std::vector<cv::Mat> MatPyramid;

template<int motionType> struct MotionTraits {};

template<> struct MotionTraits<MOTION_TRANSLATION> {
    enum { paramAmount = 2 };
    static inline void tailHandlerGetCoord(float& sx,
                                              float& sy,
                                              float& denominator,
                                              int col,
                                              float numeratorX0,
                                              float numeratorY0,
                                              float /*denominator0*/,
                                              float /*a00*/,
                                              float /*a10*/,
                                              float /*a20*/)
    {
        denominator = 0;
        sx = (numeratorX0 + col);
        sy = numeratorY0;
    }
    template<typename elemtype>
    static constexpr std::array<float, paramAmount> fillJacobian(int /*col*/, int /*row*/, float/*sx*/, float/*sy*/, float fVal,
                                                     elemtype gx, elemtype gy, float /*a00*/, float /*a10*/,
                                                     float/*denominator*/) {
#define GX (fVal * gx)
#define GY (fVal * gy)
        return std::array<float, paramAmount>{GX, GY};
#undef GX
#undef GY
    }
};

template<> struct MotionTraits<MOTION_EUCLIDEAN> {
    enum { paramAmount = 3 };
    static inline void tailHandlerGetCoord(float& sx,
                                              float& sy,
                                              float& denominator,
                                              int col,
                                              float numeratorX0,
                                              float numeratorY0,
                                              float /*denominator0*/,
                                              float a00,
                                              float a10,
                                              float /*a20*/)
    {
        denominator = 0;
        sx = (numeratorX0 + a00 * col);
        sy = (numeratorY0 + a10 * col);
    }

    template<typename elemtype>
    static constexpr std::array<float, paramAmount> fillJacobian(int col, int row, float/*sx*/, float/*sy*/, float fVal,
                                                     elemtype gx, elemtype gy, float a00, float a10,
                                                     float/*denominator*/) {
#define GX (fVal * gx)
#define GY (fVal * gy)
#define HATX (-col * a10 - row * a00)
#define HATY (col * a00 - row * a10)
#define GZ (GX * HATX + GY * HATY)
        return std::array<float, paramAmount>{GZ, GX, GY};
#undef GX
#undef GY
#undef HATX
#undef HATY
#undef GZ
    }
};

template<> struct MotionTraits<MOTION_AFFINE> {
    enum { paramAmount = 6};
    static inline void tailHandlerGetCoord(float& sx,
                                              float& sy,
                                              float& denominator,
                                              int col,
                                              float numeratorX0,
                                              float numeratorY0,
                                              float /*denominator0*/,
                                              float a00,
                                              float a10,
                                              float /*a20*/)
    {
        denominator = 0;
        sx = (numeratorX0 + a00 * col);
        sy = (numeratorY0 + a10 * col);
    }

    template<typename elemtype>
    static constexpr std::array<float, paramAmount> fillJacobian(int col, int row, float/*sx*/, float/*sy*/, float fVal,
                                                     elemtype gx, elemtype gy, float /*a00*/, float /*a10*/,
                                                     float/*denominator*/) {
#define GX (fVal * gx)
#define GY (fVal * gy)
        return std::array<float, paramAmount>{GX * col, GY * col, GX * row, GY * row, GX, GY};
#undef GX
#undef GY
    }
};

template<> struct MotionTraits<MOTION_HOMOGRAPHY> {
    enum { paramAmount = 8};
    static inline void tailHandlerGetCoord(float& sx,
                                              float& sy,
                                              float& denominator,
                                              int col,
                                              float numeratorX0,
                                              float numeratorY0,
                                              float denominator0,
                                              float a00,
                                              float a10,
                                              float a20)
    {
        denominator = 1.f / (col * a20 + denominator0);
        sx = (numeratorX0 + a00 * col) * denominator;
        sy = (numeratorY0 + a10 * col) * denominator;
    }

    template<typename elemtype>
    static constexpr std::array<float, paramAmount> fillJacobian(int col, int row, float sx, float sy, float fVal,
                                                     elemtype gx, elemtype gy, float/*a00*/, float/*a10*/,
                                                     float denominator) {
#define GX (fVal * float(gx) * denominator)
#define GY (fVal * float(gy) * denominator)
#define GZ (-(GX * sx + GY * sy))
        return std::array<float, paramAmount>{GX * col, GY * col, GZ * col, GX * row, GY * row, GZ * row, GX, GY};
#undef GX
#undef GY
#undef GZ
    }
};

inline void reinterpret(Mat& mat, int newdepth) {
    mat.flags = (mat.flags & ~CV_MAT_DEPTH_MASK) | newdepth;
}

template<int N, class F>
class constexprForClass
{
public:
    static inline void execute(F&& fVal) {
        constexprForClass<N-1, F>::execute(std::forward<F>(fVal));
        fVal(N-1);
    }
};

template<class F>
class constexprForClass<0, F>
{
public:
    static inline void execute(F&&) {}
};

template<int N, class F>
void constexprFor(F&& fVal) {
    constexprForClass<N, F>::execute(std::forward<F>(fVal));
}
template<int R, int C, class F>
class constexprForUpperTriangleClassOneRow
{
public:
    static inline void execute(F&& fVal) {
        constexprForUpperTriangleClassOneRow<R, C-1, F>::execute(std::forward<F>(fVal));
        fVal(R, R + C - 1);
    }
};

template<int R, class F>
class constexprForUpperTriangleClassOneRow<R, 0, F>
{
public:
    static inline void execute(F&&) {}
};

template<int R, int D, class F>
class constexprForUpperTriangleClass
{
public:
    static inline void execute(F&& fVal) {
        constexprForUpperTriangleClass<R-1, D, F>::execute(std::forward<F>(fVal));
        constexprForUpperTriangleClassOneRow<R-1, D-R+1, F>::execute(std::forward<F>(fVal));
    }
};

template<int D, class F>
class constexprForUpperTriangleClass<0, D, F>
{
public:
    static inline void execute(F&&) {}
};

template<int M, class F>
void constexprForUpperTriangle(F&& fVal) {
    constexprForUpperTriangleClass<M,M,F>::execute(std::forward<F>(fVal));
}

template<int MotionType>
constexpr int hessianRowStart(int row) {
    return row == 0 ? 0 : (MotionTraits<MotionType>::paramAmount - row + 1 + hessianRowStart<MotionType>(row - 1));
}

template<int motionType, typename elemtype>
static double imageHessianProjECC(const Mat& map,
                           const Mat& sampleWithGrad,
                           const Mat& ref,
                           double& sampSum,
                           double& sampSqSum,
                           double& refSum,
                           double& refSqSum,
                           int& nz,
                           Mat& hessian,
                           Mat& sampleProj,
                           Mat& refProj,
                           int deltaY,
                           int interpolation) {
    static_assert(std::is_same<float, elemtype>::value, "imageHessianProjECC: f16 is not supported yet");
#define HESSIAN_PARAMS (MotionTraits<motionType>::paramAmount)

    CV_Assert(map.type() == CV_64F);
    CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR);
    CV_Assert(hessian.type() == CV_64F && sampleProj.type() == CV_64F && refProj.type() == CV_64F);
    if (sampleProj.size() != Size(1, HESSIAN_PARAMS) || refProj.size() != Size(1, HESSIAN_PARAMS)) {
        CV_Error(Error::BadImageSize, format("imageHessianProjECC: Wrong sample projection/reference projection size. 1x%d expected", HESSIAN_PARAMS));
    }
    if (hessian.size() != Size(HESSIAN_PARAMS, HESSIAN_PARAMS)) {
        CV_Error(Error::BadImageSize, format("imageHessianProjECC: Wrong hessian size. %dx%d expected", HESSIAN_PARAMS, HESSIAN_PARAMS));
    }
    if (!map.isContinuous()) {
        CV_Error(Error::BadStep, "imageHessianProjECC: Map should be continuous");
    }
    if (std::is_same<float, elemtype>::value) {
        CV_Assert(sampleWithGrad.type() == CV_32FC4 && ref.type() == CV_32FC2);
    }

    int hr = ref.rows;
    int wr = ref.cols;
    int hs = sampleWithGrad.rows;
    int ws = sampleWithGrad.cols;
    unsigned int ycond = hs - (INTER_LINEAR ? 1 : 0);
    unsigned int xcond = ws - (INTER_LINEAR ? 1 : 0);

    hessian = Mat::zeros(hessian.size(), hessian.type());
    sampleProj = Mat::zeros(sampleProj.size(), sampleProj.type());
    refProj = Mat::zeros(refProj.size(), refProj.type());

    const int MAX_STRIPES = 128;
    int stripesAmount = std::min(MAX_STRIPES, hr / deltaY);
    std::vector<Matx<double, MotionTraits<motionType>::paramAmount, MotionTraits<motionType>::paramAmount> > hessPs(stripesAmount);
    std::vector<Vec<double, MotionTraits<motionType>::paramAmount> > iprojs(stripesAmount);
    std::vector<Vec<double, MotionTraits<motionType>::paramAmount> > tprojs(stripesAmount);
    std::vector<Vec<double, MotionTraits<motionType>::paramAmount> > projSubs(stripesAmount);
    std::vector<double> correlations(stripesAmount, 0.);

    std::vector<double> sampSums(stripesAmount, 0);
    std::vector<double> sampSqSums(stripesAmount, 0);
    std::vector<double> refSums(stripesAmount, 0);
    std::vector<double> refSqSums(stripesAmount, 0);
    std::vector<int> nzs(stripesAmount, 0);
    std::vector<double> sampMaskedSums(stripesAmount, 0);
    std::vector<double> refMaskedSums(stripesAmount, 0);

    double a00 = map.at<double>(0, 0);
    double a01 = map.at<double>(0, 1);
    double a02 = map.at<double>(0, 2);
    double a10 = map.at<double>(1, 0);
    double a11 = map.at<double>(1, 1);
    double a12 = map.at<double>(1, 2);
    double a20 = 0;
    double a21 = 0;
    double a22 = 0;
    if (motionType == MOTION_HOMOGRAPHY) {
        a20 = map.at<double>(2, 0);
        a21 = map.at<double>(2, 1);
        a22 = map.at<double>(2, 2);
    }

    const elemtype* samplePtr0 = sampleWithGrad.ptr<elemtype>(0);

    parallel_for_(Range(0, stripesAmount), [&](const Range& range) {
        int stripeIdx = range.start;
        int ystart = (hr * stripeIdx) / stripesAmount;
        ystart = roundUp(ystart, deltaY);
        int yend = (hr * (range.end)) / stripesAmount;
        // we don't store intermediate jacobian; instead, we iteratively update Hessian, sampleProj and refProj
        for (int y = ystart; y < yend; y += deltaY) {
            const elemtype* refPtr = ref.ptr<elemtype>(y);

            std::array<float, (HESSIAN_PARAMS * HESSIAN_PARAMS + HESSIAN_PARAMS) / 2> hessPcache{};
            std::array<float, HESSIAN_PARAMS> iprojCache{};
            std::array<float, HESSIAN_PARAMS> tprojCache{};
            std::array<float, HESSIAN_PARAMS> projSubCache{};

            const float numeratorX0 = y * (float)a01 + (float)a02;
            const float numeratorY0 = y * (float)a11 + (float)a12;
            const float denominator0 = y * (float)a21 + (float)a22;
            int x = 0;
            for (; x < wr; x++) { //Tail handler
                float sx, sy, denominator;
                MotionTraits<motionType>::tailHandlerGetCoord(sx, sy, denominator, x, numeratorX0, numeratorY0,
                                                denominator0, (float)a00, (float)a10, (float)a20);
                const unsigned int x0 = (interpolation == INTER_LINEAR) ? static_cast<int>(std::floor(sx)) : saturate_cast<unsigned>(sx);
                const unsigned int y0 = (interpolation == INTER_LINEAR) ? static_cast<int>(std::floor(sy)) : saturate_cast<unsigned>(sy);
                if(interpolation == INTER_LINEAR && (static_cast<int>(x0 < xcond) & static_cast<int>(y0 < ycond)) == 0)
                    continue;
                if (interpolation == INTER_NEAREST && (static_cast<int>(x0 < xcond) & static_cast<int>(y0 < ycond)) == 0)
                    continue;
                float sampleVal = 0;
                float gx = 0;
                float gy = 0;
                float fVal = 0;
                if(interpolation == INTER_LINEAR) {
                    const int x1 = x0 + 1;
                    const int y1 = y0 + 1;

                    const float dx = sx - x0;
                    const float dy = sy - y0;

                    const float p00_val = samplePtr0[4 * y0 * ws + 4 * x0];
                    const float p01_val = samplePtr0[4 * y0 * ws + 4 * x1];
                    const float p10_val = samplePtr0[4 * y1 * ws + 4 * x0];
                    const float p11_val = samplePtr0[4 * y1 * ws + 4 * x1];
                    const float p0_val = p00_val * (1.0f - dx) + p01_val * dx;
                    const float p1_val = p10_val * (1.0f - dx) + p11_val * dx;

                    const float p00_gx = samplePtr0[4 * y0 * ws + 4 * x0 + 1];
                    const float p01_gx = samplePtr0[4 * y0 * ws + 4 * x1 + 1];
                    const float p10_gx = samplePtr0[4 * y1 * ws + 4 * x0 + 1];
                    const float p11_gx = samplePtr0[4 * y1 * ws + 4 * x1 + 1];
                    const float p0_gx = p00_gx * (1.0f - dx) + p01_gx * dx;
                    const float p1_gx = p10_gx * (1.0f - dx) + p11_gx * dx;

                    const float p00_gy = samplePtr0[4 * y0 * ws + 4 * x0 + 2];
                    const float p01_gy = samplePtr0[4 * y0 * ws + 4 * x1 + 2];
                    const float p10_gy = samplePtr0[4 * y1 * ws + 4 * x0 + 2];
                    const float p11_gy = samplePtr0[4 * y1 * ws + 4 * x1 + 2];
                    const float p0_gy = p00_gy * (1.0f - dx) + p01_gy * dx;
                    const float p1_gy = p10_gy * (1.0f - dx) + p11_gy * dx;

                    const float p00_mask = samplePtr0[4 * y0 * ws + 4 * x0 + 2] == 0.f ? 0.f : 1.f;
                    const float p01_mask = samplePtr0[4 * y0 * ws + 4 * x1 + 2] == 0.f ? 0.f : 1.f;
                    const float p10_mask = samplePtr0[4 * y1 * ws + 4 * x0 + 2] == 0.f ? 0.f : 1.f;
                    const float p11_mask = samplePtr0[4 * y1 * ws + 4 * x1 + 2] == 0.f ? 0.f : 1.f;

                    sampleVal = p0_val * (1.0f - dy) + p1_val * dy;
                    gx = p0_gx * (1.0f - dy) + p1_gx * dy;
                    gy = p0_gy * (1.0f - dy) + p1_gy * dy;
                    fVal = p00_mask * p01_mask * p10_mask * p11_mask;
                }
                else { // if(interpolation == INTER_NEAREST)
                    const elemtype* samplePtr = samplePtr0 + y0 * (ws * 4) + x0 * 4;
                    sampleVal = samplePtr[0];
                    gx = samplePtr[1];
                    gy = samplePtr[2];
                    fVal = float(samplePtr[3]) == 0.f ? 0.f : 1.f;
                }

                float refVal = refPtr[2 * x];
                fVal *= float(refPtr[2 * x + 1]) == 0.f ? 0.f : 1.f;
                sampleVal *= fVal;
                refVal *= fVal;
                sampSums[stripeIdx] += sampleVal;
                sampSqSums[stripeIdx] += sampleVal * sampleVal;
                refSums[stripeIdx] += refVal;
                refSqSums[stripeIdx] += refVal * refVal;
                nzs[stripeIdx] += (int)fVal;
                sampMaskedSums[stripeIdx] += sampleVal;
                refMaskedSums[stripeIdx] += refVal;
                std::array<float, HESSIAN_PARAMS> jac = MotionTraits<motionType>::fillJacobian(x, y, sx, sy,
                                                                                        fVal, gx,
                                                                                        gy, (float)a00,
                                                                                        (float)a10, denominator);
                constexprForUpperTriangle<HESSIAN_PARAMS>([&](int row_i, int col_i) {
                    hessPcache[hessianRowStart<motionType>(row_i) + (col_i - row_i)] += jac[row_i] * jac[col_i];
                });
                constexprFor<HESSIAN_PARAMS>([&](int elem) {
                    iprojCache[elem] += jac[elem] * sampleVal;
                    tprojCache[elem] += jac[elem] * refVal;
                    projSubCache[elem] += jac[elem] * fVal;
                });
                correlations[stripeIdx] += sampleVal * refVal;
            }

            constexprForUpperTriangle<HESSIAN_PARAMS>([&](int row, int col) {
                hessPs[stripeIdx](row, col) += hessPcache[hessianRowStart<motionType>(row) + (col - row)];
            });
            constexprFor<HESSIAN_PARAMS>([&](int elem) {
                iprojs[stripeIdx][elem] += iprojCache[elem];
                tprojs[stripeIdx][elem] += tprojCache[elem];
                projSubs[stripeIdx][elem] += projSubCache[elem];
            });
        }
    });
    double sampMaskedSum = 0;
    double refMaskedSum = 0;
    double correlation = 0;
    sampSum = sampSqSum = refSum = refSqSum = nz = 0;

    for (int stripeIdx = 0; stripeIdx < stripesAmount; stripeIdx++) {
        correlation += correlations[stripeIdx];
        sampSum += sampSums[stripeIdx];
        sampSqSum += sampSqSums[stripeIdx];
        refSum += refSums[stripeIdx];
        refSqSum += refSqSums[stripeIdx];
        sampMaskedSum += sampMaskedSums[stripeIdx];
        refMaskedSum += refMaskedSums[stripeIdx];
        nz += nzs[stripeIdx];
    }
    double scale = nz == 0 ? 0. : 1. / nz;
    double sampMean = sampSum * scale;
    double refMean = refSum * scale;
    correlation += nz * sampMean * refMean - sampMaskedSum * refMean - refMaskedSum * sampMean;
    double* hessPtr = hessian.ptr<double>(0);
    double* sampleProjPtr = sampleProj.ptr<double>(0);
    double* refProjPtr = refProj.ptr<double>(0);
    for (int stripeIdx = 0; stripeIdx < stripesAmount; stripeIdx++) {
        for (int hessNum = 0; hessNum < HESSIAN_PARAMS * HESSIAN_PARAMS; hessNum++) {
            hessPtr[hessNum] += hessPs[stripeIdx].val[hessNum];
        }
        for (int projNum = 0; projNum < HESSIAN_PARAMS; projNum++) {
            sampleProjPtr[projNum] += iprojs[stripeIdx][projNum] - projSubs[stripeIdx][projNum] * sampMean;
            refProjPtr[projNum] += tprojs[stripeIdx][projNum] - projSubs[stripeIdx][projNum] * refMean;
        }
    }
    constexprForUpperTriangle<HESSIAN_PARAMS>([&](int row, int col) {
        hessPtr[col * HESSIAN_PARAMS + row] = hessPtr[row * HESSIAN_PARAMS + col];
    });
    return correlation;
#undef HESSIAN_PARAMS
}

static void updateWarpingMatrixECC(Mat& map_matrix, const Mat& update, const int motionType) {
    CV_Assert(map_matrix.type() == CV_64FC1);
    CV_Assert(update.type() == CV_64FC1);

    CV_Assert(motionType == MOTION_TRANSLATION || motionType == MOTION_EUCLIDEAN || motionType == MOTION_AFFINE ||
              motionType == MOTION_HOMOGRAPHY);

    if (motionType == MOTION_HOMOGRAPHY)
        CV_Assert(map_matrix.rows == 3 && update.rows == 8);
    else if (motionType == MOTION_AFFINE)
        CV_Assert(map_matrix.rows == 2 && update.rows == 6);
    else if (motionType == MOTION_EUCLIDEAN)
        CV_Assert(map_matrix.rows == 2 && update.rows == 3);
    else
        CV_Assert(map_matrix.rows == 2 && update.rows == 2);

    CV_Assert(update.cols == 1);

    CV_Assert(map_matrix.isContinuous());
    CV_Assert(update.isContinuous());

    double* mapPtr = map_matrix.ptr<double>(0);
    const double* updatePtr = update.ptr<double>(0);

    if (motionType == MOTION_TRANSLATION) {
        mapPtr[2] += updatePtr[0];
        mapPtr[5] += updatePtr[1];
    }
    if (motionType == MOTION_AFFINE) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[1] += updatePtr[2];
        mapPtr[4] += updatePtr[3];
        mapPtr[2] += updatePtr[4];
        mapPtr[5] += updatePtr[5];
    }
    if (motionType == MOTION_HOMOGRAPHY) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[6] += updatePtr[2];
        mapPtr[1] += updatePtr[3];
        mapPtr[4] += updatePtr[4];
        mapPtr[7] += updatePtr[5];
        mapPtr[2] += updatePtr[6];
        mapPtr[5] += updatePtr[7];
    }
    if (motionType == MOTION_EUCLIDEAN) {
        double new_theta = updatePtr[0];
        new_theta += asin(mapPtr[3]);

        mapPtr[2] += updatePtr[1];
        mapPtr[5] += updatePtr[2];
        mapPtr[0] = mapPtr[4] = cos(new_theta);
        mapPtr[3] = sin(new_theta);
        mapPtr[1] = -mapPtr[3];
    }
}

static void optimizeECC(Mat& sampleWithGrad,
                 const Mat& reference,
                 Mat& map,
                 int motionType,
                 double* rho,
                 double* lastRho,
                 int deltaY,
                 int nparams,
                 int interpolation) {
    CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR);

    // warp-back portion of the inputImage and gradients to the coordinate space of the referenceFloat
    double correlation = 0;

    // matrices needed for solving linear equation system for maximizing ECC
    Mat hessian = Mat(nparams, nparams, CV_64F);
    Mat hessianInv = Mat(nparams, nparams, CV_64F);
    Mat sampleProjection = Mat(nparams, 1, CV_64F);
    Mat referenceProjection = Mat(nparams, 1, CV_64F);
    Mat sampleProjectionHessian = Mat(nparams, 1, CV_64F);
    Mat errorProjection = Mat(nparams, 1, CV_64F);
    Mat deltaP = Mat(nparams, 1, CV_64F);

    double sampSum;
    double sampSqSum;
    double referenceSum;
    double referenceSqSum;
    int nz;

    {  // if(imageWithGrad.type() == CV_32FC4)
        if (motionType == MOTION_TRANSLATION) {
            correlation = imageHessianProjECC<MOTION_TRANSLATION, float>(map,
                                                                       sampleWithGrad,
                                                                       reference,
                                                                       sampSum,
                                                                       sampSqSum,
                                                                       referenceSum,
                                                                       referenceSqSum,
                                                                       nz,
                                                                       hessian,
                                                                       sampleProjection,
                                                                       referenceProjection,
                                                                       deltaY,
                                                                       interpolation);
        } else if (motionType == MOTION_EUCLIDEAN) {
            correlation = imageHessianProjECC<MOTION_EUCLIDEAN, float>(map,
                                                                       sampleWithGrad,
                                                                       reference,
                                                                       sampSum,
                                                                       sampSqSum,
                                                                       referenceSum,
                                                                       referenceSqSum,
                                                                       nz,
                                                                       hessian,
                                                                       sampleProjection,
                                                                       referenceProjection,
                                                                       deltaY,
                                                                       interpolation);
        } else if (motionType == MOTION_AFFINE) {
            correlation = imageHessianProjECC<MOTION_AFFINE, float>(map,
                                                                       sampleWithGrad,
                                                                       reference,
                                                                       sampSum,
                                                                       sampSqSum,
                                                                       referenceSum,
                                                                       referenceSqSum,
                                                                       nz,
                                                                       hessian,
                                                                       sampleProjection,
                                                                       referenceProjection,
                                                                       deltaY,
                                                                       interpolation);
        } else {
            correlation = imageHessianProjECC<MOTION_HOMOGRAPHY, float>(map,
                                                                        sampleWithGrad,
                                                                        reference,
                                                                        sampSum,
                                                                        sampSqSum,
                                                                        referenceSum,
                                                                        referenceSqSum,
                                                                        nz,
                                                                        hessian,
                                                                        sampleProjection,
                                                                        referenceProjection,
                                                                        deltaY,
                                                                        interpolation);
        }
    }
    double scale = nz == 0 ? 0. : 1. / nz;
    double sampMean = sampSum * scale;
    double refMean = referenceSum * scale;
    double sampStd = std::sqrt(std::max(sampSqSum * scale - sampMean * sampMean, 0.));
    double refStd = std::sqrt(std::max(referenceSqSum * scale - refMean * refMean, 0.));

    // inverse of Hessian
    hessianInv = hessian.inv();
    // calculate enhanced correlation coefficient (ECC)->rho
    *lastRho = *rho;
    double refNorm = std::sqrt(nz * refStd * refStd);
    double sampNorm = std::sqrt(nz * sampStd * sampStd);

    *rho = correlation / (sampNorm * refNorm);
    if ((bool)cvIsNaN(*rho)) {
        CV_Error(Error::StsNoConv, "NaN encountered.");
    }

    // calculate the parameter lambda to account for illumination variation
    sampleProjectionHessian = hessianInv * sampleProjection;
    const double lambdaN = (sampNorm * sampNorm) - sampleProjection.dot(sampleProjectionHessian);
    const double lambdaD = correlation - referenceProjection.dot(sampleProjectionHessian);

    if (lambdaD <= 0.0) {
        CV_Error(Error::StsNoConv, "The algorithm stopped before its convergence. The correlation is going to be minimized. "
            "Images may be uncorrelated or non-overlapped");
    }
    const double lambda = (lambdaN / lambdaD);

    // estimate the update step delta_p
    errorProjection = lambda * referenceProjection - sampleProjection;
    gemm(hessianInv, errorProjection, 1., noArray(), 0., deltaP);

    // update warping matrix
    updateWarpingMatrixECC(map, deltaP, motionType);
}

static Mat prepareGradients(const Mat& sample) {
    CV_Assert(sample.type() == CV_32FC2 || sample.type() == CV_16FC2);

    const int ws = sample.cols;
    const int hs = sample.rows;

    Mat sampleWithGrad;
    int ntasks = std::min(4, hs);

    {
        sampleWithGrad = Mat(hs, ws, CV_32FC4);
        float* dstPtr = sampleWithGrad.ptr<float>();
        parallel_for_(Range(0, ntasks), [&](const Range& range) {
            int rowstart = range.start * hs / ntasks;
            int rowend = range.end * hs / ntasks;
            for (int row = rowstart; row < rowend; row++) {
                const float* sampleCurLine = sample.ptr<float>(row);
                const float* samplePrevLine = sample.ptr<float>(std::max(row - 1, 0));
                const float* sampleNextLine = sample.ptr<float>(std::min(row + 1, hs - 1));
                float gradDivY = (row > 0 && row + 1 < hs) ? 0.5f : 0.25f;
                int col = 0;
                for (; col < ws; col++) {
                    int prevCol = std::max(col - 1, 0);
                    int nextCol = std::min(col + 1, ws - 1);
                    float gradDivX = (col > 0 && col + 1 < ws) ? 0.5f : 0.25f;
                    dstPtr[row * ws * 4 + col * 4] = sampleCurLine[2 * col];
                    dstPtr[row * ws * 4 + col * 4 + 1] =
                        gradDivX * (sampleCurLine[2 * nextCol] - sampleCurLine[2 * prevCol]);
                    dstPtr[row * ws * 4 + col * 4 + 2] = gradDivY * (sampleNextLine[2 * col] - samplePrevLine[2 * col]);
                    dstPtr[row * ws * 4 + col * 4 + 3] = sampleCurLine[2 * col + 1];
                }
            }
        }, ntasks);
    }
    return sampleWithGrad;
}

static void buildPyramidECC(InputArray inputImage,
                  MatPyramid& imgPyramid,
                  InputArray& mask,
                  MatPyramid& maskPyramid,
                  int nlevels) {
    imgPyramid.resize(nlevels);
    inputImage.getMat().convertTo(imgPyramid[0], CV_8UC1);
    maskPyramid.resize(nlevels);
    if (!mask.empty()) {
        mask.getMat().convertTo(maskPyramid[0], CV_8UC1);
    }
    for (int pyrLevel = 0; pyrLevel < nlevels - 1; ++pyrLevel) {
        Size size = Size((imgPyramid[pyrLevel].cols + 1) / 2, (imgPyramid[pyrLevel].rows + 1) / 2);
        pyrDown(imgPyramid[pyrLevel], imgPyramid[pyrLevel + 1], size);
        if (!mask.empty()) {
            pyrDown(maskPyramid[pyrLevel], maskPyramid[pyrLevel + 1], size);
            threshold(maskPyramid[pyrLevel + 1], maskPyramid[pyrLevel + 1], 254, 0xff, THRESH_BINARY);
        }
    }
}

static Mat spliceWithMask(const Mat& image, const Mat& mask) {
    CV_Assert(image.type() == CV_32F && (mask.empty() || mask.type() == CV_8U));
    if (!mask.empty() && image.size() != mask.size()) {
        CV_Error(Error::BadImageSize, "spliceWithMask: Mask and image have to be of same size.");
    }
    const int hs = image.rows;
    const int ws = image.cols;

    Mat result;
    int ntasks = std::min(4, hs);
    {
        union conv_ {
            uint32_t valU;
            float val;
            conv_() : valU(0xffffffff) {}
        } conv;
        result = Mat(hs, ws, CV_32FC2);
        parallel_for_(Range(0, ntasks), [&](const Range& range) {
            int rowstart = range.start * hs / ntasks;
            int rowend = range.end * hs / ntasks;
            for (int row = rowstart; row < rowend; row++) {
                float* dstPtr = result.ptr<float>(row);
                const float* srcPtr = image.ptr<float>(row);
                const uint8_t* maskPtr = !mask.empty() ? mask.ptr<uint8_t>(row) : nullptr;
                int col = 0;
                for (; col < ws; col++) {
                    dstPtr[col * 2] = srcPtr[col];
                    dstPtr[col * 2 + 1] = (!maskPtr || maskPtr[col]) ? conv.val : 0;
                }
            }
        }, ntasks);
    }
    return result;
}

static void scaleWarpMatrix(Mat& warpMatrix, float scale) {
    if (warpMatrix.rows == 3) {
        Mat invertScaleMat = Mat(3, 3, CV_64F, 0.f);
        invertScaleMat.at<double>(0, 0) = 1.f / scale;
        invertScaleMat.at<double>(1, 1) = 1.f / scale;
        invertScaleMat.at<double>(2, 2) = 1.f;
        Mat scaleMatrix = invertScaleMat.clone();
        scaleMatrix.at<double>(0, 0) = scale;
        scaleMatrix.at<double>(1, 1) = scale;
        gemm(warpMatrix, invertScaleMat, 1., noArray(), 0., warpMatrix);
        gemm(scaleMatrix, warpMatrix, 1., noArray(), 0., warpMatrix);
        // Normalization, internal algorithms assumes, that a22 = 1.0f
        for (int mel = 0; mel < 8; mel++) {
            (reinterpret_cast<double*>(warpMatrix.data))[mel] /=
                (reinterpret_cast<double*>(warpMatrix.data))[8];
        }
        (reinterpret_cast<double*>(warpMatrix.data))[8] = 1.f;
    } else {
        warpMatrix.at<double>(0, 2) *= scale;
        warpMatrix.at<double>(1, 2) *= scale;
    }
}

static void checkParams(const MatPyramid& referencePyramid,
                 const MatPyramid& samplePyramid,
                 Mat& map,
                 std::vector<int>& itersPerLevel,
                 const ECCParameters& eccParams) {
    if (itersPerLevel.empty()) {
        itersPerLevel.resize(eccParams.nlevels, eccParams.criteria.maxCount);
    }
    CV_Assert(eccParams.interpolation == INTER_NEAREST || eccParams.interpolation == INTER_LINEAR);
    CV_Assert(static_cast<int>(itersPerLevel.size()) == eccParams.nlevels);
    for (const auto& lvl : referencePyramid) {
        CV_Assert(!lvl.empty() && lvl.type() == referencePyramid[0].type());
    }
    CV_Assert(!samplePyramid.empty());
    for (const auto& lvl : samplePyramid) {
        CV_Assert(!lvl.empty() && lvl.type() == samplePyramid[0].type());
    }
    CV_Assert(samplePyramid.size() == referencePyramid.size() && samplePyramid.size() == itersPerLevel.size());
    CV_Assert(referencePyramid.back().rows > 1 && referencePyramid.back().cols > 1 &&
              samplePyramid.back().rows > 1 && samplePyramid.back().cols > 1);
    // If the user passed an un-initialized warpMatrix, initialize to identity
    if (referencePyramid[0].type() != CV_32FC2 && referencePyramid[0].type() != CV_16FC2) {
        CV_Error(Error::StsError, "Reference pyramid have to be prepared via prepareReferencePyramid function");
    }
    // accept only 1-channel images
    CV_Assert(samplePyramid[0].type() == CV_32FC2 || samplePyramid[0].type() != CV_16FC2);
    CV_Assert(map.type() == CV_64FC1);
    if (map.cols != 3 || (map.rows != 2 && map.rows != 3)) {
        CV_Error(Error::BadImageSize, "warpMatrix has incorrect size");
    }

    if (eccParams.motionType != MOTION_TRANSLATION && eccParams.motionType != MOTION_EUCLIDEAN &&
        eccParams.motionType != MOTION_AFFINE && eccParams.motionType != MOTION_HOMOGRAPHY) {
        CV_Error(Error::StsError, "Incorrect motion type");
    }

    if (eccParams.motionType == MOTION_HOMOGRAPHY && map.rows != 3) {
        CV_Error(Error::BadImageSize, "warpMatrix has incorrect size");
    }

    if (!((bool)(eccParams.criteria.type & TermCriteria::COUNT) || (bool)(eccParams.criteria.type & TermCriteria::EPS))) {
        CV_Error(Error::StsError, "Incorrect stop eccParams.criteria");
    }
}

static MatPyramid prepareECCPyramid(InputArray image,
                             InputArray imageMask,  // Can be empty
                             int gaussFiltSize,
                             int nlevels) {
    MatPyramid imagePyramid, maskPyramid;
    buildPyramidECC(image, imagePyramid, imageMask, maskPyramid, nlevels);
    for (int lvl = 0; lvl < nlevels; lvl++) {
        Mat imgFloat;
        imagePyramid[lvl].convertTo(imgFloat, CV_32F, 1. / 255.);
        if (gaussFiltSize != 0) {
            GaussianBlur(imgFloat, imgFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);
        }
        imagePyramid[lvl] = spliceWithMask(
            imgFloat,
            (static_cast<int>(maskPyramid.size()) > lvl && !maskPyramid[lvl].empty()) ? maskPyramid[lvl] : Mat());
    }
    return imagePyramid;
}

double findTransformECCMultiScale(InputArray reference,
                        InputArray sample,
                        InputOutputArray warpMatrixA,
                        const ECCParameters& eccParams,
                        InputArray referenceMask,
                        InputArray sampleMask) {
    MatPyramid referencePyramid = prepareECCPyramid(reference, referenceMask, eccParams.gaussFiltSize, eccParams.nlevels);
    MatPyramid samplePyramid = prepareECCPyramid(sample, sampleMask, eccParams.gaussFiltSize, eccParams.nlevels);
    Mat& warpMatrix = warpMatrixA.getMatRef();
    std::vector<int> itersPerLevelCopy = eccParams.itersPerLevel;
    // If the user passed an un-initialized warpMatrix, initialize to identity
    if (warpMatrix.empty())
    {
        int rowCount = eccParams.motionType == MOTION_HOMOGRAPHY ? 3 : 2;
        warpMatrix = Mat::eye(rowCount, 3, CV_64FC1);
    }
    int warpMatrixType = warpMatrix.type();
    if (warpMatrixType != CV_64FC1)
    {
        warpMatrix.convertTo(warpMatrix, CV_64FC1);
    }

    checkParams(referencePyramid,
                samplePyramid,
                warpMatrix,
                itersPerLevelCopy,
                eccParams);

    int nparams = 0;
    switch (eccParams.motionType) {
        case MOTION_TRANSLATION:
            nparams = MotionTraits<MOTION_TRANSLATION>::paramAmount;
            break;
        case MOTION_EUCLIDEAN:
            nparams = MotionTraits<MOTION_EUCLIDEAN>::paramAmount;
            break;
        case MOTION_AFFINE:
            nparams = MotionTraits<MOTION_AFFINE>::paramAmount;
            break;
        case MOTION_HOMOGRAPHY:
            nparams = MotionTraits<MOTION_HOMOGRAPHY>::paramAmount;
            break;
        default:
            CV_Error(Error::StsBadArg, "Incorrect motion type");
    }

    const std::vector<int> numberOfIterations = ((eccParams.criteria.type & TermCriteria::COUNT) != 0)
                                                    ? itersPerLevelCopy
                                                    : std::vector<int>(eccParams.nlevels, 200);
    const double terminationEPS = (bool)(eccParams.criteria.type & TermCriteria::EPS) ? eccParams.criteria.epsilon : -1;

    // Scale warp matrix multiple times to lower pyramid level
    for (int pyrLevel = 0; pyrLevel < eccParams.nlevels - 1; pyrLevel++) {
        scaleWarpMatrix(warpMatrix, 0.5);
    }
    double rho = -1;
    for (int pyrLevel = eccParams.nlevels - 1; pyrLevel >= 0; --pyrLevel) {
        const int hr = referencePyramid[pyrLevel].rows;

        Mat sampleWithGrad = prepareGradients(samplePyramid[pyrLevel]);

        const int LOW_SIZE = 200;
        int deltaY = hr < LOW_SIZE ? 1 : 2;

        // iteratively update mapMatrix
        double lastRho = -terminationEPS;
        for (int i = 1; (i <= numberOfIterations[pyrLevel]) && (fabs(rho - lastRho) >= terminationEPS); i++) {
            optimizeECC(sampleWithGrad, referencePyramid[pyrLevel], warpMatrix, eccParams.motionType, &rho, &lastRho, deltaY, nparams, eccParams.interpolation);
        }
        if (pyrLevel > 0) {
            scaleWarpMatrix(warpMatrix, 2);
        }
    }
    if(warpMatrixType != CV_64FC1)
    {
        warpMatrix.convertTo(warpMatrix, warpMatrixType);
    }
    // return final correlation coefficient
    return rho;
}
};
/* End of file. */