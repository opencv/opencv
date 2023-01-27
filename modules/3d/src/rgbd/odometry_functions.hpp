// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_ODOMETRY_FUNCTIONS_HPP
#define OPENCV_3D_ODOMETRY_FUNCTIONS_HPP

#include "utils.hpp"
#include <opencv2/imgproc.hpp>

namespace cv
{
enum class OdometryTransformType
{
    // rotation, translation, rotation+translation
    ROTATION = 1, TRANSLATION = 2, RIGID_TRANSFORMATION = 4
};

static inline int getTransformDim(OdometryTransformType transformType)
{
    switch(transformType)
    {
    case OdometryTransformType::RIGID_TRANSFORMATION:
        return 6;
    case OdometryTransformType::ROTATION:
    case OdometryTransformType::TRANSLATION:
        return 3;
    default:
        CV_Error(Error::StsBadArg, "Incorrect transformation type");
    }
}


static inline
Vec6d calcRgbdEquationCoeffs(double dIdx, double dIdy, const Point3d& p3d, double fx, double fy)
{
    double invz = 1. / p3d.z,
           v0 = dIdx * fx * invz,
           v1 = dIdy * fy * invz,
           v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;
    Point3d v(v0, v1, v2);
    Point3d pxv = p3d.cross(v);

    return Vec6d(pxv.x, pxv.y, pxv.z, v0, v1, v2);
}

static inline
Vec3d calcRgbdEquationCoeffsRotation(double dIdx, double dIdy, const Point3d& p3d, double fx, double fy)
{
    double invz = 1. / p3d.z,
           v0 = dIdx * fx * invz,
           v1 = dIdy * fy * invz,
           v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;

    Point3d v(v0, v1, v2);
    Point3d pxv = p3d.cross(v);

    return Vec3d(pxv);
}

static inline
Vec3d calcRgbdEquationCoeffsTranslation(double dIdx, double dIdy, const Point3d& p3d, double fx, double fy)
{
    double invz = 1. / p3d.z,
           v0 = dIdx * fx * invz,
           v1 = dIdy * fy * invz,
           v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;

    return Vec3d(v0, v1, v2);
}

static inline void rgbdCoeffsFunc(OdometryTransformType transformType,
                                  double* C, double dIdx, double dIdy, const Point3d& p3d, double fx, double fy)
{
    int dim = getTransformDim(transformType);
    Vec6d ret;
    switch(transformType)
    {
    case OdometryTransformType::RIGID_TRANSFORMATION:
    {
        ret = calcRgbdEquationCoeffs(dIdx, dIdy, p3d, fx, fy);
        break;
    }
    case OdometryTransformType::ROTATION:
    {
        Vec3d r = calcRgbdEquationCoeffsRotation(dIdx, dIdy, p3d, fx, fy);
        ret = Vec6d(r[0], r[1], r[2], 0, 0, 0);
        break;
    }
    case OdometryTransformType::TRANSLATION:
    {
        Vec3d r = calcRgbdEquationCoeffsTranslation(dIdx, dIdy, p3d, fx, fy);
        ret = Vec6d(r[0], r[1], r[2], 0, 0, 0);
        break;
    }
    default:
        CV_Error(Error::StsBadArg, "Incorrect transformation type");
    }
    for (int i = 0; i < dim; i++)
        C[i] = ret[i];
}


static inline
Vec6d calcICPEquationCoeffs(const Point3d& psrc, const Vec3d& ndst)
{
    Point3d pxv = psrc.cross(Point3d(ndst));

    return Vec6d(pxv.x, pxv.y, pxv.z, ndst[0], ndst[1], ndst[2]);
}

static inline
Vec3d calcICPEquationCoeffsRotation(const Point3d& psrc, const Vec3d& ndst)
{
    Point3d pxv = psrc.cross(Point3d(ndst));

    return Vec3d(pxv);
}

static inline
Vec3d calcICPEquationCoeffsTranslation( const Point3d& /*p0*/, const Vec3d& ndst)
{
    return Vec3d(ndst);
}

static inline
void icpCoeffsFunc(OdometryTransformType transformType, double* C, const Point3d& p0, const Point3d& /*p1*/, const Vec3d& n1)
{
    int dim = getTransformDim(transformType);
    Vec6d ret;
    switch(transformType)
    {
    case OdometryTransformType::RIGID_TRANSFORMATION:
    {
        ret = calcICPEquationCoeffs(p0, n1);
        break;
    }
    case OdometryTransformType::ROTATION:
    {
        Vec3d r = calcICPEquationCoeffsRotation(p0, n1);
        ret = Vec6d(r[0], r[1], r[2], 0, 0, 0);
        break;
    }
    case OdometryTransformType::TRANSLATION:
    {
        Vec3d r = calcICPEquationCoeffsTranslation(p0, n1);
        ret = Vec6d(r[0], r[1], r[2], 0, 0, 0);
        break;
    }
    default:
        CV_Error(Error::StsBadArg, "Incorrect transformation type");
    }
    for (int i = 0; i < dim; i++)
        C[i] = ret[i];
}

void prepareRGBDFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, Ptr<RgbdNormals>& normalsComputer, const OdometrySettings settings, OdometryAlgoType algtype);
void prepareRGBFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, OdometrySettings settings);
void prepareICPFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, Ptr<RgbdNormals>& normalsComputer, const OdometrySettings settings, OdometryAlgoType algtype);

bool RGBDICPOdometryImpl(OutputArray _Rt, const Mat& initRt,
                         const OdometryFrame& srcFrame,
                         const OdometryFrame& dstFrame,
                         const Matx33f& cameraMatrix,
                         float maxDepthDiff, float angleThreshold, const std::vector<int>& iterCounts,
                         double maxTranslation, double maxRotation, double sobelScale,
                         OdometryType method, OdometryTransformType transfromType, OdometryAlgoType algtype);

void computeCorresps(const Matx33f& _K, const Mat& Rt,
                     const Mat& image0, const Mat& depth0, const Mat& validMask0,
                     const Mat& image1, const Mat& depth1, const Mat& selectMask1, float maxDepthDiff,
                     Mat& _corresps, Mat& _diffs, double& _sigma, OdometryType method);

void calcRgbdLsmMatrices(const Mat& cloud0, const Mat& Rt,
                         const Mat& dI_dx1, const Mat& dI_dy1,
                         const Mat& corresps, const Mat& diffs, const double sigma,
                         double fx, double fy, double sobelScaleIn,
                         Mat& AtA, Mat& AtB, OdometryTransformType transformType);

void calcICPLsmMatrices(const Mat& cloud0, const Mat& Rt,
                        const Mat& cloud1, const Mat& normals1,
                        const Mat& corresps,
                        Mat& AtA, Mat& AtB, OdometryTransformType transformType);

void calcICPLsmMatricesFast(Matx33f cameraMatrix, const UMat& oldPts, const UMat& oldNrm, const UMat& newPts, const UMat& newNrm,
                            cv::Affine3f pose, int level, float maxDepthDiff, float angleThreshold, cv::Matx66f& A, cv::Vec6f& b);

#ifdef HAVE_OPENCL
bool ocl_calcICPLsmMatricesFast(Matx33f cameraMatrix, const UMat& oldPts, const UMat& oldNrm, const UMat& newPts, const UMat& newNrm,
                                cv::Affine3f pose, int level, float maxDepthDiff, float angleThreshold, cv::Matx66f& A, cv::Vec6f& b);
#endif

void computeProjectiveMatrix(const Mat& ksi, Mat& Rt);

bool solveSystem(const Mat& AtA, const Mat& AtB, double detThreshold, Mat& x);

bool testDeltaTransformation(const Mat& deltaRt, double maxTranslation, double maxRotation);

}
#endif //OPENCV_3D_ODOMETRY_FUNCTIONS_HPP
