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
void checkImage(InputArray image)
{
    if (image.empty())
        CV_Error(Error::StsBadSize, "Image is empty.");
    if (image.type() != CV_8UC1)
        CV_Error(Error::StsBadSize, "Image type has to be CV_8UC1.");
}

static inline
void checkDepth(InputArray depth, const Size& imageSize)
{
    if (depth.empty())
        CV_Error(Error::StsBadSize, "Depth is empty.");
    if (depth.size() != imageSize)
        CV_Error(Error::StsBadSize, "Depth has to have the size equal to the image size.");
    if (depth.type() != CV_32FC1)
        CV_Error(Error::StsBadSize, "Depth type has to be CV_32FC1.");
}

static inline
void checkMask(InputArray mask, const Size& imageSize)
{
    if (!mask.empty())
    {
        if (mask.size() != imageSize)
            CV_Error(Error::StsBadSize, "Mask has to have the size equal to the image size.");
        if (mask.type() != CV_8UC1)
            CV_Error(Error::StsBadSize, "Mask type has to be CV_8UC1.");
    }
}

static inline
void checkNormals(InputArray normals, const Size& depthSize)
{
    if (normals.size() != depthSize)
        CV_Error(Error::StsBadSize, "Normals has to have the size equal to the depth size.");
    if (normals.type() != CV_32FC3)
        CV_Error(Error::StsBadSize, "Normals type has to be CV_32FC3.");
}


static inline
Vec6d calcRgbdEquationCoeffs(double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
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
Vec3d calcRgbdEquationCoeffsRotation(double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
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
Vec3d calcRgbdEquationCoeffsTranslation(double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz = 1. / p3d.z,
           v0 = dIdx * fx * invz,
           v1 = dIdy * fy * invz,
           v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;

    return Vec3d(v0, v1, v2);
}

static inline void rgbdCoeffsFunc(OdometryTransformType transformType,
                                  double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
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
Vec6d calcICPEquationCoeffs(const Point3f& psrc, const Vec3f& ndst)
{
    Point3d pxv = psrc.cross(Point3d(ndst));

    return Vec6d(pxv.x, pxv.y, pxv.z, ndst[0], ndst[1], ndst[2]);
}

static inline
Vec3d calcICPEquationCoeffsRotation(const Point3f& psrc, const Vec3f& ndst)
{
    Point3d pxv = psrc.cross(Point3d(ndst));

    return Vec3d(pxv);
}

static inline
Vec3d calcICPEquationCoeffsTranslation( const Point3f& /*p0*/, const Vec3f& ndst)
{
    return Vec3d(ndst);
}

static inline
void icpCoeffsFunc(OdometryTransformType transformType, double* C, const Point3f& p0, const Point3f& /*p1*/, const Vec3f& n1)
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

void prepareRGBDFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, const OdometrySettings settings, OdometryAlgoType algtype);
void prepareRGBFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, const OdometrySettings settings, bool useDepth);
void prepareICPFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, const OdometrySettings settings, OdometryAlgoType algtype);

void prepareRGBFrameBase(OdometryFrame& frame, const OdometrySettings settings, bool useDepth);
void prepareRGBFrameSrc (OdometryFrame& frame, const OdometrySettings settings);
void prepareRGBFrameDst (OdometryFrame& frame, const OdometrySettings settings);

void prepareICPFrameBase(OdometryFrame& frame, const OdometrySettings settings);
void prepareICPFrameSrc (OdometryFrame& frame, const OdometrySettings settings);
void prepareICPFrameDst (OdometryFrame& frame, const OdometrySettings settings);


void setPyramids(OdometryFrame& odf, OdometryFramePyramidType oftype, InputArrayOfArrays pyramidImage);
void getPyramids(OdometryFrame& odf, OdometryFramePyramidType oftype, OutputArrayOfArrays _pyramid);

void preparePyramidImage(InputArray image, InputOutputArrayOfArrays pyramidImage, size_t levelCount);

template<typename TMat>
void preparePyramidMask(InputArray mask, InputArrayOfArrays pyramidDepth, float minDepth, float maxDepth, InputArrayOfArrays pyramidNormal, InputOutputArrayOfArrays pyramidMask);

template<typename TMat>
void preparePyramidCloud(InputArrayOfArrays pyramidDepth, const Matx33f& cameraMatrix, InputOutputArrayOfArrays pyramidCloud);

void buildPyramidCameraMatrix(const Matx33f& cameraMatrix, int levels, std::vector<Matx33f>& pyramidCameraMatrix);

template<typename TMat>
void preparePyramidSobel(InputArrayOfArrays pyramidImage, int dx, int dy, InputOutputArrayOfArrays pyramidSobel, int sobelSize);

void preparePyramidTexturedMask(InputArrayOfArrays pyramid_dI_dx, InputArrayOfArrays pyramid_dI_dy,
                                InputArray minGradMagnitudes, InputArrayOfArrays pyramidMask, double maxPointsPart,
                                InputOutputArrayOfArrays pyramidTexturedMask, double sobelScale);

void randomSubsetOfMask(InputOutputArray _mask, float part);

void preparePyramidNormals(InputArray normals, InputArrayOfArrays pyramidDepth, InputOutputArrayOfArrays pyramidNormals);

void preparePyramidNormalsMask(InputArray pyramidNormals, InputArray pyramidMask, double maxPointsPart,
                               InputOutputArrayOfArrays /*std::vector<Mat>&*/ pyramidNormalsMask);


bool RGBDICPOdometryImpl(OutputArray _Rt, const Mat& initRt,
                         const OdometryFrame srcFrame,
                         const OdometryFrame dstFrame,
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

void calcICPLsmMatricesFast(Matx33f cameraMatrix, const Mat& oldPts, const Mat& oldNrm, const Mat& newPts, const Mat& newNrm,
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
