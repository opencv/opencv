// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_ODOMETRY_FUNCTIONS_HPP
#define OPENCV_3D_ODOMETRY_FUNCTIONS_HPP

#include "../precomp.hpp"
#include "utils.hpp"
#include <opencv2/imgproc.hpp>

namespace cv
{
enum class OdometryTransformType
{
    ROTATION = 1, TRANSLATION = 2, RIGID_TRANSFORMATION = 4
};

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
void calcRgbdEquationCoeffs(double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz = 1. / p3d.z,
        v0 = dIdx * fx * invz,
        v1 = dIdy * fy * invz,
        v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;

    C[0] = -p3d.z * v1 + p3d.y * v2;
    C[1] = p3d.z * v0 - p3d.x * v2;
    C[2] = -p3d.y * v0 + p3d.x * v1;
    C[3] = v0;
    C[4] = v1;
    C[5] = v2;
}

static inline
void calcRgbdEquationCoeffsRotation(double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz = 1. / p3d.z,
        v0 = dIdx * fx * invz,
        v1 = dIdy * fy * invz,
        v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;
    C[0] = -p3d.z * v1 + p3d.y * v2;
    C[1] = p3d.z * v0 - p3d.x * v2;
    C[2] = -p3d.y * v0 + p3d.x * v1;
}

static inline
void calcRgbdEquationCoeffsTranslation(double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz = 1. / p3d.z,
        v0 = dIdx * fx * invz,
        v1 = dIdy * fy * invz,
        v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;
    C[0] = v0;
    C[1] = v1;
    C[2] = v2;
}

typedef
void (*CalcRgbdEquationCoeffsPtr)(double*, double, double, const Point3f&, double, double);

static inline
void calcICPEquationCoeffs(double* C, const Point3f& p0, const Vec3f& n1)
{
    C[0] = -p0.z * n1[1] + p0.y * n1[2];
    C[1] = p0.z * n1[0] - p0.x * n1[2];
    C[2] = -p0.y * n1[0] + p0.x * n1[1];
    C[3] = n1[0];
    C[4] = n1[1];
    C[5] = n1[2];
}

static inline
void calcICPEquationCoeffsRotation(double* C, const Point3f& p0, const Vec3f& n1)
{
    C[0] = -p0.z * n1[1] + p0.y * n1[2];
    C[1] = p0.z * n1[0] - p0.x * n1[2];
    C[2] = -p0.y * n1[0] + p0.x * n1[1];
}

static inline
void calcICPEquationCoeffsTranslation(double* C, const Point3f& /*p0*/, const Vec3f& n1)
{
    C[0] = n1[0];
    C[1] = n1[1];
    C[2] = n1[2];
}

typedef
void (*CalcICPEquationCoeffsPtr)(double*, const Point3f&, const Vec3f&);

void prepareRGBDFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, OdometrySettings settings, OdometryAlgoType algtype);
void prepareRGBFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, OdometrySettings settings);
void prepareICPFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, OdometrySettings settings, OdometryAlgoType algtype);

void prepareRGBFrameBase(OdometryFrame& frame, OdometrySettings settings);
void prepareRGBFrameSrc (OdometryFrame& frame, OdometrySettings settings);
void prepareRGBFrameDst (OdometryFrame& frame, OdometrySettings settings);

void prepareICPFrameBase(OdometryFrame& frame, OdometrySettings settings);
void prepareICPFrameSrc (OdometryFrame& frame, OdometrySettings settings);
void prepareICPFrameDst (OdometryFrame& frame, OdometrySettings settings);


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

void computeCorresps(const Matx33f& _K, const Matx33f& _K_inv, const Mat& Rt,
    const Mat& image0, const Mat& depth0, const Mat& validMask0,
    const Mat& image1, const Mat& depth1, const Mat& selectMask1, float maxDepthDiff,
    Mat& _corresps, Mat& _diffs, double& _sigma, OdometryType method);

void calcRgbdLsmMatrices(const Mat& cloud0, const Mat& Rt,
    const Mat& dI_dx1, const Mat& dI_dy1,
    const Mat& corresps, const Mat& diffs, const double sigma,
    double fx, double fy, double sobelScaleIn,
    Mat& AtA, Mat& AtB, CalcRgbdEquationCoeffsPtr func, int transformDim);

void calcICPLsmMatrices(const Mat& cloud0, const Mat& Rt,
    const Mat& cloud1, const Mat& normals1,
    const Mat& corresps,
    Mat& AtA, Mat& AtB, CalcICPEquationCoeffsPtr func, int transformDim);

void calcICPLsmMatricesFast(Matx33f cameraMatrix, const Mat& oldPts, const Mat& oldNrm, const Mat& newPts, const Mat& newNrm,
    cv::Affine3f pose, int level, float maxDepthDiff, float angleThreshold, cv::Matx66f& A, cv::Vec6f& b);

#ifdef HAVE_OPENCL
void calcICPLsmMatricesFast_ocl(Matx33f cameraMatrix, const UMat& oldPts, const UMat& oldNrm, const UMat& newPts, const UMat& newNrm,
    cv::Affine3f pose, int level, float maxDepthDiff, float angleThreshold, cv::Matx66f& A, cv::Vec6f& b);
#endif

void computeProjectiveMatrix(const Mat& ksi, Mat& Rt);

bool solveSystem(const Mat& AtA, const Mat& AtB, double detThreshold, Mat& x);

bool testDeltaTransformation(const Mat& deltaRt, double maxTranslation, double maxRotation);

}
#endif //OPENCV_3D_ODOMETRY_FUNCTIONS_HPP
