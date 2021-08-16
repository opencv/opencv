// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// FastICPOdometry is based on kinfu-remake code
// Copyright(c) 2012, Anatoly Baksheev
// All rights reserved.

#include "../precomp.hpp"
#include "utils.hpp"
#include "fast_icp.hpp"

namespace cv
{

enum
{
    RGBD_ODOMETRY = 1,
    ICP_ODOMETRY = 2,
    MERGED_ODOMETRY = RGBD_ODOMETRY + ICP_ODOMETRY
};

static const int sobelSize = 3;
static const double sobelScale = 1./8.;
static const int normalWinSize = 5;
static const RgbdNormals::RgbdNormalsMethod normalMethod = RgbdNormals::RGBD_NORMALS_METHOD_FALS;

static
void buildPyramidCameraMatrix(const Matx33f& cameraMatrix, int levels, std::vector<Matx33f>& pyramidCameraMatrix)
{
    pyramidCameraMatrix.resize(levels);

    for(int i = 0; i < levels; i++)
    {
        Matx33f levelCameraMatrix = (i == 0) ? cameraMatrix : 0.5f * pyramidCameraMatrix[i-1];
        levelCameraMatrix(2, 2) = 1.0;
        pyramidCameraMatrix[i] = levelCameraMatrix;
    }
}

static inline
void checkImage(InputArray image)
{
    if(image.empty())
        CV_Error(Error::StsBadSize, "Image is empty.");
    if(image.type() != CV_8UC1)
        CV_Error(Error::StsBadSize, "Image type has to be CV_8UC1.");
}

static inline
void checkDepth(InputArray depth, const Size& imageSize)
{
    if(depth.empty())
        CV_Error(Error::StsBadSize, "Depth is empty.");
    if(depth.size() != imageSize)
        CV_Error(Error::StsBadSize, "Depth has to have the size equal to the image size.");
    if(depth.type() != CV_32FC1)
        CV_Error(Error::StsBadSize, "Depth type has to be CV_32FC1.");
}

static inline
void checkMask(InputArray mask, const Size& imageSize)
{
    if(!mask.empty())
    {
        if(mask.size() != imageSize)
            CV_Error(Error::StsBadSize, "Mask has to have the size equal to the image size.");
        if(mask.type() != CV_8UC1)
            CV_Error(Error::StsBadSize, "Mask type has to be CV_8UC1.");
    }
}

static inline
void checkNormals(InputArray normals, const Size& depthSize)
{
    if(normals.size() != depthSize)
        CV_Error(Error::StsBadSize, "Normals has to have the size equal to the depth size.");
    if(normals.type() != CV_32FC3)
        CV_Error(Error::StsBadSize, "Normals type has to be CV_32FC3.");
}

static
void preparePyramidImage(InputArray image, InputOutputArrayOfArrays pyramidImage, size_t levelCount)
{
    if(!pyramidImage.empty())
    {
        size_t nLevels = pyramidImage.size(-1).width;
        if(nLevels < levelCount)
            CV_Error(Error::StsBadSize, "Levels count of pyramidImage has to be equal or less than size of iterCounts.");

        CV_Assert(pyramidImage.size(0) == image.size());
        for(size_t i = 0; i < nLevels; i++)
            CV_Assert(pyramidImage.type((int)i) == image.type());
    }
    else
        buildPyramid(image, pyramidImage, (int)levelCount - 1);
}

static
void preparePyramidDepth(InputArray depth, InputOutputArrayOfArrays pyramidDepth, size_t levelCount)
{
    if(!pyramidDepth.empty())
    {
        size_t nLevels = pyramidDepth.size(-1).width;
        if(nLevels < levelCount)
            CV_Error(Error::StsBadSize, "Levels count of pyramidDepth has to be equal or less than size of iterCounts.");

        CV_Assert(pyramidDepth.size(0) == depth.size());
        for(size_t i = 0; i < nLevels; i++)
            CV_Assert(pyramidDepth.type((int)i) == depth.type());
    }
    else
        buildPyramid(depth, pyramidDepth, (int)levelCount - 1);
}

template<typename TMat>
static TMat getTMat(InputArray, int = -1);

template<>
Mat getTMat<Mat>(InputArray a, int i)
{
    return a.getMat(i);
}

template<>
UMat getTMat<UMat>(InputArray a, int i)
{
    return a.getUMat(i);
}

template<typename TMat>
static TMat& getTMatRef(InputOutputArray, int = -1);

template<>
Mat& getTMatRef<Mat>(InputOutputArray a, int i)
{
    return a.getMatRef(i);
}

//TODO: uncomment it when it's in use
//template<>
//UMat& getTMatRef<UMat>(InputOutputArray a, int i)
//{
//    return a.getUMatRef(i);
//}

template<typename TMat>
static
void preparePyramidMask(InputArray mask, InputArrayOfArrays pyramidDepth, float minDepth, float maxDepth,
                        InputArrayOfArrays pyramidNormal,
                        InputOutputArrayOfArrays pyramidMask)
{
    minDepth = std::max(0.f, minDepth);

    int nLevels = pyramidDepth.size(-1).width;
    if(!pyramidMask.empty())
    {
        if(pyramidMask.size(-1).width != nLevels)
            CV_Error(Error::StsBadSize, "Levels count of pyramidMask has to be equal to size of pyramidDepth.");

        for(int i = 0; i < pyramidMask.size(-1).width; i++)
        {
            CV_Assert(pyramidMask.size(i) == pyramidDepth.size(i));
            CV_Assert(pyramidMask.type(i) == CV_8UC1);
        }
    }
    else
    {
        TMat validMask;
        if(mask.empty())
            validMask = TMat(pyramidDepth.size(0), CV_8UC1, Scalar(255));
        else
            validMask = getTMat<TMat>(mask, -1).clone();

        buildPyramid(validMask, pyramidMask, nLevels - 1);

        for(int i = 0; i < pyramidMask.size(-1).width; i++)
        {
            TMat levelDepth = getTMat<TMat>(pyramidDepth, i).clone();
            patchNaNs(levelDepth, 0);

            TMat& levelMask = getTMatRef<TMat>(pyramidMask, i);
            TMat gtmin, ltmax, tmpMask;
            cv::compare(levelDepth, Scalar(minDepth), gtmin, CMP_GT);
            cv::compare(levelDepth, Scalar(maxDepth), ltmax, CMP_LT);
            cv::bitwise_and(gtmin, ltmax, tmpMask);
            cv::bitwise_and(levelMask, tmpMask, levelMask);

            if(!pyramidNormal.empty())
            {
                CV_Assert(pyramidNormal.type(i) == CV_32FC3);
                CV_Assert(pyramidNormal.size(i) == pyramidDepth.size(i));
                TMat levelNormal = getTMat<TMat>(pyramidNormal, i).clone();

                TMat validNormalMask;
                // NaN check
                cv::compare(levelNormal, levelNormal, validNormalMask, CMP_EQ);
                CV_Assert(validNormalMask.type() == CV_8UC3);

                std::vector<TMat> channelMasks;
                split(validNormalMask, channelMasks);
                TMat tmpChMask;
                cv::bitwise_and(channelMasks[0], channelMasks[1], tmpChMask);
                cv::bitwise_and(channelMasks[2], tmpChMask, validNormalMask);
                cv::bitwise_and(levelMask, validNormalMask, levelMask);
            }
        }
    }
}

template<typename TMat>
static
void preparePyramidCloud(InputArrayOfArrays pyramidDepth, const Matx33f& cameraMatrix, InputOutputArrayOfArrays pyramidCloud)
{
    size_t depthSize = pyramidDepth.size(-1).width;
    size_t cloudSize = pyramidCloud.size(-1).width;
    if(!pyramidCloud.empty())
    {
        if(cloudSize != depthSize)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidCloud.");

        for(size_t i = 0; i < depthSize; i++)
        {
            CV_Assert(pyramidCloud.size((int)i) == pyramidDepth.size((int)i));
            CV_Assert(pyramidCloud.type((int)i) == CV_32FC3);
        }
    }
    else
    {
        std::vector<Matx33f> pyramidCameraMatrix;
        buildPyramidCameraMatrix(cameraMatrix, (int)depthSize, pyramidCameraMatrix);

        pyramidCloud.create((int)depthSize, 1, CV_32FC3, -1);
        for(size_t i = 0; i < depthSize; i++)
        {
            TMat cloud;
            depthTo3d(getTMat<TMat>(pyramidDepth, (int)i), pyramidCameraMatrix[i], cloud);
            getTMatRef<TMat>(pyramidCloud, (int)i) = cloud;
        }
    }
}

template<typename TMat>
static
void preparePyramidSobel(InputArrayOfArrays pyramidImage, int dx, int dy, InputOutputArrayOfArrays pyramidSobel)
{
    size_t imgLevels = pyramidImage.size(-1).width;
    size_t sobelLvls = pyramidSobel.size(-1).width;
    if(!pyramidSobel.empty())
    {
        if(sobelLvls != imgLevels)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidSobel.");

        for(size_t i = 0; i < sobelLvls; i++)
        {
            CV_Assert(pyramidSobel.size((int)i) == pyramidImage.size((int)i));
            CV_Assert(pyramidSobel.type((int)i) == CV_16SC1);
        }
    }
    else
    {
        pyramidSobel.create((int)imgLevels, 1, CV_16SC1, -1);
        for(size_t i = 0; i < imgLevels; i++)
        {
            Sobel(getTMat<TMat>(pyramidImage, (int)i), getTMatRef<TMat>(pyramidSobel, (int)i), CV_16S, dx, dy, sobelSize);
        }
    }
}

static
void randomSubsetOfMask(InputOutputArray _mask, float part)
{
    const int minPointsCount = 1000; // minimum point count (we can process them fast)
    const int nonzeros = countNonZero(_mask);
    const int needCount = std::max(minPointsCount, int(_mask.total() * part));
    if(needCount < nonzeros)
    {
        RNG rng;
        Mat mask = _mask.getMat();
        Mat subset(mask.size(), CV_8UC1, Scalar(0));

        int subsetSize = 0;
        while(subsetSize < needCount)
        {
            int y = rng(mask.rows);
            int x = rng(mask.cols);
            if(mask.at<uchar>(y,x))
            {
                subset.at<uchar>(y,x) = 255;
                mask.at<uchar>(y,x) = 0;
                subsetSize++;
            }
        }
        _mask.assign(subset);
    }
}

static
void preparePyramidTexturedMask(InputArrayOfArrays pyramid_dI_dx, InputArrayOfArrays pyramid_dI_dy,
                                InputArray minGradMagnitudes, InputArrayOfArrays pyramidMask, double maxPointsPart,
                                InputOutputArrayOfArrays pyramidTexturedMask)
{
    size_t didxLevels = pyramid_dI_dx.size(-1).width;
    size_t texLevels = pyramidTexturedMask.size(-1).width;
    if(!pyramidTexturedMask.empty())
    {
        if(texLevels != didxLevels)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidTexturedMask.");

        for(size_t i = 0; i < texLevels; i++)
        {
            CV_Assert(pyramidTexturedMask.size((int)i) == pyramid_dI_dx.size((int)i));
            CV_Assert(pyramidTexturedMask.type((int)i) == CV_8UC1);
        }
    }
    else
    {
        CV_Assert(minGradMagnitudes.type() == CV_32F);
        Mat_<float> mgMags = minGradMagnitudes.getMat();

        const float sobelScale2_inv = 1.f / (float)(sobelScale * sobelScale);
        pyramidTexturedMask.create((int)didxLevels, 1, CV_8UC1, -1);
        for(size_t i = 0; i < didxLevels; i++)
        {
            const float minScaledGradMagnitude2 = mgMags((int)i) * mgMags((int)i) * sobelScale2_inv;
            const Mat& dIdx = pyramid_dI_dx.getMat((int)i);
            const Mat& dIdy = pyramid_dI_dy.getMat((int)i);

            Mat texturedMask(dIdx.size(), CV_8UC1, Scalar(0));

            for(int y = 0; y < dIdx.rows; y++)
            {
                const short *dIdx_row = dIdx.ptr<short>(y);
                const short *dIdy_row = dIdy.ptr<short>(y);
                uchar *texturedMask_row = texturedMask.ptr<uchar>(y);
                for(int x = 0; x < dIdx.cols; x++)
                {
                    float magnitude2 = static_cast<float>(dIdx_row[x] * dIdx_row[x] + dIdy_row[x] * dIdy_row[x]);
                    if(magnitude2 >= minScaledGradMagnitude2)
                        texturedMask_row[x] = 255;
                }
            }
            Mat texMask = texturedMask & pyramidMask.getMat((int)i);

            randomSubsetOfMask(texMask, (float)maxPointsPart);
            pyramidTexturedMask.getMatRef((int)i) = texMask;
        }
    }
}

static
void preparePyramidNormals(InputArray normals, InputArrayOfArrays pyramidDepth, InputOutputArrayOfArrays pyramidNormals)
{
    size_t depthLevels = pyramidDepth.size(-1).width;
    size_t normalsLevels = pyramidNormals.size(-1).width;
    if(!pyramidNormals.empty())
    {
        if(normalsLevels != depthLevels)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidNormals.");

        for(size_t i = 0; i < normalsLevels; i++)
        {
            CV_Assert(pyramidNormals.size((int)i) == pyramidDepth.size((int)i));
            CV_Assert(pyramidNormals.type((int)i) == CV_32FC3);
        }
    }
    else
    {
        buildPyramid(normals, pyramidNormals, (int)depthLevels - 1);
        // renormalize normals
        for(size_t i = 1; i < depthLevels; i++)
        {
            Mat& currNormals = pyramidNormals.getMatRef((int)i);
            for(int y = 0; y < currNormals.rows; y++)
            {
                Point3f* normals_row = currNormals.ptr<Point3f>(y);
                for(int x = 0; x < currNormals.cols; x++)
                {
                    double nrm = norm(normals_row[x]);
                    normals_row[x] *= 1./nrm;
                }
            }
        }
    }
}

static
void preparePyramidNormalsMask(InputArray pyramidNormals, InputArray pyramidMask, double maxPointsPart,
                               InputOutputArrayOfArrays /*std::vector<Mat>&*/ pyramidNormalsMask)
{
    size_t maskLevels = pyramidMask.size(-1).width;
    size_t norMaskLevels = pyramidNormalsMask.size(-1).width;
    if(!pyramidNormalsMask.empty())
    {
        if(norMaskLevels != maskLevels)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidNormalsMask.");

        for(size_t i = 0; i < norMaskLevels; i++)
        {
            CV_Assert(pyramidNormalsMask.size((int)i) == pyramidMask.size((int)i));
            CV_Assert(pyramidNormalsMask.type((int)i) == pyramidMask.type((int)i));
        }
    }
    else
    {
        pyramidNormalsMask.create((int)maskLevels, 1, CV_8U, -1);
        for(size_t i = 0; i < maskLevels; i++)
        {
            Mat& normalsMask = pyramidNormalsMask.getMatRef((int)i);
            normalsMask = pyramidMask.getMat((int)i).clone();

            const Mat normals = pyramidNormals.getMat((int)i);
            for(int y = 0; y < normalsMask.rows; y++)
            {
                const Vec3f *normals_row = normals.ptr<Vec3f>(y);
                uchar *normalsMask_row = normalsMask.ptr<uchar>(y);
                for(int x = 0; x < normalsMask.cols; x++)
                {
                    Vec3f n = normals_row[x];
                    if(cvIsNaN(n[0]))
                    {
                        CV_DbgAssert(cvIsNaN(n[1]) && cvIsNaN(n[2]));
                        normalsMask_row[x] = 0;
                    }
                }
            }
            randomSubsetOfMask(normalsMask, (float)maxPointsPart);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////

static
void computeProjectiveMatrix(const Mat& ksi, Mat& Rt)
{
    CV_Assert(ksi.size() == Size(1,6) && ksi.type() == CV_64FC1);

    const double* ksi_ptr = ksi.ptr<const double>();
    // 0.5 multiplication is here because (dual) quaternions keep half an angle/twist inside
    Matx44d matdq = (DualQuatd(0, ksi_ptr[0], ksi_ptr[1], ksi_ptr[2],
                               0, ksi_ptr[3], ksi_ptr[4], ksi_ptr[5])*0.5).exp().toMat(QUAT_ASSUME_UNIT);

    matdq.copyTo(Rt);
}

static
void computeCorresps(const Matx33f& _K, const Matx33f& _K_inv, const Mat& Rt,
                     const Mat& depth0, const Mat& validMask0,
                     const Mat& depth1, const Mat& selectMask1, float maxDepthDiff,
                     Mat& _corresps)
{
    CV_Assert(Rt.type() == CV_64FC1);

    Mat corresps(depth1.size(), CV_16SC2, Scalar::all(-1));

    Matx33d K(_K), K_inv(_K_inv);
    Rect r(0, 0, depth1.cols, depth1.rows);
    Mat Kt = Rt(Rect(3,0,1,3)).clone();
    Kt = K * Kt;
    const double * Kt_ptr = Kt.ptr<const double>();

    AutoBuffer<float> buf(3 * (depth1.cols + depth1.rows));
    float *KRK_inv0_u1 = buf.data();
    float *KRK_inv1_v1_plus_KRK_inv2 = KRK_inv0_u1 + depth1.cols;
    float *KRK_inv3_u1 = KRK_inv1_v1_plus_KRK_inv2 + depth1.rows;
    float *KRK_inv4_v1_plus_KRK_inv5 = KRK_inv3_u1 + depth1.cols;
    float *KRK_inv6_u1 = KRK_inv4_v1_plus_KRK_inv5 + depth1.rows;
    float *KRK_inv7_v1_plus_KRK_inv8 = KRK_inv6_u1 + depth1.cols;
    {
        Mat R = Rt(Rect(0,0,3,3)).clone();

        Mat KRK_inv = K * R * K_inv;
        const double * KRK_inv_ptr = KRK_inv.ptr<const double>();
        for(int u1 = 0; u1 < depth1.cols; u1++)
        {
            KRK_inv0_u1[u1] = (float)(KRK_inv_ptr[0] * u1);
            KRK_inv3_u1[u1] = (float)(KRK_inv_ptr[3] * u1);
            KRK_inv6_u1[u1] = (float)(KRK_inv_ptr[6] * u1);
        }

        for(int v1 = 0; v1 < depth1.rows; v1++)
        {
            KRK_inv1_v1_plus_KRK_inv2[v1] = (float)(KRK_inv_ptr[1] * v1 + KRK_inv_ptr[2]);
            KRK_inv4_v1_plus_KRK_inv5[v1] = (float)(KRK_inv_ptr[4] * v1 + KRK_inv_ptr[5]);
            KRK_inv7_v1_plus_KRK_inv8[v1] = (float)(KRK_inv_ptr[7] * v1 + KRK_inv_ptr[8]);
        }
    }

    int correspCount = 0;
    for(int v1 = 0; v1 < depth1.rows; v1++)
    {
        const float *depth1_row = depth1.ptr<float>(v1);
        const uchar *mask1_row = selectMask1.ptr<uchar>(v1);
        for(int u1 = 0; u1 < depth1.cols; u1++)
        {
            float d1 = depth1_row[u1];
            if(mask1_row[u1])
            {
                CV_DbgAssert(!cvIsNaN(d1));
                float transformed_d1 = static_cast<float>(d1 * (KRK_inv6_u1[u1] + KRK_inv7_v1_plus_KRK_inv8[v1]) +
                                                          Kt_ptr[2]);
                if(transformed_d1 > 0)
                {
                    float transformed_d1_inv = 1.f / transformed_d1;
                    int u0 = cvRound(transformed_d1_inv * (d1 * (KRK_inv0_u1[u1] + KRK_inv1_v1_plus_KRK_inv2[v1]) +
                                                           Kt_ptr[0]));
                    int v0 = cvRound(transformed_d1_inv * (d1 * (KRK_inv3_u1[u1] + KRK_inv4_v1_plus_KRK_inv5[v1]) +
                                                           Kt_ptr[1]));

                    if(r.contains(Point(u0,v0)))
                    {
                        float d0 = depth0.at<float>(v0,u0);
                        if(validMask0.at<uchar>(v0, u0) && std::abs(transformed_d1 - d0) <= maxDepthDiff)
                        {
                            CV_DbgAssert(!cvIsNaN(d0));
                            Vec2s& c = corresps.at<Vec2s>(v0,u0);
                            if(c[0] != -1)
                            {
                                int exist_u1 = c[0], exist_v1 = c[1];

                                float exist_d1 = (float)(depth1.at<float>(exist_v1,exist_u1) *
                                    (KRK_inv6_u1[exist_u1] + KRK_inv7_v1_plus_KRK_inv8[exist_v1]) + Kt_ptr[2]);

                                if(transformed_d1 > exist_d1)
                                    continue;
                            }
                            else
                                correspCount++;

                            c = Vec2s((short)u1, (short)v1);
                        }
                    }
                }
            }
        }
    }

    _corresps.create(correspCount, 1, CV_32SC4);
    Vec4i * corresps_ptr = _corresps.ptr<Vec4i>();
    for(int v0 = 0, i = 0; v0 < corresps.rows; v0++)
    {
        const Vec2s* corresps_row = corresps.ptr<Vec2s>(v0);
        for(int u0 = 0; u0 < corresps.cols; u0++)
        {
            const Vec2s& c = corresps_row[u0];
            if(c[0] != -1)
                corresps_ptr[i++] = Vec4i(u0,v0,c[0],c[1]);
        }
    }
}

static inline
void calcRgbdEquationCoeffs(double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz  = 1. / p3d.z,
           v0 = dIdx * fx * invz,
           v1 = dIdy * fy * invz,
           v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;

    C[0] = -p3d.z * v1 + p3d.y * v2;
    C[1] =  p3d.z * v0 - p3d.x * v2;
    C[2] = -p3d.y * v0 + p3d.x * v1;
    C[3] = v0;
    C[4] = v1;
    C[5] = v2;
}

static inline
void calcRgbdEquationCoeffsRotation(double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz  = 1. / p3d.z,
           v0 = dIdx * fx * invz,
           v1 = dIdy * fy * invz,
           v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;
    C[0] = -p3d.z * v1 + p3d.y * v2;
    C[1] =  p3d.z * v0 - p3d.x * v2;
    C[2] = -p3d.y * v0 + p3d.x * v1;
}

static inline
void calcRgbdEquationCoeffsTranslation(double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz  = 1. / p3d.z,
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
    C[1] =  p0.z * n1[0] - p0.x * n1[2];
    C[2] = -p0.y * n1[0] + p0.x * n1[1];
    C[3] = n1[0];
    C[4] = n1[1];
    C[5] = n1[2];
}

static inline
void calcICPEquationCoeffsRotation(double* C, const Point3f& p0, const Vec3f& n1)
{
    C[0] = -p0.z * n1[1] + p0.y * n1[2];
    C[1] =  p0.z * n1[0] - p0.x * n1[2];
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

static
void calcRgbdLsmMatrices(const Mat& image0, const Mat& cloud0, const Mat& Rt,
               const Mat& image1, const Mat& dI_dx1, const Mat& dI_dy1,
               const Mat& corresps, double fx, double fy, double sobelScaleIn,
               Mat& AtA, Mat& AtB, CalcRgbdEquationCoeffsPtr func, int transformDim)
{
    AtA = Mat(transformDim, transformDim, CV_64FC1, Scalar(0));
    AtB = Mat(transformDim, 1, CV_64FC1, Scalar(0));
    double* AtB_ptr = AtB.ptr<double>();

    const int correspsCount = corresps.rows;

    CV_Assert(Rt.type() == CV_64FC1);
    const double * Rt_ptr = Rt.ptr<const double>();

    AutoBuffer<float> diffs(correspsCount);
    float* diffs_ptr = diffs.data();

    const Vec4i* corresps_ptr = corresps.ptr<Vec4i>();

    double sigma = 0;
    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
         const Vec4i& c = corresps_ptr[correspIndex];
         int u0 = c[0], v0 = c[1];
         int u1 = c[2], v1 = c[3];

         diffs_ptr[correspIndex] = static_cast<float>(static_cast<int>(image0.at<uchar>(v0,u0)) -
                                                      static_cast<int>(image1.at<uchar>(v1,u1)));
         sigma += diffs_ptr[correspIndex] * diffs_ptr[correspIndex];
    }
    sigma = std::sqrt(sigma/correspsCount);

    std::vector<double> A_buf(transformDim);
    double* A_ptr = &A_buf[0];

    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
         const Vec4i& c = corresps_ptr[correspIndex];
         int u0 = c[0], v0 = c[1];
         int u1 = c[2], v1 = c[3];

         double w = sigma + std::abs(diffs_ptr[correspIndex]);
         w = w > DBL_EPSILON ? 1./w : 1.;

         double w_sobelScale = w * sobelScaleIn;

         const Point3f& p0 = cloud0.at<Point3f>(v0,u0);
         Point3f tp0;
         tp0.x = (float)(p0.x * Rt_ptr[0] + p0.y * Rt_ptr[1] + p0.z * Rt_ptr[2] + Rt_ptr[3]);
         tp0.y = (float)(p0.x * Rt_ptr[4] + p0.y * Rt_ptr[5] + p0.z * Rt_ptr[6] + Rt_ptr[7]);
         tp0.z = (float)(p0.x * Rt_ptr[8] + p0.y * Rt_ptr[9] + p0.z * Rt_ptr[10] + Rt_ptr[11]);

         func(A_ptr,
              w_sobelScale * dI_dx1.at<short int>(v1,u1),
              w_sobelScale * dI_dy1.at<short int>(v1,u1),
              tp0, fx, fy);

        for(int y = 0; y < transformDim; y++)
        {
            double* AtA_ptr = AtA.ptr<double>(y);
            for(int x = y; x < transformDim; x++)
                AtA_ptr[x] += A_ptr[y] * A_ptr[x];

            AtB_ptr[y] += A_ptr[y] * w * diffs_ptr[correspIndex];
        }
    }

    for(int y = 0; y < transformDim; y++)
        for(int x = y+1; x < transformDim; x++)
            AtA.at<double>(x,y) = AtA.at<double>(y,x);
}

static
void calcICPLsmMatrices(const Mat& cloud0, const Mat& Rt,
                        const Mat& cloud1, const Mat& normals1,
                        const Mat& corresps,
                        Mat& AtA, Mat& AtB, CalcICPEquationCoeffsPtr func, int transformDim)
{
    AtA = Mat(transformDim, transformDim, CV_64FC1, Scalar(0));
    AtB = Mat(transformDim, 1, CV_64FC1, Scalar(0));
    double* AtB_ptr = AtB.ptr<double>();

    const int correspsCount = corresps.rows;

    CV_Assert(Rt.type() == CV_64FC1);
    const double * Rt_ptr = Rt.ptr<const double>();

    AutoBuffer<float> diffs(correspsCount);
    float * diffs_ptr = diffs.data();

    AutoBuffer<Point3f> transformedPoints0(correspsCount);
    Point3f * tps0_ptr = transformedPoints0.data();

    const Vec4i* corresps_ptr = corresps.ptr<Vec4i>();

    double sigma = 0;
    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u0 = c[0], v0 = c[1];
        int u1 = c[2], v1 = c[3];

        const Point3f& p0 = cloud0.at<Point3f>(v0,u0);
        Point3f tp0;
        tp0.x = (float)(p0.x * Rt_ptr[0] + p0.y * Rt_ptr[1] + p0.z * Rt_ptr[2] + Rt_ptr[3]);
        tp0.y = (float)(p0.x * Rt_ptr[4] + p0.y * Rt_ptr[5] + p0.z * Rt_ptr[6] + Rt_ptr[7]);
        tp0.z = (float)(p0.x * Rt_ptr[8] + p0.y * Rt_ptr[9] + p0.z * Rt_ptr[10] + Rt_ptr[11]);

        Vec3f n1 = normals1.at<Vec3f>(v1, u1);
        Point3f v = cloud1.at<Point3f>(v1,u1) - tp0;

        tps0_ptr[correspIndex] = tp0;
        diffs_ptr[correspIndex] = n1[0] * v.x + n1[1] * v.y + n1[2] * v.z;
        sigma += diffs_ptr[correspIndex] * diffs_ptr[correspIndex];
    }

    sigma = std::sqrt(sigma/correspsCount);

    std::vector<double> A_buf(transformDim);
    double* A_ptr = &A_buf[0];
    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u1 = c[2], v1 = c[3];

        double w = sigma + std::abs(diffs_ptr[correspIndex]);
        w = w > DBL_EPSILON ? 1./w : 1.;

        func(A_ptr, tps0_ptr[correspIndex], normals1.at<Vec3f>(v1, u1) * w);

        for(int y = 0; y < transformDim; y++)
        {
            double* AtA_ptr = AtA.ptr<double>(y);
            for(int x = y; x < transformDim; x++)
                AtA_ptr[x] += A_ptr[y] * A_ptr[x];

            AtB_ptr[y] += A_ptr[y] * w * diffs_ptr[correspIndex];
        }
    }

    for(int y = 0; y < transformDim; y++)
        for(int x = y+1; x < transformDim; x++)
            AtA.at<double>(x,y) = AtA.at<double>(y,x);
}

static
bool solveSystem(const Mat& AtA, const Mat& AtB, double detThreshold, Mat& x)
{
    double det = determinant(AtA);

    if(fabs (det) < detThreshold || cvIsNaN(det) || cvIsInf(det))
        return false;

    solve(AtA, AtB, x, DECOMP_CHOLESKY);

    return true;
}

static
bool testDeltaTransformation(const Mat& deltaRt, double maxTranslation, double maxRotation)
{
    double translation = norm(deltaRt(Rect(3, 0, 1, 3)));

    Mat rvec;
    Rodrigues(deltaRt(Rect(0,0,3,3)), rvec);

    double rotation = norm(rvec) * 180. / CV_PI;

    return translation <= maxTranslation && rotation <= maxRotation;
}

static
bool RGBDICPOdometryImpl(OutputArray _Rt, const Mat& initRt,
                         const Ptr<OdometryFrame>& srcFrame,
                         const Ptr<OdometryFrame>& dstFrame,
                         const Matx33f& cameraMatrix,
                         float maxDepthDiff, const std::vector<int>& iterCounts,
                         double maxTranslation, double maxRotation,
                         int method, int transfromType)
{
    int transformDim = -1;
    CalcRgbdEquationCoeffsPtr rgbdEquationFuncPtr = 0;
    CalcICPEquationCoeffsPtr icpEquationFuncPtr = 0;
    switch(transfromType)
    {
    case Odometry::RIGID_BODY_MOTION:
        transformDim = 6;
        rgbdEquationFuncPtr = calcRgbdEquationCoeffs;
        icpEquationFuncPtr = calcICPEquationCoeffs;
        break;
    case Odometry::ROTATION:
        transformDim = 3;
        rgbdEquationFuncPtr = calcRgbdEquationCoeffsRotation;
        icpEquationFuncPtr = calcICPEquationCoeffsRotation;
        break;
    case Odometry::TRANSLATION:
        transformDim = 3;
        rgbdEquationFuncPtr = calcRgbdEquationCoeffsTranslation;
        icpEquationFuncPtr = calcICPEquationCoeffsTranslation;
        break;
    default:
        CV_Error(Error::StsBadArg, "Incorrect transformation type");
    }

    const int minOverdetermScale = 20;
    const int minCorrespsCount = minOverdetermScale * transformDim;

    std::vector<Matx33f> pyramidCameraMatrix;
    buildPyramidCameraMatrix(cameraMatrix, (int)iterCounts.size(), pyramidCameraMatrix);

    Mat resultRt = initRt.empty() ? Mat::eye(4,4,CV_64FC1) : initRt.clone();
    Mat currRt, ksi;

    bool isOk = false;
    for(int level = (int)iterCounts.size() - 1; level >= 0; level--)
    {
        const Matx33f& levelCameraMatrix = pyramidCameraMatrix[level];
        const Matx33f& levelCameraMatrix_inv = levelCameraMatrix.inv(DECOMP_SVD);
        const Mat srcLevelDepth, dstLevelDepth;
        srcFrame->getPyramidAt(srcLevelDepth, OdometryFrame::PYR_DEPTH, level);
        dstFrame->getPyramidAt(dstLevelDepth, OdometryFrame::PYR_DEPTH, level);

        const double fx = levelCameraMatrix(0, 0);
        const double fy = levelCameraMatrix(1, 1);
        const double determinantThreshold = 1e-6;

        Mat AtA_rgbd, AtB_rgbd, AtA_icp, AtB_icp;
        Mat corresps_rgbd, corresps_icp;

        // Run transformation search on current level iteratively.
        for(int iter = 0; iter < iterCounts[level]; iter ++)
        {
            Mat resultRt_inv = resultRt.inv(DECOMP_SVD);

            const Mat pyramidMask;
            srcFrame->getPyramidAt(pyramidMask, OdometryFrame::PYR_MASK, level);

            if(method & RGBD_ODOMETRY)
            {
                const Mat pyramidTexturedMask;
                dstFrame->getPyramidAt(pyramidTexturedMask, OdometryFrame::PYR_TEXMASK, level);
                computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt_inv,
                                srcLevelDepth, pyramidMask, dstLevelDepth, pyramidTexturedMask,
                                maxDepthDiff, corresps_rgbd);
            }

            if(method & ICP_ODOMETRY)
            {
                const Mat pyramidNormalsMask;
                dstFrame->getPyramidAt(pyramidNormalsMask, OdometryFrame::PYR_NORMMASK, level);
                computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt_inv,
                                srcLevelDepth, pyramidMask, dstLevelDepth, pyramidNormalsMask,
                                maxDepthDiff, corresps_icp);
            }

            if(corresps_rgbd.rows < minCorrespsCount && corresps_icp.rows < minCorrespsCount)
                break;

            const Mat srcPyrCloud;
            srcFrame->getPyramidAt(srcPyrCloud, OdometryFrame::PYR_CLOUD, level);


            Mat AtA(transformDim, transformDim, CV_64FC1, Scalar(0)), AtB(transformDim, 1, CV_64FC1, Scalar(0));
            if(corresps_rgbd.rows >= minCorrespsCount)
            {
                const Mat srcPyrImage, dstPyrImage, dstPyrIdx, dstPyrIdy;
                srcFrame->getPyramidAt(srcPyrImage, OdometryFrame::PYR_IMAGE, level);
                dstFrame->getPyramidAt(dstPyrImage, OdometryFrame::PYR_IMAGE, level);
                dstFrame->getPyramidAt(dstPyrIdx, OdometryFrame::PYR_DIX, level);
                dstFrame->getPyramidAt(dstPyrIdy, OdometryFrame::PYR_DIY, level);
                calcRgbdLsmMatrices(srcPyrImage, srcPyrCloud, resultRt, dstPyrImage, dstPyrIdx, dstPyrIdy,
                                    corresps_rgbd, fx, fy, sobelScale,
                                    AtA_rgbd, AtB_rgbd, rgbdEquationFuncPtr, transformDim);

                AtA += AtA_rgbd;
                AtB += AtB_rgbd;
            }
            if(corresps_icp.rows >= minCorrespsCount)
            {
                const Mat dstPyrCloud, dstPyrNormals;
                dstFrame->getPyramidAt(dstPyrCloud, OdometryFrame::PYR_CLOUD, level);
                dstFrame->getPyramidAt(dstPyrNormals, OdometryFrame::PYR_NORM, level);
                calcICPLsmMatrices(srcPyrCloud, resultRt, dstPyrCloud, dstPyrNormals,
                                   corresps_icp, AtA_icp, AtB_icp, icpEquationFuncPtr, transformDim);
                AtA += AtA_icp;
                AtB += AtB_icp;
            }

            bool solutionExist = solveSystem(AtA, AtB, determinantThreshold, ksi);
            if(!solutionExist)
                break;

            if(transfromType == Odometry::ROTATION)
            {
                Mat tmp(6, 1, CV_64FC1, Scalar(0));
                ksi.copyTo(tmp.rowRange(0,3));
                ksi = tmp;
            }
            else if(transfromType == Odometry::TRANSLATION)
            {
                Mat tmp(6, 1, CV_64FC1, Scalar(0));
                ksi.copyTo(tmp.rowRange(3,6));
                ksi = tmp;
            }

            computeProjectiveMatrix(ksi, currRt);
            resultRt = currRt * resultRt;
            isOk = true;
        }
    }
    _Rt.create(resultRt.size(), resultRt.type());
    Mat Rt = _Rt.getMat();
    resultRt.copyTo(Rt);

    if(isOk)
    {
        Mat deltaRt;
        if(initRt.empty())
            deltaRt = resultRt;
        else
            deltaRt = resultRt * initRt.inv(DECOMP_SVD);

        isOk = testDeltaTransformation(deltaRt, maxTranslation, maxRotation);
    }

    return isOk;
}

//

template<class ImageElemType>
static void
warpFrameImpl(InputArray _image, InputArray depth, InputArray _mask,
              const Mat& Rt, const Mat& cameraMatrix, const Mat& distCoeff,
              OutputArray _warpedImage, OutputArray warpedDepth, OutputArray warpedMask)
{
    CV_Assert(_image.size() == depth.size());

    Mat cloud;
    depthTo3d(depth, cameraMatrix, cloud);

    std::vector<Point2f> points2d;
    Mat transformedCloud;
    perspectiveTransform(cloud, transformedCloud, Rt);
    projectPoints(transformedCloud.reshape(3, 1), Mat::eye(3, 3, CV_64FC1), Mat::zeros(3, 1, CV_64FC1), cameraMatrix,
                  distCoeff, points2d);

    Mat image = _image.getMat();
    Size sz = _image.size();
    Mat mask = _mask.getMat();
    _warpedImage.create(sz, image.type());
    Mat warpedImage = _warpedImage.getMat();

    Mat zBuffer(sz, CV_32FC1, std::numeric_limits<float>::max());
    const Rect rect = Rect(Point(), sz);

    for (int y = 0; y < sz.height; y++)
    {
        //const Point3f* cloud_row = cloud.ptr<Point3f>(y);
        const Point3f* transformedCloud_row = transformedCloud.ptr<Point3f>(y);
        const Point2f* points2d_row = &points2d[y*sz.width];
        const ImageElemType* image_row = image.ptr<ImageElemType>(y);
        const uchar* mask_row = mask.empty() ? 0 : mask.ptr<uchar>(y);
        for (int x = 0; x < sz.width; x++)
        {
            const float transformed_z = transformedCloud_row[x].z;
            const Point2i p2d = points2d_row[x];
            if((!mask_row || mask_row[x]) && transformed_z > 0 && rect.contains(p2d) && /*!cvIsNaN(cloud_row[x].z) && */zBuffer.at<float>(p2d) > transformed_z)
            {
                warpedImage.at<ImageElemType>(p2d) = image_row[x];
                zBuffer.at<float>(p2d) = transformed_z;
            }
        }
    }

    if(warpedMask.needed())
        Mat(zBuffer != std::numeric_limits<float>::max()).copyTo(warpedMask);

   if(warpedDepth.needed())
    {
        zBuffer.setTo(std::numeric_limits<float>::quiet_NaN(), zBuffer == std::numeric_limits<float>::max());
        zBuffer.copyTo(warpedDepth);
    }
}

void warpFrame(InputArray image, InputArray depth, InputArray mask,
               InputArray Rt, InputArray cameraMatrix, InputArray distCoeff,
               OutputArray warpedImage, OutputArray warpedDepth, OutputArray warpedMask)
{
    if (image.type() == CV_8UC1)
        warpFrameImpl<uchar>(image, depth, mask, Rt.getMat(), cameraMatrix.getMat(), distCoeff.getMat(), warpedImage, warpedDepth, warpedMask);
    else if (image.type() == CV_8UC3)
        warpFrameImpl<Point3_<uchar> >(image, depth, mask, Rt.getMat(), cameraMatrix.getMat(), distCoeff.getMat(), warpedImage, warpedDepth, warpedMask);
    else
        CV_Error(Error::StsBadArg, "Image has to be type of CV_8UC1 or CV_8UC3");
}

///////////////////////////////////////////////////////////////////////////////////////////////

template<typename TMat>
struct OdometryFrameImpl : public OdometryFrame
{
    OdometryFrameImpl() : OdometryFrame(), image(), depth(), mask(), normals(), pyramids(N_PYRAMIDS) { }
    OdometryFrameImpl(InputArray _image, InputArray _depth, InputArray _mask = noArray(), InputArray _normals = noArray(), int _ID = -1);
    virtual ~OdometryFrameImpl() { }

    virtual void setImage(InputArray  _image) CV_OVERRIDE
    {
        image = getTMat<TMat>(_image);
    }
    virtual void getImage(OutputArray _image) CV_OVERRIDE
    {
        _image.assign(image);
    }
    virtual void setDepth(InputArray  _depth) CV_OVERRIDE
    {
        depth = getTMat<TMat>(_depth);
    }
    virtual void getDepth(OutputArray _depth) CV_OVERRIDE
    {
        _depth.assign(depth);
    }
    virtual void setMask(InputArray  _mask) CV_OVERRIDE
    {
        mask = getTMat<TMat>(_mask);
    }
    virtual void getMask(OutputArray _mask) CV_OVERRIDE
    {
        _mask.assign(mask);
    }
    virtual void setNormals(InputArray  _normals) CV_OVERRIDE
    {
        normals = getTMat<TMat>(_normals);
    }
    virtual void getNormals(OutputArray _normals) CV_OVERRIDE
    {
        _normals.assign(normals);
    }

    virtual void setPyramidLevels(size_t _nLevels) CV_OVERRIDE
    {
        for (auto& p : pyramids)
        {
            p.resize(_nLevels, TMat());
        }
    }

    virtual size_t getPyramidLevels(OdometryFramePyramidType what = PYR_IMAGE) CV_OVERRIDE
    {
        if (what < N_PYRAMIDS)
            return pyramids[what].size();
        else
            return 0;
    }

    virtual void setPyramidAt(InputArray  _pyrImage, OdometryFramePyramidType pyrType, size_t level) CV_OVERRIDE
    {
        TMat img = getTMat<TMat>(_pyrImage);
        pyramids[pyrType][level] = img;
    }

    virtual void getPyramidAt(OutputArray _pyrImage, OdometryFramePyramidType pyrType, size_t level) CV_OVERRIDE
    {
        TMat img;
        img = pyramids[pyrType][level];
        _pyrImage.assign(img);
    }

    TMat image;
    TMat depth;
    TMat mask;
    TMat normals;

    std::vector< std::vector<TMat> > pyramids;
};

template<>
OdometryFrameImpl<Mat>::OdometryFrameImpl(InputArray _image, InputArray _depth, InputArray _mask, InputArray _normals, int _ID) :
    OdometryFrame(),
    image(_image.getMat()), depth(_depth.getMat()), mask(_mask.getMat()), normals(_normals.getMat()),
    pyramids(N_PYRAMIDS)
{
    ID = _ID;
}

template<>
OdometryFrameImpl<UMat>::OdometryFrameImpl(InputArray _image, InputArray _depth, InputArray _mask, InputArray _normals, int _ID) :
    OdometryFrame(),
    image(_image.getUMat()), depth(_depth.getUMat()), mask(_mask.getUMat()), normals(_normals.getUMat()),
    pyramids(N_PYRAMIDS)
{
    ID = _ID;
}


Ptr<OdometryFrame> OdometryFrame::create(InputArray _image, InputArray _depth, InputArray _mask, InputArray _normals, int _ID)
{
    bool allEmpty = _image.empty() && _depth.empty() && _mask.empty() && _normals.empty();
    bool useOcl = (_image.isUMat() || _image.empty()) &&
                  (_depth.isUMat() || _depth.empty()) &&
                  (_mask.isUMat()  || _mask.empty()) &&
                  (_normals.isUMat() || _normals.empty());
    if (useOcl && !allEmpty)
        return makePtr<OdometryFrameImpl<UMat>>(_image, _depth, _mask, _normals, _ID);
    else
        return makePtr<OdometryFrameImpl<Mat>> (_image, _depth, _mask, _normals, _ID);
}


class OdometryImpl : public Odometry
{
public:
    // initialized outside of a class
    static const float defaultMaxTranslation;
    static const float defaultMaxRotation;
    static const float defaultMinGradientMagnitude;
    static const std::vector<int> defaultIterCounts;
    static const cv::Matx33f defaultCameraMatrix;
    static const float defaultMinDepth;
    static const float defaultMaxDepth;
    static const float defaultMaxDepthDiff;
    static const float defaultMaxPointsPart;

    OdometryImpl(InputArray _cameraMatrix = noArray(),
                 InputArray _iterCounts = noArray(),
                 InputArray _minGradientMagnitudes = noArray(),
                 Odometry::OdometryTransformType _transformType = Odometry::RIGID_BODY_MOTION)
    {
        setTransformType(_transformType);
        setMaxTranslation(defaultMaxTranslation);
        setMaxRotation(defaultMaxRotation);
        setCameraMatrix(_cameraMatrix);
        setIterationCounts(_iterCounts);
        setMinGradientMagnitudes(_minGradientMagnitudes);
    }

    virtual ~OdometryImpl() { }

    virtual bool compute(InputArray srcImage, InputArray srcDepth, InputArray srcMask, InputArray dstImage, InputArray dstDepth,
                         InputArray dstMask, OutputArray Rt, InputArray initRt = noArray()) const CV_OVERRIDE
    {
        Ptr<OdometryFrame> srcFrame = makeOdometryFrame(srcImage, srcDepth, srcMask);
        Ptr<OdometryFrame> dstFrame = makeOdometryFrame(dstImage, dstDepth, dstMask);

        return compute(srcFrame, dstFrame, Rt, initRt);
    }

    virtual bool compute(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame,
                         OutputArray Rt, InputArray initRt) const CV_OVERRIDE
    {
        Size srcSize = prepareFrameCache(srcFrame, OdometryFrame::CACHE_SRC);
        Size dstSize = prepareFrameCache(dstFrame, OdometryFrame::CACHE_DST);

        if (srcSize != dstSize)
            CV_Error(Error::StsBadSize, "srcFrame and dstFrame have to have the same size (resolution).");

        return computeImpl(srcFrame, dstFrame, Rt, initRt);
    }

    virtual Size prepareFrameCache(Ptr<OdometryFrame> frame, OdometryFrame::OdometryFrameCacheType /*cacheType*/) const CV_OVERRIDE
    {
        if (!frame)
            CV_Error(Error::StsBadArg, "Null frame pointer.");

        return Size();
    }

    virtual void getCameraMatrix(OutputArray val) const CV_OVERRIDE
    {
        cameraMatrix.copyTo(val);
    }
    virtual void setCameraMatrix(InputArray val) CV_OVERRIDE
    {
        if (val.empty())
        {
            cameraMatrix = defaultCameraMatrix;
        }
        else
        {
            CV_Assert(val.rows() == 3 && val.cols() == 3 && val.channels() == 1);
            CV_Assert(val.type() == CV_32F);
            val.copyTo(cameraMatrix);
        }
    }

    virtual Odometry::OdometryTransformType getTransformType() const CV_OVERRIDE
    {
        return transformType;
    }
    virtual void setTransformType(Odometry::OdometryTransformType val) CV_OVERRIDE
    {
        transformType = val;
    }

    virtual void getIterationCounts(OutputArray val) const CV_OVERRIDE
    {
        val.create((int)iterCounts.size(), 1, CV_32S);
        Mat(iterCounts).copyTo(val);
    }
    virtual void setIterationCounts(InputArray val) CV_OVERRIDE
    {
        if (val.empty())
        {
            iterCounts = defaultIterCounts;
        }
        else
        {
            CV_Assert(val.type() == CV_32SC1);
            CV_Assert(val.rows() == 1 || val.cols() == 1);
            iterCounts.resize(val.rows() * val.cols());
            val.copyTo(iterCounts);

            minGradientMagnitudes.resize(iterCounts.size(), defaultMinGradientMagnitude);
        }
    }

    virtual void getMinGradientMagnitudes(OutputArray val) const CV_OVERRIDE
    {
        val.create((int)minGradientMagnitudes.size(), 1, CV_32F);
        Mat(minGradientMagnitudes).copyTo(val);
    }
    virtual void setMinGradientMagnitudes(InputArray val) CV_OVERRIDE
    {
        if (val.empty())
        {
            minGradientMagnitudes = std::vector<float>(iterCounts.size(), defaultMinGradientMagnitude);
        }
        else
        {
            CV_Assert(val.type() == CV_32FC1);
            CV_Assert(val.rows() == 1 || val.cols() == 1);
            size_t valSize = val.rows() * val.cols();
            CV_Assert(valSize == iterCounts.size());
            minGradientMagnitudes.resize(valSize);
            val.copyTo(minGradientMagnitudes);
        }
    }

    /** Get max allowed translation in meters.
    Found delta transform is considered successful only if the translation is in given limits. */
    virtual double getMaxTranslation() const CV_OVERRIDE
    {
        return maxTranslation;
    }
    /** Set max allowed translation in meters.
    * Found delta transform is considered successful only if the translation is in given limits. */
    virtual void setMaxTranslation(double val) CV_OVERRIDE
    {
        maxTranslation = val;
    }
    /** Get max allowed rotation in degrees.
    * Found delta transform is considered successful only if the rotation is in given limits. */
    virtual double getMaxRotation() const CV_OVERRIDE
    {
        return maxRotation;
    }
    /** Set max allowed rotation in degrees.
    * Found delta transform is considered successful only if the rotation is in given limits. */
    virtual void setMaxRotation(double val) CV_OVERRIDE
    {
        maxRotation = val;
    }

    virtual bool computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt,
                             InputArray initRt) const = 0;

    double maxTranslation, maxRotation;

    Odometry::OdometryTransformType transformType;

    Matx33f cameraMatrix;
    std::vector<int> iterCounts;
    std::vector<float> minGradientMagnitudes;
};

const float OdometryImpl::defaultMaxTranslation = 0.15f;
const float OdometryImpl::defaultMaxRotation = 15.f;
const float OdometryImpl::defaultMinGradientMagnitude = 10.f;
const std::vector<int> OdometryImpl::defaultIterCounts = { 7, 7, 7, 10 };
const cv::Matx33f OdometryImpl::defaultCameraMatrix = { /* fx, 0, cx*/ 525.f, 0, 319.5f, /* 0, fy, cy */ 0, 525.f, 239.5f, /**/ 0, 0, 1.f };
const float OdometryImpl::defaultMinDepth = 0.f;
const float OdometryImpl::defaultMaxDepth = 4.f;
const float OdometryImpl::defaultMaxDepthDiff = 0.07f;
const float OdometryImpl::defaultMaxPointsPart = 0.07f;

//

// Public Odometry classes are pure abstract, therefore a sin of multiple inheritance should be forgiven
class RgbdOdometryImpl : public OdometryImpl, public RgbdOdometry
{
public:
    /** Constructor.
     * @param cameraMatrix Camera matrix
     * @param minDepth Pixels with depth less than minDepth will not be used (in meters)
     * @param maxDepth Pixels with depth larger than maxDepth will not be used (in meters)
     * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out
     *                     if their depth difference is larger than maxDepthDiff (in meters)
     * @param iterCounts Count of iterations on each pyramid level.
     * @param minGradientMagnitudes For each pyramid level the pixels will be filtered out
     *                              if they have gradient magnitude less than minGradientMagnitudes[level].
     * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart
     * @param transformType Class of transformation
     */
    RgbdOdometryImpl(InputArray _cameraMatrix = noArray(),
                     float _minDepth = defaultMinDepth,
                     float _maxDepth = defaultMaxDepth,
                     float _maxDepthDiff = defaultMaxDepthDiff,
                     InputArray _iterCounts = noArray(),
                     InputArray _minGradientMagnitudes = noArray(),
                     float _maxPointsPart = defaultMaxPointsPart,
                     Odometry::OdometryTransformType _transformType = Odometry::RIGID_BODY_MOTION) :
        OdometryImpl(_cameraMatrix, _iterCounts, _minGradientMagnitudes, _transformType)
    {
        setMinDepth(_minDepth);
        setMaxDepth(_maxDepth);
        setMaxDepthDiff(_maxDepthDiff);
        setMaxPointsPart(_maxPointsPart);
    }

    virtual ~RgbdOdometryImpl() { }

    virtual Size prepareFrameCache(Ptr<OdometryFrame> frame, OdometryFrame::OdometryFrameCacheType cacheType) const CV_OVERRIDE;

    virtual Ptr<OdometryFrame> makeOdometryFrame(InputArray image, InputArray depth, InputArray mask) const CV_OVERRIDE;

    virtual double getMinDepth() const CV_OVERRIDE
    {
        return minDepth;
    }
    virtual void setMinDepth(double val) CV_OVERRIDE
    {
        minDepth = val;
    }
    virtual double getMaxDepth() const CV_OVERRIDE
    {
        return maxDepth;
    }
    virtual void setMaxDepth(double val) CV_OVERRIDE
    {
        maxDepth = val;
    }
    virtual double getMaxDepthDiff() const CV_OVERRIDE
    {
        return maxDepthDiff;
    }
    virtual void setMaxDepthDiff(double val) CV_OVERRIDE
    {
        maxDepthDiff = val;
    }
    virtual double getMaxPointsPart() const CV_OVERRIDE
    {
        return maxPointsPart;
    }
    virtual void setMaxPointsPart(double val) CV_OVERRIDE
    {
        CV_Assert(val > 0. && val <= 1.);
        maxPointsPart = val;
    }

    virtual void getIterationCounts(OutputArray val) const CV_OVERRIDE
    {
        OdometryImpl::getIterationCounts(val);
    }
    virtual void setIterationCounts(InputArray val) CV_OVERRIDE
    {
        OdometryImpl::setIterationCounts(val);
    }
    virtual void getMinGradientMagnitudes(OutputArray val) const CV_OVERRIDE
    {
        OdometryImpl::getMinGradientMagnitudes(val);
    }
    virtual void setMinGradientMagnitudes(InputArray val) CV_OVERRIDE
    {
        OdometryImpl::setMinGradientMagnitudes(val);
    }
    virtual void getCameraMatrix(OutputArray val) const CV_OVERRIDE
    {
        OdometryImpl::getCameraMatrix(val);
    }
    virtual void setCameraMatrix(InputArray val) CV_OVERRIDE
    {
        OdometryImpl::setCameraMatrix(val);
    }
    virtual Odometry::OdometryTransformType getTransformType() const CV_OVERRIDE
    {
        return OdometryImpl::getTransformType();
    }
    virtual void setTransformType(Odometry::OdometryTransformType val) CV_OVERRIDE
    {
        OdometryImpl::setTransformType(val);
    }
    virtual double getMaxTranslation() const CV_OVERRIDE
    {
        return OdometryImpl::getMaxTranslation();
    }
    virtual void setMaxTranslation(double val) CV_OVERRIDE
    {
        OdometryImpl::setMaxTranslation(val);
    }
    virtual double getMaxRotation() const CV_OVERRIDE
    {
        return OdometryImpl::getMaxRotation();
    }
    virtual void setMaxRotation(double val) CV_OVERRIDE
    {
        OdometryImpl::setMaxRotation(val);
    }

    virtual bool compute(InputArray srcImage, InputArray srcDepth, InputArray srcMask, InputArray dstImage, InputArray dstDepth,
                         InputArray dstMask, OutputArray Rt, InputArray initRt = noArray()) const CV_OVERRIDE
    {
        return OdometryImpl::compute(srcImage, srcDepth, srcMask, dstImage, dstDepth, dstMask, Rt, initRt);
    }

    virtual bool compute(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame,
                         OutputArray Rt, InputArray initRt) const CV_OVERRIDE
    {
        return OdometryImpl::compute(srcFrame, dstFrame, Rt, initRt);
    }

protected:

    virtual bool computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt,
                             InputArray initRt) const CV_OVERRIDE;

    // Some params have commented desired type. It's due to AlgorithmInfo::addParams does not support it now.
    /*float*/
    double minDepth, maxDepth, maxDepthDiff;

    double maxPointsPart;
};

Ptr<RgbdOdometry> RgbdOdometry::create(InputArray _cameraMatrix, float _minDepth, float _maxDepth,
                                       float _maxDepthDiff, InputArray _iterCounts,
                                       InputArray _minGradientMagnitudes, float _maxPointsPart,
                                       Odometry::OdometryTransformType _transformType)
{
    return makePtr<RgbdOdometryImpl>(_cameraMatrix, _minDepth, _maxDepth, _maxDepthDiff, _iterCounts, _minGradientMagnitudes, _maxPointsPart, _transformType);
}

Ptr<OdometryFrame> RgbdOdometryImpl::makeOdometryFrame(InputArray _image, InputArray _depth, InputArray _mask) const
{
    // Can get rid of getMat() calls as soon as this Odometry algorithm supports UMats
    return OdometryFrame::create(_image.getMat(), _depth.getMat(), _mask.getMat());
}


Size RgbdOdometryImpl::prepareFrameCache(Ptr<OdometryFrame> frame, OdometryFrame::OdometryFrameCacheType cacheType) const
{
    OdometryImpl::prepareFrameCache(frame, cacheType);

    // Can be transformed into template argument in the future
    // when this algorithm supports OCL UMats too
    typedef Mat TMat;

    if (frame.dynamicCast<OdometryFrameImpl<UMat>>())
        CV_Error(cv::Error::Code::StsBadArg, "RgbdOdometry does not support UMats yet");

    TMat image;
    frame->getImage(image);
    if(image.empty())
    {
        if (frame->getPyramidLevels(OdometryFrame::PYR_IMAGE) > 0)
        {
            TMat pyr0;
            frame->getPyramidAt(pyr0, OdometryFrame::PYR_IMAGE, 0);
            frame->setImage(pyr0);
        }
        else
            CV_Error(Error::StsBadSize, "Image or pyramidImage have to be set.");
    }
    checkImage(image);

    TMat depth;
    frame->getDepth(depth);
    if(depth.empty())
    {
        if (frame->getPyramidLevels(OdometryFrame::PYR_DEPTH) > 0)
        {
            TMat pyr0;
            frame->getPyramidAt(pyr0, OdometryFrame::PYR_DEPTH, 0);
            frame->setDepth(pyr0);
        }
        else if(frame->getPyramidLevels(OdometryFrame::PYR_CLOUD) > 0)
        {
            TMat cloud;
            frame->getPyramidAt(cloud, OdometryFrame::PYR_CLOUD, 0);
            std::vector<TMat> xyz;
            split(cloud, xyz);
            frame->setDepth(xyz[2]);
        }
        else
            CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
    }
    checkDepth(depth, image.size());

    TMat mask;
    frame->getMask(mask);
    if (mask.empty() && frame->getPyramidLevels(OdometryFrame::PYR_MASK) > 0)
    {
        TMat pyr0;
        frame->getPyramidAt(pyr0, OdometryFrame::PYR_MASK, 0);
        frame->setMask(pyr0);
    }
    checkMask(mask, image.size());

    auto tframe = frame.dynamicCast<OdometryFrameImpl<TMat>>();
    preparePyramidImage(image, tframe->pyramids[OdometryFrame::PYR_IMAGE], iterCounts.size());

    preparePyramidDepth(depth, tframe->pyramids[OdometryFrame::PYR_DEPTH], iterCounts.size());

    preparePyramidMask<TMat>(mask, tframe->pyramids[OdometryFrame::PYR_DEPTH], (float)minDepth, (float)maxDepth,
                             tframe->pyramids[OdometryFrame::PYR_NORM], tframe->pyramids[OdometryFrame::PYR_MASK]);

    if(cacheType & OdometryFrame::CACHE_SRC)
        preparePyramidCloud<TMat>(tframe->pyramids[OdometryFrame::PYR_DEPTH], cameraMatrix, tframe->pyramids[OdometryFrame::PYR_CLOUD]);

    if(cacheType & OdometryFrame::CACHE_DST)
    {
        preparePyramidSobel<TMat>(tframe->pyramids[OdometryFrame::PYR_IMAGE], 1, 0, tframe->pyramids[OdometryFrame::PYR_DIX]);
        preparePyramidSobel<TMat>(tframe->pyramids[OdometryFrame::PYR_IMAGE], 0, 1, tframe->pyramids[OdometryFrame::PYR_DIY]);
        preparePyramidTexturedMask(tframe->pyramids[OdometryFrame::PYR_DIX], tframe->pyramids[OdometryFrame::PYR_DIY], minGradientMagnitudes,
                                   tframe->pyramids[OdometryFrame::PYR_MASK], maxPointsPart, tframe->pyramids[OdometryFrame::PYR_TEXMASK]);
    }

    return image.size();
}


bool RgbdOdometryImpl::computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt, InputArray initRt) const
{
    return RGBDICPOdometryImpl(Rt, initRt.getMat(), srcFrame, dstFrame, cameraMatrix, (float)maxDepthDiff, iterCounts, maxTranslation, maxRotation, RGBD_ODOMETRY, transformType);
}

//

// Public Odometry classes are pure abstract, therefore a sin of multiple inheritance should be forgiven
class ICPOdometryImpl : public OdometryImpl, public ICPOdometry
{
public:

    /** Constructor.
     * @param cameraMatrix Camera matrix
     * @param minDepth Pixels with depth less than minDepth will not be used
     * @param maxDepth Pixels with depth larger than maxDepth will not be used
     * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out
     *                     if their depth difference is larger than maxDepthDiff
     * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart
     * @param iterCounts Count of iterations on each pyramid level.
     * @param transformType Class of trasformation
     */
    ICPOdometryImpl(InputArray _cameraMatrix = noArray(),
                    float _minDepth = defaultMinDepth,
                    float _maxDepth = defaultMaxDepth,
                    float _maxDepthDiff = defaultMaxDepthDiff,
                    float _maxPointsPart = defaultMaxPointsPart,
                    InputArray _iterCounts = noArray(),
                    Odometry::OdometryTransformType _transformType = Odometry::RIGID_BODY_MOTION) :
        OdometryImpl(_cameraMatrix, _iterCounts, noArray(), _transformType)
    {
        setMinDepth(_minDepth);
        setMaxDepth(_maxDepth);
        setMaxDepthDiff(_maxDepthDiff);
        setMaxPointsPart(_maxPointsPart);
    }

    virtual Size prepareFrameCache(Ptr<OdometryFrame> frame, OdometryFrame::OdometryFrameCacheType cacheType) const CV_OVERRIDE;

    virtual Ptr<OdometryFrame> makeOdometryFrame(InputArray image, InputArray depth, InputArray mask) const CV_OVERRIDE;

    virtual double getMinDepth() const CV_OVERRIDE
    {
        return minDepth;
    }
    virtual void setMinDepth(double val) CV_OVERRIDE
    {
        minDepth = val;
    }
    virtual double getMaxDepth() const CV_OVERRIDE
    {
        return maxDepth;
    }
    virtual void setMaxDepth(double val) CV_OVERRIDE
    {
        maxDepth = val;
    }
    virtual double getMaxDepthDiff() const CV_OVERRIDE
    {
        return maxDepthDiff;
    }
    virtual void setMaxDepthDiff(double val) CV_OVERRIDE
    {
        maxDepthDiff = val;
    }
    virtual double getMaxPointsPart() const CV_OVERRIDE
    {
        return maxPointsPart;
    }
    virtual void setMaxPointsPart(double val) CV_OVERRIDE
    {
        CV_Assert(val > 0. && val <= 1.);
        maxPointsPart = val;
    }

    virtual void getIterationCounts(OutputArray val) const CV_OVERRIDE
    {
        OdometryImpl::getIterationCounts(val);
    }
    virtual void setIterationCounts(InputArray val) CV_OVERRIDE
    {
        OdometryImpl::setIterationCounts(val);
    }
    virtual void getMinGradientMagnitudes(OutputArray /*val*/) const CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, "This Odometry class does not use minGradientMagnitudes");
    }
    virtual void setMinGradientMagnitudes(InputArray /*val*/) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, "This Odometry class does not use minGradientMagnitudes");
    }
    virtual void getCameraMatrix(OutputArray val) const CV_OVERRIDE
    {
        OdometryImpl::getCameraMatrix(val);
    }
    virtual void setCameraMatrix(InputArray val) CV_OVERRIDE
    {
        OdometryImpl::setCameraMatrix(val);
    }
    virtual Odometry::OdometryTransformType getTransformType() const CV_OVERRIDE
    {
        return OdometryImpl::getTransformType();
    }
    virtual void setTransformType(Odometry::OdometryTransformType val) CV_OVERRIDE
    {
        OdometryImpl::setTransformType(val);
    }
    virtual double getMaxTranslation() const CV_OVERRIDE
    {
        return OdometryImpl::getMaxTranslation();
    }
    virtual void setMaxTranslation(double val) CV_OVERRIDE
    {
        OdometryImpl::setMaxTranslation(val);
    }
    virtual double getMaxRotation() const CV_OVERRIDE
    {
        return OdometryImpl::getMaxRotation();
    }
    virtual void setMaxRotation(double val) CV_OVERRIDE
    {
        OdometryImpl::setMaxRotation(val);
    }

    virtual Ptr<RgbdNormals> getNormalsComputer() const CV_OVERRIDE
    {
        return normalsComputer;
    }

    virtual bool compute(InputArray srcImage, InputArray srcDepth, InputArray srcMask, InputArray dstImage, InputArray dstDepth,
                         InputArray dstMask, OutputArray Rt, InputArray initRt = noArray()) const CV_OVERRIDE
    {
        return OdometryImpl::compute(srcImage, srcDepth, srcMask, dstImage, dstDepth, dstMask, Rt, initRt);
    }

    virtual bool compute(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame,
                         OutputArray Rt, InputArray initRt) const CV_OVERRIDE
    {
        return OdometryImpl::compute(srcFrame, dstFrame, Rt, initRt);
    }

protected:

    virtual bool computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt,
                             InputArray initRt) const CV_OVERRIDE;

    // Some params have commented desired type. It's due to AlgorithmInfo::addParams does not support it now.
    /*float*/
    double minDepth, maxDepth, maxDepthDiff;
    /*float*/
    double maxPointsPart;

    mutable Ptr<RgbdNormals> normalsComputer;
};

Ptr<ICPOdometry> ICPOdometry::create(InputArray _cameraMatrix, float _minDepth, float _maxDepth,
                                     float _maxDepthDiff, float _maxPointsPart, InputArray _iterCounts,
                                     Odometry::OdometryTransformType _transformType)
{
    return makePtr<ICPOdometryImpl>(_cameraMatrix, _minDepth, _maxDepth, _maxDepthDiff, _maxPointsPart, _iterCounts, _transformType);
}

Ptr<OdometryFrame> ICPOdometryImpl::makeOdometryFrame(InputArray _image, InputArray _depth, InputArray _mask) const
{
    // Can get rid of getMat() calls as soon as this Odometry algorithm supports UMats
    return OdometryFrame::create(_image.getMat(), _depth.getMat(), _mask.getMat());
}


Size ICPOdometryImpl::prepareFrameCache(Ptr<OdometryFrame> frame, OdometryFrame::OdometryFrameCacheType cacheType) const
{
    OdometryImpl::prepareFrameCache(frame, cacheType);

    // Can be transformed into template argument in the future
    // when this algorithm supports OCL UMats too
    typedef Mat TMat;

    if (frame.dynamicCast<OdometryFrameImpl<UMat>>())
        CV_Error(cv::Error::Code::StsBadArg, "ICPOdometry does not support UMats yet");

    TMat depth;
    frame->getDepth(depth);
    if(depth.empty())
    {
        if (frame->getPyramidLevels(OdometryFrame::PYR_DEPTH))
        {
            TMat pyr0;
            frame->getPyramidAt(pyr0, OdometryFrame::PYR_DEPTH, 0);
            frame->setDepth(pyr0);
        }
        else if(frame->getPyramidLevels(OdometryFrame::PYR_CLOUD))
        {
            TMat cloud;
            frame->getPyramidAt(cloud, OdometryFrame::PYR_CLOUD, 0);
            std::vector<TMat> xyz;
            split(cloud, xyz);
            frame->setDepth(xyz[2]);
        }
        else
            CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
    }
    checkDepth(depth, depth.size());

    TMat mask;
    frame->getMask(mask);
    if (mask.empty() && frame->getPyramidLevels(OdometryFrame::PYR_MASK))
    {
        Mat m0;
        frame->getPyramidAt(m0, OdometryFrame::PYR_MASK, 0);
        frame->setMask(m0);
    }
    checkMask(mask, depth.size());

    auto tframe = frame.dynamicCast<OdometryFrameImpl<TMat>>();
    preparePyramidDepth(depth, tframe->pyramids[OdometryFrame::PYR_DEPTH], iterCounts.size());

    preparePyramidCloud<TMat>(tframe->pyramids[OdometryFrame::PYR_DEPTH], cameraMatrix, tframe->pyramids[OdometryFrame::PYR_CLOUD]);

    if(cacheType & OdometryFrame::CACHE_DST)
    {
        TMat normals;
        frame->getNormals(normals);
        if(normals.empty())
        {
            if (frame->getPyramidLevels(OdometryFrame::PYR_NORM))
            {
                TMat n0;
                frame->getPyramidAt(n0, OdometryFrame::PYR_NORM, 0);
                frame->setNormals(n0);
            }
            else
            {
                Matx33f K;
                if (!normalsComputer.empty())
                    normalsComputer->getK(K);
                if(normalsComputer.empty() ||
                   normalsComputer->getRows() != depth.rows ||
                   normalsComputer->getCols() != depth.cols ||
                   norm(K, cameraMatrix) > FLT_EPSILON)
                    normalsComputer = RgbdNormals::create(depth.rows,
                                                          depth.cols,
                                                          depth.depth(),
                                                          cameraMatrix,
                                                          normalWinSize,
                                                          normalMethod);
                TMat c0;
                frame->getPyramidAt(c0, OdometryFrame::PYR_CLOUD, 0);
                normalsComputer->apply(c0, normals);
                frame->setNormals(normals);
            }
        }
        checkNormals(normals, depth.size());

        preparePyramidNormals(normals, tframe->pyramids[OdometryFrame::PYR_DEPTH], tframe->pyramids[OdometryFrame::PYR_NORM]);

        preparePyramidMask<TMat>(mask, tframe->pyramids[OdometryFrame::PYR_DEPTH], (float)minDepth, (float)maxDepth,
                                 tframe->pyramids[OdometryFrame::PYR_NORM], tframe->pyramids[OdometryFrame::PYR_MASK]);

        preparePyramidNormalsMask(tframe->pyramids[OdometryFrame::PYR_NORM], tframe->pyramids[OdometryFrame::PYR_MASK],
                                  maxPointsPart, tframe->pyramids[OdometryFrame::PYR_NORMMASK]);
    }
    else
        preparePyramidMask<TMat>(mask, tframe->pyramids[OdometryFrame::PYR_DEPTH], (float)minDepth, (float)maxDepth,
                                 tframe->pyramids[OdometryFrame::PYR_NORM], tframe->pyramids[OdometryFrame::PYR_MASK]);

    return depth.size();
}


bool ICPOdometryImpl::computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt, InputArray initRt) const
{
    return RGBDICPOdometryImpl(Rt, initRt.getMat(), srcFrame, dstFrame, cameraMatrix, (float)maxDepthDiff, iterCounts, maxTranslation, maxRotation, ICP_ODOMETRY, transformType);
}

//

// Public Odometry classes are pure abstract, therefore a sin of multiple inheritance should be forgiven
class RgbdICPOdometryImpl : public OdometryImpl, public RgbdICPOdometry
{
public:
    /** Constructor.
     * @param cameraMatrix Camera matrix
     * @param minDepth Pixels with depth less than minDepth will not be used
     * @param maxDepth Pixels with depth larger than maxDepth will not be used
     * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out
     *                     if their depth difference is larger than maxDepthDiff
     * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart
     * @param iterCounts Count of iterations on each pyramid level.
     * @param minGradientMagnitudes For each pyramid level the pixels will be filtered out
     *                              if they have gradient magnitude less than minGradientMagnitudes[level].
     * @param transformType Class of trasformation
     */
    RgbdICPOdometryImpl(InputArray _cameraMatrix = noArray(),
                        float _minDepth = defaultMinDepth,
                        float _maxDepth = defaultMaxDepth,
                        float _maxDepthDiff = defaultMaxDepthDiff,
                        float _maxPointsPart = defaultMaxPointsPart,
                        InputArray _iterCounts = noArray(),
                        InputArray _minGradientMagnitudes = noArray(),
                        Odometry::OdometryTransformType _transformType = Odometry::RIGID_BODY_MOTION) :
        OdometryImpl(_cameraMatrix, _iterCounts, _minGradientMagnitudes, _transformType)
    {
        setMinDepth(_minDepth);
        setMaxDepth(_maxDepth);
        setMaxDepthDiff(_maxDepthDiff);
        setMaxPointsPart(_maxPointsPart);
    }

    virtual Size prepareFrameCache(Ptr<OdometryFrame> frame, OdometryFrame::OdometryFrameCacheType cacheType) const CV_OVERRIDE;

    virtual Ptr<OdometryFrame> makeOdometryFrame(InputArray image, InputArray depth, InputArray mask) const CV_OVERRIDE;

    virtual double getMinDepth() const CV_OVERRIDE
    {
        return minDepth;
    }
    virtual void setMinDepth(double val) CV_OVERRIDE
    {
        minDepth = val;
    }
    virtual double getMaxDepth() const CV_OVERRIDE
    {
        return maxDepth;
    }
    virtual void setMaxDepth(double val) CV_OVERRIDE
    {
        maxDepth = val;
    }
    virtual double getMaxDepthDiff() const CV_OVERRIDE
    {
        return maxDepthDiff;
    }
    virtual void setMaxDepthDiff(double val) CV_OVERRIDE
    {
        maxDepthDiff = val;
    }
    virtual double getMaxPointsPart() const CV_OVERRIDE
    {
        return maxPointsPart;
    }
    virtual void setMaxPointsPart(double val) CV_OVERRIDE
    {
        CV_Assert(val > 0. && val <= 1.);
        maxPointsPart = val;
    }

    virtual void getIterationCounts(OutputArray val) const CV_OVERRIDE
    {
        OdometryImpl::getIterationCounts(val);
    }
    virtual void setIterationCounts(InputArray val) CV_OVERRIDE
    {
        OdometryImpl::setIterationCounts(val);
    }
    virtual void getMinGradientMagnitudes(OutputArray val) const CV_OVERRIDE
    {
        OdometryImpl::getMinGradientMagnitudes(val);
    }
    virtual void setMinGradientMagnitudes(InputArray val) CV_OVERRIDE
    {
        OdometryImpl::setMinGradientMagnitudes(val);
    }
    virtual void getCameraMatrix(OutputArray val) const CV_OVERRIDE
    {
        OdometryImpl::getCameraMatrix(val);
    }
    virtual void setCameraMatrix(InputArray val) CV_OVERRIDE
    {
        OdometryImpl::setCameraMatrix(val);
    }
    virtual Odometry::OdometryTransformType getTransformType() const CV_OVERRIDE
    {
        return OdometryImpl::getTransformType();
    }
    virtual void setTransformType(Odometry::OdometryTransformType val) CV_OVERRIDE
    {
        OdometryImpl::setTransformType(val);
    }
    virtual double getMaxTranslation() const CV_OVERRIDE
    {
        return OdometryImpl::getMaxTranslation();
    }
    virtual void setMaxTranslation(double val) CV_OVERRIDE
    {
        OdometryImpl::setMaxTranslation(val);
    }
    virtual double getMaxRotation() const CV_OVERRIDE
    {
        return OdometryImpl::getMaxRotation();
    }
    virtual void setMaxRotation(double val) CV_OVERRIDE
    {
        OdometryImpl::setMaxRotation(val);
    }

    virtual Ptr<RgbdNormals> getNormalsComputer() const CV_OVERRIDE
    {
        return normalsComputer;
    }

    virtual bool compute(InputArray srcImage, InputArray srcDepth, InputArray srcMask, InputArray dstImage, InputArray dstDepth,
                         InputArray dstMask, OutputArray Rt, InputArray initRt = noArray()) const CV_OVERRIDE
    {
        return OdometryImpl::compute(srcImage, srcDepth, srcMask, dstImage, dstDepth, dstMask, Rt, initRt);
    }

    virtual bool compute(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame,
                         OutputArray Rt, InputArray initRt) const CV_OVERRIDE
    {
        return OdometryImpl::compute(srcFrame, dstFrame, Rt, initRt);
    }

protected:

    virtual bool computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt,
                             InputArray initRt) const CV_OVERRIDE;

    // Some params have commented desired type. It's due to AlgorithmInfo::addParams does not support it now.
    /*float*/
    double minDepth, maxDepth, maxDepthDiff;
    /*float*/
    double maxPointsPart;

    mutable Ptr<RgbdNormals> normalsComputer;
};


Ptr<RgbdICPOdometry> RgbdICPOdometry::create(InputArray _cameraMatrix, float _minDepth, float _maxDepth,
                                             float _maxDepthDiff, float _maxPointsPart, InputArray _iterCounts,
                                             InputArray _minGradientMagnitudes,
                                             Odometry::OdometryTransformType _transformType)
{
    return makePtr<RgbdICPOdometryImpl>(_cameraMatrix, _minDepth, _maxDepth, _maxDepthDiff, _maxPointsPart, _iterCounts, _minGradientMagnitudes, _transformType);
}

Ptr<OdometryFrame> RgbdICPOdometryImpl::makeOdometryFrame(InputArray _image, InputArray _depth, InputArray _mask) const
{
    // Can get rid of getMat() calls as soon as this Odometry algorithm supports UMats
    return OdometryFrame::create(_image.getMat(), _depth.getMat(), _mask.getMat());
}


Size RgbdICPOdometryImpl::prepareFrameCache(Ptr<OdometryFrame> frame, OdometryFrame::OdometryFrameCacheType cacheType) const
{
    OdometryImpl::prepareFrameCache(frame, cacheType);

    // Can be transformed into template argument in the future
    // when this algorithm supports OCL UMats too
    typedef Mat TMat;

    if (frame.dynamicCast<OdometryFrameImpl<UMat>>())
        CV_Error(cv::Error::Code::StsBadArg, "RgbdICPOdometry does not support UMats yet");

    TMat image;
    frame->getImage(image);
    if(image.empty())
    {
        if (frame->getPyramidLevels(OdometryFrame::PYR_IMAGE))
        {
            TMat p0;
            frame->getPyramidAt(p0, OdometryFrame::PYR_IMAGE, 0);
            frame->setImage(p0);
        }
        else
            CV_Error(Error::StsBadSize, "Image or pyramidImage have to be set.");
    }
    checkImage(image);

    TMat depth;
    frame->getDepth(depth);
    if (depth.empty())
    {
        if (frame->getPyramidLevels(OdometryFrame::PYR_DEPTH))
        {
            TMat d0;
            frame->getPyramidAt(d0, OdometryFrame::PYR_DEPTH, 0);
            frame->setDepth(d0);
        }
        else if(frame->getPyramidLevels(OdometryFrame::PYR_CLOUD))
        {
            TMat cloud;
            frame->getPyramidAt(cloud, OdometryFrame::PYR_CLOUD, 0);
            std::vector<TMat> xyz;
            split(cloud, xyz);
            frame->setDepth(xyz[2]);
        }
        else
            CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
    }
    checkDepth(depth, image.size());

    TMat mask;
    frame->getMask(mask);
    if(mask.empty() && frame->getPyramidLevels(OdometryFrame::PYR_MASK))
    {
        TMat m0;
        frame->getPyramidAt(m0, OdometryFrame::PYR_MASK, 0);
        frame->setMask(m0);
    }
    checkMask(mask, image.size());

    auto tframe = frame.dynamicCast<OdometryFrameImpl<TMat>>();
    preparePyramidImage(image, tframe->pyramids[OdometryFrame::PYR_IMAGE], iterCounts.size());

    preparePyramidDepth(depth, tframe->pyramids[OdometryFrame::PYR_DEPTH], iterCounts.size());

    preparePyramidCloud<TMat>(tframe->pyramids[OdometryFrame::PYR_DEPTH], cameraMatrix, tframe->pyramids[OdometryFrame::PYR_CLOUD]);

    if(cacheType & OdometryFrame::CACHE_DST)
    {
        TMat normals;
        frame->getNormals(normals);
        if (normals.empty())
        {
            if (frame->getPyramidLevels(OdometryFrame::PYR_NORM))
            {
                TMat n0;
                frame->getPyramidAt(n0, OdometryFrame::PYR_NORM, 0);
                frame->setNormals(n0);
            }
            else
            {
                Matx33f K;
                if (!normalsComputer.empty())
                    normalsComputer->getK(K);
                if(normalsComputer.empty() ||
                   normalsComputer->getRows() != depth.rows ||
                   normalsComputer->getCols() != depth.cols ||
                   norm(K, cameraMatrix) > FLT_EPSILON)
                    normalsComputer = RgbdNormals::create(depth.rows,
                                                          depth.cols,
                                                          depth.depth(),
                                                          cameraMatrix,
                                                          normalWinSize,
                                                          normalMethod);

                TMat c0;
                frame->getPyramidAt(c0, OdometryFrame::PYR_CLOUD, 0);
                normalsComputer->apply(c0, normals);
                frame->setNormals(normals);
            }
        }
        checkNormals(normals, depth.size());

        preparePyramidNormals(normals, tframe->pyramids[OdometryFrame::PYR_DEPTH], tframe->pyramids[OdometryFrame::PYR_NORM]);

        preparePyramidMask<TMat>(mask, tframe->pyramids[OdometryFrame::PYR_DEPTH], (float)minDepth, (float)maxDepth,
                                 tframe->pyramids[OdometryFrame::PYR_NORM], tframe->pyramids[OdometryFrame::PYR_MASK]);

        preparePyramidSobel<TMat>(tframe->pyramids[OdometryFrame::PYR_IMAGE], 1, 0, tframe->pyramids[OdometryFrame::PYR_DIX]);
        preparePyramidSobel<TMat>(tframe->pyramids[OdometryFrame::PYR_IMAGE], 0, 1, tframe->pyramids[OdometryFrame::PYR_DIY]);
        preparePyramidTexturedMask(tframe->pyramids[OdometryFrame::PYR_DIX], tframe->pyramids[OdometryFrame::PYR_DIY],
                                   minGradientMagnitudes, tframe->pyramids[OdometryFrame::PYR_MASK],
                                   maxPointsPart, tframe->pyramids[OdometryFrame::PYR_TEXMASK]);

        preparePyramidNormalsMask(tframe->pyramids[OdometryFrame::PYR_NORM], tframe->pyramids[OdometryFrame::PYR_MASK],
                                  maxPointsPart, tframe->pyramids[OdometryFrame::PYR_NORMMASK]);
    }
    else
        preparePyramidMask<TMat>(mask, tframe->pyramids[OdometryFrame::PYR_DEPTH], (float)minDepth, (float)maxDepth,
                                 tframe->pyramids[OdometryFrame::PYR_NORM], tframe->pyramids[OdometryFrame::PYR_MASK]);

    return image.size();
}


bool RgbdICPOdometryImpl::computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt, InputArray initRt) const
{
    return RGBDICPOdometryImpl(Rt, initRt.getMat(), srcFrame, dstFrame, cameraMatrix, (float)maxDepthDiff, iterCounts,  maxTranslation, maxRotation, MERGED_ODOMETRY, transformType);
}

//

// Public Odometry classes are pure abstract, therefore a sin of multiple inheritance should be forgiven
class FastICPOdometryImpl : public OdometryImpl, public FastICPOdometry
{
public:
    /** Creates FastICPOdometry object
     * @param cameraMatrix Camera matrix
     * @param maxDistDiff Correspondences between pixels of two given frames will be filtered out
     *                     if their depth difference is larger than maxDepthDiff
     * @param angleThreshold Correspondence will be filtered out
     *                     if an angle between their normals is bigger than threshold
     * @param sigmaDepth Depth sigma in meters for bilateral smooth
     * @param sigmaSpatial Spatial sigma in pixels for bilateral smooth
     * @param kernelSize Kernel size in pixels for bilateral smooth
     * @param iterCounts Count of iterations on each pyramid level
     * @param depthFactor pre-scale per 1 meter for input values
     * @param truncateThreshold Threshold for depth truncation in meters
     *        All depth values beyond this threshold will be set to zero
     */
    FastICPOdometryImpl(InputArray _cameraMatrix = noArray(),
                        float _maxDistDiff = 0.1f,
                        float _angleThreshold = (float)(30. * CV_PI / 180.),
                        float _sigmaDepth = 0.04f,
                        float _sigmaSpatial = 4.5f,
                        int _kernelSize = 7,
                        InputArray _iterCounts = noArray(),
                        float _depthFactor = 1.f,
                        float _truncateThreshold = 0.f) :
        OdometryImpl(_cameraMatrix, _iterCounts, noArray(), Odometry::RIGID_BODY_MOTION)
    {
        setMaxDistDiff(_maxDistDiff);
        setAngleThreshold(_angleThreshold);
        setSigmaDepth(_sigmaDepth);
        setSigmaSpatial(_sigmaSpatial);
        setKernelSize(_kernelSize);
        setDepthFactor(_depthFactor);
        setTruncateThreshold(_truncateThreshold);
    }

    virtual Size prepareFrameCache(Ptr<OdometryFrame> frame, OdometryFrame::OdometryFrameCacheType cacheType) const CV_OVERRIDE;

    virtual Ptr<OdometryFrame> makeOdometryFrame(InputArray image, InputArray depth, InputArray mask) const CV_OVERRIDE;

    virtual double getMaxDistDiff() const CV_OVERRIDE
    {
        return maxDistDiff;
    }
    virtual void setMaxDistDiff(float val) CV_OVERRIDE
    {
        CV_Assert(val > 0);
        maxDistDiff = val;
    }
    virtual float getAngleThreshold() const CV_OVERRIDE
    {
        return angleThreshold;
    }
    virtual void setAngleThreshold(float f) CV_OVERRIDE
    {
        CV_Assert(f > 0);
        angleThreshold = f;
    }
    virtual float getSigmaDepth() const CV_OVERRIDE
    {
        return sigmaDepth;
    }
    virtual void setSigmaDepth(float f) CV_OVERRIDE
    {
        CV_Assert(f > 0);
        sigmaDepth = f;
    }
    virtual float getSigmaSpatial() const CV_OVERRIDE
    {
        return sigmaSpatial;
    }
    virtual void setSigmaSpatial(float f) CV_OVERRIDE
    {
        CV_Assert(f > 0);
        sigmaSpatial = f;
    }
    virtual int getKernelSize() const CV_OVERRIDE
    {
        return kernelSize;
    }
    virtual void setKernelSize(int f) CV_OVERRIDE
    {
        CV_Assert(f > 0);
        kernelSize = f;
    }
    virtual float getDepthFactor() const CV_OVERRIDE
    {
        return depthFactor;
    }
    virtual void setDepthFactor(float _depthFactor) CV_OVERRIDE
    {
        depthFactor = _depthFactor;
    }
    virtual float getTruncateThreshold() const CV_OVERRIDE
    {
        return truncateThreshold;
    }
    virtual void setTruncateThreshold(float _truncateThreshold) CV_OVERRIDE
    {
        truncateThreshold = _truncateThreshold;
    }

    virtual void getIterationCounts(OutputArray val) const CV_OVERRIDE
    {
        OdometryImpl::getIterationCounts(val);
    }
    virtual void setIterationCounts(InputArray val) CV_OVERRIDE
    {
        OdometryImpl::setIterationCounts(val);
    }
    virtual void getMinGradientMagnitudes(OutputArray /*val*/) const CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, "This Odometry class does not use minGradientMagnitudes");
    }
    virtual void setMinGradientMagnitudes(InputArray /*val*/) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, "This Odometry class does not use minGradientMagnitudes");
    }
    virtual void getCameraMatrix(OutputArray val) const CV_OVERRIDE
    {
        OdometryImpl::getCameraMatrix(val);
    }
    virtual void setCameraMatrix(InputArray val) CV_OVERRIDE
    {
        OdometryImpl::setCameraMatrix(val);
    }
    virtual double getMaxTranslation() const CV_OVERRIDE
    {
        return OdometryImpl::getMaxTranslation();
    }
    virtual void setMaxTranslation(double val) CV_OVERRIDE
    {
        OdometryImpl::setMaxTranslation(val);
    }
    virtual double getMaxRotation() const CV_OVERRIDE
    {
        return OdometryImpl::getMaxRotation();
    }
    virtual void setMaxRotation(double val) CV_OVERRIDE
    {
        OdometryImpl::setMaxRotation(val);
    }
    virtual Odometry::OdometryTransformType getTransformType() const CV_OVERRIDE
    {
        return Odometry::RIGID_BODY_MOTION;
    }
    virtual void setTransformType(Odometry::OdometryTransformType val) CV_OVERRIDE
    {
        if (val != Odometry::RIGID_BODY_MOTION)
            CV_Error(CV_StsBadArg, "Rigid Body Motion is the only accepted transformation type for this odometry method");
    }

    virtual bool compute(InputArray srcImage, InputArray srcDepth, InputArray srcMask, InputArray dstImage, InputArray dstDepth,
                         InputArray dstMask, OutputArray Rt, InputArray initRt = noArray()) const CV_OVERRIDE
    {
        return OdometryImpl::compute(srcImage, srcDepth, srcMask, dstImage, dstDepth, dstMask, Rt, initRt);
    }

    virtual bool compute(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame,
                         OutputArray Rt, InputArray initRt) const CV_OVERRIDE
    {
        return OdometryImpl::compute(srcFrame, dstFrame, Rt, initRt);
    }

protected:

    virtual bool computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt,
                             InputArray initRt) const CV_OVERRIDE;

    template<typename TMat>
    Size prepareFrameCacheT(Ptr<OdometryFrame> frame, OdometryFrame::OdometryFrameCacheType cacheType) const;

    // Some params have commented desired type. It's due to AlgorithmInfo::addParams does not support it now.
    float maxDistDiff;

    float angleThreshold;

    float sigmaDepth;

    float sigmaSpatial;

    int kernelSize;

    float depthFactor;

    float truncateThreshold;
};

using namespace cv::kinfu;

Ptr<FastICPOdometry> FastICPOdometry::create(InputArray _cameraMatrix,
                                             float _maxDistDiff,
                                             float _angleThreshold,
                                             float _sigmaDepth,
                                             float _sigmaSpatial,
                                             int _kernelSize,
                                             InputArray _iterCounts,
                                             float _depthFactor,
                                             float _truncateThreshold)
{
    return makePtr<FastICPOdometryImpl>(_cameraMatrix, _maxDistDiff, _angleThreshold,
                                        _sigmaDepth, _sigmaSpatial, _kernelSize, _iterCounts,
                                        _depthFactor, _truncateThreshold);
}

Ptr<OdometryFrame> FastICPOdometryImpl::makeOdometryFrame(InputArray _image, InputArray _depth, InputArray _mask) const
{
    return OdometryFrame::create(_image, _depth, _mask);
}

template<typename TMat>
Size FastICPOdometryImpl::prepareFrameCacheT(Ptr<OdometryFrame> frame, OdometryFrame::OdometryFrameCacheType cacheType) const
{
    OdometryImpl::prepareFrameCache(frame, cacheType);

    TMat depth;
    frame->getDepth(depth);
    if (depth.empty())
    {
        if (frame->getPyramidLevels(OdometryFrame::PYR_CLOUD))
        {
            if (frame->getPyramidLevels(OdometryFrame::PYR_NORM))
            {
                TMat points, normals;
                frame->getPyramidAt(points, OdometryFrame::PYR_CLOUD, 0);
                frame->getPyramidAt(normals, OdometryFrame::PYR_NORM, 0);
                std::vector<TMat> pyrPoints, pyrNormals;
                // in, in, out, out
                size_t nLevels = iterCounts.size();
                buildPyramidPointsNormals(points, normals, pyrPoints, pyrNormals, (int)nLevels);
                for (size_t i = 1; i < nLevels; i++)
                {
                    frame->setPyramidAt(pyrPoints [i], OdometryFrame::PYR_CLOUD, i);
                    frame->setPyramidAt(pyrNormals[i], OdometryFrame::PYR_NORM,  i);
                }

                return points.size();
            }
            else
            {
                TMat cloud;
                frame->getPyramidAt(cloud, OdometryFrame::PYR_CLOUD, 0);
                std::vector<TMat> xyz;
                split(cloud, xyz);
                frame->setDepth(xyz[2]);
            }
        }
        else if (frame->getPyramidLevels(OdometryFrame::PYR_DEPTH))
        {
            TMat d0;
            frame->getPyramidAt(d0, OdometryFrame::PYR_DEPTH, 0);
            frame->setDepth(d0);
        }
        else
            CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
    }
    frame->getDepth(depth);
    checkDepth(depth, depth.size());

    // mask isn't used by FastICP
    auto tframe = frame.dynamicCast<OdometryFrameImpl<TMat>>();
    makeFrameFromDepth(depth, tframe->pyramids[OdometryFrame::PYR_CLOUD], tframe->pyramids[OdometryFrame::PYR_NORM], cameraMatrix, (int)iterCounts.size(),
                       depthFactor, sigmaDepth, sigmaSpatial, kernelSize, truncateThreshold);

    return depth.size();
}

Size FastICPOdometryImpl::prepareFrameCache(Ptr<OdometryFrame> frame, OdometryFrame::OdometryFrameCacheType cacheType) const
{
    auto oclFrame = frame.dynamicCast<OdometryFrameImpl<UMat>>();
    auto cpuFrame = frame.dynamicCast<OdometryFrameImpl<Mat>>();
    if (oclFrame != nullptr)
    {
        return prepareFrameCacheT<UMat>(frame, cacheType);
    }
    else if (cpuFrame != nullptr)
    {
        return prepareFrameCacheT<Mat>(frame, cacheType);
    }
    else
    {
        CV_Error(Error::StsBadArg, "Incorrect OdometryFrame type");
    }
}


bool FastICPOdometryImpl::computeImpl(const Ptr<OdometryFrame>& srcFrame,
                                      const Ptr<OdometryFrame>& dstFrame,
                                      OutputArray Rt, InputArray /*initRt*/) const
{
    Intr intr(cameraMatrix);
    Ptr<kinfu::ICP> icp = kinfu::makeICP(intr,
                                         iterCounts,
                                         angleThreshold,
                                         maxDistDiff);

    // KinFu's ICP calculates transformation from new frame to old one (src to dst)
    Affine3f transform;
    bool result;
    auto srcOclFrame = srcFrame.dynamicCast<OdometryFrameImpl<UMat>>();
    auto srcCpuFrame = srcFrame.dynamicCast<OdometryFrameImpl<Mat>>();
    auto dstOclFrame = dstFrame.dynamicCast<OdometryFrameImpl<UMat>>();
    auto dstCpuFrame = dstFrame.dynamicCast<OdometryFrameImpl<Mat>>();
    bool useOcl = (srcOclFrame != nullptr) && (dstOclFrame != nullptr);
    bool useCpu = (srcCpuFrame != nullptr) && (dstCpuFrame != nullptr);
    if (useOcl)
    {
        result = icp->estimateTransform(transform,
                                        dstOclFrame->pyramids[OdometryFrame::PYR_CLOUD], dstOclFrame->pyramids[OdometryFrame::PYR_NORM],
                                        srcOclFrame->pyramids[OdometryFrame::PYR_CLOUD], srcOclFrame->pyramids[OdometryFrame::PYR_NORM]);
    }
    else if (useCpu)
    {
        result = icp->estimateTransform(transform,
                                        dstCpuFrame->pyramids[OdometryFrame::PYR_CLOUD], dstCpuFrame->pyramids[OdometryFrame::PYR_NORM],
                                        srcCpuFrame->pyramids[OdometryFrame::PYR_CLOUD], srcCpuFrame->pyramids[OdometryFrame::PYR_NORM]);
    }
    else
    {
        CV_Error(Error::StsBadArg, "Both OdometryFrames should have the same storage type: Mat or UMat");
    }

    Rt.create(Size(4, 4), CV_64FC1);
    Mat(Matx44d(transform.matrix)).copyTo(Rt.getMat());

    result = result && testDeltaTransformation(Mat(Matx44d(transform.matrix)), maxTranslation, maxRotation);

    return result;
}

//

Ptr<Odometry> Odometry::createFromName(const std::string& odometryType)
{
    if (odometryType == "RgbdOdometry")
        return RgbdOdometry::create();
    else if (odometryType == "ICPOdometry")
        return ICPOdometry::create();
    else if (odometryType == "RgbdICPOdometry")
        return RgbdICPOdometry::create();
    else if (odometryType == "FastICPOdometry")
        return FastICPOdometry::create();
    return Ptr<Odometry>();
}

} // namespace cv
