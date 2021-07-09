// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

// This code is also subject to the license terms in the LICENSE_WillowGarage.md file found in this module's directory

#include "../precomp.hpp"
#include "utils.hpp"
#include "fast_icp.hpp"

#if defined(HAVE_EIGEN) && EIGEN_WORLD_VERSION == 3
#  define HAVE_EIGEN3_HERE
#  if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable:4701)  // potentially uninitialized local variable
#    pragma warning(disable:4702)  // unreachable code
#    pragma warning(disable:4714)  // const marked as __forceinline not inlined
#  endif
#  include <Eigen/Core>
#  include <unsupported/Eigen/MatrixFunctions>
#  include <Eigen/Dense>
#  if defined(_MSC_VER)
#    pragma warning(pop)
#  endif
#endif

namespace cv
{

enum
{
    RGBD_ODOMETRY = 1,
    ICP_ODOMETRY = 2,
    MERGED_ODOMETRY = RGBD_ODOMETRY + ICP_ODOMETRY
};

const int sobelSize = 3;
const double sobelScale = 1./8.;
int normalWinSize = 5;
int normalMethod = RgbdNormals::RGBD_NORMALS_METHOD_FALS;

static inline
void setDefaultIterCounts(Mat& iterCounts)
{
    iterCounts = Mat(Vec4i(7,7,7,10));
}

static inline
void setDefaultMinGradientMagnitudes(Mat& minGradientMagnitudes)
{
    minGradientMagnitudes = Mat(Vec4f(10,10,10,10));
}

static
void buildPyramidCameraMatrix(const Mat& cameraMatrix, int levels, std::vector<Mat>& pyramidCameraMatrix)
{
    pyramidCameraMatrix.resize(levels);

    Mat cameraMatrix_dbl;
    cameraMatrix.convertTo(cameraMatrix_dbl, CV_64FC1);

    for(int i = 0; i < levels; i++)
    {
        Mat levelCameraMatrix = i == 0 ? cameraMatrix_dbl : 0.5f * pyramidCameraMatrix[i-1];
        levelCameraMatrix.at<double>(2,2) = 1.;
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
void preparePyramidCloud(InputArrayOfArrays pyramidDepth, const Mat& cameraMatrix, InputOutputArrayOfArrays pyramidCloud)
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
        std::vector<Mat> pyramidCameraMatrix;
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

#ifdef HAVE_EIGEN3_HERE
    const double* ksi_ptr = ksi.ptr<const double>();
    Eigen::Matrix<double,4,4> twist, g;
    twist << 0.,          -ksi_ptr[2], ksi_ptr[1],  ksi_ptr[3],
             ksi_ptr[2],  0.,          -ksi_ptr[0], ksi_ptr[4],
             -ksi_ptr[1], ksi_ptr[0],  0,           ksi_ptr[5],
             0.,          0.,          0.,          0.;
    g = twist.exp();

    eigen2cv(g, Rt);
#else
    // TODO: check computeProjectiveMatrix when there is not eigen library,
    //       because it gives less accurate pose of the camera
    Rt = Mat::eye(4, 4, CV_64FC1);

    Mat R = Rt(Rect(0,0,3,3));
    Mat rvec = ksi.rowRange(0,3);

    Rodrigues(rvec, R);

    Rt.at<double>(0,3) = ksi.at<double>(3);
    Rt.at<double>(1,3) = ksi.at<double>(4);
    Rt.at<double>(2,3) = ksi.at<double>(5);
#endif
}

static
void computeCorresps(const Mat& K, const Mat& K_inv, const Mat& Rt,
                     const Mat& depth0, const Mat& validMask0,
                     const Mat& depth1, const Mat& selectMask1, float maxDepthDiff,
                     Mat& _corresps)
{
    CV_Assert(K.type() == CV_64FC1);
    CV_Assert(K_inv.type() == CV_64FC1);
    CV_Assert(Rt.type() == CV_64FC1);

    Mat corresps(depth1.size(), CV_16SC2, Scalar::all(-1));

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
                         const Mat& cameraMatrix,
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

    std::vector<Mat> pyramidCameraMatrix;
    buildPyramidCameraMatrix(cameraMatrix, (int)iterCounts.size(), pyramidCameraMatrix);

    Mat resultRt = initRt.empty() ? Mat::eye(4,4,CV_64FC1) : initRt.clone();
    Mat currRt, ksi;

    bool isOk = false;
    for(int level = (int)iterCounts.size() - 1; level >= 0; level--)
    {
        const Mat& levelCameraMatrix = pyramidCameraMatrix[level];
        const Mat& levelCameraMatrix_inv = levelCameraMatrix.inv(DECOMP_SVD);
        const Mat srcLevelDepth, dstLevelDepth;
        srcFrame->getPyramidAt(srcLevelDepth, OdometryFrame::PYR_DEPTH, level);
        dstFrame->getPyramidAt(dstLevelDepth, OdometryFrame::PYR_DEPTH, level);

        const double fx = levelCameraMatrix.at<double>(0,0);
        const double fy = levelCameraMatrix.at<double>(1,1);
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

///////////////////////////////////////////////////////////////////////////////////////////////

Ptr<RgbdNormals> RgbdNormals::create(int rows_in, int cols_in, int depth_in, InputArray K_in, int window_size_in, int method_in) {
  return makePtr<RgbdNormals>(rows_in, cols_in, depth_in, K_in, window_size_in, method_in);
}

Ptr<DepthCleaner> DepthCleaner::create(int depth_in, int window_size_in, int method_in) {
  return makePtr<DepthCleaner>(depth_in, window_size_in, method_in);
}

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

    virtual size_t getPyramidLevels(int what = PYR_IMAGE) CV_OVERRIDE
    {
        if (what < N_PYRAMIDS)
            return pyramids[what].size();
        else
            return 0;
    }

    virtual void setPyramidAt(InputArray  _pyrImage, int pyrType, size_t level) CV_OVERRIDE
    {
        TMat img = getTMat<TMat>(_pyrImage);
        pyramids[pyrType][level] = img;
    }

    CV_WRAP virtual void getPyramidAt(OutputArray _pyrImage, int pyrType, size_t level) CV_OVERRIDE
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


bool Odometry::compute(InputArray srcImage, InputArray srcDepth, InputArray srcMask,
                       InputArray dstImage, InputArray dstDepth, InputArray dstMask,
                       OutputArray Rt, const Mat& initRt) const
{
    //std::cout << "compute1" << std::endl;
    Ptr<OdometryFrame> srcFrame = makeOdometryFrame(srcImage, srcDepth, srcMask);
    Ptr<OdometryFrame> dstFrame = makeOdometryFrame(dstImage, dstDepth, dstMask);

    return compute(srcFrame, dstFrame, Rt, initRt);
}

bool Odometry::compute(Ptr<OdometryFrame> srcFrame, Ptr<OdometryFrame> dstFrame, OutputArray Rt, const Mat& initRt) const
{
    checkParams();
    //std::cout << "compute2" << std::endl;
    Size srcSize = prepareFrameCache(srcFrame, OdometryFrame::CACHE_SRC);
    Size dstSize = prepareFrameCache(dstFrame, OdometryFrame::CACHE_DST);

    if(srcSize != dstSize)
        CV_Error(Error::StsBadSize, "srcFrame and dstFrame have to have the same size (resolution).");

    return computeImpl(srcFrame, dstFrame, Rt, initRt);
}

Size Odometry::prepareFrameCache(Ptr<OdometryFrame> frame, int /*cacheType*/) const
{
    if (!frame)
        CV_Error(Error::StsBadArg, "Null frame pointer.");

    return Size();
}

Ptr<Odometry> Odometry::createFromName(const String & odometryType)
{
    if (odometryType == "RgbdOdometry")
        return makePtr<RgbdOdometry>();
    else if (odometryType == "ICPOdometry")
        return makePtr<ICPOdometry>();
    else if (odometryType == "RgbdICPOdometry")
        return makePtr<RgbdICPOdometry>();
    else if (odometryType == "FastICPOdometry")
        return makePtr<FastICPOdometry>();
    return Ptr<Odometry>();
}

//
RgbdOdometry::RgbdOdometry() :
    minDepth(DEFAULT_MIN_DEPTH()),
    maxDepth(DEFAULT_MAX_DEPTH()),
    maxDepthDiff(DEFAULT_MAX_DEPTH_DIFF()),
    maxPointsPart(DEFAULT_MAX_POINTS_PART()),
    transformType(Odometry::RIGID_BODY_MOTION),
    maxTranslation(DEFAULT_MAX_TRANSLATION()),
    maxRotation(DEFAULT_MAX_ROTATION())

{
    setDefaultIterCounts(iterCounts);
    setDefaultMinGradientMagnitudes(minGradientMagnitudes);
}

RgbdOdometry::RgbdOdometry(const Mat& _cameraMatrix,
                           float _minDepth, float _maxDepth, float _maxDepthDiff,
                           const std::vector<int>& _iterCounts,
                           const std::vector<float>& _minGradientMagnitudes,
                           float _maxPointsPart,
                           int _transformType) :
                           minDepth(_minDepth), maxDepth(_maxDepth), maxDepthDiff(_maxDepthDiff),
                           iterCounts(Mat(_iterCounts).clone()),
                           minGradientMagnitudes(Mat(_minGradientMagnitudes).clone()),
                           maxPointsPart(_maxPointsPart),
                           cameraMatrix(_cameraMatrix), transformType(_transformType),
                           maxTranslation(DEFAULT_MAX_TRANSLATION()), maxRotation(DEFAULT_MAX_ROTATION())
{
    if(iterCounts.empty() || minGradientMagnitudes.empty())
    {
        setDefaultIterCounts(iterCounts);
        setDefaultMinGradientMagnitudes(minGradientMagnitudes);
    }
}

Ptr<RgbdOdometry> RgbdOdometry::create(const Mat& _cameraMatrix, float _minDepth, float _maxDepth,
                 float _maxDepthDiff, const std::vector<int>& _iterCounts,
                 const std::vector<float>& _minGradientMagnitudes, float _maxPointsPart,
                 int _transformType) {
  return makePtr<RgbdOdometry>(_cameraMatrix, _minDepth, _maxDepth, _maxDepthDiff, _iterCounts, _minGradientMagnitudes, _maxPointsPart, _transformType);
}

Ptr<OdometryFrame> RgbdOdometry::makeOdometryFrame(InputArray _image, InputArray _depth, InputArray _mask) const
{
    // Can get rid of getMat() calls as soon as this Odometry algorithm supports UMats
    return OdometryFrame::create(_image.getMat(), _depth.getMat(), _mask.getMat());
}


Size RgbdOdometry::prepareFrameCache(Ptr<OdometryFrame> frame, int cacheType) const
{
    Odometry::prepareFrameCache(frame, cacheType);

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
    preparePyramidImage(image, tframe->pyramids[OdometryFrame::PYR_IMAGE], iterCounts.total());

    preparePyramidDepth(depth, tframe->pyramids[OdometryFrame::PYR_DEPTH], iterCounts.total());

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


void RgbdOdometry::checkParams() const
{
    CV_Assert(maxPointsPart > 0. && maxPointsPart <= 1.);
    CV_Assert(cameraMatrix.size() == Size(3,3) && (cameraMatrix.type() == CV_32FC1 || cameraMatrix.type() == CV_64FC1));
    CV_Assert(minGradientMagnitudes.size() == iterCounts.size() || minGradientMagnitudes.size() == iterCounts.t().size());
}

bool RgbdOdometry::computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt, const Mat& initRt) const
{
    return RGBDICPOdometryImpl(Rt, initRt, srcFrame, dstFrame, cameraMatrix, (float)maxDepthDiff, iterCounts, maxTranslation, maxRotation, RGBD_ODOMETRY, transformType);
}

//
ICPOdometry::ICPOdometry() :
    minDepth(DEFAULT_MIN_DEPTH()), maxDepth(DEFAULT_MAX_DEPTH()),
    maxDepthDiff(DEFAULT_MAX_DEPTH_DIFF()), maxPointsPart(DEFAULT_MAX_POINTS_PART()), transformType(Odometry::RIGID_BODY_MOTION),
    maxTranslation(DEFAULT_MAX_TRANSLATION()), maxRotation(DEFAULT_MAX_ROTATION())
{
    setDefaultIterCounts(iterCounts);
}

ICPOdometry::ICPOdometry(const Mat& _cameraMatrix,
                         float _minDepth, float _maxDepth, float _maxDepthDiff,
                         float _maxPointsPart, const std::vector<int>& _iterCounts,
                         int _transformType) :
                         minDepth(_minDepth), maxDepth(_maxDepth), maxDepthDiff(_maxDepthDiff),
                         maxPointsPart(_maxPointsPart), iterCounts(Mat(_iterCounts).clone()),
                         cameraMatrix(_cameraMatrix), transformType(_transformType),
                         maxTranslation(DEFAULT_MAX_TRANSLATION()), maxRotation(DEFAULT_MAX_ROTATION())
{
    if(iterCounts.empty())
        setDefaultIterCounts(iterCounts);
}

Ptr<ICPOdometry> ICPOdometry::create(const Mat& _cameraMatrix, float _minDepth, float _maxDepth,
                 float _maxDepthDiff, float _maxPointsPart, const std::vector<int>& _iterCounts,
                 int _transformType) {
  return makePtr<ICPOdometry>(_cameraMatrix, _minDepth, _maxDepth, _maxDepthDiff, _maxPointsPart, _iterCounts, _transformType);
}

Ptr<OdometryFrame> ICPOdometry::makeOdometryFrame(InputArray _image, InputArray _depth, InputArray _mask) const
{
    // Can get rid of getMat() calls as soon as this Odometry algorithm supports UMats
    return OdometryFrame::create(_image.getMat(), _depth.getMat(), _mask.getMat());
}


Size ICPOdometry::prepareFrameCache(Ptr<OdometryFrame> frame, int cacheType) const
{
    Odometry::prepareFrameCache(frame, cacheType);

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
    preparePyramidDepth(depth, tframe->pyramids[OdometryFrame::PYR_DEPTH], iterCounts.total());

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
                if(normalsComputer.empty() ||
                   normalsComputer->getRows() != depth.rows ||
                   normalsComputer->getCols() != depth.cols ||
                   norm(normalsComputer->getK(), cameraMatrix) > FLT_EPSILON)
                   normalsComputer = makePtr<RgbdNormals>(depth.rows,
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


void ICPOdometry::checkParams() const
{
    CV_Assert(maxPointsPart > 0. && maxPointsPart <= 1.);
    CV_Assert(cameraMatrix.size() == Size(3,3) && (cameraMatrix.type() == CV_32FC1 || cameraMatrix.type() == CV_64FC1));
}

bool ICPOdometry::computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt, const Mat& initRt) const
{
    return RGBDICPOdometryImpl(Rt, initRt, srcFrame, dstFrame, cameraMatrix, (float)maxDepthDiff, iterCounts, maxTranslation, maxRotation, ICP_ODOMETRY, transformType);
}

//
RgbdICPOdometry::RgbdICPOdometry() :
    minDepth(DEFAULT_MIN_DEPTH()), maxDepth(DEFAULT_MAX_DEPTH()),
    maxDepthDiff(DEFAULT_MAX_DEPTH_DIFF()), maxPointsPart(DEFAULT_MAX_POINTS_PART()), transformType(Odometry::RIGID_BODY_MOTION),
    maxTranslation(DEFAULT_MAX_TRANSLATION()), maxRotation(DEFAULT_MAX_ROTATION())
{
    setDefaultIterCounts(iterCounts);
    setDefaultMinGradientMagnitudes(minGradientMagnitudes);
}

RgbdICPOdometry::RgbdICPOdometry(const Mat& _cameraMatrix,
                                 float _minDepth, float _maxDepth, float _maxDepthDiff,
                                 float _maxPointsPart, const std::vector<int>& _iterCounts,
                                 const std::vector<float>& _minGradientMagnitudes,
                                 int _transformType) :
                                 minDepth(_minDepth), maxDepth(_maxDepth), maxDepthDiff(_maxDepthDiff),
                                 maxPointsPart(_maxPointsPart), iterCounts(Mat(_iterCounts).clone()),
                                 minGradientMagnitudes(Mat(_minGradientMagnitudes).clone()),
                                 cameraMatrix(_cameraMatrix), transformType(_transformType),
                                 maxTranslation(DEFAULT_MAX_TRANSLATION()), maxRotation(DEFAULT_MAX_ROTATION())
{
    if(iterCounts.empty() || minGradientMagnitudes.empty())
    {
        setDefaultIterCounts(iterCounts);
        setDefaultMinGradientMagnitudes(minGradientMagnitudes);
    }
}

Ptr<RgbdICPOdometry> RgbdICPOdometry::create(const Mat& _cameraMatrix, float _minDepth, float _maxDepth,
                 float _maxDepthDiff, float _maxPointsPart, const std::vector<int>& _iterCounts,
                 const std::vector<float>& _minGradientMagnitudes,
                 int _transformType) {
  return makePtr<RgbdICPOdometry>(_cameraMatrix, _minDepth, _maxDepth, _maxDepthDiff, _maxPointsPart, _iterCounts, _minGradientMagnitudes, _transformType);
}

Ptr<OdometryFrame> RgbdICPOdometry::makeOdometryFrame(InputArray _image, InputArray _depth, InputArray _mask) const
{
    // Can get rid of getMat() calls as soon as this Odometry algorithm supports UMats
    return OdometryFrame::create(_image.getMat(), _depth.getMat(), _mask.getMat());
}


Size RgbdICPOdometry::prepareFrameCache(Ptr<OdometryFrame> frame, int cacheType) const
{
    Odometry::prepareFrameCache(frame, cacheType);

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
    preparePyramidImage(image, tframe->pyramids[OdometryFrame::PYR_IMAGE], iterCounts.total());

    preparePyramidDepth(depth, tframe->pyramids[OdometryFrame::PYR_DEPTH], iterCounts.total());

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
                if(normalsComputer.empty() ||
                   normalsComputer->getRows() != depth.rows ||
                   normalsComputer->getCols() != depth.cols ||
                   norm(normalsComputer->getK(), cameraMatrix) > FLT_EPSILON)
                   normalsComputer = makePtr<RgbdNormals>(depth.rows,
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


void RgbdICPOdometry::checkParams() const
{
    CV_Assert(maxPointsPart > 0. && maxPointsPart <= 1.);
    CV_Assert(cameraMatrix.size() == Size(3,3) && (cameraMatrix.type() == CV_32FC1 || cameraMatrix.type() == CV_64FC1));
    CV_Assert(minGradientMagnitudes.size() == iterCounts.size() || minGradientMagnitudes.size() == iterCounts.t().size());
}

bool RgbdICPOdometry::computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt, const Mat& initRt) const
{
    return RGBDICPOdometryImpl(Rt, initRt, srcFrame, dstFrame, cameraMatrix, (float)maxDepthDiff, iterCounts,  maxTranslation, maxRotation, MERGED_ODOMETRY, transformType);
}

//

using namespace cv::kinfu;

FastICPOdometry::FastICPOdometry() :
    minDepth(DEFAULT_MIN_DEPTH()),
    maxDepth(DEFAULT_MAX_DEPTH()),
    maxDepthDiff(DEFAULT_MAX_DEPTH_DIFF()),
    maxDistDiff(DEFAULT_MAX_DEPTH_DIFF()),
    angleThreshold((float)(30. * CV_PI / 180.)),
    sigmaDepth(0.04f),
    sigmaSpatial(4.5f),
    kernelSize(7),
    depthFactor(1.f),
    truncateThreshold(0.f)
{
    setDefaultIterCounts(iterCounts);
}

FastICPOdometry::FastICPOdometry(const Mat& _cameraMatrix,
                                 float _maxDistDiff,
                                 float _angleThreshold,
                                 float _sigmaDepth,
                                 float _sigmaSpatial,
                                 int _kernelSize,
                                 const std::vector<int>& _iterCounts,
                                 float _depthFactor,
                                 float _truncateThreshold
                                ) :
    maxDistDiff(_maxDistDiff),
    angleThreshold(_angleThreshold),
    sigmaDepth(_sigmaDepth),
    sigmaSpatial(_sigmaSpatial),
    kernelSize(_kernelSize),
    iterCounts(Mat(_iterCounts).clone()),
    cameraMatrix(_cameraMatrix),
    depthFactor(_depthFactor),
    truncateThreshold(_truncateThreshold)
{
    if(iterCounts.empty())
        setDefaultIterCounts(iterCounts);
}

Ptr<FastICPOdometry> FastICPOdometry::create(const Mat& _cameraMatrix,
                                             float _maxDistDiff,
                                             float _angleThreshold,
                                             float _sigmaDepth,
                                             float _sigmaSpatial,
                                             int _kernelSize,
                                             const std::vector<int>& _iterCounts,
                                             float _depthFactor,
                                             float _truncateThreshold)
{
    return makePtr<FastICPOdometry>(_cameraMatrix, _maxDistDiff, _angleThreshold,
                                    _sigmaDepth, _sigmaSpatial, _kernelSize, _iterCounts,
                                    _depthFactor, _truncateThreshold);
}

Ptr<OdometryFrame> FastICPOdometry::makeOdometryFrame(InputArray _image, InputArray _depth, InputArray _mask) const
{
    return OdometryFrame::create(_image, _depth, _mask);
}

template<typename TMat>
Size FastICPOdometry::prepareFrameCacheT(Ptr<OdometryFrame> frame, int cacheType) const
{
    Odometry::prepareFrameCache(frame, cacheType);

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
                size_t nLevels = iterCounts.total();
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
    
    TMat mask;
    frame->getMask(mask);
    if (mask.empty() && frame->getPyramidLevels(OdometryFrame::PYR_MASK) > 0)
    {
        TMat pyr0;
        frame->getPyramidAt(pyr0, OdometryFrame::PYR_MASK, 0);
        frame->setMask(pyr0);
    }
    checkMask(mask, depth.size());


    auto tframe = frame.dynamicCast<OdometryFrameImpl<TMat>>();

    makeFrameFromDepth(depth, mask, tframe->pyramids[OdometryFrame::PYR_CLOUD], tframe->pyramids[OdometryFrame::PYR_NORM], tframe->pyramids[OdometryFrame::PYR_MASK], tframe->pyramids[OdometryFrame::PYR_NORMMASK], cameraMatrix, (int)iterCounts.total(),
                       depthFactor, sigmaDepth, sigmaSpatial, kernelSize, truncateThreshold);

    return depth.size();
}

Size FastICPOdometry::prepareFrameCache(Ptr<OdometryFrame> frame, int cacheType) const
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

void FastICPOdometry::checkParams() const
{
    CV_Assert(cameraMatrix.size() == Size(3,3) &&
              (cameraMatrix.type() == CV_32FC1 ||
               cameraMatrix.type() == CV_64FC1));

    CV_Assert(maxDistDiff > 0);
    CV_Assert(angleThreshold > 0);
    CV_Assert(sigmaDepth > 0 && sigmaSpatial > 0 && kernelSize > 0);
}

bool FastICPOdometry::computeImpl(const Ptr<OdometryFrame>& srcFrame,
                                  const Ptr<OdometryFrame>& dstFrame,
                                  OutputArray Rt, const Mat& /*initRt*/) const
{
    Intr intr(cameraMatrix);
    std::vector<int> iterations = iterCounts;
    Ptr<kinfu::ICP> icp = kinfu::makeICP(intr,
                                         iterations,
                                         angleThreshold,
                                         maxDistDiff);
    //std::cout << "computeImpl" << std::endl;

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
        //std::cout << "<== GPU ==>" << std::endl;
        result = icp->estimateTransform(transform,
                                        dstOclFrame->pyramids[OdometryFrame::PYR_CLOUD], dstOclFrame->pyramids[OdometryFrame::PYR_NORM],
                                        srcOclFrame->pyramids[OdometryFrame::PYR_CLOUD], srcOclFrame->pyramids[OdometryFrame::PYR_NORM],
                                        dstOclFrame->pyramids[OdometryFrame::PYR_MASK], dstOclFrame->pyramids[OdometryFrame::PYR_NORMMASK],
                                        srcOclFrame->pyramids[OdometryFrame::PYR_MASK], srcOclFrame->pyramids[OdometryFrame::PYR_NORMMASK]
        );
    }
    else if (useCpu)
    {
        result = icp->estimateTransform(transform,
                                        dstCpuFrame->pyramids[OdometryFrame::PYR_CLOUD], dstCpuFrame->pyramids[OdometryFrame::PYR_NORM],
                                        srcCpuFrame->pyramids[OdometryFrame::PYR_CLOUD], srcCpuFrame->pyramids[OdometryFrame::PYR_NORM],
                                        dstCpuFrame->pyramids[OdometryFrame::PYR_MASK], dstCpuFrame->pyramids[OdometryFrame::PYR_NORMMASK],
                                        srcCpuFrame->pyramids[OdometryFrame::PYR_MASK], srcCpuFrame->pyramids[OdometryFrame::PYR_NORMMASK]
        );
    }
    else
    {
        CV_Error(Error::StsBadArg, "Both OdometryFrames should have the same storage type: Mat or UMat");
    }

    Rt.create(Size(4, 4), CV_64FC1);
    Mat(Matx44d(transform.matrix)).copyTo(Rt.getMat());
    return result;
}

//
void
warpFrame(InputArray image, InputArray depth, InputArray mask,
          const Mat& Rt, const Mat& cameraMatrix, const Mat& distCoeff,
          OutputArray warpedImage, OutputArray warpedDepth, OutputArray warpedMask)
{
    if(image.type() == CV_8UC1)
        warpFrameImpl<uchar>(image, depth, mask, Rt, cameraMatrix, distCoeff, warpedImage, warpedDepth, warpedMask);
    else if(image.type() == CV_8UC3)
        warpFrameImpl<Point3_<uchar> >(image, depth, mask, Rt, cameraMatrix, distCoeff, warpedImage, warpedDepth, warpedMask);
    else
        CV_Error(Error::StsBadArg, "Image has to be type of CV_8UC1 or CV_8UC3");
}

} // namespace cv
