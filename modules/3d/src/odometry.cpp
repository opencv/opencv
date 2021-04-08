// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

// This code is also subject to the license terms in the LICENSE_WillowGarage.md file found in this module's directory

#include "precomp.hpp"
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
namespace rgbd
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
void checkImage(const Mat& image)
{
    if(image.empty())
        CV_Error(Error::StsBadSize, "Image is empty.");
    if(image.type() != CV_8UC1)
        CV_Error(Error::StsBadSize, "Image type has to be CV_8UC1.");
}

static inline
void checkDepth(const Mat& depth, const Size& imageSize)
{
    if(depth.empty())
        CV_Error(Error::StsBadSize, "Depth is empty.");
    if(depth.size() != imageSize)
        CV_Error(Error::StsBadSize, "Depth has to have the size equal to the image size.");
    if(depth.type() != CV_32FC1)
        CV_Error(Error::StsBadSize, "Depth type has to be CV_32FC1.");
}

static inline
void checkMask(const Mat& mask, const Size& imageSize)
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
void checkNormals(const Mat& normals, const Size& depthSize)
{
    if(normals.size() != depthSize)
        CV_Error(Error::StsBadSize, "Normals has to have the size equal to the depth size.");
    if(normals.type() != CV_32FC3)
        CV_Error(Error::StsBadSize, "Normals type has to be CV_32FC3.");
}

static
void preparePyramidImage(const Mat& image, std::vector<Mat>& pyramidImage, size_t levelCount)
{
    if(!pyramidImage.empty())
    {
        if(pyramidImage.size() < levelCount)
            CV_Error(Error::StsBadSize, "Levels count of pyramidImage has to be equal or less than size of iterCounts.");

        CV_Assert(pyramidImage[0].size() == image.size());
        for(size_t i = 0; i < pyramidImage.size(); i++)
            CV_Assert(pyramidImage[i].type() == image.type());
    }
    else
        buildPyramid(image, pyramidImage, (int)levelCount - 1);
}

static
void preparePyramidDepth(const Mat& depth, std::vector<Mat>& pyramidDepth, size_t levelCount)
{
    if(!pyramidDepth.empty())
    {
        if(pyramidDepth.size() < levelCount)
            CV_Error(Error::StsBadSize, "Levels count of pyramidDepth has to be equal or less than size of iterCounts.");

        CV_Assert(pyramidDepth[0].size() == depth.size());
        for(size_t i = 0; i < pyramidDepth.size(); i++)
            CV_Assert(pyramidDepth[i].type() == depth.type());
    }
    else
        buildPyramid(depth, pyramidDepth, (int)levelCount - 1);
}

static
void preparePyramidMask(const Mat& mask, const std::vector<Mat>& pyramidDepth, float minDepth, float maxDepth,
                        const std::vector<Mat>& pyramidNormal,
                        std::vector<Mat>& pyramidMask)
{
    minDepth = std::max(0.f, minDepth);

    if(!pyramidMask.empty())
    {
        if(pyramidMask.size() != pyramidDepth.size())
            CV_Error(Error::StsBadSize, "Levels count of pyramidMask has to be equal to size of pyramidDepth.");

        for(size_t i = 0; i < pyramidMask.size(); i++)
        {
            CV_Assert(pyramidMask[i].size() == pyramidDepth[i].size());
            CV_Assert(pyramidMask[i].type() == CV_8UC1);
        }
    }
    else
    {
        Mat validMask;
        if(mask.empty())
            validMask = Mat(pyramidDepth[0].size(), CV_8UC1, Scalar(255));
        else
            validMask = mask.clone();

        buildPyramid(validMask, pyramidMask, (int)pyramidDepth.size() - 1);

        for(size_t i = 0; i < pyramidMask.size(); i++)
        {
            Mat levelDepth = pyramidDepth[i].clone();
            patchNaNs(levelDepth, 0);

            Mat& levelMask = pyramidMask[i];
            levelMask &= (levelDepth > minDepth) & (levelDepth < maxDepth);

            if(!pyramidNormal.empty())
            {
                CV_Assert(pyramidNormal[i].type() == CV_32FC3);
                CV_Assert(pyramidNormal[i].size() == pyramidDepth[i].size());
                Mat levelNormal = pyramidNormal[i].clone();

                Mat validNormalMask = levelNormal == levelNormal; // otherwise it's Nan
                CV_Assert(validNormalMask.type() == CV_8UC3);

                std::vector<Mat> channelMasks;
                split(validNormalMask, channelMasks);
                validNormalMask = channelMasks[0] & channelMasks[1] & channelMasks[2];

                levelMask &= validNormalMask;
            }
        }
    }
}

static
void preparePyramidCloud(const std::vector<Mat>& pyramidDepth, const Mat& cameraMatrix, std::vector<Mat>& pyramidCloud)
{
    if(!pyramidCloud.empty())
    {
        if(pyramidCloud.size() != pyramidDepth.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidCloud.");

        for(size_t i = 0; i < pyramidDepth.size(); i++)
        {
            CV_Assert(pyramidCloud[i].size() == pyramidDepth[i].size());
            CV_Assert(pyramidCloud[i].type() == CV_32FC3);
        }
    }
    else
    {
        std::vector<Mat> pyramidCameraMatrix;
        buildPyramidCameraMatrix(cameraMatrix, (int)pyramidDepth.size(), pyramidCameraMatrix);

        pyramidCloud.resize(pyramidDepth.size());
        for(size_t i = 0; i < pyramidDepth.size(); i++)
        {
            Mat cloud;
            depthTo3d(pyramidDepth[i], pyramidCameraMatrix[i], cloud);
            pyramidCloud[i] = cloud;
        }
    }
}

static
void preparePyramidSobel(const std::vector<Mat>& pyramidImage, int dx, int dy, std::vector<Mat>& pyramidSobel)
{
    if(!pyramidSobel.empty())
    {
        if(pyramidSobel.size() != pyramidImage.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidSobel.");

        for(size_t i = 0; i < pyramidSobel.size(); i++)
        {
            CV_Assert(pyramidSobel[i].size() == pyramidImage[i].size());
            CV_Assert(pyramidSobel[i].type() == CV_16SC1);
        }
    }
    else
    {
        pyramidSobel.resize(pyramidImage.size());
        for(size_t i = 0; i < pyramidImage.size(); i++)
        {
            Sobel(pyramidImage[i], pyramidSobel[i], CV_16S, dx, dy, sobelSize);
        }
    }
}

static
void randomSubsetOfMask(Mat& mask, float part)
{
    const int minPointsCount = 1000; // minimum point count (we can process them fast)
    const int nonzeros = countNonZero(mask);
    const int needCount = std::max(minPointsCount, int(mask.total() * part));
    if(needCount < nonzeros)
    {
        RNG rng;
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
        mask = subset;
    }
}

static
void preparePyramidTexturedMask(const std::vector<Mat>& pyramid_dI_dx, const std::vector<Mat>& pyramid_dI_dy,
                                const std::vector<float>& minGradMagnitudes, const std::vector<Mat>& pyramidMask, double maxPointsPart,
                                std::vector<Mat>& pyramidTexturedMask)
{
    if(!pyramidTexturedMask.empty())
    {
        if(pyramidTexturedMask.size() != pyramid_dI_dx.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidTexturedMask.");

        for(size_t i = 0; i < pyramidTexturedMask.size(); i++)
        {
            CV_Assert(pyramidTexturedMask[i].size() == pyramid_dI_dx[i].size());
            CV_Assert(pyramidTexturedMask[i].type() == CV_8UC1);
        }
    }
    else
    {
        const float sobelScale2_inv = 1.f / (float)(sobelScale * sobelScale);
        pyramidTexturedMask.resize(pyramid_dI_dx.size());
        for(size_t i = 0; i < pyramidTexturedMask.size(); i++)
        {
            const float minScaledGradMagnitude2 = minGradMagnitudes[i] * minGradMagnitudes[i] * sobelScale2_inv;
            const Mat& dIdx = pyramid_dI_dx[i];
            const Mat& dIdy = pyramid_dI_dy[i];

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
            pyramidTexturedMask[i] = texturedMask & pyramidMask[i];

            randomSubsetOfMask(pyramidTexturedMask[i], (float)maxPointsPart);
        }
    }
}

static
void preparePyramidNormals(const Mat& normals, const std::vector<Mat>& pyramidDepth, std::vector<Mat>& pyramidNormals)
{
    if(!pyramidNormals.empty())
    {
        if(pyramidNormals.size() != pyramidDepth.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidNormals.");

        for(size_t i = 0; i < pyramidNormals.size(); i++)
        {
            CV_Assert(pyramidNormals[i].size() == pyramidDepth[i].size());
            CV_Assert(pyramidNormals[i].type() == CV_32FC3);
        }
    }
    else
    {
        buildPyramid(normals, pyramidNormals, (int)pyramidDepth.size() - 1);
        // renormalize normals
        for(size_t i = 1; i < pyramidNormals.size(); i++)
        {
            Mat& currNormals = pyramidNormals[i];
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
void preparePyramidNormalsMask(const std::vector<Mat>& pyramidNormals, const std::vector<Mat>& pyramidMask, double maxPointsPart,
                               std::vector<Mat>& pyramidNormalsMask)
{
    if(!pyramidNormalsMask.empty())
    {
        if(pyramidNormalsMask.size() != pyramidMask.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidNormalsMask.");

        for(size_t i = 0; i < pyramidNormalsMask.size(); i++)
        {
            CV_Assert(pyramidNormalsMask[i].size() == pyramidMask[i].size());
            CV_Assert(pyramidNormalsMask[i].type() == pyramidMask[i].type());
        }
    }
    else
    {
        pyramidNormalsMask.resize(pyramidMask.size());

        for(size_t i = 0; i < pyramidNormalsMask.size(); i++)
        {
            pyramidNormalsMask[i] = pyramidMask[i].clone();
            Mat& normalsMask = pyramidNormalsMask[i];
            for(int y = 0; y < normalsMask.rows; y++)
            {
                const Vec3f *normals_row = pyramidNormals[i].ptr<Vec3f>(y);
                uchar *normalsMask_row = pyramidNormalsMask[i].ptr<uchar>(y);
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
        const Mat& srcLevelDepth = srcFrame->pyramidDepth[level];
        const Mat& dstLevelDepth = dstFrame->pyramidDepth[level];

        const double fx = levelCameraMatrix.at<double>(0,0);
        const double fy = levelCameraMatrix.at<double>(1,1);
        const double determinantThreshold = 1e-6;

        Mat AtA_rgbd, AtB_rgbd, AtA_icp, AtB_icp;
        Mat corresps_rgbd, corresps_icp;

        // Run transformation search on current level iteratively.
        for(int iter = 0; iter < iterCounts[level]; iter ++)
        {
            Mat resultRt_inv = resultRt.inv(DECOMP_SVD);

            if(method & RGBD_ODOMETRY)
                computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt_inv,
                                srcLevelDepth, srcFrame->pyramidMask[level], dstLevelDepth, dstFrame->pyramidTexturedMask[level],
                                maxDepthDiff, corresps_rgbd);

            if(method & ICP_ODOMETRY)
                computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt_inv,
                                srcLevelDepth, srcFrame->pyramidMask[level], dstLevelDepth, dstFrame->pyramidNormalsMask[level],
                                maxDepthDiff, corresps_icp);

            if(corresps_rgbd.rows < minCorrespsCount && corresps_icp.rows < minCorrespsCount)
                break;

            Mat AtA(transformDim, transformDim, CV_64FC1, Scalar(0)), AtB(transformDim, 1, CV_64FC1, Scalar(0));
            if(corresps_rgbd.rows >= minCorrespsCount)
            {
                calcRgbdLsmMatrices(srcFrame->pyramidImage[level], srcFrame->pyramidCloud[level], resultRt,
                                    dstFrame->pyramidImage[level], dstFrame->pyramid_dI_dx[level], dstFrame->pyramid_dI_dy[level],
                                    corresps_rgbd, fx, fy, sobelScale,
                                    AtA_rgbd, AtB_rgbd, rgbdEquationFuncPtr, transformDim);

                AtA += AtA_rgbd;
                AtB += AtB_rgbd;
            }
            if(corresps_icp.rows >= minCorrespsCount)
            {
                calcICPLsmMatrices(srcFrame->pyramidCloud[level], resultRt,
                                   dstFrame->pyramidCloud[level], dstFrame->pyramidNormals[level],
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
warpFrameImpl(const Mat& image, const Mat& depth, const Mat& mask,
              const Mat& Rt, const Mat& cameraMatrix, const Mat& distCoeff,
              OutputArray _warpedImage, OutputArray warpedDepth, OutputArray warpedMask)
{
    CV_Assert(image.size() == depth.size());

    Mat cloud;
    depthTo3d(depth, cameraMatrix, cloud);

    std::vector<Point2f> points2d;
    Mat transformedCloud;
    perspectiveTransform(cloud, transformedCloud, Rt);
    projectPoints(transformedCloud.reshape(3, 1), Mat::eye(3, 3, CV_64FC1), Mat::zeros(3, 1, CV_64FC1), cameraMatrix,
                distCoeff, points2d);

    _warpedImage.create(image.size(), image.type());
    Mat warpedImage = _warpedImage.getMat();

    Mat zBuffer(image.size(), CV_32FC1, std::numeric_limits<float>::max());
    const Rect rect = Rect(0, 0, image.cols, image.rows);

    for (int y = 0; y < image.rows; y++)
    {
        //const Point3f* cloud_row = cloud.ptr<Point3f>(y);
        const Point3f* transformedCloud_row = transformedCloud.ptr<Point3f>(y);
        const Point2f* points2d_row = &points2d[y*image.cols];
        const ImageElemType* image_row = image.ptr<ImageElemType>(y);
        const uchar* mask_row = mask.empty() ? 0 : mask.ptr<uchar>(y);
        for (int x = 0; x < image.cols; x++)
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

RgbdFrame::RgbdFrame() : ID(-1)
{}

RgbdFrame::RgbdFrame(const Mat& image_in, const Mat& depth_in, const Mat& mask_in, const Mat& normals_in, int ID_in)
    : ID(ID_in), image(image_in), depth(depth_in), mask(mask_in), normals(normals_in)
{}

RgbdFrame::~RgbdFrame()
{}

Ptr<RgbdFrame> RgbdFrame::create(const Mat& image_in, const Mat& depth_in, const Mat& mask_in, const Mat& normals_in, int ID_in) {
  return makePtr<RgbdFrame>(image_in, depth_in, mask_in, normals_in, ID_in);
}

void RgbdFrame::release()
{
    ID = -1;
    image.release();
    depth.release();
    mask.release();
    normals.release();
}

OdometryFrame::OdometryFrame() : RgbdFrame()
{}

OdometryFrame::OdometryFrame(const Mat& image_in, const Mat& depth_in, const Mat& mask_in, const Mat& normals_in, int ID_in)
    : RgbdFrame(image_in, depth_in, mask_in, normals_in, ID_in)
{}

Ptr<OdometryFrame> OdometryFrame::create(const Mat& image_in, const Mat& depth_in, const Mat& mask_in, const Mat& normals_in, int ID_in) {
  return makePtr<OdometryFrame>(image_in, depth_in, mask_in, normals_in, ID_in);
}

void OdometryFrame::release()
{
    RgbdFrame::release();
    releasePyramids();
}

void OdometryFrame::releasePyramids()
{
    pyramidImage.clear();
    pyramidDepth.clear();
    pyramidMask.clear();

    pyramidCloud.clear();

    pyramid_dI_dx.clear();
    pyramid_dI_dy.clear();
    pyramidTexturedMask.clear();

    pyramidNormals.clear();
    pyramidNormalsMask.clear();
}

bool Odometry::compute(const Mat& srcImage, const Mat& srcDepth, const Mat& srcMask,
                       const Mat& dstImage, const Mat& dstDepth, const Mat& dstMask,
                       OutputArray Rt, const Mat& initRt) const
{
    Ptr<OdometryFrame> srcFrame(new OdometryFrame(srcImage, srcDepth, srcMask));
    Ptr<OdometryFrame> dstFrame(new OdometryFrame(dstImage, dstDepth, dstMask));

    return compute(srcFrame, dstFrame, Rt, initRt);
}

bool Odometry::compute(Ptr<OdometryFrame>& srcFrame, Ptr<OdometryFrame>& dstFrame, OutputArray Rt, const Mat& initRt) const
{
    checkParams();

    Size srcSize = prepareFrameCache(srcFrame, OdometryFrame::CACHE_SRC);
    Size dstSize = prepareFrameCache(dstFrame, OdometryFrame::CACHE_DST);

    if(srcSize != dstSize)
        CV_Error(Error::StsBadSize, "srcFrame and dstFrame have to have the same size (resolution).");

    return computeImpl(srcFrame, dstFrame, Rt, initRt);
}

Size Odometry::prepareFrameCache(Ptr<OdometryFrame> &frame, int /*cacheType*/) const
{
    if (!frame)
        CV_Error(Error::StsBadArg, "Null frame pointer.");

    return Size();
}

Ptr<Odometry> Odometry::create(const String & odometryType)
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

Size RgbdOdometry::prepareFrameCache(Ptr<OdometryFrame>& frame, int cacheType) const
{
    Odometry::prepareFrameCache(frame, cacheType);

    if(frame->image.empty())
    {
        if(!frame->pyramidImage.empty())
            frame->image = frame->pyramidImage[0];
        else
            CV_Error(Error::StsBadSize, "Image or pyramidImage have to be set.");
    }
    checkImage(frame->image);

    if(frame->depth.empty())
    {
        if(!frame->pyramidDepth.empty())
            frame->depth = frame->pyramidDepth[0];
        else if(!frame->pyramidCloud.empty())
        {
            Mat cloud = frame->pyramidCloud[0];
            std::vector<Mat> xyz;
            split(cloud, xyz);
            frame->depth = xyz[2];
        }
        else
            CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
    }
    checkDepth(frame->depth, frame->image.size());

    if(frame->mask.empty() && !frame->pyramidMask.empty())
        frame->mask = frame->pyramidMask[0];
    checkMask(frame->mask, frame->image.size());

    preparePyramidImage(frame->image, frame->pyramidImage, iterCounts.total());

    preparePyramidDepth(frame->depth, frame->pyramidDepth, iterCounts.total());

    preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
                       frame->pyramidNormals, frame->pyramidMask);

    if(cacheType & OdometryFrame::CACHE_SRC)
        preparePyramidCloud(frame->pyramidDepth, cameraMatrix, frame->pyramidCloud);

    if(cacheType & OdometryFrame::CACHE_DST)
    {
        preparePyramidSobel(frame->pyramidImage, 1, 0, frame->pyramid_dI_dx);
        preparePyramidSobel(frame->pyramidImage, 0, 1, frame->pyramid_dI_dy);
        preparePyramidTexturedMask(frame->pyramid_dI_dx, frame->pyramid_dI_dy, minGradientMagnitudes,
                                   frame->pyramidMask, maxPointsPart, frame->pyramidTexturedMask);
    }

    return frame->image.size();
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

Size ICPOdometry::prepareFrameCache(Ptr<OdometryFrame>& frame, int cacheType) const
{
    Odometry::prepareFrameCache(frame, cacheType);

    if(frame->depth.empty())
    {
        if(!frame->pyramidDepth.empty())
            frame->depth = frame->pyramidDepth[0];
        else if(!frame->pyramidCloud.empty())
        {
            Mat cloud = frame->pyramidCloud[0];
            std::vector<Mat> xyz;
            split(cloud, xyz);
            frame->depth = xyz[2];
        }
        else
            CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
    }
    checkDepth(frame->depth, frame->depth.size());

    if(frame->mask.empty() && !frame->pyramidMask.empty())
        frame->mask = frame->pyramidMask[0];
    checkMask(frame->mask, frame->depth.size());

    preparePyramidDepth(frame->depth, frame->pyramidDepth, iterCounts.total());

    preparePyramidCloud(frame->pyramidDepth, cameraMatrix, frame->pyramidCloud);

    if(cacheType & OdometryFrame::CACHE_DST)
    {
        if(frame->normals.empty())
        {
            if(!frame->pyramidNormals.empty())
                frame->normals = frame->pyramidNormals[0];
            else
            {
                if(normalsComputer.empty() ||
                   normalsComputer->getRows() != frame->depth.rows ||
                   normalsComputer->getCols() != frame->depth.cols ||
                   norm(normalsComputer->getK(), cameraMatrix) > FLT_EPSILON)
                   normalsComputer = makePtr<RgbdNormals>(frame->depth.rows,
                                                          frame->depth.cols,
                                                          frame->depth.depth(),
                                                          cameraMatrix,
                                                          normalWinSize,
                                                          normalMethod);

                (*normalsComputer)(frame->pyramidCloud[0], frame->normals);
            }
        }
        checkNormals(frame->normals, frame->depth.size());

        preparePyramidNormals(frame->normals, frame->pyramidDepth, frame->pyramidNormals);

        preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
                           frame->pyramidNormals, frame->pyramidMask);

        preparePyramidNormalsMask(frame->pyramidNormals, frame->pyramidMask, maxPointsPart, frame->pyramidNormalsMask);
    }
    else
        preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
                           frame->pyramidNormals, frame->pyramidMask);

    return frame->depth.size();
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

Size RgbdICPOdometry::prepareFrameCache(Ptr<OdometryFrame>& frame, int cacheType) const
{
    if(frame->image.empty())
    {
        if(!frame->pyramidImage.empty())
            frame->image = frame->pyramidImage[0];
        else
            CV_Error(Error::StsBadSize, "Image or pyramidImage have to be set.");
    }
    checkImage(frame->image);

    if(frame->depth.empty())
    {
        if(!frame->pyramidDepth.empty())
            frame->depth = frame->pyramidDepth[0];
        else if(!frame->pyramidCloud.empty())
        {
            Mat cloud = frame->pyramidCloud[0];
            std::vector<Mat> xyz;
            split(cloud, xyz);
            frame->depth = xyz[2];
        }
        else
            CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
    }
    checkDepth(frame->depth, frame->image.size());

    if(frame->mask.empty() && !frame->pyramidMask.empty())
        frame->mask = frame->pyramidMask[0];
    checkMask(frame->mask, frame->image.size());

    preparePyramidImage(frame->image, frame->pyramidImage, iterCounts.total());

    preparePyramidDepth(frame->depth, frame->pyramidDepth, iterCounts.total());

    preparePyramidCloud(frame->pyramidDepth, cameraMatrix, frame->pyramidCloud);

    if(cacheType & OdometryFrame::CACHE_DST)
    {
        if(frame->normals.empty())
        {
            if(!frame->pyramidNormals.empty())
                frame->normals = frame->pyramidNormals[0];
            else
            {
                if(normalsComputer.empty() ||
                   normalsComputer->getRows() != frame->depth.rows ||
                   normalsComputer->getCols() != frame->depth.cols ||
                   norm(normalsComputer->getK(), cameraMatrix) > FLT_EPSILON)
                   normalsComputer = makePtr<RgbdNormals>(frame->depth.rows,
                                                          frame->depth.cols,
                                                          frame->depth.depth(),
                                                          cameraMatrix,
                                                          normalWinSize,
                                                          normalMethod);

                (*normalsComputer)(frame->pyramidCloud[0], frame->normals);
            }
        }
        checkNormals(frame->normals, frame->depth.size());

        preparePyramidNormals(frame->normals, frame->pyramidDepth, frame->pyramidNormals);

        preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
                           frame->pyramidNormals, frame->pyramidMask);

        preparePyramidSobel(frame->pyramidImage, 1, 0, frame->pyramid_dI_dx);
        preparePyramidSobel(frame->pyramidImage, 0, 1, frame->pyramid_dI_dy);
        preparePyramidTexturedMask(frame->pyramid_dI_dx, frame->pyramid_dI_dy,
                                   minGradientMagnitudes, frame->pyramidMask,
                                   maxPointsPart, frame->pyramidTexturedMask);

        preparePyramidNormalsMask(frame->pyramidNormals, frame->pyramidMask, maxPointsPart, frame->pyramidNormalsMask);
    }
    else
        preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
                           frame->pyramidNormals, frame->pyramidMask);

    return frame->image.size();
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
    maxDistDiff(DEFAULT_MAX_DEPTH_DIFF()),
    angleThreshold((float)(30. * CV_PI / 180.)),
    sigmaDepth(0.04f),
    sigmaSpatial(4.5f),
    kernelSize(7)
{
    setDefaultIterCounts(iterCounts);
}

FastICPOdometry::FastICPOdometry(const Mat& _cameraMatrix,
                                 float _maxDistDiff,
                                 float _angleThreshold,
                                 float _sigmaDepth,
                                 float _sigmaSpatial,
                                 int _kernelSize,
                                 const std::vector<int>& _iterCounts) :
    maxDistDiff(_maxDistDiff),
    angleThreshold(_angleThreshold),
    sigmaDepth(_sigmaDepth),
    sigmaSpatial(_sigmaSpatial),
    kernelSize(_kernelSize),
    iterCounts(Mat(_iterCounts).clone()),
    cameraMatrix(_cameraMatrix)
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
                                             const std::vector<int>& _iterCounts)
{
    return makePtr<FastICPOdometry>(_cameraMatrix, _maxDistDiff, _angleThreshold,
                                   _sigmaDepth, _sigmaSpatial, _kernelSize, _iterCounts);
}

Size FastICPOdometry::prepareFrameCache(Ptr<OdometryFrame>& frame, int cacheType) const
{
    Odometry::prepareFrameCache(frame, cacheType);

    if(frame->depth.empty())
    {
        if(!frame->pyramidDepth.empty())
            frame->depth = frame->pyramidDepth[0];
        else if(!frame->pyramidCloud.empty())
        {
            Mat cloud = frame->pyramidCloud[0];
            std::vector<Mat> xyz;
            split(cloud, xyz);
            frame->depth = xyz[2];
        }
        else
            CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
    }
    checkDepth(frame->depth, frame->depth.size());

    // mask isn't used by FastICP
    Intr intr(cameraMatrix);
    float depthFactor = 1.f; // user should rescale depth manually
    float truncateThreshold = 0.f; // disabled
    makeFrameFromDepth(frame->depth, frame->pyramidCloud, frame->pyramidNormals, intr, (int)iterCounts.total(),
                       depthFactor, sigmaDepth, sigmaSpatial, kernelSize, truncateThreshold);

    return frame->depth.size();
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
    kinfu::Intr intr(cameraMatrix);
    std::vector<int> iterations = iterCounts;
    Ptr<kinfu::ICP> icp = kinfu::makeICP(intr,
                                         iterations,
                                         angleThreshold,
                                         maxDistDiff);

    // KinFu's ICP calculates transformation from new frame to old one (src to dst)
    Affine3f transform;
    bool result = icp->estimateTransform(transform,
                                         dstFrame->pyramidCloud, dstFrame->pyramidNormals,
                                         srcFrame->pyramidCloud, srcFrame->pyramidNormals);

    Rt.create(Size(4, 4), CV_64FC1);
    Mat(Matx44d(transform.matrix)).copyTo(Rt.getMat());
    return result;
}

//

void
warpFrame(const Mat& image, const Mat& depth, const Mat& mask,
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
} // namespace rgbd
} // namespace cv
