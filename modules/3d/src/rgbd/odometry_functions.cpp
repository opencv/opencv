// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "odometry_functions.hpp"
#include "../precomp.hpp"
#include "utils.hpp"
#include "opencl_kernels_3d.hpp"

#include "opencv2/imgproc.hpp"
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/dualquaternion.hpp>

namespace cv
{

static void randomSubsetOfMask(InputOutputArray _mask, float part)
{
    const int minPointsCount = 1000; // minimum point count (we can process them fast)
    const int nonzeros = countNonZero(_mask);
    const int needCount = std::max(minPointsCount, int(_mask.total() * part));
    if (needCount < nonzeros)
    {
        RNG rng;
        Mat mask = _mask.getMat();
        Mat subset(mask.size(), CV_8UC1, Scalar(0));

        int subsetSize = 0;
        while (subsetSize < needCount)
        {
            int y = rng(mask.rows);
            int x = rng(mask.cols);
            if (mask.at<uchar>(y, x))
            {
                subset.at<uchar>(y, x) = 255;
                mask.at<uchar>(y, x) = 0;
                subsetSize++;
            }
        }
        _mask.assign(subset);
    }
}


static UMat prepareScaledDepth(OdometryFrame& frame)
{
    UMat depth;
    frame.getDepth(depth);
    CV_Assert(!depth.empty());

    // Odometry works well with depth values in range [0, 10)
    // If it's bigger, let's scale it down by 5000, a typical depth factor
    double maxv;
    cv::minMaxLoc(depth, nullptr, &maxv);
    UMat depthFlt;
    depth.convertTo(depthFlt, CV_32FC1, maxv > 10 ? (1.f / 5000.f) : 1.f);
    patchNaNs(depthFlt, 0);
    frame.impl->scaledDepth = depthFlt;

    return depthFlt;
}


//TODO: this with bilateral, maybe masks
static void preparePyramidDepth(UMat depth, std::vector<UMat>& dpyramids, int maxLevel)
{
    //TODO: make resize down instead
    buildPyramid(depth, dpyramids, maxLevel);
    //resize(depth, resized, dsize, fx, fy, INTER_NEAREST);
}


static void preparePyramidMask(UMat mask, const std::vector<UMat> pyramidDepth, int nLevels, float minDepth, float maxDepth,
                               std::vector<UMat>& pyramidMask)
{
    minDepth = std::max(0.f, minDepth);

    buildPyramid(mask, pyramidMask, nLevels - 1);

    for (int i = 0; i < nLevels; i++)
    {
        UMat maski = pyramidMask[i];
        threshold(maski, maski, 254, 255, THRESH_TOZERO);
        const UMat depthi = pyramidDepth[i];

        UMat gtmin, ltmax, tmpMask;
        cv::compare(depthi, Scalar(minDepth), gtmin, CMP_GT);
        cv::compare(depthi, Scalar(maxDepth), ltmax, CMP_LT);
        cv::bitwise_and(gtmin, ltmax, tmpMask);
        cv::bitwise_and(maski, tmpMask, maski);
    }
}

static void extendPyrMaskByPyrNormals(const std::vector<UMat>& pyramidNormals,  std::vector<UMat>& pyramidMask)
{
    if (!pyramidNormals.empty())
    {
        int nLevels = (int)pyramidNormals.size();

        for (int i = 0; i < nLevels; i++)
        {
            UMat maski = pyramidMask[i];
            UMat normali = pyramidNormals[i];
            UMat validNormalMask;
            finiteMask(normali, validNormalMask);
            cv::bitwise_and(maski, validNormalMask, maski);
        }
    }
}

static void buildPyramidCameraMatrix(const Matx33f& cameraMatrix, int levels, std::vector<Matx33f>& pyramidCameraMatrix)
{
    pyramidCameraMatrix.resize(levels);

    for (int i = 0; i < levels; i++)
    {
        Matx33f levelCameraMatrix = (i == 0) ? cameraMatrix : 0.5f * pyramidCameraMatrix[i - 1];
        levelCameraMatrix(2, 2) = 1.0;
        pyramidCameraMatrix[i] = levelCameraMatrix;
    }
}

static void preparePyramidCloud(const std::vector<UMat>& pyramidDepth, const Matx33f& cameraMatrix, std::vector<UMat>& pyramidCloud)
{
    int nLevels = (int)pyramidDepth.size();

    std::vector<Matx33f> pyramidCameraMatrix;
    buildPyramidCameraMatrix(cameraMatrix, nLevels, pyramidCameraMatrix);

    pyramidCloud.resize(nLevels, UMat());

    for (int i = 0; i < nLevels; i++)
    {
        UMat cloud;
        depthTo3d(pyramidDepth[i], pyramidCameraMatrix[i], cloud, noArray());
        pyramidCloud[i] = cloud;
    }
}


static void preparePyramidTexturedMask(const std::vector<UMat>& pyramid_dI_dx, const std::vector<UMat>& pyramid_dI_dy,
                                       std::vector<float> minGradMagnitudes, const std::vector<UMat>& pyramidMask, double maxPointsPart,
                                       std::vector<UMat>& pyramidTexturedMask, double sobelScale)
{
    int nLevels = (int)pyramid_dI_dx.size();

    const float sobelScale2_inv = (float)(1. / (sobelScale * sobelScale));
    pyramidTexturedMask.resize(nLevels, UMat());
    for (int i = 0; i < nLevels; i++)
    {
        const float minScaledGradMagnitude2 = minGradMagnitudes[i] * minGradMagnitudes[i] * sobelScale2_inv;
        const Mat dIdx = pyramid_dI_dx[i].getMat(ACCESS_READ);
        const Mat dIdy = pyramid_dI_dy[i].getMat(ACCESS_READ);

        Mat texturedMask(dIdx.size(), CV_8UC1, Scalar(0));

        for (int y = 0; y < dIdx.rows; y++)
        {
            const short *dIdx_row = dIdx.ptr<short>(y);
            const short *dIdy_row = dIdy.ptr<short>(y);
            uchar *texturedMask_row = texturedMask.ptr<uchar>(y);
            for (int x = 0; x < dIdx.cols; x++)
            {
                float magnitude2 = static_cast<float>(dIdx_row[x] * dIdx_row[x] + dIdy_row[x] * dIdy_row[x]);
                if (magnitude2 >= minScaledGradMagnitude2)
                    texturedMask_row[x] = 255;
            }
        }
        Mat texMask = texturedMask & pyramidMask[i].getMat(ACCESS_READ);
        randomSubsetOfMask(texMask, (float)maxPointsPart);
        texMask.copyTo(pyramidTexturedMask[i]);
    }
}


static void preparePyramidNormals(const UMat &normals, const std::vector<UMat> &pyramidDepth, std::vector<UMat> &pyramidNormals)
{
    int nLevels = (int)pyramidDepth.size();

    buildPyramid(normals, pyramidNormals, nLevels - 1);
    // renormalize normals
    for (int i = 1; i < nLevels; i++)
    {
        Mat currNormals = pyramidNormals[i].getMat(ACCESS_RW);
        CV_Assert(currNormals.type() == CV_32FC4);
        for (int y = 0; y < currNormals.rows; y++)
        {
            Vec4f *normals_row = currNormals.ptr<Vec4f>(y);
            for (int x = 0; x < currNormals.cols; x++)
            {
                Vec4f n4 = normals_row[x];
                Point3f n(n4[0], n4[1], n4[2]);
                double nrm = norm(n);
                n *= 1.f / nrm;
                normals_row[x] = Vec4f(n.x, n.y, n.z, 0);
            }
        }
    }
}


static void preparePyramidNormalsMask(const std::vector<UMat> &pyramidNormals, const std::vector<UMat> &pyramidMask, double maxPointsPart,
                                      std::vector<UMat> &pyramidNormalsMask)
{
    int nLevels = (int)pyramidNormals.size();
    pyramidNormalsMask.resize(nLevels, UMat());
    for (int i = 0; i < nLevels; i++)
    {
        Mat pyrMask = pyramidMask[i].getMat(ACCESS_READ);

        const Mat normals = pyramidNormals[i].getMat(ACCESS_READ);
        Mat_<uchar> normalsMask(pyrMask.size(), (uchar)255);
        for (int y = 0; y < normalsMask.rows; y++)
        {
            const Vec4f *normals_row = normals.ptr<Vec4f>(y);
            uchar *normalsMask_row = normalsMask.ptr<uchar>(y);
            for (int x = 0; x < normalsMask.cols; x++)
            {
                Vec4f n = normals_row[x];
                if (!(std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])))
                {
                    normalsMask_row[x] = 0;
                }
            }
        }
        cv::bitwise_and(pyrMask, normalsMask, normalsMask);

        randomSubsetOfMask(normalsMask, (float)maxPointsPart);
        normalsMask.copyTo(pyramidNormalsMask[i]);
    }
}


static void prepareRGBFrameBase(OdometryFrame& frame, OdometrySettings settings)
{
    UMat grayImage;
    frame.getGrayImage(grayImage);
    if (grayImage.empty())
    {
        UMat image;
        frame.getImage(image);
        CV_Assert(!image.empty() && image.depth() == CV_8U);

        int ch = image.channels();
        if (ch == 3 || ch == 4)
            cvtColor(image, grayImage, ch == 3 ? COLOR_BGR2GRAY : COLOR_BGRA2GRAY, 1);
        else if (ch == 1)
                grayImage = image;
        else
            CV_Error(Error::StsBadArg, "Image should have 3 or 4 channels (RGB) or 1 channel (grayscale)");
        grayImage.convertTo(grayImage, CV_8U);
        frame.impl->imageGray = grayImage;
    }

    //TODO: don't use scaled when scale bug is fixed
    UMat scaledDepth;
    frame.getProcessedDepth(scaledDepth);
    if (scaledDepth.empty())
    {
        scaledDepth = prepareScaledDepth(frame);
        CV_Assert(scaledDepth.size() == grayImage.size());
    }

    UMat depthMask;
    // ignore small, negative, Inf, NaN values
    cv::compare(scaledDepth, Scalar(FLT_EPSILON), depthMask, CMP_GT);

    UMat mask;
    frame.getMask(mask);
    if (mask.empty())
    {
        frame.impl->mask = depthMask;
    }
    else
    {
        CV_Assert(mask.type() == CV_8UC1 || mask.type() == CV_8SC1 || mask.type() == CV_BoolC1);
        CV_Assert(mask.size() == grayImage.size());
        cv::bitwise_and(mask, depthMask, frame.impl->mask);
    }
    frame.getMask(mask);

    std::vector<int> iterCounts;
    settings.getIterCounts(iterCounts);

    int maxLevel = (int)iterCounts.size() - 1;
    std::vector<UMat>& ipyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_IMAGE];
    if (ipyramids.empty())
        buildPyramid(grayImage, ipyramids, maxLevel);

    std::vector<UMat>& dpyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_DEPTH];
    if (dpyramids.empty())
        preparePyramidDepth(scaledDepth, dpyramids, maxLevel);

    std::vector<UMat>& mpyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_MASK];
    if (mpyramids.empty())
        preparePyramidMask(mask, dpyramids, maxLevel + 1, settings.getMinDepth(), settings.getMaxDepth(), mpyramids);
}


static void prepareRGBFrameSrc(OdometryFrame& frame, OdometrySettings settings)
{
    Matx33f cameraMatrix;
    settings.getCameraMatrix(cameraMatrix);

    std::vector<UMat>&       cpyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_CLOUD];
    const std::vector<UMat>& dpyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_DEPTH];

    preparePyramidCloud(dpyramids, cameraMatrix, cpyramids);
}


static void prepareRGBFrameDst(OdometryFrame& frame, OdometrySettings settings)
{
    const std::vector<UMat>& ipyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_IMAGE];
    const std::vector<UMat>& mpyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_MASK];

    std::vector<UMat>& dxpyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_DIX];
    std::vector<UMat>& dypyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_DIY];

    std::vector<float> minGradientMagnitudes;
    settings.getMinGradientMagnitudes(minGradientMagnitudes);

    int nLevels = (int)ipyramids.size();
    dxpyramids.resize(nLevels, UMat());
    dypyramids.resize(nLevels, UMat());
    int sobelSize = settings.getSobelSize();
    for (int i = 0; i < nLevels; i++)
    {
        Sobel(ipyramids[i], dxpyramids[i], CV_16S, 1, 0, sobelSize);
        Sobel(ipyramids[i], dypyramids[i], CV_16S, 0, 1, sobelSize);
    }

    std::vector<UMat>& tmpyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_TEXMASK];
    preparePyramidTexturedMask(dxpyramids, dypyramids, minGradientMagnitudes,
                               mpyramids, settings.getMaxPointsPart(), tmpyramids, settings.getSobelScale());
}


static void prepareICPFrameBase(OdometryFrame& frame, OdometrySettings settings)
{
    //TODO: don't use scaled when scale bug is fixed
    UMat scaledDepth;
    frame.getProcessedDepth(scaledDepth);
    if (scaledDepth.empty())
    {
        scaledDepth = prepareScaledDepth(frame);
    }

    UMat depthMask;
    // ignore small, negative, Inf, NaN values
    cv::compare(scaledDepth, Scalar(FLT_EPSILON), depthMask, CMP_GT);

    UMat mask;
    frame.getMask(mask);
    if (mask.empty())
    {
        frame.impl->mask = depthMask;
    }
    else
    {
        CV_Assert(mask.type() == CV_8UC1 || mask.type() == CV_8SC1 || mask.type() == CV_BoolC1);
        CV_Assert(mask.size() == scaledDepth.size());
        cv::bitwise_and(mask, depthMask, frame.impl->mask);
    }
    frame.getMask(mask);

    std::vector<int> iterCounts;
    settings.getIterCounts(iterCounts);

    int maxLevel = (int)iterCounts.size() - 1;
    std::vector<UMat>& dpyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_DEPTH];
    if (dpyramids.empty())
        preparePyramidDepth(scaledDepth, dpyramids, maxLevel);

    Matx33f cameraMatrix;
    settings.getCameraMatrix(cameraMatrix);

    std::vector<UMat>& cpyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_CLOUD];
    if (cpyramids.empty())
        preparePyramidCloud(dpyramids, cameraMatrix, cpyramids);
}


static void prepareICPFrameSrc(OdometryFrame& frame, OdometrySettings settings)
{
    UMat mask;
    frame.getMask(mask);

    const std::vector<UMat>& dpyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_DEPTH];

    std::vector<int> iterCounts;
    settings.getIterCounts(iterCounts);
    std::vector<UMat>& mpyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_MASK];
    if (mpyramids.empty())
        preparePyramidMask(mask, dpyramids, (int)iterCounts.size(), settings.getMinDepth(), settings.getMaxDepth(), mpyramids);
}


static void prepareICPFrameDst(OdometryFrame& frame, OdometrySettings settings, Ptr<RgbdNormals>& normalsComputer)
{
    Matx33f cameraMatrix;
    settings.getCameraMatrix(cameraMatrix);

    UMat scaledDepth, mask, normals;
    frame.getProcessedDepth(scaledDepth);
    frame.getMask(mask);
    frame.getNormals(normals);

    if (normals.empty())
    {
        Matx33f K;
        if (!normalsComputer.empty())
            normalsComputer->getK(K);
        if (normalsComputer.empty() ||
            normalsComputer->getRows() != scaledDepth.rows ||
            normalsComputer->getCols() != scaledDepth.cols ||
            norm(K, cameraMatrix) > FLT_EPSILON)
        {
            int normalWinSize = settings.getNormalWinSize();
            float diffThreshold = settings.getNormalDiffThreshold();
            RgbdNormals::RgbdNormalsMethod normalMethod = settings.getNormalMethod();
            normalsComputer = RgbdNormals::create(scaledDepth.rows,
                                                  scaledDepth.cols,
                                                  scaledDepth.depth(),
                                                  cameraMatrix,
                                                  normalWinSize,
                                                  diffThreshold,
                                                  normalMethod);
        }
        const UMat& c0 = frame.impl->pyramids[OdometryFramePyramidType::PYR_CLOUD][0];
        normalsComputer->apply(c0, normals);
        frame.impl->normals = normals;
    }
    CV_Assert(normals.type() == CV_32FC4);

    const std::vector<UMat>& dpyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_DEPTH];

    std::vector<UMat>& npyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_NORM];
    if (npyramids.empty())
        preparePyramidNormals(normals, dpyramids, npyramids);

    std::vector<UMat>& mpyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_MASK];
    if (mpyramids.empty())
    {
        std::vector<int> iterCounts;
        settings.getIterCounts(iterCounts);
        preparePyramidMask(mask, dpyramids, (int)iterCounts.size(), settings.getMinDepth(), settings.getMaxDepth(), mpyramids);
        extendPyrMaskByPyrNormals(npyramids, mpyramids);
    }

    std::vector<UMat>& nmpyramids = frame.impl->pyramids[OdometryFramePyramidType::PYR_NORMMASK];
    if (nmpyramids.empty())
        preparePyramidNormalsMask(npyramids, mpyramids, settings.getMaxPointsPart(), nmpyramids);
}


void prepareRGBFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, OdometrySettings settings)
{
    prepareRGBFrameBase(srcFrame, settings);
    prepareRGBFrameBase(dstFrame, settings);

    prepareRGBFrameSrc(srcFrame, settings);
    prepareRGBFrameDst(dstFrame, settings);
}

void prepareICPFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, Ptr<RgbdNormals>& normalsComputer, OdometrySettings settings, OdometryAlgoType algtype)
{
    prepareICPFrameBase(srcFrame, settings);
    prepareICPFrameBase(dstFrame, settings);

    prepareICPFrameSrc(srcFrame, settings);
    if (algtype == OdometryAlgoType::FAST)
        prepareICPFrameDst(srcFrame, settings, normalsComputer);
    prepareICPFrameDst(dstFrame, settings, normalsComputer);
}

void prepareRGBDFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, Ptr<RgbdNormals>& normalsComputer, OdometrySettings settings, OdometryAlgoType algtype)
{
    prepareRGBFrame(srcFrame, dstFrame, settings);
    prepareICPFrame(srcFrame, dstFrame, normalsComputer, settings, algtype);
}

bool RGBDICPOdometryImpl(OutputArray _Rt, const Mat& initRt,
                         const OdometryFrame& srcFrame,
                         const OdometryFrame& dstFrame,
                         const Matx33f& cameraMatrix,
                         float maxDepthDiff, float angleThreshold, const std::vector<int>& iterCounts,
                         double maxTranslation, double maxRotation, double sobelScale,
                         OdometryType method, OdometryTransformType transformType, OdometryAlgoType algtype)
{
    if (srcFrame.getPyramidLevels() != (int)(iterCounts.size()))
        CV_Error(Error::StsBadArg, "srcFrame has incorrect number of pyramid levels. Did you forget to call prepareFrame()?");
    if (dstFrame.getPyramidLevels() != (int)(iterCounts.size()))
        CV_Error(Error::StsBadArg, "dstFrame has incorrect number of pyramid levels. Did you forget to call prepareFrame()?");

    int transformDim = getTransformDim(transformType);

    const int minOverdetermScale = 20;
    const int minCorrespsCount = minOverdetermScale * transformDim;

    std::vector<Matx33f> pyramidCameraMatrix;
    buildPyramidCameraMatrix(cameraMatrix, (int)iterCounts.size(), pyramidCameraMatrix);

    Mat resultRt = initRt.empty() ? Mat::eye(4,4,CV_64FC1) : initRt.clone();
    Mat currRt, ksi;
    Affine3f transform = Affine3f::Identity();

    bool isOk = false;
    for(int level = (int)iterCounts.size() - 1; level >= 0; level--)
    {
        const Matx33f& levelCameraMatrix = pyramidCameraMatrix[level];
        const Mat srcLevelDepth, dstLevelDepth;
        const Mat srcLevelImage, dstLevelImage;
        srcFrame.getPyramidAt(srcLevelDepth, OdometryFramePyramidType::PYR_DEPTH, level);
        dstFrame.getPyramidAt(dstLevelDepth, OdometryFramePyramidType::PYR_DEPTH, level);

        if (method != OdometryType::DEPTH) // RGB(D)
        {
            srcFrame.getPyramidAt(srcLevelImage, OdometryFramePyramidType::PYR_IMAGE, level);
            dstFrame.getPyramidAt(dstLevelImage, OdometryFramePyramidType::PYR_IMAGE, level);
        }

        const double fx = levelCameraMatrix(0, 0);
        const double fy = levelCameraMatrix(1, 1);
        const double determinantThreshold = 1e-6;

        Mat AtA_rgbd, AtB_rgbd, AtA_icp, AtB_icp;

        // Run transformation search on current level iteratively.
        for(int iter = 0; iter < iterCounts[level]; iter ++)
        {
            Mat resultRt_inv = resultRt.inv(DECOMP_SVD);
            Mat corresps_rgbd, corresps_icp, diffs_rgbd;
            Mat dummy;
            double sigma_rgbd = 0, dummyFloat = 0;

            const Mat pyramidMask;
            srcFrame.getPyramidAt(pyramidMask, OdometryFramePyramidType::PYR_MASK, level);

            if(method != OdometryType::DEPTH) // RGB(D)
            {
                const Mat pyramidTexturedMask;
                dstFrame.getPyramidAt(pyramidTexturedMask, OdometryFramePyramidType::PYR_TEXMASK, level);
                computeCorresps(levelCameraMatrix, resultRt,
                                srcLevelImage, srcLevelDepth, pyramidMask,
                                dstLevelImage, dstLevelDepth, pyramidTexturedMask, maxDepthDiff,
                                corresps_rgbd, diffs_rgbd, sigma_rgbd, OdometryType::RGB);
            }

            if(method != OdometryType::RGB) // ICP, RGBD
            {
                if (algtype == OdometryAlgoType::COMMON)
                {
                    const Mat pyramidNormalsMask;
                    dstFrame.getPyramidAt(pyramidNormalsMask, OdometryFramePyramidType::PYR_NORMMASK, level);
                    computeCorresps(levelCameraMatrix, resultRt,
                                    Mat(), srcLevelDepth, pyramidMask,
                                    Mat(), dstLevelDepth, pyramidNormalsMask, maxDepthDiff,
                                    corresps_icp, dummy, dummyFloat, OdometryType::DEPTH);
                }
            }

            if(corresps_rgbd.rows < minCorrespsCount && corresps_icp.rows < minCorrespsCount && algtype != OdometryAlgoType::FAST)
                break;

            const UMat srcPyrCloud;
            srcFrame.getPyramidAt(srcPyrCloud, OdometryFramePyramidType::PYR_CLOUD, level);


            Mat AtA(transformDim, transformDim, CV_64FC1, Scalar(0)), AtB(transformDim, 1, CV_64FC1, Scalar(0));
            if(corresps_rgbd.rows >= minCorrespsCount)
            {
                const Mat srcPyrImage, dstPyrImage, dstPyrIdx, dstPyrIdy;
                srcFrame.getPyramidAt(srcPyrImage, OdometryFramePyramidType::PYR_IMAGE, level);
                dstFrame.getPyramidAt(dstPyrImage, OdometryFramePyramidType::PYR_IMAGE, level);
                dstFrame.getPyramidAt(dstPyrIdx, OdometryFramePyramidType::PYR_DIX, level);
                dstFrame.getPyramidAt(dstPyrIdy, OdometryFramePyramidType::PYR_DIY, level);
                calcRgbdLsmMatrices(srcPyrCloud.getMat(ACCESS_READ), resultRt, dstPyrIdx, dstPyrIdy,
                                    corresps_rgbd, diffs_rgbd, sigma_rgbd, fx, fy, sobelScale,
                                    AtA_rgbd, AtB_rgbd, transformType);
                AtA += AtA_rgbd;
                AtB += AtB_rgbd;
            }
            if(corresps_icp.rows >= minCorrespsCount || algtype == OdometryAlgoType::FAST)
            {
                if (algtype == OdometryAlgoType::COMMON)
                {
                    const Mat dstPyrCloud, dstPyrNormals;
                    dstFrame.getPyramidAt(dstPyrCloud, OdometryFramePyramidType::PYR_CLOUD, level);
                    dstFrame.getPyramidAt(dstPyrNormals, OdometryFramePyramidType::PYR_NORM, level);
                    calcICPLsmMatrices(srcPyrCloud.getMat(ACCESS_READ), resultRt, dstPyrCloud, dstPyrNormals,
                                       corresps_icp, AtA_icp, AtB_icp, transformType);
                }
                else
                {
                    const UMat dstPyrCloud, dstPyrNormals, srcPyrNormals;
                    dstFrame.getPyramidAt(dstPyrCloud, OdometryFramePyramidType::PYR_CLOUD, level);
                    dstFrame.getPyramidAt(dstPyrNormals, OdometryFramePyramidType::PYR_NORM, level);
                    srcFrame.getPyramidAt(srcPyrNormals, OdometryFramePyramidType::PYR_NORM, level);
                    cv::Matx66f A;
                    cv::Vec6f b;
                    calcICPLsmMatricesFast(cameraMatrix, dstPyrCloud, dstPyrNormals, srcPyrCloud, srcPyrNormals, transform, level, maxDepthDiff, angleThreshold, A, b);
                    AtA_icp = Mat(A);
                    AtB_icp = Mat(b);
                }

                AtA += AtA_icp;
                AtB += AtB_icp;
            }

            bool solutionExist = solveSystem(AtA, AtB, determinantThreshold, ksi);

            if (!solutionExist)
            {
                break;
            }

            Mat tmp61(6, 1, CV_64FC1, Scalar(0));
            if(transformType == OdometryTransformType::ROTATION)
            {
                ksi.copyTo(tmp61.rowRange(0,3));
                ksi = tmp61;
            }
            else if(transformType == OdometryTransformType::TRANSLATION)
            {
                ksi.copyTo(tmp61.rowRange(3,6));
                ksi = tmp61;
            }

            computeProjectiveMatrix(ksi, currRt);
            resultRt = currRt * resultRt;

            //TODO: fixit, transform is used for Fast ICP only
            Vec6f x(ksi);
            Affine3f tinc(Vec3f(x.val), Vec3f(x.val + 3));
            transform = tinc * transform;

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

// Rotate dst by RtInv to get corresponding src pixels
// In RGB case compute sigma and diffs too
void computeCorresps(const Matx33f& _K, const Mat& rt,
                     const Mat& imageSrc, const Mat& depthSrc, const Mat& validMaskSrc,
                     const Mat& imageDst, const Mat& depthDst, const Mat& selectMaskDst, float maxDepthDiff,
                     Mat& _corresps, Mat& _diffs, double& _sigma, OdometryType method)
{
    Mat mrtInv = rt.inv(DECOMP_SVD);
    Matx44d rtInv = mrtInv;

    Mat corresps(depthDst.size(), CV_16SC2, Scalar::all(-1));
    Mat diffs;
    if (method == OdometryType::RGB)
        diffs = Mat(depthDst.size(), CV_32F, Scalar::all(-1));

    // src_2d = K * src_3d, src_3d = K_inv * [src_2d | z]
    //

    Matx33d K(_K);
    Matx33d K_inv = K.inv(DECOMP_SVD);
    Rect r(0, 0, depthDst.cols, depthDst.rows);
    Matx31d tinv = rtInv.get_minor<3, 1>(0, 3);
    Matx31d ktinvm = K * tinv;
    //const double* Kt_ptr = Kt.ptr<const double>();
    Point3d ktinv(ktinvm(0, 0), ktinvm(1, 0), ktinvm(2, 0));

    AutoBuffer<float> buf(3 * (depthDst.cols + depthDst.rows));
    float* KRK_inv0_u1 = buf.data();
    float* KRK_inv1_v1_plus_KRK_inv2 = buf.data() + depthDst.cols;
    float* KRK_inv3_u1               = buf.data() + depthDst.cols     + depthDst.rows;
    float* KRK_inv4_v1_plus_KRK_inv5 = buf.data() + depthDst.cols * 2 + depthDst.rows;
    float* KRK_inv6_u1               = buf.data() + depthDst.cols * 2 + depthDst.rows * 2;
    float* KRK_inv7_v1_plus_KRK_inv8 = buf.data() + depthDst.cols * 3 + depthDst.rows * 2;
    {
        Matx33d rinv = rtInv.get_minor<3, 3>(0, 0);

        Matx33d kriki = K * rinv * K_inv;
        for (int udst = 0; udst < depthDst.cols; udst++)
        {
            KRK_inv0_u1[udst] = (float)(kriki(0, 0) * udst);
            KRK_inv3_u1[udst] = (float)(kriki(1, 0) * udst);
            KRK_inv6_u1[udst] = (float)(kriki(2, 0) * udst);
        }

        for (int vdst = 0; vdst < depthDst.rows; vdst++)
        {
            KRK_inv1_v1_plus_KRK_inv2[vdst] = (float)(kriki(0, 1) * vdst + kriki(0, 2));
            KRK_inv4_v1_plus_KRK_inv5[vdst] = (float)(kriki(1, 1) * vdst + kriki(1, 2));
            KRK_inv7_v1_plus_KRK_inv8[vdst] = (float)(kriki(2, 1) * vdst + kriki(2, 2));
        }
    }

    double sigma = 0;
    int correspCount = 0;
    for (int vdst = 0; vdst < depthDst.rows; vdst++)
    {
        const float* depthDst_row = depthDst.ptr<float>(vdst);
        const uchar* maskDst_row = selectMaskDst.ptr<uchar>(vdst);
        for (int udst = 0; udst < depthDst.cols; udst++)
        {
            float ddst = depthDst_row[udst];

            if (maskDst_row[udst])
            {
                float transformed_ddst = static_cast<float>(ddst * (KRK_inv6_u1[udst] + KRK_inv7_v1_plus_KRK_inv8[vdst]) + ktinv.z);

                if (transformed_ddst > 0)
                {
                    float transformed_ddst_inv = 1.f / transformed_ddst;
                    int usrc = cvRound(transformed_ddst_inv * (ddst * (KRK_inv0_u1[udst] + KRK_inv1_v1_plus_KRK_inv2[vdst]) + ktinv.x));
                    int vsrc = cvRound(transformed_ddst_inv * (ddst * (KRK_inv3_u1[udst] + KRK_inv4_v1_plus_KRK_inv5[vdst]) + ktinv.y));
                    if (r.contains(Point(usrc, vsrc)))
                    {
                        float dsrc = depthSrc.at<float>(vsrc, usrc);
                        if (validMaskSrc.at<uchar>(vsrc, usrc) && std::abs(transformed_ddst - dsrc) <= maxDepthDiff)
                        {
                            CV_DbgAssert(!cvIsNaN(dsrc));
                            Vec2s& c = corresps.at<Vec2s>(vsrc, usrc);
                            float diff = 0;
                            if (c[0] != -1)
                            {
                                diff = 0;
                                int exist_u1 = c[0], exist_v1 = c[1];

                                float exist_d1 = (float)(depthDst.at<float>(exist_v1, exist_u1) *
                                                 (KRK_inv6_u1[exist_u1] + KRK_inv7_v1_plus_KRK_inv8[exist_v1]) + ktinv.z);

                                if (transformed_ddst > exist_d1)
                                    continue;
                                if (method == OdometryType::RGB)
                                    diff = static_cast<float>(static_cast<int>(imageSrc.at<uchar>(vsrc, usrc)) -
                                                              static_cast<int>(imageDst.at<uchar>(vdst, udst)));
                            }
                            else
                            {
                                if (method == OdometryType::RGB)
                                    diff = static_cast<float>(static_cast<int>(imageSrc.at<uchar>(vsrc, usrc)) -
                                                              static_cast<int>(imageDst.at<uchar>(vdst, udst)));
                                correspCount++;
                            }
                            c = Vec2s((short)udst, (short)vdst);
                            if (method == OdometryType::RGB)
                            {
                                diffs.at<float>(vsrc, usrc) = diff;
                                sigma += diff * diff;
                            }
                        }
                    }
                }
            }
        }
    }

    _sigma = std::sqrt(sigma / double(correspCount));

    _corresps.create(correspCount, 1, CV_32SC4);
    Vec4i* corresps_ptr = _corresps.ptr<Vec4i>();
    float* diffs_ptr = nullptr;
    if (method == OdometryType::RGB)
    {
        _diffs.create(correspCount, 1, CV_32F);
        diffs_ptr = _diffs.ptr<float>();
    }
    for (int vsrc = 0, i = 0; vsrc < corresps.rows; vsrc++)
    {
        const Vec2s* corresps_row = corresps.ptr<Vec2s>(vsrc);
        const float* diffs_row = nullptr;
        if (method == OdometryType::RGB)
            diffs_row = diffs.ptr<float>(vsrc);
        for (int usrc = 0; usrc < corresps.cols; usrc++)
        {
            const Vec2s& c = corresps_row[usrc];
            const float& d = diffs_row[usrc];
            if (c[0] != -1)
            {
                corresps_ptr[i] = Vec4i(usrc, vsrc, c[0], c[1]);
                if (method == OdometryType::RGB)
                    diffs_ptr[i] = d;
                i++;
            }
        }
    }
}

void calcRgbdLsmMatrices(const Mat& cloud0, const Mat& Rt,
                         const Mat& dI_dx1, const Mat& dI_dy1,
                         const Mat& corresps, const Mat& _diffs, const double _sigma,
                         double fx, double fy, double sobelScaleIn,
                         Mat& AtA, Mat& AtB, OdometryTransformType transformType)
{
    int transformDim = getTransformDim(transformType);
    AtA = Mat(transformDim, transformDim, CV_64FC1, Scalar(0));
    AtB = Mat(transformDim, 1, CV_64FC1, Scalar(0));
    double* AtB_ptr = AtB.ptr<double>();

    CV_Assert(Rt.type() == CV_64FC1);
    Affine3d rtmat(Rt);

    const float* diffs_ptr = _diffs.ptr<float>();
    const Vec4i* corresps_ptr = corresps.ptr<Vec4i>();
    double sigma = _sigma;

    std::vector<double> A_buf(transformDim);
    double* A_ptr = &A_buf[0];

    for (int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u0 = c[0], v0 = c[1];
        int u1 = c[2], v1 = c[3];

        double w = sigma + std::abs(diffs_ptr[correspIndex]);
        w = w > DBL_EPSILON ? 1. / w : 1.;

        double w_sobelScale = w * sobelScaleIn;

        const Vec4f& p0 = cloud0.at<Vec4f>(v0, u0);
        Point3d tp0 = rtmat * Point3d(p0[0], p0[1], p0[2]);

        rgbdCoeffsFunc(transformType,
                       A_ptr,
                       w_sobelScale * dI_dx1.at<short int>(v1, u1),
                       w_sobelScale * dI_dy1.at<short int>(v1, u1),
                       tp0, fx, fy);

        for (int y = 0; y < transformDim; y++)
        {
            double* AtA_ptr = AtA.ptr<double>(y);
            for (int x = y; x < transformDim; x++)
            {
                AtA_ptr[x] += A_ptr[y] * A_ptr[x];
            }
            AtB_ptr[y] += A_ptr[y] * w * diffs_ptr[correspIndex];
        }
    }

    for (int y = 0; y < transformDim; y++)
        for (int x = y + 1; x < transformDim; x++)
            AtA.at<double>(x, y) = AtA.at<double>(y, x);
}


void calcICPLsmMatrices(const Mat& cloud0, const Mat& Rt,
                        const Mat& cloud1, const Mat& normals1,
                        const Mat& corresps,
                        Mat& AtA, Mat& AtB, OdometryTransformType transformType)
{
    int transformDim = getTransformDim(transformType);
    AtA = Mat(transformDim, transformDim, CV_64FC1, Scalar(0));
    AtB = Mat(transformDim, 1, CV_64FC1, Scalar(0));
    double* AtB_ptr = AtB.ptr<double>();

    const int correspsCount = corresps.rows;

    CV_Assert(Rt.type() == CV_64FC1);
    const double* Rt_ptr = Rt.ptr<const double>();

    AutoBuffer<float> diffs(correspsCount);
    float* diffs_ptr = diffs.data();

    AutoBuffer<Point3f> transformedPoints0(correspsCount);
    Point3f* tps0_ptr = transformedPoints0.data();

    const Vec4i* corresps_ptr = corresps.ptr<Vec4i>();

    double sigma = 0;
    for (int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u0 = c[0], v0 = c[1];
        int u1 = c[2], v1 = c[3];

        const Vec4f& p0 = cloud0.at<Vec4f>(v0, u0);
        Point3f tp0;
        tp0.x = (float)(p0[0] * Rt_ptr[0] + p0[1] * Rt_ptr[1] + p0[2] * Rt_ptr[2] + Rt_ptr[3]);
        tp0.y = (float)(p0[0] * Rt_ptr[4] + p0[1] * Rt_ptr[5] + p0[2] * Rt_ptr[6] + Rt_ptr[7]);
        tp0.z = (float)(p0[0] * Rt_ptr[8] + p0[1] * Rt_ptr[9] + p0[2] * Rt_ptr[10] + Rt_ptr[11]);

        Vec4f n1 = normals1.at<Vec4f>(v1, u1);
        Vec4f _v = cloud1.at<Vec4f>(v1, u1);
        Point3f v = Point3f(_v[0], _v[1], _v[2]) - tp0;

        tps0_ptr[correspIndex] = tp0;
        diffs_ptr[correspIndex] = n1[0] * v.x + n1[1] * v.y + n1[2] * v.z;
        sigma += diffs_ptr[correspIndex] * diffs_ptr[correspIndex];
    }

    sigma = std::sqrt(sigma / correspsCount);

    std::vector<double> A_buf(transformDim);
    double* A_ptr = &A_buf[0];
    for (int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u1 = c[2], v1 = c[3];

        double w = sigma +std::abs(diffs_ptr[correspIndex]);
        w = w > DBL_EPSILON ? 1. / w : 1.;

        Vec4f n4 = normals1.at<Vec4f>(v1, u1);
        Vec4f p1 = cloud1.at<Vec4f>(v1, u1);

        icpCoeffsFunc(transformType,
                      A_ptr, tps0_ptr[correspIndex], Point3d(p1[0], p1[1], p1[2]), Vec3d(n4[0], n4[1], n4[2]) * w);
        for (int y = 0; y < transformDim; y++)
        {
            double* AtA_ptr = AtA.ptr<double>(y);
            for (int x = y; x < transformDim; x++)
            {
                AtA_ptr[x] += A_ptr[y] * A_ptr[x];
            }
            AtB_ptr[y] += A_ptr[y] * w * diffs_ptr[correspIndex];
        }
    }

    for (int y = 0; y < transformDim; y++)
        for (int x = y + 1; x < transformDim; x++)
            AtA.at<double>(x, y) = AtA.at<double>(y, x);
}

void computeProjectiveMatrix(const Mat& ksi, Mat& Rt)
{
    CV_Assert(ksi.size() == Size(1, 6) && ksi.type() == CV_64FC1);

    const double* ksi_ptr = ksi.ptr<const double>();
    // 0.5 multiplication is here because (dual) quaternions keep half an angle/twist inside
    Matx44d matdq = (DualQuatd(0, ksi_ptr[0], ksi_ptr[1], ksi_ptr[2],
                               0, ksi_ptr[3], ksi_ptr[4], ksi_ptr[5]) * 0.5).exp().toMat(QUAT_ASSUME_UNIT);
    Mat(matdq).copyTo(Rt);
}

bool solveSystem(const Mat& AtA, const Mat& AtB, double detThreshold, Mat& x)
{
    double det = determinant(AtA);
    if (fabs(det) < detThreshold || cvIsNaN(det) || cvIsInf(det))
        return false;

    solve(AtA, AtB, x, DECOMP_CHOLESKY);

    return true;
}

bool testDeltaTransformation(const Mat& deltaRt, double maxTranslation, double maxRotation)
{
    double translation = norm(deltaRt(Rect(3, 0, 1, 3)));

    Mat rvec;
    Rodrigues(deltaRt(Rect(0, 0, 3, 3)), rvec);

    double rotation = norm(rvec) * 180. / CV_PI;

    return translation <= maxTranslation && rotation <= maxRotation;
}

// Upper triangle buffer size (for symmetric matrix build)
const size_t UTSIZE = 27;

#if USE_INTRINSICS
static inline bool fastCheck(const v_float32x4& p0, const v_float32x4& p1)
{
    float check = v_reduce_sum(p0) + v_reduce_sum(p1);
    return !cvIsNaN(check);
}

static inline void getCrossPerm(const v_float32x4& a, v_float32x4& yzx, v_float32x4& zxy)
{
    v_uint32x4 aa = v_reinterpret_as_u32(a);
    v_uint32x4 yz00 = v_extract<1>(aa, v_setzero_u32());
    v_uint32x4 x0y0, tmp;
    v_zip(aa, v_setzero_u32(), x0y0, tmp);
    v_uint32x4 yzx0 = v_combine_low(yz00, x0y0);
    v_uint32x4 y000 = v_extract<2>(x0y0, v_setzero_u32());
    v_uint32x4 zx00 = v_extract<1>(yzx0, v_setzero_u32());
    zxy = v_reinterpret_as_f32(v_combine_low(zx00, y000));
    yzx = v_reinterpret_as_f32(yzx0);
}

static inline v_float32x4 crossProduct(const v_float32x4& a, const v_float32x4& b)
{
    v_float32x4 ayzx, azxy, byzx, bzxy;
    getCrossPerm(a, ayzx, azxy);
    getCrossPerm(b, byzx, bzxy);
    return v_sub(v_mul(ayzx, bzxy), v_mul(azxy, byzx));
}
#else
static inline bool fastCheck(const Point3f& p)
{
    return !cvIsNaN(p.x);
}

#endif

typedef Matx<float, 6, 7> ABtype;

struct GetAbInvoker : ParallelLoopBody
{
    GetAbInvoker(ABtype& _globalAb, Mutex& _mtx,
        const Points& _oldPts, const Normals& _oldNrm, const Points& _newPts, const Normals& _newNrm,
        Affine3f _pose, Intr::Projector _proj, float _sqDistanceThresh, float _minCos) :
        ParallelLoopBody(),
        globalSumAb(_globalAb), mtx(_mtx),
        oldPts(_oldPts), oldNrm(_oldNrm), newPts(_newPts), newNrm(_newNrm), pose(_pose),
        proj(_proj), sqDistanceThresh(_sqDistanceThresh), minCos(_minCos)
    { }

    virtual void operator ()(const Range& range) const override
    {
#if USE_INTRINSICS
        CV_Assert(ptype::channels == 4);

        const size_t utBufferSize = 9;
        float CV_DECL_ALIGNED(16) upperTriangle[utBufferSize * 4];
        for (size_t i = 0; i < utBufferSize * 4; i++)
            upperTriangle[i] = 0;
        // how values are kept in upperTriangle
        const int NA = 0;
        const size_t utPos[] =
        {
           0,  1,  2,  4,  5,  6,  3,
          NA,  9, 10, 12, 13, 14, 11,
          NA, NA, 18, 20, 21, 22, 19,
          NA, NA, NA, 24, 28, 30, 32,
          NA, NA, NA, NA, 25, 29, 33,
          NA, NA, NA, NA, NA, 26, 34
        };

        const float(&pm)[16] = pose.matrix.val;
        v_float32x4 poseRot0(pm[0], pm[4], pm[8], 0);
        v_float32x4 poseRot1(pm[1], pm[5], pm[9], 0);
        v_float32x4 poseRot2(pm[2], pm[6], pm[10], 0);
        v_float32x4 poseTrans(pm[3], pm[7], pm[11], 0);

        v_float32x4 vfxy(proj.fx, proj.fy, 0, 0), vcxy(proj.cx, proj.cy, 0, 0);
        v_float32x4 vframe((float)(oldPts.cols - 1), (float)(oldPts.rows - 1), 1.f, 1.f);

        float sqThresh = sqDistanceThresh;
        float cosThresh = minCos;

        for (int y = range.start; y < range.end; y++)
        {
            const CV_DECL_ALIGNED(16) float* newPtsRow = (const float*)newPts[y];
            const CV_DECL_ALIGNED(16) float* newNrmRow = (const float*)newNrm[y];

            for (int x = 0; x < newPts.cols; x++)
            {
                v_float32x4 newP = v_load_aligned(newPtsRow + x * 4);
                v_float32x4 newN = v_load_aligned(newNrmRow + x * 4);

                if (!fastCheck(newP, newN))
                    continue;

                //transform to old coord system
                newP = v_matmuladd(newP, poseRot0, poseRot1, poseRot2, poseTrans);
                newN = v_matmuladd(newN, poseRot0, poseRot1, poseRot2, v_setzero_f32());

                //find correspondence by projecting the point
                v_float32x4 oldCoords;
                float pz = v_get0(v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(newP))));
                // x, y, 0, 0
                oldCoords = v_muladd(v_div(newP, v_setall_f32(pz)), vfxy, vcxy);

                if (!v_check_all(v_and(v_ge(oldCoords, v_setzero_f32()), v_lt(oldCoords, vframe))))
                    continue;

                // bilinearly interpolate oldPts and oldNrm under oldCoords point
                v_float32x4 oldP;
                v_float32x4 oldN;
                {
                    v_int32x4 ixy = v_floor(oldCoords);
                    v_float32x4 txy = v_sub(oldCoords, v_cvt_f32(ixy));
                    int xi = v_get0(ixy);
                    int yi = v_get0(v_rotate_right<1>(ixy));
                    v_float32x4 tx = v_setall_f32(v_get0(txy));
                    txy = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(txy)));
                    v_float32x4 ty = v_setall_f32(v_get0(txy));

                    const float* prow0 = (const float*)oldPts[yi + 0];
                    const float* prow1 = (const float*)oldPts[yi + 1];

                    v_float32x4 p00 = v_load(prow0 + (xi + 0) * 4);
                    v_float32x4 p01 = v_load(prow0 + (xi + 1) * 4);
                    v_float32x4 p10 = v_load(prow1 + (xi + 0) * 4);
                    v_float32x4 p11 = v_load(prow1 + (xi + 1) * 4);

                    // do not fix missing data
                    // NaN check is done later

                    const float* nrow0 = (const float*)oldNrm[yi + 0];
                    const float* nrow1 = (const float*)oldNrm[yi + 1];

                    v_float32x4 n00 = v_load(nrow0 + (xi + 0) * 4);
                    v_float32x4 n01 = v_load(nrow0 + (xi + 1) * 4);
                    v_float32x4 n10 = v_load(nrow1 + (xi + 0) * 4);
                    v_float32x4 n11 = v_load(nrow1 + (xi + 1) * 4);

                    // NaN check is done later

                    v_float32x4 p0 = v_add(p00, v_mul(tx, v_sub(p01, p00)));
                    v_float32x4 p1 = v_add(p10, v_mul(tx, v_sub(p11, p10)));
                    oldP = v_add(p0, v_mul(ty, v_sub(p1, p0)));

                    v_float32x4 n0 = v_add(n00, v_mul(tx, v_sub(n01, n00)));
                    v_float32x4 n1 = v_add(n10, v_mul(tx, v_sub(n11, n10)));
                    oldN = v_add(n0, v_mul(ty, v_sub(n1, n0)));
                }

                bool oldPNcheck = fastCheck(oldP, oldN);

                //filter by distance
                v_float32x4 diff = v_sub(newP, oldP);
                bool distCheck = !(v_reduce_sum(v_mul(diff, diff)) > sqThresh);

                //filter by angle
                bool angleCheck = !(abs(v_reduce_sum(v_mul(newN, oldN))) < cosThresh);

                if (!(oldPNcheck && distCheck && angleCheck))
                    continue;

                // build point-wise vector ab = [ A | b ]
                v_float32x4 VxNv = crossProduct(newP, oldN);
                Point3f VxN;
                VxN.x = v_get0(VxNv);
                VxN.y = v_get0(v_reinterpret_as_f32(v_extract<1>(v_reinterpret_as_u32(VxNv), v_setzero_u32())));
                VxN.z = v_get0(v_reinterpret_as_f32(v_extract<2>(v_reinterpret_as_u32(VxNv), v_setzero_u32())));

                float dotp = -v_reduce_sum(v_mul(oldN, diff));

                // build point-wise upper-triangle matrix [ab^T * ab] w/o last row
                // which is [A^T*A | A^T*b]
                // and gather sum

                v_float32x4 vd = v_or(VxNv, v_float32x4(0, 0, 0, dotp));
                v_float32x4 n = oldN;
                v_float32x4 nyzx;
                {
                    v_uint32x4 aa = v_reinterpret_as_u32(n);
                    v_uint32x4 yz00 = v_extract<1>(aa, v_setzero_u32());
                    v_uint32x4 x0y0, tmp;
                    v_zip(aa, v_setzero_u32(), x0y0, tmp);
                    nyzx = v_reinterpret_as_f32(v_combine_low(yz00, x0y0));
                }

                v_float32x4 vutg[utBufferSize];
                for (size_t i = 0; i < utBufferSize; i++)
                    vutg[i] = v_load_aligned(upperTriangle + i * 4);

                int p = 0;
                v_float32x4 v;
                // vx * vd, vx * n
                v = v_setall_f32(VxN.x);
                v_store_aligned(upperTriangle + p * 4, v_muladd(v, vd, vutg[p])); p++;
                v_store_aligned(upperTriangle + p * 4, v_muladd(v, n, vutg[p])); p++;
                // vy * vd, vy * n
                v = v_setall_f32(VxN.y);
                v_store_aligned(upperTriangle + p * 4, v_muladd(v, vd, vutg[p])); p++;
                v_store_aligned(upperTriangle + p * 4, v_muladd(v, n, vutg[p])); p++;
                // vz * vd, vz * n
                v = v_setall_f32(VxN.z);
                v_store_aligned(upperTriangle + p * 4, v_muladd(v, vd, vutg[p])); p++;
                v_store_aligned(upperTriangle + p * 4, v_muladd(v, n, vutg[p])); p++;
                // nx^2, ny^2, nz^2
                v_store_aligned(upperTriangle + p * 4, v_muladd(n, n, vutg[p])); p++;
                // nx*ny, ny*nz, nx*nz
                v_store_aligned(upperTriangle + p * 4, v_muladd(n, nyzx, vutg[p])); p++;
                // nx*d, ny*d, nz*d
                v = v_setall_f32(dotp);
                v_store_aligned(upperTriangle + p * 4, v_muladd(n, v, vutg[p])); p++;
            }
        }

        ABtype sumAB = ABtype::zeros();
        for (int i = 0; i < 6; i++)
        {
            for (int j = i; j < 7; j++)
            {
                size_t p = utPos[i * 7 + j];
                sumAB(i, j) = upperTriangle[p];
            }
        }

        CV_UNUSED(UTSIZE);

#else
        float upperTriangle[UTSIZE];
        for (size_t i = 0; i < UTSIZE; i++)
            upperTriangle[i] = 0;

        for (int y = range.start; y < range.end; y++)
        {
            const ptype* newPtsRow = newPts[y];
            const ptype* newNrmRow = newNrm[y];

            for (int x = 0; x < newPts.cols; x++)
            {
                Point3f newP = fromPtype(newPtsRow[x]);
                Point3f newN = fromPtype(newNrmRow[x]);

                Point3f oldP(nan3), oldN(nan3);

                if (!(fastCheck(newP) && fastCheck(newN)))
                    continue;
                //transform to old coord system
                newP = pose * newP;
                newN = pose.rotation() * newN;

                //find correspondence by projecting the point
                Point2f oldCoords = proj(newP);
                if (!(oldCoords.x >= 0 && oldCoords.x < oldPts.cols - 1 &&
                    oldCoords.y >= 0 && oldCoords.y < oldPts.rows - 1))
                    continue;

                // bilinearly interpolate oldPts and oldNrm under oldCoords point
                int xi = cvFloor(oldCoords.x), yi = cvFloor(oldCoords.y);
                float tx = oldCoords.x - xi, ty = oldCoords.y - yi;

                const ptype* prow0 = oldPts[yi + 0];
                const ptype* prow1 = oldPts[yi + 1];

                Point3f p00 = fromPtype(prow0[xi + 0]);
                Point3f p01 = fromPtype(prow0[xi + 1]);
                Point3f p10 = fromPtype(prow1[xi + 0]);
                Point3f p11 = fromPtype(prow1[xi + 1]);

                //do not fix missing data
                if (!(fastCheck(p00) && fastCheck(p01) &&
                    fastCheck(p10) && fastCheck(p11)))
                    continue;

                const ptype* nrow0 = oldNrm[yi + 0];
                const ptype* nrow1 = oldNrm[yi + 1];

                Point3f n00 = fromPtype(nrow0[xi + 0]);
                Point3f n01 = fromPtype(nrow0[xi + 1]);
                Point3f n10 = fromPtype(nrow1[xi + 0]);
                Point3f n11 = fromPtype(nrow1[xi + 1]);

                if (!(fastCheck(n00) && fastCheck(n01) &&
                    fastCheck(n10) && fastCheck(n11)))
                    continue;

                Point3f p0 = p00 + tx * (p01 - p00);
                Point3f p1 = p10 + tx * (p11 - p10);
                oldP = p0 + ty * (p1 - p0);

                Point3f n0 = n00 + tx * (n01 - n00);
                Point3f n1 = n10 + tx * (n11 - n10);
                oldN = n0 + ty * (n1 - n0);

                if (!(fastCheck(oldP) && fastCheck(oldN)))
                    continue;

                //filter by distance
                Point3f diff = newP - oldP;
                if (diff.dot(diff) > sqDistanceThresh)
                {
                    continue;
                }

                //filter by angle
                if (abs(newN.dot(oldN)) < minCos)
                {
                    continue;
                }
                // build point-wise vector ab = [ A | b ]

                //try to optimize
                Point3f VxN = newP.cross(oldN);
                float ab[7] = { VxN.x, VxN.y, VxN.z, oldN.x, oldN.y, oldN.z, oldN.dot(-diff) };
                // build point-wise upper-triangle matrix [ab^T * ab] w/o last row
                // which is [A^T*A | A^T*b]
                // and gather sum
                int pos = 0;
                for (int i = 0; i < 6; i++)
                {
                    for (int j = i; j < 7; j++)
                    {
                        upperTriangle[pos++] += ab[i] * ab[j];
                    }
                }
            }
        }

        ABtype sumAB = ABtype::zeros();
        int pos = 0;
        for (int i = 0; i < 6; i++)
        {
            for (int j = i; j < 7; j++)
            {
                sumAB(i, j) = upperTriangle[pos++];
            }
        }
#endif
        AutoLock al(mtx);
        globalSumAb += sumAB;
    }

    ABtype& globalSumAb;
    Mutex& mtx;
    const Points& oldPts;
    const Normals& oldNrm;
    const Points& newPts;
    const Normals& newNrm;
    Affine3f pose;
    const Intr::Projector proj;
    float sqDistanceThresh;
    float minCos;
};

void calcICPLsmMatricesFast(Matx33f cameraMatrix, const UMat& oldPts, const UMat& oldNrm, const UMat& newPts, const UMat& newNrm,
                            cv::Affine3f pose, int level, float maxDepthDiff, float angleThreshold, cv::Matx66f& A, cv::Vec6f& b)
{
    CV_Assert(oldPts.size() == oldNrm.size());
    CV_Assert(newPts.size() == newNrm.size());

    CV_OCL_RUN(ocl::isOpenCLActivated(),
        ocl_calcICPLsmMatricesFast(cameraMatrix,
            oldPts, oldNrm, newPts, newNrm,
            pose, level, maxDepthDiff, angleThreshold,
            A, b)
        );

    ABtype sumAB = ABtype::zeros();
    Mutex mutex;
    const Points  op(oldPts.getMat(AccessFlag::ACCESS_READ)), np(newPts.getMat(AccessFlag::ACCESS_READ));
    const Normals on(oldNrm.getMat(AccessFlag::ACCESS_READ)), nn(newNrm.getMat(AccessFlag::ACCESS_READ));

    Intr intrinsics(cameraMatrix);
    GetAbInvoker invoker(sumAB, mutex, op, on, np, nn, pose,
        intrinsics.scale(level).makeProjector(),
        maxDepthDiff * maxDepthDiff, std::cos(angleThreshold));
    Range range(0, newPts.rows);
    const int nstripes = -1;
    parallel_for_(range, invoker, nstripes);

    // splitting AB matrix to A and b
    for (int i = 0; i < 6; i++)
    {
        // augment lower triangle of A by symmetry
        for (int j = i; j < 6; j++)
        {
            A(i, j) = A(j, i) = sumAB(i, j);
        }

        b(i) = sumAB(i, 6);
    }
}

#ifdef HAVE_OPENCL

bool ocl_calcICPLsmMatricesFast(Matx33f cameraMatrix, const UMat& oldPts, const UMat& oldNrm, const UMat& newPts, const UMat& newNrm,
                                cv::Affine3f pose, int level, float maxDepthDiff, float angleThreshold, cv::Matx66f& A, cv::Vec6f& b)
{
    CV_TRACE_FUNCTION();

    Size oldSize = oldPts.size(), newSize = newPts.size();
    CV_Assert(oldSize == oldNrm.size());
    CV_Assert(newSize == newNrm.size());

    // calculate 1x7 vector ab to produce b and upper triangle of A:
    // [A|b] = ab*(ab^t)
    // and then reduce it across work groups

    UMat groupedSumBuffer;
    cv::String errorStr;
    String name = "getAb";
    ocl::ProgramSource source = ocl::_3d::icp_oclsrc;
    cv::String options = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if (k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    size_t globalSize[2];
    globalSize[0] = (size_t)newPts.cols;
    globalSize[1] = (size_t)newPts.rows;

    const ocl::Device& device = ocl::Device::getDefault();
    size_t wgsLimit = device.maxWorkGroupSize();
    size_t memSize = device.localMemSize();
    // local memory should keep upperTriangles for all threads in group for reduce
    const size_t ltsz = UTSIZE * sizeof(float);
    const size_t lcols = 32;
    size_t lrows = min(memSize / ltsz, wgsLimit) / lcols;
    // round lrows down to 2^n
    lrows = roundDownPow2(lrows);
    size_t localSize[2] = { lcols, lrows };
    Size ngroups((int)divUp(globalSize[0], (unsigned int)localSize[0]),
                 (int)divUp(globalSize[1], (unsigned int)localSize[1]));

    // size of local buffer for group-wide reduce
    size_t lsz = localSize[0] * localSize[1] * ltsz;

    Intr intrinsics(cameraMatrix);
    Intr::Projector proj = intrinsics.scale(level).makeProjector();
    Vec2f fxy(proj.fx, proj.fy), cxy(proj.cx, proj.cy);

    UMat& groupedSumGpu = groupedSumBuffer;
    groupedSumGpu.create(Size(ngroups.width * UTSIZE, ngroups.height),
        CV_32F);
    groupedSumGpu.setTo(0);

    // TODO: optimization possible:
    // samplers instead of oldPts/oldNrm (mask needed)
    k.args(ocl::KernelArg::ReadOnlyNoSize(oldPts),
        ocl::KernelArg::ReadOnlyNoSize(oldNrm),
        oldSize,
        ocl::KernelArg::ReadOnlyNoSize(newPts),
        ocl::KernelArg::ReadOnlyNoSize(newNrm),
        newSize,
        ocl::KernelArg::Constant(pose.matrix.val,
            sizeof(pose.matrix.val)),
        fxy.val, cxy.val,
        maxDepthDiff * maxDepthDiff,
        std::cos(angleThreshold),
        ocl::KernelArg::Local(lsz),
        ocl::KernelArg::WriteOnlyNoSize(groupedSumGpu)
    );

    if (!k.run(2, globalSize, localSize, true))
        throw std::runtime_error("Failed to run kernel");

    float upperTriangle[UTSIZE];
    for (size_t i = 0; i < UTSIZE; i++)
        upperTriangle[i] = 0;

    Mat groupedSumCpu = groupedSumGpu.getMat(ACCESS_READ);

    for (int y = 0; y < ngroups.height; y++)
    {
        const float* rowr = groupedSumCpu.ptr<float>(y);
        for (size_t x = 0; x < size_t(ngroups.width); x++)
        {
            const float* p = rowr + x * UTSIZE;
            for (size_t j = 0; j < UTSIZE; j++)
            {
                upperTriangle[j] += p[j];
            }
        }
    }
    groupedSumCpu.release();

    ABtype sumAB = ABtype::zeros();
    int pos = 0;
    for (int i = 0; i < 6; i++)
    {
        for (int j = i; j < 7; j++)
        {
            sumAB(i, j) = upperTriangle[pos++];
        }
    }

    // splitting AB matrix to A and b
    for (int i = 0; i < 6; i++)
    {
        // augment lower triangle of A by symmetry
        for (int j = i; j < 6; j++)
        {
            A(i, j) = A(j, i) = sumAB(i, j);
        }

        b(i) = sumAB(i, 6);
    }
    return true;
}

#endif

}
