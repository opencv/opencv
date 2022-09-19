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

static const int normalWinSize = 5;
static const RgbdNormals::RgbdNormalsMethod normalMethod = RgbdNormals::RGBD_NORMALS_METHOD_FALS;

enum
{
    UTSIZE = 27
};

void prepareRGBDFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, OdometrySettings settings, OdometryAlgoType algtype)
{
    prepareRGBFrame(srcFrame, dstFrame, settings, true);
    prepareICPFrame(srcFrame, dstFrame, settings, algtype);
}

void prepareRGBFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, OdometrySettings settings, bool useDepth)
{
    prepareRGBFrameBase(srcFrame, settings, useDepth);
    prepareRGBFrameBase(dstFrame, settings, useDepth);

    prepareRGBFrameSrc(srcFrame, settings);
    prepareRGBFrameDst(dstFrame, settings);
}

void prepareICPFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, OdometrySettings settings, OdometryAlgoType algtype)
{
    prepareICPFrameBase(srcFrame, settings);
    prepareICPFrameBase(dstFrame, settings);

    prepareICPFrameSrc(srcFrame, settings);
    if (algtype == OdometryAlgoType::FAST)
        prepareICPFrameDst(srcFrame, settings);
    prepareICPFrameDst(dstFrame, settings);
}

void prepareRGBFrameBase(OdometryFrame& frame, OdometrySettings settings, bool useDepth)
{
    // Can be transformed into template argument in the future
    // when this algorithm supports OCL UMats too

    typedef Mat TMat;

    TMat image;
    frame.getGrayImage(image);
    if (image.empty())
    {
        if (frame.getPyramidLevels(OdometryFramePyramidType::PYR_IMAGE) > 0)
        {
            TMat pyr0;
            frame.getPyramidAt(pyr0, OdometryFramePyramidType::PYR_IMAGE, 0);
            frame.setImage(pyr0);
            frame.getGrayImage(image);
        }
        else
            CV_Error(Error::StsBadSize, "Image or pyramidImage have to be set.");
    }
    checkImage(image);

    TMat depth;
    if (useDepth)
    {
        frame.getScaledDepth(depth);
        if (depth.empty())
        {
            if (frame.getPyramidLevels(OdometryFramePyramidType::PYR_DEPTH) > 0)
            {
                TMat pyr0;
                frame.getPyramidAt(pyr0, OdometryFramePyramidType::PYR_DEPTH, 0);
                frame.setDepth(pyr0);
            }
            else if (frame.getPyramidLevels(OdometryFramePyramidType::PYR_CLOUD) > 0)
            {
                TMat cloud;
                frame.getPyramidAt(cloud, OdometryFramePyramidType::PYR_CLOUD, 0);
                std::vector<TMat> xyz;
                split(cloud, xyz);
                frame.setDepth(xyz[2]);
                frame.getScaledDepth(depth);
            }
            else
                CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
        }
        checkDepth(depth, image.size());
    }
    else
        depth = TMat(image.size(), CV_32F, 1);

    TMat mask;
    frame.getMask(mask);
    if (mask.empty() && frame.getPyramidLevels(OdometryFramePyramidType::PYR_MASK) > 0)
    {
        TMat pyr0;
        frame.getPyramidAt(pyr0, OdometryFramePyramidType::PYR_MASK, 0);
        frame.setMask(pyr0);
        frame.getMask(mask);
    }
    checkMask(mask, image.size());

    std::vector<int> iterCounts;
    Mat miterCounts;
    settings.getIterCounts(miterCounts);
    for (int i = 0; i < miterCounts.size().height; i++)
        iterCounts.push_back(miterCounts.at<int>(i));

    std::vector<TMat> ipyramids;
    preparePyramidImage(image, ipyramids, iterCounts.size());
    setPyramids(frame, OdometryFramePyramidType::PYR_IMAGE, ipyramids);

    std::vector<TMat> dpyramids;
    preparePyramidImage(depth, dpyramids, iterCounts.size());
    setPyramids(frame, OdometryFramePyramidType::PYR_DEPTH, dpyramids);

    std::vector<TMat> mpyramids;
    std::vector<TMat> npyramids;
    preparePyramidMask<TMat>(mask, dpyramids, settings.getMinDepth(), settings.getMaxDepth(), npyramids, mpyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_MASK, mpyramids);
}

void prepareRGBFrameSrc(OdometryFrame& frame, OdometrySettings settings)
{
    typedef Mat TMat;

    std::vector<TMat> dpyramids(frame.getPyramidLevels(OdometryFramePyramidType::PYR_DEPTH));
    getPyramids(frame, OdometryFramePyramidType::PYR_DEPTH, dpyramids);
    std::vector<TMat> mpyramids(frame.getPyramidLevels(OdometryFramePyramidType::PYR_MASK));
    getPyramids(frame, OdometryFramePyramidType::PYR_MASK, mpyramids);

    std::vector<TMat> cpyramids;
    Matx33f cameraMatrix;
    settings.getCameraMatrix(cameraMatrix);

    preparePyramidCloud<TMat>(dpyramids, cameraMatrix, cpyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_CLOUD, cpyramids);
}

void prepareRGBFrameDst(OdometryFrame& frame, OdometrySettings settings)
{
    typedef Mat TMat;

    std::vector<TMat> ipyramids(frame.getPyramidLevels(OdometryFramePyramidType::PYR_IMAGE));
    getPyramids(frame, OdometryFramePyramidType::PYR_IMAGE, ipyramids);
    std::vector<TMat> mpyramids(frame.getPyramidLevels(OdometryFramePyramidType::PYR_MASK));
    getPyramids(frame, OdometryFramePyramidType::PYR_MASK, mpyramids);

    std::vector<TMat> dxpyramids, dypyramids, tmpyramids;

    Mat _minGradientMagnitudes;
    std::vector<float> minGradientMagnitudes;
    settings.getMinGradientMagnitudes(_minGradientMagnitudes);
    for (int i = 0; i < _minGradientMagnitudes.size().height; i++)
        minGradientMagnitudes.push_back(_minGradientMagnitudes.at<float>(i));

    preparePyramidSobel<TMat>(ipyramids, 1, 0, dxpyramids, settings.getSobelSize());
    preparePyramidSobel<TMat>(ipyramids, 0, 1, dypyramids, settings.getSobelSize());
    preparePyramidTexturedMask(dxpyramids, dypyramids, minGradientMagnitudes,
        mpyramids, settings.getMaxPointsPart(), tmpyramids, settings.getSobelScale());

    setPyramids(frame, OdometryFramePyramidType::PYR_DIX, dxpyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_DIY, dypyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_TEXMASK, tmpyramids);
}

void prepareICPFrameBase(OdometryFrame& frame, OdometrySettings settings)
{
    typedef Mat TMat;

    TMat depth;
    frame.getScaledDepth(depth);
    if (depth.empty())
    {
        if (frame.getPyramidLevels(OdometryFramePyramidType::PYR_DEPTH) > 0)
        {
            TMat pyr0;
            frame.getPyramidAt(pyr0, OdometryFramePyramidType::PYR_DEPTH, 0);
            frame.setDepth(pyr0);
            frame.getScaledDepth(depth);
        }
        else if (frame.getPyramidLevels(OdometryFramePyramidType::PYR_CLOUD) > 0)
        {
            TMat cloud;
            frame.getPyramidAt(cloud, OdometryFramePyramidType::PYR_CLOUD, 0);
            std::vector<TMat> xyz;
            split(cloud, xyz);
            frame.setDepth(xyz[2]);
            frame.getScaledDepth(depth);
        }
        else
            CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
    }

    checkDepth(depth, depth.size());

    TMat mask;
    frame.getMask(mask);
    if (mask.empty() && frame.getPyramidLevels(OdometryFramePyramidType::PYR_MASK) > 0)
    {
        TMat pyr0;
        frame.getPyramidAt(pyr0, OdometryFramePyramidType::PYR_MASK, 0);
        frame.setMask(pyr0);
        frame.getMask(mask);
    }
    checkMask(mask, depth.size());

    std::vector<int> iterCounts;
    Mat miterCounts;
    settings.getIterCounts(miterCounts);
    for (int i = 0; i < miterCounts.size().height; i++)
        iterCounts.push_back(miterCounts.at<int>(i));

    std::vector<TMat> dpyramids;
    preparePyramidImage(depth, dpyramids, iterCounts.size());
    setPyramids(frame, OdometryFramePyramidType::PYR_DEPTH, dpyramids);

    std::vector<TMat> mpyramids(frame.getPyramidLevels(OdometryFramePyramidType::PYR_MASK));
    getPyramids(frame, OdometryFramePyramidType::PYR_MASK, mpyramids);

    std::vector<TMat> cpyramids;
    Matx33f cameraMatrix;
    settings.getCameraMatrix(cameraMatrix);

    preparePyramidCloud<TMat>(dpyramids, cameraMatrix, cpyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_CLOUD, cpyramids);
}

void prepareICPFrameSrc(OdometryFrame& frame, OdometrySettings settings)
{
    typedef Mat TMat;

    TMat mask;
    frame.getMask(mask);

    std::vector<TMat> dpyramids(frame.getPyramidLevels(OdometryFramePyramidType::PYR_DEPTH));
    getPyramids(frame, OdometryFramePyramidType::PYR_DEPTH, dpyramids);

    std::vector<TMat> mpyramids;
    std::vector<TMat> npyramids;
    preparePyramidMask<TMat>(mask, dpyramids, settings.getMinDepth(), settings.getMaxDepth(),
        npyramids, mpyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_MASK, mpyramids);
}

void prepareICPFrameDst(OdometryFrame& frame, OdometrySettings settings)
{
    typedef Mat TMat;

    Ptr<RgbdNormals> normalsComputer;
    Matx33f cameraMatrix;
    settings.getCameraMatrix(cameraMatrix);

    TMat depth, mask, normals;
    frame.getScaledDepth(depth);
    frame.getMask(mask);
    frame.getNormals(normals);

    if (normals.empty())
    {
        if ( frame.getPyramidLevels(OdometryFramePyramidType::PYR_NORM))
        {
            TMat n0;
            frame.getPyramidAt(n0, OdometryFramePyramidType::PYR_NORM, 0);
            frame.setNormals(n0);
            frame.getNormals(normals);
        }
        else
        {
            Matx33f K;
            if (!normalsComputer.empty())
                normalsComputer->getK(K);
            if (normalsComputer.empty() ||
                normalsComputer->getRows() != depth.rows ||
                normalsComputer->getCols() != depth.cols ||
                norm(K, cameraMatrix) > FLT_EPSILON)
                normalsComputer = RgbdNormals::create(depth.rows,
                    depth.cols,
                    depth.depth(),
                    cameraMatrix,
                    normalWinSize,
                    50.f,
                    normalMethod);
            TMat c0;
            frame.getPyramidAt(c0, OdometryFramePyramidType::PYR_CLOUD, 0);
            normalsComputer->apply(c0, normals);
            frame.setNormals(normals);
            frame.getNormals(normals);
        }
    }

    std::vector<TMat> npyramids;
    std::vector<TMat> dpyramids(frame.getPyramidLevels(OdometryFramePyramidType::PYR_DEPTH));
    getPyramids(frame, OdometryFramePyramidType::PYR_DEPTH, dpyramids);
    preparePyramidNormals(normals, dpyramids, npyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_NORM, npyramids);

    std::vector<TMat> mpyramids;
    preparePyramidMask<TMat>(mask, dpyramids, settings.getMinDepth(), settings.getMaxDepth(),
        npyramids, mpyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_MASK, mpyramids);

    std::vector<TMat> nmpyramids;
    preparePyramidNormalsMask(npyramids, mpyramids, settings.getMaxPointsPart(), nmpyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_NORMMASK, nmpyramids);
}

void setPyramids(OdometryFrame& odf, OdometryFramePyramidType oftype, InputArrayOfArrays pyramidImage)
{
    size_t nLevels = pyramidImage.size(-1).width;
    std::vector<Mat> pyramids;
    pyramidImage.getMatVector(pyramids);
    odf.setPyramidLevel(nLevels, oftype);
    for (size_t l = 0; l < nLevels; l++)
    {
        odf.setPyramidAt(pyramids[l], oftype, l);
    }
}

void getPyramids(OdometryFrame& odf, OdometryFramePyramidType oftype, OutputArrayOfArrays _pyramid)
{
    typedef Mat TMat;

    size_t nLevels = odf.getPyramidLevels(oftype);
    for (size_t l = 0; l < nLevels; l++)
    {
        TMat img;
        odf.getPyramidAt(img, oftype, l);
        TMat& p = _pyramid.getMatRef(int(l));
        img.copyTo(p);
    }
}

void preparePyramidImage(InputArray image, InputOutputArrayOfArrays pyramidImage, size_t levelCount)
{
    if (!pyramidImage.empty())
    {
        size_t nLevels = pyramidImage.size(-1).width;
        if (nLevels < levelCount)
            CV_Error(Error::StsBadSize, "Levels count of pyramidImage has to be equal or less than size of iterCounts.");

        CV_Assert(pyramidImage.size(0) == image.size());
        for (size_t i = 0; i < nLevels; i++)
            CV_Assert(pyramidImage.type((int)i) == image.type());
    }
    else
        buildPyramid(image, pyramidImage, (int)levelCount - 1);
}

template<typename TMat>
void preparePyramidMask(InputArray mask, InputArrayOfArrays pyramidDepth, float minDepth, float maxDepth,
                        InputArrayOfArrays pyramidNormal,
                        InputOutputArrayOfArrays pyramidMask)
{
    minDepth = std::max(0.f, minDepth);

    int nLevels = pyramidDepth.size(-1).width;
    if (!pyramidMask.empty())
    {
        if (pyramidMask.size(-1).width != nLevels)
            CV_Error(Error::StsBadSize, "Levels count of pyramidMask has to be equal to size of pyramidDepth.");

        for (int i = 0; i < pyramidMask.size(-1).width; i++)
        {
            CV_Assert(pyramidMask.size(i) == pyramidDepth.size(i));
            CV_Assert(pyramidMask.type(i) == CV_8UC1);
        }
    }
    else
    {
        TMat validMask;
        if (mask.empty())
            validMask = TMat(pyramidDepth.size(0), CV_8UC1, Scalar(255));
        else
            validMask = getTMat<TMat>(mask, -1).clone();

        buildPyramid(validMask, pyramidMask, nLevels - 1);

        for (int i = 0; i < pyramidMask.size(-1).width; i++)
        {
            TMat levelDepth = getTMat<TMat>(pyramidDepth, i).clone();
            patchNaNs(levelDepth, 0);

            TMat& levelMask = getTMatRef<TMat>(pyramidMask, i);
            TMat gtmin, ltmax, tmpMask;
            cv::compare(levelDepth, Scalar(minDepth), gtmin, CMP_GT);
            cv::compare(levelDepth, Scalar(maxDepth), ltmax, CMP_LT);
            cv::bitwise_and(gtmin, ltmax, tmpMask);
            cv::bitwise_and(levelMask, tmpMask, levelMask);

            if (!pyramidNormal.empty())
            {
                CV_Assert(pyramidNormal.type(i) == CV_32FC4);
                CV_Assert(pyramidNormal.size(i) == pyramidDepth.size(i));
                TMat levelNormal = getTMat<TMat>(pyramidNormal, i).clone();

                TMat validNormalMask;
                // NaN check
                cv::compare(levelNormal, levelNormal, validNormalMask, CMP_EQ);
                CV_Assert(validNormalMask.type() == CV_8UC4);

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
void preparePyramidCloud(InputArrayOfArrays pyramidDepth, const Matx33f& cameraMatrix, InputOutputArrayOfArrays pyramidCloud)
{
    size_t depthSize = pyramidDepth.size(-1).width;
    size_t cloudSize = pyramidCloud.size(-1).width;
    if (!pyramidCloud.empty())
    {
        if (cloudSize != depthSize)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidCloud.");

        for (size_t i = 0; i < depthSize; i++)
        {
            CV_Assert(pyramidCloud.size((int)i) == pyramidDepth.size((int)i));
            CV_Assert(pyramidCloud.type((int)i) == CV_32FC4);
        }
    }
    else
    {
        std::vector<Matx33f> pyramidCameraMatrix;
        buildPyramidCameraMatrix(cameraMatrix, (int)depthSize, pyramidCameraMatrix);

        pyramidCloud.create((int)depthSize, 1, CV_32FC4, -1);
        for (size_t i = 0; i < depthSize; i++)
        {
            TMat cloud;
            depthTo3d(getTMat<TMat>(pyramidDepth, (int)i), pyramidCameraMatrix[i], cloud, Mat());
            getTMatRef<TMat>(pyramidCloud, (int)i) = cloud;

        }
    }
}

void buildPyramidCameraMatrix(const Matx33f& cameraMatrix, int levels, std::vector<Matx33f>& pyramidCameraMatrix)
{
    pyramidCameraMatrix.resize(levels);

    for (int i = 0; i < levels; i++)
    {
        Matx33f levelCameraMatrix = (i == 0) ? cameraMatrix : 0.5f * pyramidCameraMatrix[i - 1];
        levelCameraMatrix(2, 2) = 1.0;
        pyramidCameraMatrix[i] = levelCameraMatrix;
    }
}


template<typename TMat>
void preparePyramidSobel(InputArrayOfArrays pyramidImage, int dx, int dy, InputOutputArrayOfArrays pyramidSobel, int sobelSize)
{
    size_t imgLevels = pyramidImage.size(-1).width;
    size_t sobelLvls = pyramidSobel.size(-1).width;
    if (!pyramidSobel.empty())
    {
        if (sobelLvls != imgLevels)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidSobel.");

        for (size_t i = 0; i < sobelLvls; i++)
        {
            CV_Assert(pyramidSobel.size((int)i) == pyramidImage.size((int)i));
            CV_Assert(pyramidSobel.type((int)i) == CV_16SC1);
        }
    }
    else
    {
        pyramidSobel.create((int)imgLevels, 1, CV_16SC1, -1);
        for (size_t i = 0; i < imgLevels; i++)
        {
            Sobel(getTMat<TMat>(pyramidImage, (int)i), getTMatRef<TMat>(pyramidSobel, (int)i), CV_16S, dx, dy, sobelSize);
        }
    }
}

void preparePyramidTexturedMask(InputArrayOfArrays pyramid_dI_dx, InputArrayOfArrays pyramid_dI_dy,
                                InputArray minGradMagnitudes, InputArrayOfArrays pyramidMask, double maxPointsPart,
                                InputOutputArrayOfArrays pyramidTexturedMask, double sobelScale)
{
    size_t didxLevels = pyramid_dI_dx.size(-1).width;
    size_t texLevels = pyramidTexturedMask.size(-1).width;
    if (!pyramidTexturedMask.empty())
    {
        if (texLevels != didxLevels)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidTexturedMask.");

        for (size_t i = 0; i < texLevels; i++)
        {
            CV_Assert(pyramidTexturedMask.size((int)i) == pyramid_dI_dx.size((int)i));
            CV_Assert(pyramidTexturedMask.type((int)i) == CV_8UC1);
        }
    }
    else
    {
        CV_Assert(minGradMagnitudes.type() == CV_32F);
        Mat_<float> mgMags = minGradMagnitudes.getMat();

        const float sobelScale2_inv = (float) (1. / (sobelScale * sobelScale));
        pyramidTexturedMask.create((int)didxLevels, 1, CV_8UC1, -1);
        for (size_t i = 0; i < didxLevels; i++)
        {
            const float minScaledGradMagnitude2 = mgMags((int)i) * mgMags((int)i) * sobelScale2_inv;
            const Mat& dIdx = pyramid_dI_dx.getMat((int)i);
            const Mat& dIdy = pyramid_dI_dy.getMat((int)i);

            Mat texturedMask(dIdx.size(), CV_8UC1, Scalar(0));

            for (int y = 0; y < dIdx.rows; y++)
            {
                const short* dIdx_row = dIdx.ptr<short>(y);
                const short* dIdy_row = dIdy.ptr<short>(y);
                uchar* texturedMask_row = texturedMask.ptr<uchar>(y);
                for (int x = 0; x < dIdx.cols; x++)
                {
                    float magnitude2 = static_cast<float>(dIdx_row[x] * dIdx_row[x] + dIdy_row[x] * dIdy_row[x]);
                    if (magnitude2 >= minScaledGradMagnitude2)
                        texturedMask_row[x] = 255;
                }
            }
            Mat texMask = texturedMask & pyramidMask.getMat((int)i);

            randomSubsetOfMask(texMask, (float)maxPointsPart);
            pyramidTexturedMask.getMatRef((int)i) = texMask;
        }
    }
}

void randomSubsetOfMask(InputOutputArray _mask, float part)
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

void preparePyramidNormals(InputArray normals, InputArrayOfArrays pyramidDepth, InputOutputArrayOfArrays pyramidNormals)
{
    size_t depthLevels = pyramidDepth.size(-1).width;
    size_t normalsLevels = pyramidNormals.size(-1).width;
    if (!pyramidNormals.empty())
    {
        if (normalsLevels != depthLevels)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidNormals.");

        for (size_t i = 0; i < normalsLevels; i++)
        {
            CV_Assert(pyramidNormals.size((int)i) == pyramidDepth.size((int)i));
            CV_Assert(pyramidNormals.type((int)i) == CV_32FC3);
        }
    }
    else
    {
        buildPyramid(normals, pyramidNormals, (int)depthLevels - 1);
        // renormalize normals
        for (size_t i = 1; i < depthLevels; i++)
        {
            Mat& currNormals = pyramidNormals.getMatRef((int)i);
            for (int y = 0; y < currNormals.rows; y++)
            {
                Point3f* normals_row = currNormals.ptr<Point3f>(y);
                for (int x = 0; x < currNormals.cols; x++)
                {
                    double nrm = norm(normals_row[x]);
                    normals_row[x] *= 1. / nrm;
                }
            }
        }
    }
}

void preparePyramidNormalsMask(InputArray pyramidNormals, InputArray pyramidMask, double maxPointsPart,
                               InputOutputArrayOfArrays /*std::vector<Mat>&*/ pyramidNormalsMask)
{
    size_t maskLevels = pyramidMask.size(-1).width;
    size_t norMaskLevels = pyramidNormalsMask.size(-1).width;
    if (!pyramidNormalsMask.empty())
    {
        if (norMaskLevels != maskLevels)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidNormalsMask.");

        for (size_t i = 0; i < norMaskLevels; i++)
        {
            CV_Assert(pyramidNormalsMask.size((int)i) == pyramidMask.size((int)i));
            CV_Assert(pyramidNormalsMask.type((int)i) == pyramidMask.type((int)i));
        }
    }
    else
    {
        pyramidNormalsMask.create((int)maskLevels, 1, CV_8U, -1);
        for (size_t i = 0; i < maskLevels; i++)
        {
            Mat& normalsMask = pyramidNormalsMask.getMatRef((int)i);
            normalsMask = pyramidMask.getMat((int)i).clone();

            const Mat normals = pyramidNormals.getMat((int)i);
            for (int y = 0; y < normalsMask.rows; y++)
            {
                const Vec4f* normals_row = normals.ptr<Vec4f>(y);
                uchar* normalsMask_row = normalsMask.ptr<uchar>(y);
                for (int x = 0; x < normalsMask.cols; x++)
                {
                    Vec4f n = normals_row[x];
                    if (cvIsNaN(n[0]))
                    {
                        normalsMask_row[x] = 0;
                    }
                }
            }
            randomSubsetOfMask(normalsMask, (float)maxPointsPart);
        }
    }
}

bool RGBDICPOdometryImpl(OutputArray _Rt, const Mat& initRt,
                         const OdometryFrame srcFrame,
                         const OdometryFrame dstFrame,
                         const Matx33f& cameraMatrix,
                         float maxDepthDiff, float angleThreshold, const std::vector<int>& iterCounts,
                         double maxTranslation, double maxRotation, double sobelScale,
                         OdometryType method, OdometryTransformType transformType, OdometryAlgoType algtype)
{
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

        if (method != OdometryType::DEPTH)
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

            if(method != OdometryType::DEPTH) // RGB
            {
                const Mat pyramidTexturedMask;
                dstFrame.getPyramidAt(pyramidTexturedMask, OdometryFramePyramidType::PYR_TEXMASK, level);
                computeCorresps(levelCameraMatrix, resultRt,
                                srcLevelImage, srcLevelDepth, pyramidMask,
                                dstLevelImage, dstLevelDepth, pyramidTexturedMask, maxDepthDiff,
                                corresps_rgbd, diffs_rgbd, sigma_rgbd, OdometryType::RGB);
            }

            if(method != OdometryType::RGB) // ICP
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

            const Mat srcPyrCloud;
            srcFrame.getPyramidAt(srcPyrCloud, OdometryFramePyramidType::PYR_CLOUD, level);


            Mat AtA(transformDim, transformDim, CV_64FC1, Scalar(0)), AtB(transformDim, 1, CV_64FC1, Scalar(0));
            if(corresps_rgbd.rows >= minCorrespsCount)
            {
                const Mat srcPyrImage, dstPyrImage, dstPyrIdx, dstPyrIdy;
                srcFrame.getPyramidAt(srcPyrImage, OdometryFramePyramidType::PYR_IMAGE, level);
                dstFrame.getPyramidAt(dstPyrImage, OdometryFramePyramidType::PYR_IMAGE, level);
                dstFrame.getPyramidAt(dstPyrIdx, OdometryFramePyramidType::PYR_DIX, level);
                dstFrame.getPyramidAt(dstPyrIdy, OdometryFramePyramidType::PYR_DIY, level);
                calcRgbdLsmMatrices(srcPyrCloud, resultRt, dstPyrIdx, dstPyrIdy,
                                    corresps_rgbd, diffs_rgbd, sigma_rgbd, fx, fy, sobelScale,
                                    AtA_rgbd, AtB_rgbd, transformType);
                AtA += AtA_rgbd;
                AtB += AtB_rgbd;
            }
            if(corresps_icp.rows >= minCorrespsCount || algtype == OdometryAlgoType::FAST)
            {
                const Mat dstPyrCloud, dstPyrNormals, srcPyrNormals;
                dstFrame.getPyramidAt(dstPyrCloud, OdometryFramePyramidType::PYR_CLOUD, level);
                dstFrame.getPyramidAt(dstPyrNormals, OdometryFramePyramidType::PYR_NORM, level);

                if (algtype == OdometryAlgoType::COMMON)
                {
                    calcICPLsmMatrices(srcPyrCloud, resultRt, dstPyrCloud, dstPyrNormals,
                                       corresps_icp, AtA_icp, AtB_icp, transformType);
                }
                else
                {
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

            if (maskDst_row[udst] && !cvIsNaN(ddst))
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
        Point3f tp0 = rtmat * Point3f(p0[0], p0[1], p0[2]);

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
                      A_ptr, tps0_ptr[correspIndex], Point3f(p1[0], p1[1], p1[2]), Vec3f(n4[0], n4[1], n4[2]) * w);
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
    return ayzx * bzxy - azxy * byzx;
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
                float pz = (v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(newP))).get0());
                // x, y, 0, 0
                oldCoords = v_muladd(newP / v_setall_f32(pz), vfxy, vcxy);

                if (!v_check_all((oldCoords >= v_setzero_f32()) & (oldCoords < vframe)))
                    continue;

                // bilinearly interpolate oldPts and oldNrm under oldCoords point
                v_float32x4 oldP;
                v_float32x4 oldN;
                {
                    v_int32x4 ixy = v_floor(oldCoords);
                    v_float32x4 txy = oldCoords - v_cvt_f32(ixy);
                    int xi = ixy.get0();
                    int yi = v_rotate_right<1>(ixy).get0();
                    v_float32x4 tx = v_setall_f32(txy.get0());
                    txy = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(txy)));
                    v_float32x4 ty = v_setall_f32(txy.get0());

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

                    v_float32x4 p0 = p00 + tx * (p01 - p00);
                    v_float32x4 p1 = p10 + tx * (p11 - p10);
                    oldP = p0 + ty * (p1 - p0);

                    v_float32x4 n0 = n00 + tx * (n01 - n00);
                    v_float32x4 n1 = n10 + tx * (n11 - n10);
                    oldN = n0 + ty * (n1 - n0);
                }

                bool oldPNcheck = fastCheck(oldP, oldN);

                //filter by distance
                v_float32x4 diff = newP - oldP;
                bool distCheck = !(v_reduce_sum(diff * diff) > sqThresh);

                //filter by angle
                bool angleCheck = !(abs(v_reduce_sum(newN * oldN)) < cosThresh);

                if (!(oldPNcheck && distCheck && angleCheck))
                    continue;

                // build point-wise vector ab = [ A | b ]
                v_float32x4 VxNv = crossProduct(newP, oldN);
                Point3f VxN;
                VxN.x = VxNv.get0();
                VxN.y = v_reinterpret_as_f32(v_extract<1>(v_reinterpret_as_u32(VxNv), v_setzero_u32())).get0();
                VxN.z = v_reinterpret_as_f32(v_extract<2>(v_reinterpret_as_u32(VxNv), v_setzero_u32())).get0();

                float dotp = -v_reduce_sum(oldN * diff);

                // build point-wise upper-triangle matrix [ab^T * ab] w/o last row
                // which is [A^T*A | A^T*b]
                // and gather sum

                v_float32x4 vd = VxNv | v_float32x4(0, 0, 0, dotp);
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

#else
        float upperTriangle[UTSIZE];
        for (int i = 0; i < UTSIZE; i++)
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

void calcICPLsmMatricesFast(Matx33f cameraMatrix, const Mat& oldPts, const Mat& oldNrm, const Mat& newPts, const Mat& newNrm,
    cv::Affine3f pose, int level, float maxDepthDiff, float angleThreshold, cv::Matx66f& A, cv::Vec6f& b)
{
    CV_Assert(oldPts.size() == oldNrm.size());
    CV_Assert(newPts.size() == newNrm.size());

    CV_OCL_RUN(ocl::isOpenCLActivated(),
        ocl_calcICPLsmMatricesFast(cameraMatrix,
            oldPts.getUMat(AccessFlag::ACCESS_READ), oldNrm.getUMat(AccessFlag::ACCESS_READ),
            newPts.getUMat(AccessFlag::ACCESS_READ), newNrm.getUMat(AccessFlag::ACCESS_READ),
            pose, level, maxDepthDiff, angleThreshold,
            A, b)
        );

    ABtype sumAB = ABtype::zeros();
    Mutex mutex;
    const Points  op(oldPts), np(newPts);
    const Normals on(oldNrm),  nn(newNrm);


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
    for (int i = 0; i < UTSIZE; i++)
        upperTriangle[i] = 0;

    Mat groupedSumCpu = groupedSumGpu.getMat(ACCESS_READ);

    for (int y = 0; y < ngroups.height; y++)
    {
        const float* rowr = groupedSumCpu.ptr<float>(y);
        for (int x = 0; x < ngroups.width; x++)
        {
            const float* p = rowr + x * UTSIZE;
            for (int j = 0; j < UTSIZE; j++)
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
