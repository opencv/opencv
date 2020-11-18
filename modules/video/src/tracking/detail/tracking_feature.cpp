// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "opencv2/video/detail/tracking.private.hpp"
#include "opencv2/video/detail/tracking_feature.private.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

/*
 * TODO This implementation is based on apps/traincascade/
 * TODO Changed CvHaarEvaluator based on ADABOOSTING implementation (Grabner et al.)
 */

CvParams::CvParams()
{
    // nothing
}

//---------------------------- FeatureParams --------------------------------------

CvFeatureParams::CvFeatureParams()
    : maxCatCount(0)
    , featSize(1)
    , numFeatures(1)
{
    // nothing
}

//------------------------------------- FeatureEvaluator ---------------------------------------

void CvFeatureEvaluator::init(const CvFeatureParams* _featureParams, int _maxSampleCount, Size _winSize)
{
    CV_Assert(_featureParams);
    CV_Assert(_maxSampleCount > 0);
    featureParams = (CvFeatureParams*)_featureParams;
    winSize = _winSize;
    numFeatures = _featureParams->numFeatures;
    cls.create((int)_maxSampleCount, 1, CV_32FC1);
    generateFeatures();
}

void CvFeatureEvaluator::setImage(const Mat& img, uchar clsLabel, int idx)
{
    winSize.width = img.cols;
    winSize.height = img.rows;
    //CV_Assert( img.cols == winSize.width );
    //CV_Assert( img.rows == winSize.height );
    CV_Assert(idx < cls.rows);
    cls.ptr<float>(idx)[0] = clsLabel;
}

CvHaarFeatureParams::CvHaarFeatureParams()
{
    isIntegral = false;
}

//--------------------- HaarFeatureEvaluator ----------------

void CvHaarEvaluator::init(const CvFeatureParams* _featureParams, int /*_maxSampleCount*/, Size _winSize)
{
    CV_Assert(_featureParams);
    int cols = (_winSize.width + 1) * (_winSize.height + 1);
    sum.create((int)1, cols, CV_32SC1);
    isIntegral = ((CvHaarFeatureParams*)_featureParams)->isIntegral;
    CvFeatureEvaluator::init(_featureParams, 1, _winSize);
}

void CvHaarEvaluator::setImage(const Mat& img, uchar /*clsLabel*/, int /*idx*/)
{
    CV_DbgAssert(!sum.empty());

    winSize.width = img.cols;
    winSize.height = img.rows;

    CvFeatureEvaluator::setImage(img, 1, 0);
    if (!isIntegral)
    {
        std::vector<Mat_<float>> ii_imgs;
        compute_integral(img, ii_imgs);
        _ii_img = ii_imgs[0];
    }
    else
    {
        _ii_img = img;
    }
}

void CvHaarEvaluator::generateFeatures()
{
    generateFeatures(featureParams->numFeatures);
}

void CvHaarEvaluator::generateFeatures(int nFeatures)
{
    for (int i = 0; i < nFeatures; i++)
    {
        CvHaarEvaluator::FeatureHaar feature(Size(winSize.width, winSize.height));
        features.push_back(feature);
    }
}

#define INITSIGMA(numAreas) (static_cast<float>(sqrt(256.0f * 256.0f / 12.0f * (numAreas))));

CvHaarEvaluator::FeatureHaar::FeatureHaar(Size patchSize)
{
    try
    {
        generateRandomFeature(patchSize);
    }
    catch (...)
    {
        // FIXIT
        throw;
    }
}

void CvHaarEvaluator::FeatureHaar::generateRandomFeature(Size patchSize)
{
    cv::Point2i position;
    Size baseDim;
    Size sizeFactor;
    int area;

    CV_Assert(!patchSize.empty());

    //Size minSize = Size( 3, 3 );
    int minArea = 9;

    bool valid = false;
    while (!valid)
    {
        //choose position and scale
        position.y = rand() % (patchSize.height);
        position.x = rand() % (patchSize.width);

        baseDim.width = (int)((1 - sqrt(1 - (float)rand() * (float)(1.0 / RAND_MAX))) * patchSize.width);
        baseDim.height = (int)((1 - sqrt(1 - (float)rand() * (float)(1.0 / RAND_MAX))) * patchSize.height);

        //select types
        //float probType[11] = {0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0950f};
        float probType[11] = { 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        float prob = (float)rand() * (float)(1.0 / RAND_MAX);

        if (prob < probType[0])
        {
            //check if feature is valid
            sizeFactor.height = 2;
            sizeFactor.width = 1;
            if (position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width)
                continue;
            area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
            if (area < minArea)
                continue;

            m_type = 1;
            m_numAreas = 2;
            m_weights.resize(m_numAreas);
            m_weights[0] = 1;
            m_weights[1] = -1;
            m_areas.resize(m_numAreas);
            m_areas[0].x = position.x;
            m_areas[0].y = position.y;
            m_areas[0].height = baseDim.height;
            m_areas[0].width = baseDim.width;
            m_areas[1].x = position.x;
            m_areas[1].y = position.y + baseDim.height;
            m_areas[1].height = baseDim.height;
            m_areas[1].width = baseDim.width;
            m_initMean = 0;
            m_initSigma = INITSIGMA(m_numAreas);

            valid = true;
        }
        else if (prob < probType[0] + probType[1])
        {
            //check if feature is valid
            sizeFactor.height = 1;
            sizeFactor.width = 2;
            if (position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width)
                continue;
            area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
            if (area < minArea)
                continue;

            m_type = 2;
            m_numAreas = 2;
            m_weights.resize(m_numAreas);
            m_weights[0] = 1;
            m_weights[1] = -1;
            m_areas.resize(m_numAreas);
            m_areas[0].x = position.x;
            m_areas[0].y = position.y;
            m_areas[0].height = baseDim.height;
            m_areas[0].width = baseDim.width;
            m_areas[1].x = position.x + baseDim.width;
            m_areas[1].y = position.y;
            m_areas[1].height = baseDim.height;
            m_areas[1].width = baseDim.width;
            m_initMean = 0;
            m_initSigma = INITSIGMA(m_numAreas);
            valid = true;
        }
        else if (prob < probType[0] + probType[1] + probType[2])
        {
            //check if feature is valid
            sizeFactor.height = 4;
            sizeFactor.width = 1;
            if (position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width)
                continue;
            area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
            if (area < minArea)
                continue;

            m_type = 3;
            m_numAreas = 3;
            m_weights.resize(m_numAreas);
            m_weights[0] = 1;
            m_weights[1] = -2;
            m_weights[2] = 1;
            m_areas.resize(m_numAreas);
            m_areas[0].x = position.x;
            m_areas[0].y = position.y;
            m_areas[0].height = baseDim.height;
            m_areas[0].width = baseDim.width;
            m_areas[1].x = position.x;
            m_areas[1].y = position.y + baseDim.height;
            m_areas[1].height = 2 * baseDim.height;
            m_areas[1].width = baseDim.width;
            m_areas[2].y = position.y + 3 * baseDim.height;
            m_areas[2].x = position.x;
            m_areas[2].height = baseDim.height;
            m_areas[2].width = baseDim.width;
            m_initMean = 0;
            m_initSigma = INITSIGMA(m_numAreas);
            valid = true;
        }
        else if (prob < probType[0] + probType[1] + probType[2] + probType[3])
        {
            //check if feature is valid
            sizeFactor.height = 1;
            sizeFactor.width = 4;
            if (position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width)
                continue;
            area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
            if (area < minArea)
                continue;

            m_type = 3;
            m_numAreas = 3;
            m_weights.resize(m_numAreas);
            m_weights[0] = 1;
            m_weights[1] = -2;
            m_weights[2] = 1;
            m_areas.resize(m_numAreas);
            m_areas[0].x = position.x;
            m_areas[0].y = position.y;
            m_areas[0].height = baseDim.height;
            m_areas[0].width = baseDim.width;
            m_areas[1].x = position.x + baseDim.width;
            m_areas[1].y = position.y;
            m_areas[1].height = baseDim.height;
            m_areas[1].width = 2 * baseDim.width;
            m_areas[2].y = position.y;
            m_areas[2].x = position.x + 3 * baseDim.width;
            m_areas[2].height = baseDim.height;
            m_areas[2].width = baseDim.width;
            m_initMean = 0;
            m_initSigma = INITSIGMA(m_numAreas);
            valid = true;
        }
        else if (prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4])
        {
            //check if feature is valid
            sizeFactor.height = 2;
            sizeFactor.width = 2;
            if (position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width)
                continue;
            area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
            if (area < minArea)
                continue;

            m_type = 5;
            m_numAreas = 4;
            m_weights.resize(m_numAreas);
            m_weights[0] = 1;
            m_weights[1] = -1;
            m_weights[2] = -1;
            m_weights[3] = 1;
            m_areas.resize(m_numAreas);
            m_areas[0].x = position.x;
            m_areas[0].y = position.y;
            m_areas[0].height = baseDim.height;
            m_areas[0].width = baseDim.width;
            m_areas[1].x = position.x + baseDim.width;
            m_areas[1].y = position.y;
            m_areas[1].height = baseDim.height;
            m_areas[1].width = baseDim.width;
            m_areas[2].y = position.y + baseDim.height;
            m_areas[2].x = position.x;
            m_areas[2].height = baseDim.height;
            m_areas[2].width = baseDim.width;
            m_areas[3].y = position.y + baseDim.height;
            m_areas[3].x = position.x + baseDim.width;
            m_areas[3].height = baseDim.height;
            m_areas[3].width = baseDim.width;
            m_initMean = 0;
            m_initSigma = INITSIGMA(m_numAreas);
            valid = true;
        }
        else if (prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5])
        {
            //check if feature is valid
            sizeFactor.height = 3;
            sizeFactor.width = 3;
            if (position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width)
                continue;
            area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
            if (area < minArea)
                continue;

            m_type = 6;
            m_numAreas = 2;
            m_weights.resize(m_numAreas);
            m_weights[0] = 1;
            m_weights[1] = -9;
            m_areas.resize(m_numAreas);
            m_areas[0].x = position.x;
            m_areas[0].y = position.y;
            m_areas[0].height = 3 * baseDim.height;
            m_areas[0].width = 3 * baseDim.width;
            m_areas[1].x = position.x + baseDim.width;
            m_areas[1].y = position.y + baseDim.height;
            m_areas[1].height = baseDim.height;
            m_areas[1].width = baseDim.width;
            m_initMean = -8 * 128;
            m_initSigma = INITSIGMA(m_numAreas);
            valid = true;
        }
        else if (prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6])
        {
            //check if feature is valid
            sizeFactor.height = 3;
            sizeFactor.width = 1;
            if (position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width)
                continue;
            area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
            if (area < minArea)
                continue;

            m_type = 7;
            m_numAreas = 3;
            m_weights.resize(m_numAreas);
            m_weights[0] = 1;
            m_weights[1] = -2;
            m_weights[2] = 1;
            m_areas.resize(m_numAreas);
            m_areas[0].x = position.x;
            m_areas[0].y = position.y;
            m_areas[0].height = baseDim.height;
            m_areas[0].width = baseDim.width;
            m_areas[1].x = position.x;
            m_areas[1].y = position.y + baseDim.height;
            m_areas[1].height = baseDim.height;
            m_areas[1].width = baseDim.width;
            m_areas[2].y = position.y + baseDim.height * 2;
            m_areas[2].x = position.x;
            m_areas[2].height = baseDim.height;
            m_areas[2].width = baseDim.width;
            m_initMean = 0;
            m_initSigma = INITSIGMA(m_numAreas);
            valid = true;
        }
        else if (prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6] + probType[7])
        {
            //check if feature is valid
            sizeFactor.height = 1;
            sizeFactor.width = 3;
            if (position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width)
                continue;

            area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;

            if (area < minArea)
                continue;

            m_type = 8;
            m_numAreas = 3;
            m_weights.resize(m_numAreas);
            m_weights[0] = 1;
            m_weights[1] = -2;
            m_weights[2] = 1;
            m_areas.resize(m_numAreas);
            m_areas[0].x = position.x;
            m_areas[0].y = position.y;
            m_areas[0].height = baseDim.height;
            m_areas[0].width = baseDim.width;
            m_areas[1].x = position.x + baseDim.width;
            m_areas[1].y = position.y;
            m_areas[1].height = baseDim.height;
            m_areas[1].width = baseDim.width;
            m_areas[2].y = position.y;
            m_areas[2].x = position.x + 2 * baseDim.width;
            m_areas[2].height = baseDim.height;
            m_areas[2].width = baseDim.width;
            m_initMean = 0;
            m_initSigma = INITSIGMA(m_numAreas);
            valid = true;
        }
        else if (prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6] + probType[7] + probType[8])
        {
            //check if feature is valid
            sizeFactor.height = 3;
            sizeFactor.width = 3;
            if (position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width)
                continue;
            area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
            if (area < minArea)
                continue;

            m_type = 9;
            m_numAreas = 2;
            m_weights.resize(m_numAreas);
            m_weights[0] = 1;
            m_weights[1] = -2;
            m_areas.resize(m_numAreas);
            m_areas[0].x = position.x;
            m_areas[0].y = position.y;
            m_areas[0].height = 3 * baseDim.height;
            m_areas[0].width = 3 * baseDim.width;
            m_areas[1].x = position.x + baseDim.width;
            m_areas[1].y = position.y + baseDim.height;
            m_areas[1].height = baseDim.height;
            m_areas[1].width = baseDim.width;
            m_initMean = 0;
            m_initSigma = INITSIGMA(m_numAreas);
            valid = true;
        }
        else if (prob
                < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6] + probType[7] + probType[8] + probType[9])
        {
            //check if feature is valid
            sizeFactor.height = 3;
            sizeFactor.width = 1;
            if (position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width)
                continue;
            area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
            if (area < minArea)
                continue;

            m_type = 10;
            m_numAreas = 3;
            m_weights.resize(m_numAreas);
            m_weights[0] = 1;
            m_weights[1] = -1;
            m_weights[2] = 1;
            m_areas.resize(m_numAreas);
            m_areas[0].x = position.x;
            m_areas[0].y = position.y;
            m_areas[0].height = baseDim.height;
            m_areas[0].width = baseDim.width;
            m_areas[1].x = position.x;
            m_areas[1].y = position.y + baseDim.height;
            m_areas[1].height = baseDim.height;
            m_areas[1].width = baseDim.width;
            m_areas[2].y = position.y + baseDim.height * 2;
            m_areas[2].x = position.x;
            m_areas[2].height = baseDim.height;
            m_areas[2].width = baseDim.width;
            m_initMean = 128;
            m_initSigma = INITSIGMA(m_numAreas);
            valid = true;
        }
        else if (prob
                < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6] + probType[7] + probType[8] + probType[9]
                        + probType[10])
        {
            //check if feature is valid
            sizeFactor.height = 1;
            sizeFactor.width = 3;
            if (position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width)
                continue;
            area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
            if (area < minArea)
                continue;

            m_type = 11;
            m_numAreas = 3;
            m_weights.resize(m_numAreas);
            m_weights[0] = 1;
            m_weights[1] = -1;
            m_weights[2] = 1;
            m_areas.resize(m_numAreas);
            m_areas[0].x = position.x;
            m_areas[0].y = position.y;
            m_areas[0].height = baseDim.height;
            m_areas[0].width = baseDim.width;
            m_areas[1].x = position.x + baseDim.width;
            m_areas[1].y = position.y;
            m_areas[1].height = baseDim.height;
            m_areas[1].width = baseDim.width;
            m_areas[2].y = position.y;
            m_areas[2].x = position.x + 2 * baseDim.width;
            m_areas[2].height = baseDim.height;
            m_areas[2].width = baseDim.width;
            m_initMean = 128;
            m_initSigma = INITSIGMA(m_numAreas);
            valid = true;
        }
        else
            CV_Error(Error::StsAssert, "");
    }

    m_initSize = patchSize;
    m_curSize = m_initSize;
    m_scaleFactorWidth = m_scaleFactorHeight = 1.0f;
    m_scaleAreas.resize(m_numAreas);
    m_scaleWeights.resize(m_numAreas);
    for (int curArea = 0; curArea < m_numAreas; curArea++)
    {
        m_scaleAreas[curArea] = m_areas[curArea];
        m_scaleWeights[curArea] = (float)m_weights[curArea] / (float)(m_areas[curArea].width * m_areas[curArea].height);
    }
}

bool CvHaarEvaluator::FeatureHaar::eval(const Mat& image, Rect /*ROI*/, float* result) const
{

    *result = 0.0f;

    for (int curArea = 0; curArea < m_numAreas; curArea++)
    {
        *result += (float)getSum(image, Rect(m_areas[curArea].x, m_areas[curArea].y, m_areas[curArea].width, m_areas[curArea].height))
                * m_scaleWeights[curArea];
    }

    /*
   if( image->getUseVariance() )
   {
   float variance = (float) image->getVariance( ROI );
   *result /= variance;
   }
   */

    return true;
}

float CvHaarEvaluator::FeatureHaar::getSum(const Mat& image, Rect imageROI) const
{
    // left upper Origin
    int OriginX = imageROI.x;
    int OriginY = imageROI.y;

    // Check and fix width and height
    int Width = imageROI.width;
    int Height = imageROI.height;

    if (OriginX + Width >= image.cols - 1)
        Width = (image.cols - 1) - OriginX;
    if (OriginY + Height >= image.rows - 1)
        Height = (image.rows - 1) - OriginY;

    float value = 0;
    int depth = image.depth();

    if (depth == CV_8U || depth == CV_32S)
        value = static_cast<float>(image.at<int>(OriginY + Height, OriginX + Width) + image.at<int>(OriginY, OriginX) - image.at<int>(OriginY, OriginX + Width)
                - image.at<int>(OriginY + Height, OriginX));
    else if (depth == CV_64F)
        value = static_cast<float>(image.at<double>(OriginY + Height, OriginX + Width) + image.at<double>(OriginY, OriginX)
                - image.at<double>(OriginY, OriginX + Width) - image.at<double>(OriginY + Height, OriginX));
    else if (depth == CV_32F)
        value = static_cast<float>(image.at<float>(OriginY + Height, OriginX + Width) + image.at<float>(OriginY, OriginX) - image.at<float>(OriginY, OriginX + Width)
                - image.at<float>(OriginY + Height, OriginX));

    return value;
}

}}}  // namespace cv::detail::tracking
