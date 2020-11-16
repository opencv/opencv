// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "opencv2/video/detail/tracking.private.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

/*
 * TODO This implementation is based on apps/traincascade/
 * TODO Changed CvHaarEvaluator based on ADABOOSTING implementation (Grabner et al.)
 */

CvParams::CvParams()
    : name("params")
{
}
void CvParams::printDefaults() const
{
    //std::cout << "--" << name << "--" << std::endl;
}
void CvParams::printAttrs() const
{
}
bool CvParams::scanAttr(const std::string, const std::string)
{
    return false;
}

//---------------------------- FeatureParams --------------------------------------

CvFeatureParams::CvFeatureParams()
    : maxCatCount(0)
    , featSize(1)
    , numFeatures(1)
{
    name = CC_FEATURE_PARAMS;
}

void CvFeatureParams::init(const CvFeatureParams& fp)
{
    maxCatCount = fp.maxCatCount;
    featSize = fp.featSize;
    numFeatures = fp.numFeatures;
}

void CvFeatureParams::write(FileStorage& fs) const
{
    fs << CC_MAX_CAT_COUNT << maxCatCount;
    fs << CC_FEATURE_SIZE << featSize;
    fs << CC_NUM_FEATURES << numFeatures;
}

bool CvFeatureParams::read(const FileNode& node)
{
    if (node.empty())
        return false;
    maxCatCount = node[CC_MAX_CAT_COUNT];
    featSize = node[CC_FEATURE_SIZE];
    numFeatures = node[CC_NUM_FEATURES];
    return (maxCatCount >= 0 && featSize >= 1);
}

Ptr<CvFeatureParams> CvFeatureParams::create(FeatureType featureType)
{
    return featureType == HAAR ? Ptr<CvFeatureParams>(new CvHaarFeatureParams) : featureType == LBP ? Ptr<CvFeatureParams>(new CvLBPFeatureParams)
            : featureType == HOG                                                                    ? Ptr<CvFeatureParams>(new CvHOGFeatureParams)
                                                                                                    : Ptr<CvFeatureParams>();
}

//------------------------------------- FeatureEvaluator ---------------------------------------

void CvFeatureEvaluator::init(const CvFeatureParams* _featureParams, int _maxSampleCount, Size _winSize)
{
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

Ptr<CvFeatureEvaluator> CvFeatureEvaluator::create(CvFeatureParams::FeatureType type)
{
    return type == CvFeatureParams::HAAR ? Ptr<CvFeatureEvaluator>(new CvHaarEvaluator) : type == CvFeatureParams::LBP ? Ptr<CvFeatureEvaluator>(new CvLBPEvaluator)
            : type == CvFeatureParams::HOG                                                                             ? Ptr<CvFeatureEvaluator>(new CvHOGEvaluator)
                                                                                                                       : Ptr<CvFeatureEvaluator>();
}

CvHaarFeatureParams::CvHaarFeatureParams()
{
    name = HFP_NAME;
    isIntegral = false;
}

void CvHaarFeatureParams::init(const CvFeatureParams& fp)
{
    CvFeatureParams::init(fp);
    isIntegral = ((const CvHaarFeatureParams&)fp).isIntegral;
}

void CvHaarFeatureParams::write(FileStorage& fs) const
{
    CvFeatureParams::write(fs);
    fs << CC_ISINTEGRAL << isIntegral;
}

bool CvHaarFeatureParams::read(const FileNode& node)
{
    if (!CvFeatureParams::read(node))
        return false;

    FileNode rnode = node[CC_ISINTEGRAL];
    if (!rnode.isString())
        return false;
    String intStr;
    rnode >> intStr;
    isIntegral = !intStr.compare("0") ? false : !true;
    return true;
}

void CvHaarFeatureParams::printDefaults() const
{
    CvFeatureParams::printDefaults();
    //std::cout << "isIntegral: false" << std::endl;
}

void CvHaarFeatureParams::printAttrs() const
{
    CvFeatureParams::printAttrs();
    //std::string int_str = isIntegral == true ? "true" : "false";
    //std::cout << "isIntegral: " << int_str << std::endl;
}

bool CvHaarFeatureParams::scanAttr(const std::string /*prmName*/, const std::string /*val*/)
{

    return true;
}

//--------------------- HaarFeatureEvaluator ----------------

void CvHaarEvaluator::init(const CvFeatureParams* _featureParams, int /*_maxSampleCount*/, Size _winSize)
{
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

void CvHaarEvaluator::writeFeatures(FileStorage& fs, const Mat& featureMap) const
{
    _writeFeatures(features, fs, featureMap);
}

void CvHaarEvaluator::writeFeature(FileStorage& fs) const
{
    String modeStr = isIntegral == true ? "1" : "0";
    CV_Assert(!modeStr.empty());
    fs << "isIntegral" << modeStr;
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

const std::vector<CvHaarEvaluator::FeatureHaar>& CvHaarEvaluator::getFeatures() const
{
    return features;
}

float CvHaarEvaluator::operator()(int featureIdx, int /*sampleIdx*/)
{
    /* TODO Added from MIL implementation */
    //return features[featureIdx].calc( _ii_img, Mat(), 0 );
    float res;
    features.at(featureIdx).eval(_ii_img, Rect(0, 0, winSize.width, winSize.height), &res);
    return res;
}

void CvHaarEvaluator::setWinSize(Size patchSize)
{
    winSize.width = patchSize.width;
    winSize.height = patchSize.height;
}

Size CvHaarEvaluator::setWinSize() const
{
    return Size(winSize.width, winSize.height);
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
        throw;
    }
}

float CvHaarEvaluator::FeatureHaar::getInitMean() const
{
    return m_initMean;
}

float CvHaarEvaluator::FeatureHaar::getInitSigma() const
{
    return m_initSigma;
}

void CvHaarEvaluator::FeatureHaar::generateRandomFeature(Size patchSize)
{
    cv::Point2i position;
    Size baseDim;
    Size sizeFactor;
    int area;

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

int CvHaarEvaluator::FeatureHaar::getNumAreas()
{
    return m_numAreas;
}

const std::vector<float>& CvHaarEvaluator::FeatureHaar::getWeights() const
{
    return m_weights;
}

const std::vector<Rect>& CvHaarEvaluator::FeatureHaar::getAreas() const
{
    return m_areas;
}

CvHOGFeatureParams::CvHOGFeatureParams()
{
    maxCatCount = 0;
    name = HOGF_NAME;
    featSize = N_BINS * N_CELLS;
}

void CvHOGEvaluator::init(const CvFeatureParams* _featureParams, int _maxSampleCount, Size _winSize)
{
    CV_Assert(_maxSampleCount > 0);
    int cols = (_winSize.width + 1) * (_winSize.height + 1);
    for (int bin = 0; bin < N_BINS; bin++)
    {
        hist.push_back(Mat(_maxSampleCount, cols, CV_32FC1));
    }
    normSum.create((int)_maxSampleCount, cols, CV_32FC1);
    CvFeatureEvaluator::init(_featureParams, _maxSampleCount, _winSize);
}

void CvHOGEvaluator::setImage(const Mat& img, uchar clsLabel, int idx)
{
    CV_DbgAssert(!hist.empty());
    CvFeatureEvaluator::setImage(img, clsLabel, idx);
    std::vector<Mat> integralHist;
    for (int bin = 0; bin < N_BINS; bin++)
    {
        integralHist.push_back(Mat(winSize.height + 1, winSize.width + 1, hist[bin].type(), hist[bin].ptr<float>((int)idx)));
    }
    Mat integralNorm(winSize.height + 1, winSize.width + 1, normSum.type(), normSum.ptr<float>((int)idx));
    integralHistogram(img, integralHist, integralNorm, (int)N_BINS);
}

//void CvHOGEvaluator::writeFeatures( FileStorage &fs, const Mat& featureMap ) const
//{
//    _writeFeatures( features, fs, featureMap );
//}

void CvHOGEvaluator::writeFeatures(FileStorage& fs, const Mat& featureMap) const
{
    int featIdx;
    int componentIdx;
    const Mat_<int>& featureMap_ = (const Mat_<int>&)featureMap;
    fs << FEATURES << "[";
    for (int fi = 0; fi < featureMap.cols; fi++)
        if (featureMap_(0, fi) >= 0)
        {
            fs << "{";
            featIdx = fi / getFeatureSize();
            componentIdx = fi % getFeatureSize();
            features[featIdx].write(fs, componentIdx);
            fs << "}";
        }
    fs << "]";
}

void CvHOGEvaluator::generateFeatures()
{
    int offset = winSize.width + 1;
    Size blockStep;
    int x, y, t, w, h;

    for (t = 8; t <= winSize.width / 2; t += 8)  //t = size of a cell. blocksize = 4*cellSize
    {
        blockStep = Size(4, 4);
        w = 2 * t;  //width of a block
        h = 2 * t;  //height of a block
        for (x = 0; x <= winSize.width - w; x += blockStep.width)
        {
            for (y = 0; y <= winSize.height - h; y += blockStep.height)
            {
                features.push_back(Feature(offset, x, y, t, t));
            }
        }
        w = 2 * t;
        h = 4 * t;
        for (x = 0; x <= winSize.width - w; x += blockStep.width)
        {
            for (y = 0; y <= winSize.height - h; y += blockStep.height)
            {
                features.push_back(Feature(offset, x, y, t, 2 * t));
            }
        }
        w = 4 * t;
        h = 2 * t;
        for (x = 0; x <= winSize.width - w; x += blockStep.width)
        {
            for (y = 0; y <= winSize.height - h; y += blockStep.height)
            {
                features.push_back(Feature(offset, x, y, 2 * t, t));
            }
        }
    }

    numFeatures = (int)features.size();
}

CvHOGEvaluator::Feature::Feature()
{
    for (int i = 0; i < N_CELLS; i++)
    {
        rect[i] = Rect(0, 0, 0, 0);
    }
}

CvHOGEvaluator::Feature::Feature(int offset, int x, int y, int cellW, int cellH)
{
    rect[0] = Rect(x, y, cellW, cellH);  //cell0
    rect[1] = Rect(x + cellW, y, cellW, cellH);  //cell1
    rect[2] = Rect(x, y + cellH, cellW, cellH);  //cell2
    rect[3] = Rect(x + cellW, y + cellH, cellW, cellH);  //cell3

    for (int i = 0; i < N_CELLS; i++)
    {
        CV_SUM_OFFSETS(fastRect[i].p0, fastRect[i].p1, fastRect[i].p2, fastRect[i].p3, rect[i], offset);
    }
}

void CvHOGEvaluator::Feature::write(FileStorage& fs) const
{
    fs << CC_RECTS << "[";
    for (int i = 0; i < N_CELLS; i++)
    {
        fs << "[:" << rect[i].x << rect[i].y << rect[i].width << rect[i].height << "]";
    }
    fs << "]";
}

//cell and bin idx writing
//void CvHOGEvaluator::Feature::write(FileStorage &fs, int varIdx) const
//{
//    int featComponent = varIdx % (N_CELLS * N_BINS);
//    int cellIdx = featComponent / N_BINS;
//    int binIdx = featComponent % N_BINS;
//
//    fs << CC_RECTS << "[:" << rect[cellIdx].x << rect[cellIdx].y <<
//        rect[cellIdx].width << rect[cellIdx].height << binIdx << "]";
//}

//cell[0] and featComponent idx writing. By cell[0] it's possible to recover all block
//All block is necessary for block normalization
void CvHOGEvaluator::Feature::write(FileStorage& fs, int featComponentIdx) const
{
    fs << CC_RECT << "[:" << rect[0].x << rect[0].y << rect[0].width << rect[0].height << featComponentIdx << "]";
}

void CvHOGEvaluator::integralHistogram(const Mat& img, std::vector<Mat>& histogram, Mat& norm, int nbins) const
{
    CV_Assert(img.type() == CV_8U || img.type() == CV_8UC3);
    int x, y, binIdx;

    Size gradSize(img.size());
    Size histSize(histogram[0].size());
    Mat grad(gradSize, CV_32F);
    Mat qangle(gradSize, CV_8U);

    AutoBuffer<int> mapbuf(gradSize.width + gradSize.height + 4);
    int* xmap = mapbuf.data() + 1;
    int* ymap = xmap + gradSize.width + 2;

    const int borderType = (int)BORDER_REPLICATE;

    for (x = -1; x < gradSize.width + 1; x++)
        xmap[x] = borderInterpolate(x, gradSize.width, borderType);
    for (y = -1; y < gradSize.height + 1; y++)
        ymap[y] = borderInterpolate(y, gradSize.height, borderType);

    int width = gradSize.width;
    AutoBuffer<float> _dbuf(width * 4);
    float* dbuf = _dbuf.data();
    Mat Dx(1, width, CV_32F, dbuf);
    Mat Dy(1, width, CV_32F, dbuf + width);
    Mat Mag(1, width, CV_32F, dbuf + width * 2);
    Mat Angle(1, width, CV_32F, dbuf + width * 3);

    float angleScale = (float)(nbins / CV_PI);

    for (y = 0; y < gradSize.height; y++)
    {
        const uchar* currPtr = img.data + img.step * ymap[y];
        const uchar* prevPtr = img.data + img.step * ymap[y - 1];
        const uchar* nextPtr = img.data + img.step * ymap[y + 1];
        float* gradPtr = (float*)grad.ptr(y);
        uchar* qanglePtr = (uchar*)qangle.ptr(y);

        for (x = 0; x < width; x++)
        {
            dbuf[x] = (float)(currPtr[xmap[x + 1]] - currPtr[xmap[x - 1]]);
            dbuf[width + x] = (float)(nextPtr[xmap[x]] - prevPtr[xmap[x]]);
        }
        cartToPolar(Dx, Dy, Mag, Angle, false);
        for (x = 0; x < width; x++)
        {
            float mag = dbuf[x + width * 2];
            float angle = dbuf[x + width * 3];
            angle = angle * angleScale - 0.5f;
            int bidx = cvFloor(angle);
            angle -= bidx;
            if (bidx < 0)
                bidx += nbins;
            else if (bidx >= nbins)
                bidx -= nbins;

            qanglePtr[x] = (uchar)bidx;
            gradPtr[x] = mag;
        }
    }
    integral(grad, norm, grad.depth());

    float* histBuf;
    const float* magBuf;
    const uchar* binsBuf;

    int binsStep = (int)(qangle.step / sizeof(uchar));
    int histStep = (int)(histogram[0].step / sizeof(float));
    int magStep = (int)(grad.step / sizeof(float));
    for (binIdx = 0; binIdx < nbins; binIdx++)
    {
        histBuf = (float*)histogram[binIdx].data;
        magBuf = (const float*)grad.data;
        binsBuf = (const uchar*)qangle.data;

        memset(histBuf, 0, histSize.width * sizeof(histBuf[0]));
        histBuf += histStep + 1;
        for (y = 0; y < qangle.rows; y++)
        {
            histBuf[-1] = 0.f;
            float strSum = 0.f;
            for (x = 0; x < qangle.cols; x++)
            {
                if (binsBuf[x] == binIdx)
                    strSum += magBuf[x];
                histBuf[x] = histBuf[-histStep + x] + strSum;
            }
            histBuf += histStep;
            binsBuf += binsStep;
            magBuf += magStep;
        }
    }
}

CvLBPFeatureParams::CvLBPFeatureParams()
{
    maxCatCount = 256;
    name = LBPF_NAME;
}

void CvLBPEvaluator::init(const CvFeatureParams* _featureParams, int _maxSampleCount, Size _winSize)
{
    CV_Assert(_maxSampleCount > 0);
    sum.create((int)_maxSampleCount, (_winSize.width + 1) * (_winSize.height + 1), CV_32SC1);
    CvFeatureEvaluator::init(_featureParams, _maxSampleCount, _winSize);
}

void CvLBPEvaluator::setImage(const Mat& img, uchar clsLabel, int idx)
{
    CV_DbgAssert(!sum.empty());
    CvFeatureEvaluator::setImage(img, clsLabel, idx);
    Mat innSum(winSize.height + 1, winSize.width + 1, sum.type(), sum.ptr<int>((int)idx));
    integral(img, innSum);
}

void CvLBPEvaluator::writeFeatures(FileStorage& fs, const Mat& featureMap) const
{
    _writeFeatures(features, fs, featureMap);
}

void CvLBPEvaluator::generateFeatures()
{
    int offset = winSize.width + 1;
    for (int x = 0; x < winSize.width; x++)
        for (int y = 0; y < winSize.height; y++)
            for (int w = 1; w <= winSize.width / 3; w++)
                for (int h = 1; h <= winSize.height / 3; h++)
                    if ((x + 3 * w <= winSize.width) && (y + 3 * h <= winSize.height))
                        features.push_back(Feature(offset, x, y, w, h));
    numFeatures = (int)features.size();
}

CvLBPEvaluator::Feature::Feature()
{
    rect = Rect(0, 0, 0, 0);
}

CvLBPEvaluator::Feature::Feature(int offset, int x, int y, int _blockWidth, int _blockHeight)
{
    Rect tr = rect = Rect(x, y, _blockWidth, _blockHeight);
    CV_SUM_OFFSETS(p[0], p[1], p[4], p[5], tr, offset)
    tr.x += 2 * rect.width;
    CV_SUM_OFFSETS(p[2], p[3], p[6], p[7], tr, offset)
    tr.y += 2 * rect.height;
    CV_SUM_OFFSETS(p[10], p[11], p[14], p[15], tr, offset)
    tr.x -= 2 * rect.width;
    CV_SUM_OFFSETS(p[8], p[9], p[12], p[13], tr, offset)
}

void CvLBPEvaluator::Feature::write(FileStorage& fs) const
{
    fs << CC_RECT << "[:" << rect.x << rect.y << rect.width << rect.height << "]";
}

}}}  // namespace cv::detail::tracking
