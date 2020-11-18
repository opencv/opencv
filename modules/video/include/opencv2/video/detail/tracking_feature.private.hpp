// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_VIDEO_DETAIL_TRACKING_FEATURE_HPP
#define OPENCV_VIDEO_DETAIL_TRACKING_FEATURE_HPP

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

/*
 * TODO This implementation is based on apps/traincascade/
 * TODO Changed CvHaarEvaluator based on ADABOOSTING implementation (Grabner et al.)
 */

namespace cv {
namespace detail {
inline namespace tracking {

//! @addtogroup tracking_detail
//! @{

inline namespace feature {

class CvParams
{
public:
    CvParams();
    virtual ~CvParams()
    {
    }
};

class CvFeatureParams : public CvParams
{
public:
    enum FeatureType
    {
        HAAR = 0,
        LBP = 1,
        HOG = 2
    };

    CvFeatureParams();
    static Ptr<CvFeatureParams> create(CvFeatureParams::FeatureType featureType);
    int maxCatCount;  // 0 in case of numerical features
    int featSize;  // 1 in case of simple features (HAAR, LBP) and N_BINS(9)*N_CELLS(4) in case of Dalal's HOG features
    int numFeatures;
};

class CvFeatureEvaluator
{
public:
    virtual ~CvFeatureEvaluator()
    {
    }
    virtual void init(const CvFeatureParams* _featureParams, int _maxSampleCount, Size _winSize);
    virtual void setImage(const Mat& img, uchar clsLabel, int idx);
    static Ptr<CvFeatureEvaluator> create(CvFeatureParams::FeatureType type);

    int getNumFeatures() const
    {
        return numFeatures;
    }
    int getMaxCatCount() const
    {
        return featureParams->maxCatCount;
    }
    int getFeatureSize() const
    {
        return featureParams->featSize;
    }
    const Mat& getCls() const
    {
        return cls;
    }
    float getCls(int si) const
    {
        return cls.at<float>(si, 0);
    }

protected:
    virtual void generateFeatures() = 0;

    int npos, nneg;
    int numFeatures;
    Size winSize;
    CvFeatureParams* featureParams;
    Mat cls;
};

class CvHaarFeatureParams : public CvFeatureParams
{
public:
    CvHaarFeatureParams();
    bool isIntegral;
};

class CvHaarEvaluator : public CvFeatureEvaluator
{
public:
    class FeatureHaar
    {

    public:
        FeatureHaar(Size patchSize);
        bool eval(const Mat& image, Rect ROI, float* result) const;
        inline int getNumAreas() const { return m_numAreas; }
        inline const std::vector<float>& getWeights() const { return m_weights; }
        inline const std::vector<Rect>& getAreas() const { return m_areas; }

    private:
        int m_type;
        int m_numAreas;
        std::vector<float> m_weights;
        float m_initMean;
        float m_initSigma;
        void generateRandomFeature(Size imageSize);
        float getSum(const Mat& image, Rect imgROI) const;
        std::vector<Rect> m_areas;  // areas within the patch over which to compute the feature
        cv::Size m_initSize;  // size of the patch used during training
        cv::Size m_curSize;  // size of the patches currently under investigation
        float m_scaleFactorHeight;  // scaling factor in vertical direction
        float m_scaleFactorWidth;  // scaling factor in horizontal direction
        std::vector<Rect> m_scaleAreas;  // areas after scaling
        std::vector<float> m_scaleWeights;  // weights after scaling
    };

    virtual void init(const CvFeatureParams* _featureParams, int _maxSampleCount, Size _winSize) CV_OVERRIDE;
    virtual void setImage(const Mat& img, uchar clsLabel = 0, int idx = 1) CV_OVERRIDE;
    inline const std::vector<CvHaarEvaluator::FeatureHaar>& getFeatures() const { return features; }
    inline CvHaarEvaluator::FeatureHaar& getFeatures(int idx)
    {
        return features[idx];
    }
    inline void setWinSize(Size patchSize) { winSize = patchSize; }
    inline Size getWinSize() const { return winSize; }
    virtual void generateFeatures() CV_OVERRIDE;

    /**
    * \brief Overload the original generateFeatures in order to limit the number of the features
    * @param numFeatures Number of the features
    */
    virtual void generateFeatures(int numFeatures);

protected:
    bool isIntegral;

    /* TODO Added from MIL implementation */
    Mat _ii_img;
    void compute_integral(const cv::Mat& img, std::vector<cv::Mat_<float>>& ii_imgs)
    {
        Mat ii_img;
        integral(img, ii_img, CV_32F);
        split(ii_img, ii_imgs);
    }

    std::vector<FeatureHaar> features;
    Mat sum; /* sum images (each row represents image) */
};

}  // namespace feature

//! @}

}}}  // namespace cv::detail::tracking

#endif
