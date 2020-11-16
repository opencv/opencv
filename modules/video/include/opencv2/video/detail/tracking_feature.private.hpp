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

#define FEATURES "features"

#define CC_FEATURES FEATURES
#define CC_FEATURE_PARAMS "featureParams"
#define CC_MAX_CAT_COUNT "maxCatCount"
#define CC_FEATURE_SIZE "featSize"
#define CC_NUM_FEATURES "numFeat"
#define CC_ISINTEGRAL "isIntegral"
#define CC_RECTS "rects"
#define CC_TILTED "tilted"
#define CC_RECT "rect"

#define LBPF_NAME "lbpFeatureParams"
#define HOGF_NAME "HOGFeatureParams"
#define HFP_NAME "haarFeatureParams"

#define CV_HAAR_FEATURE_MAX 3
#define N_BINS 9
#define N_CELLS 4

#define CV_SUM_OFFSETS(p0, p1, p2, p3, rect, step)         \
    /* (x, y) */                                           \
    (p0) = (rect).x + (step) * (rect).y;                   \
    /* (x + w, y) */                                       \
    (p1) = (rect).x + (rect).width + (step) * (rect).y;    \
    /* (x + w, y) */                                       \
    (p2) = (rect).x + (step) * ((rect).y + (rect).height); \
    /* (x + w, y + h) */                                   \
    (p3) = (rect).x + (rect).width + (step) * ((rect).y + (rect).height);

#define CV_TILTED_OFFSETS(p0, p1, p2, p3, rect, step)                      \
    /* (x, y) */                                                           \
    (p0) = (rect).x + (step) * (rect).y;                                   \
    /* (x - h, y + h) */                                                   \
    (p1) = (rect).x - (rect).height + (step) * ((rect).y + (rect).height); \
    /* (x + w, y + w) */                                                   \
    (p2) = (rect).x + (rect).width + (step) * ((rect).y + (rect).width);   \
    /* (x + w - h, y + w + h) */                                           \
    (p3) = (rect).x + (rect).width - (rect).height                         \
            + (step) * ((rect).y + (rect).width + (rect).height);

float calcNormFactor(const Mat& sum, const Mat& sqSum);

template <class Feature>
void _writeFeatures(const std::vector<Feature> features, FileStorage& fs, const Mat& featureMap)
{
    fs << FEATURES << "[";
    const Mat_<int>& featureMap_ = (const Mat_<int>&)featureMap;
    for (int fi = 0; fi < featureMap.cols; fi++)
        if (featureMap_(0, fi) >= 0)
        {
            fs << "{";
            features[fi].write(fs);
            fs << "}";
        }
    fs << "]";
}

class CvParams
{
public:
    CvParams();
    virtual ~CvParams()
    {
    }
    // from|to file
    virtual void write(FileStorage& fs) const = 0;
    virtual bool read(const FileNode& node) = 0;
    // from|to screen
    virtual void printDefaults() const;
    virtual void printAttrs() const;
    virtual bool scanAttr(const std::string prmName, const std::string val);
    std::string name;
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
    virtual void init(const CvFeatureParams& fp);
    virtual void write(FileStorage& fs) const CV_OVERRIDE;
    virtual bool read(const FileNode& node) CV_OVERRIDE;
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
    virtual void writeFeatures(FileStorage& fs, const Mat& featureMap) const = 0;
    virtual float operator()(int featureIdx, int sampleIdx) = 0;
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

    virtual void init(const CvFeatureParams& fp) CV_OVERRIDE;
    virtual void write(FileStorage& fs) const CV_OVERRIDE;
    virtual bool read(const FileNode& node) CV_OVERRIDE;

    virtual void printDefaults() const CV_OVERRIDE;
    virtual void printAttrs() const CV_OVERRIDE;
    virtual bool scanAttr(const std::string prm, const std::string val) CV_OVERRIDE;

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
        int getNumAreas();
        const std::vector<float>& getWeights() const;
        const std::vector<Rect>& getAreas() const;
        void write(FileStorage) const {}
        float getInitMean() const;
        float getInitSigma() const;

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
    virtual float operator()(int featureIdx, int sampleIdx) CV_OVERRIDE;
    virtual void writeFeatures(FileStorage& fs, const Mat& featureMap) const CV_OVERRIDE;
    void writeFeature(FileStorage& fs) const;  // for old file format
    const std::vector<CvHaarEvaluator::FeatureHaar>& getFeatures() const;
    inline CvHaarEvaluator::FeatureHaar& getFeatures(int idx)
    {
        return features[idx];
    }
    void setWinSize(Size patchSize);
    Size setWinSize() const;
    virtual void generateFeatures() CV_OVERRIDE;

    /** @brief Overload the original generateFeatures in order to limit the number of the features
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

struct CvHOGFeatureParams : public CvFeatureParams
{
    CvHOGFeatureParams();
};

class CvHOGEvaluator : public CvFeatureEvaluator
{
public:
    virtual ~CvHOGEvaluator()
    {
    }
    virtual void init(const CvFeatureParams* _featureParams, int _maxSampleCount, Size _winSize) CV_OVERRIDE;
    virtual void setImage(const Mat& img, uchar clsLabel, int idx) CV_OVERRIDE;
    virtual float operator()(int varIdx, int sampleIdx) CV_OVERRIDE;
    virtual void writeFeatures(FileStorage& fs, const Mat& featureMap) const CV_OVERRIDE;

protected:
    virtual void generateFeatures() CV_OVERRIDE;
    virtual void integralHistogram(const Mat& img, std::vector<Mat>& histogram, Mat& norm, int nbins) const;
    class Feature
    {
    public:
        Feature();
        Feature(int offset, int x, int y, int cellW, int cellH);
        float calc(const std::vector<Mat>& _hists, const Mat& _normSum, size_t y, int featComponent) const;
        void write(FileStorage& fs) const;
        void write(FileStorage& fs, int varIdx) const;

        Rect rect[N_CELLS];  //cells

        struct
        {
            int p0, p1, p2, p3;
        } fastRect[N_CELLS];
    };
    std::vector<Feature> features;

    Mat normSum;  //for nomalization calculation (L1 or L2)
    std::vector<Mat> hist;
};

inline float CvHOGEvaluator::operator()(int varIdx, int sampleIdx)
{
    int featureIdx = varIdx / (N_BINS * N_CELLS);
    int componentIdx = varIdx % (N_BINS * N_CELLS);
    //return features[featureIdx].calc( hist, sampleIdx, componentIdx);
    return features[featureIdx].calc(hist, normSum, sampleIdx, componentIdx);
}

inline float CvHOGEvaluator::Feature::calc(const std::vector<Mat>& _hists, const Mat& _normSum, size_t y, int featComponent) const
{
    float normFactor;
    float res;

    int binIdx = featComponent % N_BINS;
    int cellIdx = featComponent / N_BINS;

    const float* phist = _hists[binIdx].ptr<float>((int)y);
    res = phist[fastRect[cellIdx].p0] - phist[fastRect[cellIdx].p1] - phist[fastRect[cellIdx].p2] + phist[fastRect[cellIdx].p3];

    const float* pnormSum = _normSum.ptr<float>((int)y);
    normFactor = (float)(pnormSum[fastRect[0].p0] - pnormSum[fastRect[1].p1] - pnormSum[fastRect[2].p2] + pnormSum[fastRect[3].p3]);
    res = (res > 0.001f) ? (res / (normFactor + 0.001f)) : 0.f;  //for cutting negative values, which apper due to floating precision

    return res;
}

struct CvLBPFeatureParams : CvFeatureParams
{
    CvLBPFeatureParams();
};

class CvLBPEvaluator : public CvFeatureEvaluator
{
public:
    virtual ~CvLBPEvaluator() CV_OVERRIDE
    {
    }
    virtual void init(const CvFeatureParams* _featureParams, int _maxSampleCount, Size _winSize) CV_OVERRIDE;
    virtual void setImage(const Mat& img, uchar clsLabel, int idx) CV_OVERRIDE;
    virtual float operator()(int featureIdx, int sampleIdx) CV_OVERRIDE
    {
        return (float)features[featureIdx].calc(sum, sampleIdx);
    }
    virtual void writeFeatures(FileStorage& fs, const Mat& featureMap) const CV_OVERRIDE;

protected:
    virtual void generateFeatures() CV_OVERRIDE;

    class Feature
    {
    public:
        Feature();
        Feature(int offset, int x, int y, int _block_w, int _block_h);
        uchar calc(const Mat& _sum, size_t y) const;
        void write(FileStorage& fs) const;

        Rect rect;
        int p[16];
    };
    std::vector<Feature> features;

    Mat sum;
};

inline uchar CvLBPEvaluator::Feature::calc(const Mat& _sum, size_t y) const
{
    const int* psum = _sum.ptr<int>((int)y);
    int cval = psum[p[5]] - psum[p[6]] - psum[p[9]] + psum[p[10]];

    return (uchar)((psum[p[0]] - psum[p[1]] - psum[p[4]] + psum[p[5]] >= cval ? 128 : 0) |  // 0
            (psum[p[1]] - psum[p[2]] - psum[p[5]] + psum[p[6]] >= cval ? 64 : 0) |  // 1
            (psum[p[2]] - psum[p[3]] - psum[p[6]] + psum[p[7]] >= cval ? 32 : 0) |  // 2
            (psum[p[6]] - psum[p[7]] - psum[p[10]] + psum[p[11]] >= cval ? 16 : 0) |  // 5
            (psum[p[10]] - psum[p[11]] - psum[p[14]] + psum[p[15]] >= cval ? 8 : 0) |  // 8
            (psum[p[9]] - psum[p[10]] - psum[p[13]] + psum[p[14]] >= cval ? 4 : 0) |  // 7
            (psum[p[8]] - psum[p[9]] - psum[p[12]] + psum[p[13]] >= cval ? 2 : 0) |  // 6
            (psum[p[4]] - psum[p[5]] - psum[p[8]] + psum[p[9]] >= cval ? 1 : 0));  // 3
}

}  // namespace feature

//! @}

}}}  // namespace cv::detail::tracking

#endif
