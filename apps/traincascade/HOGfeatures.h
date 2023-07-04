#ifndef _OPENCV_HOGFEATURES_H_
#define _OPENCV_HOGFEATURES_H_

#include "traincascade_features.h"

//#define TEST_INTHIST_BUILD
//#define TEST_FEAT_CALC

#define N_BINS 9
#define N_CELLS 4

#define HOGF_NAME "HOGFeatureParams"
struct CvHOGFeatureParams : public CvFeatureParams
{
    CvHOGFeatureParams();
};

class CvHOGEvaluator : public CvFeatureEvaluator
{
public:
    virtual ~CvHOGEvaluator() {}
    virtual void init(const CvFeatureParams *_featureParams,
        int _maxSampleCount, cv::Size _winSize );
    virtual void setImage(const cv::Mat& img, uchar clsLabel, int idx);
    virtual float operator()(int varIdx, int sampleIdx) const;
    virtual void writeFeatures( cv::FileStorage &fs, const cv::Mat& featureMap ) const;
protected:
    virtual void generateFeatures();
    virtual void integralHistogram(const cv::Mat &img, std::vector<cv::Mat> &histogram, cv::Mat &norm, int nbins) const;
    class Feature
    {
    public:
        Feature();
        Feature( int offset, int x, int y, int cellW, int cellH );
        float calc( const std::vector<cv::Mat> &_hists, const cv::Mat &_normSum, size_t y, int featComponent ) const;
        void write( cv::FileStorage &fs ) const;
        void write( cv::FileStorage &fs, int varIdx ) const;

        cv::Rect rect[N_CELLS]; //cells

        struct
        {
            int p0, p1, p2, p3;
        } fastRect[N_CELLS];
    };
    std::vector<Feature> features;

    cv::Mat normSum; //for normalization calculation (L1 or L2)
    std::vector<cv::Mat> hist;
};

inline float CvHOGEvaluator::operator()(int varIdx, int sampleIdx) const
{
    int featureIdx = varIdx / (N_BINS * N_CELLS);
    int componentIdx = varIdx % (N_BINS * N_CELLS);
    //return features[featureIdx].calc( hist, sampleIdx, componentIdx);
    return features[featureIdx].calc( hist, normSum, sampleIdx, componentIdx);
}

inline float CvHOGEvaluator::Feature::calc( const std::vector<cv::Mat>& _hists, const cv::Mat& _normSum, size_t y, int featComponent ) const
{
    float normFactor;
    float res;

    int binIdx = featComponent % N_BINS;
    int cellIdx = featComponent / N_BINS;

    const float *phist = _hists[binIdx].ptr<float>((int)y);
    res = phist[fastRect[cellIdx].p0] - phist[fastRect[cellIdx].p1] - phist[fastRect[cellIdx].p2] + phist[fastRect[cellIdx].p3];

    const float *pnormSum = _normSum.ptr<float>((int)y);
    normFactor = (float)(pnormSum[fastRect[0].p0] - pnormSum[fastRect[1].p1] - pnormSum[fastRect[2].p2] + pnormSum[fastRect[3].p3]);
    res = (res > 0.001f) ? ( res / (normFactor + 0.001f) ) : 0.f; //for cutting negative values, which appear due to floating precision

    return res;
}

#endif // _OPENCV_HOGFEATURES_H_
