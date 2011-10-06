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
        int _maxSampleCount, Size _winSize );
    virtual void setImage(const Mat& img, uchar clsLabel, int idx);    
    virtual float operator()(int varIdx, int sampleIdx) const;
    virtual void writeFeatures( FileStorage &fs, const Mat& featureMap ) const;
protected:
    virtual void generateFeatures();
    virtual void integralHistogram(const Mat &img, vector<Mat> &histogram, Mat &norm, int nbins) const;
    class Feature
    {
    public:
        Feature();
        Feature( int offset, int x, int y, int cellW, int cellH ); 
        float calc( const vector<Mat> &_hists, const Mat &_normSum, size_t y, int featComponent ) const; 
        void write( FileStorage &fs ) const;
        void write( FileStorage &fs, int varIdx ) const;

        Rect rect[N_CELLS]; //cells

        struct
        {
            int p0, p1, p2, p3;
        } fastRect[N_CELLS];
    };
    vector<Feature> features;

    Mat normSum; //for nomalization calculation (L1 or L2)
    vector<Mat> hist;
};

inline float CvHOGEvaluator::operator()(int varIdx, int sampleIdx) const
{
    int featureIdx = varIdx / (N_BINS * N_CELLS);
    int componentIdx = varIdx % (N_BINS * N_CELLS);
    //return features[featureIdx].calc( hist, sampleIdx, componentIdx); 
    return features[featureIdx].calc( hist, normSum, sampleIdx, componentIdx); 
}

inline float CvHOGEvaluator::Feature::calc( const vector<Mat>& _hists, const Mat& _normSum, size_t y, int featComponent ) const
{
    float normFactor;
    float res;

    int binIdx = featComponent % N_BINS;
    int cellIdx = featComponent / N_BINS;

    const float *hist = _hists[binIdx].ptr<float>(y);
    res = hist[fastRect[cellIdx].p0] - hist[fastRect[cellIdx].p1] - hist[fastRect[cellIdx].p2] + hist[fastRect[cellIdx].p3];

    const float *normSum = _normSum.ptr<float>(y);
    normFactor = (float)(normSum[fastRect[0].p0] - normSum[fastRect[1].p1] - normSum[fastRect[2].p2] + normSum[fastRect[3].p3]);
    res = (res > 0.001f) ? ( res / (normFactor + 0.001f) ) : 0.f; //for cutting negative values, which apper due to floating precision

    return res;
}

#endif // _OPENCV_HOGFEATURES_H_
