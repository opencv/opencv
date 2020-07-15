#ifndef _OPENCV_LBPFEATURES_H_
#define _OPENCV_LBPFEATURES_H_

#include "traincascade_features.h"

#define LBPF_NAME "lbpFeatureParams"
struct CvLBPFeatureParams : CvFeatureParams
{
    CvLBPFeatureParams();

};

class CvLBPEvaluator : public CvFeatureEvaluator
{
public:
    virtual ~CvLBPEvaluator() {}
    virtual void init(const CvFeatureParams *_featureParams,
        int _maxSampleCount, cv::Size _winSize );
    virtual void setImage(const cv::Mat& img, uchar clsLabel, int idx);
    virtual float operator()(int featureIdx, int sampleIdx) const
    { return (float)features[featureIdx].calc( sum, sampleIdx); }
    virtual void writeFeatures( cv::FileStorage &fs, const cv::Mat& featureMap ) const;
protected:
    virtual void generateFeatures();

    class Feature
    {
    public:
        Feature();
        Feature( int offset, int x, int y, int _block_w, int _block_h  );
        uchar calc( const cv::Mat& _sum, size_t y ) const;
        void write( cv::FileStorage &fs ) const;

        cv::Rect rect;
        int p[16];
    };
    std::vector<Feature> features;

    cv::Mat sum;
};

inline uchar CvLBPEvaluator::Feature::calc(const cv::Mat &_sum, size_t y) const
{
    const int* psum = _sum.ptr<int>((int)y);
    int cval = psum[p[5]] - psum[p[6]] - psum[p[9]] + psum[p[10]];

    return (uchar)((psum[p[0]] - psum[p[1]] - psum[p[4]] + psum[p[5]] >= cval ? 128 : 0) |   // 0
        (psum[p[1]] - psum[p[2]] - psum[p[5]] + psum[p[6]] >= cval ? 64 : 0) |    // 1
        (psum[p[2]] - psum[p[3]] - psum[p[6]] + psum[p[7]] >= cval ? 32 : 0) |    // 2
        (psum[p[6]] - psum[p[7]] - psum[p[10]] + psum[p[11]] >= cval ? 16 : 0) |  // 5
        (psum[p[10]] - psum[p[11]] - psum[p[14]] + psum[p[15]] >= cval ? 8 : 0) | // 8
        (psum[p[9]] - psum[p[10]] - psum[p[13]] + psum[p[14]] >= cval ? 4 : 0) |  // 7
        (psum[p[8]] - psum[p[9]] - psum[p[12]] + psum[p[13]] >= cval ? 2 : 0) |   // 6
        (psum[p[4]] - psum[p[5]] - psum[p[8]] + psum[p[9]] >= cval ? 1 : 0));     // 3
}

#endif
