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
        int _maxSampleCount, Size _winSize );
    virtual void setImage(const Mat& img, uchar clsLabel, int idx);
    virtual float operator()(int featureIdx, int sampleIdx) const
    { return (float)features[featureIdx].calc( sum, sampleIdx); }
    virtual void writeFeatures( FileStorage &fs, const Mat& featureMap ) const;
protected:
    virtual void generateFeatures();

    class Feature
    {
    public:
        Feature();
        Feature( int offset, int x, int y, int _block_w, int _block_h  ); 
        uchar calc( const Mat& _sum, size_t y ) const;
        void write( FileStorage &fs ) const;

        Rect rect;
        int p[16];
    };
    vector<Feature> features;

    Mat sum;
};

inline uchar CvLBPEvaluator::Feature::calc(const Mat &_sum, size_t y) const
{
    const int* sum = _sum.ptr<int>((int)y);
    int cval = sum[p[5]] - sum[p[6]] - sum[p[9]] + sum[p[10]];

    return (uchar)((sum[p[0]] - sum[p[1]] - sum[p[4]] + sum[p[5]] >= cval ? 128 : 0) |   // 0
        (sum[p[1]] - sum[p[2]] - sum[p[5]] + sum[p[6]] >= cval ? 64 : 0) |    // 1
        (sum[p[2]] - sum[p[3]] - sum[p[6]] + sum[p[7]] >= cval ? 32 : 0) |    // 2
        (sum[p[6]] - sum[p[7]] - sum[p[10]] + sum[p[11]] >= cval ? 16 : 0) |  // 5
        (sum[p[10]] - sum[p[11]] - sum[p[14]] + sum[p[15]] >= cval ? 8 : 0) | // 8
        (sum[p[9]] - sum[p[10]] - sum[p[13]] + sum[p[14]] >= cval ? 4 : 0) |  // 7
        (sum[p[8]] - sum[p[9]] - sum[p[12]] + sum[p[13]] >= cval ? 2 : 0) |   // 6
        (sum[p[4]] - sum[p[5]] - sum[p[8]] + sum[p[9]] >= cval ? 1 : 0));     // 3
}

#endif
