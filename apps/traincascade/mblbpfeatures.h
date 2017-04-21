#ifndef _OPENCV_MBLBPFEATURES_H_
#define _OPENCV_MBLBPFEATURES_H_

#include "traincascade_features.h"
#include "common.h"

#define MBLBPF_NAME "mblbpFeatureParams"
struct CvMBLBPFeatureParams : CvFeatureParams
{
    CvMBLBPFeatureParams();
};

class CvMBLBPEvaluator:public CvFeatureEvaluator
{
    public:
        virtual ~CvMBLBPEvaluator(){}
        virtual void init(const CvFeatureParams *_featureParams,
            int _maxSampleCount, cv::Size _winSize);
        virtual void setImage(const cv::Mat &img, int idx,bool isSum);
        virtual float operator()(int featureIdx,int sampleIdx) const
        { 
            return (float)features[featureIdx].calc(sum, 0);
        };
        virtual void writeFeatures(cv::FileStorage &fs, const cv::Mat& featureMap) const;
    protected:
        virtual void generateFeatures();
        class Feature
        {
        public:
            Feature();
            Feature(int offset, int x, int y, int _block_w,int _block_h);
            Feature(int x, int y, int cellwidth, int cellheight);
            uchar calc(const cv::Mat&_sum, size_t img_offset) const;
            void write (cv::FileStorage &fs) const;
            cv::Rect rect;
            int p[16];
            // MBLBP Parameters
            int x=0;
            int y=0;
            int cellwidth=0;
            int cellheight=0;
            int offsets[16];
            double soft_threshold=0.0;
            double look_up_table[MBLBP_LUTLENGTH];
        };
        cv::Mat sum;
        // feature parameters
        int numFeatures;
        bool *featuresMask;
        // cascade parameters
        int numPos;
        int numNeg;
        int numSamples;
        int maxWeakCount;
        float minHitRate;
        MBLBPCascadef cascade;
        cv::Mat samplesLBP;
        cv::Mat labels;
        std::vector<Feature> features;
};

inline uchar CvMBLBPEvaluator::Feature::calc(const cv::Mat &_sum, size_t img_offset=0) const
{
    const int* psum = _sum.ptr<int>(0);
    const int* p = offsets;

    int cval = psum[p[5]+img_offset] - psum[p[6]+img_offset] - psum[p[9]+img_offset] + psum[p[10]+img_offset]; 
    
	return        LBPMAP[(((psum[p[ 0]+img_offset ] - psum[p[ 1]+img_offset ] - psum[p[ 4]+img_offset ] + psum[p[ 5]+img_offset ] >= cval) << 7) |
                          ((psum[p[ 1]+img_offset ] - psum[p[ 2]+img_offset ] - psum[p[ 5]+img_offset ] + psum[p[ 6]+img_offset ] >= cval) << 6) |
                          ((psum[p[ 2]+img_offset ] - psum[p[ 3]+img_offset ] - psum[p[ 6]+img_offset ] + psum[p[ 7]+img_offset ] >= cval) << 5) |
                          ((psum[p[ 6]+img_offset ] - psum[p[ 7]+img_offset ] - psum[p[10]+img_offset ] + psum[p[11]+img_offset ] >= cval) << 4) |
                          ((psum[p[10]+img_offset ] - psum[p[11]+img_offset ] - psum[p[14]+img_offset ] + psum[p[15]+img_offset ] >= cval) << 3) |
                          ((psum[p[ 9]+img_offset ] - psum[p[10]+img_offset ] - psum[p[13]+img_offset ] + psum[p[14]+img_offset ] >= cval) << 2) |
                          ((psum[p[ 8]+img_offset ] - psum[p[ 9]+img_offset ] - psum[p[12]+img_offset ] + psum[p[13]+img_offset ] >= cval) << 1) |  
                          ( psum[p[ 4]+img_offset ] - psum[p[ 5]+img_offset ] - psum[p[ 8]+img_offset ] + psum[p[ 9]+img_offset ] >= cval)     )];
}

#endif
