/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_FEATURE_HPP__
#define __OPENCV_FEATURE_HPP__

#include "opencv2/core.hpp"
#include <iostream>
#include <string>

/*
 * TODO This is a copy from apps/traincascade/
 * TODO Changed CvHaarEvaluator based on MIL implementation
 */



namespace cv
{

#define FEATURES "features"

#define CC_FEATURES       FEATURES
#define CC_FEATURE_PARAMS "featureParams"
#define CC_MAX_CAT_COUNT  "maxCatCount"
#define CC_FEATURE_SIZE   "featSize"
#define CC_MODE        "mode"
#define CC_MODE_BASIC  "BASIC"
#define CC_MODE_CORE   "CORE"
#define CC_MODE_ALL    "ALL"
#define CC_RECTS       "rects"
#define CC_TILTED      "tilted"
#define CC_RECT "rect"


#define LBPF_NAME "lbpFeatureParams"
#define HOGF_NAME "HOGFeatureParams"
#define HFP_NAME "haarFeatureParams"

#define CV_HAAR_FEATURE_MAX 3
#define N_BINS 9
#define N_CELLS 4

#define CV_SUM_OFFSETS( p0, p1, p2, p3, rect, step )                      \
    /* (x, y) */                                                          \
    (p0) = (rect).x + (step) * (rect).y;                                  \
    /* (x + w, y) */                                                      \
    (p1) = (rect).x + (rect).width + (step) * (rect).y;                   \
    /* (x + w, y) */                                                      \
    (p2) = (rect).x + (step) * ((rect).y + (rect).height);                \
    /* (x + w, y + h) */                                                  \
    (p3) = (rect).x + (rect).width + (step) * ((rect).y + (rect).height);

#define CV_TILTED_OFFSETS( p0, p1, p2, p3, rect, step )                   \
    /* (x, y) */                                                          \
    (p0) = (rect).x + (step) * (rect).y;                                  \
    /* (x - h, y + h) */                                                  \
    (p1) = (rect).x - (rect).height + (step) * ((rect).y + (rect).height);\
    /* (x + w, y + w) */                                                  \
    (p2) = (rect).x + (rect).width + (step) * ((rect).y + (rect).width);  \
    /* (x + w - h, y + w + h) */                                          \
    (p3) = (rect).x + (rect).width - (rect).height                        \
           + (step) * ((rect).y + (rect).width + (rect).height);

float calcNormFactor( const Mat& sum, const Mat& sqSum );

template<class Feature>
void _writeFeatures( const std::vector<Feature> features, FileStorage &fs, const Mat& featureMap )
{
    fs << FEATURES << "[";
    const Mat_<int>& featureMap_ = (const Mat_<int>&)featureMap;
    for ( int fi = 0; fi < featureMap.cols; fi++ )
        if ( featureMap_(0, fi) >= 0 )
        {
            fs << "{";
            features[fi].write( fs );
            fs << "}";
        }
    fs << "]";
}

class CvParams
{
public:
    CvParams();
    virtual ~CvParams() {}
    // from|to file
    virtual void write( FileStorage &fs ) const = 0;
    virtual bool read( const FileNode &node ) = 0;
    // from|to screen
    virtual void printDefaults() const;
    virtual void printAttrs() const;
    virtual bool scanAttr( const std::string prmName, const std::string val );
    std::string name;
};

class CvFeatureParams : public CvParams
{
public:
    enum { HAAR = 0, LBP = 1, HOG = 2 };
    CvFeatureParams();
    virtual void init( const CvFeatureParams& fp );
    virtual void write( FileStorage &fs ) const;
    virtual bool read( const FileNode &node );
    static Ptr<CvFeatureParams> create( int featureType );
    int maxCatCount; // 0 in case of numerical features
    int featSize; // 1 in case of simple features (HAAR, LBP) and N_BINS(9)*N_CELLS(4) in case of Dalal's HOG features
    int numFeatures;
};

class CvFeatureEvaluator
{
public:
    virtual ~CvFeatureEvaluator() {}
    virtual void init(const CvFeatureParams *_featureParams,
                      int _maxSampleCount, Size _winSize );
    virtual void setImage(const Mat& img, uchar clsLabel, int idx);
    virtual void writeFeatures( FileStorage &fs, const Mat& featureMap ) const = 0;
    virtual float operator()(int featureIdx, int sampleIdx) const = 0;
    static Ptr<CvFeatureEvaluator> create(int type);

    int getNumFeatures() const { return numFeatures; }
    int getMaxCatCount() const { return featureParams->maxCatCount; }
    int getFeatureSize() const { return featureParams->featSize; }
    const Mat& getCls() const { return cls; }
    float getCls(int si) const { return cls.at<float>(si, 0); }
protected:
    virtual void generateFeatures() = 0;

    int npos, nneg;
    int numFeatures;
    Size winSize;
    CvFeatureParams *featureParams;
    Mat cls;
};


class CvHaarFeatureParams : public CvFeatureParams
{
public:
    enum { BASIC = 0, CORE = 1, ALL = 2 };
     /* 0 - BASIC = Viola
     *  1 - CORE  = All upright
     *  2 - ALL   = All features */

    CvHaarFeatureParams();
    CvHaarFeatureParams( int _mode );

    virtual void init( const CvFeatureParams& fp );
    virtual void write( FileStorage &fs ) const;
    virtual bool read( const FileNode &node );

    virtual void printDefaults() const;
    virtual void printAttrs() const;
    virtual bool scanAttr( const std::string prm, const std::string val);

    int mode;
};

class CvHaarEvaluator : public CvFeatureEvaluator
{
public:
    virtual void init(const CvFeatureParams *_featureParams,
        int _maxSampleCount, Size _winSize );
    virtual void setImage(const Mat& img, uchar clsLabel, int idx);
    virtual float operator()(int featureIdx, int sampleIdx) const;
    virtual void writeFeatures( FileStorage &fs, const Mat& featureMap ) const;
    void writeFeature( FileStorage &fs, int fi ) const; // for old file fornat
protected:
    /* TODO Added from MIL implementation */
    Mat _ii_img;
    void
    compute_integral(const cv::Mat & img, std::vector<cv::Mat_<float> > & ii_imgs)
    {
      cv::Mat ii_img;
      cv::integral(img, ii_img, CV_32F);
      cv::split(ii_img, ii_imgs);
    }


    virtual void generateFeatures();

    /**
     * TODO new method
     * \brief Overload the original generateFeatures in order to limit the number of the features
     * @param numFeatures Number of the features
     */

    virtual void generateFeatures( int numFeatures );

    class Feature
    {
    public:
        Feature();
        Feature( int offset, bool _tilted,
            int x0, int y0, int w0, int h0, float wt0,
            int x1, int y1, int w1, int h1, float wt1,
            int x2 = 0, int y2 = 0, int w2 = 0, int h2 = 0, float wt2 = 0.0F );
        float calc( const Mat &sum, const Mat &tilted, size_t y) const;
        void write( FileStorage &fs ) const;

        bool  tilted;
        struct
        {
            Rect r;
            float weight;
        } rect[CV_HAAR_FEATURE_MAX];

        struct
        {
            int p0, p1, p2, p3;
        } fastRect[CV_HAAR_FEATURE_MAX];
    };

    std::vector<Feature> features;
    Mat  sum;         /* sum images (each row represents image) */
    Mat  tilted;      /* tilted sum images (each row represents image) */
    Mat  normfactor;  /* normalization factor */
};

inline float CvHaarEvaluator::operator()(int featureIdx, int sampleIdx) const
{
  /* TODO Added from MIL implementation */
   // float nf = normfactor.at<float>(0, sampleIdx);
    //return !nf ? 0.0f : (features[featureIdx].calc( sum, tilted, sampleIdx)/nf);
  return features[featureIdx].calc(_ii_img, Mat(), 0);
}

inline float CvHaarEvaluator::Feature::calc( const Mat &_sum, const Mat &_tilted, size_t y) const
{
  /* TODO Added from MIL implementation */
  Mat_<float> ii_img( _sum );
  cv::Rect r;
  float sum = 0.0f;

  for (int k = 0; k < CV_HAAR_FEATURE_MAX; k++)
  {
    r = rect[k].r;
    sum +=
        rect[k].weight * (ii_img(r.y + r.height, r.x + r.width)
            + ii_img(r.y, r.x)
            - ii_img(r.y + r.height, r.x)
            - ii_img(r.y, r.x + r.width)); ///_rsums[k];
  }

  return (float) (sum);
  //return 0;
   /* const int* img = tilted ? _tilted.ptr<int>((int)y) : _sum.ptr<int>((int)y);
    float ret = rect[0].weight * (img[fastRect[0].p0] - img[fastRect[0].p1] - img[fastRect[0].p2] + img[fastRect[0].p3] ) +
        rect[1].weight * (img[fastRect[1].p0] - img[fastRect[1].p1] - img[fastRect[1].p2] + img[fastRect[1].p3] );
    if( rect[2].weight != 0.0f )
        ret += rect[2].weight * (img[fastRect[2].p0] - img[fastRect[2].p1] - img[fastRect[2].p2] + img[fastRect[2].p3] );
    return ret;*/
}


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
    virtual void integralHistogram(const Mat &img, std::vector<Mat> &histogram, Mat &norm, int nbins) const;
    class Feature
    {
    public:
        Feature();
        Feature( int offset, int x, int y, int cellW, int cellH );
        float calc( const std::vector<Mat> &_hists, const Mat &_normSum, size_t y, int featComponent ) const;
        void write( FileStorage &fs ) const;
        void write( FileStorage &fs, int varIdx ) const;

        Rect rect[N_CELLS]; //cells

        struct
        {
            int p0, p1, p2, p3;
        } fastRect[N_CELLS];
    };
    std::vector<Feature> features;

    Mat normSum; //for nomalization calculation (L1 or L2)
    std::vector<Mat> hist;
};

inline float CvHOGEvaluator::operator()(int varIdx, int sampleIdx) const
{
    int featureIdx = varIdx / (N_BINS * N_CELLS);
    int componentIdx = varIdx % (N_BINS * N_CELLS);
    //return features[featureIdx].calc( hist, sampleIdx, componentIdx);
    return features[featureIdx].calc( hist, normSum, sampleIdx, componentIdx);
}

inline float CvHOGEvaluator::Feature::calc( const std::vector<Mat>& _hists, const Mat& _normSum, size_t y, int featComponent ) const
{
    float normFactor;
    float res;

    int binIdx = featComponent % N_BINS;
    int cellIdx = featComponent / N_BINS;

    const float *phist = _hists[binIdx].ptr<float>((int)y);
    res = phist[fastRect[cellIdx].p0] - phist[fastRect[cellIdx].p1] - phist[fastRect[cellIdx].p2] + phist[fastRect[cellIdx].p3];

    const float *pnormSum = _normSum.ptr<float>((int)y);
    normFactor = (float)(pnormSum[fastRect[0].p0] - pnormSum[fastRect[1].p1] - pnormSum[fastRect[2].p2] + pnormSum[fastRect[3].p3]);
    res = (res > 0.001f) ? ( res / (normFactor + 0.001f) ) : 0.f; //for cutting negative values, which apper due to floating precision

    return res;
}

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
    std::vector<Feature> features;

    Mat sum;
};

inline uchar CvLBPEvaluator::Feature::calc(const Mat &_sum, size_t y) const
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


} /* namespace cv */

#endif
