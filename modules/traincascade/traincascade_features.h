#ifndef _OPENCV_FEATURES_H_
#define _OPENCV_FEATURES_H_

#include "imagestorage.h"
#include "cxcore.h"
#include "cv.h"
#include "ml.h"
#include <stdio.h>

#define FEATURES "features"

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
void _writeFeatures( const vector<Feature> features, FileStorage &fs, const Mat& featureMap )
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
    virtual bool scanAttr( const String prmName, const String val );
    String name;
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

#endif
