#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"

#include "lbpfeatures.h"
#include "cascadeclassifier.h"

CvLBPFeatureParams::CvLBPFeatureParams()
{
    maxCatCount = 256;
    name = LBPF_NAME;
}

void CvLBPEvaluator::init(const CvFeatureParams *_featureParams, int _maxSampleCount, Size _winSize)
{
    CV_Assert( _maxSampleCount > 0);
    sum.create((int)_maxSampleCount, (_winSize.width + 1) * (_winSize.height + 1), CV_32SC1);
    CvFeatureEvaluator::init( _featureParams, _maxSampleCount, _winSize );
}

void CvLBPEvaluator::setImage(const Mat &img, uchar clsLabel, int idx)
{
    CV_DbgAssert( !sum.empty() );
    CvFeatureEvaluator::setImage( img, clsLabel, idx );
    Mat innSum(winSize.height + 1, winSize.width + 1, sum.type(), sum.ptr<int>((int)idx));
    integral( img, innSum );
}

void CvLBPEvaluator::writeFeatures( FileStorage &fs, const Mat& featureMap ) const
{
    _writeFeatures( features, fs, featureMap );
}

void CvLBPEvaluator::generateFeatures()
{
    int offset = winSize.width + 1;
    for( int x = 0; x < winSize.width; x++ )
        for( int y = 0; y < winSize.height; y++ )
            for( int w = 1; w <= winSize.width / 3; w++ )
                for( int h = 1; h <= winSize.height / 3; h++ )
                    if ( (x+3*w <= winSize.width) && (y+3*h <= winSize.height) )
                        features.push_back( Feature(offset, x, y, w, h ) );
    numFeatures = (int)features.size();
}

CvLBPEvaluator::Feature::Feature()
{
    rect = cvRect(0, 0, 0, 0);
}

CvLBPEvaluator::Feature::Feature( int offset, int x, int y, int _blockWidth, int _blockHeight )
{
    Rect tr = rect = cvRect(x, y, _blockWidth, _blockHeight);
    CV_SUM_OFFSETS( p[0], p[1], p[4], p[5], tr, offset )
    tr.x += 2*rect.width;
    CV_SUM_OFFSETS( p[2], p[3], p[6], p[7], tr, offset )
    tr.y +=2*rect.height;
    CV_SUM_OFFSETS( p[10], p[11], p[14], p[15], tr, offset )
    tr.x -= 2*rect.width;
    CV_SUM_OFFSETS( p[8], p[9], p[12], p[13], tr, offset )
}

void CvLBPEvaluator::Feature::write(FileStorage &fs) const
{
    fs << CC_RECT << "[:" << rect.x << rect.y << rect.width << rect.height << "]";
}
