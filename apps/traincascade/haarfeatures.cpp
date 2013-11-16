#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"

#include "haarfeatures.h"
#include "cascadeclassifier.h"

using namespace std;
using namespace cv;

CvHaarFeatureParams::CvHaarFeatureParams() : mode(BASIC)
{
    name = HFP_NAME;
}

CvHaarFeatureParams::CvHaarFeatureParams( int _mode ) : mode( _mode )
{
    name = HFP_NAME;
}

void CvHaarFeatureParams::init( const CvFeatureParams& fp )
{
    CvFeatureParams::init( fp );
    mode = ((const CvHaarFeatureParams&)fp).mode;
}

void CvHaarFeatureParams::write( FileStorage &fs ) const
{
    CvFeatureParams::write( fs );
    string modeStr = mode == BASIC ? CC_MODE_BASIC :
                     mode == CORE ? CC_MODE_CORE :
                     mode == ALL ? CC_MODE_ALL : string();
    CV_Assert( !modeStr.empty() );
    fs << CC_MODE << modeStr;
}

bool CvHaarFeatureParams::read( const FileNode &node )
{
    if( !CvFeatureParams::read( node ) )
        return false;

    FileNode rnode = node[CC_MODE];
    if( !rnode.isString() )
        return false;
    string modeStr;
    rnode >> modeStr;
    mode = !modeStr.compare( CC_MODE_BASIC ) ? BASIC :
           !modeStr.compare( CC_MODE_CORE ) ? CORE :
           !modeStr.compare( CC_MODE_ALL ) ? ALL : -1;
    return (mode >= 0);
}

void CvHaarFeatureParams::printDefaults() const
{
    CvFeatureParams::printDefaults();
    cout << "  [-mode <" CC_MODE_BASIC << "(default) | "
            << CC_MODE_CORE <<" | " << CC_MODE_ALL << endl;
}

void CvHaarFeatureParams::printAttrs() const
{
    CvFeatureParams::printAttrs();
    string mode_str = mode == BASIC ? CC_MODE_BASIC :
                       mode == CORE ? CC_MODE_CORE :
                       mode == ALL ? CC_MODE_ALL : 0;
    cout << "mode: " <<  mode_str << endl;
}

bool CvHaarFeatureParams::scanAttr( const string prmName, const string val)
{
    if ( !CvFeatureParams::scanAttr( prmName, val ) )
    {
        if( !prmName.compare("-mode") )
        {
            mode = !val.compare( CC_MODE_CORE ) ? CORE :
                   !val.compare( CC_MODE_ALL ) ? ALL :
                   !val.compare( CC_MODE_BASIC ) ? BASIC : -1;
            if (mode == -1)
                return false;
        }
        return false;
    }
    return true;
}

//--------------------- HaarFeatureEvaluator ----------------

void CvHaarEvaluator::init(const CvFeatureParams *_featureParams,
                           int _maxSampleCount, Size _winSize )
{
    CV_Assert(_maxSampleCount > 0);
    int cols = (_winSize.width + 1) * (_winSize.height + 1);
    sum.create((int)_maxSampleCount, cols, CV_32SC1);
    tilted.create((int)_maxSampleCount, cols, CV_32SC1);
    normfactor.create(1, (int)_maxSampleCount, CV_32FC1);
    CvFeatureEvaluator::init( _featureParams, _maxSampleCount, _winSize );
}

void CvHaarEvaluator::setImage(const Mat& img, uchar clsLabel, int idx)
{
    CV_DbgAssert( !sum.empty() && !tilted.empty() && !normfactor.empty() );
    CvFeatureEvaluator::setImage( img, clsLabel, idx);
    Mat innSum(winSize.height + 1, winSize.width + 1, sum.type(), sum.ptr<int>((int)idx));
    Mat innTilted(winSize.height + 1, winSize.width + 1, tilted.type(), tilted.ptr<int>((int)idx));
    Mat innSqSum;
    integral(img, innSum, innSqSum, innTilted);
    normfactor.ptr<float>(0)[idx] = calcNormFactor( innSum, innSqSum );
}

void CvHaarEvaluator::writeFeatures( FileStorage &fs, const Mat& featureMap ) const
{
    _writeFeatures( features, fs, featureMap );
}

void CvHaarEvaluator::writeFeature(FileStorage &fs, int fi) const
{
    CV_DbgAssert( fi < (int)features.size() );
    features[fi].write(fs);
}

void CvHaarEvaluator::generateFeatures()
{
    int mode = ((const CvHaarFeatureParams*)((CvFeatureParams*)featureParams))->mode;
    int offset = winSize.width + 1;
    for( int x = 0; x < winSize.width; x++ )
    {
        for( int y = 0; y < winSize.height; y++ )
        {
            for( int dx = 1; dx <= winSize.width; dx++ )
            {
                for( int dy = 1; dy <= winSize.height; dy++ )
                {
                    // haar_x2
                    if ( (x+dx*2 <= winSize.width) && (y+dy <= winSize.height) )
                    {
                        features.push_back( Feature( offset, false,
                            x,    y, dx*2, dy, -1,
                            x+dx, y, dx  , dy, +2 ) );
                    }
                    // haar_y2
                    if ( (x+dx <= winSize.width) && (y+dy*2 <= winSize.height) )
                    {
                        features.push_back( Feature( offset, false,
                            x,    y, dx, dy*2, -1,
                            x, y+dy, dx, dy,   +2 ) );
                    }
                    // haar_x3
                    if ( (x+dx*3 <= winSize.width) && (y+dy <= winSize.height) )
                    {
                        features.push_back( Feature( offset, false,
                            x,    y, dx*3, dy, -1,
                            x+dx, y, dx  , dy, +3 ) );
                    }
                    // haar_y3
                    if ( (x+dx <= winSize.width) && (y+dy*3 <= winSize.height) )
                    {
                        features.push_back( Feature( offset, false,
                            x, y,    dx, dy*3, -1,
                            x, y+dy, dx, dy,   +3 ) );
                    }
                    if( mode != CvHaarFeatureParams::BASIC )
                    {
                        // haar_x4
                        if ( (x+dx*4 <= winSize.width) && (y+dy <= winSize.height) )
                        {
                            features.push_back( Feature( offset, false,
                                x,    y, dx*4, dy, -1,
                                x+dx, y, dx*2, dy, +2 ) );
                        }
                        // haar_y4
                        if ( (x+dx <= winSize.width ) && (y+dy*4 <= winSize.height) )
                        {
                            features.push_back( Feature( offset, false,
                                x, y,    dx, dy*4, -1,
                                x, y+dy, dx, dy*2, +2 ) );
                        }
                    }
                    // x2_y2
                    if ( (x+dx*2 <= winSize.width) && (y+dy*2 <= winSize.height) )
                    {
                        features.push_back( Feature( offset, false,
                            x,    y,    dx*2, dy*2, -1,
                            x,    y,    dx,   dy,   +2,
                            x+dx, y+dy, dx,   dy,   +2 ) );
                    }
                    if (mode != CvHaarFeatureParams::BASIC)
                    {
                        if ( (x+dx*3 <= winSize.width) && (y+dy*3 <= winSize.height) )
                        {
                            features.push_back( Feature( offset, false,
                                x   , y   , dx*3, dy*3, -1,
                                x+dx, y+dy, dx  , dy  , +9) );
                        }
                    }
                    if (mode == CvHaarFeatureParams::ALL)
                    {
                        // tilted haar_x2
                        if ( (x+2*dx <= winSize.width) && (y+2*dx+dy <= winSize.height) && (x-dy>= 0) )
                        {
                            features.push_back( Feature( offset, true,
                                x, y, dx*2, dy, -1,
                                x, y, dx,   dy, +2 ) );
                        }
                        // tilted haar_y2
                        if ( (x+dx <= winSize.width) && (y+dx+2*dy <= winSize.height) && (x-2*dy>= 0) )
                        {
                            features.push_back( Feature( offset, true,
                                x, y, dx, 2*dy, -1,
                                x, y, dx, dy,   +2 ) );
                        }
                        // tilted haar_x3
                        if ( (x+3*dx <= winSize.width) && (y+3*dx+dy <= winSize.height) && (x-dy>= 0) )
                        {
                            features.push_back( Feature( offset, true,
                                x,    y,    dx*3, dy, -1,
                                x+dx, y+dx, dx,   dy, +3 ) );
                        }
                        // tilted haar_y3
                        if ( (x+dx <= winSize.width) && (y+dx+3*dy <= winSize.height) && (x-3*dy>= 0) )
                        {
                            features.push_back( Feature( offset, true,
                                x,    y,    dx, 3*dy, -1,
                                x-dy, y+dy, dx, dy,   +3 ) );
                        }
                        // tilted haar_x4
                        if ( (x+4*dx <= winSize.width) && (y+4*dx+dy <= winSize.height) && (x-dy>= 0) )
                        {
                            features.push_back( Feature( offset, true,
                                x,    y,    dx*4, dy, -1,
                                x+dx, y+dx, dx*2, dy, +2 ) );
                        }
                        // tilted haar_y4
                        if ( (x+dx <= winSize.width) && (y+dx+4*dy <= winSize.height) && (x-4*dy>= 0) )
                        {
                            features.push_back( Feature( offset, true,
                                x,    y,    dx, 4*dy, -1,
                                x-dy, y+dy, dx, 2*dy, +2 ) );
                        }
                    }
                }
            }
        }
    }
    numFeatures = (int)features.size();
}

CvHaarEvaluator::Feature::Feature()
{
    tilted = false;
    rect[0].r = rect[1].r = rect[2].r = Rect(0,0,0,0);
    rect[0].weight = rect[1].weight = rect[2].weight = 0;
}

CvHaarEvaluator::Feature::Feature( int offset, bool _tilted,
                                          int x0, int y0, int w0, int h0, float wt0,
                                          int x1, int y1, int w1, int h1, float wt1,
                                          int x2, int y2, int w2, int h2, float wt2 )
{
    tilted = _tilted;

    rect[0].r.x = x0;
    rect[0].r.y = y0;
    rect[0].r.width  = w0;
    rect[0].r.height = h0;
    rect[0].weight   = wt0;

    rect[1].r.x = x1;
    rect[1].r.y = y1;
    rect[1].r.width  = w1;
    rect[1].r.height = h1;
    rect[1].weight   = wt1;

    rect[2].r.x = x2;
    rect[2].r.y = y2;
    rect[2].r.width  = w2;
    rect[2].r.height = h2;
    rect[2].weight   = wt2;

    if( !tilted )
    {
        for( int j = 0; j < CV_HAAR_FEATURE_MAX; j++ )
        {
            if( rect[j].weight == 0.0F )
                break;
            CV_SUM_OFFSETS( fastRect[j].p0, fastRect[j].p1, fastRect[j].p2, fastRect[j].p3, rect[j].r, offset )
        }
    }
    else
    {
        for( int j = 0; j < CV_HAAR_FEATURE_MAX; j++ )
        {
            if( rect[j].weight == 0.0F )
                break;
            CV_TILTED_OFFSETS( fastRect[j].p0, fastRect[j].p1, fastRect[j].p2, fastRect[j].p3, rect[j].r, offset )
        }
    }
}

void CvHaarEvaluator::Feature::write( FileStorage &fs ) const
{
    fs << CC_RECTS << "[";
    for( int ri = 0; ri < CV_HAAR_FEATURE_MAX && rect[ri].r.width != 0; ++ri )
    {
        fs << "[:" << rect[ri].r.x << rect[ri].r.y <<
            rect[ri].r.width << rect[ri].r.height << rect[ri].weight << "]";
    }
    fs << "]" << CC_TILTED << tilted;
}
