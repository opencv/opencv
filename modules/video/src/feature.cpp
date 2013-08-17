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

#include "precomp.hpp"
#include "opencv2/video/feature.hpp"

namespace cv
{

/*
 * TODO This is a copy from apps/traincascade/
 * TODO Changed CvHaarEvaluator based on MIL implementation
 */


float calcNormFactor( const Mat& sum, const Mat& sqSum )
{
    CV_DbgAssert( sum.cols > 3 && sqSum.rows > 3 );
    Rect normrect( 1, 1, sum.cols - 3, sum.rows - 3 );
    size_t p0, p1, p2, p3;
    CV_SUM_OFFSETS( p0, p1, p2, p3, normrect, sum.step1() )
    double area = normrect.width * normrect.height;
    const int *sp = (const int*)sum.data;
    int valSum = sp[p0] - sp[p1] - sp[p2] + sp[p3];
    const double *sqp = (const double *)sqSum.data;
    double valSqSum = sqp[p0] - sqp[p1] - sqp[p2] + sqp[p3];
    return (float) sqrt( (double) (area * valSqSum - (double)valSum * valSum) );
}

CvParams::CvParams() : name( "params" ) {}
void CvParams::printDefaults() const
{ std::cout << "--" << name << "--" << std::endl; }
void CvParams::printAttrs() const {}
bool CvParams::scanAttr( const std::string, const std::string ) { return false; }


//---------------------------- FeatureParams --------------------------------------

CvFeatureParams::CvFeatureParams() : maxCatCount( 0 ), featSize( 1 ), numFeatures( 0 )
{
    name = CC_FEATURE_PARAMS;
}

void CvFeatureParams::init( const CvFeatureParams& fp )
{
    maxCatCount = fp.maxCatCount;
    featSize = fp.featSize;
}

void CvFeatureParams::write( FileStorage &fs ) const
{
    fs << CC_MAX_CAT_COUNT << maxCatCount;
    fs << CC_FEATURE_SIZE << featSize;
}

bool CvFeatureParams::read( const FileNode &node )
{
    if ( node.empty() )
        return false;
    maxCatCount = node[CC_MAX_CAT_COUNT];
    featSize = node[CC_FEATURE_SIZE];
    return ( maxCatCount >= 0 && featSize >= 1 );
}

Ptr<CvFeatureParams> CvFeatureParams::create( int featureType )
{
    return featureType == HAAR ? Ptr<CvFeatureParams>(new CvHaarFeatureParams) :
        featureType == LBP ? Ptr<CvFeatureParams>(new CvLBPFeatureParams) :
        featureType == HOG ? Ptr<CvFeatureParams>(new CvHOGFeatureParams) :
        Ptr<CvFeatureParams>();
}

//------------------------------------- FeatureEvaluator ---------------------------------------

void CvFeatureEvaluator::init(const CvFeatureParams *_featureParams,
                              int _maxSampleCount, Size _winSize )
{
    CV_Assert(_maxSampleCount > 0);
    featureParams = (CvFeatureParams *)_featureParams;
    winSize = _winSize;
    numFeatures = 0;
    cls.create( (int)_maxSampleCount, 1, CV_32FC1 );
    generateFeatures();
}

void CvFeatureEvaluator::setImage(const Mat &img, uchar clsLabel, int idx)
{
    CV_Assert(img.cols == winSize.width);
    CV_Assert(img.rows == winSize.height);
    CV_Assert(idx < cls.rows);
    cls.ptr<float>(idx)[0] = clsLabel;
}

Ptr<CvFeatureEvaluator> CvFeatureEvaluator::create(int type)
{
    return type == CvFeatureParams::HAAR ? Ptr<CvFeatureEvaluator>(new CvHaarEvaluator) :
        type == CvFeatureParams::LBP ? Ptr<CvFeatureEvaluator>(new CvLBPEvaluator) :
        type == CvFeatureParams::HOG ? Ptr<CvFeatureEvaluator>(new CvHOGEvaluator) :
        Ptr<CvFeatureEvaluator>();
}




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
    String modeStr = mode == BASIC ? CC_MODE_BASIC :
                     mode == CORE ? CC_MODE_CORE :
                     mode == ALL ? CC_MODE_ALL : String();
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
    String modeStr;
    rnode >> modeStr;
    mode = !modeStr.compare( CC_MODE_BASIC ) ? BASIC :
           !modeStr.compare( CC_MODE_CORE ) ? CORE :
           !modeStr.compare( CC_MODE_ALL ) ? ALL : -1;
    return (mode >= 0);
}

void CvHaarFeatureParams::printDefaults() const
{
    CvFeatureParams::printDefaults();
    std::cout << "  [-mode <" CC_MODE_BASIC << "(default) | "
            << CC_MODE_CORE <<" | " << CC_MODE_ALL << std::endl;
}

void CvHaarFeatureParams::printAttrs() const
{
    CvFeatureParams::printAttrs();
    std::string mode_str = mode == BASIC ? CC_MODE_BASIC :
                       mode == CORE ? CC_MODE_CORE :
                       mode == ALL ? CC_MODE_ALL : 0;
    std::cout << "mode: " <<  mode_str << std::endl;
}

bool CvHaarFeatureParams::scanAttr( const std::string prmName, const std::string val)
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
    std::vector<Mat_<float> > ii_imgs;
    compute_integral( img, ii_imgs );
    _ii_img = ii_imgs[0];
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

void CvHaarEvaluator::generateFeatures( int numFeatures )
{
	int mode = ((const CvHaarFeatureParams*)((CvFeatureParams*)featureParams))->mode;
	int offset = winSize.width + 1;
	bool tilted = false;
	if( mode == CvHaarFeatureParams::ALL )
		tilted = true;

	RNG rng = RNG( (int) time( 0 ) );

	for( int i = 0; i < numFeatures; i++ )
	{
		//generates new HAAR feature
		 int x0  = rng.uniform( 0, (uint) ( winSize.width - 3 ) );
		 int y0  = rng.uniform( 0, (uint) ( winSize.height - 3 ) );
		 int w0  = rng.uniform( 1, ( winSize.width - x0 - 2 ) );
		 int h0  = rng.uniform( 1, ( winSize.height - y0 - 2 ) );
		 float wt0 = rng.uniform( -1.f, 1.f);

		 int x1  = rng.uniform( 0, (uint) ( winSize.width - 3 ) );
		 int y1  = rng.uniform( 0, (uint) ( winSize.height - 3 ) );
		 int w1  = rng.uniform( 1, ( winSize.width - x1 - 2 ) );
		 int h1  = rng.uniform( 1, ( winSize.height - y1 - 2 ) );
		 float wt1 = rng.uniform( -1.f, 1.f);

		 int x2  = rng.uniform( 0, (uint) ( winSize.width - 3 ) );
		 int y2  = rng.uniform( 0, (uint) ( winSize.height - 3 ) );
		 int w2  = rng.uniform( 1, ( winSize.width - x2 - 2 ) );
		 int h2  = rng.uniform( 1, ( winSize.height - y2 - 2 ) );
		 float wt2 = rng.uniform( -1.f, 1.f );

		 features.push_back( Feature( offset, tilted,
			 x0, y0, w0, h0, wt0,
			 x1, y1, w1, h1, wt1,
			 x2, y2, w2, h2, wt2 ) );
	}

}
void CvHaarEvaluator::generateFeatures()
{
	if( featureParams->numFeatures != 0)
	{
		generateFeatures( featureParams->numFeatures );
	}
	else
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




CvHOGFeatureParams::CvHOGFeatureParams()
{
    maxCatCount = 0;
    name = HOGF_NAME;
    featSize = N_BINS * N_CELLS;
}

void CvHOGEvaluator::init(const CvFeatureParams *_featureParams, int _maxSampleCount, Size _winSize)
{
    CV_Assert( _maxSampleCount > 0);
    int cols = (_winSize.width + 1) * (_winSize.height + 1);
    for (int bin = 0; bin < N_BINS; bin++)
    {
        hist.push_back(Mat(_maxSampleCount, cols, CV_32FC1));
    }
    normSum.create( (int)_maxSampleCount, cols, CV_32FC1 );
    CvFeatureEvaluator::init( _featureParams, _maxSampleCount, _winSize );
}

void CvHOGEvaluator::setImage(const Mat &img, uchar clsLabel, int idx)
{
    CV_DbgAssert( !hist.empty());
    CvFeatureEvaluator::setImage( img, clsLabel, idx );
    std:: vector<Mat> integralHist;
    for (int bin = 0; bin < N_BINS; bin++)
    {
        integralHist.push_back( Mat(winSize.height + 1, winSize.width + 1, hist[bin].type(), hist[bin].ptr<float>((int)idx)) );
    }
    Mat integralNorm(winSize.height + 1, winSize.width + 1, normSum.type(), normSum.ptr<float>((int)idx));
    integralHistogram(img, integralHist, integralNorm, (int)N_BINS);
}

//void CvHOGEvaluator::writeFeatures( FileStorage &fs, const Mat& featureMap ) const
//{
//    _writeFeatures( features, fs, featureMap );
//}

void CvHOGEvaluator::writeFeatures( FileStorage &fs, const Mat& featureMap ) const
{
    int featIdx;
    int componentIdx;
    const Mat_<int>& featureMap_ = (const Mat_<int>&)featureMap;
    fs << FEATURES << "[";
    for ( int fi = 0; fi < featureMap.cols; fi++ )
        if ( featureMap_(0, fi) >= 0 )
        {
            fs << "{";
            featIdx = fi / getFeatureSize();
            componentIdx = fi % getFeatureSize();
            features[featIdx].write( fs, componentIdx );
            fs << "}";
        }
    fs << "]";
}

void CvHOGEvaluator::generateFeatures()
{
    int offset = winSize.width + 1;
    Size blockStep;
    int x, y, t, w, h;

    for (t = 8; t <= winSize.width/2; t+=8) //t = size of a cell. blocksize = 4*cellSize
    {
        blockStep = Size(4,4);
        w = 2*t; //width of a block
        h = 2*t; //height of a block
        for (x = 0; x <= winSize.width - w; x += blockStep.width)
        {
            for (y = 0; y <= winSize.height - h; y += blockStep.height)
            {
                features.push_back(Feature(offset, x, y, t, t));
            }
        }
        w = 2*t;
        h = 4*t;
        for (x = 0; x <= winSize.width - w; x += blockStep.width)
        {
            for (y = 0; y <= winSize.height - h; y += blockStep.height)
            {
                features.push_back(Feature(offset, x, y, t, 2*t));
            }
        }
        w = 4*t;
        h = 2*t;
        for (x = 0; x <= winSize.width - w; x += blockStep.width)
        {
            for (y = 0; y <= winSize.height - h; y += blockStep.height)
            {
                features.push_back(Feature(offset, x, y, 2*t, t));
            }
        }
    }

    numFeatures = (int)features.size();
}

CvHOGEvaluator::Feature::Feature()
{
    for (int i = 0; i < N_CELLS; i++)
    {
        rect[i] = Rect(0, 0, 0, 0);
    }
}

CvHOGEvaluator::Feature::Feature( int offset, int x, int y, int cellW, int cellH )
{
    rect[0] = Rect(x, y, cellW, cellH); //cell0
    rect[1] = Rect(x+cellW, y, cellW, cellH); //cell1
    rect[2] = Rect(x, y+cellH, cellW, cellH); //cell2
    rect[3] = Rect(x+cellW, y+cellH, cellW, cellH); //cell3

    for (int i = 0; i < N_CELLS; i++)
    {
        CV_SUM_OFFSETS(fastRect[i].p0, fastRect[i].p1, fastRect[i].p2, fastRect[i].p3, rect[i], offset);
    }
}

void CvHOGEvaluator::Feature::write(FileStorage &fs) const
{
    fs << CC_RECTS << "[";
    for( int i = 0; i < N_CELLS; i++ )
    {
        fs << "[:" << rect[i].x << rect[i].y << rect[i].width << rect[i].height << "]";
    }
    fs << "]";
}

//cell and bin idx writing
//void CvHOGEvaluator::Feature::write(FileStorage &fs, int varIdx) const
//{
//    int featComponent = varIdx % (N_CELLS * N_BINS);
//    int cellIdx = featComponent / N_BINS;
//    int binIdx = featComponent % N_BINS;
//
//    fs << CC_RECTS << "[:" << rect[cellIdx].x << rect[cellIdx].y <<
//        rect[cellIdx].width << rect[cellIdx].height << binIdx << "]";
//}

//cell[0] and featComponent idx writing. By cell[0] it's possible to recover all block
//All block is nessesary for block normalization
void CvHOGEvaluator::Feature::write(FileStorage &fs, int featComponentIdx) const
{
    fs << CC_RECT << "[:" << rect[0].x << rect[0].y <<
        rect[0].width << rect[0].height << featComponentIdx << "]";
}


void CvHOGEvaluator::integralHistogram(const Mat &img, std::vector<Mat> &histogram, Mat &norm, int nbins) const
{
    CV_Assert( img.type() == CV_8U || img.type() == CV_8UC3 );
    int x, y, binIdx;

    Size gradSize(img.size());
    Size histSize(histogram[0].size());
    Mat grad(gradSize, CV_32F);
    Mat qangle(gradSize, CV_8U);

    AutoBuffer<int> mapbuf(gradSize.width + gradSize.height + 4);
    int* xmap = (int*)mapbuf + 1;
    int* ymap = xmap + gradSize.width + 2;

    const int borderType = (int)BORDER_REPLICATE;

    for( x = -1; x < gradSize.width + 1; x++ )
        xmap[x] = borderInterpolate(x, gradSize.width, borderType);
    for( y = -1; y < gradSize.height + 1; y++ )
        ymap[y] = borderInterpolate(y, gradSize.height, borderType);

    int width = gradSize.width;
    AutoBuffer<float> _dbuf(width*4);
    float* dbuf = _dbuf;
    Mat Dx(1, width, CV_32F, dbuf);
    Mat Dy(1, width, CV_32F, dbuf + width);
    Mat Mag(1, width, CV_32F, dbuf + width*2);
    Mat Angle(1, width, CV_32F, dbuf + width*3);

    float angleScale = (float)(nbins/CV_PI);

    for( y = 0; y < gradSize.height; y++ )
    {
        const uchar* currPtr = img.data + img.step*ymap[y];
        const uchar* prevPtr = img.data + img.step*ymap[y-1];
        const uchar* nextPtr = img.data + img.step*ymap[y+1];
        float* gradPtr = (float*)grad.ptr(y);
        uchar* qanglePtr = (uchar*)qangle.ptr(y);

        for( x = 0; x < width; x++ )
        {
            dbuf[x] = (float)(currPtr[xmap[x+1]] - currPtr[xmap[x-1]]);
            dbuf[width + x] = (float)(nextPtr[xmap[x]] - prevPtr[xmap[x]]);
        }
        cartToPolar( Dx, Dy, Mag, Angle, false );
        for( x = 0; x < width; x++ )
        {
            float mag = dbuf[x+width*2];
            float angle = dbuf[x+width*3];
            angle = angle*angleScale - 0.5f;
            int bidx = cvFloor(angle);
            angle -= bidx;
            if( bidx < 0 )
                bidx += nbins;
            else if( bidx >= nbins )
                bidx -= nbins;

            qanglePtr[x] = (uchar)bidx;
            gradPtr[x] = mag;
        }
    }
    integral(grad, norm, grad.depth());

    float* histBuf;
    const float* magBuf;
    const uchar* binsBuf;

    int binsStep = (int)( qangle.step / sizeof(uchar) );
    int histStep = (int)( histogram[0].step / sizeof(float) );
    int magStep = (int)( grad.step / sizeof(float) );
    for( binIdx = 0; binIdx < nbins; binIdx++ )
    {
        histBuf = (float*)histogram[binIdx].data;
        magBuf = (const float*)grad.data;
        binsBuf = (const uchar*)qangle.data;

        memset( histBuf, 0, histSize.width * sizeof(histBuf[0]) );
        histBuf += histStep + 1;
        for( y = 0; y < qangle.rows; y++ )
        {
            histBuf[-1] = 0.f;
            float strSum = 0.f;
            for( x = 0; x < qangle.cols; x++ )
            {
                if( binsBuf[x] == binIdx )
                    strSum += magBuf[x];
                histBuf[x] = histBuf[-histStep + x] + strSum;
            }
            histBuf += histStep;
            binsBuf += binsStep;
            magBuf += magStep;
        }
    }
}


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
    rect = Rect(0, 0, 0, 0);
}

CvLBPEvaluator::Feature::Feature( int offset, int x, int y, int _blockWidth, int _blockHeight )
{
    Rect tr = rect = Rect(x, y, _blockWidth, _blockHeight);
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


} /* namespace cv */
