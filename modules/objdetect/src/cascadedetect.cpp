/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
#include <cstdio>

#include "cascadedetect.hpp"

namespace cv
{
    
// class for grouping object candidates, detected by Cascade Classifier, HOG etc.
// instance of the class is to be passed to cv::partition (see cxoperations.hpp)
class CV_EXPORTS SimilarRects
{
public:    
    SimilarRects(double _eps) : eps(_eps) {}
    inline bool operator()(const Rect& r1, const Rect& r2) const
    {
        double delta = eps*(std::min(r1.width, r2.width) + std::min(r1.height, r2.height))*0.5;
        return std::abs(r1.x - r2.x) <= delta &&
        std::abs(r1.y - r2.y) <= delta &&
        std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
        std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
    }
    double eps;
};    
    

void groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps, vector<int>* weights, vector<double>* levelWeights)
{
    if( groupThreshold <= 0 || rectList.empty() )
    {
        if( weights )
        {
            size_t i, sz = rectList.size();
            weights->resize(sz);
            for( i = 0; i < sz; i++ )
                (*weights)[i] = 1;
        }
        return;
    }
    
    vector<int> labels;
    int nclasses = partition(rectList, labels, SimilarRects(eps));
    
    vector<Rect> rrects(nclasses);
    vector<int> rweights(nclasses, 0);
	vector<int> rejectLevels(nclasses, 0);
    vector<double> rejectWeights(nclasses, DBL_MIN);
    int i, j, nlabels = (int)labels.size();
    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        rweights[cls]++;
    }
    if ( levelWeights && weights && !weights->empty() && !levelWeights->empty() )
	{
		for( i = 0; i < nlabels; i++ )
		{
			int cls = labels[i];
            if( (*weights)[i] > rejectLevels[cls] )
            {
                rejectLevels[cls] = (*weights)[i];
                rejectWeights[cls] = (*levelWeights)[i];
            }
            else if( ( (*weights)[i] == rejectLevels[cls] ) && ( (*levelWeights)[i] > rejectWeights[cls] ) )
                rejectWeights[cls] = (*levelWeights)[i];
		}
	}
    
    for( i = 0; i < nclasses; i++ )
    {
        Rect r = rrects[i];
        float s = 1.f/rweights[i];
        rrects[i] = Rect(saturate_cast<int>(r.x*s),
             saturate_cast<int>(r.y*s),
             saturate_cast<int>(r.width*s),
             saturate_cast<int>(r.height*s));
    }
    
    rectList.clear();
    if( weights )
        weights->clear();
	if( levelWeights )
		levelWeights->clear();
    
    for( i = 0; i < nclasses; i++ )
    {
        Rect r1 = rrects[i];
        int n1 = levelWeights ? rejectLevels[i] : rweights[i];
		double w1 = rejectWeights[i];
        if( n1 <= groupThreshold )
            continue;
        // filter out small face rectangles inside large rectangles
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = rweights[j];
            
            if( j == i || n2 <= groupThreshold )
                continue;
            Rect r2 = rrects[j];
            
            int dx = saturate_cast<int>( r2.width * eps );
            int dy = saturate_cast<int>( r2.height * eps );
            
            if( i != j &&
                r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }
        
        if( j == nclasses )
        {
            rectList.push_back(r1);
            if( weights )
                weights->push_back(n1);
			if( levelWeights )
				levelWeights->push_back(w1);
        }
    }
}

class MeanshiftGrouping
{
public:
	MeanshiftGrouping(const Point3d& densKer, const vector<Point3d>& posV, 
		const vector<double>& wV, double, int maxIter = 20)
    {
	    densityKernel = densKer;
        weightsV = wV;
        positionsV = posV;
        positionsCount = (int)posV.size();
	    meanshiftV.resize(positionsCount);
        distanceV.resize(positionsCount);
	    iterMax = maxIter;
        
	    for (unsigned i = 0; i<positionsV.size(); i++)
	    {
		    meanshiftV[i] = getNewValue(positionsV[i]);
		    distanceV[i] = moveToMode(meanshiftV[i]);
		    meanshiftV[i] -= positionsV[i];
	    }
    }
	
	void getModes(vector<Point3d>& modesV, vector<double>& resWeightsV, const double eps)
    {
	    for (size_t i=0; i <distanceV.size(); i++)
	    {
		    bool is_found = false;
		    for(size_t j=0; j<modesV.size(); j++)
		    {
			    if ( getDistance(distanceV[i], modesV[j]) < eps)
			    {
				    is_found=true;
				    break;
			    }
		    }
		    if (!is_found)
		    {
			    modesV.push_back(distanceV[i]);
		    }
	    }
    	
	    resWeightsV.resize(modesV.size());

	    for (size_t i=0; i<modesV.size(); i++)
	    {
		    resWeightsV[i] = getResultWeight(modesV[i]);
	    }
    }

protected:
	vector<Point3d> positionsV;
	vector<double> weightsV;

	Point3d densityKernel;
	int positionsCount;

	vector<Point3d> meanshiftV;
	vector<Point3d> distanceV;
	int iterMax;
	double modeEps;

	Point3d getNewValue(const Point3d& inPt) const
    {
	    Point3d resPoint(.0);
	    Point3d ratPoint(.0);
	    for (size_t i=0; i<positionsV.size(); i++)
	    {
		    Point3d aPt= positionsV[i];
		    Point3d bPt = inPt;
		    Point3d sPt = densityKernel;
    		
		    sPt.x *= exp(aPt.z);
		    sPt.y *= exp(aPt.z);
    		
		    aPt.x /= sPt.x;
		    aPt.y /= sPt.y;
		    aPt.z /= sPt.z;

		    bPt.x /= sPt.x;
		    bPt.y /= sPt.y;
		    bPt.z /= sPt.z;
    		
		    double w = (weightsV[i])*std::exp(-((aPt-bPt).dot(aPt-bPt))/2)/std::sqrt(sPt.dot(Point3d(1,1,1)));
    		
		    resPoint += w*aPt;

		    ratPoint.x += w/sPt.x;
		    ratPoint.y += w/sPt.y;
		    ratPoint.z += w/sPt.z;
	    }
	    resPoint.x /= ratPoint.x;
	    resPoint.y /= ratPoint.y;
	    resPoint.z /= ratPoint.z;
	    return resPoint;
    }

	double getResultWeight(const Point3d& inPt) const
    {
	    double sumW=0;
	    for (size_t i=0; i<positionsV.size(); i++)
	    {
		    Point3d aPt = positionsV[i];
		    Point3d sPt = densityKernel;

		    sPt.x *= exp(aPt.z);
		    sPt.y *= exp(aPt.z);

		    aPt -= inPt;
    		
		    aPt.x /= sPt.x;
		    aPt.y /= sPt.y;
		    aPt.z /= sPt.z;
    		
		    sumW+=(weightsV[i])*std::exp(-(aPt.dot(aPt))/2)/std::sqrt(sPt.dot(Point3d(1,1,1)));
	    }
	    return sumW;
    }
    
	Point3d moveToMode(Point3d aPt) const
    {
	    Point3d bPt;
	    for (int i = 0; i<iterMax; i++)
	    {
		    bPt = aPt;
		    aPt = getNewValue(bPt);
		    if ( getDistance(aPt, bPt) <= modeEps )
		    {
			    break;
		    }
	    }
	    return aPt;
    }

    double getDistance(Point3d p1, Point3d p2) const
    {
	    Point3d ns = densityKernel;
	    ns.x *= exp(p2.z);
	    ns.y *= exp(p2.z);
	    p2 -= p1;
	    p2.x /= ns.x;
	    p2.y /= ns.y;
	    p2.z /= ns.z;
	    return p2.dot(p2);
    }
};
//new grouping function with using meanshift
static void groupRectangles_meanshift(vector<Rect>& rectList, double detectThreshold, vector<double>* foundWeights, 
									  vector<double>& scales, Size winDetSize)
{
    int detectionCount = (int)rectList.size();
    vector<Point3d> hits(detectionCount), resultHits;
    vector<double> hitWeights(detectionCount), resultWeights;
    Point2d hitCenter;

    for (int i=0; i < detectionCount; i++) 
    {
        hitWeights[i] = (*foundWeights)[i];
        hitCenter = (rectList[i].tl() + rectList[i].br())*(0.5); //center of rectangles
        hits[i] = Point3d(hitCenter.x, hitCenter.y, std::log(scales[i]));
    }

    rectList.clear();
    if (foundWeights)
        foundWeights->clear();

    double logZ = std::log(1.3);
    Point3d smothing(8, 16, logZ);

    MeanshiftGrouping msGrouping(smothing, hits, hitWeights, 1e-5, 100);

    msGrouping.getModes(resultHits, resultWeights, 1);

    for (unsigned i=0; i < resultHits.size(); ++i) 
    {

        double scale = exp(resultHits[i].z);
        hitCenter.x = resultHits[i].x;
        hitCenter.y = resultHits[i].y;
        Size s( int(winDetSize.width * scale), int(winDetSize.height * scale) );
        Rect resultRect( int(hitCenter.x-s.width/2), int(hitCenter.y-s.height/2), 
            int(s.width), int(s.height) ); 

        if (resultWeights[i] > detectThreshold) 
        {
            rectList.push_back(resultRect);
            foundWeights->push_back(resultWeights[i]);
        }
    }
}

void groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, 0, 0);
}

void groupRectangles(vector<Rect>& rectList, vector<int>& weights, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, &weights, 0);
}
//used for cascade detection algorithm for ROC-curve calculating
void groupRectangles(vector<Rect>& rectList, vector<int>& rejectLevels, vector<double>& levelWeights, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, &rejectLevels, &levelWeights);
}
//can be used for HOG detection algorithm only
void groupRectangles_meanshift(vector<Rect>& rectList, vector<double>& foundWeights, 
							   vector<double>& foundScales, double detectThreshold, Size winDetSize)
{
	groupRectangles_meanshift(rectList, detectThreshold, &foundWeights, foundScales, winDetSize);
}

    

FeatureEvaluator::~FeatureEvaluator() {}
bool FeatureEvaluator::read(const FileNode&) {return true;}
Ptr<FeatureEvaluator> FeatureEvaluator::clone() const { return Ptr<FeatureEvaluator>(); }
int FeatureEvaluator::getFeatureType() const {return -1;}
bool FeatureEvaluator::setImage(const Mat&, Size) {return true;}
bool FeatureEvaluator::setWindow(Point) { return true; }
double FeatureEvaluator::calcOrd(int) const { return 0.; }
int FeatureEvaluator::calcCat(int) const { return 0; }

//----------------------------------------------  HaarEvaluator ---------------------------------------

bool HaarEvaluator::Feature :: read( const FileNode& node )
{
    FileNode rnode = node[CC_RECTS];
    FileNodeIterator it = rnode.begin(), it_end = rnode.end();
    
    int ri;
    for( ri = 0; ri < RECT_NUM; ri++ )
    {
        rect[ri].r = Rect();
        rect[ri].weight = 0.f;
    }
    
    for(ri = 0; it != it_end; ++it, ri++)
    {
        FileNodeIterator it2 = (*it).begin();
        it2 >> rect[ri].r.x >> rect[ri].r.y >>
            rect[ri].r.width >> rect[ri].r.height >> rect[ri].weight;
    }
    
    tilted = (int)node[CC_TILTED] != 0;
    return true;
}

HaarEvaluator::HaarEvaluator()
{
    features = new vector<Feature>();
}
HaarEvaluator::~HaarEvaluator()
{
}

bool HaarEvaluator::read(const FileNode& node)
{
    features->resize(node.size());
    featuresPtr = &(*features)[0];
    FileNodeIterator it = node.begin(), it_end = node.end();
    hasTiltedFeatures = false;
    
    for(int i = 0; it != it_end; ++it, i++)
    {
        if(!featuresPtr[i].read(*it))
            return false;
        if( featuresPtr[i].tilted )
            hasTiltedFeatures = true;
    }
    return true;
}
    
Ptr<FeatureEvaluator> HaarEvaluator::clone() const
{
    HaarEvaluator* ret = new HaarEvaluator;
    ret->origWinSize = origWinSize;
    ret->features = features;
    ret->featuresPtr = &(*ret->features)[0];
    ret->hasTiltedFeatures = hasTiltedFeatures;
    ret->sum0 = sum0, ret->sqsum0 = sqsum0, ret->tilted0 = tilted0;
    ret->sum = sum, ret->sqsum = sqsum, ret->tilted = tilted;
    ret->normrect = normrect;
    memcpy( ret->p, p, 4*sizeof(p[0]) );
    memcpy( ret->pq, pq, 4*sizeof(pq[0]) );
    ret->offset = offset;
    ret->varianceNormFactor = varianceNormFactor; 
    return ret;
}

bool HaarEvaluator::setImage( const Mat &image, Size _origWinSize )
{
    int rn = image.rows+1, cn = image.cols+1;
    origWinSize = _origWinSize;
    normrect = Rect(1, 1, origWinSize.width-2, origWinSize.height-2);
    
    if (image.cols < origWinSize.width || image.rows < origWinSize.height)
        return false;
    
    if( sum0.rows < rn || sum0.cols < cn )
    {
        sum0.create(rn, cn, CV_32S);
        sqsum0.create(rn, cn, CV_64F);
        if (hasTiltedFeatures)
            tilted0.create( rn, cn, CV_32S);
    }
    sum = Mat(rn, cn, CV_32S, sum0.data);
    sqsum = Mat(rn, cn, CV_64F, sqsum0.data);

    if( hasTiltedFeatures )
    {
        tilted = Mat(rn, cn, CV_32S, tilted0.data);
        integral(image, sum, sqsum, tilted);
    }
    else
        integral(image, sum, sqsum);
    const int* sdata = (const int*)sum.data;
    const double* sqdata = (const double*)sqsum.data;
    size_t sumStep = sum.step/sizeof(sdata[0]);
    size_t sqsumStep = sqsum.step/sizeof(sqdata[0]);
    
    CV_SUM_PTRS( p[0], p[1], p[2], p[3], sdata, normrect, sumStep );
    CV_SUM_PTRS( pq[0], pq[1], pq[2], pq[3], sqdata, normrect, sqsumStep );
    
    size_t fi, nfeatures = features->size();

    for( fi = 0; fi < nfeatures; fi++ )
        featuresPtr[fi].updatePtrs( !featuresPtr[fi].tilted ? sum : tilted );
    return true;
}

bool  HaarEvaluator::setWindow( Point pt )
{
    if( pt.x < 0 || pt.y < 0 ||
        pt.x + origWinSize.width >= sum.cols ||
        pt.y + origWinSize.height >= sum.rows )
        return false;

    size_t pOffset = pt.y * (sum.step/sizeof(int)) + pt.x;
    size_t pqOffset = pt.y * (sqsum.step/sizeof(double)) + pt.x;
    int valsum = CALC_SUM(p, pOffset);
    double valsqsum = CALC_SUM(pq, pqOffset);

    double nf = (double)normrect.area() * valsqsum - (double)valsum * valsum;
    if( nf > 0. )
        nf = sqrt(nf);
    else
        nf = 1.;
    varianceNormFactor = 1./nf;
    offset = (int)pOffset;

    return true;
}

//----------------------------------------------  LBPEvaluator -------------------------------------
bool LBPEvaluator::Feature :: read(const FileNode& node )
{
    FileNode rnode = node[CC_RECT];
    FileNodeIterator it = rnode.begin();
    it >> rect.x >> rect.y >> rect.width >> rect.height;
    return true;
}

LBPEvaluator::LBPEvaluator()
{
    features = new vector<Feature>();
}
LBPEvaluator::~LBPEvaluator()
{
}

bool LBPEvaluator::read( const FileNode& node )
{
    features->resize(node.size());
    featuresPtr = &(*features)[0];
    FileNodeIterator it = node.begin(), it_end = node.end();
    for(int i = 0; it != it_end; ++it, i++)
    {
        if(!featuresPtr[i].read(*it))
            return false;
    }
    return true;
}

Ptr<FeatureEvaluator> LBPEvaluator::clone() const
{
    LBPEvaluator* ret = new LBPEvaluator;
    ret->origWinSize = origWinSize;
    ret->features = features;
    ret->featuresPtr = &(*ret->features)[0];
    ret->sum0 = sum0, ret->sum = sum;
    ret->normrect = normrect;
    ret->offset = offset;
    return ret;
}

bool LBPEvaluator::setImage( const Mat& image, Size _origWinSize )
{
    int rn = image.rows+1, cn = image.cols+1;
    origWinSize = _origWinSize;

    if( image.cols < origWinSize.width || image.rows < origWinSize.height )
        return false;
    
    if( sum0.rows < rn || sum0.cols < cn )
        sum0.create(rn, cn, CV_32S);
    sum = Mat(rn, cn, CV_32S, sum0.data);
    integral(image, sum);
    
    size_t fi, nfeatures = features->size();
    
    for( fi = 0; fi < nfeatures; fi++ )
        featuresPtr[fi].updatePtrs( sum );
    return true;
}
    
bool LBPEvaluator::setWindow( Point pt )
{
    if( pt.x < 0 || pt.y < 0 ||
        pt.x + origWinSize.width >= sum.cols ||
        pt.y + origWinSize.height >= sum.rows )
        return false;
    offset = pt.y * ((int)sum.step/sizeof(int)) + pt.x;
    return true;
}          

//----------------------------------------------  HOGEvaluator ---------------------------------------
bool HOGEvaluator::Feature :: read( const FileNode& node )
{
    FileNode rnode = node[CC_RECT];
    FileNodeIterator it = rnode.begin();
    it >> rect[0].x >> rect[0].y >> rect[0].width >> rect[0].height >> featComponent;
    rect[1].x = rect[0].x + rect[0].width;
    rect[1].y = rect[0].y;
    rect[2].x = rect[0].x;
    rect[2].y = rect[0].y + rect[0].height;
    rect[3].x = rect[0].x + rect[0].width;
    rect[3].y = rect[0].y + rect[0].height;
    rect[1].width = rect[2].width = rect[3].width = rect[0].width;
    rect[1].height = rect[2].height = rect[3].height = rect[0].height;
    return true;
}

HOGEvaluator::HOGEvaluator()
{
    features = new vector<Feature>();
}

HOGEvaluator::~HOGEvaluator()
{
}

bool HOGEvaluator::read( const FileNode& node )
{
    features->resize(node.size());
    featuresPtr = &(*features)[0];
    FileNodeIterator it = node.begin(), it_end = node.end();
    for(int i = 0; it != it_end; ++it, i++)
    {
        if(!featuresPtr[i].read(*it))
            return false;
    }
    return true;
}

Ptr<FeatureEvaluator> HOGEvaluator::clone() const
{
    HOGEvaluator* ret = new HOGEvaluator;
    ret->origWinSize = origWinSize;
    ret->features = features;
    ret->featuresPtr = &(*ret->features)[0];
    ret->offset = offset;
    ret->hist = hist;
    ret->normSum = normSum;    
    return ret;
}

bool HOGEvaluator::setImage( const Mat& image, Size winSize )
{
    int rows = image.rows + 1;
    int cols = image.cols + 1;
    origWinSize = winSize;
    if( image.cols < origWinSize.width || image.rows < origWinSize.height )
        return false;
    hist.clear();
    for( int bin = 0; bin < Feature::BIN_NUM; bin++ )
    {
        hist.push_back( Mat(rows, cols, CV_32FC1) );
    }
    normSum.create( rows, cols, CV_32FC1 );

    integralHistogram( image, hist, normSum, Feature::BIN_NUM );

    size_t featIdx, featCount = features->size();

    for( featIdx = 0; featIdx < featCount; featIdx++ )
    {
        featuresPtr[featIdx].updatePtrs( hist, normSum );
    }
    return true;
}

bool HOGEvaluator::setWindow(Point pt)
{
    if( pt.x < 0 || pt.y < 0 ||
        pt.x + origWinSize.width >= hist[0].cols-2 ||
        pt.y + origWinSize.height >= hist[0].rows-2 )
        return false;
    offset = pt.y * ((int)hist[0].step/sizeof(float)) + pt.x;
    return true;
}

void HOGEvaluator::integralHistogram(const Mat &img, vector<Mat> &histogram, Mat &norm, int nbins) const
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

Ptr<FeatureEvaluator> FeatureEvaluator::create( int featureType )
{
    return featureType == HAAR ? Ptr<FeatureEvaluator>(new HaarEvaluator) :
        featureType == LBP ? Ptr<FeatureEvaluator>(new LBPEvaluator) : 
        featureType == HOG ? Ptr<FeatureEvaluator>(new HOGEvaluator) :
        Ptr<FeatureEvaluator>();
}

//---------------------------------------- Classifier Cascade --------------------------------------------

CascadeClassifier::CascadeClassifier()
{
}

CascadeClassifier::CascadeClassifier(const string& filename)
{ 
    load(filename); 
}

CascadeClassifier::~CascadeClassifier()
{
}    

bool CascadeClassifier::empty() const
{
    return oldCascade.empty() && data.stages.empty();
}

bool CascadeClassifier::load(const string& filename)
{
    oldCascade.release();
    data = Data();
    featureEvaluator.release();
    
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    
    if( read(fs.getFirstTopLevelNode()) )
        return true;
    
    fs.release();
    
    oldCascade = Ptr<CvHaarClassifierCascade>((CvHaarClassifierCascade*)cvLoad(filename.c_str(), 0, 0, 0));
    return !oldCascade.empty();
}
    
int CascadeClassifier::runAt( Ptr<FeatureEvaluator>& featureEvaluator, Point pt, double& weight )
{
    CV_Assert( oldCascade.empty() );
        
    assert( data.featureType == FeatureEvaluator::HAAR ||
            data.featureType == FeatureEvaluator::LBP ||
            data.featureType == FeatureEvaluator::HOG );

    if( !featureEvaluator->setWindow(pt) )
        return -1;
    if( data.isStumpBased )
    {
        if( data.featureType == FeatureEvaluator::HAAR )
            return predictOrderedStump<HaarEvaluator>( *this, featureEvaluator, weight );
        else if( data.featureType == FeatureEvaluator::LBP )
            return predictCategoricalStump<LBPEvaluator>( *this, featureEvaluator, weight );
        else if( data.featureType == FeatureEvaluator::HOG )
            return predictOrderedStump<HOGEvaluator>( *this, featureEvaluator, weight );
        else
            return -2;
    }
    else
    {
        if( data.featureType == FeatureEvaluator::HAAR )
            return predictOrdered<HaarEvaluator>( *this, featureEvaluator, weight );
        else if( data.featureType == FeatureEvaluator::LBP )
            return predictCategorical<LBPEvaluator>( *this, featureEvaluator, weight );
        else if( data.featureType == FeatureEvaluator::HOG )
            return predictOrdered<HOGEvaluator>( *this, featureEvaluator, weight );
        else
            return -2;
    }
}
    
bool CascadeClassifier::setImage( Ptr<FeatureEvaluator>& featureEvaluator, const Mat& image )
{
    return empty() ? false : featureEvaluator->setImage(image, data.origWinSize);
}

void CascadeClassifier::setMaskGenerator(Ptr<MaskGenerator> _maskGenerator)
{
    maskGenerator=_maskGenerator;
}
Ptr<CascadeClassifier::MaskGenerator> CascadeClassifier::getMaskGenerator()
{
    return maskGenerator;
}

void CascadeClassifier::setFaceDetectionMaskGenerator()
{
#ifdef HAVE_TEGRA_OPTIMIZATION
    setMaskGenerator(tegra::getCascadeClassifierMaskGenerator(*this));
#else
    setMaskGenerator(Ptr<CascadeClassifier::MaskGenerator>());
#endif
}

struct CascadeClassifierInvoker
{
    CascadeClassifierInvoker( CascadeClassifier& _cc, Size _sz1, int _stripSize, int _yStep, double _factor, 
        ConcurrentRectVector& _vec, vector<int>& _levels, vector<double>& _weights, bool outputLevels, const Mat& _mask)
    {
        classifier = &_cc;
        processingRectSize = _sz1;
        stripSize = _stripSize;
        yStep = _yStep;
        scalingFactor = _factor;
        rectangles = &_vec;
        rejectLevels  = outputLevels ? &_levels : 0;
        levelWeights  = outputLevels ? &_weights : 0;
        mask=_mask;
    }
    
    void operator()(const BlockedRange& range) const
    {
        Ptr<FeatureEvaluator> evaluator = classifier->featureEvaluator->clone();

        Size winSize(cvRound(classifier->data.origWinSize.width * scalingFactor), cvRound(classifier->data.origWinSize.height * scalingFactor));

        int y1 = range.begin() * stripSize;
        int y2 = min(range.end() * stripSize, processingRectSize.height);
        for( int y = y1; y < y2; y += yStep )
        {
            for( int x = 0; x < processingRectSize.width; x += yStep )
            {
                if ( (!mask.empty()) && (mask.at<uchar>(Point(x,y))==0)) {
                    continue;
                }

                double gypWeight;
                int result = classifier->runAt(evaluator, Point(x, y), gypWeight);
                if( rejectLevels )
                {
                    if( result == 1 )
                        result =  -(int)classifier->data.stages.size();
                    if( classifier->data.stages.size() + result < 4 )
                    {
                        rectangles->push_back(Rect(cvRound(x*scalingFactor), cvRound(y*scalingFactor), winSize.width, winSize.height)); 
                        rejectLevels->push_back(-result);
                        levelWeights->push_back(gypWeight);
                    }
                }                    
                else if( result > 0 )
                    rectangles->push_back(Rect(cvRound(x*scalingFactor), cvRound(y*scalingFactor),
                                               winSize.width, winSize.height));
                if( result == 0 )
                    x += yStep;
            }
        }
    }
    
    CascadeClassifier* classifier;
    ConcurrentRectVector* rectangles;
    Size processingRectSize;
    int stripSize, yStep;
    double scalingFactor;
    vector<int> *rejectLevels;
    vector<double> *levelWeights;
    Mat mask;
};
    
struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };

bool CascadeClassifier::detectSingleScale( const Mat& image, int stripCount, Size processingRectSize,
                                           int stripSize, int yStep, double factor, vector<Rect>& candidates,
                                           vector<int>& levels, vector<double>& weights, bool outputRejectLevels )
{
    if( !featureEvaluator->setImage( image, data.origWinSize ) )
        return false;

    Mat currentMask;
    if (!maskGenerator.empty()) {
        currentMask=maskGenerator->generateMask(image);
    }

    ConcurrentRectVector concurrentCandidates;
    vector<int> rejectLevels;
    vector<double> levelWeights;
    if( outputRejectLevels )
    {
        parallel_for(BlockedRange(0, stripCount), CascadeClassifierInvoker( *this, processingRectSize, stripSize, yStep, factor,
            concurrentCandidates, rejectLevels, levelWeights, true, currentMask));
        levels.insert( levels.end(), rejectLevels.begin(), rejectLevels.end() );
        weights.insert( weights.end(), levelWeights.begin(), levelWeights.end() );
    }
    else
    {
         parallel_for(BlockedRange(0, stripCount), CascadeClassifierInvoker( *this, processingRectSize, stripSize, yStep, factor,
            concurrentCandidates, rejectLevels, levelWeights, false, currentMask));
    }
    candidates.insert( candidates.end(), concurrentCandidates.begin(), concurrentCandidates.end() );

    return true;
}

bool CascadeClassifier::isOldFormatCascade() const
{
    return !oldCascade.empty();
}


int CascadeClassifier::getFeatureType() const
{
    return featureEvaluator->getFeatureType();
}

Size CascadeClassifier::getOriginalWindowSize() const
{
    return data.origWinSize;
}

bool CascadeClassifier::setImage(const Mat& image)
{
    return featureEvaluator->setImage(image, data.origWinSize);
}

void CascadeClassifier::detectMultiScale( const Mat& image, vector<Rect>& objects, 
                                          vector<int>& rejectLevels,
                                          vector<double>& levelWeights,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize, 
                                          bool outputRejectLevels )
{
    const double GROUP_EPS = 0.2;
    
    CV_Assert( scaleFactor > 1 && image.depth() == CV_8U );
    
    if( empty() )
        return;

    if( isOldFormatCascade() )
    {
        MemStorage storage(cvCreateMemStorage(0));
        CvMat _image = image;
        CvSeq* _objects = cvHaarDetectObjectsForROC( &_image, oldCascade, storage, rejectLevels, levelWeights, scaleFactor,
                                              minNeighbors, flags, minObjectSize, maxObjectSize, outputRejectLevels );
        vector<CvAvgComp> vecAvgComp;
        Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
        objects.resize(vecAvgComp.size());
        std::transform(vecAvgComp.begin(), vecAvgComp.end(), objects.begin(), getRect());
        return;
    }

    objects.clear();

    if (!maskGenerator.empty()) {
        maskGenerator->initializeMask(image);
    }


    if( maxObjectSize.height == 0 || maxObjectSize.width == 0 )
        maxObjectSize = image.size();
    
    Mat grayImage = image;
    if( grayImage.channels() > 1 )
    {
        Mat temp;
        cvtColor(grayImage, temp, CV_BGR2GRAY);
        grayImage = temp;
    }
    
    Mat imageBuffer(image.rows + 1, image.cols + 1, CV_8U);
    vector<Rect> candidates;

    for( double factor = 1; ; factor *= scaleFactor )
    {
        Size originalWindowSize = getOriginalWindowSize();

        Size windowSize( cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor) );
        Size scaledImageSize( cvRound( grayImage.cols/factor ), cvRound( grayImage.rows/factor ) );
        Size processingRectSize( scaledImageSize.width - originalWindowSize.width + 1, scaledImageSize.height - originalWindowSize.height + 1 );
        
        if( processingRectSize.width <= 0 || processingRectSize.height <= 0 )
            break;
        if( windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height )
            break;
        if( windowSize.width < minObjectSize.width || windowSize.height < minObjectSize.height )
            continue;
        
        Mat scaledImage( scaledImageSize, CV_8U, imageBuffer.data );
        resize( grayImage, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR );

        int yStep;
        if( getFeatureType() == cv::FeatureEvaluator::HOG )
        {
            yStep = 4;
        }
        else
        {
            yStep = factor > 2. ? 1 : 2;
        }

        int stripCount, stripSize;

    #ifdef HAVE_TBB
        const int PTS_PER_THREAD = 1000;
        stripCount = ((processingRectSize.width/yStep)*(processingRectSize.height + yStep-1)/yStep + PTS_PER_THREAD/2)/PTS_PER_THREAD;
        stripCount = std::min(std::max(stripCount, 1), 100);
        stripSize = (((processingRectSize.height + stripCount - 1)/stripCount + yStep-1)/yStep)*yStep;
    #else
        stripCount = 1;
        stripSize = processingRectSize.height;
    #endif

        if( !detectSingleScale( scaledImage, stripCount, processingRectSize, stripSize, yStep, factor, candidates, 
            rejectLevels, levelWeights, outputRejectLevels ) )
            break;
    }

    
    objects.resize(candidates.size());
    std::copy(candidates.begin(), candidates.end(), objects.begin());

    if( outputRejectLevels )
    {
        groupRectangles( objects, rejectLevels, levelWeights, minNeighbors, GROUP_EPS );
    }
    else
    {
        groupRectangles( objects, minNeighbors, GROUP_EPS );
    }
}

void CascadeClassifier::detectMultiScale( const Mat& image, vector<Rect>& objects,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize)
{
    vector<int> fakeLevels;
    vector<double> fakeWeights;
    detectMultiScale( image, objects, fakeLevels, fakeWeights, scaleFactor, 
        minNeighbors, flags, minObjectSize, maxObjectSize, false );
}    

bool CascadeClassifier::Data::read(const FileNode &root)
{
    static const float THRESHOLD_EPS = 1e-5f;
    
    // load stage params
    string stageTypeStr = (string)root[CC_STAGE_TYPE];
    if( stageTypeStr == CC_BOOST )
        stageType = BOOST;
    else
        return false;

    string featureTypeStr = (string)root[CC_FEATURE_TYPE];
    if( featureTypeStr == CC_HAAR )
        featureType = FeatureEvaluator::HAAR;
    else if( featureTypeStr == CC_LBP )
        featureType = FeatureEvaluator::LBP;
    else if( featureTypeStr == CC_HOG )
        featureType = FeatureEvaluator::HOG;

    else
        return false;

    origWinSize.width = (int)root[CC_WIDTH];
    origWinSize.height = (int)root[CC_HEIGHT];
    CV_Assert( origWinSize.height > 0 && origWinSize.width > 0 );

    isStumpBased = (int)(root[CC_STAGE_PARAMS][CC_MAX_DEPTH]) == 1 ? true : false;

    // load feature params
    FileNode fn = root[CC_FEATURE_PARAMS];
    if( fn.empty() )
        return false;

    ncategories = fn[CC_MAX_CAT_COUNT];
    int subsetSize = (ncategories + 31)/32,
        nodeStep = 3 + ( ncategories>0 ? subsetSize : 1 );

    // load stages
    fn = root[CC_STAGES];
    if( fn.empty() )
        return false;

    stages.reserve(fn.size());
    classifiers.clear();
    nodes.clear();

    FileNodeIterator it = fn.begin(), it_end = fn.end();

    for( int si = 0; it != it_end; si++, ++it )
    {
        FileNode fns = *it;
        Stage stage;
        stage.threshold = (float)fns[CC_STAGE_THRESHOLD] - THRESHOLD_EPS;
        fns = fns[CC_WEAK_CLASSIFIERS];
        if(fns.empty())
            return false;
        stage.ntrees = (int)fns.size();
        stage.first = (int)classifiers.size();
        stages.push_back(stage);
        classifiers.reserve(stages[si].first + stages[si].ntrees);

        FileNodeIterator it1 = fns.begin(), it1_end = fns.end();
        for( ; it1 != it1_end; ++it1 ) // weak trees
        {
            FileNode fnw = *it1;
            FileNode internalNodes = fnw[CC_INTERNAL_NODES];
            FileNode leafValues = fnw[CC_LEAF_VALUES];
            if( internalNodes.empty() || leafValues.empty() )
                return false;

            DTree tree;
            tree.nodeCount = (int)internalNodes.size()/nodeStep;
            classifiers.push_back(tree);

            nodes.reserve(nodes.size() + tree.nodeCount);
            leaves.reserve(leaves.size() + leafValues.size());
            if( subsetSize > 0 )
                subsets.reserve(subsets.size() + tree.nodeCount*subsetSize);

            FileNodeIterator internalNodesIter = internalNodes.begin(), internalNodesEnd = internalNodes.end();

            for( ; internalNodesIter != internalNodesEnd; ) // nodes
            {
                DTreeNode node;
                node.left = (int)*internalNodesIter; ++internalNodesIter;
                node.right = (int)*internalNodesIter; ++internalNodesIter;
                node.featureIdx = (int)*internalNodesIter; ++internalNodesIter;
                if( subsetSize > 0 )
                {
                    for( int j = 0; j < subsetSize; j++, ++internalNodesIter )
                        subsets.push_back((int)*internalNodesIter);
                    node.threshold = 0.f;
                }
                else
                {
                    node.threshold = (float)*internalNodesIter; ++internalNodesIter;
                }
                nodes.push_back(node);
            }

            internalNodesIter = leafValues.begin(), internalNodesEnd = leafValues.end();

            for( ; internalNodesIter != internalNodesEnd; ++internalNodesIter ) // leaves
                leaves.push_back((float)*internalNodesIter);
        }
    }

    return true;
}

bool CascadeClassifier::read(const FileNode& root)
{
    if( !data.read(root) )
        return false;

    // load features
    featureEvaluator = FeatureEvaluator::create(data.featureType);
    FileNode fn = root[CC_FEATURES];
    if( fn.empty() )
        return false;
    
    return featureEvaluator->read(fn);
}
    
template<> void Ptr<CvHaarClassifierCascade>::delete_obj()
{ cvReleaseHaarClassifierCascade(&obj); }    

} // namespace cv
