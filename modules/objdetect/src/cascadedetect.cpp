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
// Copyright (C) 2008-2013, Itseez Inc., all rights reserved.
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
//   * The name of Itseez Inc. may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
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
#include "opencv2/objdetect/objdetect_c.h"
#include "opencl_kernels.hpp"

namespace cv
{

template<typename _Tp> void copyVectorToUMat(const std::vector<_Tp>& v, UMat& um)
{
    if(v.empty())
        um.release();
    Mat(1, (int)(v.size()*sizeof(v[0])), CV_8U, (void*)&v[0]).copyTo(um);
}

void groupRectangles(std::vector<Rect>& rectList, int groupThreshold, double eps,
                     std::vector<int>* weights, std::vector<double>* levelWeights)
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

    std::vector<int> labels;
    int nclasses = partition(rectList, labels, SimilarRects(eps));

    std::vector<Rect> rrects(nclasses);
    std::vector<int> rweights(nclasses, 0);
    std::vector<int> rejectLevels(nclasses, 0);
    std::vector<double> rejectWeights(nclasses, DBL_MIN);
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
        int n1 = rweights[i];
        double w1 = rejectWeights[i];
        int l1 = rejectLevels[i];

        // filter out rectangles which don't have enough similar rectangles
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
                weights->push_back(l1);
            if( levelWeights )
                levelWeights->push_back(w1);
        }
    }
}

class MeanshiftGrouping
{
public:
    MeanshiftGrouping(const Point3d& densKer, const std::vector<Point3d>& posV,
        const std::vector<double>& wV, double eps, int maxIter = 20)
    {
        densityKernel = densKer;
        weightsV = wV;
        positionsV = posV;
        positionsCount = (int)posV.size();
        meanshiftV.resize(positionsCount);
        distanceV.resize(positionsCount);
        iterMax = maxIter;
        modeEps = eps;

        for (unsigned i = 0; i<positionsV.size(); i++)
        {
            meanshiftV[i] = getNewValue(positionsV[i]);
            distanceV[i] = moveToMode(meanshiftV[i]);
            meanshiftV[i] -= positionsV[i];
        }
    }

    void getModes(std::vector<Point3d>& modesV, std::vector<double>& resWeightsV, const double eps)
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
    std::vector<Point3d> positionsV;
    std::vector<double> weightsV;

    Point3d densityKernel;
    int positionsCount;

    std::vector<Point3d> meanshiftV;
    std::vector<Point3d> distanceV;
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

            sPt.x *= std::exp(aPt.z);
            sPt.y *= std::exp(aPt.z);

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

            sPt.x *= std::exp(aPt.z);
            sPt.y *= std::exp(aPt.z);

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
        ns.x *= std::exp(p2.z);
        ns.y *= std::exp(p2.z);
        p2 -= p1;
        p2.x /= ns.x;
        p2.y /= ns.y;
        p2.z /= ns.z;
        return p2.dot(p2);
    }
};
//new grouping function with using meanshift
static void groupRectangles_meanshift(std::vector<Rect>& rectList, double detectThreshold, std::vector<double>* foundWeights,
                                      std::vector<double>& scales, Size winDetSize)
{
    int detectionCount = (int)rectList.size();
    std::vector<Point3d> hits(detectionCount), resultHits;
    std::vector<double> hitWeights(detectionCount), resultWeights;
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

        double scale = std::exp(resultHits[i].z);
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

void groupRectangles(std::vector<Rect>& rectList, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, 0, 0);
}

void groupRectangles(std::vector<Rect>& rectList, std::vector<int>& weights, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, &weights, 0);
}
//used for cascade detection algorithm for ROC-curve calculating
void groupRectangles(std::vector<Rect>& rectList, std::vector<int>& rejectLevels, std::vector<double>& levelWeights, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, &rejectLevels, &levelWeights);
}
//can be used for HOG detection algorithm only
void groupRectangles_meanshift(std::vector<Rect>& rectList, std::vector<double>& foundWeights,
                               std::vector<double>& foundScales, double detectThreshold, Size winDetSize)
{
    groupRectangles_meanshift(rectList, detectThreshold, &foundWeights, foundScales, winDetSize);
}


FeatureEvaluator::~FeatureEvaluator() {}
bool FeatureEvaluator::read(const FileNode&) {return true;}
Ptr<FeatureEvaluator> FeatureEvaluator::clone() const { return Ptr<FeatureEvaluator>(); }
int FeatureEvaluator::getFeatureType() const {return -1;}
bool FeatureEvaluator::setImage(InputArray, Size, const std::vector<float>&) {return true;}
bool FeatureEvaluator::setWindow(Point, int) { return true; }
const FeatureEvaluator::ScaleData& FeatureEvaluator::getScaleData(int) const
{
    CV_Error(CV_StsNotImplemented, "");
    static ScaleData dummy;
    return dummy;
}
void FeatureEvaluator::getUMats(std::vector<UMat>& umats)
{
    umats.clear();
}

float FeatureEvaluator::calcOrd(int) const { return 0.; }
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
    optfeaturesPtr = 0;
    pwin = 0;
}
HaarEvaluator::~HaarEvaluator()
{
}

bool HaarEvaluator::read(const FileNode& node)
{
    size_t i, n = node.size();
    CV_Assert(n > 0);
    if(features.empty())
        features = makePtr<std::vector<Feature> >();
    if(optfeatures.empty())
        optfeatures = makePtr<std::vector<OptFeature> >();
    if(scaleData.empty())
        scaleData = makePtr<std::vector<ScaleData> >();
    features->resize(n);
    FileNodeIterator it = node.begin();
    hasTiltedFeatures = false;
    std::vector<Feature>& ff = *features;
    sbufSize = Size();
    ufbuf.release();

    for(i = 0; i < n; i++, ++it)
    {
        if(!ff[i].read(*it))
            return false;
        if( ff[i].tilted )
            hasTiltedFeatures = true;
    }
    return true;
}

Ptr<FeatureEvaluator> HaarEvaluator::clone() const
{
    Ptr<HaarEvaluator> ret = makePtr<HaarEvaluator>();
    ret->origWinSize = origWinSize;
    ret->features = features;
    ret->optfeatures = optfeatures;
    ret->optfeaturesPtr = optfeaturesPtr;
    ret->scaleData = scaleData;
    ret->hasTiltedFeatures = hasTiltedFeatures;
    ret->sbuf = sbuf;
    ret->rbuf0 = rbuf0;
    ret->rbuf1 = rbuf1;
    ret->usbuf = usbuf;
    ret->ufbuf = ufbuf;
    ret->uscaleData = uscaleData;
    ret->nofs = nofs;
    ret->sqofs = sqofs;
    ret->normrect = normrect;
    ret->pwin = 0;
    ret->varianceNormFactor = 0;
    ret->sbufSize = sbufSize;
    return ret;
}


static bool updateScaleData( Size imgsz, const std::vector<float>& _scales,
                             std::vector<FeatureEvaluator::ScaleData>& scaleData,
                             Size& sbufsize )
{
    size_t i, nscales = _scales.size();
    bool recalcOptFeatures = nscales != scaleData.size();
    scaleData.resize(nscales);

    int layer_dy = 0;
    Point layer_ofs(0,0);
    Size prevBufSize = sbufsize;
    sbufsize.width = std::max(sbufsize.width, (int)alignSize(cvRound(imgsz.width/_scales[0]) + 31, 32));
    recalcOptFeatures = recalcOptFeatures || sbufsize.width != prevBufSize.width;

    for( i = 0; i < nscales; i++ )
    {
        FeatureEvaluator::ScaleData& s = scaleData[i];
        if( !recalcOptFeatures && fabs(s.scale - _scales[i]) > FLT_EPSILON*100*_scales[i] )
            recalcOptFeatures = true;
        float sc = _scales[i];
        Size sz;
        sz.width = cvRound(imgsz.width/sc);
        sz.height = cvRound(imgsz.height/sc);
        s.ystep = sc >= 2 ? 1 : 2;
        s.scale = sc;
        s.szi = Size(sz.width+1, sz.height+1);
        if( layer_ofs.x + s.szi.width > sbufsize.width )
        {
            layer_ofs = Point(0, layer_ofs.y + layer_dy);
            layer_dy = s.szi.height;
        }
        s.layer_ofs = layer_ofs.y*sbufsize.width + layer_ofs.x;
        layer_ofs.x += s.szi.width;
    }

    layer_ofs.y += layer_dy;
    sbufsize.height = std::max(sbufsize.height, layer_ofs.y);
    recalcOptFeatures = recalcOptFeatures || sbufsize.height != prevBufSize.height;
    return recalcOptFeatures;
}

bool HaarEvaluator::setImage( InputArray _image, Size _origWinSize,
                              const std::vector<float>& _scales )
{
    Mat image;

    if( _image.isUMat() )
        _image.copyTo(image);
    else
        image = _image.getMat();
    Size imgsz = image.size();

    bool recalcOptFeatures = updateScaleData( imgsz, _scales, *scaleData, sbufSize ) || _origWinSize != origWinSize;
    origWinSize = _origWinSize;
    sqofs = sbufSize.width*sbufSize.height;
    int tofs = hasTiltedFeatures ? sqofs*2 : 0;

    sbuf.create(sbufSize.height*(hasTiltedFeatures ? 3 : 2), sbufSize.width, CV_32S);
    size_t i, nscales = scaleData->size();
    Size sz0 = scaleData->at(0).szi;
    sz0 = Size(std::max(rbuf0.cols, (int)alignSize(sz0.width, 16)), std::max(rbuf0.rows, sz0.height));

    rbuf0.create(sz0, CV_8U);
    rbuf1.create(sz0, CV_8U);
    Mat prev = image;
    float prevscale = 1.f;

    for( i = 0; i < nscales; i++ )
    {
        const ScaleData& s = scaleData->at(i);
        Mat& dbuf = i%2 == 0 ? rbuf0 : rbuf1;
        Mat src = prev, dst(s.szi.height-1, s.szi.width-1, CV_8U, dbuf.data, dbuf.step);
        resize(src, dst, dst.size(), prevscale/s.scale, prevscale/s.scale, INTER_LINEAR);
        Mat sum(s.szi, CV_32S, sbuf.ptr<int>() + s.layer_ofs, sbuf.step);
        Mat sqsum(s.szi.height-1, s.szi.width-1, CV_32S, sum.ptr<int>() + sqofs, sbuf.step);

        if( hasTiltedFeatures )
        {
            Mat tilted(s.szi, CV_32S, sum.ptr<int>() + tofs, sbuf.step);
            integral(dst, sum, noArray(), tilted, CV_32S, CV_64F);
        }
        else
            integral(dst, sum, noArray(), noArray(), CV_32S, CV_64F);

        sqrBoxFilter(dst, sqsum, CV_32S,
                     Size(origWinSize.width-2, origWinSize.height-2),
                     Point(0,0), false);
        prev = dst;
        prevscale = s.scale;
    }

    int sstep = (int)(sbuf.step/sizeof(int));
    normrect = Rect(1, 1, origWinSize.width-2, origWinSize.height-2);
    CV_SUM_OFS( nofs[0], nofs[1], nofs[2], nofs[3], 0, normrect, sstep );

    if( recalcOptFeatures )
    {
        size_t fi, nfeatures = features->size();
        const std::vector<Feature>& ff = *features;
        optfeatures->resize(nfeatures);
        optfeaturesPtr = &(*optfeatures)[0];
        for( fi = 0; fi < nfeatures; fi++ )
            optfeaturesPtr[fi].setOffsets( ff[fi], sstep, tofs );
    }

    if(recalcOptFeatures || ufbuf.empty())
    {
        copyVectorToUMat(*optfeatures, ufbuf);
        copyVectorToUMat(*scaleData, uscaleData);
    }

    return true;
}

const FeatureEvaluator::ScaleData& HaarEvaluator::getScaleData( int scaleIdx ) const
{
    CV_Assert(0 <= scaleIdx && scaleIdx < (int)scaleData->size());
    return scaleData->at(scaleIdx);
}

bool HaarEvaluator::setWindow( Point pt, int scaleIdx )
{
    CV_Assert(0 <= scaleIdx && scaleIdx < (int)scaleData->size());
    const ScaleData& s = scaleData->at(scaleIdx);

    if( pt.x < 0 || pt.y < 0 ||
        pt.x + origWinSize.width > s.szi.width ||
        pt.y + origWinSize.height > s.szi.height )
        return false;

    pwin = &sbuf.at<int>(pt) + s.layer_ofs;
    const int* pq = pwin + sqofs;
    int valsum = CALC_SUM_OFS(nofs, pwin);
    int valsqsum = pq[nofs[0]];

    double nf = (double)normrect.area() * valsqsum - (double)valsum * valsum;
    if( nf > 0. )
        nf = std::sqrt(nf);
    else
        nf = 1.;
    varianceNormFactor = (1./nf);

    return true;
}


void HaarEvaluator::OptFeature::setOffsets( const Feature& _f, int step, int tofs )
{
    weight[0] = _f.rect[0].weight;
    weight[1] = _f.rect[1].weight;
    weight[2] = _f.rect[2].weight;

    if( _f.tilted )
    {
        CV_TILTED_OFS( ofs[0][0], ofs[0][1], ofs[0][2], ofs[0][3], tofs, _f.rect[0].r, step );
        CV_TILTED_OFS( ofs[1][0], ofs[1][1], ofs[1][2], ofs[1][3], tofs, _f.rect[1].r, step );
        CV_TILTED_OFS( ofs[2][0], ofs[2][1], ofs[2][2], ofs[2][3], tofs, _f.rect[2].r, step );
    }
    else
    {
        CV_SUM_OFS( ofs[0][0], ofs[0][1], ofs[0][2], ofs[0][3], 0, _f.rect[0].r, step );
        CV_SUM_OFS( ofs[1][0], ofs[1][1], ofs[1][2], ofs[1][3], 0, _f.rect[1].r, step );
        CV_SUM_OFS( ofs[2][0], ofs[2][1], ofs[2][2], ofs[2][3], 0, _f.rect[2].r, step );
    }
}

Rect HaarEvaluator::getNormRect() const
{
    return normrect;
}

int HaarEvaluator::getSquaresOffset() const
{
    return sqofs;
}

void HaarEvaluator::getUMats(std::vector<UMat>& bufs)
{
    sbuf.copyTo(usbuf);

    bufs.clear();
    bufs.push_back(uscaleData);
    bufs.push_back(usbuf);
    bufs.push_back(ufbuf);
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
    features = makePtr<std::vector<Feature> >();
    optfeatures = makePtr<std::vector<OptFeature> >();
    scaleData = makePtr<std::vector<ScaleData> >();
}

LBPEvaluator::~LBPEvaluator()
{
}

bool LBPEvaluator::read( const FileNode& node )
{
    features->resize(node.size());
    optfeaturesPtr = 0;
    FileNodeIterator it = node.begin(), it_end = node.end();
    std::vector<Feature>& ff = *features;
    for(int i = 0; it != it_end; ++it, i++)
    {
        if(!ff[i].read(*it))
            return false;
    }
    return true;
}

Ptr<FeatureEvaluator> LBPEvaluator::clone() const
{
    Ptr<LBPEvaluator> ret = makePtr<LBPEvaluator>();
    ret->origWinSize = origWinSize;
    ret->features = features;
    ret->optfeatures = optfeatures;
    ret->scaleData = scaleData;
    ret->optfeaturesPtr = optfeaturesPtr;
    ret->sbuf = sbuf;
    ret->usbuf = usbuf;
    ret->ufbuf = ufbuf;
    ret->uscaleData = uscaleData;
    ret->pwin = pwin;
    ret->sbufSize = sbufSize;
    return ret;
}

bool LBPEvaluator::setImage( InputArray _image, Size _origWinSize, const std::vector<float>& _scales )
{
    Mat image;

    if( _image.isUMat() )
        _image.copyTo(image);
    else
        image = _image.getMat();
    Size imgsz = image.size();

    bool recalcOptFeatures = updateScaleData( imgsz, _scales, *scaleData, sbufSize ) || _origWinSize != origWinSize;
    origWinSize = _origWinSize;

    sbuf.create(sbufSize.height, sbufSize.width, CV_32S);
    size_t i, nscales = scaleData->size();
    Size sz0 = scaleData->at(0).szi;
    sz0 = Size(std::max(rbuf0.cols, (int)alignSize(sz0.width, 16)), std::max(rbuf0.rows, sz0.height));

    rbuf0.create(sz0, CV_8U);
    rbuf1.create(sz0, CV_8U);
    Mat prev = image;
    float prevscale = 1.f;

    for( i = 0; i < nscales; i++ )
    {
        const ScaleData& s = scaleData->at(i);
        Mat& dbuf = i%2 == 0 ? rbuf0 : rbuf1;
        Mat src = prev, dst(s.szi.height-1, s.szi.width-1, CV_8U, dbuf.data, dbuf.step);
        resize(src, dst, dst.size(), prevscale/s.scale, prevscale/s.scale, INTER_LINEAR);
        Mat sum(s.szi, CV_32S, sbuf.ptr<int>() + s.layer_ofs, sbuf.step);

        integral(dst, sum, noArray(), noArray(), CV_32S);

        prev = dst;
        prevscale = s.scale;
    }

    if( recalcOptFeatures )
    {
        size_t fi, nfeatures = features->size();
        const std::vector<Feature>& ff = *features;
        optfeatures->resize(nfeatures);
        optfeaturesPtr = &(*optfeatures)[0];
        int sstep = (int)(sbuf.step/sizeof(int));

        for( fi = 0; fi < nfeatures; fi++ )
            optfeaturesPtr[fi].setOffsets( ff[fi], sstep );
    }

    if(recalcOptFeatures || ufbuf.empty())
    {
        copyVectorToUMat(*optfeatures, ufbuf);
        copyVectorToUMat(*scaleData, uscaleData);
    }
    
    return true;
}

void LBPEvaluator::OptFeature::setOffsets( const Feature& _f, int step )
{
    Rect tr = _f.rect;
    int w0 = tr.width;
    int h0 = tr.height;

    CV_SUM_OFS( ofs[0], ofs[1], ofs[4], ofs[5], 0, tr, step );
    tr.x += 2*w0;
    CV_SUM_OFS( ofs[2], ofs[3], ofs[6], ofs[7], 0, tr, step );
    tr.y += 2*h0;
    CV_SUM_OFS( ofs[10], ofs[11], ofs[14], ofs[15], 0, tr, step );
    tr.x -= 2*w0;
    CV_SUM_OFS( ofs[8], ofs[9], ofs[12], ofs[13], 0, tr, step );
}


bool LBPEvaluator::setWindow( Point pt, int scaleIdx )
{
    CV_Assert(0 <= scaleIdx && scaleIdx < (int)scaleData->size());
    const ScaleData& s = scaleData->at(scaleIdx);

    if( pt.x < 0 || pt.y < 0 ||
        pt.x + origWinSize.width > s.szi.width ||
        pt.y + origWinSize.height > s.szi.height )
        return false;

    pwin = &sbuf.at<int>(pt) + s.layer_ofs;
    return true;
}

const FeatureEvaluator::ScaleData& LBPEvaluator::getScaleData( int scaleIdx ) const
{
    CV_Assert(0 <= scaleIdx && scaleIdx < (int)scaleData->size());
    return scaleData->at(scaleIdx);
}

void LBPEvaluator::getUMats(std::vector<UMat>& bufs)
{
    sbuf.copyTo(usbuf);

    bufs.clear();
    bufs.push_back(uscaleData);
    bufs.push_back(usbuf);
    bufs.push_back(ufbuf);
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
    features = makePtr<std::vector<Feature> >();
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
    Ptr<HOGEvaluator> ret = makePtr<HOGEvaluator>();
    ret->origWinSize = origWinSize;
    ret->features = features;
    ret->featuresPtr = &(*ret->features)[0];
    ret->offset = offset;
    ret->hist = hist;
    ret->normSum = normSum;
    return ret;
}

bool HOGEvaluator::setImage( InputArray _image, Size winSize, const std::vector<float>& _scales )
{
    Mat image = _image.getMat();
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

bool HOGEvaluator::setWindow(Point pt, int scaleIdx)
{
    if( pt.x < 0 || pt.y < 0 ||
        pt.x + origWinSize.width >= hist[0].cols-2 ||
        pt.y + origWinSize.height >= hist[0].rows-2 )
        return false;
    offset = pt.y * ((int)hist[0].step/sizeof(float)) + pt.x;
    return true;
}

void HOGEvaluator::integralHistogram(const Mat &img, std::vector<Mat> &histogram, Mat &norm, int nbins) const
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

CascadeClassifierImpl::CascadeClassifierImpl()
{
}

CascadeClassifierImpl::~CascadeClassifierImpl()
{
}

bool CascadeClassifierImpl::empty() const
{
    return !oldCascade && data.stages.empty();
}

bool CascadeClassifierImpl::load(const String& filename)
{
    oldCascade.release();
    data = Data();
    featureEvaluator.release();

    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;

    if( read_(fs.getFirstTopLevelNode()) )
        return true;

    fs.release();

    oldCascade.reset((CvHaarClassifierCascade*)cvLoad(filename.c_str(), 0, 0, 0));
    return !oldCascade.empty();
}

void CascadeClassifierImpl::read(const FileNode& node)
{
    read_(node);
}

int CascadeClassifierImpl::runAt( Ptr<FeatureEvaluator>& evaluator, Point pt, int scaleIdx, double& weight )
{
    assert( !oldCascade &&
            (data.featureType == FeatureEvaluator::HAAR ||
            data.featureType == FeatureEvaluator::LBP ||
            data.featureType == FeatureEvaluator::HOG) );

    if( !evaluator->setWindow(pt, scaleIdx) )
        return -1;
    if( data.isStumpBased() )
    {
        if( data.featureType == FeatureEvaluator::HAAR )
            return predictOrderedStump<HaarEvaluator>( *this, evaluator, weight );
        else if( data.featureType == FeatureEvaluator::LBP )
            return predictCategoricalStump<LBPEvaluator>( *this, evaluator, weight );
        else if( data.featureType == FeatureEvaluator::HOG )
            return predictOrderedStump<HOGEvaluator>( *this, evaluator, weight );
        else
            return -2;
    }
    else
    {
        if( data.featureType == FeatureEvaluator::HAAR )
            return predictOrdered<HaarEvaluator>( *this, evaluator, weight );
        else if( data.featureType == FeatureEvaluator::LBP )
            return predictCategorical<LBPEvaluator>( *this, evaluator, weight );
        else if( data.featureType == FeatureEvaluator::HOG )
            return predictOrdered<HOGEvaluator>( *this, evaluator, weight );
        else
            return -2;
    }
}

void CascadeClassifierImpl::setMaskGenerator(const Ptr<MaskGenerator>& _maskGenerator)
{
    maskGenerator=_maskGenerator;
}
Ptr<CascadeClassifierImpl::MaskGenerator> CascadeClassifierImpl::getMaskGenerator()
{
    return maskGenerator;
}

Ptr<BaseCascadeClassifier::MaskGenerator> createFaceDetectionMaskGenerator()
{
#ifdef HAVE_TEGRA_OPTIMIZATION
    return tegra::getCascadeClassifierMaskGenerator(*this);
#else
    return Ptr<BaseCascadeClassifier::MaskGenerator>();
#endif
}

class CascadeClassifierInvoker : public ParallelLoopBody
{
public:
    CascadeClassifierInvoker( CascadeClassifierImpl& _cc, int _nscales, int _nstripes,
                              const FeatureEvaluator::ScaleData* _scaleData,
                              const int* _stripeSizes, std::vector<Rect>& _vec,
                              std::vector<int>& _levels, std::vector<double>& _weights,
                              bool outputLevels, const Mat& _mask, Mutex* _mtx)
    {
        classifier = &_cc;
        nscales = _nscales;
        nstripes = _nstripes;
        scaleData = _scaleData;
        stripeSizes = _stripeSizes;
        rectangles = &_vec;
        rejectLevels = outputLevels ? &_levels : 0;
        levelWeights = outputLevels ? &_weights : 0;
        mask = _mask;
        mtx = _mtx;
    }

    void operator()(const Range& range) const
    {
        Ptr<FeatureEvaluator> evaluator = classifier->featureEvaluator->clone();
        double gypWeight = 0.;
        Size origWinSize = classifier->data.origWinSize;

        for( int scaleIdx = 0; scaleIdx < nscales; scaleIdx++ )
        {
            const FeatureEvaluator::ScaleData& s = scaleData[scaleIdx];
            float scalingFactor = s.scale;
            int yStep = s.ystep;
            int stripeSize = stripeSizes[scaleIdx];
            int y0 = range.start*stripeSize;
            Size szw = s.getWorkingSize(origWinSize);
            int y1 = std::min(range.end*stripeSize, szw.height);
            Size winSize(cvRound(origWinSize.width * scalingFactor),
                         cvRound(origWinSize.height * scalingFactor));

            for( int y = y0; y < y1; y += yStep )
            {
                for( int x = 0; x < szw.width; x += yStep )
                {
                    int result = classifier->runAt(evaluator, Point(x, y), scaleIdx, gypWeight);
                    if( rejectLevels )
                    {
                        if( result == 1 )
                            result = -(int)classifier->data.stages.size();
                        if( classifier->data.stages.size() + result == 0 )
                        {
                            mtx->lock();
                            rectangles->push_back(Rect(cvRound(x*scalingFactor),
                                                       cvRound(y*scalingFactor),
                                                       winSize.width, winSize.height));
                            rejectLevels->push_back(-result);
                            levelWeights->push_back(gypWeight);
                            mtx->unlock();
                        }
                    }
                    else if( result > 0 )
                    {
                        mtx->lock();
                        rectangles->push_back(Rect(cvRound(x*scalingFactor),
                                                   cvRound(y*scalingFactor),
                                                   winSize.width, winSize.height));
                        mtx->unlock();
                    }
                    if( result == 0 )
                        x += yStep;
                }
            }
        }
    }

    CascadeClassifierImpl* classifier;
    std::vector<Rect>* rectangles;
    int nscales, nstripes;
    const FeatureEvaluator::ScaleData* scaleData;
    const int* stripeSizes;
    std::vector<int> *rejectLevels;
    std::vector<double> *levelWeights;
    std::vector<float> scales;
    Mat mask;
    Mutex* mtx;
};

struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };
struct getNeighbors { int operator ()(const CvAvgComp& e) const { return e.neighbors; } };


bool CascadeClassifierImpl::ocl_detectMultiScaleNoGrouping( const std::vector<float>& scales,
                                                            std::vector<Rect>& candidates )
{
    int featureType = getFeatureType();
    std::vector<UMat> bufs;
    featureEvaluator->getUMats(bufs);
    size_t localsize[] = { 8, 8 };
    const int grp_per_CU = 12;
    size_t globalsize[] = { grp_per_CU*ocl::Device::getDefault().maxComputeUnits()*localsize[0], localsize[1] };
    bool ok = false;

    ufacepos.create(1, MAX_FACES*4+1, CV_32S);
    UMat ufacepos_count(ufacepos, Rect(0, 0, 1, 1));
    ufacepos_count.setTo(Scalar::all(0));

    if( ustages.empty() )
    {
        copyVectorToUMat(data.stages, ustages);
        copyVectorToUMat(data.stumps, ustumps);
        if( !data.subsets.empty() )
            copyVectorToUMat(data.subsets, usubsets);
    }

    if( featureType == FeatureEvaluator::HAAR )
    {
        Ptr<HaarEvaluator> haar = featureEvaluator.dynamicCast<HaarEvaluator>();
        if( haar.empty() )
            return false;

        if( haarKernel.empty() )
        {
            haarKernel.create("runHaarClassifierStump", ocl::objdetect::cascadedetect_oclsrc, "");
            if( haarKernel.empty() )
                return false;
        }

        Rect normrect = haar->getNormRect();
        int sqofs = haar->getSquaresOffset();

        haarKernel.args((int)scales.size(),
                        ocl::KernelArg::PtrReadOnly(bufs[0]), // scaleData
                        ocl::KernelArg::ReadOnlyNoSize(bufs[1]), // sum
                        ocl::KernelArg::PtrReadOnly(bufs[2]), // optfeatures

                        // cascade classifier
                        (int)data.stages.size(),
                        ocl::KernelArg::PtrReadOnly(ustages),
                        ocl::KernelArg::PtrReadOnly(ustumps),

                        ocl::KernelArg::PtrWriteOnly(ufacepos), // positions
                        normrect, sqofs, data.origWinSize, (int)MAX_FACES);
        ok = haarKernel.run(2, globalsize, localsize, true);
    }
    else if( featureType == FeatureEvaluator::LBP )
    {
        Ptr<LBPEvaluator> lbp = featureEvaluator.dynamicCast<LBPEvaluator>();
        if( lbp.empty() )
            return false;

        if( lbpKernel.empty() )
        {
            lbpKernel.create("runLBPClassifierStump", ocl::objdetect::cascadedetect_oclsrc, "");
            if( lbpKernel.empty() )
                return false;
        }

        int subsetSize = (data.ncategories + 31)/32;
        lbpKernel.args((int)scales.size(),
                       ocl::KernelArg::PtrReadOnly(bufs[0]), // scaleData
                       ocl::KernelArg::ReadOnlyNoSize(bufs[1]), // sum
                       ocl::KernelArg::PtrReadOnly(bufs[2]), // optfeatures

                       // cascade classifier
                       (int)data.stages.size(),
                       ocl::KernelArg::PtrReadOnly(ustages),
                       ocl::KernelArg::PtrReadOnly(ustumps),
                       ocl::KernelArg::PtrReadOnly(usubsets),
                       subsetSize,

                       ocl::KernelArg::PtrWriteOnly(ufacepos), // positions
                       data.origWinSize, (int)MAX_FACES);
        ok = lbpKernel.run(2, globalsize, localsize, true);
    }

    if( ok )
    {
        Mat facepos = ufacepos.getMat(ACCESS_READ);
        const int* fptr = facepos.ptr<int>();
        int i, nfaces = fptr[0];
        for( i = 0; i < nfaces; i++ )
        {
            candidates.push_back(Rect(fptr[i*4+1], fptr[i*4+2], fptr[i*4+3], fptr[i*4+4]));
        }
    }
    return ok;
}

bool CascadeClassifierImpl::isOldFormatCascade() const
{
    return !oldCascade.empty();
}

int CascadeClassifierImpl::getFeatureType() const
{
    return featureEvaluator->getFeatureType();
}

Size CascadeClassifierImpl::getOriginalWindowSize() const
{
    return data.origWinSize;
}

void* CascadeClassifierImpl::getOldCascade()
{
    return oldCascade;
}

static void detectMultiScaleOldFormat( const Mat& image, Ptr<CvHaarClassifierCascade> oldCascade,
                                       std::vector<Rect>& objects,
                                       std::vector<int>& rejectLevels,
                                       std::vector<double>& levelWeights,
                                       std::vector<CvAvgComp>& vecAvgComp,
                                       double scaleFactor, int minNeighbors,
                                       int flags, Size minObjectSize, Size maxObjectSize,
                                       bool outputRejectLevels = false )
{
    MemStorage storage(cvCreateMemStorage(0));
    CvMat _image = image;
    CvSeq* _objects = cvHaarDetectObjectsForROC( &_image, oldCascade, storage, rejectLevels, levelWeights, scaleFactor,
                                                 minNeighbors, flags, minObjectSize, maxObjectSize, outputRejectLevels );
    Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
    objects.resize(vecAvgComp.size());
    std::transform(vecAvgComp.begin(), vecAvgComp.end(), objects.begin(), getRect());
}


void CascadeClassifierImpl::detectMultiScaleNoGrouping( InputArray _image, std::vector<Rect>& candidates,
                                                    std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
                                                    double scaleFactor, Size minObjectSize, Size maxObjectSize,
                                                    bool outputRejectLevels )
{
    int featureType = getFeatureType();
    Size imgsz = _image.size();
    int imgtype = _image.type();

    Mat grayImage;

    candidates.clear();
    rejectLevels.clear();
    levelWeights.clear();

    if( maxObjectSize.height == 0 || maxObjectSize.width == 0 )
        maxObjectSize = imgsz;

    bool use_ocl = tryOpenCL && ocl::useOpenCL() &&
        (featureType == FeatureEvaluator::HAAR ||
         featureType == FeatureEvaluator::LBP) &&
        ocl::Device::getDefault().type() != ocl::Device::TYPE_CPU &&
        !isOldFormatCascade() &&
        data.isStumpBased() &&
        maskGenerator.empty() &&
        !outputRejectLevels;

    Mat image = _image.getMat();

    grayImage = image;
    if( CV_MAT_CN(imgtype) > 1 )
    {
        Mat temp;
        cvtColor(grayImage, temp, COLOR_BGR2GRAY);
        grayImage = temp;
    }

    std::vector<float> scales;
    scales.reserve(1024);

    for( double factor = 1; ; factor *= scaleFactor )
    {
        Size originalWindowSize = getOriginalWindowSize();

        Size windowSize( cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor) );
        if( windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height ||
            windowSize.width > imgsz.width || windowSize.height > imgsz.height )
            break;
        if( windowSize.width < minObjectSize.width || windowSize.height < minObjectSize.height )
            continue;
        scales.push_back((float)factor);
    }

    if( !featureEvaluator->setImage(grayImage, data.origWinSize, scales) )
        return;

    // OpenCL code
    if( use_ocl )
    {
        if( ocl_detectMultiScaleNoGrouping( scales, candidates ))
            return;
        tryOpenCL = false;
    }

    // CPU code
    {
        Mat currentMask;
        if (maskGenerator)
            currentMask = maskGenerator->generateMask(grayImage);

        size_t i, nscales = scales.size();
        cv::AutoBuffer<int> stripeSizeBuf(nscales);
        int* stripeSizes = stripeSizeBuf;
        const FeatureEvaluator::ScaleData* s = &featureEvaluator->getScaleData(0);
        Size szw = s->getWorkingSize(data.origWinSize);
        int nstripes = cvCeil(szw.width/32.);
        for( i = 0; i < nscales; i++ )
        {
            szw = s[i].getWorkingSize(data.origWinSize);
            stripeSizes[i] = std::max((szw.height/s[i].ystep + nstripes-1)/nstripes, 1)*s[i].ystep;
        }

        CascadeClassifierInvoker invoker(*this, (int)nscales, nstripes, s, stripeSizes,
                                         candidates, rejectLevels, levelWeights,
                                         outputRejectLevels, currentMask, &mtx);
        parallel_for_(Range(0, nstripes), invoker);
    }
}


void CascadeClassifierImpl::detectMultiScale( InputArray _image, std::vector<Rect>& objects,
                                          std::vector<int>& rejectLevels,
                                          std::vector<double>& levelWeights,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize,
                                          bool outputRejectLevels )
{
    CV_Assert( scaleFactor > 1 && _image.depth() == CV_8U );

    if( empty() )
        return;

    if( isOldFormatCascade() )
    {
        Mat image = _image.getMat();
        std::vector<CvAvgComp> fakeVecAvgComp;
        detectMultiScaleOldFormat( image, oldCascade, objects, rejectLevels, levelWeights, fakeVecAvgComp, scaleFactor,
                                   minNeighbors, flags, minObjectSize, maxObjectSize, outputRejectLevels );
    }
    else
    {
        detectMultiScaleNoGrouping( _image, objects, rejectLevels, levelWeights, scaleFactor, minObjectSize, maxObjectSize,
                                    outputRejectLevels );
        const double GROUP_EPS = 0.2;
        if( outputRejectLevels )
        {
            groupRectangles( objects, rejectLevels, levelWeights, minNeighbors, GROUP_EPS );
        }
        else
        {
            groupRectangles( objects, minNeighbors, GROUP_EPS );
        }
    }
}

void CascadeClassifierImpl::detectMultiScale( InputArray _image, std::vector<Rect>& objects,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize)
{
    Mat image = _image.getMat();
    std::vector<int> fakeLevels;
    std::vector<double> fakeWeights;
    detectMultiScale( image, objects, fakeLevels, fakeWeights, scaleFactor,
        minNeighbors, flags, minObjectSize, maxObjectSize );
}

void CascadeClassifierImpl::detectMultiScale( InputArray _image, std::vector<Rect>& objects,
                                          std::vector<int>& numDetections, double scaleFactor,
                                          int minNeighbors, int flags, Size minObjectSize,
                                          Size maxObjectSize )
{
    Mat image = _image.getMat();
    CV_Assert( scaleFactor > 1 && image.depth() == CV_8U );

    if( empty() )
        return;

    std::vector<int> fakeLevels;
    std::vector<double> fakeWeights;
    if( isOldFormatCascade() )
    {
        std::vector<CvAvgComp> vecAvgComp;
        detectMultiScaleOldFormat( image, oldCascade, objects, fakeLevels, fakeWeights, vecAvgComp, scaleFactor,
                                   minNeighbors, flags, minObjectSize, maxObjectSize );
        numDetections.resize(vecAvgComp.size());
        std::transform(vecAvgComp.begin(), vecAvgComp.end(), numDetections.begin(), getNeighbors());
    }
    else
    {
        detectMultiScaleNoGrouping( image, objects, fakeLevels, fakeWeights, scaleFactor, minObjectSize, maxObjectSize );
        const double GROUP_EPS = 0.2;
        groupRectangles( objects, numDetections, minNeighbors, GROUP_EPS );
    }
}


CascadeClassifierImpl::Data::Data()
{
    stageType = featureType = ncategories = maxNodesPerTree = 0;
}

bool CascadeClassifierImpl::Data::read(const FileNode &root)
{
    static const float THRESHOLD_EPS = 1e-5f;

    // load stage params
    String stageTypeStr = (String)root[CC_STAGE_TYPE];
    if( stageTypeStr == CC_BOOST )
        stageType = BOOST;
    else
        return false;

    String featureTypeStr = (String)root[CC_FEATURE_TYPE];
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
    stumps.clear();

    FileNodeIterator it = fn.begin(), it_end = fn.end();
    maxNodesPerTree = 0;

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
            maxNodesPerTree = std::max(maxNodesPerTree, tree.nodeCount);

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

    if( isStumpBased() )
    {
        int nodeOfs = 0, leafOfs = 0;
        size_t nstages = stages.size();
        for( size_t stageIdx = 0; stageIdx < nstages; stageIdx++ )
        {
            const Stage& stage = stages[stageIdx];

            int ntrees = stage.ntrees;
            for( int i = 0; i < ntrees; i++, nodeOfs++, leafOfs+= 2 )
            {
                const DTreeNode& node = nodes[nodeOfs];
                stumps.push_back(Stump(node.featureIdx, node.threshold,
                                       leaves[leafOfs], leaves[leafOfs+1]));
            }
        }
    }

    return true;
}


bool CascadeClassifierImpl::read_(const FileNode& root)
{
    tryOpenCL = true;
    haarKernel = ocl::Kernel();
    lbpKernel = ocl::Kernel();
    ustages.release();
    ustumps.release();
    if( !data.read(root) )
        return false;

    // load features
    featureEvaluator = FeatureEvaluator::create(data.featureType);
    FileNode fn = root[CC_FEATURES];
    if( fn.empty() )
        return false;

    return featureEvaluator->read(fn);
}

template<> void DefaultDeleter<CvHaarClassifierCascade>::operator ()(CvHaarClassifierCascade* obj) const
{ cvReleaseHaarClassifierCascade(&obj); }


BaseCascadeClassifier::~BaseCascadeClassifier()
{
}

CascadeClassifier::CascadeClassifier() {}
CascadeClassifier::CascadeClassifier(const String& filename)
{
    load(filename);
}

CascadeClassifier::~CascadeClassifier()
{
}

bool CascadeClassifier::empty() const
{
    return cc.empty() || cc->empty();
}

bool CascadeClassifier::load( const String& filename )
{
    cc = makePtr<CascadeClassifierImpl>();
    if(!cc->load(filename))
        cc.release();
    return !empty();
}

bool CascadeClassifier::read(const FileNode &root)
{
    Ptr<CascadeClassifierImpl> ccimpl;
    bool ok = ccimpl->read_(root);
    if( ok )
        cc = ccimpl.staticCast<BaseCascadeClassifier>();
    else
        cc.release();
    return ok;
}

void CascadeClassifier::detectMultiScale( InputArray image,
                      CV_OUT std::vector<Rect>& objects,
                      double scaleFactor,
                      int minNeighbors, int flags,
                      Size minSize,
                      Size maxSize )
{
    CV_Assert(!empty());
    cc->detectMultiScale(image, objects, scaleFactor, minNeighbors, flags, minSize, maxSize);
}

void CascadeClassifier::detectMultiScale( InputArray image,
                      CV_OUT std::vector<Rect>& objects,
                      CV_OUT std::vector<int>& numDetections,
                      double scaleFactor,
                      int minNeighbors, int flags,
                      Size minSize, Size maxSize )
{
    CV_Assert(!empty());
    cc->detectMultiScale(image, objects, numDetections,
                         scaleFactor, minNeighbors, flags, minSize, maxSize);
}

void CascadeClassifier::detectMultiScale( InputArray image,
                      CV_OUT std::vector<Rect>& objects,
                      CV_OUT std::vector<int>& rejectLevels,
                      CV_OUT std::vector<double>& levelWeights,
                      double scaleFactor,
                      int minNeighbors, int flags,
                      Size minSize, Size maxSize,
                      bool outputRejectLevels )
{
    CV_Assert(!empty());
    cc->detectMultiScale(image, objects, rejectLevels, levelWeights,
                         scaleFactor, minNeighbors, flags,
                         minSize, maxSize, outputRejectLevels);
}

bool CascadeClassifier::isOldFormatCascade() const
{
    CV_Assert(!empty());
    return cc->isOldFormatCascade();
}

Size CascadeClassifier::getOriginalWindowSize() const
{
    CV_Assert(!empty());
    return cc->getOriginalWindowSize();
}

int CascadeClassifier::getFeatureType() const
{
    CV_Assert(!empty());
    return cc->getFeatureType();
}

void* CascadeClassifier::getOldCascade()
{
    CV_Assert(!empty());
    return cc->getOldCascade();
}

void CascadeClassifier::setMaskGenerator(const Ptr<BaseCascadeClassifier::MaskGenerator>& maskGenerator)
{
    CV_Assert(!empty());
    cc->setMaskGenerator(maskGenerator);
}

Ptr<BaseCascadeClassifier::MaskGenerator> CascadeClassifier::getMaskGenerator()
{
    CV_Assert(!empty());
    return cc->getMaskGenerator();
}

} // namespace cv
