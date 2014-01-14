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
bool FeatureEvaluator::setImage(InputArray, Size, const std::vector<double>&) {return true;}
bool FeatureEvaluator::setWindow(Point, int) { return true; }
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
    sumSize0 = Size();
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
    ret->optfeaturesPtr = 0;
    ret->scaleData = scaleData;
    ret->hasTiltedFeatures = hasTiltedFeatures;
    ret->sum0 = sum0; ret->sqsum0 = sqsum0;
    ret->sum = sum; ret->sqsum = sqsum;
    ret->usum0 = usum0; ret->usqsum0 = usqsum0; ret->ufbuf = ufbuf;
    ret->pwin = 0;
    ret->varianceNormFactor = 0;
    ret->sumSize0 = sumSize0;
    return ret;
}


bool HaarEvaluator::setImage( InputArray _image, Size _origWinSize,
                              const std::vector<double>& _scales )
{
    Size imgsz = _image.size();
    int irows = imgsz.height+1, icols = imgsz.width+1;
    int row_scale = hasTiltedFeatures ? 2 : 1;

    Size prevSumSize = sumSize0;
    sumSize0.width = (int)alignSize(std::max(sumSize0.width, icols), 32);
    sumSize0.height = std::max(sumSize0.height, irows*row_scale);

    origWinSize = _origWinSize;
    int sumStep, tofs = 0;

    if( _image.isUMat() )
    {
        usum0.create(sumSize0, CV_32S);
        usqsum0.create(sumSize0, CV_64F);
        usum = UMat(usum0, Rect(0, 0, icols, irows));
        usqsum = UMat(usqsum0, Rect(0, 0, icols, irows));

        if( hasTiltedFeatures )
        {
            UMat utilted(usum0, Rect(0, irows, icols, irows));
            integral(_image, usum, usqsum, utilted, CV_32S);
            tofs = (int)((utilted.offset - usum.offset)/sizeof(int));
        }
        else
        {
            integral(_image, usum, usqsum, noArray(), CV_32S);
        }
        sumStep = (int)(usum.step/usum.elemSize());
    }
    else
    {
        sum0.create(sumSize0, CV_32S);
        sqsum0.create(sumSize0, CV_64F);
        sum = Mat(sum0, Rect(0, 0, icols, irows));
        sqsum = Mat(sqsum0, Rect(0, 0, icols, irows));

        if( hasTiltedFeatures )
        {
            Mat tilted = sum0(Rect(0, irows, icols, irows));
            integral(_image, sum, sqsum, tilted, CV_32S, CV_64F);
            tofs = (int)((tilted.data - sum.data)/sizeof(int));
        }
        else
            integral(_image, sum, sqsum, noArray(), CV_32S, CV_64F);
        sumStep = (int)(sum.step/sum.elemSize());
    }

    size_t prevNScales = scaleData->size();
    size_t i, nscales = _scales.size();
    scaleData->resize(nscales);

    bool recalcFeatures = prevSumSize != sumSize0 || prevNScales != nscales;
    Rect normrect0(1, 1, origWinSize.width-2, origWinSize.height-2);

    for( i = 0; i < nscales; i++ )
    {
        ScaleData& s = scaleData->at(i);
        if( !recalcFeatures && fabs(s.scale - _scales[i]) > FLT_EPSILON*100*_scales[i] )
            recalcFeatures = true;
        double sc = _scales[i];
        s.scale = (float)sc;
        s.normrect.x = cvRound(normrect0.x*sc);
        s.normrect.y = cvRound(normrect0.y*sc);
        s.normrect.width = cvRound(normrect0.width*sc);
        s.normrect.height = cvRound(normrect0.height*sc);
        s.nscale = 1;//(float)(normrect0.area()/s.normrect.area());
        s.winSize.width = cvRound(origWinSize.width*sc);
        s.winSize.height = cvRound(origWinSize.height*sc);

        CV_SUM_OFS( s.nofs[0], s.nofs[1], s.nofs[2], s.nofs[3], 0, s.normrect, sumStep );
    }

    size_t fi, nfeatures = features->size();
    const std::vector<Feature>& ff = *features;

    if( recalcFeatures )
    {
        optfeatures->resize(nfeatures*nscales);
        optfeaturesPtr = &(*optfeatures)[0];
        for( fi = 0; fi < nfeatures*nscales; fi++ )
        {
            const ScaleData& s = scaleData->at(fi/nfeatures);
            optfeaturesPtr[fi].setOffsets( ff[fi%nfeatures], sumStep, tofs, s.scale, s.nscale, s.winSize );
        }
    }
    if( _image.isUMat() && (recalcFeatures || ufbuf.empty()) )
        copyVectorToUMat(*optfeatures, ufbuf);

    return true;
}


bool HaarEvaluator::setWindow( Point pt, int scaleIdx )
{
    CV_Assert(0 <= scaleIdx && scaleIdx < (int)scaleData->size());

    if( pt.x < 0 || pt.y < 0 ||
        pt.x + origWinSize.width >= sum.cols ||
        pt.y + origWinSize.height >= sum.rows )
        return false;

    const int* p = &sum.at<int>(pt);
    const double* pq = &sqsum.at<double>(pt);
    const ScaleData& s = scaleData->at(scaleIdx);
    const int* nofs = s.nofs;
    int valsum = CALC_SUM_OFS(nofs, p);
    double valsqsum = CALC_SUM_OFS(nofs, pq);

    double nf = ((double)s.normrect.area() * valsqsum - (double)valsum * valsum)*s.nscale;
    if( nf > 0. )
        nf = std::sqrt(nf);
    else
        nf = 1.;
    varianceNormFactor = (1./nf);
    pwin = p;
    optfeaturesPtr = &(*optfeatures)[scaleIdx*features->size()];

    return true;
}


void HaarEvaluator::OptFeature::setOffsets( const Feature& _f, int step,
                                            int tofs, double scale, double nscale,
                                            Size winSize )
{
#if 0
    double wsum0 = 0;
    int area0 = 0;
    for( int k = 0; k < Feature::RECT_NUM; k++ )
    {
        Rect tr = _f.rect[k].r;
        double area = tr.area();
        if( fabs(_f.rect[k].weight) < FLT_EPSILON )
            tr = Rect(0,0,0,0);

        tr.x = cvRound(tr.x*scale);
        tr.y = cvRound(tr.y*scale);
        tr.width = cvRound(tr.width*scale);
        tr.height = cvRound(tr.height*scale);
        tr &= Rect(0, 0, winSize.width, winSize.height);
        double correction_ratio = 0.;
        if( area > 0 )
            correction_ratio = nscale;

        if( _f.tilted )
        {
            CV_TILTED_OFS( ofs[k][0], ofs[k][1], ofs[k][2], ofs[k][3], tofs, tr, step );
            correction_ratio *= 0.5;
        }
        else
        {
            CV_SUM_OFS( ofs[k][0], ofs[k][1], ofs[k][2], ofs[k][3], 0, tr, step );
        }
        weight[k] = _f.rect[k].weight*correction_ratio;
        if( k == 0 )
            area0 = tr.area();
        else
            wsum0 += tr.area()*weight[k];
    }
    weight[0] = -wsum0/area0;
#else
    Rect r[Feature::RECT_NUM];
    int base_w = -1, base_h = -1;
    int new_base_w = 0, new_base_h = 0;
    int flagx = 0, flagy = 0;
    int k, x0 = 0, y0 = 0;

    // align blocks
    for( k = 0; k < Feature::RECT_NUM; k++ )
    {
        if( fabs(_f.rect[k].weight) < FLT_EPSILON )
            break;
        r[k] = _f.rect[k].r;
        base_w = (int)std::min( (unsigned)base_w, (unsigned)(r[k].width-1) );
        base_w = (int)std::min( (unsigned)base_w, (unsigned)(r[k].x - r[0].x-1) );
        base_h = (int)std::min( (unsigned)base_h, (unsigned)(r[k].height-1) );
        base_h = (int)std::min( (unsigned)base_h, (unsigned)(r[k].y - r[0].y-1) );
    }

    int nr = k;

    base_w += 1;
    base_h += 1;
    int kx = r[0].width / base_w;
    int ky = r[0].height / base_h;

    if( kx <= 0 )
    {
        flagx = 1;
        new_base_w = cvRound( r[0].width * scale ) / kx;
        x0 = cvRound( r[0].x * scale );
    }

    if( ky <= 0 )
    {
        flagy = 1;
        new_base_h = cvRound( r[0].height * scale ) / ky;
        y0 = cvRound( r[0].y * scale );
    }

    int area0 = 0;
    double wsum0 = 0;

    for( k = 0; k < nr; k++ )
    {
        Rect tr;
        if( fabs(_f.rect[k].weight) >= FLT_EPSILON )
        {
            if( flagx )
            {
                tr.x = (r[k].x - r[0].x) * new_base_w / base_w + x0;
                tr.width = r[k].width * new_base_w / base_w;
            }
            else
            {
                tr.x = cvRound( r[k].x * scale );
                tr.width = cvRound( r[k].width * scale );
            }

            if( flagy )
            {
                tr.y = (r[k].y - r[0].y) * new_base_h / base_h + y0;
                tr.height = r[k].height * new_base_h / base_h;
            }
            else
            {
                tr.y = cvRound( r[k].y * scale );
                tr.height = cvRound( r[k].height * scale );
            }
            tr &= Rect(0, 0, winSize.width, winSize.height);
        }

        double correction_ratio = nscale;

        if( _f.tilted )
        {
            CV_TILTED_OFS( ofs[k][0], ofs[k][1], ofs[k][2], ofs[k][3], tofs, tr, step );
            correction_ratio *= 0.5;
        }
        else
        {
            CV_SUM_OFS( ofs[k][0], ofs[k][1], ofs[k][2], ofs[k][3], 0, tr, step );
        }

        weight[k] = (float)(_f.rect[k].weight * correction_ratio);
        
        if( k == 0 )
            area0 = tr.width * tr.height;
        else
            wsum0 += weight[k] * tr.width * tr.height;
    }
    
    weight[0] = (float)(-wsum0/area0);
#endif
}

/*Rect HaarEvaluator::getNormRect() const
{
    return normrect;
}*/

void HaarEvaluator::getUMats(std::vector<UMat>& bufs)
{
    bufs.clear();
    bufs.push_back(usum);
    bufs.push_back(usqsum);
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
    ret->optfeaturesPtr = 0;
    ret->sum0 = sum0, ret->sum = sum;
    ret->pwin = pwin;
    return ret;
}

bool LBPEvaluator::setImage( InputArray _image, Size _origWinSize, const std::vector<double>& _scales )
{
    Size imgsz = _image.size();
    int irows = imgsz.height+1, icols = imgsz.width+1;

    Size prevSumSize = sumSize0;
    sumSize0.width = (int)alignSize(std::max(sumSize0.width, icols), 32);
    sumSize0.height = std::max(sumSize0.height, irows);

    origWinSize = _origWinSize;
    int sumStep;

    if( _image.isUMat() )
    {
        usum0.create(sumSize0, CV_32S);
        usum = UMat(usum0, Rect(0, 0, icols, irows));

        integral(_image, usum, noArray(), noArray(), CV_32S);
        sumStep = (int)(usum.step/usum.elemSize());
    }
    else
    {
        sum0.create(sumSize0, CV_32S);
        sum = Mat(sum0, Rect(0, 0, icols, irows));

        integral(_image, sum, noArray(), noArray(), CV_32S);
        sumStep = (int)(sum.step/sum.elemSize());
    }

    size_t prevNScales = scaleData->size();
    size_t i, nscales = _scales.size();
    scaleData->resize(nscales);

    bool recalcFeatures = prevSumSize != sumSize0 || prevNScales != nscales;
    Rect normrect0(1, 1, origWinSize.width-2, origWinSize.height-2);

    for( i = 0; i < nscales; i++ )
    {
        ScaleData& s = scaleData->at(i);
        if( !recalcFeatures && fabs(s.scale - _scales[i]) > FLT_EPSILON*100*_scales[i] )
            recalcFeatures = true;
        double sc = _scales[i];
        s.scale = (float)sc;
        s.winSize.width = cvRound(origWinSize.width*sc);
        s.winSize.height = cvRound(origWinSize.height*sc);
    }

    size_t fi, nfeatures = features->size();
    const std::vector<Feature>& ff = *features;

    if( recalcFeatures )
    {
        optfeatures->resize(nfeatures*nscales);
        optfeaturesPtr = &(*optfeatures)[0];
        for( fi = 0; fi < nfeatures*nscales; fi++ )
        {
            const ScaleData& s = scaleData->at(fi/nfeatures);
            optfeaturesPtr[fi].setOffsets( ff[fi%nfeatures], sumStep, s.scale, s.winSize );
        }
    }
    if( _image.isUMat() && (recalcFeatures || ufbuf.empty()) )
        copyVectorToUMat(*optfeatures, ufbuf);
    
    return true;
}

void LBPEvaluator::OptFeature::setOffsets( const Feature& _f, int step, double scale, Size winSize )
{
    Rect tr = _f.rect;
    tr.x = cvRound(tr.x*scale);
    tr.y = cvRound(tr.y*scale);
    tr.width = cvRound(tr.width*scale);
    tr.height = cvRound(tr.height*scale);
    tr.width *= 3;
    tr.height *= 3;
    tr &= Rect(0, 0, winSize.width, winSize.height);
    tr.width /= 3;
    tr.height /= 3;
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
    if( pt.x < 0 || pt.y < 0 ||
        pt.x + origWinSize.width >= sum.cols ||
        pt.y + origWinSize.height >= sum.rows )
        return false;
    pwin = &sum.at<int>(pt);
    optfeaturesPtr = &(*optfeatures)[scaleIdx*features->size()];
    return true;
}


void LBPEvaluator::getUMats(std::vector<UMat>& bufs)
{
    bufs.clear();
    bufs.push_back(usum);
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

bool HOGEvaluator::setImage( InputArray _image, Size winSize, const std::vector<double>& _scales )
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
    CV_Assert( !oldCascade );

    assert( data.featureType == FeatureEvaluator::HAAR ||
            data.featureType == FeatureEvaluator::LBP ||
            data.featureType == FeatureEvaluator::HOG );

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
    CascadeClassifierInvoker( CascadeClassifierImpl& _cc,
                              Size _imgSize, Size _tileSize,
                              const std::vector<double>& _scales,
                              std::vector<Rect>& _vec, std::vector<int>& _levels,
                              std::vector<double>& _weights,
                              bool outputLevels, const Mat& _mask, Mutex* _mtx)
    {
        classifier = &_cc;
        imgSize = _imgSize;
        tileSize = _tileSize;
        scales = _scales;
        rectangles = &_vec;
        rejectLevels = outputLevels ? &_levels : 0;
        levelWeights = outputLevels ? &_weights : 0;
        mask = _mask;
        mtx = _mtx;
    }

    void operator()(const Range& range) const
    {
        Ptr<FeatureEvaluator> evaluator = classifier->featureEvaluator->clone();
        int nScales = (int)scales.size();
        int xTiles = (imgSize.width + tileSize.width-1)/tileSize.width;
        uchar* mdata = mask.data;
        double gypWeight = 0.;

        for( int t = range.start; t < range.end; t++ )
        {
            int x0 = (t % xTiles)*tileSize.width;
            int y0 = (t / xTiles)*tileSize.height;

            for( int scaleIdx = 0; scaleIdx < nScales; scaleIdx++ )
            {
                double scalingFactor = scales[scaleIdx];
                double yStep = max(2., scalingFactor);
                Size winSize(cvRound(classifier->data.origWinSize.width * scalingFactor),
                             cvRound(classifier->data.origWinSize.height * scalingFactor));

                double startX = cvCeil(x0/yStep)*yStep;
                double startY = cvCeil(y0/yStep)*yStep;
                double fx, fy;

                for( fy = startY; fy < y0 + tileSize.height; fy += yStep )
                {
                    int y = cvRound(fy);
                    if( y + winSize.height >= imgSize.height )
                        break;
                    for( fx = startX; fx < x0 + tileSize.width; fx += yStep )
                    {
                        int x = cvRound(fx);
                        if( x + winSize.width >= imgSize.width )
                            break;
                        if( mdata && !mask.at<uchar>(y,x) )
                            continue;
                        int result = classifier->runAt(evaluator, Point(x, y), scaleIdx, gypWeight);
                        if( rejectLevels )
                        {
                            if( result == 1 )
                                result = -(int)classifier->data.stages.size();
                            if( classifier->data.stages.size() + result == 0 )
                            {
                                mtx->lock();
                                rectangles->push_back(Rect(x, y, winSize.width, winSize.height));
                                rejectLevels->push_back(-result);
                                levelWeights->push_back(gypWeight);
                                mtx->unlock();
                            }
                        }
                        else if( result > 0 )
                        {
                            mtx->lock();
                            rectangles->push_back(Rect(x, y, winSize.width, winSize.height));
                            mtx->unlock();
                        }
                        if( result == 0 )
                            fx += yStep;
                    }
                }
            }
        }
    }

    CascadeClassifierImpl* classifier;
    std::vector<Rect>* rectangles;
    Size imgSize, tileSize;
    std::vector<int> *rejectLevels;
    std::vector<double> *levelWeights;
    std::vector<double> scales;
    Mat mask;
    Mutex* mtx;
};

struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };
struct getNeighbors { int operator ()(const CvAvgComp& e) const { return e.neighbors; } };


#if 0
bool CascadeClassifierImpl::ocl_detectSingleScale( InputArray _image, Size processingRectSize,
                                                   int yStep, double factor, Size sumSize0 )
{
    int featureType = getFeatureType();
    std::vector<UMat> bufs;
    size_t globalsize[] = { processingRectSize.width/yStep, processingRectSize.height/yStep };
    bool ok = false;

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

        haar->setImage(_image, data.origWinSize, sumSize0);
        if( haarKernel.empty() )
        {
            haarKernel.create("runHaarClassifierStump", ocl::objdetect::cascadedetect_oclsrc, "");
            if( haarKernel.empty() )
                return false;
        }

        haar->getUMats(bufs);
        Rect normrect = haar->getNormRect();

        haarKernel.args(ocl::KernelArg::ReadOnlyNoSize(bufs[0]), // sum
                        ocl::KernelArg::ReadOnlyNoSize(bufs[1]), // sqsum
                        ocl::KernelArg::PtrReadOnly(bufs[2]), // optfeatures

                        // cascade classifier
                        (int)data.stages.size(),
                        ocl::KernelArg::PtrReadOnly(ustages),
                        ocl::KernelArg::PtrReadOnly(ustumps),

                        ocl::KernelArg::PtrWriteOnly(ufacepos), // positions
                        processingRectSize,
                        yStep, (float)factor,
                        normrect, data.origWinSize, (int)MAX_FACES);
        ok = haarKernel.run(2, globalsize, 0, true);
    }
    else if( featureType == FeatureEvaluator::LBP )
    {
        Ptr<LBPEvaluator> lbp = featureEvaluator.dynamicCast<LBPEvaluator>();
        if( lbp.empty() )
            return false;

        lbp->setImage(_image, data.origWinSize, sumSize0);
        if( lbpKernel.empty() )
        {
            lbpKernel.create("runLBPClassifierStump", ocl::objdetect::cascadedetect_oclsrc, "");
            if( lbpKernel.empty() )
                return false;
        }

        lbp->getUMats(bufs);

        int subsetSize = (data.ncategories + 31)/32;
        lbpKernel.args(ocl::KernelArg::ReadOnlyNoSize(bufs[0]), // sum
                        ocl::KernelArg::PtrReadOnly(bufs[1]), // optfeatures

                        // cascade classifier
                        (int)data.stages.size(),
                        ocl::KernelArg::PtrReadOnly(ustages),
                        ocl::KernelArg::PtrReadOnly(ustumps),
                        ocl::KernelArg::PtrReadOnly(usubsets),
                        subsetSize,

                        ocl::KernelArg::PtrWriteOnly(ufacepos), // positions
                        processingRectSize,
                        yStep, (float)factor,
                        data.origWinSize, (int)MAX_FACES);
        ok = lbpKernel.run(2, globalsize, 0, true);
    }

    if( use_ocl && tryOpenCL )
    {
        Mat facepos = ufacepos.getMat(ACCESS_READ);
        const int* fptr = facepos.ptr<int>();
        int i, nfaces = fptr[0];
        for( i = 0; i < nfaces; i++ )
        {
            candidates.push_back(Rect(fptr[i*4+1], fptr[i*4+2], fptr[i*4+3], fptr[i*4+4]));
        }
    }
    //CV_Assert(ok);
    return ok;
}
#endif

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

    bool use_ocl = false;

    /*tryOpenCL && ocl::useOpenCL() &&
        (featureType == FeatureEvaluator::HAAR ||
         featureType == FeatureEvaluator::LBP) &&
        ocl::Device::getDefault().type() != ocl::Device::TYPE_CPU &&
        !isOldFormatCascade() &&
        data.isStumpBased() &&
        maskGenerator.empty() &&
        !outputRejectLevels;

    if( use_ocl )
    {
        UMat uimage = _image.getUMat();
        if( CV_MAT_CN(imgtype) > 1 )
            cvtColor(uimage, ugrayImage, COLOR_BGR2GRAY);
        else
            uimage.copyTo(ugrayImage);
        uimageBuffer.create(imgsz.height + 1, imgsz.width + 1, CV_8U);
     
        ufacepos.create(1, MAX_FACES*4 + 1, CV_32S);
        UMat ufacecount(ufacepos, Rect(0,0,1,1));
        ufacecount.setTo(Scalar::all(0));
    }
    else*/
    {
        Mat image = _image.getMat();
        if (maskGenerator)
            maskGenerator->initializeMask(image);

        grayImage = image;
        if( CV_MAT_CN(imgtype) > 1 )
        {
            Mat temp;
            cvtColor(grayImage, temp, COLOR_BGR2GRAY);
            grayImage = temp;
        }
    }

    Size sumSize0((imgsz.width + SUM_ALIGN) & -SUM_ALIGN, imgsz.height+1);
    std::vector<double> scales;
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
        scales.push_back(factor);
    }

    // CPU code
    if( !use_ocl || !tryOpenCL )
    {
        if( !featureEvaluator->setImage(grayImage, data.origWinSize, scales) )
            return;

        Mat currentMask;
        if (maskGenerator)
            currentMask = maskGenerator->generateMask(grayImage);

        Size tileSize(128, 128);
        int nTiles = ((imgsz.width + tileSize.width - 1)/tileSize.width)*((imgsz.height + tileSize.height - 1)/tileSize.height);

        parallel_for_(Range(0, nTiles),
                      CascadeClassifierInvoker(*this, imgsz, tileSize, scales,
                                               candidates, rejectLevels, levelWeights,
                                               outputRejectLevels, currentMask, &mtx));
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
