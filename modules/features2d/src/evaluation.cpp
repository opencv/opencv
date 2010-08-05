//*M///////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
#include <limits>

using namespace cv;
using namespace std;

inline Point2f applyHomography( const Mat_<double>& H, const Point2f& pt )
{
    double z = H(2,0)*pt.x + H(2,1)*pt.y + H(2,2);
    if( z )
    {
        double w = 1./z;
        return Point2f( (H(0,0)*pt.x + H(0,1)*pt.y + H(0,2))*w, (H(1,0)*pt.x + H(1,1)*pt.y + H(1,2))*w );
    }
    return Point2f( numeric_limits<double>::max(), numeric_limits<double>::max() );
}

inline void linearizeHomographyAt( const Mat_<double>& H, const Point2f& pt, Mat_<double>& A )
{
    A.create(2,2);
    double p1 = H(0,0)*pt.x + H(0,1)*pt.y + H(0,2),
           p2 = H(1,0)*pt.x + H(1,1)*pt.y + H(1,2),
           p3 = H(2,0)*pt.x + H(2,1)*pt.y + H(2,2),
           p3_2 = p3*p3;
    if( p3 )
    {
        A(0,0) = H(0,0)/p3 - p1*H(2,0)/p3_2; // fxdx
        A(0,1) = H(0,1)/p3 - p1*H(2,1)/p3_2; // fxdy

        A(1,0) = H(1,0)/p3 - p2*H(2,0)/p3_2; // fydx
        A(1,1) = H(1,1)/p3 - p2*H(2,1)/p3_2; // fydx
    }
    else
        A.setTo(Scalar::all(numeric_limits<double>::max()));
}

class EllipticKeyPoint
{
public:
    EllipticKeyPoint();
    EllipticKeyPoint( const Point2f& _center, const Scalar& _ellipse );

    static void convert( const vector<KeyPoint>& src, vector<EllipticKeyPoint>& dst );
    static void convert( const vector<EllipticKeyPoint>& src, vector<KeyPoint>& dst );

    static Mat_<double> getSecondMomentsMatrix( const Scalar& _ellipse );
    Mat_<double> getSecondMomentsMatrix() const;

    void calcProjection( const Mat_<double>& H, EllipticKeyPoint& projection ) const;
    static void calcProjection( const vector<EllipticKeyPoint>& src, const Mat_<double>& H, vector<EllipticKeyPoint>& dst );

    Point2f center;
    Scalar ellipse; // 3 elements a, b, c: ax^2+2bxy+cy^2=1
    Size_<float> axes; // half lenght of elipse axes
    Size_<float> boundingBox; // half sizes of bounding box which sides are parallel to the coordinate axes
};

EllipticKeyPoint::EllipticKeyPoint()
{
    *this = EllipticKeyPoint(Point2f(0,0), Scalar(1, 0, 1) );
}

EllipticKeyPoint::EllipticKeyPoint( const Point2f& _center, const Scalar& _ellipse )
{
    center = _center;
    ellipse = _ellipse;

    Mat_<double> M = getSecondMomentsMatrix(_ellipse), eval;
    eigen( M, eval );
    assert( eval.rows == 2 && eval.cols == 1 );
    axes.width = 1.f / sqrt(eval(0,0));
    axes.height = 1.f / sqrt(eval(1,0));

    float ac_b2 = ellipse[0]*ellipse[2] - ellipse[1]*ellipse[1];
    boundingBox.width = sqrt(ellipse[2]/ac_b2);
    boundingBox.height = sqrt(ellipse[0]/ac_b2);
}

Mat_<double> EllipticKeyPoint::getSecondMomentsMatrix( const Scalar& _ellipse )
{
    Mat_<double> M(2, 2);
    M(0,0) = _ellipse[0];
    M(1,0) = M(0,1) = _ellipse[1];
    M(1,1) = _ellipse[2];
    return M;
}

Mat_<double> EllipticKeyPoint::getSecondMomentsMatrix() const
{
    return getSecondMomentsMatrix(ellipse);
}

void EllipticKeyPoint::calcProjection( const Mat_<double>& H, EllipticKeyPoint& projection ) const
{
    Point2f dstCenter = applyHomography(H, center);

    Mat_<double> invM; invert(getSecondMomentsMatrix(), invM);
    Mat_<double> Aff; linearizeHomographyAt(H, center, Aff);
    Mat_<double> dstM; invert(Aff*invM*Aff.t(), dstM);

    projection = EllipticKeyPoint( dstCenter, Scalar(dstM(0,0), dstM(0,1), dstM(1,1)) );
}

void EllipticKeyPoint::convert( const vector<KeyPoint>& src, vector<EllipticKeyPoint>& dst )
{
    if( !src.empty() )
    {
        dst.resize(src.size());
        for( size_t i = 0; i < src.size(); i++ )
        {
            float rad = src[i].size/2;
            assert( rad );
            float fac = 1.f/(rad*rad);
            dst[i] = EllipticKeyPoint( src[i].pt, Scalar(fac, 0, fac) );
        }
    }
}

void EllipticKeyPoint::convert( const vector<EllipticKeyPoint>& src, vector<KeyPoint>& dst )
{
    if( !src.empty() )
    {
        dst.resize(src.size());
        for( size_t i = 0; i < src.size(); i++ )
        {
            Size_<float> axes = src[i].axes;
            float rad = sqrt(axes.height*axes.width);
            dst[i] = KeyPoint(src[i].center, 2*rad );
        }
    }
}

void EllipticKeyPoint::calcProjection( const vector<EllipticKeyPoint>& src, const Mat_<double>& H, vector<EllipticKeyPoint>& dst )
{
    if( !src.empty() )
    {
        assert( !H.empty() && H.cols == 3 && H.rows == 3);
        dst.resize(src.size());
        vector<EllipticKeyPoint>::const_iterator srcIt = src.begin();
        vector<EllipticKeyPoint>::iterator       dstIt = dst.begin();
        for( ; srcIt != src.end(); ++srcIt, ++dstIt )
            srcIt->calcProjection(H, *dstIt);
    }
}

static void filterEllipticKeyPointsByImageSize( vector<EllipticKeyPoint>& keypoints, const Size& imgSize )
{
    if( !keypoints.empty() )
    {
        vector<EllipticKeyPoint> filtered;
        filtered.reserve(keypoints.size());
        vector<EllipticKeyPoint>::const_iterator it = keypoints.begin();
        for( int i = 0; it != keypoints.end(); ++it, i++ )
        {
            if( it->center.x + it->boundingBox.width < imgSize.width &&
                it->center.x - it->boundingBox.width > 0 &&
                it->center.y + it->boundingBox.height < imgSize.height &&
                it->center.y - it->boundingBox.height > 0 )
                filtered.push_back(*it);
        }
        keypoints.assign(filtered.begin(), filtered.end());
    }
}

static void overlap( const vector<EllipticKeyPoint>& keypoints1, const vector<EllipticKeyPoint>& keypoints2t, bool commonPart,
                     SparseMat_<float>& overlaps )
{
    overlaps.clear();
    if( keypoints1.empty() || keypoints2t.empty() )
        return;

    int size[] = { keypoints1.size(), keypoints2t.size() };
    overlaps.create( 2, size );

    for( size_t i1 = 0; i1 < keypoints1.size(); i1++ )
    {
        EllipticKeyPoint kp1 = keypoints1[i1];
        float maxDist = sqrt(kp1.axes.width*kp1.axes.height),
              fac = 30.f/maxDist;
        if( !commonPart )
            fac=3;

        maxDist = maxDist*4;
        fac = 1.0/(fac*fac);

        EllipticKeyPoint keypoint1a = EllipticKeyPoint( kp1.center, Scalar(fac*kp1.ellipse[0], fac*kp1.ellipse[1], fac*kp1.ellipse[2]) );

        for( size_t i2 = 0; i2 < keypoints2t.size(); i2++ )
        {
            EllipticKeyPoint kp2 = keypoints2t[i2];
            Point2f diff = kp2.center - kp1.center;

            if( norm(diff) < maxDist )
            {
                EllipticKeyPoint keypoint2a = EllipticKeyPoint( kp2.center, Scalar(fac*kp2.ellipse[0], fac*kp2.ellipse[1], fac*kp2.ellipse[2]) );
                //find the largest eigenvalue
                float maxx =  ceil(( keypoint1a.boundingBox.width > (diff.x+keypoint2a.boundingBox.width)) ?
                                     keypoint1a.boundingBox.width : (diff.x+keypoint2a.boundingBox.width));
                float minx = floor((-keypoint1a.boundingBox.width < (diff.x-keypoint2a.boundingBox.width)) ?
                                    -keypoint1a.boundingBox.width : (diff.x-keypoint2a.boundingBox.width));

                float maxy =  ceil(( keypoint1a.boundingBox.height > (diff.y+keypoint2a.boundingBox.height)) ?
                                     keypoint1a.boundingBox.height : (diff.y+keypoint2a.boundingBox.height));
                float miny = floor((-keypoint1a.boundingBox.height < (diff.y-keypoint2a.boundingBox.height)) ?
                                    -keypoint1a.boundingBox.height : (diff.y-keypoint2a.boundingBox.height));
                float mina = (maxx-minx) < (maxy-miny) ? (maxx-minx) : (maxy-miny) ;
                float dr = mina/50.0;
                float bua = 0, bna = 0;
                //compute the area
                for( float rx1 = minx; rx1 <= maxx; rx1+=dr )
                {
                    float rx2 = rx1-diff.x;
                    for( float ry1=miny; ry1<=maxy; ry1+=dr )
                    {
                        float ry2=ry1-diff.y;
                        //compute the distance from the ellipse center
                        float e1 = keypoint1a.ellipse[0]*rx1*rx1+2*keypoint1a.ellipse[1]*rx1*ry1+keypoint1a.ellipse[2]*ry1*ry1;
                        float e2 = keypoint2a.ellipse[0]*rx2*rx2+2*keypoint2a.ellipse[1]*rx2*ry2+keypoint2a.ellipse[2]*ry2*ry2;
                        //compute the area
                        if( e1<1 && e2<1 ) bna++;
                        if( e1<1 || e2<1 ) bua++;
                    }
                }
                if( bna > 0)
                    overlaps.ref(i1,i2) = bna/bua;
            }
        }
    }
}

static void calculateRepeatability( const Mat& img1, const Mat& img2, const Mat& H1to2,
                                    const vector<KeyPoint>& _keypoints1, const vector<KeyPoint>& _keypoints2,
                                    float& repeatability, int& correspondencesCount,
                                    SparseMat_<uchar>* thresholdedOverlapMask=0 )
{
    vector<EllipticKeyPoint> keypoints1, keypoints2, keypoints1t, keypoints2t;
    EllipticKeyPoint::convert( _keypoints1, keypoints1 );
    EllipticKeyPoint::convert( _keypoints2, keypoints2 );

    // calculate projections of key points
    EllipticKeyPoint::calcProjection( keypoints1, H1to2, keypoints1t );
    Mat H2to1; invert(H1to2, H2to1);
    EllipticKeyPoint::calcProjection( keypoints2, H2to1, keypoints2t );

    bool ifEvaluateDetectors = !thresholdedOverlapMask; // == commonPart
    float overlapThreshold;
    if( ifEvaluateDetectors )
    {
        overlapThreshold = 1.f - 0.4f;

        // remove key points from outside of the common image part
        Size sz1 = img1.size(), sz2 = img2.size();
        filterEllipticKeyPointsByImageSize( keypoints1, sz1 );
        filterEllipticKeyPointsByImageSize( keypoints1t, sz2 );
        filterEllipticKeyPointsByImageSize( keypoints2, sz2 );
        filterEllipticKeyPointsByImageSize( keypoints2t, sz1 );
    }
    else
    {
        overlapThreshold = 1.f - 0.5f;
    }
    int minCount = min( keypoints1.size(), keypoints2t.size() );

    // calculate overlap errors
    SparseMat_<float> overlaps;
    overlap( keypoints1, keypoints2t, ifEvaluateDetectors, overlaps );

    correspondencesCount = -1;
    repeatability = -1.f;
    const int* size = overlaps.size();
    if( !size || overlaps.nzcount() == 0 )
        return;

    if( ifEvaluateDetectors )
    {
        // threshold the overlaps
        for( int y = 0; y < size[0]; y++ )
        {
            for( int x = 0; x < size[1]; x++ )
            {
                if ( overlaps(y,x) < overlapThreshold )
                    overlaps.erase(y,x);
            }
        }

        // regions one-to-one matching
        correspondencesCount = 0;
        while( overlaps.nzcount() > 0 )
        {
            double maxOverlap = 0;
            int maxIdx[2];
            minMaxLoc( overlaps, 0, &maxOverlap, 0, maxIdx );
            for( size_t i1 = 0; i1 < keypoints1.size(); i1++ )
                overlaps.erase(i1, maxIdx[1]);
            for( size_t i2 = 0; i2 < keypoints2t.size(); i2++ )
                overlaps.erase(maxIdx[0], i2);
            correspondencesCount++;
        }
        repeatability = minCount ? (float)correspondencesCount/minCount : -1;
    }
    else
    {
        thresholdedOverlapMask->create( 2, size );
        for( int y = 0; y < size[0]; y++ )
        {
            for( int x = 0; x < size[1]; x++ )
            {
                float val = overlaps(y,x);
                if ( val >= overlapThreshold )
                    thresholdedOverlapMask->ref(y,x) = 1;
            }
        }
    }
}

void cv::evaluateFeatureDetector( const Mat& img1, const Mat& img2, const Mat& H1to2,
                              vector<KeyPoint>* _keypoints1, vector<KeyPoint>* _keypoints2,
                              float& repeatability, int& correspCount,
                              const Ptr<FeatureDetector>& _fdetector )
{
    Ptr<FeatureDetector> fdetector(_fdetector);
    vector<KeyPoint> *keypoints1, *keypoints2, buf1, buf2;
    keypoints1 = _keypoints1 != 0 ? _keypoints1 : &buf1;
    keypoints2 = _keypoints2 != 0 ? _keypoints2 : &buf2;

    if( (keypoints1->empty() || keypoints2->empty()) && fdetector.empty() )
        CV_Error( CV_StsBadArg, "fdetector must be no empty when keypoints1 or keypoints2 is empty" );

    if( keypoints1->empty() )
        fdetector->detect( img1, *keypoints1 );
    if( keypoints2->empty() )
        fdetector->detect( img1, *keypoints2 );

    calculateRepeatability( img1, img2, H1to2, *keypoints1, *keypoints2, repeatability, correspCount );
}

struct DMatchForEvaluation : public DMatch
{
    uchar isCorrect;
    DMatchForEvaluation( const DMatch &dm ) : DMatch( dm ) {}
};

static inline float recall( int correctMatchCount, int correspondenceCount )
{
    return correspondenceCount ? (float)correctMatchCount / (float)correspondenceCount : -1;
}

static inline float precision( int correctMatchCount, int falseMatchCount )
{
    return correctMatchCount + falseMatchCount ? (float)correctMatchCount / (float)(correctMatchCount + falseMatchCount) : -1;
}

void cv::computeRecallPrecisionCurve( const vector<vector<DMatch> >& matches1to2,
                                      const vector<vector<uchar> >& correctMatches1to2Mask,
                                      vector<Point2f>& recallPrecisionCurve )
{
    CV_Assert( matches1to2.size() == correctMatches1to2Mask.size() );

    vector<DMatchForEvaluation> allMatches;
    int correspondenceCount = 0;
    for( size_t i = 0; i < matches1to2.size(); i++ )
    {
        for( size_t j = 0; j < matches1to2[i].size(); j++ )
        {
            DMatchForEvaluation match = matches1to2[i][j];
            match.isCorrect = correctMatches1to2Mask[i][j] ;
            allMatches.push_back( match );
            correspondenceCount += match.isCorrect != 0 ? 1 : 0;
        }
    }

    std::sort( allMatches.begin(), allMatches.end() );

    int correctMatchCount = 0, falseMatchCount = 0;
    recallPrecisionCurve.resize( allMatches.size() );
    for( size_t i = 0; i < allMatches.size(); i++ )
    {
        if( allMatches[i].isCorrect )
            correctMatchCount++;
        else
            falseMatchCount++;

        float r = recall( correctMatchCount, correspondenceCount );
        float p =  precision( correctMatchCount, falseMatchCount );
        recallPrecisionCurve[i] = Point2f(1-p, r);
    }
}

float cv::getRecall( const vector<Point2f>& recallPrecisionCurve, float l_precision )
{
    float recall = -1;

    if( l_precision >= 0 && l_precision <= 1 )
    {
        int bestIdx = -1;
        float minDiff = FLT_MAX;
        for( size_t i = 0; i < recallPrecisionCurve.size(); i++ )
        {
            float curDiff = std::fabs(l_precision - recallPrecisionCurve[i].x);
            if( curDiff <= minDiff )
            {
                bestIdx = i;
                minDiff = curDiff;
            }
        }

        recall = recallPrecisionCurve[bestIdx].y;
    }

    return recall;
}

void cv::evaluateDescriptorMatch( const Mat& img1, const Mat& img2, const Mat& H1to2,
                                  vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2,
                                  vector<vector<DMatch> >* _matches1to2, vector<vector<uchar> >* _correctMatches1to2Mask,
                                  vector<Point2f>& recallPrecisionCurve,
                                  const Ptr<GenericDescriptorMatch>& _dmatch )
{
    Ptr<GenericDescriptorMatch> dmatch = _dmatch;
    dmatch->clear();

    vector<vector<DMatch> > *matches1to2, buf1;
    vector<vector<uchar> > *correctMatches1to2Mask, buf2;
    matches1to2 = _matches1to2 != 0 ? _matches1to2 : &buf1;
    correctMatches1to2Mask = _correctMatches1to2Mask != 0 ? _correctMatches1to2Mask : &buf2;

    if( keypoints1.empty() || keypoints2.empty() )
        CV_Error( CV_StsBadArg, "keypoints1 and keypoints2 must be no empty" );
    if( matches1to2->empty() && dmatch.empty() )
        CV_Error( CV_StsBadArg, "dmatch must be no empty when matches1to2 is empty" );
    if( matches1to2->empty() )
    {
        dmatch->add( img2, keypoints2 );
        //TODO: use more sophisticated strategy to choose threshold
        dmatch->match( img1, keypoints1, *matches1to2, std::numeric_limits<float>::max() );
    }
    float repeatability;
    int correspCount;
    SparseMat_<uchar> thresholdedOverlapMask; // thresholded allOverlapErrors
    calculateRepeatability( img1, img2, H1to2,
                            keypoints1, keypoints2,
                            repeatability, correspCount,
                            &thresholdedOverlapMask );

    correctMatches1to2Mask->resize(matches1to2->size());
    int ddd = 0;
    for( size_t i = 0; i < matches1to2->size(); i++ )
    {
        (*correctMatches1to2Mask)[i].resize((*matches1to2)[i].size());
        for( size_t j = 0;j < (*matches1to2)[i].size(); j++ )
        {
            int indexQuery = (*matches1to2)[i][j].indexQuery;
            int indexTrain = (*matches1to2)[i][j].indexTrain;
            (*correctMatches1to2Mask)[i][j] = thresholdedOverlapMask( indexQuery, indexTrain );
            ddd += thresholdedOverlapMask( indexQuery, indexTrain ) != 0 ? 1 : 0;
        }
    }

    computeRecallPrecisionCurve( *matches1to2, *correctMatches1to2Mask, recallPrecisionCurve );
}
