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

template<typename _Tp> static int solveQuadratic(_Tp a, _Tp b, _Tp c, _Tp& x1, _Tp& x2)
{
    if( a == 0 )
    {
        if( b == 0 )
        {
            x1 = x2 = 0;
            return c == 0;
        }
        x1 = x2 = -c/b;
        return 1;
    }

    _Tp d = b*b - 4*a*c;
    if( d < 0 )
    {
        x1 = x2 = 0;
        return 0;
    }
    if( d > 0 )
    {
        d = std::sqrt(d);
        double s = 1/(2*a);
        x1 = (-b - d)*s;
        x2 = (-b + d)*s;
        if( x1 > x2 )
            std::swap(x1, x2);
        return 2;
    }
    x1 = x2 = -b/(2*a);
    return 1;
}

//for android ndk
#undef _S
static inline Point2f applyHomography( const Mat_<double>& H, const Point2f& pt )
{
    double z = H(2,0)*pt.x + H(2,1)*pt.y + H(2,2);
    if( z )
    {
        double w = 1./z;
        return Point2f( (float)((H(0,0)*pt.x + H(0,1)*pt.y + H(0,2))*w), (float)((H(1,0)*pt.x + H(1,1)*pt.y + H(1,2))*w) );
    }
    return Point2f( std::numeric_limits<float>::max(), std::numeric_limits<float>::max() );
}

static inline void linearizeHomographyAt( const Mat_<double>& H, const Point2f& pt, Mat_<double>& A )
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
        A.setTo(Scalar::all(std::numeric_limits<double>::max()));
}

class EllipticKeyPoint
{
public:
    EllipticKeyPoint();
    EllipticKeyPoint( const Point2f& _center, const Scalar& _ellipse );

    static void convert( const std::vector<KeyPoint>& src, std::vector<EllipticKeyPoint>& dst );
    static void convert( const std::vector<EllipticKeyPoint>& src, std::vector<KeyPoint>& dst );

    static Mat_<double> getSecondMomentsMatrix( const Scalar& _ellipse );
    Mat_<double> getSecondMomentsMatrix() const;

    void calcProjection( const Mat_<double>& H, EllipticKeyPoint& projection ) const;
    static void calcProjection( const std::vector<EllipticKeyPoint>& src, const Mat_<double>& H, std::vector<EllipticKeyPoint>& dst );

    Point2f center;
    Scalar ellipse; // 3 elements a, b, c: ax^2+2bxy+cy^2=1
    Size_<float> axes; // half length of ellipse axes
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

    double a = ellipse[0], b = ellipse[1], c = ellipse[2];
    double ac_b2 = a*c - b*b;
    double x1, x2;
    solveQuadratic(1., -(a+c), ac_b2, x1, x2);
    axes.width = (float)(1/sqrt(x1));
    axes.height = (float)(1/sqrt(x2));

    boundingBox.width = (float)sqrt(ellipse[2]/ac_b2);
    boundingBox.height = (float)sqrt(ellipse[0]/ac_b2);
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

void EllipticKeyPoint::convert( const std::vector<KeyPoint>& src, std::vector<EllipticKeyPoint>& dst )
{
    CV_INSTRUMENT_REGION()

    if( !src.empty() )
    {
        dst.resize(src.size());
        for( size_t i = 0; i < src.size(); i++ )
        {
            float rad = src[i].size/2;
            CV_Assert( rad );
            float fac = 1.f/(rad*rad);
            dst[i] = EllipticKeyPoint( src[i].pt, Scalar(fac, 0, fac) );
        }
    }
}

void EllipticKeyPoint::convert( const std::vector<EllipticKeyPoint>& src, std::vector<KeyPoint>& dst )
{
    CV_INSTRUMENT_REGION()

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

void EllipticKeyPoint::calcProjection( const std::vector<EllipticKeyPoint>& src, const Mat_<double>& H, std::vector<EllipticKeyPoint>& dst )
{
    if( !src.empty() )
    {
        CV_Assert( !H.empty() && H.cols == 3 && H.rows == 3);
        dst.resize(src.size());
        std::vector<EllipticKeyPoint>::const_iterator srcIt = src.begin();
        std::vector<EllipticKeyPoint>::iterator       dstIt = dst.begin();
        for( ; srcIt != src.end(); ++srcIt, ++dstIt )
            srcIt->calcProjection(H, *dstIt);
    }
}

static void filterEllipticKeyPointsByImageSize( std::vector<EllipticKeyPoint>& keypoints, const Size& imgSize )
{
    if( !keypoints.empty() )
    {
        std::vector<EllipticKeyPoint> filtered;
        filtered.reserve(keypoints.size());
        std::vector<EllipticKeyPoint>::const_iterator it = keypoints.begin();
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

struct IntersectAreaCounter
{
    IntersectAreaCounter( float _dr, int _minx,
                          int _miny, int _maxy,
                          const Point2f& _diff,
                          const Scalar& _ellipse1, const Scalar& _ellipse2 ) :
               dr(_dr), bua(0), bna(0), minx(_minx), miny(_miny), maxy(_maxy),
               diff(_diff), ellipse1(_ellipse1), ellipse2(_ellipse2) {}
    IntersectAreaCounter( const IntersectAreaCounter& counter, Split )
    {
        *this = counter;
        bua = 0;
        bna = 0;
    }

    void operator()( const BlockedRange& range )
    {
        CV_Assert( miny < maxy );
        CV_Assert( dr > FLT_EPSILON );

        int temp_bua = bua, temp_bna = bna;
        for( int i = range.begin(); i != range.end(); i++ )
        {
            float rx1 = minx + i*dr;
            float rx2 = rx1 - diff.x;
            for( float ry1 = (float)miny; ry1 <= (float)maxy; ry1 += dr )
            {
                float ry2 = ry1 - diff.y;
                //compute the distance from the ellipse center
                float e1 = (float)(ellipse1[0]*rx1*rx1 + 2*ellipse1[1]*rx1*ry1 + ellipse1[2]*ry1*ry1);
                float e2 = (float)(ellipse2[0]*rx2*rx2 + 2*ellipse2[1]*rx2*ry2 + ellipse2[2]*ry2*ry2);
                //compute the area
                if( e1<1 && e2<1 ) temp_bna++;
                if( e1<1 || e2<1 ) temp_bua++;
            }
        }
        bua = temp_bua;
        bna = temp_bna;
    }

    void join( IntersectAreaCounter& ac )
    {
        bua += ac.bua;
        bna += ac.bna;
    }

    float dr;
    int bua, bna;

    int minx;
    int miny, maxy;

    Point2f diff;
    Scalar ellipse1, ellipse2;

};

struct SIdx
{
    SIdx() : S(-1), i1(-1), i2(-1) {}
    SIdx(float _S, int _i1, int _i2) : S(_S), i1(_i1), i2(_i2) {}
    float S;
    int i1;
    int i2;

    bool operator<(const SIdx& v) const { return S > v.S; }

    struct UsedFinder
    {
        UsedFinder(const SIdx& _used) : used(_used) {}
        const SIdx& used;
        bool operator()(const SIdx& v) const { return  (v.i1 == used.i1 || v.i2 == used.i2); }
        UsedFinder& operator=(const UsedFinder&);
    };
};

static void computeOneToOneMatchedOverlaps( const std::vector<EllipticKeyPoint>& keypoints1, const std::vector<EllipticKeyPoint>& keypoints2t,
                                            bool commonPart, std::vector<SIdx>& overlaps, float minOverlap )
{
    CV_Assert( minOverlap >= 0.f );
    overlaps.clear();
    if( keypoints1.empty() || keypoints2t.empty() )
        return;

    overlaps.clear();
    overlaps.reserve(cvRound(keypoints1.size() * keypoints2t.size() * 0.01));

    for( size_t i1 = 0; i1 < keypoints1.size(); i1++ )
    {
        EllipticKeyPoint kp1 = keypoints1[i1];
        float maxDist = sqrt(kp1.axes.width*kp1.axes.height),
              fac = 30.f/maxDist;
        if( !commonPart )
            fac=3;

        maxDist = maxDist*4;
        fac = 1.f/(fac*fac);

        EllipticKeyPoint keypoint1a = EllipticKeyPoint( kp1.center, Scalar(fac*kp1.ellipse[0], fac*kp1.ellipse[1], fac*kp1.ellipse[2]) );

        for( size_t i2 = 0; i2 < keypoints2t.size(); i2++ )
        {
            EllipticKeyPoint kp2 = keypoints2t[i2];
            Point2f diff = kp2.center - kp1.center;

            if( norm(diff) < maxDist )
            {
                EllipticKeyPoint keypoint2a = EllipticKeyPoint( kp2.center, Scalar(fac*kp2.ellipse[0], fac*kp2.ellipse[1], fac*kp2.ellipse[2]) );
                //find the largest eigenvalue
                int maxx =  (int)ceil(( keypoint1a.boundingBox.width > (diff.x+keypoint2a.boundingBox.width)) ?
                                     keypoint1a.boundingBox.width : (diff.x+keypoint2a.boundingBox.width));
                int minx = (int)floor((-keypoint1a.boundingBox.width < (diff.x-keypoint2a.boundingBox.width)) ?
                                    -keypoint1a.boundingBox.width : (diff.x-keypoint2a.boundingBox.width));

                int maxy =  (int)ceil(( keypoint1a.boundingBox.height > (diff.y+keypoint2a.boundingBox.height)) ?
                                     keypoint1a.boundingBox.height : (diff.y+keypoint2a.boundingBox.height));
                int miny = (int)floor((-keypoint1a.boundingBox.height < (diff.y-keypoint2a.boundingBox.height)) ?
                                    -keypoint1a.boundingBox.height : (diff.y-keypoint2a.boundingBox.height));
                int mina = (maxx-minx) < (maxy-miny) ? (maxx-minx) : (maxy-miny) ;

                //compute the area
                float dr = (float)mina/50.f;
                int N = (int)floor((float)(maxx - minx) / dr);
                IntersectAreaCounter ac( dr, minx, miny, maxy, diff, keypoint1a.ellipse, keypoint2a.ellipse );
                parallel_reduce( BlockedRange(0, N+1), ac );
                if( ac.bna > 0 )
                {
                    float ov =  (float)ac.bna / (float)ac.bua;
                    if( ov >= minOverlap )
                        overlaps.push_back(SIdx(ov, (int)i1, (int)i2));
                }
            }
        }
    }

    std::sort( overlaps.begin(), overlaps.end() );

    typedef std::vector<SIdx>::iterator It;

    It pos = overlaps.begin();
    It end = overlaps.end();

    while(pos != end)
    {
        It prev = pos++;
        end = std::remove_if(pos, end, SIdx::UsedFinder(*prev));
    }
    overlaps.erase(pos, overlaps.end());
}

static void calculateRepeatability( const Mat& img1, const Mat& img2, const Mat& H1to2,
                                    const std::vector<KeyPoint>& _keypoints1, const std::vector<KeyPoint>& _keypoints2,
                                    float& repeatability, int& correspondencesCount,
                                    Mat* thresholdedOverlapMask=0  )
{
    std::vector<EllipticKeyPoint> keypoints1, keypoints2, keypoints1t, keypoints2t;
    EllipticKeyPoint::convert( _keypoints1, keypoints1 );
    EllipticKeyPoint::convert( _keypoints2, keypoints2 );

    // calculate projections of key points
    EllipticKeyPoint::calcProjection( keypoints1, H1to2, keypoints1t );
    Mat H2to1; invert(H1to2, H2to1);
    EllipticKeyPoint::calcProjection( keypoints2, H2to1, keypoints2t );

    float overlapThreshold;
    bool ifEvaluateDetectors = thresholdedOverlapMask == 0;
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

        thresholdedOverlapMask->create( (int)keypoints1.size(), (int)keypoints2t.size(), CV_8UC1 );
        thresholdedOverlapMask->setTo( Scalar::all(0) );
    }
    size_t size1 = keypoints1.size(), size2 = keypoints2t.size();
    size_t minCount = MIN( size1, size2 );

    // calculate overlap errors
    std::vector<SIdx> overlaps;
    computeOneToOneMatchedOverlaps( keypoints1, keypoints2t, ifEvaluateDetectors, overlaps, overlapThreshold/*min overlap*/ );

    correspondencesCount = -1;
    repeatability = -1.f;
    if( overlaps.empty() )
        return;

    if( ifEvaluateDetectors )
    {
        // regions one-to-one matching
        correspondencesCount = (int)overlaps.size();
        repeatability = minCount ? (float)correspondencesCount / minCount : -1;
    }
    else
    {
        for( size_t i = 0; i < overlaps.size(); i++ )
        {
            int y = overlaps[i].i1;
            int x = overlaps[i].i2;
            thresholdedOverlapMask->at<uchar>(y,x) = 1;
        }
    }
}

void cv::evaluateFeatureDetector( const Mat& img1, const Mat& img2, const Mat& H1to2,
                              std::vector<KeyPoint>* _keypoints1, std::vector<KeyPoint>* _keypoints2,
                              float& repeatability, int& correspCount,
                              const Ptr<FeatureDetector>& _fdetector )
{
    CV_INSTRUMENT_REGION()

    Ptr<FeatureDetector> fdetector(_fdetector);
    std::vector<KeyPoint> *keypoints1, *keypoints2, buf1, buf2;
    keypoints1 = _keypoints1 != 0 ? _keypoints1 : &buf1;
    keypoints2 = _keypoints2 != 0 ? _keypoints2 : &buf2;

    if( (keypoints1->empty() || keypoints2->empty()) && !fdetector )
        CV_Error( Error::StsBadArg, "fdetector must not be empty when keypoints1 or keypoints2 is empty" );

    if( keypoints1->empty() )
        fdetector->detect( img1, *keypoints1 );
    if( keypoints2->empty() )
        fdetector->detect( img2, *keypoints2 );

    calculateRepeatability( img1, img2, H1to2, *keypoints1, *keypoints2, repeatability, correspCount );
}

struct DMatchForEvaluation : public DMatch
{
    uchar isCorrect;
    DMatchForEvaluation( const DMatch &dm ) : DMatch( dm ), isCorrect(0) {}
};

static inline float recall( int correctMatchCount, int correspondenceCount )
{
    return correspondenceCount ? (float)correctMatchCount / (float)correspondenceCount : -1;
}

static inline float precision( int correctMatchCount, int falseMatchCount )
{
    return correctMatchCount + falseMatchCount ? (float)correctMatchCount / (float)(correctMatchCount + falseMatchCount) : -1;
}

void cv::computeRecallPrecisionCurve( const std::vector<std::vector<DMatch> >& matches1to2,
                                      const std::vector<std::vector<uchar> >& correctMatches1to2Mask,
                                      std::vector<Point2f>& recallPrecisionCurve )
{
    CV_INSTRUMENT_REGION()

    CV_Assert( matches1to2.size() == correctMatches1to2Mask.size() );

    std::vector<DMatchForEvaluation> allMatches;
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

float cv::getRecall( const std::vector<Point2f>& recallPrecisionCurve, float l_precision )
{
    CV_INSTRUMENT_REGION()

    int nearestPointIndex = getNearestPoint( recallPrecisionCurve, l_precision );

    float recall = -1.f;

    if( nearestPointIndex >= 0 )
        recall = recallPrecisionCurve[nearestPointIndex].y;

    return recall;
}

int cv::getNearestPoint( const std::vector<Point2f>& recallPrecisionCurve, float l_precision )
{
    CV_INSTRUMENT_REGION()

    int nearestPointIndex = -1;

    if( l_precision >= 0 && l_precision <= 1 )
    {
        float minDiff = FLT_MAX;
        for( size_t i = 0; i < recallPrecisionCurve.size(); i++ )
        {
            float curDiff = std::fabs(l_precision - recallPrecisionCurve[i].x);
            if( curDiff <= minDiff )
            {
                nearestPointIndex = (int)i;
                minDiff = curDiff;
            }
        }
    }

    return nearestPointIndex;
}
