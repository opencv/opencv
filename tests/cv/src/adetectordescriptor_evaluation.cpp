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

#include "cvtest.h"
#include <limits>
#include <cstdio>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

/****************************************************************************************\
*           Functions to evaluate affine covariant detectors and descriptors.            *
\****************************************************************************************/
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

    static Mat_<double> getSecondMomentsMatrix( const Scalar& _ellipse );
    Mat_<double> getSecondMomentsMatrix() const;

    void calcProjection( const Mat_<double>& H, EllipticKeyPoint& projection ) const;

    Point2f center;
    Scalar ellipse; // 3 elements a, b, c: ax^2+2bxy+cy^2=1
    Size_<float> axes; // half lenght of elipse axes
    Size_<float> boundingBox; // half sizes of bounding box
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

void calcEllipticKeyPointProjections( const vector<EllipticKeyPoint>& src, const Mat_<double>& H, vector<EllipticKeyPoint>& dst )
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

void transformToEllipticKeyPoints( const vector<KeyPoint>& src, vector<EllipticKeyPoint>& dst )
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

void transformToKeyPoints( const vector<EllipticKeyPoint>& src, vector<KeyPoint>& dst )
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

void calcKeyPointProjections( const vector<KeyPoint>& src, const Mat_<double>& H, vector<KeyPoint>& dst )
{
    if(  !src.empty() )
    {
        assert( !H.empty() && H.cols == 3 && H.rows == 3);
        dst.resize(src.size());
        vector<KeyPoint>::const_iterator srcIt = src.begin();
        vector<KeyPoint>::iterator       dstIt = dst.begin();
        for( ; srcIt != src.end(); ++srcIt, ++dstIt )
        {
            Point2f dstPt = applyHomography(H, srcIt->pt);

            float srcSize2 = srcIt->size * srcIt->size;
            Mat_<double> invM; invert(EllipticKeyPoint::getSecondMomentsMatrix( Scalar(1./srcSize2, 0., 1./srcSize2)), invM);
            Mat_<double> Aff; linearizeHomographyAt(H, srcIt->pt, Aff);
            Mat_<double> dstM; invert(Aff*invM*Aff.t(), dstM);
            Mat_<double> eval; eigen( dstM, eval );
            assert( eval(0,0) && eval(1,0) );
            float dstSize = pow(1./(eval(0,0)*eval(1,0)), 0.25);

            // TODO: check angle projection
            float srcAngleRad = srcIt->angle*CV_PI/180;
            Point2f vec1(cos(srcAngleRad), sin(srcAngleRad)), vec2;
            vec2.x = Aff(0,0)*vec1.x + Aff(0,1)*vec1.y;
            vec2.y = Aff(1,0)*vec1.x + Aff(0,1)*vec1.y;
            float dstAngleGrad = fastAtan2(vec2.y, vec2.x);

            *dstIt = KeyPoint( dstPt, dstSize, dstAngleGrad, srcIt->response, srcIt->octave, srcIt->class_id );
        }
    }
}

void filterKeyPointsByImageSize( vector<KeyPoint>& keypoints, const Size& imgSize )
{
    if( !keypoints.empty() )
    {
        vector<KeyPoint> filtered;
        filtered.reserve(keypoints.size());
        Rect r(0, 0, imgSize.width, imgSize.height);
        vector<KeyPoint>::const_iterator it = keypoints.begin();
        for( int i = 0; it != keypoints.end(); ++it, i++ )
            if( r.contains(it->pt) )
                filtered.push_back(*it);
        keypoints.assign(filtered.begin(), filtered.end());
    }
}

/*
 * calulate ovelap errors
 */
void overlap( const vector<EllipticKeyPoint>& keypoints1, const vector<EllipticKeyPoint>& keypoints2t, bool commonPart,
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
        if( !commonPart)
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
                    overlaps.ref(i1,i2) = 100.0*bna/bua;
            }
        }
    }
}

void filterEllipticKeyPointsByImageSize( vector<EllipticKeyPoint>& keypoints, const Size& imgSize )
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

void getEllipticKeyPointsInCommonPart( vector<EllipticKeyPoint>& keypoints1, vector<EllipticKeyPoint>& keypoints2,
                              vector<EllipticKeyPoint>& keypoints1t, vector<EllipticKeyPoint>& keypoints2t,
                              Size& imgSize1, const Size& imgSize2 )
{
    filterEllipticKeyPointsByImageSize( keypoints1, imgSize1 );
    filterEllipticKeyPointsByImageSize( keypoints1t, imgSize2 );
    filterEllipticKeyPointsByImageSize( keypoints2, imgSize2 );
    filterEllipticKeyPointsByImageSize( keypoints2t, imgSize1 );
}

void calculateRepeatability( const vector<EllipticKeyPoint>& _keypoints1, const vector<EllipticKeyPoint>& _keypoints2,
                             const Mat& img1, const Mat& img2, const Mat& H1to2,
                             float& repeatability, int& correspondencesCount,
                             SparseMat_<uchar>* thresholdedOverlapMask=0 )
{
    vector<EllipticKeyPoint> keypoints1( _keypoints1.begin(), _keypoints1.end() ),
                             keypoints2( _keypoints2.begin(), _keypoints2.end() ),
                             keypoints1t( keypoints1.size() ),
                             keypoints2t( keypoints2.size() );

    // calculate projections of key points
    calcEllipticKeyPointProjections( keypoints1, H1to2, keypoints1t );
    Mat H2to1; invert(H1to2, H2to1);
    calcEllipticKeyPointProjections( keypoints2, H2to1, keypoints2t );

    bool ifEvaluateDetectors = !thresholdedOverlapMask; // == commonPart
    float overlapThreshold;
    if( ifEvaluateDetectors )
    {
        overlapThreshold = 100.f - 40.f;

        // remove key points from outside of the common image part
        Size sz1 = img1.size(), sz2 = img2.size();
        getEllipticKeyPointsInCommonPart( keypoints1, keypoints2, keypoints1t, keypoints2t, sz1, sz2 );
    }
    else
    {
        overlapThreshold = 100.f - 50.f;
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
        repeatability = minCount ? (float)(correspondencesCount*100)/minCount : -1;
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
                    thresholdedOverlapMask->ref(y,x) = val;
            }
        }
    }
}


void evaluateDetectors( const vector<EllipticKeyPoint>& keypoints1, const vector<EllipticKeyPoint>& keypoints2,
                        const Mat& img1, const Mat& img2, const Mat& H1to2,
                        float& repeatability, int& correspCount )
{
    calculateRepeatability( keypoints1, keypoints2,
                            img1, img2, H1to2,
                            repeatability, correspCount );
}

inline float recall( int correctMatchCount, int correspondenceCount )
{
    return correspondenceCount ? (float)correctMatchCount / (float)correspondenceCount : -1;
}

inline float precision( int correctMatchCount, int falseMatchCount )
{
    return correctMatchCount + falseMatchCount ? (float)correctMatchCount / (float)(correctMatchCount + falseMatchCount) : -1;
}


struct DMatchForEvaluation : public DMatch
{
    int isCorrect;

    DMatchForEvaluation( const DMatch &dm )
    : DMatch( dm )
    {
    }
};


void evaluateDescriptors( const vector<EllipticKeyPoint>& keypoints1, const vector<EllipticKeyPoint>& keypoints2,
                          const vector<vector<DMatch> >& matches1to2, vector<DMatchForEvaluation> &allMatches,
                          const Mat& img1, const Mat& img2, const Mat& H1to2,
                          int &correctMatchCount, int &falseMatchCount, int& correspondenceCount )
{
    assert( !keypoints1.empty() && !keypoints2.empty() && !matches1to2.empty() );
    assert( keypoints1.size() == matches1to2.size() );

    float repeatability;
    int correspCount;
    SparseMat_<uchar> thresholdedOverlapMask; // thresholded allOverlapErrors
    calculateRepeatability( keypoints1, keypoints2,
                            img1, img2, H1to2,
                            repeatability, correspCount,
                            &thresholdedOverlapMask );
    correspondenceCount = thresholdedOverlapMask.nzcount();

    correctMatchCount = 0;
    falseMatchCount = 0;

    for( size_t i = 0; i < matches1to2.size(); i++ )
    {
        for( size_t j = 0;j < matches1to2[i].size(); j++ )
        {
        //if( matches1to2[i].match.indexTrain > 0 )
        //{
            DMatchForEvaluation match = matches1to2[i][j];
            match.isCorrect = thresholdedOverlapMask( match.indexQuery, match.indexTrain);
            if( match.isCorrect )
                correctMatchCount++;
            else
                falseMatchCount++;
            allMatches.push_back( match );
        //}
        //else
        //{
        //    matches1to2[i].isCorrect = -1;
        //}
        }
    }
}

/****************************************************************************************\
*                                  Detectors evaluation                                 *
\****************************************************************************************/
const int DATASETS_COUNT = 8;
const int TEST_CASE_COUNT = 5;

const string IMAGE_DATASETS_DIR = "detectors_descriptors_evaluation/images_datasets/";
const string DETECTORS_DIR = "detectors_descriptors_evaluation/detectors/";
const string DESCRIPTORS_DIR = "detectors_descriptors_evaluation/descriptors/";
const string KEYPOINTS_DIR = "detectors_descriptors_evaluation/keypoints_datasets/";

const string PARAMS_POSTFIX = "_params.xml";
const string RES_POSTFIX = "_res.xml";

const string REPEAT = "repeatability";
const string CORRESP_COUNT = "correspondence_count";

string DATASET_NAMES[DATASETS_COUNT] = { "bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall"};

string DEFAULT_PARAMS = "default";

string IS_ACTIVE_PARAMS = "isActiveParams";
string IS_SAVE_KEYPOINTS = "isSaveKeypoints";


class BaseQualityTest : public CvTest
{
public:
    BaseQualityTest( const char* _algName, const char* _testName, const char* _testFuncs ) :
            CvTest( _testName, _testFuncs ), algName(_algName)
    {
        //TODO: change this
        isWriteGraphicsData = true;
    }

protected:
    virtual string getRunParamsFilename() const = 0;
    virtual string getResultsFilename() const = 0;
    virtual string getPlotPath() const = 0;

    virtual void validQualityClear( int datasetIdx ) = 0;
    virtual void calcQualityClear( int datasetIdx ) = 0;
    virtual void validQualityCreate( int datasetIdx ) = 0;
    virtual bool isValidQualityEmpty( int datasetIdx ) const = 0;
    virtual bool isCalcQualityEmpty( int datasetIdx ) const = 0;

    void readAllDatasetsRunParams();
    virtual void readDatasetRunParams( FileNode& fn, int datasetIdx ) = 0;
    void writeAllDatasetsRunParams() const;
    virtual void writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const = 0;
    void setDefaultAllDatasetsRunParams();
    virtual void setDefaultDatasetRunParams( int datasetIdx ) = 0;
    virtual void readDefaultRunParams( FileNode &fn ) {};
    virtual void writeDefaultRunParams( FileStorage &fs ) const {};

    virtual void readResults();
    virtual void readResults( FileNode& fn, int datasetIdx, int caseIdx ) = 0;
    void writeResults() const;
    virtual void writeResults( FileStorage& fs, int datasetIdx, int caseIdx ) const = 0;

    bool readDataset( const string& datasetName, vector<Mat>& Hs, vector<Mat>& imgs );

    virtual void readAlgorithm( ) {};
    virtual void processRunParamsFile () {};
    virtual void runDatasetTest( const vector<Mat> &imgs, const vector<Mat> &Hs, int di, int &progress ) {};
    void run( int );

    virtual void processResults( int datasetIdx );
    virtual int processResults( int datasetIdx, int caseIdx ) = 0;
    virtual void processResults();
    virtual void writePlotData( int datasetIdx ) const {};
    virtual void writeAveragePlotData() const {};

    string algName;
    bool isWriteParams, isWriteResults, isWriteGraphicsData;
};

void BaseQualityTest::readAllDatasetsRunParams()
{
    string filename = getRunParamsFilename();
    FileStorage fs( filename, FileStorage::READ );
    if( !fs.isOpened() )
    {
        isWriteParams = true;
        setDefaultAllDatasetsRunParams();
        ts->printf(CvTS::LOG, "all runParams are default\n");
    }
    else
    {
        isWriteParams = false;
        FileNode topfn = fs.getFirstTopLevelNode();

        FileNode fn = topfn[DEFAULT_PARAMS];
        readDefaultRunParams(fn);

        for( int i = 0; i < DATASETS_COUNT; i++ )
        {
            FileNode fn = topfn[DATASET_NAMES[i]];
            if( fn.empty() )
            {
                ts->printf( CvTS::LOG, "%d-runParams is default\n", i);
                setDefaultDatasetRunParams(i);
            }
            else
                readDatasetRunParams(fn, i);
        }
    }
}

void BaseQualityTest::writeAllDatasetsRunParams() const
{
    string filename = getRunParamsFilename();
    FileStorage fs( filename, FileStorage::WRITE );
    if( fs.isOpened() )
    {
        fs << "run_params" << "{"; // top file node
        fs << DEFAULT_PARAMS << "{";
        writeDefaultRunParams(fs);
        fs << "}";
        for( int i = 0; i < DATASETS_COUNT; i++ )
        {
            fs << DATASET_NAMES[i] << "{";
            writeDatasetRunParams(fs, i);
            fs << "}";
        }
        fs << "}";
    }
    else
        ts->printf(CvTS::LOG, "file %s for writing run params can not be opened\n", filename.c_str() );
}

void BaseQualityTest::setDefaultAllDatasetsRunParams()
{
    for( int i = 0; i < DATASETS_COUNT; i++ )
        setDefaultDatasetRunParams(i);
}

bool BaseQualityTest::readDataset( const string& datasetName, vector<Mat>& Hs, vector<Mat>& imgs )
{
    Hs.resize( TEST_CASE_COUNT );
    imgs.resize( TEST_CASE_COUNT+1 );
    string dirname = string(ts->get_data_path()) + IMAGE_DATASETS_DIR + datasetName + "/";

    for( int i = 0; i < (int)Hs.size(); i++ )
    {
        stringstream filename; filename << "H1to" << i+2 << "p.xml";
        FileStorage fs( dirname + filename.str(), FileStorage::READ );
        if( !fs.isOpened() )
            return false;
        fs.getFirstTopLevelNode() >> Hs[i];
    }

    for( int i = 0; i < (int)imgs.size(); i++ )
    {
        stringstream filename; filename << "img" << i+1 << ".png";
        imgs[i] = imread( dirname + filename.str(), 0 );
        if( imgs[i].empty() )
            return false;
    }
    return true;
}

void BaseQualityTest::readResults()
{
    string filename = getResultsFilename();
    FileStorage fs( filename, FileStorage::READ );
    if( fs.isOpened() )
    {
        isWriteResults = false;
        FileNode topfn = fs.getFirstTopLevelNode();
        for( int di = 0; di < DATASETS_COUNT; di++ )
        {
            FileNode datafn = topfn[DATASET_NAMES[di]];
            if( datafn.empty() )
            {
                validQualityClear(di);
                ts->printf( CvTS::LOG, "results for %s dataset were not read\n",
                            DATASET_NAMES[di].c_str() );
            }
            else
            {
                validQualityCreate(di);
                for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
                {
                    stringstream ss; ss << "case" << ci;
                    FileNode casefn = datafn[ss.str()];
                    CV_Assert( !casefn.empty() );
                    readResults( casefn , di, ci );
                }
            }
        }
    }
    else
        isWriteResults = true;
}

void BaseQualityTest::writeResults() const
{
    string filename = getResultsFilename();;
    FileStorage fs( filename, FileStorage::WRITE );
    if( fs.isOpened() )
    {
        fs << "results" << "{";
        for( int di = 0; di < DATASETS_COUNT; di++ )
        {
            if( isCalcQualityEmpty(di) )
            {
                ts->printf(CvTS::LOG, "results on %s dataset were not write because of empty\n",
                    DATASET_NAMES[di].c_str());
            }
            else
            {
                fs << DATASET_NAMES[di] << "{";
                for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
                {
                    stringstream ss; ss << "case" << ci;
                    fs << ss.str() << "{";
                    writeResults( fs, di, ci );
                    fs << "}"; //ss.str()
                }
                fs << "}"; //DATASET_NAMES[di]
            }
        }
        fs << "}"; //results
    }
    else
        ts->printf(CvTS::LOG, "results were not written because file %s can not be opened\n", filename.c_str() );
}

void BaseQualityTest::processResults( int datasetIdx )
{
    if( isWriteGraphicsData )
        writePlotData( datasetIdx );
}

void BaseQualityTest::processResults()
{
    if( isWriteParams )
        writeAllDatasetsRunParams();

    if( isWriteGraphicsData )
        writeAveragePlotData();

    int res = CvTS::OK;
    if( isWriteResults )
        writeResults();
    else
    {
        for( int di = 0; di < DATASETS_COUNT; di++ )
        {
            if( isValidQualityEmpty(di) || isCalcQualityEmpty(di) )
                continue;

            ts->printf(CvTS::LOG, "\nDataset: %s\n", DATASET_NAMES[di].c_str() );

            for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
            {
                ts->printf(CvTS::LOG, "case%d\n", ci);
                int currRes = processResults( di, ci );
                res = currRes == CvTS::OK ? res : currRes;
            }
        }
    }

    if( res != CvTS::OK )
        ts->printf(CvTS::LOG, "BAD ACCURACY\n");
    ts->set_failed_test_info( res );
}

void BaseQualityTest::run ( int )
{
    readAlgorithm ();
    processRunParamsFile ();
    readResults();

    int notReadDatasets = 0;
    int progress = 0;

    FileStorage runParamsFS( getRunParamsFilename(), FileStorage::READ );
    isWriteParams = (! runParamsFS.isOpened());
    FileNode topfn = runParamsFS.getFirstTopLevelNode();
    FileNode defaultParams = topfn[DEFAULT_PARAMS];
    readDefaultRunParams (defaultParams);

    for(int di = 0; di < DATASETS_COUNT; di++ )
    {
        vector<Mat> imgs, Hs;
        if( !readDataset( DATASET_NAMES[di], Hs, imgs ) )
        {
            calcQualityClear (di);
            ts->printf( CvTS::LOG, "images or homography matrices of dataset named %s can not be read\n",
                        DATASET_NAMES[di].c_str());
            notReadDatasets++;
            continue;
        }

        FileNode fn = topfn[DATASET_NAMES[di]];
        readDatasetRunParams(fn, di);

        runDatasetTest (imgs, Hs, di, progress);
        processResults( di );
    }
    if( notReadDatasets == DATASETS_COUNT )
    {
        ts->printf(CvTS::LOG, "All datasets were not be read\n");
        ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
    }
    else
        processResults();
    runParamsFS.release();
}



class DetectorQualityTest : public BaseQualityTest
{
public:
    DetectorQualityTest( const char* _detectorName, const char* _testName ) :
            BaseQualityTest( _detectorName, _testName, "quality-of-detector" )
    {
        validQuality.resize(DATASETS_COUNT);
        calcQuality.resize(DATASETS_COUNT);
        isSaveKeypoints.resize(DATASETS_COUNT);
        isActiveParams.resize(DATASETS_COUNT);

        isSaveKeypointsDefault = false;
        isActiveParamsDefault = false;
    }

protected:
    using BaseQualityTest::readResults;
    using BaseQualityTest::writeResults;
    using BaseQualityTest::processResults;

    virtual string getRunParamsFilename() const;
    virtual string getResultsFilename() const;
    virtual string getPlotPath() const;

    virtual void validQualityClear( int datasetIdx );
    virtual void calcQualityClear( int datasetIdx );
    virtual void validQualityCreate( int datasetIdx );
    virtual bool isValidQualityEmpty( int datasetIdx ) const;
    virtual bool isCalcQualityEmpty( int datasetIdx ) const;

    virtual void readResults( FileNode& fn, int datasetIdx, int caseIdx );
    virtual void writeResults( FileStorage& fs, int datasetIdx, int caseIdx ) const;

    virtual void readDatasetRunParams( FileNode& fn, int datasetIdx );
    virtual void writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const;
    virtual void setDefaultDatasetRunParams( int datasetIdx );
    virtual void readDefaultRunParams( FileNode &fn );
    virtual void writeDefaultRunParams( FileStorage &fs ) const;

    virtual void writePlotData( int di ) const;
    virtual void writeAveragePlotData() const;

    void openToWriteKeypointsFile( FileStorage& fs, int datasetIdx );

    virtual void readAlgorithm( );
    virtual void processRunParamsFile () {};
    virtual void runDatasetTest( const vector<Mat> &imgs, const vector<Mat> &Hs, int di, int &progress );
    virtual int processResults( int datasetIdx, int caseIdx );

    Ptr<FeatureDetector> specificDetector;
    Ptr<FeatureDetector> defaultDetector;

    struct Quality
    {
        float repeatability;
        int correspondenceCount;
    };
    vector<vector<Quality> > validQuality;
    vector<vector<Quality> > calcQuality;

    vector<bool> isSaveKeypoints;
    vector<bool> isActiveParams;

    bool isSaveKeypointsDefault;
    bool isActiveParamsDefault;
};

string DetectorQualityTest::getRunParamsFilename() const
{
     return string(ts->get_data_path()) + DETECTORS_DIR + algName + PARAMS_POSTFIX;
}

string DetectorQualityTest::getResultsFilename() const
{
    return string(ts->get_data_path()) + DETECTORS_DIR + algName + RES_POSTFIX;
}

string DetectorQualityTest::getPlotPath() const
{
    return string(ts->get_data_path()) + DETECTORS_DIR + "plots/";
}

void DetectorQualityTest::validQualityClear( int datasetIdx )
{
    validQuality[datasetIdx].clear();
}

void DetectorQualityTest::calcQualityClear( int datasetIdx )
{
    calcQuality[datasetIdx].clear();
}

void DetectorQualityTest::validQualityCreate( int datasetIdx )
{
    validQuality[datasetIdx].resize(TEST_CASE_COUNT);
}

bool DetectorQualityTest::isValidQualityEmpty( int datasetIdx ) const
{
    return validQuality[datasetIdx].empty();
}

bool DetectorQualityTest::isCalcQualityEmpty( int datasetIdx ) const
{
    return calcQuality[datasetIdx].empty();
}

void DetectorQualityTest::readResults( FileNode& fn, int datasetIdx, int caseIdx )
{
    validQuality[datasetIdx][caseIdx].repeatability = fn[REPEAT];
    validQuality[datasetIdx][caseIdx].correspondenceCount = fn[CORRESP_COUNT];
}

void DetectorQualityTest::writeResults( FileStorage& fs, int datasetIdx, int caseIdx ) const
{
    fs << REPEAT << calcQuality[datasetIdx][caseIdx].repeatability;
    fs << CORRESP_COUNT << calcQuality[datasetIdx][caseIdx].correspondenceCount;
}

void DetectorQualityTest::readDefaultRunParams (FileNode &fn)
{
    if (! fn.empty() )
    {
        isSaveKeypointsDefault = (int)fn[IS_SAVE_KEYPOINTS] != 0;
        defaultDetector->read (fn);
    }
}

void DetectorQualityTest::writeDefaultRunParams (FileStorage &fs) const
{
    fs << IS_SAVE_KEYPOINTS << isSaveKeypointsDefault;
    defaultDetector->write (fs);
}

void DetectorQualityTest::readDatasetRunParams( FileNode& fn, int datasetIdx )
{
    isActiveParams[datasetIdx] = (int)fn[IS_ACTIVE_PARAMS] != 0;
    if (isActiveParams[datasetIdx])
    {
        isSaveKeypoints[datasetIdx] = (int)fn[IS_SAVE_KEYPOINTS] != 0;
        specificDetector->read (fn);
    }
    else
    {
        setDefaultDatasetRunParams(datasetIdx);
    }
}

void DetectorQualityTest::writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const
{
    fs << IS_ACTIVE_PARAMS << isActiveParams[datasetIdx];
    fs << IS_SAVE_KEYPOINTS << isSaveKeypoints[datasetIdx];
    defaultDetector->write (fs);
}

void DetectorQualityTest::setDefaultDatasetRunParams( int datasetIdx )
{
    isSaveKeypoints[datasetIdx] = isSaveKeypointsDefault;
    isActiveParams[datasetIdx] = isActiveParamsDefault;
}

void DetectorQualityTest::writePlotData(int di ) const
{
    int imgXVals[] = { 2, 3, 4, 5, 6 }; // if scale, blur or light changes
    int viewpointXVals[] = { 20, 30, 40, 50, 60 }; // if viewpoint changes
    int jpegXVals[] = { 60, 80, 90, 95, 98 }; // if jpeg compression

    int* xVals = 0;
    if( !DATASET_NAMES[di].compare("ubc") )
    {
        xVals = jpegXVals;
    }
    else if( !DATASET_NAMES[di].compare("graf") || !DATASET_NAMES[di].compare("wall") )
    {
        xVals = viewpointXVals;
    }
    else
        xVals = imgXVals;

    stringstream rFilename, cFilename;
    rFilename << getPlotPath() << algName << "_" << DATASET_NAMES[di]  << "_repeatability.csv";
    cFilename << getPlotPath() << algName << "_" << DATASET_NAMES[di]  << "_correspondenceCount.csv";
    ofstream rfile(rFilename.str().c_str()), cfile(cFilename.str().c_str());
    for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
    {
        rfile << xVals[ci] << ", " << calcQuality[di][ci].repeatability << endl;
        cfile << xVals[ci] << ", " << calcQuality[di][ci].correspondenceCount << endl;
    }
}

void DetectorQualityTest::writeAveragePlotData() const
{
    stringstream rFilename, cFilename;
    rFilename << getPlotPath() << algName << "_average_repeatability.csv";
    cFilename << getPlotPath() << algName << "_average_correspondenceCount.csv";
    ofstream rfile(rFilename.str().c_str()), cfile(cFilename.str().c_str());
    float avRep = 0, avCorCount = 0;
    for( int di = 0; di < DATASETS_COUNT; di++ )
    {
        for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
        {
            avRep += calcQuality[di][ci].repeatability;
            avCorCount += calcQuality[di][ci].correspondenceCount;
        }
    }
    avRep /= DATASETS_COUNT*TEST_CASE_COUNT;
    avCorCount /= DATASETS_COUNT*TEST_CASE_COUNT;
    rfile << algName << ", " << avRep << endl;
    cfile << algName << ", " << cvRound(avCorCount) << endl;
}

void DetectorQualityTest::openToWriteKeypointsFile( FileStorage& fs, int datasetIdx )
{
    string filename = string(ts->get_data_path()) + KEYPOINTS_DIR + algName + "_"+
                      DATASET_NAMES[datasetIdx] + ".xml.gz" ;

    fs.open(filename, FileStorage::WRITE);
    if( !fs.isOpened() )
        ts->printf( CvTS::LOG, "keypoints can not be written in file %s because this file can not be opened\n",
                    filename.c_str());
}

inline void writeKeypoints( FileStorage& fs, const vector<KeyPoint>& keypoints, int imgIdx )
{
    if( fs.isOpened() )
    {
        stringstream imgName; imgName << "img" << imgIdx;
        write( fs, imgName.str(), keypoints );
    }
}

inline void readKeypoints( FileStorage& fs, vector<KeyPoint>& keypoints, int imgIdx )
{
    assert( fs.isOpened() );
    stringstream imgName; imgName << "img" << imgIdx;
    read( fs[imgName.str()], keypoints);
}

void DetectorQualityTest::readAlgorithm ()
{
    defaultDetector = createDetector( algName );
    specificDetector = createDetector( algName );
    if( defaultDetector == 0 )
    {
        ts->printf(CvTS::LOG, "Algorithm can not be read\n");
        ts->set_failed_test_info( CvTS::FAIL_GENERIC);
    }
}

void DetectorQualityTest::runDatasetTest (const vector<Mat> &imgs, const vector<Mat> &Hs, int di, int &progress)
{
    Ptr<FeatureDetector> detector = isActiveParams[di] ? specificDetector : defaultDetector;
    FileStorage keypontsFS;
    if( isSaveKeypoints[di] )
        openToWriteKeypointsFile( keypontsFS, di );

    calcQuality[di].resize(TEST_CASE_COUNT);

    vector<KeyPoint> keypoints1; vector<EllipticKeyPoint> ekeypoints1;

    detector->detect( imgs[0], keypoints1 );
    writeKeypoints( keypontsFS, keypoints1, 0);
    transformToEllipticKeyPoints( keypoints1, ekeypoints1 );
    int progressCount = DATASETS_COUNT*TEST_CASE_COUNT;
    for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
    {
        progress = update_progress( progress, di*TEST_CASE_COUNT + ci, progressCount, 0 );
        vector<KeyPoint> keypoints2;
        detector->detect( imgs[ci+1], keypoints2 );
        writeKeypoints( keypontsFS, keypoints2, ci+1);
        vector<EllipticKeyPoint> ekeypoints2;
        transformToEllipticKeyPoints( keypoints2, ekeypoints2 );
        evaluateDetectors( ekeypoints1, ekeypoints2, imgs[0], imgs[ci], Hs[ci],
                           calcQuality[di][ci].repeatability, calcQuality[di][ci].correspondenceCount );
    }
}

void testLog( CvTS* ts, bool isBadAccuracy )
{
    if( isBadAccuracy )
        ts->printf(CvTS::LOG, " bad accuracy\n");
    else
        ts->printf(CvTS::LOG, "\n");
}

int DetectorQualityTest::processResults( int datasetIdx, int caseIdx )
{
    int res = CvTS::OK;

    Quality valid = validQuality[datasetIdx][caseIdx], calc = calcQuality[datasetIdx][caseIdx];

    bool isBadAccuracy;
    int countEps = 1;
    const float rltvEps = 0.001;
    ts->printf(CvTS::LOG, "%s: calc=%f, valid=%f", REPEAT.c_str(), calc.repeatability, valid.repeatability );
    isBadAccuracy = valid.repeatability - calc.repeatability > rltvEps;
    testLog( ts, isBadAccuracy );
    res = isBadAccuracy ? CvTS::FAIL_BAD_ACCURACY : res;

    ts->printf(CvTS::LOG, "%s: calc=%d, valid=%d", CORRESP_COUNT.c_str(), calc.correspondenceCount, valid.correspondenceCount );
    isBadAccuracy = valid.correspondenceCount - calc.correspondenceCount > countEps;
    testLog( ts, isBadAccuracy );
    res = isBadAccuracy ? CvTS::FAIL_BAD_ACCURACY : res;
    return res;
}

//DetectorQualityTest fastDetectorQuality = DetectorQualityTest( "FAST", "quality-detector-fast" );
//DetectorQualityTest gfttDetectorQuality = DetectorQualityTest( "GFTT", "quality-detector-gftt" );
//DetectorQualityTest harrisDetectorQuality = DetectorQualityTest( "HARRIS", "quality-detector-harris" );
//DetectorQualityTest mserDetectorQuality = DetectorQualityTest( "MSER", "quality-detector-mser" );
//DetectorQualityTest starDetectorQuality = DetectorQualityTest( "STAR", "quality-detector-star" );
//DetectorQualityTest siftDetectorQuality = DetectorQualityTest( "SIFT", "quality-detector-sift" );
//DetectorQualityTest surfDetectorQuality = DetectorQualityTest( "SURF", "quality-detector-surf" );

/****************************************************************************************\
*                                  Descriptors evaluation                                 *
\****************************************************************************************/

const string RECALL = "recall";
const string PRECISION = "precision";

const string KEYPOINTS_FILENAME = "keypointsFilename";
const string PROJECT_KEYPOINTS_FROM_1IMAGE = "projectKeypointsFrom1Image";
const string MATCH_FILTER = "matchFilter";
const string RUN_PARAMS_IS_IDENTICAL = "runParamsIsIdentical";

const string ONE_WAY_TRAIN_DIR = "detectors_descriptors_evaluation/one_way_train_images/";
const string ONE_WAY_IMAGES_LIST = "one_way_train_images.txt";

class DescriptorQualityTest : public BaseQualityTest
{
public:
    enum{ NO_MATCH_FILTER = 0 };
    DescriptorQualityTest( const char* _descriptorName, const char* _testName, const char* _matcherName = 0 ) :
            BaseQualityTest( _descriptorName, _testName, "quality-of-descriptor" )
    {
        validQuality.resize(DATASETS_COUNT);
        calcQuality.resize(DATASETS_COUNT);
        calcDatasetQuality.resize(DATASETS_COUNT);
        commRunParams.resize(DATASETS_COUNT);

        commRunParamsDefault.projectKeypointsFrom1Image = true;
        commRunParamsDefault.matchFilter = NO_MATCH_FILTER;
        commRunParamsDefault.isActiveParams = false;

        if( _matcherName )
            matcherName = _matcherName;
    }

protected:
    using BaseQualityTest::readResults;
    using BaseQualityTest::writeResults;
    using BaseQualityTest::processResults;

    virtual string getRunParamsFilename() const;
    virtual string getResultsFilename() const;
    virtual string getPlotPath() const;

    virtual void validQualityClear( int datasetIdx );
    virtual void calcQualityClear( int datasetIdx );
    virtual void validQualityCreate( int datasetIdx );
    virtual bool isValidQualityEmpty( int datasetIdx ) const;
    virtual bool isCalcQualityEmpty( int datasetIdx ) const;

    virtual void readResults( FileNode& fn, int datasetIdx, int caseIdx );
    virtual void writeResults( FileStorage& fs, int datasetIdx, int caseIdx ) const;

    virtual void readDatasetRunParams( FileNode& fn, int datasetIdx ); //
    virtual void writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const;
    virtual void setDefaultDatasetRunParams( int datasetIdx );
    virtual void readDefaultRunParams( FileNode &fn );
    virtual void writeDefaultRunParams( FileStorage &fs ) const;

    virtual void readAlgorithm( );
    virtual void processRunParamsFile () {};
    virtual void runDatasetTest( const vector<Mat> &imgs, const vector<Mat> &Hs, int di, int &progress );

    virtual int processResults( int datasetIdx, int caseIdx );

    virtual void writePlotData( int di ) const;
    void calculatePlotData( vector<DMatchForEvaluation> &allMatches, int allCorrespCount, int di );

    struct Quality
    {
        float recall;
        float precision;
    };
    vector<vector<Quality> > validQuality;
    vector<vector<Quality> > calcQuality;
    vector<vector<Quality> > calcDatasetQuality;

    struct CommonRunParams
    {
        string keypontsFilename;
        bool projectKeypointsFrom1Image;
        int matchFilter; // not used now
        bool isActiveParams;
    };
    vector<CommonRunParams> commRunParams;

    Ptr<GenericDescriptorMatch> specificDescMatch;
    Ptr<GenericDescriptorMatch> defaultDescMatch;

    CommonRunParams commRunParamsDefault;
    string matcherName;
};

string DescriptorQualityTest::getRunParamsFilename() const
{
    return string(ts->get_data_path()) + DESCRIPTORS_DIR + algName + PARAMS_POSTFIX;
}

string DescriptorQualityTest::getResultsFilename() const
{
    return string(ts->get_data_path()) + DESCRIPTORS_DIR + algName + RES_POSTFIX;
}

string DescriptorQualityTest::getPlotPath() const
{
    return string(ts->get_data_path()) + DESCRIPTORS_DIR + "plots/";
}

void DescriptorQualityTest::validQualityClear( int datasetIdx )
{
    validQuality[datasetIdx].clear();
}

void DescriptorQualityTest::calcQualityClear( int datasetIdx )
{
    calcQuality[datasetIdx].clear();
}

void DescriptorQualityTest::validQualityCreate( int datasetIdx )
{
    validQuality[datasetIdx].resize(TEST_CASE_COUNT);
}

bool DescriptorQualityTest::isValidQualityEmpty( int datasetIdx ) const
{
    return validQuality[datasetIdx].empty();
}

bool DescriptorQualityTest::isCalcQualityEmpty( int datasetIdx ) const
{
    return calcQuality[datasetIdx].empty();
}

void DescriptorQualityTest::readResults( FileNode& fn, int datasetIdx, int caseIdx )
{
    validQuality[datasetIdx][caseIdx].recall = fn[RECALL];
    validQuality[datasetIdx][caseIdx].precision = fn[PRECISION];
}

void DescriptorQualityTest::writeResults( FileStorage& fs, int datasetIdx, int caseIdx ) const
{
    fs << RECALL << calcQuality[datasetIdx][caseIdx].recall;
    fs << PRECISION << calcQuality[datasetIdx][caseIdx].precision;
}

void DescriptorQualityTest::readDefaultRunParams (FileNode &fn)
{
    if (! fn.empty() )
    {
        commRunParamsDefault.projectKeypointsFrom1Image = (int)fn[PROJECT_KEYPOINTS_FROM_1IMAGE] != 0;
        commRunParamsDefault.matchFilter = (int)fn[MATCH_FILTER];
        defaultDescMatch->read (fn);
    }
}

void DescriptorQualityTest::writeDefaultRunParams (FileStorage &fs) const
{
    fs << PROJECT_KEYPOINTS_FROM_1IMAGE << commRunParamsDefault.projectKeypointsFrom1Image;
    fs << MATCH_FILTER << commRunParamsDefault.matchFilter;
    defaultDescMatch->write (fs);
}

void DescriptorQualityTest::readDatasetRunParams( FileNode& fn, int datasetIdx )
{
    commRunParams[datasetIdx].isActiveParams = (int)fn[IS_ACTIVE_PARAMS];
    if (commRunParams[datasetIdx].isActiveParams)
    {
        commRunParams[datasetIdx].keypontsFilename = (string)fn[KEYPOINTS_FILENAME];
        commRunParams[datasetIdx].projectKeypointsFrom1Image = (int)fn[PROJECT_KEYPOINTS_FROM_1IMAGE] != 0;
        commRunParams[datasetIdx].matchFilter = (int)fn[MATCH_FILTER];
        specificDescMatch->read (fn);
    }
    else
    {
        setDefaultDatasetRunParams(datasetIdx);
    }
}

void DescriptorQualityTest::writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const
{
    fs << IS_ACTIVE_PARAMS << commRunParams[datasetIdx].isActiveParams;
    fs << KEYPOINTS_FILENAME << commRunParams[datasetIdx].keypontsFilename;
    fs << PROJECT_KEYPOINTS_FROM_1IMAGE << commRunParams[datasetIdx].projectKeypointsFrom1Image;
    fs << MATCH_FILTER << commRunParams[datasetIdx].matchFilter;

    defaultDescMatch->write (fs);
}

void DescriptorQualityTest::setDefaultDatasetRunParams( int datasetIdx )
{
    commRunParams[datasetIdx] = commRunParamsDefault;
    commRunParams[datasetIdx].keypontsFilename = "surf_" + DATASET_NAMES[datasetIdx] + ".xml.gz";
}

void DescriptorQualityTest::writePlotData( int di ) const
{
    stringstream filename;
    filename << getPlotPath() << algName << "_" << DATASET_NAMES[di] << ".csv";
    FILE *file = fopen (filename.str().c_str(), "w");
    size_t size = calcDatasetQuality[di].size();
    for (size_t i=0;i<size;i++)
    {
        fprintf( file, "%f, %f\n", 1 - calcDatasetQuality[di][i].precision, calcDatasetQuality[di][i].recall);
    }
    fclose( file );
}

void DescriptorQualityTest::readAlgorithm( )
{
    defaultDescMatch = createGenericDescriptorMatch( algName );
    specificDescMatch = createGenericDescriptorMatch( algName );

    if( defaultDescMatch == 0 )
    {
        DescriptorExtractor *extractor = createDescriptorExtractor( algName );
        DescriptorMatcher *matcher = createDescriptorMatcher( matcherName );
        defaultDescMatch = new VectorDescriptorMatch<DescriptorExtractor, DescriptorMatcher >( extractor, matcher );
        specificDescMatch = new VectorDescriptorMatch<DescriptorExtractor, DescriptorMatcher >( extractor, matcher );

        if( extractor == 0 || matcher == 0 )
        {
            ts->printf(CvTS::LOG, "Algorithm can not be read\n");
            ts->set_failed_test_info( CvTS::FAIL_GENERIC);
        }
    }
}

void DescriptorQualityTest::calculatePlotData( vector<DMatchForEvaluation> &allMatches, int allCorrespCount, int di )
{
    std::sort( allMatches.begin(), allMatches.end() );

    //calcDatasetQuality[di].resize( allMatches.size() );
    calcDatasetQuality[di].clear();
    int correctMatchCount = 0, falseMatchCount = 0;
    const float sparsePlotBound = 0.1;
    const int npoints = 10000;
    int step = 1 + allMatches.size() / npoints;
    const float resultPrecision = 0.5;
    bool isResultCalculated = false;
    for( size_t i=0;i<allMatches.size();i++)
    {
        if( allMatches[i].isCorrect )
            correctMatchCount++;
        else
            falseMatchCount++;

        if( precision( correctMatchCount, falseMatchCount ) >= sparsePlotBound || (i % step == 0) )
        {
            Quality quality;
            quality.recall = recall( correctMatchCount, allCorrespCount );
            quality.precision = precision( correctMatchCount, falseMatchCount );

            calcDatasetQuality[di].push_back( quality );

            if( !isResultCalculated && quality.precision < resultPrecision)
            {
                for(int ci=0;ci<TEST_CASE_COUNT;ci++)
                {
                    calcQuality[di][ci].recall = quality.recall;
                    calcQuality[di][ci].precision = quality.precision;
                }
                isResultCalculated = true;
            }
        }
    }

    Quality quality;
    quality.recall = recall( correctMatchCount, allCorrespCount );
    quality.precision = precision( correctMatchCount, falseMatchCount );

    calcDatasetQuality[di].push_back( quality );

}

void DescriptorQualityTest::runDatasetTest (const vector<Mat> &imgs, const vector<Mat> &Hs, int di, int &progress)
{
    FileStorage keypontsFS( string(ts->get_data_path()) + KEYPOINTS_DIR + commRunParams[di].keypontsFilename,
                                    FileStorage::READ );
    if( !keypontsFS.isOpened())
    {
       calcQuality[di].clear();
       ts->printf( CvTS::LOG, "keypoints from file %s can not be read\n", commRunParams[di].keypontsFilename.c_str() );
       return;
    }

    Ptr<GenericDescriptorMatch> descMatch = commRunParams[di].isActiveParams ? specificDescMatch : defaultDescMatch;
    calcQuality[di].resize(TEST_CASE_COUNT);

    vector<KeyPoint> keypoints1; vector<EllipticKeyPoint> ekeypoints1;
    readKeypoints( keypontsFS, keypoints1, 0);
    transformToEllipticKeyPoints( keypoints1, ekeypoints1 );

    int progressCount = DATASETS_COUNT*TEST_CASE_COUNT;
    vector<DMatchForEvaluation> allMatches;

    int allCorrespCount = 0;
    for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
    {
        progress = update_progress( progress, di*TEST_CASE_COUNT + ci, progressCount, 0 );

        vector<KeyPoint> keypoints2;
        vector<EllipticKeyPoint> ekeypoints2;
        if( commRunParams[di].projectKeypointsFrom1Image )
        {
            // TODO need to test function calcKeyPointProjections
            calcKeyPointProjections( keypoints1, Hs[ci], keypoints2 );
            filterKeyPointsByImageSize( keypoints2,  imgs[ci+1].size() );
        }
        else
            readKeypoints( keypontsFS, keypoints2, ci+1 );
        transformToEllipticKeyPoints( keypoints2, ekeypoints2 );
        descMatch->add( imgs[ci+1], keypoints2 );
        vector<vector<DMatch> > matches1to2;
        //TODO: use more sophisticated strategy to choose threshold
        descMatch->match( imgs[0], keypoints1, matches1to2, std::numeric_limits<float>::max() );

        // TODO if( commRunParams[di].matchFilter )
        int correspCount;
        int correctMatchCount = 0, falseMatchCount = 0;
        evaluateDescriptors( ekeypoints1, ekeypoints2, matches1to2, allMatches, imgs[0], imgs[ci+1], Hs[ci],
                             correctMatchCount, falseMatchCount, correspCount );

        allCorrespCount += correspCount;

        descMatch->clear ();
    }

    calculatePlotData( allMatches, allCorrespCount, di );
}

int DescriptorQualityTest::processResults( int datasetIdx, int caseIdx )
{
    int res = CvTS::OK;
    Quality valid = validQuality[datasetIdx][caseIdx], calc = calcQuality[datasetIdx][caseIdx];

    bool isBadAccuracy;
    const float rltvEps = 0.001;
    ts->printf(CvTS::LOG, "%s: calc=%f, valid=%f", RECALL.c_str(), calc.recall, valid.recall );
    isBadAccuracy = valid.recall - calc.recall > rltvEps;
    testLog( ts, isBadAccuracy );
    res = isBadAccuracy ? CvTS::FAIL_BAD_ACCURACY : res;

    ts->printf(CvTS::LOG, "%s: calc=%f, valid=%f", PRECISION.c_str(), calc.precision, valid.precision );
    isBadAccuracy = valid.precision - calc.precision > rltvEps;
    testLog( ts, isBadAccuracy );
    res = isBadAccuracy ? CvTS::FAIL_BAD_ACCURACY : res;

    return res;
}

//DescriptorQualityTest siftDescriptorQuality = DescriptorQualityTest( "SIFT", "quality-descriptor-sift", "BruteForce" );
//DescriptorQualityTest surfDescriptorQuality = DescriptorQualityTest( "SURF", "quality-descriptor-surf", "BruteForce" );
//DescriptorQualityTest siftL1DescriptorQuality = DescriptorQualityTest( "SIFT", "quality-descriptor-sift-L1", "BruteForce-L1" );
//DescriptorQualityTest surfL1DescriptorQuality = DescriptorQualityTest( "SURF", "quality-descriptor-surf-L1", "BruteForce-L1" );

//--------------------------------- One Way descriptor test --------------------------------------------
class OneWayDescriptorQualityTest : public DescriptorQualityTest
{
public:
    OneWayDescriptorQualityTest() :
        DescriptorQualityTest("ONEWAY", "quality-descriptor-one-way")
    {
    }
protected:
    virtual void processRunParamsFile ();
    virtual void writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const;
};

void OneWayDescriptorQualityTest::processRunParamsFile ()
{
    string filename = getRunParamsFilename();
    FileStorage fs = FileStorage (filename, FileStorage::READ);
    FileNode fn = fs.getFirstTopLevelNode();
    fn = fn[DEFAULT_PARAMS];

    string pcaFilename = string(ts->get_data_path()) + (string)fn["pcaFilename"];
    string trainPath = string(ts->get_data_path()) + (string)fn["trainPath"];
    string trainImagesList = (string)fn["trainImagesList"];
    int patch_width = fn["patchWidth"];
    int patch_height = fn["patchHeight"];
    Size patchSize = cvSize (patch_width, patch_height);
    int poseCount = fn["poseCount"];

    if (trainImagesList.length () == 0 )
    {
        return;
        fs.release ();
    }
    fs.release ();

    readAllDatasetsRunParams();

    OneWayDescriptorBase *base = new OneWayDescriptorBase(patchSize, poseCount, pcaFilename,
                                               trainPath, trainImagesList);

    OneWayDescriptorMatch *match = new OneWayDescriptorMatch ();
    match->initialize( OneWayDescriptorMatch::Params (), base );
    defaultDescMatch = match;
    writeAllDatasetsRunParams();
}

void OneWayDescriptorQualityTest::writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const
{
    fs << IS_ACTIVE_PARAMS << commRunParams[datasetIdx].isActiveParams;
    fs << KEYPOINTS_FILENAME << commRunParams[datasetIdx].keypontsFilename;
    fs << PROJECT_KEYPOINTS_FROM_1IMAGE << commRunParams[datasetIdx].projectKeypointsFrom1Image;
    fs << MATCH_FILTER << commRunParams[datasetIdx].matchFilter;
}


//OneWayDescriptorQualityTest oneWayDescriptorQuality;
//DescriptorQualityTest fernDescriptorQualityTest( "FERN", "quality-descriptor-fern");
//DescriptorQualityTest calonderDescriptorQualityTest( "CALONDER", "quality-descriptor-calonder");
