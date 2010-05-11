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

using namespace std;
using namespace cv;

inline Point2f applyHomography( const Mat_<double>& H, const Point2f& pt )
{
    double w = 1./(H(2,0)*pt.x + H(2,1)*pt.y + H(2,2));
    return Point2f( (H(0,0)*pt.x + H(0,1)*pt.y + H(0,2))*w, (H(1,0)*pt.x + H(1,1)*pt.y + H(1,2))*w );
}

inline void linearizeHomographyAt( const Mat_<double>& H, const Point2f& pt, Mat_<double>& A )
{
    A.create(2,2);
    double p1 = H(0,0)*pt.x + H(0,1)*pt.y + H(0,2),
           p2 = H(1,0)*pt.x + H(1,1)*pt.y + H(1,2),
           p3 = H(2,0)*pt.x + H(2,1)*pt.y + H(2,2),
           p3_2 = p3*p3;
    A(0,0) = H(0,0)/p3 - p1*H(2,0)/p3_2; // fxdx
    A(0,1) = H(0,1)/p3 - p1*H(2,1)/p3_2; // fxdy

    A(1,0) = H(1,0)/p3 - p2*H(2,0)/p3_2; // fydx
    A(1,1) = H(1,1)/p3 - p2*H(2,1)/p3_2; // fydx
}

//----------------------------------- Repeatability ---------------------------------------------------

// Find the key points located in the part of the scene present in both images
// and project keypoints2 on img1
void getCommonKeyPointsOnImg1( const Mat& img1, const Mat img2, const Mat& H12,
                               const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
                               vector<KeyPoint>& ckeypoints1, vector<KeyPoint>& hckeypoints2,
                               bool isAffineInvariant )
{
    assert( !img1.empty() && !img2.empty() );
    assert( !H12.empty() && H12.cols==3 && H12.rows==3 && H12.type()==CV_64FC1 );
    ckeypoints1.clear();
    hckeypoints2.clear();

    Rect r1(0, 0, img1.cols, img1.rows), r2(0, 0, img2.cols, img2.rows);
    Mat H21; invert( H12, H21 );

    for( vector<KeyPoint>::const_iterator it = keypoints1.begin();
                 it != keypoints1.end(); ++it )
    {
        if( r2.contains(applyHomography(H12, it->pt)) )
            ckeypoints1.push_back(*it);
    }
    for( vector<KeyPoint>::const_iterator it = keypoints2.begin();
                 it != keypoints2.end(); ++it )
    {
        Point2f pt = applyHomography(H21, it->pt);
        if( r1.contains(pt) )
        {
            KeyPoint kp = *it;
            kp.pt = pt;
            if( isAffineInvariant )
                assert(0);
            else // scale invariant
            {
                Mat_<double> A, eval;
                linearizeHomographyAt(H21, it->pt, A);
                eigen(A, eval);
                assert( eval.type()==CV_64FC1 && eval.cols==1 && eval.rows==2 );
                kp.size *= sqrt(eval(0,0) * eval(1,0)) /*scale from linearized homography matrix*/;
            }
            hckeypoints2.push_back(kp);
        }
    }
}

// Locations p1 and p2 are repeated if ||p1 - H21*p2|| < 1.5 pixels.
// Regions are repeated if Es < 0.4 (Es differs for scale invariant and affine invarian detectors).
// For more details see "Scale&Affine Invariant Interest Point Detectors", Mikolajczyk, Schmid.
void repeatability( const Mat& img1, const Mat img2, const Mat& H12,
                    const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
                    int& repeatingLocationCount, float& repeatingLocationRltv,
                    int& repeatingRegionCount, float& repeatingRegionRltv,
                    bool isAffineInvariant )
{
    const double locThreshold = 1.5,
                 regThreshold = 0.4;
    assert( !img1.empty() && !img2.empty() );
    assert( !H12.empty() && H12.cols==3 && H12.rows==3 && H12.type()==CV_64FC1 );

    Mat H21; invert( H12, H21 );

    vector<KeyPoint> ckeypoints1, hckeypoints2;
    getCommonKeyPointsOnImg1( img1, img2, H12, keypoints1, keypoints2, ckeypoints1, hckeypoints2, false );

    vector<KeyPoint> *smallKPSet, *bigKPSet;
    if( ckeypoints1.size() < hckeypoints2.size() )
    {
        smallKPSet = &ckeypoints1;
        bigKPSet = &hckeypoints2;
    }
    else
    {
        smallKPSet = &hckeypoints2;
        bigKPSet = &ckeypoints1;
    }

    if( smallKPSet->size() == 0 )
    {
        repeatingLocationCount = repeatingRegionCount = -1;
        repeatingLocationRltv = repeatingRegionRltv = -1.f;
    }
    else
    {
        vector<bool> matchedMask( bigKPSet->size(), false);
        repeatingLocationCount = repeatingRegionCount = 0;
        for( vector<KeyPoint>::const_iterator skpIt = smallKPSet->begin(); skpIt != smallKPSet->end(); ++skpIt )
        {
            int nearestIdx = -1, bkpIdx = 0;
            double minDist = numeric_limits<double>::max();
            vector<KeyPoint>::const_iterator nearestBkp;
            for( vector<KeyPoint>::const_iterator bkpIt = bigKPSet->begin(); bkpIt != bigKPSet->end(); ++bkpIt, bkpIdx++ )
            {
                if( !matchedMask[bkpIdx] )
                {
                    Point p1(cvRound(skpIt->pt.x), cvRound(skpIt->pt.y)),
                          p2(cvRound(bkpIt->pt.x), cvRound(bkpIt->pt.y));
                    double dist = norm(p1 - p2);
                    if( dist < minDist )
                    {
                        nearestIdx = bkpIdx;
                        minDist = dist;
                        nearestBkp = bkpIt;
                    }
                }
            }
            if( minDist < locThreshold )
            {
                matchedMask[nearestIdx] = true;
                repeatingLocationCount++;
                if( isAffineInvariant )
                    assert(0);
                else // scale invariant
                {
                    double minRadius = min( skpIt->size, nearestBkp->size ),
                           maxRadius = max( skpIt->size, nearestBkp->size );
                    double Es = abs(1 - (minRadius*minRadius)/(maxRadius*maxRadius));
                    if( Es < regThreshold )
                        repeatingRegionCount++;
                }
            }
        }
        repeatingLocationRltv = (float)repeatingLocationCount / smallKPSet->size();
        repeatingRegionRltv = (float)repeatingRegionCount / smallKPSet->size();
    }
}

//----------------------------------- base class of detector test ------------------------------------

const int DATASETS_COUNT = 8;
const int TEST_CASE_COUNT = 5;

const string DATASET_DIR = "detectors/datasets/";
const string ALGORITHMS_DIR = "detectors/algorithms/";

const string PARAMS_POSTFIX = "_params.xml";
const string RES_POSTFIX = "_res.xml";

const string RLC = "repeating_locations_count";
const string RLR = "repeating_locations_rltv";
const string RRC = "repeating_regions_count";
const string RRR = "repeating_regions_rltv";

string DATASET_NAMES[DATASETS_COUNT] = { "bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall"};

class CV_DetectorRepeatabilityTest : public CvTest
{
public:
    CV_DetectorRepeatabilityTest( const char* _detectorName, const char* testName ) : CvTest( testName, "repeatability-of-detector" )
    {
        detectorName = _detectorName;
        isAffineInvariant = false;

        validRepeatability.resize(DATASETS_COUNT);
        calcRepeatability.resize(DATASETS_COUNT);
    }

protected:
    virtual FeatureDetector* createDetector( int datasetIdx ) = 0;

    void readAllRunParams();
    virtual void readRunParams( FileNode& fn, int datasetIdx ) = 0;
    void writeAllRunParams();
    virtual void writeRunParams( FileStorage& fs, int datasetIdx ) = 0;
    void setDefaultAllRunParams();
    virtual void setDefaultRunParams( int datasetIdx ) = 0;

    void readResults();
    void writeResults();

    bool readDataset( const string& datasetName, vector<Mat>& Hs, vector<Mat>& imgs );

    void run( int );
    void processResults();

    bool isAffineInvariant;
    string detectorName;
    bool isWriteParams, isWriteResults;

    struct Repeatability
    {
        int repeatingLocationCount;
        float repeatingLocationRltv;
        int repeatingRegionCount;
        float repeatingRegionRltv;
    };
    vector<vector<Repeatability> > validRepeatability;
    vector<vector<Repeatability> > calcRepeatability;
};

void CV_DetectorRepeatabilityTest::readAllRunParams()
{
    string filename = string(ts->get_data_path()) + ALGORITHMS_DIR + detectorName + PARAMS_POSTFIX;
    FileStorage fs( filename, FileStorage::READ );
    if( !fs.isOpened() )
    {
        isWriteParams = true;
        setDefaultAllRunParams();
        ts->printf(CvTS::LOG, "all runParams are default\n");
    }
    else
    {
        isWriteParams = false;
        FileNode topfn = fs.getFirstTopLevelNode();
        for( int i = 0; i < DATASETS_COUNT; i++ )
        {
            FileNode fn = topfn[DATASET_NAMES[i]];
            if( fn.empty() )
            {
                ts->printf( CvTS::LOG, "%d-runParams is default\n", i);
                setDefaultRunParams(i);
            }
            else
                readRunParams(fn, i);
        }
    }
}

void CV_DetectorRepeatabilityTest::writeAllRunParams()
{
    string filename = string(ts->get_data_path()) + ALGORITHMS_DIR + detectorName + PARAMS_POSTFIX;
    FileStorage fs( filename, FileStorage::WRITE );
    if( fs.isOpened() )
    {
        fs << "run_params" << "{"; // top file node
        for( int i = 0; i < DATASETS_COUNT; i++ )
        {
            fs << DATASET_NAMES[i] << "{";
            writeRunParams(fs, i);
            fs << "}";
        }
        fs << "}";
    }
    else
        ts->printf(CvTS::LOG, "file %s for writing run params can not be opened\n", filename.c_str() );
}

void CV_DetectorRepeatabilityTest::setDefaultAllRunParams()
{
    for( int i = 0; i < DATASETS_COUNT; i++ )
        setDefaultRunParams(i);
}

bool CV_DetectorRepeatabilityTest::readDataset( const string& datasetName, vector<Mat>& Hs, vector<Mat>& imgs )
{
    Hs.resize( TEST_CASE_COUNT );
    imgs.resize( TEST_CASE_COUNT+1 );
    string dirname = string(ts->get_data_path()) + DATASET_DIR + datasetName + "/";

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

void CV_DetectorRepeatabilityTest::readResults()
{
    string filename = string(ts->get_data_path()) + ALGORITHMS_DIR + detectorName + RES_POSTFIX;
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
                validRepeatability[di].clear();
                ts->printf( CvTS::LOG, "results for %s dataset were not read\n",
                        DATASET_NAMES[di].c_str());
            }
            else
            {
                validRepeatability[di].resize(TEST_CASE_COUNT);
                for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
                {
                    stringstream ss; ss << "case" << ci;
                    FileNode casefn = datafn[ss.str()];
                    CV_Assert( !casefn.empty() );
                    validRepeatability[di][ci].repeatingLocationCount = casefn[RLC];
                    validRepeatability[di][ci].repeatingLocationRltv = casefn[RLR];
                    validRepeatability[di][ci].repeatingRegionCount = casefn[RRC];
                    validRepeatability[di][ci].repeatingRegionRltv = casefn[RRR];
                }
            }
        }
    }
    else
        isWriteResults = true;
}

void CV_DetectorRepeatabilityTest::writeResults()
{
    string filename = string(ts->get_data_path()) + ALGORITHMS_DIR + detectorName + RES_POSTFIX;
    FileStorage fs( filename, FileStorage::WRITE );
    if( fs.isOpened() )
    {
        fs << "results" << "{";
        for( int di = 0; di < DATASETS_COUNT; di++ )
        {
            if( calcRepeatability[di].empty() )
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
                    fs << RLC << calcRepeatability[di][ci].repeatingLocationCount;
                    fs << RLR << calcRepeatability[di][ci].repeatingLocationRltv;
                    fs << RRC << calcRepeatability[di][ci].repeatingRegionCount;
                    fs << RRR << calcRepeatability[di][ci].repeatingRegionRltv;
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

void CV_DetectorRepeatabilityTest::run( int )
{
    readAllRunParams();
    readResults();

    int notReadDatasets = 0;
    int progress = 0, progressCount = DATASETS_COUNT*TEST_CASE_COUNT;
    for(int di = 0; di < DATASETS_COUNT; di++ )
    {   
        vector<Mat> imgs, Hs;
        if( !readDataset( DATASET_NAMES[di], Hs, imgs ) )
        {
            calcRepeatability[di].clear();
            ts->printf( CvTS::LOG, "images or homography matrices of dataset named %s can not be read\n",
                        DATASET_NAMES[di].c_str());
            notReadDatasets++;
        }
        else
        {
            calcRepeatability[di].resize(TEST_CASE_COUNT);
            Ptr<FeatureDetector> detector = createDetector(di);

            vector<KeyPoint> keypoints1;
            detector->detect( imgs[0], keypoints1 );
            for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
            {
                progress = update_progress( progress, di*TEST_CASE_COUNT + ci, progressCount, 0 );
                vector<KeyPoint> keypoints2;
                detector->detect( imgs[ci+1], keypoints2 );
                repeatability( imgs[0], imgs[ci+1], Hs[ci], keypoints1, keypoints2,
                    calcRepeatability[di][ci].repeatingLocationCount, calcRepeatability[di][ci].repeatingLocationRltv,
                    calcRepeatability[di][ci].repeatingRegionCount, calcRepeatability[di][ci].repeatingRegionRltv,
                    isAffineInvariant );
            }
        }
    }
    if( notReadDatasets == DATASETS_COUNT )
    {
        ts->printf(CvTS::LOG, "All datasets were not be read\n");
        ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
    }
    else
        processResults();
}

void testLog( CvTS* ts, bool isBadAccuracy )
{
    if( isBadAccuracy )
        ts->printf(CvTS::LOG, " bad accuracy\n");
    else
        ts->printf(CvTS::LOG, "\n");
}

void CV_DetectorRepeatabilityTest::processResults()
{
    if( isWriteParams )
        writeAllRunParams();

    bool isBadAccuracy;
    int res = CvTS::OK;
    if( isWriteResults )
        writeResults();
    else
    {
        for( int di = 0; di < DATASETS_COUNT; di++ )
        {
            if( validRepeatability[di].empty() || calcRepeatability[di].empty() )
                continue;

            ts->printf(CvTS::LOG, "\nDataset: %s\n", DATASET_NAMES[di].c_str() );

            int countEps = 1;
            float rltvEps = 0.001f;
            for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
            {
                ts->printf(CvTS::LOG, "case%d\n", ci);
                Repeatability valid = validRepeatability[di][ci], calc = calcRepeatability[di][ci];

                ts->printf(CvTS::LOG, "%s: calc=%d, valid=%d", RLC.c_str(), calc.repeatingLocationCount, valid.repeatingLocationCount );
                isBadAccuracy = valid.repeatingLocationCount - calc.repeatingLocationCount > countEps;
                testLog( ts, isBadAccuracy );
                res = isBadAccuracy ? CvTS::FAIL_BAD_ACCURACY : res;

                ts->printf(CvTS::LOG, "%s: calc=%f, valid=%f", RLR.c_str(), calc.repeatingLocationRltv, valid.repeatingLocationRltv );
                isBadAccuracy = valid.repeatingLocationRltv - calc.repeatingLocationRltv > rltvEps;
                testLog( ts, isBadAccuracy );
                res = isBadAccuracy ? CvTS::FAIL_BAD_ACCURACY : res;

                ts->printf(CvTS::LOG, "%s: calc=%d, valid=%d", RRC.c_str(), calc.repeatingRegionCount, valid.repeatingRegionCount );
                isBadAccuracy = valid.repeatingRegionCount - calc.repeatingRegionCount > countEps;
                testLog( ts, isBadAccuracy );
                res = isBadAccuracy ? CvTS::FAIL_BAD_ACCURACY : res;

                ts->printf(CvTS::LOG, "%s: calc=%f, valid=%f", RRR.c_str(), calc.repeatingRegionRltv, valid.repeatingRegionRltv );
                isBadAccuracy = valid.repeatingRegionRltv - calc.repeatingRegionRltv > rltvEps;
                testLog( ts, isBadAccuracy );
                res = isBadAccuracy ? CvTS::FAIL_BAD_ACCURACY : res;
            }
        }
    }

    if( res != CvTS::OK )
        ts->printf(CvTS::LOG, "BAD ACCURACY\n");
    ts->set_failed_test_info( res );
}

//--------------------------------- FAST detector test --------------------------------------------
class CV_FastDetectorTest : public CV_DetectorRepeatabilityTest
{
public:
    CV_FastDetectorTest() : CV_DetectorRepeatabilityTest( "fast", "repeatability-fast-detector" )
    { runParams.resize(DATASETS_COUNT); }

protected:
    virtual FeatureDetector* createDetector( int datasetIdx );
    virtual void readRunParams( FileNode& fn, int datasetIdx );
    virtual void writeRunParams( FileStorage& fs, int datasetIdx );
    virtual void setDefaultRunParams( int datasetIdx );

    struct RunParams
    {
        int threshold;
        bool nonmaxSuppression;
    };
    vector<RunParams> runParams;
};

FeatureDetector* CV_FastDetectorTest::createDetector( int datasetIdx )
{
    return new FastFeatureDetector( runParams[datasetIdx].threshold, runParams[datasetIdx].nonmaxSuppression );
}

void CV_FastDetectorTest::readRunParams( FileNode& fn, int datasetIdx )
{
    runParams[datasetIdx].threshold = fn["threshold"];
    runParams[datasetIdx].nonmaxSuppression = (int)fn["nonmaxSuppression"] ? true : false;
}

void CV_FastDetectorTest::writeRunParams( FileStorage& fs, int datasetIdx )
{
    fs << "threshold" << runParams[datasetIdx].threshold;
    fs << "nonmaxSuppression" << runParams[datasetIdx].nonmaxSuppression;
}

void CV_FastDetectorTest::setDefaultRunParams( int datasetIdx )
{
    runParams[datasetIdx].threshold = 1;
    runParams[datasetIdx].nonmaxSuppression = true;
}

CV_FastDetectorTest fastDetector;

//--------------------------------- GFTT & HARRIS detectors tests --------------------------------------------
class CV_BaseGfttDetectorTest : public CV_DetectorRepeatabilityTest
{
public:
    CV_BaseGfttDetectorTest( const char* detectorName, const char* testName )
        : CV_DetectorRepeatabilityTest( detectorName, testName )
    {
        runParams.resize(DATASETS_COUNT);
        useHarrisDetector = false;
    }

protected:
    virtual FeatureDetector* createDetector( int datasetIdx );
    virtual void readRunParams( FileNode& fn, int datasetIdx );
    virtual void writeRunParams( FileStorage& fs, int datasetIdx );
    virtual void setDefaultRunParams( int datasetIdx );

    struct RunParams
    {
        int maxCorners;
        double qualityLevel;
        double minDistance;
        int blockSize;
        double k;
    };
    vector<RunParams> runParams;
    bool useHarrisDetector;
};

FeatureDetector* CV_BaseGfttDetectorTest::createDetector( int datasetIdx )
{
    return new GoodFeaturesToTrackDetector( runParams[datasetIdx].maxCorners,
                                            runParams[datasetIdx].qualityLevel,
                                            runParams[datasetIdx].minDistance,
                                            runParams[datasetIdx].blockSize,
                                            useHarrisDetector,
                                            runParams[datasetIdx].k );
}

void CV_BaseGfttDetectorTest::readRunParams( FileNode& fn, int datasetIdx )
{
    runParams[datasetIdx].maxCorners = fn["maxCorners"];
    runParams[datasetIdx].qualityLevel = fn["qualityLevel"];
    runParams[datasetIdx].minDistance = fn["minDistance"];
    runParams[datasetIdx].blockSize = fn["blockSize"];
    runParams[datasetIdx].k = fn["k"];
}

void CV_BaseGfttDetectorTest::writeRunParams( FileStorage& fs, int datasetIdx )
{
    fs << "maxCorners" << runParams[datasetIdx].maxCorners;
    fs << "qualityLevel" << runParams[datasetIdx].qualityLevel;
    fs << "minDistance" << runParams[datasetIdx].minDistance;
    fs << "blockSize" << runParams[datasetIdx].blockSize;
    fs << "k" << runParams[datasetIdx].k;
}

void CV_BaseGfttDetectorTest::setDefaultRunParams( int datasetIdx )
{
    runParams[datasetIdx].maxCorners = 1500;
    runParams[datasetIdx].qualityLevel = 0.01;
    runParams[datasetIdx].minDistance = 2.0;
    runParams[datasetIdx].blockSize = 3;
    runParams[datasetIdx].k = 0.04;
}

class CV_GfttDetectorTest : public CV_BaseGfttDetectorTest
{
public:
    CV_GfttDetectorTest() : CV_BaseGfttDetectorTest( "gftt", "repeatability-gftt-detector" ) {}
};

CV_GfttDetectorTest gfttDetector;

class CV_HarrisDetectorTest : public CV_BaseGfttDetectorTest
{
public:
    CV_HarrisDetectorTest() : CV_BaseGfttDetectorTest( "harris", "repeatability-harris-detector" )
        { useHarrisDetector = true; }
};

CV_HarrisDetectorTest harrisDetector;

//--------------------------------- MSER detector test --------------------------------------------
class CV_MserDetectorTest : public CV_DetectorRepeatabilityTest
{
public:
    CV_MserDetectorTest() : CV_DetectorRepeatabilityTest( "mser", "repeatability-mser-detector" )
    { runParams.resize(DATASETS_COUNT); }

protected:
    virtual FeatureDetector* createDetector( int datasetIdx );
    virtual void readRunParams( FileNode& fn, int datasetIdx );
    virtual void writeRunParams( FileStorage& fs, int datasetIdx );
    virtual void setDefaultRunParams( int datasetIdx );

    struct RunParams
    {
        int delta;
        int minArea;
        int maxArea;
        float maxVariation;
        float minDiversity;
        int maxEvolution;
        double areaThreshold;
        double minMargin;
        int edgeBlurSize;
    };
    vector<RunParams> runParams;
};

FeatureDetector* CV_MserDetectorTest::createDetector( int datasetIdx )
{
    return new MserFeatureDetector( runParams[datasetIdx].delta,
                                    runParams[datasetIdx].minArea,
                                    runParams[datasetIdx].maxArea,
                                    runParams[datasetIdx].maxVariation,
                                    runParams[datasetIdx].minDiversity,
                                    runParams[datasetIdx].maxEvolution,
                                    runParams[datasetIdx].areaThreshold,
                                    runParams[datasetIdx].minMargin,
                                    runParams[datasetIdx].edgeBlurSize );
}

void CV_MserDetectorTest::readRunParams( FileNode& fn, int datasetIdx )
{
    runParams[datasetIdx].delta = fn["delta"];
    runParams[datasetIdx].minArea = fn["minArea"];
    runParams[datasetIdx].maxArea = fn["maxArea"];
    runParams[datasetIdx].maxVariation = fn["maxVariation"];
    runParams[datasetIdx].minDiversity = fn["minDiversity"];
    runParams[datasetIdx].maxEvolution = fn["maxEvolution"];
    runParams[datasetIdx].areaThreshold = fn["areaThreshold"];
    runParams[datasetIdx].minMargin = fn["minMargin"];
    runParams[datasetIdx].edgeBlurSize = fn["edgeBlurSize"];
}

void CV_MserDetectorTest::writeRunParams( FileStorage& fs, int datasetIdx )
{
    fs << "delta" << runParams[datasetIdx].delta;
    fs << "minArea" << runParams[datasetIdx].minArea;
    fs << "maxArea" << runParams[datasetIdx].maxArea;
    fs << "maxVariation" << runParams[datasetIdx].maxVariation;
    fs << "minDiversity" << runParams[datasetIdx].minDiversity;
    fs << "maxEvolution" << runParams[datasetIdx].maxEvolution;
    fs << "areaThreshold" << runParams[datasetIdx].areaThreshold;
    fs << "minMargin" << runParams[datasetIdx].minMargin;
    fs << "edgeBlurSize" << runParams[datasetIdx].edgeBlurSize;
}

void CV_MserDetectorTest::setDefaultRunParams( int datasetIdx )
{
    runParams[datasetIdx].delta = 5;
    runParams[datasetIdx].minArea = 60;
    runParams[datasetIdx].maxArea = 14400;
    runParams[datasetIdx].maxVariation = 0.25f;
    runParams[datasetIdx].minDiversity = 0.2;
    runParams[datasetIdx].maxEvolution = 200;
    runParams[datasetIdx].areaThreshold = 1.01;
    runParams[datasetIdx].minMargin = 0.003;
    runParams[datasetIdx].edgeBlurSize = 5;
}

CV_MserDetectorTest mserDetector;

//--------------------------------- STAR detector test --------------------------------------------
class CV_StarDetectorTest : public CV_DetectorRepeatabilityTest
{
public:
    CV_StarDetectorTest() : CV_DetectorRepeatabilityTest( "star", "repeatability-star-detector" )
    { runParams.resize(DATASETS_COUNT); }

protected:
    virtual FeatureDetector* createDetector( int datasetIdx );
    virtual void readRunParams( FileNode& fn, int datasetIdx );
    virtual void writeRunParams( FileStorage& fs, int datasetIdx );
    virtual void setDefaultRunParams( int datasetIdx );

    struct RunParams
    {
        int maxSize;
        int responseThreshold;
        int lineThresholdProjected;
        int lineThresholdBinarized;
        int suppressNonmaxSize;
    };
    vector<RunParams> runParams;
};

FeatureDetector* CV_StarDetectorTest::createDetector( int datasetIdx )
{
    return new StarFeatureDetector( runParams[datasetIdx].maxSize,
                                    runParams[datasetIdx].responseThreshold,
                                    runParams[datasetIdx].lineThresholdProjected,
                                    runParams[datasetIdx].lineThresholdBinarized,
                                    runParams[datasetIdx].suppressNonmaxSize );
}

void CV_StarDetectorTest::readRunParams( FileNode& fn, int datasetIdx )
{
    runParams[datasetIdx].maxSize = fn["maxSize"];
    runParams[datasetIdx].responseThreshold = fn["responseThreshold"];
    runParams[datasetIdx].lineThresholdProjected = fn["lineThresholdProjected"];
    runParams[datasetIdx].lineThresholdBinarized = fn["lineThresholdBinarized"];
    runParams[datasetIdx].suppressNonmaxSize = fn["suppressNonmaxSize"];
}

void CV_StarDetectorTest::writeRunParams( FileStorage& fs, int datasetIdx )
{
    fs << "maxSize" << runParams[datasetIdx].maxSize;
    fs << "responseThreshold" << runParams[datasetIdx].responseThreshold;
    fs << "lineThresholdProjected" << runParams[datasetIdx].lineThresholdProjected;
    fs << "lineThresholdBinarized" << runParams[datasetIdx].lineThresholdBinarized;
    fs << "suppressNonmaxSize" << runParams[datasetIdx].suppressNonmaxSize;
}

void CV_StarDetectorTest::setDefaultRunParams( int datasetIdx )
{
    runParams[datasetIdx].maxSize = 16;
    runParams[datasetIdx].responseThreshold = 30;
    runParams[datasetIdx].lineThresholdProjected = 10;
    runParams[datasetIdx].lineThresholdBinarized = 8;
    runParams[datasetIdx].suppressNonmaxSize = 5;
}

CV_StarDetectorTest starDetector;

//--------------------------------- SIFT detector test --------------------------------------------
class CV_SiftDetectorTest : public CV_DetectorRepeatabilityTest
{
public:
    CV_SiftDetectorTest() : CV_DetectorRepeatabilityTest( "sift", "repeatability-sift-detector" )
    { runParams.resize(DATASETS_COUNT); }

protected:
    virtual FeatureDetector* createDetector( int datasetIdx );
    virtual void readRunParams( FileNode& fn, int datasetIdx );
    virtual void writeRunParams( FileStorage& fs, int datasetIdx );
    virtual void setDefaultRunParams( int datasetIdx );

    struct RunParams
    {
        SIFT::CommonParams comm;
        SIFT::DetectorParams detect;
    };

    vector<RunParams> runParams;
};

FeatureDetector* CV_SiftDetectorTest::createDetector( int datasetIdx )
{
    return new SiftFeatureDetector( runParams[datasetIdx].detect.threshold,
                                    runParams[datasetIdx].detect.edgeThreshold,
                                    runParams[datasetIdx].detect.angleMode,
                                    runParams[datasetIdx].comm.nOctaves,
                                    runParams[datasetIdx].comm.nOctaveLayers,
                                    runParams[datasetIdx].comm.firstOctave );
}

void CV_SiftDetectorTest::readRunParams( FileNode& fn, int datasetIdx )
{
    runParams[datasetIdx].detect.threshold = fn["threshold"];
    runParams[datasetIdx].detect.edgeThreshold = fn["edgeThreshold"];
    runParams[datasetIdx].detect.angleMode = fn["angleMode"];
    runParams[datasetIdx].comm.nOctaves = fn["nOctaves"];
    runParams[datasetIdx].comm.nOctaveLayers = fn["nOctaveLayers"];
    runParams[datasetIdx].comm.firstOctave = fn["firstOctave"];
}

void CV_SiftDetectorTest::writeRunParams( FileStorage& fs, int datasetIdx )
{
    fs << "threshold" << runParams[datasetIdx].detect.threshold;
    fs << "edgeThreshold" << runParams[datasetIdx].detect.edgeThreshold;
    fs << "angleMode" << runParams[datasetIdx].detect.angleMode;
    fs << "nOctaves" << runParams[datasetIdx].comm.nOctaves;
    fs << "nOctaveLayers" << runParams[datasetIdx].comm.nOctaveLayers;
    fs << "firstOctave" << runParams[datasetIdx].comm.firstOctave;
 }

void CV_SiftDetectorTest::setDefaultRunParams( int datasetIdx )
{
    runParams[datasetIdx].detect = SIFT::DetectorParams();
    runParams[datasetIdx].comm = SIFT::CommonParams();
}

CV_SiftDetectorTest siftDetector;

//--------------------------------- SURF detector test --------------------------------------------
class CV_SurfDetectorTest : public CV_DetectorRepeatabilityTest
{
public:
    CV_SurfDetectorTest() : CV_DetectorRepeatabilityTest( "surf", "repeatability-surf-detector" )
    { runParams.resize(DATASETS_COUNT); }

protected:
    virtual FeatureDetector* createDetector( int datasetIdx );
    virtual void readRunParams( FileNode& fn, int datasetIdx );
    virtual void writeRunParams( FileStorage& fs, int datasetIdx );
    virtual void setDefaultRunParams( int datasetIdx );

    struct RunParams
    {
        double hessianThreshold;
        int octaves;
        int octaveLayers;
    };
    vector<RunParams> runParams;
};

FeatureDetector* CV_SurfDetectorTest::createDetector( int datasetIdx )
{
    return new SurfFeatureDetector( runParams[datasetIdx].hessianThreshold,
                                    runParams[datasetIdx].octaves,
                                    runParams[datasetIdx].octaveLayers );
}

void CV_SurfDetectorTest::readRunParams( FileNode& fn, int datasetIdx )
{
    runParams[datasetIdx].hessianThreshold = fn["hessianThreshold"];
    runParams[datasetIdx].octaves = fn["octaves"];
    runParams[datasetIdx].octaveLayers = fn["octaveLayers"];
}

void CV_SurfDetectorTest::writeRunParams( FileStorage& fs, int datasetIdx )
{
    fs << "hessianThreshold" << runParams[datasetIdx].hessianThreshold;
    fs << "octaves" << runParams[datasetIdx].octaves;
    fs << "octaveLayers" << runParams[datasetIdx].octaveLayers;
}

void CV_SurfDetectorTest::setDefaultRunParams( int datasetIdx )
{
    runParams[datasetIdx].hessianThreshold = 400.;
    runParams[datasetIdx].octaves = 3;
    runParams[datasetIdx].octaveLayers = 4;
}

CV_SurfDetectorTest surfDetector;
