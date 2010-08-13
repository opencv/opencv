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

static inline Point2f applyHomography( const Mat_<double>& H, const Point2f& pt )
{
    double z = H(2,0)*pt.x + H(2,1)*pt.y + H(2,2);
    if( z )
    {
        double w = 1./z;
        return Point2f( (H(0,0)*pt.x + H(0,1)*pt.y + H(0,2))*w, (H(1,0)*pt.x + H(1,1)*pt.y + H(1,2))*w );
    }
    return Point2f( numeric_limits<float>::max(), numeric_limits<float>::max() );
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
        A.setTo(Scalar::all(numeric_limits<double>::max()));
}

static void calcKeyPointProjections( const vector<KeyPoint>& src, const Mat_<double>& H, vector<KeyPoint>& dst )
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
            Mat_<double> M(2, 2);
            M(0,0) = M(1,1) = 1./srcSize2;
            M(1,0) = M(0,1) = 0;
            Mat_<double> invM; invert(M, invM);
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

static void filterKeyPointsByImageSize( vector<KeyPoint>& keypoints, const Size& imgSize )
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

    vector<KeyPoint> keypoints1;
    detector->detect( imgs[0], keypoints1 );
    writeKeypoints( keypontsFS, keypoints1, 0);
    int progressCount = DATASETS_COUNT*TEST_CASE_COUNT;
    for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
    {
        progress = update_progress( progress, di*TEST_CASE_COUNT + ci, progressCount, 0 );
        vector<KeyPoint> keypoints2;
        evaluateFeatureDetector( imgs[0], imgs[ci+1], Hs[ci], &keypoints1, &keypoints2,
                                 calcQuality[di][ci].repeatability, calcQuality[di][ci].correspondenceCount,
                                 detector );
        writeKeypoints( keypontsFS, keypoints2, ci+1);
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
    void calculatePlotData( vector<vector<DMatch> > &allMatches, vector<vector<uchar> > &allCorrectMatchesMask, int di );

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
        Ptr<DescriptorExtractor> extractor = createDescriptorExtractor( algName );
        Ptr<DescriptorMatcher> matcher = createDescriptorMatcher( matcherName );
        defaultDescMatch = new VectorDescriptorMatch( extractor, matcher );
        specificDescMatch = new VectorDescriptorMatch( extractor, matcher );

        if( extractor == 0 || matcher == 0 )
        {
            ts->printf(CvTS::LOG, "Algorithm can not be read\n");
            ts->set_failed_test_info( CvTS::FAIL_GENERIC);
        }
    }
}

void DescriptorQualityTest::calculatePlotData( vector<vector<DMatch> > &allMatches, vector<vector<uchar> > &allCorrectMatchesMask, int di )
{
    vector<Point2f> recallPrecisionCurve;
    computeRecallPrecisionCurve( allMatches, allCorrectMatchesMask, recallPrecisionCurve );
    // you have recallPrecisionCurve for all images from dataset
    // size of recallPrecisionCurve == total matches count
#if 0

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

            if( !isResultCalculated && quality.precision < resultPrecision )
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
#endif

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

    vector<KeyPoint> keypoints1;
    readKeypoints( keypontsFS, keypoints1, 0);

    int progressCount = DATASETS_COUNT*TEST_CASE_COUNT;

    vector<vector<DMatch> > allMatches1to2;
    vector<vector<uchar> > allCorrectMatchesMask;
    for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
    {
        progress = update_progress( progress, di*TEST_CASE_COUNT + ci, progressCount, 0 );

        vector<KeyPoint> keypoints2;
        if( commRunParams[di].projectKeypointsFrom1Image )
        {
            // TODO need to test function calcKeyPointProjections
            calcKeyPointProjections( keypoints1, Hs[ci], keypoints2 );
            filterKeyPointsByImageSize( keypoints2,  imgs[ci+1].size() );
        }
        else
            readKeypoints( keypontsFS, keypoints2, ci+1 );
        // TODO if( commRunParams[di].matchFilter )

        vector<vector<DMatch> > matches1to2;
        vector<vector<uchar> > correctMatchesMask;
        vector<Point2f> recallPrecisionCurve; // not used because we need recallPrecisionCurve for
                                              // all images in dataset
        evaluateDescriptorMatch( imgs[0], imgs[ci+1], Hs[ci], keypoints1, keypoints2,
                                 &matches1to2, &correctMatchesMask, recallPrecisionCurve,
                                 descMatch );
        allMatches1to2.insert( allMatches1to2.end(), matches1to2.begin(), matches1to2.end() );
        allCorrectMatchesMask.insert( allCorrectMatchesMask.end(), correctMatchesMask.begin(), correctMatchesMask.end() );
    }

    calculatePlotData( allMatches1to2, allCorrectMatchesMask, di );
}

int DescriptorQualityTest::processResults( int datasetIdx, int caseIdx )
{
    int res = CvTS::OK;
    Quality valid = validQuality[datasetIdx][caseIdx], calc = calcQuality[datasetIdx][caseIdx];

    bool isBadAccuracy;
    const float rltvEps = 0.001f;
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
        return;

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
