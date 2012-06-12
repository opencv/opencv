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

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/legacy/legacy.hpp"

#include <limits>
#include <cstdio>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;


string data_path;
/****************************************************************************************\
*           Functions to evaluate affine covariant detectors and descriptors.            *
\****************************************************************************************/

static inline Point2f applyHomography( const Mat_<double>& H, const Point2f& pt )
{
    double z = H(2,0)*pt.x + H(2,1)*pt.y + H(2,2);
    if( z )
    {
        double w = 1./z;
        return Point2f( (float)((H(0,0)*pt.x + H(0,1)*pt.y + H(0,2))*w),
                                                (float)((H(1,0)*pt.x + H(1,1)*pt.y + H(1,2))*w) );
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
            float dstSize = (float)pow(1./(eval(0,0)*eval(1,0)), 0.25);

            // TODO: check angle projection
            float srcAngleRad = (float)(srcIt->angle*CV_PI/180);
            Point2f vec1(cos(srcAngleRad), sin(srcAngleRad)), vec2;
            vec2.x = (float)(Aff(0,0)*vec1.x + Aff(0,1)*vec1.y);
            vec2.y = (float)(Aff(1,0)*vec1.x + Aff(0,1)*vec1.y);
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


class BaseQualityEvaluator
{
public:
    BaseQualityEvaluator( const char* _algName, const char* _testName ) : algName(_algName), testName(_testName)
    {
        //TODO: change this
        isWriteGraphicsData = true;
    }

    void run();

    virtual ~BaseQualityEvaluator(){}

protected:
    virtual string getRunParamsFilename() const = 0;
    virtual string getResultsFilename() const = 0;
    virtual string getPlotPath() const = 0;

    virtual void calcQualityClear( int datasetIdx ) = 0;
    virtual bool isCalcQualityEmpty( int datasetIdx ) const = 0;

    void readAllDatasetsRunParams();
    virtual void readDatasetRunParams( FileNode& fn, int datasetIdx ) = 0;
    void writeAllDatasetsRunParams() const;
    virtual void writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const = 0;
    void setDefaultAllDatasetsRunParams();
    virtual void setDefaultDatasetRunParams( int datasetIdx ) = 0;
    virtual void readDefaultRunParams( FileNode& /*fn*/ ) {}
    virtual void writeDefaultRunParams( FileStorage& /*fs*/ ) const {}

    bool readDataset( const string& datasetName, vector<Mat>& Hs, vector<Mat>& imgs );

    virtual void readAlgorithm() {}
    virtual void processRunParamsFile() {}
    virtual void runDatasetTest( const vector<Mat>& /*imgs*/, const vector<Mat>& /*Hs*/, int /*di*/, int& /*progress*/ ) {}

    virtual void processResults( int datasetIdx );
    virtual void processResults();
    virtual void writePlotData( int /*datasetIdx*/ ) const {}

    string algName, testName;
    bool isWriteParams, isWriteGraphicsData;
};

void BaseQualityEvaluator::readAllDatasetsRunParams()
{
    string filename = getRunParamsFilename();
    FileStorage fs( filename, FileStorage::READ );
    if( !fs.isOpened() )
    {
        isWriteParams = true;
        setDefaultAllDatasetsRunParams();
        printf("All runParams are default.\n");
    }
    else
    {
        isWriteParams = false;
        FileNode topfn = fs.getFirstTopLevelNode();

        FileNode pfn = topfn[DEFAULT_PARAMS];
        readDefaultRunParams(pfn);

        for( int i = 0; i < DATASETS_COUNT; i++ )
        {
            FileNode fn = topfn[DATASET_NAMES[i]];
            if( fn.empty() )
            {
                printf( "%d-runParams is default.\n", i);
                setDefaultDatasetRunParams(i);
            }
            else
                readDatasetRunParams(fn, i);
        }
    }
}

void BaseQualityEvaluator::writeAllDatasetsRunParams() const
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
        printf( "File %s for writing run params can not be opened.\n", filename.c_str() );
}

void BaseQualityEvaluator::setDefaultAllDatasetsRunParams()
{
    for( int i = 0; i < DATASETS_COUNT; i++ )
        setDefaultDatasetRunParams(i);
}

bool BaseQualityEvaluator::readDataset( const string& datasetName, vector<Mat>& Hs, vector<Mat>& imgs )
{
    Hs.resize( TEST_CASE_COUNT );
    imgs.resize( TEST_CASE_COUNT+1 );
    string dirname = data_path + IMAGE_DATASETS_DIR + datasetName + "/";
    for( int i = 0; i < (int)Hs.size(); i++ )
    {
        stringstream filename; filename << "H1to" << i+2 << "p.xml";
        FileStorage fs( dirname + filename.str(), FileStorage::READ );
        if( !fs.isOpened() )
        {
            cout << "filename " << dirname + filename.str() << endl;
            FileStorage fs2( dirname + filename.str(), FileStorage::READ );
            return false;
        }
        fs.getFirstTopLevelNode() >> Hs[i];
    }

    for( int i = 0; i < (int)imgs.size(); i++ )
    {
        stringstream filename; filename << "img" << i+1 << ".png";
        imgs[i] = imread( dirname + filename.str(), 0 );
        if( imgs[i].empty() )
        {
            cout << "filename " << filename.str() << endl;
            return false;
        }
    }
    return true;
}

void BaseQualityEvaluator::processResults( int datasetIdx )
{
    if( isWriteGraphicsData )
        writePlotData( datasetIdx );
}

void BaseQualityEvaluator::processResults()
{
    if( isWriteParams )
        writeAllDatasetsRunParams();
}

void BaseQualityEvaluator::run()
{
    readAlgorithm ();
    processRunParamsFile ();

    int notReadDatasets = 0;
    int progress = 0;

    FileStorage runParamsFS( getRunParamsFilename(), FileStorage::READ );
    isWriteParams = (! runParamsFS.isOpened());
    FileNode topfn = runParamsFS.getFirstTopLevelNode();
    FileNode defaultParams = topfn[DEFAULT_PARAMS];
    readDefaultRunParams (defaultParams);

    cout << testName << endl;
    for(int di = 0; di < DATASETS_COUNT; di++ )
    {
        cout << "Dataset " << di << " [" << DATASET_NAMES[di] << "] " << flush;
        vector<Mat> imgs, Hs;
        if( !readDataset( DATASET_NAMES[di], Hs, imgs ) )
        {
            calcQualityClear (di);
            printf( "Images or homography matrices of dataset named %s can not be read\n",
                        DATASET_NAMES[di].c_str());
            notReadDatasets++;
            continue;
        }

        FileNode fn = topfn[DATASET_NAMES[di]];
        readDatasetRunParams(fn, di);

        runDatasetTest (imgs, Hs, di, progress);
        processResults( di );
        cout << endl;
    }
    if( notReadDatasets == DATASETS_COUNT )
    {
        printf( "All datasets were not be read\n");
        exit(-1);
    }
    else
        processResults();
    runParamsFS.release();
}



class DetectorQualityEvaluator : public BaseQualityEvaluator
{
public:
    DetectorQualityEvaluator( const char* _detectorName, const char* _testName ) : BaseQualityEvaluator( _detectorName, _testName )
    {
        calcQuality.resize(DATASETS_COUNT);
        isSaveKeypoints.resize(DATASETS_COUNT);
        isActiveParams.resize(DATASETS_COUNT);

        isSaveKeypointsDefault = false;
        isActiveParamsDefault = false;
    }

protected:
    virtual string getRunParamsFilename() const;
    virtual string getResultsFilename() const;
    virtual string getPlotPath() const;

    virtual void calcQualityClear( int datasetIdx );
    virtual bool isCalcQualityEmpty( int datasetIdx ) const;

    virtual void readDatasetRunParams( FileNode& fn, int datasetIdx );
    virtual void writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const;
    virtual void setDefaultDatasetRunParams( int datasetIdx );
    virtual void readDefaultRunParams( FileNode &fn );
    virtual void writeDefaultRunParams( FileStorage &fs ) const;

    virtual void writePlotData( int di ) const;

    void openToWriteKeypointsFile( FileStorage& fs, int datasetIdx );

    virtual void readAlgorithm();
    virtual void processRunParamsFile() {}
    virtual void runDatasetTest( const vector<Mat> &imgs, const vector<Mat> &Hs, int di, int &progress );

    Ptr<FeatureDetector> specificDetector;
    Ptr<FeatureDetector> defaultDetector;

    struct Quality
    {
        float repeatability;
        int correspondenceCount;
    };
    vector<vector<Quality> > calcQuality;

    vector<bool> isSaveKeypoints;
    vector<bool> isActiveParams;

    bool isSaveKeypointsDefault;
    bool isActiveParamsDefault;
};

string DetectorQualityEvaluator::getRunParamsFilename() const
{
     return data_path + DETECTORS_DIR + algName + PARAMS_POSTFIX;
}

string DetectorQualityEvaluator::getResultsFilename() const
{
    return data_path + DETECTORS_DIR + algName + RES_POSTFIX;
}

string DetectorQualityEvaluator::getPlotPath() const
{
    return data_path + DETECTORS_DIR + "plots/";
}

void DetectorQualityEvaluator::calcQualityClear( int datasetIdx )
{
    calcQuality[datasetIdx].clear();
}

bool DetectorQualityEvaluator::isCalcQualityEmpty( int datasetIdx ) const
{
    return calcQuality[datasetIdx].empty();
}

void DetectorQualityEvaluator::readDefaultRunParams (FileNode &fn)
{
    if (! fn.empty() )
    {
        isSaveKeypointsDefault = (int)fn[IS_SAVE_KEYPOINTS] != 0;
        defaultDetector->read (fn);
    }
}

void DetectorQualityEvaluator::writeDefaultRunParams (FileStorage &fs) const
{
    fs << IS_SAVE_KEYPOINTS << isSaveKeypointsDefault;
    defaultDetector->write (fs);
}

void DetectorQualityEvaluator::readDatasetRunParams( FileNode& fn, int datasetIdx )
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

void DetectorQualityEvaluator::writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const
{
    fs << IS_ACTIVE_PARAMS << isActiveParams[datasetIdx];
    fs << IS_SAVE_KEYPOINTS << isSaveKeypoints[datasetIdx];
    defaultDetector->write (fs);
}

void DetectorQualityEvaluator::setDefaultDatasetRunParams( int datasetIdx )
{
    isSaveKeypoints[datasetIdx] = isSaveKeypointsDefault;
    isActiveParams[datasetIdx] = isActiveParamsDefault;
}

void DetectorQualityEvaluator::writePlotData(int di ) const
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

void DetectorQualityEvaluator::openToWriteKeypointsFile( FileStorage& fs, int datasetIdx )
{
    string filename = data_path + KEYPOINTS_DIR + algName + "_"+ DATASET_NAMES[datasetIdx] + ".xml.gz" ;

    fs.open(filename, FileStorage::WRITE);
    if( !fs.isOpened() )
        printf( "keypoints can not be written in file %s because this file can not be opened\n", filename.c_str() );
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

void DetectorQualityEvaluator::readAlgorithm ()
{
    defaultDetector = FeatureDetector::create( algName );
    specificDetector = FeatureDetector::create( algName );
    if( defaultDetector == 0 )
    {
        printf( "Algorithm can not be read\n" );
        exit(-1);
    }
}

static int update_progress( const string& /*name*/, int progress, int test_case_idx, int count, double dt )
{
    int width = 60 /*- (int)name.length()*/;
    if( count > 0 )
    {
        int t = cvRound( ((double)test_case_idx * width)/count );
        if( t > progress )
        {
            cout << "." << flush;
            progress = t;
        }
    }
    else if( cvRound(dt) > progress )
    {
        cout << "." << flush;
        progress = cvRound(dt);
    }

    return progress;
}

void DetectorQualityEvaluator::runDatasetTest (const vector<Mat> &imgs, const vector<Mat> &Hs, int di, int &progress)
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
        progress = update_progress( testName, progress, di*TEST_CASE_COUNT + ci + 1, progressCount, 0 );
        vector<KeyPoint> keypoints2;
        float rep;
        evaluateFeatureDetector( imgs[0], imgs[ci+1], Hs[ci], &keypoints1, &keypoints2,
                                 rep, calcQuality[di][ci].correspondenceCount,
                                 detector );
        calcQuality[di][ci].repeatability = rep == -1 ? rep : 100.f*rep;
        writeKeypoints( keypontsFS, keypoints2, ci+1);
    }
}

// static void testLog( bool isBadAccuracy )
// {
//     if( isBadAccuracy )
//         printf(" bad accuracy\n");
//     else
//         printf("\n");
// }

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

class DescriptorQualityEvaluator : public BaseQualityEvaluator
{
public:
    enum{ NO_MATCH_FILTER = 0 };
    DescriptorQualityEvaluator( const char* _descriptorName, const char* _testName, const char* _matcherName = 0 ) :
            BaseQualityEvaluator( _descriptorName, _testName )
    {
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
    virtual string getRunParamsFilename() const;
    virtual string getResultsFilename() const;
    virtual string getPlotPath() const;

    virtual void calcQualityClear( int datasetIdx );
    virtual bool isCalcQualityEmpty( int datasetIdx ) const;

    virtual void readDatasetRunParams( FileNode& fn, int datasetIdx ); //
    virtual void writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const;
    virtual void setDefaultDatasetRunParams( int datasetIdx );
    virtual void readDefaultRunParams( FileNode &fn );
    virtual void writeDefaultRunParams( FileStorage &fs ) const;

    virtual void readAlgorithm();
    virtual void processRunParamsFile() {}
    virtual void runDatasetTest( const vector<Mat> &imgs, const vector<Mat> &Hs, int di, int &progress );

    virtual void writePlotData( int di ) const;
    void calculatePlotData( vector<vector<DMatch> > &allMatches, vector<vector<uchar> > &allCorrectMatchesMask, int di );

    struct Quality
    {
        float recall;
        float precision;
    };
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

    Ptr<GenericDescriptorMatch> specificDescMatcher;
    Ptr<GenericDescriptorMatch> defaultDescMatcher;

    CommonRunParams commRunParamsDefault;
    string matcherName;
};

string DescriptorQualityEvaluator::getRunParamsFilename() const
{
    return data_path + DESCRIPTORS_DIR + algName + PARAMS_POSTFIX;
}

string DescriptorQualityEvaluator::getResultsFilename() const
{
    return data_path + DESCRIPTORS_DIR + algName + RES_POSTFIX;
}

string DescriptorQualityEvaluator::getPlotPath() const
{
    return data_path + DESCRIPTORS_DIR + "plots/";
}

void DescriptorQualityEvaluator::calcQualityClear( int datasetIdx )
{
    calcQuality[datasetIdx].clear();
}

bool DescriptorQualityEvaluator::isCalcQualityEmpty( int datasetIdx ) const
{
    return calcQuality[datasetIdx].empty();
}

void DescriptorQualityEvaluator::readDefaultRunParams (FileNode &fn)
{
    if (! fn.empty() )
    {
        commRunParamsDefault.projectKeypointsFrom1Image = (int)fn[PROJECT_KEYPOINTS_FROM_1IMAGE] != 0;
        commRunParamsDefault.matchFilter = (int)fn[MATCH_FILTER];
        defaultDescMatcher->read (fn);
    }
}

void DescriptorQualityEvaluator::writeDefaultRunParams (FileStorage &fs) const
{
    fs << PROJECT_KEYPOINTS_FROM_1IMAGE << commRunParamsDefault.projectKeypointsFrom1Image;
    fs << MATCH_FILTER << commRunParamsDefault.matchFilter;
    defaultDescMatcher->write (fs);
}

void DescriptorQualityEvaluator::readDatasetRunParams( FileNode& fn, int datasetIdx )
{
    commRunParams[datasetIdx].isActiveParams = (int)fn[IS_ACTIVE_PARAMS] != 0;
    if (commRunParams[datasetIdx].isActiveParams)
    {
        commRunParams[datasetIdx].keypontsFilename = (string)fn[KEYPOINTS_FILENAME];
        commRunParams[datasetIdx].projectKeypointsFrom1Image = (int)fn[PROJECT_KEYPOINTS_FROM_1IMAGE] != 0;
        commRunParams[datasetIdx].matchFilter = (int)fn[MATCH_FILTER];
        specificDescMatcher->read (fn);
    }
    else
    {
        setDefaultDatasetRunParams(datasetIdx);
    }
}

void DescriptorQualityEvaluator::writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const
{
    fs << IS_ACTIVE_PARAMS << commRunParams[datasetIdx].isActiveParams;
    fs << KEYPOINTS_FILENAME << commRunParams[datasetIdx].keypontsFilename;
    fs << PROJECT_KEYPOINTS_FROM_1IMAGE << commRunParams[datasetIdx].projectKeypointsFrom1Image;
    fs << MATCH_FILTER << commRunParams[datasetIdx].matchFilter;

    defaultDescMatcher->write (fs);
}

void DescriptorQualityEvaluator::setDefaultDatasetRunParams( int datasetIdx )
{
    commRunParams[datasetIdx] = commRunParamsDefault;
    commRunParams[datasetIdx].keypontsFilename = "SURF_" + DATASET_NAMES[datasetIdx] + ".xml.gz";
}

void DescriptorQualityEvaluator::writePlotData( int di ) const
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

void DescriptorQualityEvaluator::readAlgorithm( )
{
    defaultDescMatcher = GenericDescriptorMatcher::create( algName );
    specificDescMatcher = GenericDescriptorMatcher::create( algName );

    if( defaultDescMatcher == 0 )
    {
        Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create( algName );
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( matcherName );
        defaultDescMatcher = new VectorDescriptorMatch( extractor, matcher );
        specificDescMatcher = new VectorDescriptorMatch( extractor, matcher );

        if( extractor == 0 || matcher == 0 )
        {
            printf("Algorithm can not be read\n");
            exit(-1);
        }
    }
}

void DescriptorQualityEvaluator::calculatePlotData( vector<vector<DMatch> > &allMatches, vector<vector<uchar> > &allCorrectMatchesMask, int di )
{
    vector<Point2f> recallPrecisionCurve;
    computeRecallPrecisionCurve( allMatches, allCorrectMatchesMask, recallPrecisionCurve );

    calcDatasetQuality[di].clear();
    const float resultPrecision = 0.5;
    bool isResultCalculated = false;
    const double eps = 1e-2;

    Quality initQuality;
    initQuality.recall = 0;
    initQuality.precision = 0;
    calcDatasetQuality[di].push_back( initQuality );

    for( size_t i=0;i<recallPrecisionCurve.size();i++ )
    {
        Quality quality;
        quality.recall = recallPrecisionCurve[i].y;
        quality.precision = 1 - recallPrecisionCurve[i].x;
        Quality back = calcDatasetQuality[di].back();

        if( fabs( quality.recall - back.recall ) < eps && fabs( quality.precision - back.precision ) < eps )
            continue;

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

void DescriptorQualityEvaluator::runDatasetTest (const vector<Mat> &imgs, const vector<Mat> &Hs, int di, int &progress)
{
    FileStorage keypontsFS( data_path + KEYPOINTS_DIR + commRunParams[di].keypontsFilename, FileStorage::READ );
    if( !keypontsFS.isOpened())
    {
       calcQuality[di].clear();
       printf( "keypoints from file %s can not be read\n", commRunParams[di].keypontsFilename.c_str() );
       return;
    }

    Ptr<GenericDescriptorMatcher> descMatch = commRunParams[di].isActiveParams ? specificDescMatcher : defaultDescMatcher;
    calcQuality[di].resize(TEST_CASE_COUNT);

    vector<KeyPoint> keypoints1;
    readKeypoints( keypontsFS, keypoints1, 0);

    int progressCount = DATASETS_COUNT*TEST_CASE_COUNT;

    vector<vector<DMatch> > allMatches1to2;
    vector<vector<uchar> > allCorrectMatchesMask;
    for( int ci = 0; ci < TEST_CASE_COUNT; ci++ )
    {
        progress = update_progress( testName, progress, di*TEST_CASE_COUNT + ci + 1, progressCount, 0 );

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
        evaluateGenericDescriptorMatcher( imgs[0], imgs[ci+1], Hs[ci], keypoints1, keypoints2,
                                          &matches1to2, &correctMatchesMask, recallPrecisionCurve,
                                          descMatch );
        allMatches1to2.insert( allMatches1to2.end(), matches1to2.begin(), matches1to2.end() );
        allCorrectMatchesMask.insert( allCorrectMatchesMask.end(), correctMatchesMask.begin(), correctMatchesMask.end() );
    }

    calculatePlotData( allMatches1to2, allCorrectMatchesMask, di );
}

//--------------------------------- Calonder descriptor test --------------------------------------------
class CalonderDescriptorQualityEvaluator : public DescriptorQualityEvaluator
{
public:
    CalonderDescriptorQualityEvaluator() :
            DescriptorQualityEvaluator( "Calonder", "quality-descriptor-calonder") {}
    virtual void readAlgorithm( )
    {
        string classifierFile = data_path + "/features2d/calonder_classifier.rtc";
        defaultDescMatcher = new VectorDescriptorMatch( new CalonderDescriptorExtractor<float>( classifierFile ),
                                                        new BFMatcher(NORM_L2) );
        specificDescMatcher = defaultDescMatcher;
    }
};

//--------------------------------- One Way descriptor test --------------------------------------------
class OneWayDescriptorQualityTest : public DescriptorQualityEvaluator
{
public:
    OneWayDescriptorQualityTest() :
        DescriptorQualityEvaluator("ONEWAY", "quality-descriptor-one-way")
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

    string pcaFilename = data_path + (string)fn["pcaFilename"];
    string trainPath = data_path + (string)fn["trainPath"];
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
    defaultDescMatcher = match;
    writeAllDatasetsRunParams();
}

void OneWayDescriptorQualityTest::writeDatasetRunParams( FileStorage& fs, int datasetIdx ) const
{
    fs << IS_ACTIVE_PARAMS << commRunParams[datasetIdx].isActiveParams;
    fs << KEYPOINTS_FILENAME << commRunParams[datasetIdx].keypontsFilename;
    fs << PROJECT_KEYPOINTS_FROM_1IMAGE << commRunParams[datasetIdx].projectKeypointsFrom1Image;
    fs << MATCH_FILTER << commRunParams[datasetIdx].matchFilter;
}

int main( int argc, char** argv )
{
    if( argc != 2 )
    {
        cout << "Format: " << argv[0] << " testdata path (path to testdata/cv)" << endl;
        return -1;
    }

    data_path = argv[1];
#ifdef WIN32
    if( *data_path.rbegin() != '\\' )
        data_path = data_path + "\\";
#else
    if( *data_path.rbegin() != '/' )
        data_path = data_path + "/";
#endif

    Ptr<BaseQualityEvaluator> evals[] =
    {
        new DetectorQualityEvaluator( "FAST", "quality-detector-fast" ),
        new DetectorQualityEvaluator( "GFTT", "quality-detector-gftt" ),
        new DetectorQualityEvaluator( "HARRIS", "quality-detector-harris" ),
        new DetectorQualityEvaluator( "MSER", "quality-detector-mser" ),
        new DetectorQualityEvaluator( "STAR", "quality-detector-star" ),
        new DetectorQualityEvaluator( "SIFT", "quality-detector-sift" ),
        new DetectorQualityEvaluator( "SURF", "quality-detector-surf" ),

        new DescriptorQualityEvaluator( "SIFT", "quality-descriptor-sift", "BruteForce" ),
        new DescriptorQualityEvaluator( "SURF", "quality-descriptor-surf", "BruteForce" ),
        new DescriptorQualityEvaluator( "FERN", "quality-descriptor-fern"),
        new CalonderDescriptorQualityEvaluator()
    };

    for( size_t i = 0; i < sizeof(evals)/sizeof(evals[0]); i++ )
    {
        evals[i]->run();
        cout << endl;
    }
}
