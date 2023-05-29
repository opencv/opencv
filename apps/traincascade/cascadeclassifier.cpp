#include "opencv2/core.hpp"

#include "cascadeclassifier.h"
#include <queue>

using namespace std;
using namespace cv;

static const char* stageTypes[] = { CC_BOOST };
static const char* featureTypes[] = { CC_HAAR, CC_LBP, CC_HOG };

CvCascadeParams::CvCascadeParams() : stageType( defaultStageType ),
    featureType( defaultFeatureType ), winSize( cvSize(24, 24) )
{
    name = CC_CASCADE_PARAMS;
}
CvCascadeParams::CvCascadeParams( int _stageType, int _featureType ) : stageType( _stageType ),
    featureType( _featureType ), winSize( cvSize(24, 24) )
{
    name = CC_CASCADE_PARAMS;
}

//---------------------------- CascadeParams --------------------------------------

void CvCascadeParams::write( FileStorage &fs ) const
{
    string stageTypeStr = stageType == BOOST ? CC_BOOST : string();
    CV_Assert( !stageTypeStr.empty() );
    fs << CC_STAGE_TYPE << stageTypeStr;
    string featureTypeStr = featureType == CvFeatureParams::HAAR ? CC_HAAR :
                            featureType == CvFeatureParams::LBP ? CC_LBP :
                            featureType == CvFeatureParams::HOG ? CC_HOG :
                            0;
    CV_Assert( !stageTypeStr.empty() );
    fs << CC_FEATURE_TYPE << featureTypeStr;
    fs << CC_HEIGHT << winSize.height;
    fs << CC_WIDTH << winSize.width;
}

bool CvCascadeParams::read( const FileNode &node )
{
    if ( node.empty() )
        return false;
    string stageTypeStr, featureTypeStr;
    FileNode rnode = node[CC_STAGE_TYPE];
    if ( !rnode.isString() )
        return false;
    rnode >> stageTypeStr;
    stageType = !stageTypeStr.compare( CC_BOOST ) ? BOOST : -1;
    if (stageType == -1)
        return false;
    rnode = node[CC_FEATURE_TYPE];
    if ( !rnode.isString() )
        return false;
    rnode >> featureTypeStr;
    featureType = !featureTypeStr.compare( CC_HAAR ) ? CvFeatureParams::HAAR :
                  !featureTypeStr.compare( CC_LBP ) ? CvFeatureParams::LBP :
                  !featureTypeStr.compare( CC_HOG ) ? CvFeatureParams::HOG :
                  -1;
    if (featureType == -1)
        return false;
    node[CC_HEIGHT] >> winSize.height;
    node[CC_WIDTH] >> winSize.width;
    return winSize.height > 0 && winSize.width > 0;
}

void CvCascadeParams::printDefaults() const
{
    CvParams::printDefaults();
    cout << "  [-stageType <";
    for( int i = 0; i < (int)(sizeof(stageTypes)/sizeof(stageTypes[0])); i++ )
    {
        cout << (i ? " | " : "") << stageTypes[i];
        if ( i == defaultStageType )
            cout << "(default)";
    }
    cout << ">]" << endl;

    cout << "  [-featureType <{";
    for( int i = 0; i < (int)(sizeof(featureTypes)/sizeof(featureTypes[0])); i++ )
    {
        cout << (i ? ", " : "") << featureTypes[i];
        if ( i == defaultStageType )
            cout << "(default)";
    }
    cout << "}>]" << endl;
    cout << "  [-w <sampleWidth = " << winSize.width << ">]" << endl;
    cout << "  [-h <sampleHeight = " << winSize.height << ">]" << endl;
}

void CvCascadeParams::printAttrs() const
{
    cout << "stageType: " << stageTypes[stageType] << endl;
    cout << "featureType: " << featureTypes[featureType] << endl;
    cout << "sampleWidth: " << winSize.width << endl;
    cout << "sampleHeight: " << winSize.height << endl;
}

bool CvCascadeParams::scanAttr( const string prmName, const string val )
{
    bool res = true;
    if( !prmName.compare( "-stageType" ) )
    {
        for( int i = 0; i < (int)(sizeof(stageTypes)/sizeof(stageTypes[0])); i++ )
            if( !val.compare( stageTypes[i] ) )
                stageType = i;
    }
    else if( !prmName.compare( "-featureType" ) )
    {
        for( int i = 0; i < (int)(sizeof(featureTypes)/sizeof(featureTypes[0])); i++ )
            if( !val.compare( featureTypes[i] ) )
                featureType = i;
    }
    else if( !prmName.compare( "-w" ) )
    {
        winSize.width = atoi( val.c_str() );
    }
    else if( !prmName.compare( "-h" ) )
    {
        winSize.height = atoi( val.c_str() );
    }
    else
        res = false;
    return res;
}

//---------------------------- CascadeClassifier --------------------------------------

bool CvCascadeClassifier::train( const string _cascadeDirName,
                                const string _posFilename,
                                const string _negFilename,
                                int _numPos, int _numNeg,
                                int _precalcValBufSize, int _precalcIdxBufSize,
                                int _numStages,
                                const CvCascadeParams& _cascadeParams,
                                const CvFeatureParams& _featureParams,
                                const CvCascadeBoostParams& _stageParams,
                                bool baseFormatSave,
                                double acceptanceRatioBreakValue )
{
    // Start recording clock ticks for training time output
    double time = (double)getTickCount();

    if( _cascadeDirName.empty() || _posFilename.empty() || _negFilename.empty() )
        CV_Error( CV_StsBadArg, "_cascadeDirName or _bgfileName or _vecFileName is NULL" );

    string dirName;
    if (_cascadeDirName.find_last_of("/\\") == (_cascadeDirName.length() - 1) )
        dirName = _cascadeDirName;
    else
        dirName = _cascadeDirName + '/';

    numPos = _numPos;
    numNeg = _numNeg;
    numStages = _numStages;
    if ( !imgReader.create( _posFilename, _negFilename, _cascadeParams.winSize ) )
    {
        cout << "Image reader can not be created from -vec " << _posFilename
                << " and -bg " << _negFilename << "." << endl;
        return false;
    }
    if ( !load( dirName ) )
    {
        cascadeParams = _cascadeParams;
        featureParams = CvFeatureParams::create(cascadeParams.featureType);
        featureParams->init(_featureParams);
        stageParams = makePtr<CvCascadeBoostParams>();
        *stageParams = _stageParams;
        featureEvaluator = CvFeatureEvaluator::create(cascadeParams.featureType);
        featureEvaluator->init( featureParams, numPos + numNeg, cascadeParams.winSize );
        stageClassifiers.reserve( numStages );
    }else{
        // Make sure that if model parameters are preloaded, that people are aware of this,
        // even when passing other parameters to the training command
        cout << "---------------------------------------------------------------------------------" << endl;
        cout << "Training parameters are pre-loaded from the parameter file in data folder!" << endl;
        cout << "Please empty this folder if you want to use a NEW set of training parameters." << endl;
        cout << "---------------------------------------------------------------------------------" << endl;
    }
    cout << "PARAMETERS:" << endl;
    cout << "cascadeDirName: " << _cascadeDirName << endl;
    cout << "vecFileName: " << _posFilename << endl;
    cout << "bgFileName: " << _negFilename << endl;
    cout << "numPos: " << _numPos << endl;
    cout << "numNeg: " << _numNeg << endl;
    cout << "numStages: " << numStages << endl;
    cout << "precalcValBufSize[Mb] : " << _precalcValBufSize << endl;
    cout << "precalcIdxBufSize[Mb] : " << _precalcIdxBufSize << endl;
    cout << "acceptanceRatioBreakValue : " << acceptanceRatioBreakValue << endl;
    cascadeParams.printAttrs();
    stageParams->printAttrs();
    featureParams->printAttrs();
    cout << "Number of unique features given windowSize [" << _cascadeParams.winSize.width << "," << _cascadeParams.winSize.height << "] : " << featureEvaluator->getNumFeatures() << "" << endl;

    int startNumStages = (int)stageClassifiers.size();
    if ( startNumStages > 1 )
        cout << endl << "Stages 0-" << startNumStages-1 << " are loaded" << endl;
    else if ( startNumStages == 1)
        cout << endl << "Stage 0 is loaded" << endl;

    double requiredLeafFARate = pow( (double) stageParams->maxFalseAlarm, (double) numStages ) /
                                (double)stageParams->max_depth;
    double tempLeafFARate;

    for( int i = startNumStages; i < numStages; i++ )
    {
        cout << endl << "===== TRAINING " << i << "-stage =====" << endl;
        cout << "<BEGIN" << endl;

        if ( !updateTrainingSet( requiredLeafFARate, tempLeafFARate ) )
        {
            cout << "Train dataset for temp stage can not be filled. "
                    "Branch training terminated." << endl;
            break;
        }
        if( tempLeafFARate <= requiredLeafFARate )
        {
            cout << "Required leaf false alarm rate achieved. "
                    "Branch training terminated." << endl;
            break;
        }
        if( (tempLeafFARate <= acceptanceRatioBreakValue) && (acceptanceRatioBreakValue >= 0) ){
            cout << "The required acceptanceRatio for the model has been reached to avoid overfitting of trainingdata. "
                    "Branch training terminated." << endl;
            break;
        }

        Ptr<CvCascadeBoost> tempStage = makePtr<CvCascadeBoost>();
        bool isStageTrained = tempStage->train( featureEvaluator,
                                                curNumSamples, _precalcValBufSize, _precalcIdxBufSize,
                                                *stageParams );
        cout << "END>" << endl;

        if(!isStageTrained)
            break;

        stageClassifiers.push_back( tempStage );

        // save params
        if( i == 0)
        {
            std::string paramsFilename = dirName + CC_PARAMS_FILENAME;
            FileStorage fs( paramsFilename, FileStorage::WRITE);
            if ( !fs.isOpened() )
            {
                cout << "Parameters can not be written, because file " << paramsFilename
                        << " can not be opened." << endl;
                return false;
            }
            fs << FileStorage::getDefaultObjectName(paramsFilename) << "{";
            writeParams( fs );
            fs << "}";
        }
        // save current stage
        char buf[32];
        snprintf(buf, sizeof(buf), "%s%d", "stage", i );
        string stageFilename = dirName + buf + ".xml";
        FileStorage fs( stageFilename, FileStorage::WRITE );
        if ( !fs.isOpened() )
        {
            cout << "Current stage can not be written, because file " << stageFilename
                    << " can not be opened." << endl;
            return false;
        }
        fs << FileStorage::getDefaultObjectName(stageFilename) << "{";
        tempStage->write( fs, Mat() );
        fs << "}";

        // Output training time up till now
        double seconds = ( (double)getTickCount() - time)/ getTickFrequency();
        int days = int(seconds) / 60 / 60 / 24;
        int hours = (int(seconds) / 60 / 60) % 24;
        int minutes = (int(seconds) / 60) % 60;
        int seconds_left = int(seconds) % 60;
        cout << "Training until now has taken " << days << " days " << hours << " hours " << minutes << " minutes " << seconds_left <<" seconds." << endl;
    }

    if(stageClassifiers.size() == 0)
    {
        cout << "Cascade classifier can't be trained. Check the used training parameters." << endl;
        return false;
    }

    save( dirName + CC_CASCADE_FILENAME, baseFormatSave );

    return true;
}

int CvCascadeClassifier::predict( int sampleIdx )
{
    CV_DbgAssert( sampleIdx < numPos + numNeg );
    for (vector< Ptr<CvCascadeBoost> >::iterator it = stageClassifiers.begin();
        it != stageClassifiers.end();++it )
    {
        if ( (*it)->predict( sampleIdx ) == 0.f )
            return 0;
    }
    return 1;
}

bool CvCascadeClassifier::updateTrainingSet( double minimumAcceptanceRatio, double& acceptanceRatio)
{
    int64 posConsumed = 0, negConsumed = 0;
    imgReader.restart();
    int posCount = fillPassedSamples( 0, numPos, true, 0, posConsumed );
    if( !posCount )
        return false;
    cout << "POS count : consumed   " << posCount << " : " << (int)posConsumed << endl;

    int proNumNeg = cvRound( ( ((double)numNeg) * ((double)posCount) ) / numPos ); // apply only a fraction of negative samples. double is required since overflow is possible
    int negCount = fillPassedSamples( posCount, proNumNeg, false, minimumAcceptanceRatio, negConsumed );
    if ( !negCount )
        if ( !(negConsumed > 0 && ((double)negCount+1)/(double)negConsumed <= minimumAcceptanceRatio) )
            return false;

    curNumSamples = posCount + negCount;
    acceptanceRatio = negConsumed == 0 ? 0 : ( (double)negCount/(double)(int64)negConsumed );
    cout << "NEG count : acceptanceRatio    " << negCount << " : " << acceptanceRatio << endl;
    return true;
}

int CvCascadeClassifier::fillPassedSamples( int first, int count, bool isPositive, double minimumAcceptanceRatio, int64& consumed )
{
    int getcount = 0;
    Mat img(cascadeParams.winSize, CV_8UC1);
    for( int i = first; i < first + count; i++ )
    {
        for( ; ; )
        {
            if( consumed != 0 && ((double)getcount+1)/(double)(int64)consumed <= minimumAcceptanceRatio )
                return getcount;

            bool isGetImg = isPositive ? imgReader.getPos( img ) :
                                           imgReader.getNeg( img );
            if( !isGetImg )
                return getcount;
            consumed++;

            featureEvaluator->setImage( img, isPositive ? 1 : 0, i );
            if( predict( i ) == 1 )
            {
                getcount++;
                printf("%s current samples: %d\r", isPositive ? "POS":"NEG", getcount);
                fflush(stdout);
                break;
            }
        }
    }
    return getcount;
}

void CvCascadeClassifier::writeParams( FileStorage &fs ) const
{
    cascadeParams.write( fs );
    fs << CC_STAGE_PARAMS << "{"; stageParams->write( fs ); fs << "}";
    fs << CC_FEATURE_PARAMS << "{"; featureParams->write( fs ); fs << "}";
}

void CvCascadeClassifier::writeFeatures( FileStorage &fs, const Mat& featureMap ) const
{
    featureEvaluator->writeFeatures( fs, featureMap );
}

void CvCascadeClassifier::writeStages( FileStorage &fs, const Mat& featureMap ) const
{
    char cmnt[30];
    int i = 0;
    fs << CC_STAGES << "[";
    for( vector< Ptr<CvCascadeBoost> >::const_iterator it = stageClassifiers.begin();
        it != stageClassifiers.end();++it, ++i )
    {
        snprintf( cmnt, sizeof(cmnt), "stage %d", i );
        fs.writeComment(cmnt);
        fs << "{";
        (*it)->write( fs, featureMap );
        fs << "}";
    }
    fs << "]";
}

bool CvCascadeClassifier::readParams( const FileNode &node )
{
    if ( !node.isMap() || !cascadeParams.read( node ) )
        return false;

    stageParams = makePtr<CvCascadeBoostParams>();
    FileNode rnode = node[CC_STAGE_PARAMS];
    if ( !stageParams->read( rnode ) )
        return false;

    featureParams = CvFeatureParams::create(cascadeParams.featureType);
    rnode = node[CC_FEATURE_PARAMS];
    if ( !featureParams->read( rnode ) )
        return false;
    return true;
}

bool CvCascadeClassifier::readStages( const FileNode &node)
{
    FileNode rnode = node[CC_STAGES];
    if (!rnode.empty() || !rnode.isSeq())
        return false;
    stageClassifiers.reserve(numStages);
    FileNodeIterator it = rnode.begin();
    for( int i = 0; i < min( (int)rnode.size(), numStages ); i++, it++ )
    {
        Ptr<CvCascadeBoost> tempStage = makePtr<CvCascadeBoost>();
        if ( !tempStage->read( *it, featureEvaluator, *stageParams) )
            return false;
        stageClassifiers.push_back(tempStage);
    }
    return true;
}

// For old Haar Classifier file saving
#define ICV_HAAR_TYPE_ID              "opencv-haar-classifier"
#define ICV_HAAR_SIZE_NAME            "size"
#define ICV_HAAR_STAGES_NAME          "stages"
#define ICV_HAAR_TREES_NAME             "trees"
#define ICV_HAAR_FEATURE_NAME             "feature"
#define ICV_HAAR_RECTS_NAME                 "rects"
#define ICV_HAAR_TILTED_NAME                "tilted"
#define ICV_HAAR_THRESHOLD_NAME           "threshold"
#define ICV_HAAR_LEFT_NODE_NAME           "left_node"
#define ICV_HAAR_LEFT_VAL_NAME            "left_val"
#define ICV_HAAR_RIGHT_NODE_NAME          "right_node"
#define ICV_HAAR_RIGHT_VAL_NAME           "right_val"
#define ICV_HAAR_STAGE_THRESHOLD_NAME   "stage_threshold"
#define ICV_HAAR_PARENT_NAME            "parent"
#define ICV_HAAR_NEXT_NAME              "next"

void CvCascadeClassifier::save( const string filename, bool baseFormat )
{
    FileStorage fs( filename, FileStorage::WRITE );

    if ( !fs.isOpened() )
        return;

    fs << FileStorage::getDefaultObjectName(filename);
    if ( !baseFormat )
    {
        Mat featureMap;
        getUsedFeaturesIdxMap( featureMap );
        fs << "{";
        writeParams( fs );
        fs << CC_STAGE_NUM << (int)stageClassifiers.size();
        writeStages( fs, featureMap );
        writeFeatures( fs, featureMap );
    }
    else
    {
        //char buf[256];
        CvSeq* weak;
        if ( cascadeParams.featureType != CvFeatureParams::HAAR )
            CV_Error( CV_StsBadFunc, "old file format is used for Haar-like features only");
        fs << "{:" ICV_HAAR_TYPE_ID;
        fs << ICV_HAAR_SIZE_NAME << "[:" << cascadeParams.winSize.width <<
            cascadeParams.winSize.height << "]";
        fs << ICV_HAAR_STAGES_NAME << "[";
        for( size_t si = 0; si < stageClassifiers.size(); si++ )
        {
            fs << "{"; //stage
            /*snprintf( buf, sizeof(buf), "stage %d", si );
            CV_CALL( cvWriteComment( fs, buf, 1 ) );*/
            weak = stageClassifiers[si]->get_weak_predictors();
            fs << ICV_HAAR_TREES_NAME << "[";
            for( int wi = 0; wi < weak->total; wi++ )
            {
                int total_inner_node_idx = -1;
                queue<const CvDTreeNode*> inner_nodes_queue;
                CvCascadeBoostTree* tree = *((CvCascadeBoostTree**) cvGetSeqElem( weak, wi ));

                fs << "[";
                /*snprintf( buf, sizeof(buf), "tree %d", wi );
                CV_CALL( cvWriteComment( fs, buf, 1 ) );*/

                const CvDTreeNode* tempNode;

                inner_nodes_queue.push( tree->get_root() );
                total_inner_node_idx++;

                while (!inner_nodes_queue.empty())
                {
                    tempNode = inner_nodes_queue.front();

                    fs << "{";
                    fs << ICV_HAAR_FEATURE_NAME << "{";
                    ((CvHaarEvaluator*)featureEvaluator.get())->writeFeature( fs, tempNode->split->var_idx );
                    fs << "}";

                    fs << ICV_HAAR_THRESHOLD_NAME << tempNode->split->ord.c;

                    if( tempNode->left->left || tempNode->left->right )
                    {
                        inner_nodes_queue.push( tempNode->left );
                        total_inner_node_idx++;
                        fs << ICV_HAAR_LEFT_NODE_NAME << total_inner_node_idx;
                    }
                    else
                        fs << ICV_HAAR_LEFT_VAL_NAME << tempNode->left->value;

                    if( tempNode->right->left || tempNode->right->right )
                    {
                        inner_nodes_queue.push( tempNode->right );
                        total_inner_node_idx++;
                        fs << ICV_HAAR_RIGHT_NODE_NAME << total_inner_node_idx;
                    }
                    else
                        fs << ICV_HAAR_RIGHT_VAL_NAME << tempNode->right->value;
                    fs << "}"; // ICV_HAAR_FEATURE_NAME
                    inner_nodes_queue.pop();
                }
                fs << "]";
            }
            fs << "]"; //ICV_HAAR_TREES_NAME
            fs << ICV_HAAR_STAGE_THRESHOLD_NAME << stageClassifiers[si]->getThreshold();
            fs << ICV_HAAR_PARENT_NAME << (int)si-1 << ICV_HAAR_NEXT_NAME << -1;
            fs << "}"; //stage
        } /* for each stage */
        fs << "]"; //ICV_HAAR_STAGES_NAME
    }
    fs << "}";
}

bool CvCascadeClassifier::load( const string cascadeDirName )
{
    FileStorage fs( cascadeDirName + CC_PARAMS_FILENAME, FileStorage::READ );
    if ( !fs.isOpened() )
        return false;
    FileNode node = fs.getFirstTopLevelNode();
    if ( !readParams( node ) )
        return false;
    featureEvaluator = CvFeatureEvaluator::create(cascadeParams.featureType);
    featureEvaluator->init( featureParams, numPos + numNeg, cascadeParams.winSize );
    fs.release();

    char buf[5+10+1] = {0};
    for ( int si = 0; si < numStages; si++ )
    {
        snprintf( buf, sizeof(buf), "%s%d", "stage", si);
        fs.open( cascadeDirName + buf + ".xml", FileStorage::READ );
        node = fs.getFirstTopLevelNode();
        if ( !fs.isOpened() )
            break;
        Ptr<CvCascadeBoost> tempStage = makePtr<CvCascadeBoost>();

        if ( !tempStage->read( node, featureEvaluator, *stageParams ))
        {
            fs.release();
            break;
        }
        stageClassifiers.push_back(tempStage);
    }
    return true;
}

void CvCascadeClassifier::getUsedFeaturesIdxMap( Mat& featureMap )
{
    int varCount = featureEvaluator->getNumFeatures() * featureEvaluator->getFeatureSize();
    featureMap.create( 1, varCount, CV_32SC1 );
    featureMap.setTo(Scalar(-1));

    for( vector< Ptr<CvCascadeBoost> >::const_iterator it = stageClassifiers.begin();
        it != stageClassifiers.end();++it )
        (*it)->markUsedFeaturesInMap( featureMap );

    for( int fi = 0, idx = 0; fi < varCount; fi++ )
        if ( featureMap.at<int>(0, fi) >= 0 )
            featureMap.ptr<int>(0)[fi] = idx++;
}
