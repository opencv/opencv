#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"

#include "common.h"
#include "cascadeclassifier.h"
#include <queue>

using namespace std;
using namespace cv;

static const char* stageTypes[] = { CC_BOOST };
static const char* featureTypes[] = { CC_HAAR, CC_LBP, CC_HOG, CC_MBLBP };

PosImageReader::PosImageReader()
{
    file = 0;
    vec = 0;
}
PosImageReader::~PosImageReader()
{
    if (file)
        fclose( file );
    cvFree( &vec );
}

bool PosImageReader::create( const string _filename )
{
    if ( file )
        fclose( file );
    file = fopen( _filename.c_str(), "rb" );
    
    if( !file )
        return false;
    short tmp = 0;
    if( fread( &count, sizeof( count ), 1, file ) != 1 ||
        fread( &vecSize, sizeof( vecSize ), 1, file ) != 1 ||
        fread( &tmp, sizeof( tmp ), 1, file ) != 1 ||
        fread( &tmp, sizeof( tmp ), 1, file ) != 1 )
        CV_Error_( CV_StsParseError, ("wrong file format for %s\n", _filename.c_str()) );
    base = sizeof( count ) + sizeof( vecSize ) + 2*sizeof( tmp );
    if( feof( file ) )
        return false;
    last = 0;

    if(vec) cvFree(&vec);
    vec = (short*) cvAlloc( sizeof( *vec ) * vecSize );

    CV_Assert( vec );
    return true;
}

bool PosImageReader::get( Mat &_img )
{
    cout<<"img row:"<<_img.rows <<" img cols:"<<_img.cols<<endl;
    cout<<"vec Size:"<<vecSize<<endl;
    CV_Assert( _img.rows * _img.cols == vecSize );
    uchar tmp = 0;
    fread( &tmp, sizeof( tmp ), 1, file );
    fread( vec, sizeof( vec[0] ), vecSize, file );
    
    if( feof( file ) || last++ >= count )
        return false;
    
    for( int r = 0; r < _img.rows; r++ )
    {
        for( int c = 0; c < _img.cols; c++ )
            _img.ptr(r)[c] = (uchar)vec[r * _img.cols + c];
    }

    //blur(_img, _img, Size(3,3));
    return true;
}

void PosImageReader::restart()
{
    CV_Assert( file );
    last = 0;
    fseek( file, base, SEEK_SET );
}


NegImageReader::NegImageReader()
{
    src.create( 0, 0 , CV_8UC1 );
    img.create( 0, 0, CV_8UC1 );
    //imgsum.create(0, 0, CV_32SC1);
    point = offset = Point( 0, 0 );
    scale       = 1.0F;
    scaleFactor = 1.4142135623730950488016887242097F;
    stepFactor  = 0.5F;
}

bool NegImageReader::create( const string _filename, Size _winSize )
{
    string dirname, str;
    ifstream file(_filename.c_str());
    if ( !file.is_open() )
        return false;
    
    size_t pos = _filename.rfind('\\');
    char dlmrt = '\\';
    if (pos == String::npos)
    {
        pos = _filename.rfind('/');
        dlmrt = '/';
    }
    dirname = pos == String::npos ? "" : _filename.substr(0, pos) + dlmrt;
    while( !file.eof() )
    {
        getline(file, str);
        if (str.empty()) break;
        if (str.at(0) == '#' ) continue; /* comment */
        imgFilenames.push_back(dirname + str);
    }
    file.close();
    
    winSize = _winSize;
    last = round = 0;
    return true;
}

bool NegImageReader::nextImg()
{
	Point _offset = Point(0,0);
	size_t count = imgFilenames.size();
    
	for( size_t i = 0; i < count; i++ )
	{
		src = imread( imgFilenames[last++], 0 );
		//cout <<  imgFilenames[last-1] << endl;
		if( src.empty() )
			continue;
		round += last / count;
		round = round % (winSize.width * winSize.height);
		last %= count;
        
		_offset.x = min( (int)round % winSize.width, src.cols - winSize.width );
		_offset.y = min( (int)round / winSize.width, src.rows - winSize.height );
		if( !src.empty() && src.type() == CV_8UC1
           && _offset.x >= 0 && _offset.y >= 0 )
			break;
	}
    
 	if( src.empty() )
 		return false; // no appropriate image
    
	point = offset = _offset;
	scale = max( ((float)winSize.width + point.x) / ((float)src.cols),
                ((float)winSize.height + point.y) / ((float)src.rows) );
    
	Size sz( (int)(scale*src.cols + 0.5F), (int)(scale*src.rows + 0.5F) );
	resize( src, img, sz );
    //blur(img, img, Size(3,3));
	//integral(img, imgsum );

	return true;
}
//read an image, and calculate the image channels
bool NegImageReader::nextImg2(MBLBPCascadef * pCascade)
{
    if(pCascade->count > 0)
    {
        if(pCascade->stages[pCascade->count-1]->false_alarm >= 1.0e-2)
            stepFactor = 1.f;
        else if(pCascade->stages[pCascade->count-1]->false_alarm < 1.0e-2 &&
                pCascade->stages[pCascade->count-1]->false_alarm >= 1.0e-4)
            stepFactor = 0.5f;
        else if(pCascade->stages[pCascade->count-1]->false_alarm < 1.0e-4 && 
                pCascade->stages[pCascade->count-1]->false_alarm >= 1.0e-5)
            stepFactor = 0.3f;
        else if(pCascade->stages[pCascade->count-1]->false_alarm < 1.0e-5)
            stepFactor = 0.125f;
    }

    Point _offset = Point(0,0);
	size_t count = imgFilenames.size();
    
	for( size_t i = 0; i < count; i++ )
	{
		src = imread( imgFilenames[last++], 0 );
		//cout <<  imgFilenames[last-1] << endl;
		round += last / count;
		round = round % (winSize.width * winSize.height);
		last %= count;
        
		if( src.empty() )
			continue;

        _offset.x = min( (int)round % winSize.width, src.cols - winSize.width );
		_offset.y = min( (int)round / winSize.width, src.rows - winSize.height );
		if( !src.empty() && src.type() == CV_8UC1
           && _offset.x >= 0 && _offset.y >= 0 )
			break;
	}
    
 	if( src.empty() )
 		return false; // no appropriate image
    
	point = offset = _offset;
	scale = max( ((float)winSize.width + point.x) / ((float)src.cols),
                ((float)winSize.height + point.y) / ((float)src.rows) );
    
	Size sz( (int)(scale*src.cols + 0.5F), (int)(scale*src.rows + 0.5F) );
	resize( src, img, sz );
    integral(img, this->sum);
    updateCascade(pCascade, (int)(this->sum.step/sum.elemSize()));
	return true;
}

bool NegImageReader::get( Mat& _img)
{
	CV_Assert( !_img.empty() );
	CV_Assert( _img.type() == CV_8UC1 );
	CV_Assert( _img.cols == winSize.width );
	CV_Assert( _img.rows == winSize.height );

	if( img.empty() )
		if ( !nextImg() )
			return false;
	Mat mat( winSize.height, winSize.width, CV_8UC1,
            (void*)(img.data + point.y * img.step + point.x * img.elemSize()), img.step );
	mat.copyTo(_img);

    
	if( (int)( point.x + (1.0F + stepFactor ) * winSize.width ) < img.cols )
		point.x += (int)(stepFactor * winSize.width);
	else
	{
		point.x = offset.x;
		if( (int)( point.y + (1.0F + stepFactor ) * winSize.height ) < img.rows )
			point.y += (int)(stepFactor * winSize.height);
		else
		{
			point.y = offset.y;
			scale *= scaleFactor;
			if( scale <= 1.0F )
            {
				resize( src, img, Size( (int)(scale*src.cols), (int)(scale*src.rows) ) );
                //blur(img, img, Size(3,3));
            }
			else
			{
				if ( !nextImg() )
					return false;
			}
		}
	}
	return true;
}

bool NegImageReader::get_good_negative( Mat& _img, MBLBPCascadef * pCascade, size_t &testCount)
{
	CV_Assert( !_img.empty() );
	CV_Assert( _img.type() == CV_8UC1 );
	CV_Assert( _img.cols == winSize.width );
	CV_Assert( _img.rows == winSize.height );

    testCount = 0;

	if( img.empty() )
		if ( !nextImg2(pCascade) )
			return false;

    while(1)
    {
        testCount++;
        int detect_offset = (int)(point.y * (sum.step/sum.elemSize()) + point.x);
        bool ret = detectAt(this->sum, pCascade, detect_offset);
        if(ret)//i have get a good negative sample
        {
	        Mat mat( winSize.height, winSize.width, CV_8UC1,
                    (void*)(img.data + point.y * img.step + point.x * img.elemSize()), img.step );
	        mat.copyTo(_img);
        }

	    if( (int)( point.x + (1.0F + stepFactor ) * winSize.width ) < img.cols )
		    point.x += (int)(stepFactor * winSize.width);
	    else
	    {
		    point.x = offset.x;
		    if( (int)( point.y + (1.0F + stepFactor ) * winSize.height ) < img.rows )
			    point.y += (int)(stepFactor * winSize.height);
		    else
		    {
			    point.y = offset.y;
			    scale *= scaleFactor;
			    if( scale <= 1.0F )
                {
				    resize( src, img, Size( (int)(scale*src.cols), (int)(scale*src.rows) ) );
                    integral(img, this->sum);
                    updateCascade(pCascade, (int)(sum.step/sum.elemSize()) );
                }
			    else
			    {
				    if ( !nextImg2(pCascade) )
					    return false;
			    }
		    }
        }

        if(ret) //i have get a good negative sample
            break;
	}
	return true;
}


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
                            featureType == CvFeatureParams::MBLBP ? CC_MBLBP :
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
                  !featureTypeStr.compare( CC_MBLBP ) ? CvFeatureParams::MBLBP :
                  -1;
    if (featureType == -1)
        return false;
    node[CC_HEIGHT] >> winSize.height;
    node[CC_WIDTH] >> winSize.width;
    return winSize.height > 0 && winSize.width > 0;
}

void CvCascadeParams::printDefaults() const
{
    cout <<(sizeof(featureTypes)/sizeof(featureTypes[0]))<<endl;
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
                                double acceptanceRatioBreakValue)
{
    if (_cascadeParams.featureType == 3)
    {
        if( _cascadeDirName.empty() || _posFilename.empty() || _negFilename.empty() )
        {
            cerr << "cascadeDirName or bgfileName or vecFileName is NULL" << endl;
            return false;
        }
        string dirName;
        if (_cascadeDirName.find_last_of("/\\") == (_cascadeDirName.length() - 1) )
            dirName = _cascadeDirName;
        else
            dirName = _cascadeDirName + '/';
        this->numStages = _numStages;
        this->numPos = _numPos;
        this->numNeg = _numNeg;
        this->winSize = cvSize(24, 24);;
        this->maxWeakCount = 256;
        this->minHitRate = 0.995f;
        cout << "PARAMETERS:" << endl;
        cout << "cascadeDirName: " << _cascadeDirName << endl;
        cout << "vecFile:" << _posFilename << endl;    
        cout << "numPos: " << this->numPos << endl;
        cout << "bgFileName: " << _negFilename << endl;
        cout << "numNeg: " << this->numNeg << endl;
        cout << "numStages: " << this->numStages << endl;
        cout << "winSize: (" << this->winSize.width << ", " << this->winSize.height << ")" << endl;
        cout << "maxWeakCount: " << this->maxWeakCount << endl;
        cout << "minHitRate: " << this->minHitRate << endl;
        if( ! MBLBPGenerateFeatures())
            return false;
        if(! loadPosSamples(_posFilename))
            return false;
        negReader.create(_negFilename, this->winSize);
        MBLBPLoad(dirName);
        int startNumStages = cascade.count;
        if ( startNumStages > 1 )
            cout << endl << "Stages 0-" << startNumStages-1 << " are loaded" << endl;
        else if ( startNumStages == 1)
            cout << endl << "Stage 0 is loaded" << endl;
        for( int i = startNumStages; i < numStages; i++ )
        {
            cout << "================= Training " << i << "-stage ====================" << endl;

            double rateFA = 1.0;

            if(cascade.count != i)
            {
                cerr << "inconsistent value: cascade.count" << endl;
                return false;
            }


            double t = (double)cvGetTickCount();
            if( ! loadNegSamples(rateFA) )
            {
                cout << "Negative training dataset for temp stage can not be filled. "
                    "Branch training terminated." << endl;
                break;
            }
            t = (double)cvGetTickCount() - t;
            t = t/((double)cvGetTickFrequency()*1000*1000);
            if(t < 60)
                cout << "Precalculation time: " <<  t  << " seconds" << endl;
            else if(t>=60 && t <= 3600)
                cout << "Precalculation time: " <<  t/60  << " minutes" << endl;
            else
                cout << "Precalculation time: " <<  t/3600  << " hours" << endl;
                

            //reset feature mask
            //memset(this->featuresMask, 0, sizeof(bool)*this->numFeatures);

            MBLBPStagef * pStage = (MBLBPStagef*)cvAlloc( sizeof(MBLBPStagef));
            memset(pStage, 0, sizeof(MBLBPStagef));
            cascade.stages[ cascade.count++ ] = pStage;

            pStage->false_alarm = rateFA;

            int weak_count_this_stage = maxWeakCount;

            {
                if( i == 0 )
                    weak_count_this_stage = 8;
                else if( i == 1 )
                    weak_count_this_stage = 16;
                else if( i == 2 )
                    weak_count_this_stage = 32;
            }
            //weak_count_this_stage = 24;

            t = (double)cvGetTickCount();
            if(!boostTrain(pStage,
                        samplesLBP,
                        labels,
                        features,
                        featuresMask,
                        numFeatures,
                        numPos,
                        numNeg,
                        weak_count_this_stage,
                        minHitRate) )
            {
                return false;
            }
            
            t = (double)cvGetTickCount() - t;
            t = t/((double)cvGetTickFrequency()*1000*1000);
            if(t < 60)
                cout << "===Training time: " <<  t  << " seconds" << endl;
            else if(t>=60 && t <= 3600)
                cout << "===Training time: " <<  t/60  << " minutes" << endl;
            else
                cout << "===Training time: " <<  t/3600  << " hours" << endl;

            // save current stage
            char buf[64];
            sprintf(buf, "%s%d.xml", "cascade", i);
            this->save( dirName + string(buf));

            //saveSelectedFeatureData(pStage, i, numPos, numNeg);

        }
        return true;
    }
    else
    {
        // Start recording clock ticks for training time output
        const clock_t begin_time = clock();

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
            stageParams = new CvCascadeBoostParams;
            *stageParams = _stageParams;
            featureEvaluator = CvFeatureEvaluator::create(cascadeParams.featureType);
            featureEvaluator->init( (CvFeatureParams*)featureParams, numPos + numNeg, cascadeParams.winSize );
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

            CvCascadeBoost* tempStage = new CvCascadeBoost;
            bool isStageTrained = tempStage->train( (CvFeatureEvaluator*)featureEvaluator,
                                                    curNumSamples, _precalcValBufSize, _precalcIdxBufSize,
                                                    *((CvCascadeBoostParams*)stageParams) );
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
            char buf[10];
            sprintf(buf, "%s%d", "stage", i );
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
            float seconds = float( clock () - begin_time ) / CLOCKS_PER_SEC;
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
}

int CvCascadeClassifier::predict( int sampleIdx )
{
    CV_DbgAssert( sampleIdx < numPos + numNeg );
    for (vector< Ptr<CvCascadeBoost> >::iterator it = stageClassifiers.begin();
        it != stageClassifiers.end(); it++ )
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
bool CvCascadeClassifier::setImage(Mat img, int idx, bool isSum)
{
    Mat sum;

    if( idx >= samplesLBP.cols )
    {
        cerr << "The sample index is out of rangle: index=" << idx << "; Range [0, " << samplesLBP.cols << ")" << endl;
        return false;
    }
    unsigned char * pLBP = samplesLBP.ptr(0)+idx;

    if(isSum)
        sum = img;
    else
        integral(img, sum);

    if(sum.cols != this->winSize.width + 1 || 
       sum.rows != this->winSize.height + 1)
    {
        cerr << "The sum image's size is incorrect." << endl;
        return false;
    }

    for(int i = 0; i < numFeatures; i++)
    {
        pLBP[i*samplesLBP.step] = LBPcode(sum, features[i].offsets);
    }

    return true;
}
void CvCascadeClassifier::writeFeatures( FileStorage &fs, const Mat& featureMap ) const
{
    ((CvFeatureEvaluator*)((Ptr<CvFeatureEvaluator>)featureEvaluator))->writeFeatures( fs, featureMap );
}
bool CvCascadeClassifier::loadNegSamples(double & FARate)
{
    int64 totnegtest = 0;
    int negCount = 0;

    Mat img;
    bool isGoodNeg;

    img.create(this->winSize, CV_8UC1);

    int first = this->numPos;
    int count = this->numNeg;

    for( int i = first; i < first + count; i++ )
    {
        size_t neg_test_count = 0;
        bool isGetImage = negReader.get_good_negative(img, &cascade, neg_test_count);
            
        if(!isGetImage)
                return false;

        setImage(img, i, false);

        totnegtest+=neg_test_count;
        negCount++;

        if( negCount %(count/20)==0  )
        {
            std::cout << "current neg: " << negCount << ": " << negCount*100.0f/count << "%  " <<  negCount/double(totnegtest) << endl;
            
            if( negCount/double(totnegtest) < 1.0e-7)
                return false;
        }
    }
	cout << "Total Test: " << totnegtest << endl;

    if ( !negCount )
        return false;
    
    FARate = (totnegtest == 0) ? 0.0 : ( (double)negCount/(double)totnegtest );
    cout << "NEG count : acceptanceRatio    " << negCount << " : " << FARate << endl;

    return true;

}
bool CvCascadeClassifier::loadPosSamples(string _posFilename){
    //positive sampls
    this->numSamples = 0;
    this->numSamples += this->numPos;
    this->numSamples += this->numNeg;

    //alloc memory for samples
    this->labels = Mat::zeros(1, numSamples, CV_8UC1);
    this->samplesLBP.create(this->numFeatures, numSamples, CV_8UC1);
    if(labels.empty() || samplesLBP.empty())
    {
        cerr << "Cannot alloc memory for samples." << endl;
        return false;
    }


    //load postive samples and convert them to integral images
    {
        Mat img(this->winSize, CV_8UC1);
        posReader.create(_posFilename);

        int currPos = 0;
        while( posReader.get(img))
        {
            //imshow("pos", img);
            //waitKey(10);

            labels.at<unsigned char>(0, currPos) = 1;
            setImage(img, currPos, false);
            currPos++;

            if(currPos >= this->numPos)
                break;
        }

        if( currPos != this->numPos)
        {
            cerr << "Read vec file " << _posFilename << " error" << endl;
            cerr << "There are " << this->numPos << " samples, but only " << currPos << " loaded." << endl;
            return false;
        }

        cout << currPos << " positive samples loaded." << endl;
    }

    return true;
}
void CvCascadeClassifier::writeStages( FileStorage &fs, const Mat& featureMap ) const
{
    char cmnt[30];
    int i = 0;
    fs << CC_STAGES << "[";
    for( vector< Ptr<CvCascadeBoost> >::const_iterator it = stageClassifiers.begin();
        it != stageClassifiers.end(); it++, i++ )
    {
        sprintf( cmnt, "stage %d", i );
        cvWriteComment( fs.fs, cmnt, 0 );
        fs << "{";
        ((CvCascadeBoost*)((Ptr<CvCascadeBoost>)*it))->write( fs, featureMap );
        fs << "}";
    }
    fs << "]";
}

bool CvCascadeClassifier::readParams( const FileNode &node )
{
    if ( !node.isMap() || !cascadeParams.read( node ) )
        return false;

    stageParams = new CvCascadeBoostParams;
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
        CvCascadeBoost* tempStage = new CvCascadeBoost;
        if ( !tempStage->read( *it, (CvFeatureEvaluator *)featureEvaluator, *((CvCascadeBoostParams*)stageParams) ) )
        {
            delete tempStage;
            return false;
        }
        stageClassifiers.push_back(tempStage);
    }
    return true;
}

// For old Haar Classifier file saving
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

    fs << FileStorage::getDefaultObjectName(filename) << "{";
    if ( !baseFormat )
    {
        Mat featureMap;
        getUsedFeaturesIdxMap( featureMap );
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
        fs << ICV_HAAR_SIZE_NAME << "[:" << cascadeParams.winSize.width <<
            cascadeParams.winSize.height << "]";
        fs << ICV_HAAR_STAGES_NAME << "[";
        for( size_t si = 0; si < stageClassifiers.size(); si++ )
        {
            fs << "{"; //stage
            /*sprintf( buf, "stage %d", si );
            CV_CALL( cvWriteComment( fs, buf, 1 ) );*/
            weak = stageClassifiers[si]->get_weak_predictors();
            fs << ICV_HAAR_TREES_NAME << "[";
            for( int wi = 0; wi < weak->total; wi++ )
            {
                int inner_node_idx = -1, total_inner_node_idx = -1;
                queue<const CvDTreeNode*> inner_nodes_queue;
                CvCascadeBoostTree* tree = *((CvCascadeBoostTree**) cvGetSeqElem( weak, wi ));

                fs << "[";
                /*sprintf( buf, "tree %d", wi );
                CV_CALL( cvWriteComment( fs, buf, 1 ) );*/

                const CvDTreeNode* tempNode;

                inner_nodes_queue.push( tree->get_root() );
                total_inner_node_idx++;

                while (!inner_nodes_queue.empty())
                {
                    tempNode = inner_nodes_queue.front();
                    inner_node_idx++;

                    fs << "{";
                    fs << ICV_HAAR_FEATURE_NAME << "{";
                    ((CvHaarEvaluator*)((CvFeatureEvaluator*)featureEvaluator))->writeFeature( fs, tempNode->split->var_idx );
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
    featureEvaluator->init( ((CvFeatureParams*)featureParams), numPos + numNeg, cascadeParams.winSize );
    fs.release();

    char buf[10];
    for ( int si = 0; si < numStages; si++ )
    {
        sprintf( buf, "%s%d", "stage", si);
        fs.open( cascadeDirName + buf + ".xml", FileStorage::READ );
        node = fs.getFirstTopLevelNode();
        if ( !fs.isOpened() )
            break;
        CvCascadeBoost *tempStage = new CvCascadeBoost;

        if ( !tempStage->read( node, (CvFeatureEvaluator*)featureEvaluator, *((CvCascadeBoostParams*)stageParams )) )
        {
            delete tempStage;
            fs.release();
            break;
        }
        stageClassifiers.push_back(tempStage);
    }
    return true;
}
void CvCascadeClassifier::clearCascade()
{
        for(int i = 0; i < this->cascade.count; i++)
    {
        cvFree(cascade.stages+i);
    }
     memset(&cascade, 0, sizeof(MBLBPCascadef));
}
bool CvCascadeClassifier::MBLBPGenerateFeatures()
{
    int offset = winSize.width + 1;
    int count = 0;
    this->numFeatures = 0;
    for(int round=0; round < 2; round++)
    {
        for( int x = 0; x < winSize.width; x++ )
            for( int y = 0; y < winSize.height; y++ )
                for( int w = 1; w <= winSize.width / 3; w++ )
                    for( int h = 1; h <= winSize.height / 3; h++ )
                    {
                        if((x+3*w <= winSize.width) && (y+3*h <= winSize.height) )
                        {
                            if ( round==0 )
                                this->numFeatures++; //count how many features
                            else //set features
                            {
                                this->features[count].x = x;
                                this->features[count].y = y;
                                this->features[count].cellwidth = w;
                                this->features[count].cellheight = h;
                                count++;
                            }
                        }
                    }
        

        if(round==0) //alloc memory for features in the first round
        {
            cout << "numFeatures = " << this->numFeatures << endl;
            int mem_size = sizeof(MBLBPWeakf)*numFeatures;
            this->features = (MBLBPWeakf*) cvAlloc(mem_size);
            this->featuresMask = (bool*) cvAlloc(sizeof(bool)*numFeatures);

            if(this->features == NULL || this->featuresMask == NULL)
            {
                cerr << "Can not alloc memory. size = " <<  mem_size << endl;
                return false;
            }
            else
            {
                memset(this->features, 0, mem_size);
                memset(this->featuresMask, 0, sizeof(bool)*this->numFeatures);
            }

        } //end if round
        else //check 
        {
            if( count != this->numFeatures)
            {
                cerr << "count("<< count <<") != numFeatures("<< this->numFeatures <<")" << endl;
                cvFree( &(this->features) );
                cvFree( &(this->featuresMask) );
                return false;
            }
        }//end else round
        
    }

    //cal fast pointer offsets for each features
    for(int i = 0; i < this->numFeatures; i++)
    {
        int x = this->features[i].x;
        int y = this->features[i].y;
        int w = this->features[i].cellwidth;
        int h = this->features[i].cellheight;

        this->features[i].offsets[ 0] = y * offset + (x      );
        this->features[i].offsets[ 1] = y * offset + (x + w  );
        this->features[i].offsets[ 2] = y * offset + (x + w*2);
        this->features[i].offsets[ 3] = y * offset + (x + w*3);
        
        this->features[i].offsets[ 4] = (y+h) * offset + (x      );
        this->features[i].offsets[ 5] = (y+h) * offset + (x + w  );
        this->features[i].offsets[ 6] = (y+h) * offset + (x + w*2);
        this->features[i].offsets[ 7] = (y+h) * offset + (x + w*3);
        
        this->features[i].offsets[ 8] = (y+h*2) * offset + (x      );
        this->features[i].offsets[ 9] = (y+h*2) * offset + (x + w  );
        this->features[i].offsets[10] = (y+h*2) * offset + (x + w*2);
        this->features[i].offsets[11] = (y+h*2) * offset + (x + w*3);
        
        this->features[i].offsets[12] = (y+h*3) * offset + (x      );
        this->features[i].offsets[13] = (y+h*3) * offset + (x + w  );
        this->features[i].offsets[14] = (y+h*3) * offset + (x + w*2);
        this->features[i].offsets[15] = (y+h*3) * offset + (x + w*3);       
    }
    return true;
}

bool CvCascadeClassifier::MBLBPLoad( const string dirName )
{
//find the last classifier in the directory
	int currstage = -1;
	for(int stage_idx = 0; stage_idx < MAX_NUM_STAGES; stage_idx++)
	{
		stringstream snum;
		snum << stage_idx;
		string filename =  dirName + "cascade" + snum.str() + ".xml";
		FILE * p = fopen(filename.c_str(), "r");
		if(p)
		{
			fclose(p);
			currstage = stage_idx;
		}
		else
			break;
	}
	
	if(currstage < 0)
		return true;

	//load the classifier
    //get the filename first
	stringstream snum;
	snum << currstage;
	string filename =  dirName + "cascade" + snum.str() + ".xml";

    //open the classifier file
	FileStorage fs;
	fs.open(filename, FileStorage::READ );
	if(!fs.isOpened()) {
		cerr << "Read " << filename << " error!" << endl;
		return NULL;
	}
    //clear the cascade classifier
    this->clearCascade();

    //read the classifier content 
	FileNode node = fs.getFirstTopLevelNode();


	//get window size
	cascade.win_width = node[CC_WIDTH];
	cascade.win_height = node[CC_HEIGHT];

	//get the number of stages
	FileNode stages = node[CC_STAGES];
    cascade.count = (int)stages.size();

    int stage_idx = 0;
	for(FileNodeIterator Iter = stages.begin(); Iter != stages.end(); Iter++, stage_idx++)
	{
        //alloc memory for the stage
        MBLBPStagef* pStage = (MBLBPStagef*)cvAlloc( sizeof(MBLBPStagef));
	    memset(pStage, 0, sizeof(MBLBPStagef));
        cascade.stages[stage_idx] = pStage;

		//get the number of weak classifiers
		(*Iter)[CC_WEAK_COUNT] >> pStage->count;
        //get the stage threshold
		(*Iter)[CC_STAGE_THRESHOLD] >> pStage->threshold;
        (*Iter)[CC_FALSE_ALARM] >> pStage->false_alarm;

        int weak_idx = 0;
		FileNode nextnode = (*Iter)[CC_WEAK_CLASSIFIERS];
		for( FileNodeIterator nextIter = nextnode.begin(); nextIter != nextnode.end(); nextIter++, weak_idx++ )
		{
            int feature_idx = (int)(*nextIter)[CC_FEATUREIDX];

            pStage->weak_classifiers_idx[weak_idx] = feature_idx;
            this->featuresMask[feature_idx]=true;
            pStage->weak_classifiers[weak_idx].x = (int)(*nextIter)[CC_RECT][0];
            pStage->weak_classifiers[weak_idx].y = (int)(*nextIter)[CC_RECT][1];
            pStage->weak_classifiers[weak_idx].cellwidth  = (int)(*nextIter)[CC_RECT][2];
            pStage->weak_classifiers[weak_idx].cellheight = (int)(*nextIter)[CC_RECT][3];

            //copy offset values
            memcpy(pStage->weak_classifiers[weak_idx].offsets, features[feature_idx].offsets, sizeof(features[feature_idx].offsets));

			int lutlength = (int)(*nextIter)[CC_LUT_LENGTH];

			for (int i = 0; i < lutlength; i++){
                pStage->weak_classifiers[weak_idx].look_up_table[i] = (float)(*nextIter)[CC_LUT][i];
			}
            pStage->weak_classifiers[weak_idx].soft_threshold = (*nextIter)[CC_WEAK_THRESHOLD];
		}
        CV_Assert(pStage->count == weak_idx);
    }
    cout<<"cascade: "<<cascade.count<<endl;
    return true;
}

void CvCascadeClassifier::getUsedFeaturesIdxMap( Mat& featureMap )
{
    int varCount = featureEvaluator->getNumFeatures() * featureEvaluator->getFeatureSize();
    featureMap.create( 1, varCount, CV_32SC1 );
    featureMap.setTo(Scalar(-1));

    for( vector< Ptr<CvCascadeBoost> >::const_iterator it = stageClassifiers.begin();
        it != stageClassifiers.end(); it++ )
        ((CvCascadeBoost*)((Ptr<CvCascadeBoost>)(*it)))->markUsedFeaturesInMap( featureMap );

    for( int fi = 0, idx = 0; fi < varCount; fi++ )
        if ( featureMap.at<int>(0, fi) >= 0 )
            featureMap.ptr<int>(0)[fi] = idx++;
}
