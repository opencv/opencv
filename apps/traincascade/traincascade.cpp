#include "opencv2/core.hpp"
#include "cascadeclassifier.h"

using namespace std;
using namespace cv;

/*
traincascade.cpp is the source file of the program used for cascade training.
User has to provide training input in form of positive and negative training images,
and other data related to training in form of command line argument.
*/
//testing
int main( int argc, char* argv[] )
{
    CvCascadeClassifier classifier;
    string cascadeDirName, vecName, bgName;
    int numPos    = 2000;
    int numNeg    = 1000;
    int numStages = 20;
    int numThreads = getNumThreads();
    int precalcValBufSize = 1024,
        precalcIdxBufSize = 1024;
    bool baseFormatSave = false;
    double acceptanceRatioBreakValue = -1.0;

    CvCascadeParams cascadeParams;
    CvCascadeBoostParams stageParams;
    Ptr<CvFeatureParams> featureParams[] = { makePtr<CvHaarFeatureParams>(),
                                             makePtr<CvLBPFeatureParams>(),
                                             makePtr<CvHOGFeatureParams>()
                                           };
    int fc = sizeof(featureParams)/sizeof(featureParams[0]);
    if( argc == 1 )
    {
        cout << "Usage: " << argv[0] << endl;
        cout << "  -data <cascade_dir_name>" << endl;
        cout << "  -vec <vec_file_name>" << endl;
        cout << "  -bg <background_file_name>" << endl;
        cout << "  [-numPos <number_of_positive_samples = " << numPos << ">]" << endl;
        cout << "  [-numNeg <number_of_negative_samples = " << numNeg << ">]" << endl;
        cout << "  [-numStages <number_of_stages = " << numStages << ">]" << endl;
        cout << "  [-precalcValBufSize <precalculated_vals_buffer_size_in_Mb = " << precalcValBufSize << ">]" << endl;
        cout << "  [-precalcIdxBufSize <precalculated_idxs_buffer_size_in_Mb = " << precalcIdxBufSize << ">]" << endl;
        cout << "  [-baseFormatSave]" << endl;
        cout << "  [-numThreads <max_number_of_threads = " << numThreads << ">]" << endl;
        cout << "  [-acceptanceRatioBreakValue <value> = " << acceptanceRatioBreakValue << ">]" << endl;
        cascadeParams.printDefaults();
        stageParams.printDefaults();
        for( int fi = 0; fi < fc; fi++ )
            featureParams[fi]->printDefaults();
        return 0;
    }

    for( int i = 1; i < argc; i++ )
    {
        bool set = false;
        if( !strcmp( argv[i], "-data" ) )
        {
            cascadeDirName = argv[++i];
        }
        else if( !strcmp( argv[i], "-vec" ) )
        {
            vecName = argv[++i];
        }
        else if( !strcmp( argv[i], "-bg" ) )
        {
            bgName = argv[++i];
        }
        else if( !strcmp( argv[i], "-numPos" ) )
        {
            numPos = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-numNeg" ) )
        {
            numNeg = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-numStages" ) )
        {
            numStages = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-precalcValBufSize" ) )
        {
            precalcValBufSize = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-precalcIdxBufSize" ) )
        {
            precalcIdxBufSize = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-baseFormatSave" ) )
        {
            baseFormatSave = true;
        }
        else if( !strcmp( argv[i], "-numThreads" ) )
        {
          numThreads = atoi(argv[++i]);
        }
        else if( !strcmp( argv[i], "-acceptanceRatioBreakValue" ) )
        {
          acceptanceRatioBreakValue = atof(argv[++i]);
        }
        else if ( cascadeParams.scanAttr( argv[i], argv[i+1] ) ) { i++; }
        else if ( stageParams.scanAttr( argv[i], argv[i+1] ) ) { i++; }
        else if ( !set )
        {
            for( int fi = 0; fi < fc; fi++ )
            {
                set = featureParams[fi]->scanAttr(argv[i], argv[i+1]);
                if ( !set )
                {
                    i++;
                    break;
                }
            }
        }
    }

    setNumThreads( numThreads );
    classifier.train( cascadeDirName,
                      vecName,
                      bgName,
                      numPos, numNeg,
                      precalcValBufSize, precalcIdxBufSize,
                      numStages,
                      cascadeParams,
                      *featureParams[cascadeParams.featureType],
                      stageParams,
                      baseFormatSave,
                      acceptanceRatioBreakValue );
    return 0;
}
