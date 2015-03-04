#include "opencv2/core.hpp"
#include "cascadeclassifier.h"

using namespace std;
using namespace cv;

static int check_atoi(char *value, const char *field) {
    if (value)
        return atoi(value);
    printf("You should specify value for %s\n", field);
    exit(-1);
}

int main( int argc, char* argv[] )
{
    CvCascadeClassifier classifier;
    string cascadeDirName, vecName, bgName;
    int numPos    = 2000;
    int numNeg    = 1000;
    int numStages = 20;
    int numThreads = getNumThreads();
    int precalcValBufSize = 256,
        precalcIdxBufSize = 256;
    bool baseFormatSave = false;

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
            numPos = check_atoi( argv[++i], "number_of_positive_samples" );
        }
        else if( !strcmp( argv[i], "-numNeg" ) )
        {
            numNeg = check_atoi( argv[++i], "number_of_negative_samples" );
        }
        else if( !strcmp( argv[i], "-numStages" ) )
        {
            numStages = check_atoi( argv[++i], "number_of_stages" );
        }
        else if( !strcmp( argv[i], "-precalcValBufSize" ) )
        {
            precalcValBufSize = check_atoi( argv[++i], "precalculated_vals_buffer_size_in_Mb" );
        }
        else if( !strcmp( argv[i], "-precalcIdxBufSize" ) )
        {
            precalcIdxBufSize = check_atoi( argv[++i], "precalculated_idxs_buffer_size_in_Mb" );
        }
        else if( !strcmp( argv[i], "-baseFormatSave" ) )
        {
            baseFormatSave = true;
        }
        else if( !strcmp( argv[i], "-numThreads" ) )
        {
          numThreads = check_atoi(argv[++i], "max_number_of_threads");
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
                      baseFormatSave );
    return 0;
}
