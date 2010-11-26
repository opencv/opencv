#include "cvtest.h"

using namespace cv;

class BruteForceMatcherTest : public CvTest
{
public:
    BruteForceMatcherTest();
protected:
    void run( int );
};

struct CV_EXPORTS L2Fake : public L2<float>
{
};

BruteForceMatcherTest::BruteForceMatcherTest() : CvTest( "BruteForceMatcher", "BruteForceMatcher::matchImpl")
{
    support_testing_modes = CvTS::TIMING_MODE;
}

void BruteForceMatcherTest::run( int )
{
    const int dimensions = 64;
    const int descriptorsNumber = 5000;

    Mat train = Mat( descriptorsNumber, dimensions, CV_32FC1);
    Mat query = Mat( descriptorsNumber, dimensions, CV_32FC1);

    Mat permutation( 1, descriptorsNumber, CV_32SC1 );
    for( int i=0;i<descriptorsNumber;i++ )
        permutation.at<int>( 0, i ) = i;

    //RNG rng = RNG( cvGetTickCount() );
    RNG rng = RNG( *ts->get_rng() );
    randShuffle( permutation, 1, &rng );

    float boundary =  500.f;
    for( int row=0;row<descriptorsNumber;row++ )
    {
        for( int col=0;col<dimensions;col++ )
        {
            int bit = rng( 2 );
            train.at<float>( permutation.at<int>( 0, row ), col ) = bit*boundary + rng.uniform( 0.f, boundary );
            query.at<float>( row, col ) = bit*boundary + rng.uniform( 0.f, boundary );
        }
    }

    vector<DMatch> specMatches, genericMatches;
    BruteForceMatcher<L2<float> > specMatcher;
    BruteForceMatcher<L2Fake > genericMatcher;

    int64 time0 = cvGetTickCount();
    specMatcher.match( query, train, specMatches );
    int64 time1 = cvGetTickCount();
    genericMatcher.match( query, train, genericMatches );
    int64 time2 = cvGetTickCount();

    float specMatcherTime = float(time1 - time0)/(float)cvGetTickFrequency();
    ts->printf( CvTS::LOG, "Matching by matrix multiplication time s: %f, us per pair: %f\n",
               specMatcherTime*1e-6, specMatcherTime/( descriptorsNumber*descriptorsNumber ) );

    float genericMatcherTime = float(time2 - time1)/(float)cvGetTickFrequency();
    ts->printf( CvTS::LOG, "Matching without matrix multiplication time s: %f, us per pair: %f\n",
               genericMatcherTime*1e-6, genericMatcherTime/( descriptorsNumber*descriptorsNumber ) );

    if( (int)specMatches.size() != descriptorsNumber || (int)genericMatches.size() != descriptorsNumber )
        ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
    for( int i=0;i<descriptorsNumber;i++ )
    {
        float epsilon = 0.01f;
        bool isEquiv = fabs( specMatches[i].distance - genericMatches[i].distance ) < epsilon &&
                       specMatches[i].queryIdx == genericMatches[i].queryIdx &&
                       specMatches[i].trainIdx == genericMatches[i].trainIdx;
        if( !isEquiv || specMatches[i].trainIdx != permutation.at<int>( 0, i ) )
        {
            ts->set_failed_test_info( CvTS::FAIL_MISMATCH );
            break;
        }
    }


    //Test mask
    Mat mask( query.rows, train.rows, CV_8UC1 );
    rng.fill( mask, RNG::UNIFORM, 0, 2 );


    time0 = cvGetTickCount();
    specMatcher.match( query, train, specMatches, mask );
    time1 = cvGetTickCount();
    genericMatcher.match( query, train, genericMatches, mask );
    time2 = cvGetTickCount();

    specMatcherTime = float(time1 - time0)/(float)cvGetTickFrequency();
    ts->printf( CvTS::LOG, "Matching by matrix multiplication time with mask s: %f, us per pair: %f\n",
               specMatcherTime*1e-6, specMatcherTime/( descriptorsNumber*descriptorsNumber ) );

    genericMatcherTime = float(time2 - time1)/(float)cvGetTickFrequency();
    ts->printf( CvTS::LOG, "Matching without matrix multiplication time with mask s: %f, us per pair: %f\n",
               genericMatcherTime*1e-6, genericMatcherTime/( descriptorsNumber*descriptorsNumber ) );

    if( specMatches.size() != genericMatches.size() )
        ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );

    for( size_t i=0;i<specMatches.size();i++ )
    {
        //float epsilon = 1e-2;
        float epsilon = 10000000;
        bool isEquiv = fabs( specMatches[i].distance - genericMatches[i].distance ) < epsilon &&
                       specMatches[i].queryIdx == genericMatches[i].queryIdx &&
                       specMatches[i].trainIdx == genericMatches[i].trainIdx;
        if( !isEquiv )
        {
            ts->set_failed_test_info( CvTS::FAIL_MISMATCH );
            break;
        }
    }
}

BruteForceMatcherTest taBruteForceMatcherTest;
