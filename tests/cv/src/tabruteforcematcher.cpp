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

    RNG rng (cvGetTickCount());
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

    vector<int> specMatches, genericMatches;
    BruteForceMatcher<L2<float> > specMatcher;
    BruteForceMatcher<L2Fake > genericMatcher;
    specMatcher.add( train );
    genericMatcher.add( train );


    int64 time0 = cvGetTickCount();
    specMatcher.match( query, specMatches );
    int64 time1 = cvGetTickCount();
    genericMatcher.match( query, genericMatches );
    int64 time2 = cvGetTickCount();

    float specMatcherTime = float(time1 - time0)/(float)cvGetTickFrequency();
    ts->printf( CvTS::LOG, "Matching by matrix multiplication time s: %f, us per pair: %f\n",
               specMatcherTime*1e-6, specMatcherTime/( descriptorsNumber*descriptorsNumber ) );

    float genericMatcherTime = float(time2 - time1)/(float)cvGetTickFrequency();
    ts->printf( CvTS::LOG, "Matching without matrix multiplication time s: %f, us per pair: %f\n",
               genericMatcherTime*1e-6, genericMatcherTime/( descriptorsNumber*descriptorsNumber ) );

    if( specMatches.size() != descriptorsNumber || genericMatches.size() != descriptorsNumber )
        ts->set_failed_test_info( CvTS::FAIL_INVALID_OUTPUT );
    for( int i=0;i<descriptorsNumber;i++ )
    {
        if( specMatches[i] != permutation.at<int>( 0, i ) || genericMatches[i] != permutation.at<int>( 0, i ) )
        {
            ts->set_failed_test_info( CvTS::FAIL_MISMATCH );
            break;
        }
    }
}

BruteForceMatcherTest bruteForceMatcherTest;
