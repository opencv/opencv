#include "cvtest.h"

using namespace cv;

class BruteForceMatcherTest : public CvTest
{
public:
    BruteForceMatcherTest();
protected:
    void run( int );
};

BruteForceMatcherTest::BruteForceMatcherTest() : CvTest( "BruteForceMatcher", "BruteForceMatcher")
{
}

void BruteForceMatcherTest::run( int )
{
    const int dimensions = 64;
    const int descriptorsNumber = 1024;

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

    vector<int> matches;
    BruteForceMatcher<L2<float> > matcher;
    matcher.add( train );
    matcher.match( query, matches );

    for( int i=0;i<descriptorsNumber;i++ )
    {
        if( matches[i] != permutation.at<int>( 0, i ) )
        {
            ts->set_failed_test_info( CvTS::FAIL_MISMATCH );
            break;
        }
    }
}

BruteForceMatcherTest bruteForceMatcherTest;
