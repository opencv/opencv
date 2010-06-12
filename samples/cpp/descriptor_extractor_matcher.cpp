#include <highgui.h>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <iostream>


using namespace cv;
using namespace std;

inline Point2f applyHomography( const Mat_<double>& H, const Point2f& pt )
{
    double z = H(2,0)*pt.x + H(2,1)*pt.y + H(2,2);
    if( z )
    {
        double w = 1./z;
        return Point2f( (H(0,0)*pt.x + H(0,1)*pt.y + H(0,2))*w, (H(1,0)*pt.x + H(1,1)*pt.y + H(1,2))*w );
    }
    return Point2f( numeric_limits<double>::max(), numeric_limits<double>::max() );
}

void warpPerspectiveRand( const Mat& src, Mat& dst, Mat& H, RNG* rng )
{
    H.create(3, 3, CV_32FC1);
    H.at<float>(0,0) = rng->uniform( 0.8f, 1.2f);
    H.at<float>(0,1) = rng->uniform(-0.1f, 0.1f);
    H.at<float>(0,2) = rng->uniform(-0.1f, 0.1f)*src.cols;
    H.at<float>(1,0) = rng->uniform(-0.1f, 0.1f);
    H.at<float>(1,1) = rng->uniform( 0.8f, 1.2f);
    H.at<float>(1,2) = rng->uniform(-0.1f, 0.1f)*src.rows;
    H.at<float>(2,0) = rng->uniform( -1e-4f, 1e-4f);
    H.at<float>(2,1) = rng->uniform( -1e-4f, 1e-4f);
    H.at<float>(2,2) = rng->uniform( 0.8f, 1.2f);

    warpPerspective( src, dst, H, src.size() );
}

const string winName = "correspondences";

void doIteration( const Mat& img1, Mat& img2, bool isWarpPerspective,
                  const vector<KeyPoint>& keypoints1, const Mat& descriptors1,
                  Ptr<FeatureDetector>& detector, Ptr<DescriptorExtractor>& descriptorExtractor,
                  Ptr<DescriptorMatcher>& descriptorMatcher,
                  double ransacReprojThreshold = -1, RNG* rng = 0 )
{
    assert( !img1.empty() );
    Mat H12;
    if( isWarpPerspective )
    {
        assert( rng );
        warpPerspectiveRand(img1, img2, H12, rng);
    }
    else
        assert( !img2.empty()/* && img2.cols==img1.cols && img2.rows==img1.rows*/ );

    cout << endl << "< Extracting keypoints from second image..." << endl;
    vector<KeyPoint> keypoints2;
    detector->detect( img2, keypoints2 );
    cout << keypoints2.size() << " >" << endl;

    cout << "< Computing descriptors for keypoints from second image..." << endl;
    Mat descriptors2;
    descriptorExtractor->compute( img2, keypoints2, descriptors2 );
    cout << " >" << endl;

    cout << "< Matching descriptors..." << endl;
    vector<int> matches;
    descriptorMatcher->clear();
    descriptorMatcher->add( descriptors2 );
    descriptorMatcher->match( descriptors1, matches );
    cout << ">" << endl;

    if( !isWarpPerspective && ransacReprojThreshold >= 0 )
    {
        cout << "< Computing homography (RANSAC)..." << endl;
        vector<Point2f> points1(matches.size()), points2(matches.size());
        for( size_t i = 0; i < matches.size(); i++ )
        {
            points1[i] = keypoints1[i].pt;
            points2[i] = keypoints2[matches[i]].pt;
        }
        H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold );
        cout << ">" << endl;
    }

    Mat drawImg;
    if( !H12.empty() )
    {
        vector<char> matchesMask( matches.size(), 0 );
        vector<int>::const_iterator mit = matches.begin();
        for( size_t i1 = 0; mit != matches.end(); ++mit, i1++ )
        {
            Point2f pt1 = keypoints1[i1].pt,
                    pt2 = keypoints2[*mit].pt;
            if( norm(pt2 - applyHomography(H12, pt1)) < 4 ) // inlier
                matchesMask[i1] = 1;
        }
        // draw inliers
        drawMatches( img1, keypoints1, img2, keypoints2, matches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask );
        // draw outliers
        /*for( size_t i1 = 0; i1 < matchesMask.size(); i1++ )
            matchesMask[i1] = !matchesMask[i1];
        drawMatches( img1, keypoints1, img2, keypoints2, matches, drawImg, CV_RGB(0, 0, 255), CV_RGB(255, 0, 0), matchesMask,
                     DrawMatchesFlags::DRAW_OVER_OUTIMG | DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );*/
    }
    else
    {
        drawMatches( img1, keypoints1, img2, keypoints2, matches, drawImg, CV_RGB(0, 255, 0) );
    }

    imshow( winName, drawImg );
}

int main(int argc, char** argv)
{
    if( argc != 4 && argc != 6 )
    {
        cout << "Format:" << endl;
        cout << "case1: second image is obtained from the first (given) image using random generated homography matrix" << endl;
        cout << argv[0] << " [detectorType] [descriptorType] [image1]" << endl;
        cout << "case2: both images are given. If ransacReprojThreshold>=0 then homography matrix are calculated" << endl;
        cout << argv[0] << " [detectorType] [descriptorType] [image1] [image2] [ransacReprojThreshold]" << endl;
        cout << endl << "Mathes are filtered using homography matrix in case1 and case2 (if ransacReprojThreshold>=0)" << endl;
        return -1;
    }
    bool isWarpPerspective = argc == 4;
    double ransacReprojThreshold = -1;
    if( !isWarpPerspective )
        ransacReprojThreshold = atof(argv[5]);

    cout << "< Creating detector, descriptor extractor and descriptor matcher ..." << endl;
    Ptr<FeatureDetector> detector = createDetector( argv[1] );
    Ptr<DescriptorExtractor> descriptorExtractor = createDescriptorExtractor( argv[2] );
    Ptr<DescriptorMatcher> descriptorMatcher = createDescriptorMatcher( "BruteForce" );
    cout << ">" << endl;
    if( detector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty()  )
    {
        cout << "Can not create detector or descriptor exstractor or descriptor matcher of given types" << endl;
        return -1;
	}
		
    cout << "< Reading the images..." << endl;
    Mat img1 = imread( argv[3], CV_LOAD_IMAGE_GRAYSCALE), img2;
    if( !isWarpPerspective )
        img2 = imread( argv[4], CV_LOAD_IMAGE_GRAYSCALE);
    cout << ">" << endl;
    if( img1.empty() || (!isWarpPerspective && img2.empty()) )
    {
        cout << "Can not read images" << endl;
        return -1;
    }

    cout << endl << "< Extracting keypoints from first image..." << endl;
    vector<KeyPoint> keypoints1;
    detector->detect( img1, keypoints1 );
    cout << keypoints1.size() << " >" << endl;

    cout << "< Computing descriptors for keypoints from first image..." << endl;
    Mat descriptors1;
    descriptorExtractor->compute( img1, keypoints1, descriptors1 );
    cout << " >" << endl;

    namedWindow(winName, 1);
    RNG rng;
    doIteration( img1, img2, isWarpPerspective, keypoints1, descriptors1,
                 detector, descriptorExtractor, descriptorMatcher,
                 ransacReprojThreshold, &rng );
    for(;;)
    {
        char c = (char)cvWaitKey(0);
        if( c == '\x1b' ) // esc
        {
            cout << "Exiting ..." << endl;
            return 0;
        }
        else if( isWarpPerspective )
        {
            doIteration( img1, img2, isWarpPerspective, keypoints1, descriptors1,
                         detector, descriptorExtractor, descriptorMatcher,
                         ransacReprojThreshold, &rng );
        }
    }
    waitKey(0);
    return 0;
}
