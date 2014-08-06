#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help(char** argv)
{
    cout << "\nThis program demonstrats keypoint finding and matching between 2 images using features2d framework.\n"
     << "   In one case, the 2nd image is synthesized by homography from the first, in the second case, there are 2 images\n"
     << "\n"
     << "Case1: second image is obtained from the first (given) image using random generated homography matrix\n"
     << argv[0] << " [detectorType] [descriptorType] [matcherType] [matcherFilterType] [image] [evaluate(0 or 1)]\n"
     << "Example of case1:\n"
     << "./descriptor_extractor_matcher SURF SURF FlannBased NoneFilter cola.jpg 0\n"
     << "\n"
     << "Case2: both images are given. If ransacReprojThreshold>=0 then homography matrix are calculated\n"
     << argv[0] << " [detectorType] [descriptorType] [matcherType] [matcherFilterType] [image1] [image2] [ransacReprojThreshold]\n"
     << "\n"
     << "Matches are filtered using homography matrix in case1 and case2 (if ransacReprojThreshold>=0)\n"
     << "Example of case2:\n"
     << "./descriptor_extractor_matcher SURF SURF BruteForce CrossCheckFilter cola1.jpg cola2.jpg 3\n"
     << "\n"
     << "Possible detectorType values: see in documentation on createFeatureDetector().\n"
     << "Possible descriptorType values: see in documentation on createDescriptorExtractor().\n"
     << "Possible matcherType values: see in documentation on createDescriptorMatcher().\n"
     << "Possible matcherFilterType values: NoneFilter, CrossCheckFilter." << endl;
}

#define DRAW_RICH_KEYPOINTS_MODE     0
#define DRAW_OUTLIERS_MODE           0

const string winName = "correspondences";

enum { NONE_FILTER = 0, CROSS_CHECK_FILTER = 1 };

static int getMatcherFilterType( const string& str )
{
    if( str == "NoneFilter" )
        return NONE_FILTER;
    if( str == "CrossCheckFilter" )
        return CROSS_CHECK_FILTER;
    CV_Error(Error::StsBadArg, "Invalid filter name");
    return -1;
}

static void simpleMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                     const Mat& descriptors1, const Mat& descriptors2,
                     vector<DMatch>& matches12 )
{
    vector<DMatch> matches;
    descriptorMatcher->match( descriptors1, descriptors2, matches12 );
}

static void crossCheckMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                         const Mat& descriptors1, const Mat& descriptors2,
                         vector<DMatch>& filteredMatches12, int knn=1 )
{
    filteredMatches12.clear();
    vector<vector<DMatch> > matches12, matches21;
    descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
    descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
    for( size_t m = 0; m < matches12.size(); m++ )
    {
        bool findCrossCheck = false;
        for( size_t fk = 0; fk < matches12[m].size(); fk++ )
        {
            DMatch forward = matches12[m][fk];

            for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
            {
                DMatch backward = matches21[forward.trainIdx][bk];
                if( backward.trainIdx == forward.queryIdx )
                {
                    filteredMatches12.push_back(forward);
                    findCrossCheck = true;
                    break;
                }
            }
            if( findCrossCheck ) break;
        }
    }
}

static void warpPerspectiveRand( const Mat& src, Mat& dst, Mat& H, RNG& rng )
{
    H.create(3, 3, CV_32FC1);
    H.at<float>(0,0) = rng.uniform( 0.8f, 1.2f);
    H.at<float>(0,1) = rng.uniform(-0.1f, 0.1f);
    H.at<float>(0,2) = rng.uniform(-0.1f, 0.1f)*src.cols;
    H.at<float>(1,0) = rng.uniform(-0.1f, 0.1f);
    H.at<float>(1,1) = rng.uniform( 0.8f, 1.2f);
    H.at<float>(1,2) = rng.uniform(-0.1f, 0.1f)*src.rows;
    H.at<float>(2,0) = rng.uniform( -1e-4f, 1e-4f);
    H.at<float>(2,1) = rng.uniform( -1e-4f, 1e-4f);
    H.at<float>(2,2) = rng.uniform( 0.8f, 1.2f);

    warpPerspective( src, dst, H, src.size() );
}

static void doIteration( const Mat& img1, Mat& img2, bool isWarpPerspective,
                  vector<KeyPoint>& keypoints1, const Mat& descriptors1,
                  Ptr<FeatureDetector>& detector, Ptr<DescriptorExtractor>& descriptorExtractor,
                  Ptr<DescriptorMatcher>& descriptorMatcher, int matcherFilter, bool eval,
                  double ransacReprojThreshold, RNG& rng )
{
    CV_Assert( !img1.empty() );
    Mat H12;
    if( isWarpPerspective )
        warpPerspectiveRand(img1, img2, H12, rng );
    else
        CV_Assert( !img2.empty()/* && img2.cols==img1.cols && img2.rows==img1.rows*/ );

    cout << endl << "< Extracting keypoints from second image..." << endl;
    vector<KeyPoint> keypoints2;
    detector->detect( img2, keypoints2 );
    cout << keypoints2.size() << " points" << endl << ">" << endl;

    if( !H12.empty() && eval )
    {
        cout << "< Evaluate feature detector..." << endl;
        float repeatability;
        int correspCount;
        evaluateFeatureDetector( img1, img2, H12, &keypoints1, &keypoints2, repeatability, correspCount );
        cout << "repeatability = " << repeatability << endl;
        cout << "correspCount = " << correspCount << endl;
        cout << ">" << endl;
    }

    cout << "< Computing descriptors for keypoints from second image..." << endl;
    Mat descriptors2;
    descriptorExtractor->compute( img2, keypoints2, descriptors2 );
    cout << ">" << endl;

    cout << "< Matching descriptors..." << endl;
    vector<DMatch> filteredMatches;
    switch( matcherFilter )
    {
    case CROSS_CHECK_FILTER :
        crossCheckMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches, 1 );
        break;
    default :
        simpleMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches );
    }
    cout << ">" << endl;

    if( !H12.empty() && eval )
    {
        cout << "< Evaluate descriptor matcher..." << endl;
        vector<Point2f> curve;
        Ptr<GenericDescriptorMatcher> gdm = makePtr<VectorDescriptorMatcher>( descriptorExtractor, descriptorMatcher );
        evaluateGenericDescriptorMatcher( img1, img2, H12, keypoints1, keypoints2, 0, 0, curve, gdm );

        Point2f firstPoint = *curve.begin();
        Point2f lastPoint = *curve.rbegin();
        int prevPointIndex = -1;
        cout << "1-precision = " << firstPoint.x << "; recall = " << firstPoint.y << endl;
        for( float l_p = 0; l_p <= 1 + FLT_EPSILON; l_p+=0.05f )
        {
            int nearest = getNearestPoint( curve, l_p );
            if( nearest >= 0 )
            {
                Point2f curPoint = curve[nearest];
                if( curPoint.x > firstPoint.x && curPoint.x < lastPoint.x && nearest != prevPointIndex )
                {
                    cout << "1-precision = " << curPoint.x << "; recall = " << curPoint.y << endl;
                    prevPointIndex = nearest;
                }
            }
        }
        cout << "1-precision = " << lastPoint.x << "; recall = " << lastPoint.y << endl;
        cout << ">" << endl;
    }

    vector<int> queryIdxs( filteredMatches.size() ), trainIdxs( filteredMatches.size() );
    for( size_t i = 0; i < filteredMatches.size(); i++ )
    {
        queryIdxs[i] = filteredMatches[i].queryIdx;
        trainIdxs[i] = filteredMatches[i].trainIdx;
    }

    if( !isWarpPerspective && ransacReprojThreshold >= 0 )
    {
        cout << "< Computing homography (RANSAC)..." << endl;
        vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, queryIdxs);
        vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, trainIdxs);
        H12 = findHomography( Mat(points1), Mat(points2), RANSAC, ransacReprojThreshold );
        cout << ">" << endl;
    }

    Mat drawImg;
    if( !H12.empty() ) // filter outliers
    {
        vector<char> matchesMask( filteredMatches.size(), 0 );
        vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, queryIdxs);
        vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, trainIdxs);
        Mat points1t; perspectiveTransform(Mat(points1), points1t, H12);

        double maxInlierDist = ransacReprojThreshold < 0 ? 3 : ransacReprojThreshold;
        for( size_t i1 = 0; i1 < points1.size(); i1++ )
        {
            if( norm(points2[i1] - points1t.at<Point2f>((int)i1,0)) <= maxInlierDist ) // inlier
                matchesMask[i1] = 1;
        }
        // draw inliers
        drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg, Scalar(0, 255, 0), Scalar(255, 0, 0), matchesMask
#if DRAW_RICH_KEYPOINTS_MODE
                     , DrawMatchesFlags::DRAW_RICH_KEYPOINTS
#endif
                   );

#if DRAW_OUTLIERS_MODE
        // draw outliers
        for( size_t i1 = 0; i1 < matchesMask.size(); i1++ )
            matchesMask[i1] = !matchesMask[i1];
        drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg, Scalar(255, 0, 0), Scalar(0, 0, 255), matchesMask,
                     DrawMatchesFlags::DRAW_OVER_OUTIMG | DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
#endif

        cout << "Number of inliers: " << countNonZero(matchesMask) << endl;
    }
    else
        drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg );

    imshow( winName, drawImg );
}


int main(int argc, char** argv)
{
    if( argc != 7 && argc != 8 )
    {
        help(argv);
        return -1;
    }

    cv::initModule_nonfree();

    bool isWarpPerspective = argc == 7;
    double ransacReprojThreshold = -1;
    if( !isWarpPerspective )
        ransacReprojThreshold = atof(argv[7]);

    cout << "< Creating detector, descriptor extractor and descriptor matcher ..." << endl;
    Ptr<FeatureDetector> detector = FeatureDetector::create( argv[1] );
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create( argv[2] );
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( argv[3] );
    int mactherFilterType = getMatcherFilterType( argv[4] );
    bool eval = !isWarpPerspective ? false : (atoi(argv[6]) == 0 ? false : true);
    cout << ">" << endl;
    if( !detector || !descriptorExtractor || !descriptorMatcher )
    {
        cout << "Can not create detector or descriptor exstractor or descriptor matcher of given types" << endl;
        return -1;
    }

    cout << "< Reading the images..." << endl;
    Mat img1 = imread( argv[5] ), img2;
    if( !isWarpPerspective )
        img2 = imread( argv[6] );
    cout << ">" << endl;
    if( img1.empty() || (!isWarpPerspective && img2.empty()) )
    {
        cout << "Can not read images" << endl;
        return -1;
    }

    cout << endl << "< Extracting keypoints from first image..." << endl;
    vector<KeyPoint> keypoints1;
    detector->detect( img1, keypoints1 );
    cout << keypoints1.size() << " points" << endl << ">" << endl;

    cout << "< Computing descriptors for keypoints from first image..." << endl;
    Mat descriptors1;
    descriptorExtractor->compute( img1, keypoints1, descriptors1 );
    cout << ">" << endl;

    namedWindow(winName, 1);
    RNG rng = theRNG();
    doIteration( img1, img2, isWarpPerspective, keypoints1, descriptors1,
                 detector, descriptorExtractor, descriptorMatcher, mactherFilterType, eval,
                 ransacReprojThreshold, rng );
    for(;;)
    {
        char c = (char)waitKey(0);
        if( c == '\x1b' ) // esc
        {
            cout << "Exiting ..." << endl;
            break;
        }
        else if( isWarpPerspective )
        {
            doIteration( img1, img2, isWarpPerspective, keypoints1, descriptors1,
                         detector, descriptorExtractor, descriptorMatcher, mactherFilterType, eval,
                         ransacReprojThreshold, rng );
        }
    }
    return 0;
}
