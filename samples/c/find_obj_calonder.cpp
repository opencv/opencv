#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

static void help()
{
    cout << "This program shows the use of the Calonder point descriptor classifier"
            "SURF is used to detect interest points, Calonder is used to describe/match these points\n"
            "Format:" << endl <<
            "   classifier_file(to write) test_image file_with_train_images_filenames(txt)" <<
            "   or" << endl <<
            "   classifier_file(to read) test_image" << "\n" << endl <<
            "Using OpenCV version " << CV_VERSION << "\n" << endl;

    return;
}


/*
 * Generates random perspective transform of image
 */
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

/*
 * Trains Calonder classifier and writes trained classifier in file:
 *      imgFilename - name of .txt file which contains list of full filenames of train images,
 *      classifierFilename - name of binary file in which classifier will be written.
 *
 * To train Calonder classifier RTreeClassifier class need to be used.
 */
static void trainCalonderClassifier( const string& classifierFilename, const string& imgFilename )
{
    // Reads train images
    ifstream is( imgFilename.c_str(), ifstream::in );
    vector<Mat> trainImgs;
    while( !is.eof() )
    {
        string str;
        getline( is, str );
        if (str.empty()) break;
        Mat img = imread( str, CV_LOAD_IMAGE_GRAYSCALE );
        if( !img.empty() )
            trainImgs.push_back( img );
    }
    if( trainImgs.empty() )
    {
        cout << "All train images can not be read." << endl;
        exit(-1);
    }
    cout << trainImgs.size() << " train images were read." << endl;

    // Extracts keypoints from train images
    SurfFeatureDetector detector;
    vector<BaseKeypoint> trainPoints;
    vector<IplImage> iplTrainImgs(trainImgs.size());
    for( size_t imgIdx = 0; imgIdx < trainImgs.size(); imgIdx++ )
    {
        iplTrainImgs[imgIdx] = trainImgs[imgIdx];
        vector<KeyPoint> kps; detector.detect( trainImgs[imgIdx], kps );

        for( size_t pointIdx = 0; pointIdx < kps.size(); pointIdx++ )
        {
            Point2f p = kps[pointIdx].pt;
            trainPoints.push_back( BaseKeypoint(cvRound(p.x), cvRound(p.y), &iplTrainImgs[imgIdx]) );
        }
    }

    // Trains Calonder classifier on extracted points
    RTreeClassifier classifier;
    classifier.train( trainPoints, theRNG(), 48, 9, 100 );
    // Writes classifier
    classifier.write( classifierFilename.c_str() );
}

/*
 * Test Calonder classifier to match keypoints on given image:
 *      classifierFilename - name of file from which classifier will be read,
 *      imgFilename - test image filename.
 *
 * To calculate keypoint descriptors you may use RTreeClassifier class (as to train),
 * but it is convenient to use CalonderDescriptorExtractor class which is wrapper of
 * RTreeClassifier.
 */
static void testCalonderClassifier( const string& classifierFilename, const string& imgFilename )
{
    Mat img1 = imread( imgFilename, CV_LOAD_IMAGE_GRAYSCALE ), img2, H12;
    if( img1.empty() )
    {
        cout << "Test image can not be read." << endl;
        exit(-1);
    }
    warpPerspectiveRand( img1, img2, H12, theRNG() );

    // Exstract keypoints from test images
    SurfFeatureDetector detector;
    vector<KeyPoint> keypoints1; detector.detect( img1, keypoints1 );
    vector<KeyPoint> keypoints2; detector.detect( img2, keypoints2 );

    // Compute descriptors
    CalonderDescriptorExtractor<float> de( classifierFilename );
    Mat descriptors1;  de.compute( img1, keypoints1, descriptors1 );
    Mat descriptors2;  de.compute( img2, keypoints2, descriptors2 );

    // Match descriptors
    BFMatcher matcher(NORM_L1);
    vector<DMatch> matches;
    matcher.match( descriptors1, descriptors2, matches );

    // Prepare inlier mask
    vector<char> matchesMask( matches.size(), 0 );
    vector<Point2f> points1; KeyPoint::convert( keypoints1, points1 );
    vector<Point2f> points2; KeyPoint::convert( keypoints2, points2 );
    Mat points1t; perspectiveTransform(Mat(points1), points1t, H12);
    for( size_t mi = 0; mi < matches.size(); mi++ )
    {
        if( norm(points2[matches[mi].trainIdx] - points1t.at<Point2f>((int)mi,0)) < 4 ) // inlier
            matchesMask[mi] = 1;
    }

    // Draw
    Mat drawImg;
    drawMatches( img1, keypoints1, img2, keypoints2, matches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask );
    string winName = "Matches";
    namedWindow( winName, WINDOW_AUTOSIZE );
    imshow( winName, drawImg );
    waitKey();
}

int main( int argc, char **argv )
{
    if( argc != 4 && argc != 3 )
    {
        help();
        return -1;
    }

    if( argc == 4 )
        trainCalonderClassifier( argv[1], argv[3] );

    testCalonderClassifier( argv[1], argv[2] );

    return 0;
}
