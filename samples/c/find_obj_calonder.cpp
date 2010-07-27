#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;


void warpPerspectiveRand( const Mat& src, Mat& dst, Mat& H, RNG& rng )
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

int main( int argc, char **argv )
{
    #if 0
    if( argc != 4 && argc != 3 )
    {
        cout << "Format:" << endl <<
                "   classifier(xml to write) test_image file_with_train_images_filenames(txt)" <<
                "   or" << endl <<
                "   classifier(xml to read) test_image" << endl;
        return -1;
    }

    CalonderClassifier classifier;
    if( argc == 4 ) // Train
    {
        // Read train images and test image
        ifstream fst( argv[3], ifstream::in );
        vector<Mat> trainImgs;
        while( !fst.eof() )
        {
            string str;
            getline( fst, str );
            if (str.empty()) break;
            Mat img = imread( str, CV_LOAD_IMAGE_GRAYSCALE );
            if( !img.empty() )
                trainImgs.push_back( img );
        }
        if( trainImgs.empty() )
        {
            cout << "All train images can not be read." << endl;
            return -1;
        }
        cout << trainImgs.size() << " train images were read." << endl;

        // Extract keypoints from train images
        SurfFeatureDetector detector;
        vector<vector<Point2f> > trainPoints( trainImgs.size() );
        for( size_t i = 0; i < trainImgs.size(); i++ )
        {
            vector<KeyPoint> kps;
            detector.detect( trainImgs[i], kps );
            KeyPoint::convert( kps, trainPoints[i] );
        }

        // Train Calonder classifier on extracted points
        classifier.setVerbose( true);
        classifier.train( trainPoints, trainImgs );

        // Write Calonder classifier
        FileStorage fs( argv[1], FileStorage::WRITE );
        if( fs.isOpened() ) classifier.write( fs );
    }
    else
    {
        // Read Calonder classifier
        FileStorage fs( argv[1], FileStorage::READ );
        if( fs.isOpened() ) classifier.read( fs.root() );
    }

    if( classifier.empty() )
    {
        cout << "Calonder classifier is empty" << endl;
        return -1;
    }

    // Test Calonder classifier on test image and warped one
    Mat testImg1 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE ), testImg2, H12;
    if( testImg1.empty() )
    {
        cout << "Test image can not be read." << endl;
        return -1;
    }
    warpPerspectiveRand( testImg1, testImg2, H12, theRNG() );


    // Exstract keypoints from test images
    SurfFeatureDetector detector;
    vector<KeyPoint> testKeypoints1; detector.detect( testImg1, testKeypoints1 );
    vector<KeyPoint> testKeypoints2; detector.detect( testImg2, testKeypoints2 );
    vector<Point2f> testPoints1; KeyPoint::convert( testKeypoints1, testPoints1 );
    vector<Point2f> testPoints2; KeyPoint::convert( testKeypoints2, testPoints2 );

    // Calculate Calonder descriptors
    int signatureSize = classifier.getSignatureSize();
    vector<float> r1(testPoints1.size()*signatureSize), r2(testPoints2.size()*signatureSize);
    vector<float>::iterator rit = r1.begin();
    for( size_t i = 0; i < testPoints1.size(); i++ )
    {
        vector<float> s;
        classifier( testImg1, testPoints1[i], s );
        copy( s.begin(), s.end(), rit );
        rit += s.size();
    }
    rit = r2.begin();
    for( size_t i = 0; i < testPoints2.size(); i++ )
    {
        vector<float> s;
        classifier( testImg2, testPoints2[i], s );
        copy( s.begin(), s.end(), rit );
        rit += s.size();
    }

    Mat descriptors1(testPoints1.size(), classifier.getSignatureSize(), CV_32FC1, &r1[0] ),
        descriptors2(testPoints2.size(), classifier.getSignatureSize(), CV_32FC1, &r2[0] );

    // Match descriptors
    BruteForceMatcher<L1<float> > matcher;
    matcher.add( descriptors2 );
    vector<int> matches;
    matcher.match( descriptors1, matches );

    // Draw results
    // Prepare inlier mask
    vector<char> matchesMask( matches.size(), 0 );
    Mat points1t; perspectiveTransform(Mat(testPoints1), points1t, H12);
    vector<int>::const_iterator mit = matches.begin();
    for( size_t mi = 0; mi < matches.size(); mi++ )
    {
        if( norm(testPoints2[matches[mi]] - points1t.at<Point2f>(mi,0)) < 4 ) // inlier
            matchesMask[mi] = 1;
    }
    // Draw
    Mat drawImg;
    drawMatches( testImg1, testKeypoints1, testImg2, testKeypoints2, matches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask  );
    string winName = "Matches";
    namedWindow( winName, WINDOW_AUTOSIZE );
    imshow( winName, drawImg );
    waitKey();
#endif
    return 0;
}
