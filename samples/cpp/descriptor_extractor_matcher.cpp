#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
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

FeatureDetector* createDetector( const string& detectorType )
{
    FeatureDetector* fd = 0;
    if( !detectorType.compare( "FAST" ) )
    {
        fd = new FastFeatureDetector( 10/*threshold*/, true/*nonmax_suppression*/ );
    }
    else if( !detectorType.compare( "STAR" ) )
    {
        fd = new StarFeatureDetector( 16/*max_size*/, 5/*response_threshold*/, 10/*line_threshold_projected*/,
                                      8/*line_threshold_binarized*/, 5/*suppress_nonmax_size*/ );
    }
    else if( !detectorType.compare( "SIFT" ) )
    {
        fd = new SiftFeatureDetector(SIFT::DetectorParams::GET_DEFAULT_THRESHOLD(),
                                     SIFT::DetectorParams::GET_DEFAULT_EDGE_THRESHOLD());
    }
    else if( !detectorType.compare( "SURF" ) )
    {
        fd = new SurfFeatureDetector( 100./*hessian_threshold*/, 3 /*octaves*/, 4/*octave_layers*/ );
    }
    else if( !detectorType.compare( "MSER" ) )
    {
        fd = new MserFeatureDetector( 5/*delta*/, 60/*min_area*/, 14400/*_max_area*/, 0.25f/*max_variation*/,
                0.2/*min_diversity*/, 200/*max_evolution*/, 1.01/*area_threshold*/, 0.003/*min_margin*/,
                5/*edge_blur_size*/ );
    }
    else if( !detectorType.compare( "GFTT" ) )
    {
        fd = new GoodFeaturesToTrackDetector( 1000/*maxCorners*/, 0.01/*qualityLevel*/, 1./*minDistance*/,
                                              3/*int _blockSize*/, true/*useHarrisDetector*/, 0.04/*k*/ );
    }
    else
    {
        //CV_Error( CV_StsBadArg, "unsupported feature detector type");
    }
    return fd;
}

DescriptorExtractor* createDescriptorExtractor( const string& descriptorExtractorType )
{
    DescriptorExtractor* de = 0;
    if( !descriptorExtractorType.compare( "SIFT" ) )
    {
        de = new SiftDescriptorExtractor/*( double magnification=SIFT::DescriptorParams::GET_DEFAULT_MAGNIFICATION(),
                             bool isNormalize=true, bool recalculateAngles=true,
                             int nOctaves=SIFT::CommonParams::DEFAULT_NOCTAVES,
                             int nOctaveLayers=SIFT::CommonParams::DEFAULT_NOCTAVE_LAYERS,
                             int firstOctave=SIFT::CommonParams::DEFAULT_FIRST_OCTAVE,
                             int angleMode=SIFT::CommonParams::FIRST_ANGLE )*/;
    }
    else if( !descriptorExtractorType.compare( "SURF" ) )
    {
        de = new SurfDescriptorExtractor/*( int nOctaves=4, int nOctaveLayers=2, bool extended=false )*/;
    }
    else
    {
        //CV_Error( CV_StsBadArg, "unsupported descriptor extractor type");
    }
    return de;
}

DescriptorMatcher* createDescriptorMatcher( const string& descriptorMatcherType )
{
    DescriptorMatcher* dm = 0;
    if( !descriptorMatcherType.compare( "BruteForce" ) )
    {
        dm = new BruteForceMatcher<L2<float> >();
    }
    else
    {
        //CV_Error( CV_StsBadArg, "unsupported descriptor matcher type");
    }

    return dm;
}

void drawCorrespondences( const Mat& img1, const Mat& img2,
                          const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
                          const vector<int>& matches, Mat& drawImg, const Mat& H12 = Mat() )
{
    Scalar RED =   CV_RGB(255, 0, 0); // red keypoint - point without corresponding point
    Scalar GREEN = CV_RGB(0, 255, 0); // green keypoint - point having correct corresponding point
    Scalar BLUE =  CV_RGB(0, 0, 255); // blue keypoint - point having incorrect corresponding point

    Size size(img1.cols + img2.cols, MAX(img1.rows, img2.rows));
    drawImg.create(size, CV_MAKETYPE(img1.depth(), 3));
    Mat drawImg1 = drawImg(Rect(0, 0, img1.cols, img1.rows));
    cvtColor(img1, drawImg1, CV_GRAY2RGB);
    Mat drawImg2 = drawImg(Rect(img1.cols, 0, img2.cols, img2.rows));
    cvtColor(img2, drawImg2, CV_GRAY2RGB);

    // draw keypoints
    for(vector<KeyPoint>::const_iterator it = keypoints1.begin(); it < keypoints1.end(); ++it )
    {
        circle(drawImg, it->pt, 3, RED);
    }
    for(vector<KeyPoint>::const_iterator it = keypoints2.begin(); it < keypoints2.end(); ++it )
    {
		Point p = it->pt;
        circle(drawImg, Point2f(p.x+img1.cols, p.y), 3, RED);
    }
    
    // draw matches
    vector<int>::const_iterator mit = matches.begin();
    assert( matches.size() == keypoints1.size() );
    for( int i1 = 0; mit != matches.end(); ++mit, i1++ )
    {
        Point2f pt1 = keypoints1[i1].pt,
                pt2 = keypoints2[*mit].pt,
                dpt2 = Point2f( std::min(pt2.x+img1.cols, float(drawImg.cols-1)), pt2.y);
        if( !H12.empty() )
        {
            if( norm(pt2 - applyHomography(H12, pt1)) > 3 )
            {
                circle(drawImg, pt1, 3, BLUE);
                circle(drawImg, dpt2, 3, BLUE);
                continue;
            }
        }
        circle(drawImg, pt1, 3, GREEN);
        circle(drawImg, dpt2, 3, GREEN);
        line(drawImg, pt1, dpt2, GREEN);
    }
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
        assert( !img2.empty() && img2.cols==img1.cols && img2.rows==img1.rows );

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
    drawCorrespondences( img1, img2, keypoints1, keypoints2, matches, drawImg, H12 );
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
