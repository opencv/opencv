#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace cv;

static void help(char** argv)
{
    cout
    << "This is a sample usage of AffineFeature detector/extractor.\n"
    << "And this is a C++ version of samples/python/asift.py\n"
    << "Usage: " << argv[0] << "\n"
    << "     [ --feature=<sift|orb|brisk> ]         # Feature to use.\n"
    << "     [ --flann ]                                 # use Flann-based matcher instead of bruteforce.\n"
    << "     [ --image1=<image1(aero1.jpg as default)> ]\n"
    << "     [ --image2=<image2(aero3.jpg as default)> ] # Path to images to compare."
    << endl;
}

static double timer()
{
    return getTickCount() / getTickFrequency();
}

int main(int argc, char** argv)
{
    vector<String> fileName;
    cv::CommandLineParser parser(argc, argv,
        "{help h ||}"
        "{feature|brisk|}"
        "{flann||}"
        "{@image1|aero1.jpg|}{@image2|aero3.jpg|}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    bool useFlann = parser.has("flann");
    fileName.push_back(samples::findFile(parser.get<string>(0)));
    fileName.push_back(samples::findFile(parser.get<string>(1)));
    Mat img1 = imread(fileName[0], IMREAD_GRAYSCALE);
    Mat img2 = imread(fileName[1], IMREAD_GRAYSCALE);
    if (img1.empty())
    {
        cerr << "Image " << fileName[0] << " is empty or cannot be found" << endl;
        return 1;
    }
    if (img2.empty())
    {
        cerr << "Image " << fileName[1] << " is empty or cannot be found" << endl;
        return 1;
    }

    string feature = parser.get<string>("feature");
    Ptr<Feature2D> backend;
    Ptr<DescriptorMatcher> matcher;

    if (feature == "sift")
    {
        backend = SIFT::create();
        if (useFlann)
            matcher = DescriptorMatcher::create("FlannBased");
        else
            matcher = DescriptorMatcher::create("BruteForce");
    }
    else if (feature == "orb")
    {
        backend = ORB::create();
        if (useFlann)
            matcher = makePtr<FlannBasedMatcher>(makePtr<flann::LshIndexParams>(6, 12, 1));
        else
            matcher = DescriptorMatcher::create("BruteForce-Hamming");
    }
    else if (feature == "brisk")
    {
        backend = BRISK::create();
        if (useFlann)
            matcher = makePtr<FlannBasedMatcher>(makePtr<flann::LshIndexParams>(6, 12, 1));
        else
            matcher = DescriptorMatcher::create("BruteForce-Hamming");
    }
    else
    {
        cerr << feature << " is not supported. See --help" << endl;
        return 1;
    }

    cout << "extracting with " << feature << "..." << endl;
    Ptr<AffineFeature> ext = AffineFeature::create(backend);
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;

    ext->detectAndCompute(img1, Mat(), kp1, desc1);
    ext->detectAndCompute(img2, Mat(), kp2, desc2);
    cout << "img1 - " << kp1.size() << " features, "
         << "img2 - " << kp2.size() << " features"
         << endl;

    cout << "matching with " << (useFlann ? "flann" : "bruteforce") << "..." << endl;
    double start = timer();
    // match and draw
    vector< vector<DMatch> > rawMatches;
    vector<Point2f> p1, p2;
    matcher->knnMatch(desc1, desc2, rawMatches, 2);
    // filter_matches
    for (size_t i = 0; i < rawMatches.size(); i++)
    {
        const vector<DMatch>& m = rawMatches[i];
        if (m.size() == 2 && m[0].distance < m[1].distance * 0.75)
        {
            // matches.push_back(m[0]);
            p1.push_back(kp1[m[0].queryIdx].pt);
            p2.push_back(kp2[m[0].trainIdx].pt);
        }
    }
    vector<uchar> status;
    vector< pair<const Point2f&, const Point2f&> > pointPairs;
    Mat H = findHomography(p1, p2, status, RANSAC);
    int inliers = 0;
    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i])
        {
            inliers++;
            pointPairs.emplace_back(p1[i], p2[i]);
        }
    }
    cout << "execution time: " << fixed << setprecision(2) << (timer()-start)*1000 << " ms" << endl;
    cout << inliers << " / " << status.size() << " inliers/matched" << endl;

    cout << "visualizing..." << endl;
    // explore_match
    int h1 = img1.size().height;
    int w1 = img1.size().width;
    int h2 = img2.size().height;
    int w2 = img2.size().width;
    Mat vis = Mat::zeros(max(h1, h2), w1+w2, CV_8U);
    img1.copyTo(Mat(vis, Rect(0, 0, w1, h1)));
    img2.copyTo(Mat(vis, Rect(w1, 0, w2, h2)));
    cvtColor(vis, vis, COLOR_GRAY2BGR);

    vector<Point2f> corners{{0,0}, {(float)w1,0}, {(float)w1,(float)h1}, {0,(float)h1}};
    vector<Point2i> icorners;
    perspectiveTransform(corners, corners, H);
    transform(corners, corners, Matx23f(1,0,w1,0,1,0));
    Mat(corners).convertTo(icorners, CV_32S);
    polylines(vis, {icorners}, true, Scalar(255,255,255));

    for (size_t i = 0; i < pointPairs.size(); i++)
    {
        const Point2f& p1 = pointPairs[i].first;
        const Point2f& p2 = pointPairs[i].second;
        circle(vis, p1, 2, Scalar(0,255,0), -1);
        circle(vis, p2 + Point2f(w1,0), 2, Scalar(0,255,0), -1);
        line(vis, p1, p2 + Point2f(w1,0), Scalar(0,255,0));
    }
    imshow("affine find_obj", vis);

    // Mat vis2 = Mat::zeros(max(h1, h2), w1+w2, CV_8U);
    // Mat warp1;
    // warpPerspective(img1, warp1, H, Size(w1, h1));
    // warp1.copyTo(Mat(vis2, Rect(0, 0, w1, h1)));
    // img2.copyTo(Mat(vis2, Rect(w1, 0, w2, h2)));
    // imshow("warped", vis2);

    waitKey();
    cout << "done" << endl;
    return 0;
}
