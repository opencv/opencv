#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/3d.hpp>
#include <iostream>
#include <iomanip>
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#endif

using namespace std;
using namespace cv;

static void help(char** argv)
{
    cout
    << "This is a sample usage of AffineFeature detector/extractor.\n"
    << "And this is a C++ version of samples/python/asift.py\n"
    << "Usage: " << argv[0] << "\n"
    << "     [ --feature=<sift|orb|brisk> ]         # Feature to use.\n"
    << "     [ --flann ]                            # use Flann-based matcher instead of bruteforce.\n"
    << "     [ --maxlines=<number(50 as default)> ] # The maximum number of lines in visualizing the matching result.\n"
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
        "{maxlines|50|}"
        "{image1|aero1.jpg|}{image2|aero3.jpg|}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    string feature = parser.get<string>("feature");
    bool useFlann = parser.has("flann");
    int maxlines = parser.get<int>("maxlines");
    fileName.push_back(samples::findFile(parser.get<string>("image1")));
    fileName.push_back(samples::findFile(parser.get<string>("image2")));
    if (!parser.check())
    {
        parser.printErrors();
        cout << "See --help (or missing '=' between argument name and value?)" << endl;
        return 1;
    }

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
#ifdef HAVE_OPENCV_XFEATURES2D
    else if (feature == "brisk")
    {
        backend = xfeatures2d::BRISK::create();
        if (useFlann)
            matcher = makePtr<FlannBasedMatcher>(makePtr<flann::LshIndexParams>(6, 12, 1));
        else
            matcher = DescriptorMatcher::create("BruteForce-Hamming");
    }
#endif
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
    vector<float> distances;
    matcher->knnMatch(desc1, desc2, rawMatches, 2);
    // filter_matches
    for (size_t i = 0; i < rawMatches.size(); i++)
    {
        const vector<DMatch>& m = rawMatches[i];
        if (m.size() == 2 && m[0].distance < m[1].distance * 0.75)
        {
            p1.push_back(kp1[m[0].queryIdx].pt);
            p2.push_back(kp2[m[0].trainIdx].pt);
            distances.push_back(m[0].distance);
        }
    }
    vector<uchar> status;
    vector< pair<Point2f, Point2f> > pointPairs;
    Mat H = findHomography(p1, p2, status, RANSAC);
    int inliers = 0;
    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i])
        {
            pointPairs.push_back(make_pair(p1[i], p2[i]));
            distances[inliers] = distances[i];
            // CV_Assert(inliers <= (int)i);
            inliers++;
        }
    }
    distances.resize(inliers);

    cout << "execution time: " << fixed << setprecision(2) << (timer()-start)*1000 << " ms" << endl;
    cout << inliers << " / " << status.size() << " inliers/matched" << endl;

    cout << "visualizing..." << endl;
    vector<int> indices(inliers);
    cv::sortIdx(distances, indices, SORT_EVERY_ROW+SORT_ASCENDING);

    // explore_match
    int h1 = img1.size().height;
    int w1 = img1.size().width;
    int h2 = img2.size().height;
    int w2 = img2.size().width;
    Mat vis = Mat::zeros(max(h1, h2), w1+w2, CV_8U);
    img1.copyTo(Mat(vis, Rect(0, 0, w1, h1)));
    img2.copyTo(Mat(vis, Rect(w1, 0, w2, h2)));
    cvtColor(vis, vis, COLOR_GRAY2BGR);

    vector<Point2f> corners(4);
    corners[0] = Point2f(0, 0);
    corners[1] = Point2f((float)w1, 0);
    corners[2] = Point2f((float)w1, (float)h1);
    corners[3] = Point2f(0, (float)h1);
    vector<Point2i> icorners;
    perspectiveTransform(corners, corners, H);
    transform(corners, corners, Matx23f(1,0,(float)w1,0,1,0));
    Mat(corners).convertTo(icorners, CV_32S);
    polylines(vis, icorners, true, Scalar(255,255,255));

    for (int i = 0; i < min(inliers, maxlines); i++)
    {
        int idx = indices[i];
        const Point2f& pi1 = pointPairs[idx].first;
        const Point2f& pi2 = pointPairs[idx].second;
        circle(vis, pi1, 2, Scalar(0,255,0), -1);
        circle(vis, pi2 + Point2f((float)w1,0), 2, Scalar(0,255,0), -1);
        line(vis, pi1, pi2 + Point2f((float)w1,0), Scalar(0,255,0));
    }
    if (inliers > maxlines)
        cout << "only " << maxlines << " inliers are visualized" << endl;
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
