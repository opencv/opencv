#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>     //for ROI
#include <opencv2/highgui.hpp>      //for imshow
#include <vector>
#include <iostream>
#include <iomanip>

#include "stats.h" // Stats structure definition
#include "utils.h" // Drawing and printing functions

using namespace std;
using namespace cv;

const double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
const int bb_min_inliers = 100; // Minimal number of inliers to draw bounding box
const int stats_update_period = 10; // On-screen statistics are updated every 10 frames

namespace example {
class Tracker
{
public:
    Tracker(Ptr<Feature2D> _detector, Ptr<DescriptorMatcher> _matcher) :
        detector(_detector),
        matcher(_matcher)
    {}

    void setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats);
    Mat process(const Mat frame, Stats& stats);
    Ptr<Feature2D> getDetector() {
        return detector;
    }
protected:
    Ptr<Feature2D> detector;
    Ptr<DescriptorMatcher> matcher;
    Mat first_frame, first_desc;
    vector<KeyPoint> first_kp;
    vector<Point2f> object_bb;
};

void Tracker::setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats)
{
    cv::Point *ptMask = new cv::Point[bb.size()];
    const Point* ptContain = { &ptMask[0] };
    int iSize = static_cast<int>(bb.size());
    for (size_t i=0; i<bb.size(); i++) {
        ptMask[i].x = bb[i].x;
        ptMask[i].y = bb[i].y;
    }
    first_frame = frame.clone();
    cv::Mat matMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::fillPoly(matMask, &ptContain, &iSize, 1, cv::Scalar::all(255));
    detector->detectAndCompute(first_frame, matMask, first_kp, first_desc);
    stats.keypoints = (int)first_kp.size();
    drawBoundingBox(first_frame, bb);
    putText(first_frame, title, Point(0, 60), FONT_HERSHEY_PLAIN, 5, Scalar::all(0), 4);
    object_bb = bb;
    delete ptMask;
}

Mat Tracker::process(const Mat frame, Stats& stats)
{
    vector<KeyPoint> kp;
    Mat desc;
    detector->detectAndCompute(frame, noArray(), kp, desc);
    stats.keypoints = (int)kp.size();

    vector< vector<DMatch> > matches;
    vector<KeyPoint> matched1, matched2;
    matcher->knnMatch(first_desc, desc, matches, 2);
    for(unsigned i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
            matched1.push_back(first_kp[matches[i][0].queryIdx]);
            matched2.push_back(      kp[matches[i][0].trainIdx]);
        }
    }
    stats.matches = (int)matched1.size();

    Mat inlier_mask, homography;
    vector<KeyPoint> inliers1, inliers2;
    vector<DMatch> inlier_matches;
    if(matched1.size() >= 4) {
        homography = findHomography(Points(matched1), Points(matched2),
                                    RANSAC, ransac_thresh, inlier_mask);
    }

    if(matched1.size() < 4 || homography.empty()) {
        Mat res;
        hconcat(first_frame, frame, res);
        stats.inliers = 0;
        stats.ratio = 0;
        return res;
    }
    for(unsigned i = 0; i < matched1.size(); i++) {
        if(inlier_mask.at<uchar>(i)) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            inlier_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    stats.inliers = (int)inliers1.size();
    stats.ratio = stats.inliers * 1.0 / stats.matches;

    vector<Point2f> new_bb;
    perspectiveTransform(object_bb, new_bb, homography);
    Mat frame_with_bb = frame.clone();
    if(stats.inliers >= bb_min_inliers) {
        drawBoundingBox(frame_with_bb, new_bb);
    }
    Mat res;
    drawMatches(first_frame, inliers1, frame_with_bb, inliers2,
                inlier_matches, res,
                Scalar(255, 0, 0), Scalar(255, 0, 0));
    return res;
}
}

int main(int argc, char **argv)
{
    if(argc < 3) {
        cerr << "Usage: " << endl <<
                "akaze_track input_path output_path [bounding_box_path]" << endl
                "  (for camera input_path=N for camera N)" << endl;
        return 1;
    }

    std::string video_name = argv[1];
    std::stringstream ssFormat;
    ssFormat << atoi(argv[1]);
    VideoCapture video_in;
    int iFourCC = 0, frame_count = 0;
    if (video_name.compare(ssFormat.str())==0) {      //test str==str(num)
        video_in.open(atoi(argv[1]));
        cerr << "Capturing for 10 seconds from camera..." << endl;
        iFourCC = CV_FOURCC('D', 'I', 'V', 'X');    //default to mp4 (sample)
        frame_count = 10*static_cast<int>(video_in.get(CAP_PROP_FPS));
    }
    else {
        video_in.open(video_name);
        iFourCC = static_cast<int>(video_in.get(CAP_PROP_FOURCC));
        frame_count = static_cast<int>(video_in.get(CAP_PROP_FRAME_COUNT));
    }

    VideoWriter  video_out(argv[2], iFourCC,
                           (int)video_in.get(CAP_PROP_FPS),
                           Size(2 * (int)video_in.get(CAP_PROP_FRAME_WIDTH),
                                2 * (int)video_in.get(CAP_PROP_FRAME_HEIGHT)));

    if(!video_in.isOpened()) {
        cerr << "Couldn't open " << argv[1] << endl;
        return 1;
    }
    if(!video_out.isOpened()) {
        cerr << "Couldn't open " << argv[2] << endl;
        return 1;
    }

    Stats stats, akaze_stats, orb_stats;
    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->setThreshold(akaze_thresh);
    Ptr<ORB> orb = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    example::Tracker akaze_tracker(akaze, matcher);
    example::Tracker orb_tracker(orb, matcher);

    Mat frame;
    video_in >> frame;
    vector<Point2f> bb;
    if (argc < 4) {             //attempt to alow GUI selection
        cv::Rect2d uBox = selectROI(video_name, frame);
        bb.push_back(cv::Point2f(uBox.x, uBox.y));
        bb.push_back(cv::Point2f(uBox.x+uBox.width, uBox.y));
        bb.push_back(cv::Point2f(uBox.x+uBox.width, uBox.y+uBox.height));
        bb.push_back(cv::Point2f(uBox.x, uBox.y+uBox.height));
    }
    else {
        FileStorage fs(argv[3], FileStorage::READ);
        if(fs["bounding_box"].empty()) {
            cerr << "Couldn't read bounding_box from " << argv[3] << endl;
            return 1;
        }
        fs["bounding_box"] >> bb;
    }
    akaze_tracker.setFirstFrame(frame, bb, "AKAZE", stats);
    orb_tracker.setFirstFrame(frame, bb, "ORB", stats);

    Stats akaze_draw_stats, orb_draw_stats;
    Mat akaze_res, orb_res, res_frame;
    int i = 1;
    for(i = 1; i < frame_count; i++) {
        bool update_stats = (i % stats_update_period == 0);
        video_in >> frame;
        // stop the program if no more images
        if(frame.rows==0 || frame.cols==0) break;

        akaze_res = akaze_tracker.process(frame, stats);
        akaze_stats += stats;
        if(update_stats) {
            akaze_draw_stats = stats;
        }

        orb->setMaxFeatures(stats.keypoints);
        orb_res = orb_tracker.process(frame, stats);
        orb_stats += stats;
        if(update_stats) {
            orb_draw_stats = stats;
        }

        drawStatistics(akaze_res, akaze_draw_stats);
        drawStatistics(orb_res, orb_draw_stats);
        vconcat(akaze_res, orb_res, res_frame);
        video_out << res_frame;
        cv::imshow(video_name, res_frame);
        if (i==1)                  //resize for easier display
            cv::resizeWindow(video_name, frame.cols, frame.rows);
        if(cv::waitKey(1)==27)break; //quit on ESC button
        cout << i << "/" << frame_count - 1 << endl;
    }
    akaze_stats /= i - 1;
    orb_stats /= i - 1;
    printStatistics("AKAZE", akaze_stats);
    printStatistics("ORB", orb_stats);
    return 0;
}
