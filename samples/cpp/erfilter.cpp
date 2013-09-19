
//--------------------------------------------------------------------------------------------------
//  A demo program of the Extremal Region Filter algorithm described in
//  Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
//--------------------------------------------------------------------------------------------------

#include  "opencv2/opencv.hpp"
#include  "opencv2/objdetect.hpp"
#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"

#include  <vector>
#include  <iostream>
#include  <iomanip>

using  namespace std;
using  namespace cv;

void show_help_and_exit(const char *cmd);
void groups_draw(Mat &src, vector<Rect> &groups);
void er_draw(Mat &src, Mat &dst, ERStat& er);

int  main(int argc, const char * argv[])
{

    if (argc < 2) show_help_and_exit(argv[0]);

    Mat src = imread(argv[1]);

    // Extract channels to be processed individually
    vector<Mat> channels;
    computeNMChannels(src, channels);

    int cn = (int)channels.size();
    // Append negative channels to detect ER- (bright regions over dark background)
    for (int c = 0; c < cn-1; c++)
        channels.push_back(255-channels[c]);

    // Create ERFilter objects with the 1st and 2nd stage default classifiers
    Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"),8,0.00025,0.13,0.4,true,0.1);
    Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),0.3);

    vector<vector<ERStat> > regions(channels.size());
    // Apply the default cascade classifier to each independent channel (could be done in parallel)
    for (int c=0; c<(int)channels.size(); c++)
    {
        er_filter1->run(channels[c], regions[c]);
        er_filter2->run(channels[c], regions[c]);
    }

    // Detect character groups
    vector<Rect> groups;
    erGrouping(channels, regions, groups);

    // draw groups
    groups_draw(src, groups);
    imshow("grouping",src);
    waitKey(-1);

    // memory clean-up
    er_filter1.release();
    er_filter2.release();
    regions.clear();
    if (!groups.empty())
    {
        groups.clear();
    }
}



// helper functions

void show_help_and_exit(const char *cmd)
{
    cout << endl << cmd << endl << endl;
    cout << "Demo program of the Extremal Region Filter algorithm described in " << endl;
    cout << "Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012" << endl << endl;
    cout << "    Usage: " << cmd << " <input_image> " << endl;
    cout << "    Default classifier files (trained_classifierNM*.xml) must be in current directory" << endl << endl;
    exit(-1);
}

void groups_draw(Mat &src, vector<Rect> &groups)
{
    for (int i=groups.size()-1; i>=0; i--)
    {
        if (src.type() == CV_8UC3)
            rectangle(src,groups.at(i).tl(),groups.at(i).br(),Scalar( 0, 255, 255 ), 3, 8 );
        else
            rectangle(src,groups.at(i).tl(),groups.at(i).br(),Scalar( 255 ), 3, 8 );
    }
}

void er_draw(Mat &src, Mat &dst, ERStat& er)
{

    if (er.parent != NULL) // deprecate the root region
    {
        int newMaskVal = 255;
        int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
        floodFill(src,dst,Point(er.pixel%src.cols,er.pixel/src.cols),Scalar(255),0,Scalar(er.level),Scalar(0),flags);
    }

}
