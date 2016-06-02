#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core.hpp>
#include <vector>
#include "stats.h"

using namespace std;
using namespace cv;

void drawBoundingBox(Mat image, vector<Point2f> bb);
void drawStatistics(Mat image, const Stats& stats);
void printStatistics(string name, Stats stats);
vector<Point2f> Points(vector<KeyPoint> keypoints);
Rect2d selectROI(const String &video_name, const Mat &frame);

void drawBoundingBox(Mat image, vector<Point2f> bb)
{
    for(unsigned i = 0; i < bb.size() - 1; i++) {
        line(image, bb[i], bb[i + 1], Scalar(0, 0, 255), 2);
    }
    line(image, bb[bb.size() - 1], bb[0], Scalar(0, 0, 255), 2);
}

void drawStatistics(Mat image, const Stats& stats)
{
    static const int font = FONT_HERSHEY_PLAIN;
    stringstream str1, str2, str3;

    str1 << "Matches: " << stats.matches;
    str2 << "Inliers: " << stats.inliers;
    str3 << "Inlier ratio: " << setprecision(2) << stats.ratio;

    putText(image, str1.str(), Point(0, image.rows - 90), font, 2, Scalar::all(255), 3);
    putText(image, str2.str(), Point(0, image.rows - 60), font, 2, Scalar::all(255), 3);
    putText(image, str3.str(), Point(0, image.rows - 30), font, 2, Scalar::all(255), 3);
}

void printStatistics(string name, Stats stats)
{
    cout << name << endl;
    cout << "----------" << endl;

    cout << "Matches " << stats.matches << endl;
    cout << "Inliers " << stats.inliers << endl;
    cout << "Inlier ratio " << setprecision(2) << stats.ratio << endl;
    cout << "Keypoints " << stats.keypoints << endl;
    cout << endl;
}

vector<Point2f> Points(vector<KeyPoint> keypoints)
{
    vector<Point2f> res;
    for(unsigned i = 0; i < keypoints.size(); i++) {
        res.push_back(keypoints[i].pt);
    }
    return res;
}

Rect2d selectROI(const String &video_name, const Mat &frame)
{
    struct Data
    {
        Point center;
        Rect2d box;

        static void mouseHandler(int event, int x, int y, int flags, void *param)
        {
            Data *data = (Data*)param;
            switch( event )
            {
            // start to select the bounding box
            case EVENT_LBUTTONDOWN:
                data->box = cvRect( x, y, 0, 0 );
                data->center = Point2f((float)x,(float)y);
                break;
            // update the selected bounding box
            case EVENT_MOUSEMOVE:
                if(flags == 1)
                {
                    data->box.width  = 2 * (x - data->center.x);
                    data->box.height = 2 * (y - data->center.y);
                    data->box.x = data->center.x - data->box.width / 2.0;
                    data->box.y = data->center.y - data->box.height / 2.0;
                }
                break;
            // cleaning up the selected bounding box
            case EVENT_LBUTTONUP:
                if( data->box.width < 0 )
                {
                    data->box.x += data->box.width;
                    data->box.width *= -1;
                }
                if( data->box.height < 0 )
                {
                    data->box.y += data->box.height;
                    data->box.height *= -1;
                }
                break;
            }
        }
    } data;

    setMouseCallback(video_name, Data::mouseHandler, &data);
    while(waitKey(1) < 0)
    {
        Mat draw = frame.clone();
        rectangle(draw, data.box, Scalar(255,0,0), 2, 1);
        imshow(video_name, draw);
    }
    return data.box;
}

#endif // UTILS_H
