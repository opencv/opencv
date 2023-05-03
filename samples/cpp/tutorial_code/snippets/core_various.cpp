#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    //! [Algorithm]
    Ptr<Feature2D> sbd = SimpleBlobDetector::create();
    FileStorage fs_read("SimpleBlobDetector_params.xml", FileStorage::READ);

    if (fs_read.isOpened()) // if we have file with parameters, read them
    {
        sbd->read(fs_read.root());
        fs_read.release();
    }
    else // else modify the parameters and store them; user can later edit the file to use different parameters
    {
        fs_read.release();
        FileStorage fs_write("SimpleBlobDetector_params.xml", FileStorage::WRITE);
        sbd->write(fs_write);
        fs_write.release();
    }

    Mat result, image = imread("../data/detect_blob.png", IMREAD_COLOR);
    vector<KeyPoint> keypoints;
    sbd->detect(image, keypoints, Mat());

    drawKeypoints(image, keypoints, result);
    for (vector<KeyPoint>::iterator k = keypoints.begin(); k != keypoints.end(); ++k)
        circle(result, k->pt, (int)k->size, Scalar(0, 0, 255), 2);

    imshow("result", result);
    waitKey(0);
    //! [Algorithm]

    const char * vertex_names[4] {"0", "1", "2", "3"};

    //! [RotatedRect_demo]
    Mat test_image(200, 200, CV_8UC3, Scalar(0));
    RotatedRect rRect = RotatedRect(Point2f(100,100), Size2f(100,50), 30);

    Point2f vertices[4];
    rRect.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        line(test_image, vertices[i], vertices[(i+1)%4], Scalar(0,255,0), 2);
        putText(test_image, vertex_names[i], vertices[i], FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255));
    }

    Rect brect = rRect.boundingRect();
    rectangle(test_image, brect, Scalar(255,0,0), 2);

    imshow("rectangles", test_image);
    waitKey(0);
    //! [RotatedRect_demo]

    {
        //! [TickMeter_total]
        TickMeter tm;
        tm.start();
        // do something ...
        tm.stop();
        cout << "Total time: " << tm.getTimeSec() << endl;
        //! [TickMeter_total]
    }

    {
        const int COUNT = 100;
        //! [TickMeter_average]
        TickMeter tm;
        for (int i = 0; i < COUNT; i++)
        {
            tm.start();
            // do something ...
            tm.stop();
        }
        cout << "Average time per iteration in seconds: " << tm.getAvgTimeSec() << endl;
        cout << "Average FPS: " << tm.getFPS() << endl;
        //! [TickMeter_average]
    }


    return 0;
}
