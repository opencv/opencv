#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utility.hpp"

#include <ctype.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

static void help()
{
    cout << "\nThis program demonstrates SEEDS superpixels using OpenCV class SuperpixelSEEDS\n"
            "Use [space] to toggle output mode\n"
            "\n"
            "It captures either from the camera of your choice: 0, 1, ... default 0\n"
            "Or from an input image\n"
            "Call:\n"
            "./seeds [camera #, default 0]\n"
            "./seeds [input image file]\n" << endl;
}

static const char* window_name = "SEEDS Superpixels";

static bool init = false;

void trackbarChanged(int pos, void* data)
{
    init = false;
}


int main(int argc, char** argv)
{
    VideoCapture cap;
    Mat input_image;
    bool use_video_capture = false;
    help();

    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])) )
    {
        cap.open(argc == 2 ? argv[1][0] - '0' : 0);
        use_video_capture = true;
    }
    else if( argc >= 2 )
    {
        input_image = imread(argv[1]);
    }

    if( use_video_capture )
    {
        if( !cap.isOpened() )
        {
            cout << "Could not initialize capturing...\n";
            return -1;
        }
    }
    else if( input_image.empty() )
    {
        cout << "Could not open image...\n";
        return -1;
    }

    namedWindow(window_name, 0);
    int num_iterations = 4;
    int prior = 2;
    bool double_step = false;
    int num_superpixels = 400;
    int num_levels = 4;
    int num_histogram_bins = 5;
    createTrackbar("Number of Superpixels", window_name, &num_superpixels, 1000, trackbarChanged);
    createTrackbar("Smoothing Prior", window_name, &prior, 5, trackbarChanged);
    createTrackbar("Number of Levels", window_name, &num_levels, 10, trackbarChanged);
    createTrackbar("Iterations", window_name, &num_iterations, 12, 0);

    Mat result, mask;
    Ptr<SuperpixelSEEDS> seeds;
    int width, height;
    int display_mode = 0;

    for (;;)
    {
        Mat frame;
        if( use_video_capture )
            cap >> frame;
        else
            input_image.copyTo(frame);

        if( frame.empty() )
            break;

        if( !init )
        {
            width = frame.size().width;
            height = frame.size().height;
            seeds = createSuperpixelSEEDS(width, height, frame.channels(), num_superpixels,
                    num_levels, prior, num_histogram_bins, double_step);
            init = true;
        }
        Mat converted;
        cvtColor(frame, converted, COLOR_BGR2HSV);

        double t = (double) getTickCount();

        seeds->iterate(converted, num_iterations);
        result = frame;

        t = ((double) getTickCount() - t) / getTickFrequency();
        printf("SEEDS segmentation took %i ms with %3i superpixels\n",
                (int) (t * 1000), seeds->getNumberOfSuperpixels());

        /* retrieve the segmentation result */
        Mat labels;
        seeds->getLabels(labels);

        /* get the contours for displaying */
        seeds->getLabelContourMask(mask, false);
        result.setTo(Scalar(0, 0, 255), mask);

        /* display output */
        switch (display_mode)
        {
        case 0: //superpixel contours
            imshow(window_name, result);
            break;
        case 1: //mask
            imshow(window_name, mask);
            break;
        case 2: //labels array
        {
            // use the last x bit to determine the color. Note that this does not
            // guarantee that 2 neighboring superpixels have different colors.
            const int num_label_bits = 2;
            labels &= (1 << num_label_bits) - 1;
            labels *= 1 << (16 - num_label_bits);
            imshow(window_name, labels);
        }
            break;
        }


        int c = waitKey(1);
        if( (c & 255) == 'q' || c == 'Q' || (c & 255) == 27 )
            break;
        else if( (c & 255) == ' ' )
            display_mode = (display_mode + 1) % 3;
    }

    return 0;
}
