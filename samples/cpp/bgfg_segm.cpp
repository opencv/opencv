// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
    const String keys = "{c camera     | 0 | use video stream from camera (device index starting from 0) }"
                        "{fn file_name |   | use video file as input }"
                        "{m method | mog2 | method: background subtraction algorithm ('knn', 'mog2')}"
                        "{h help | | show help message}";
    CommandLineParser parser(argc, argv, keys);
    parser.about("This sample demonstrates background segmentation.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    int camera = parser.get<int>("camera");
    String file = parser.get<String>("file_name");
    String method = parser.get<String>("method");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    VideoCapture cap;
    if (file.empty())
        cap.open(camera);
    else
    {
        file = samples::findFileOrKeep(file);  // ignore gstreamer pipelines
        cap.open(file.c_str());
    }
    if (!cap.isOpened())
    {
        cout << "Can not open video stream: '" << (file.empty() ? "<camera>" : file) << "'" << endl;
        return 2;
    }

    Ptr<BackgroundSubtractor> model;
    if (method == "knn")
        model = createBackgroundSubtractorKNN();
    else if (method == "mog2")
        model = createBackgroundSubtractorMOG2();
    if (!model)
    {
        cout << "Can not create background model using provided method: '" << method << "'" << endl;
        return 3;
    }

    bool doUpdateModel = true;
    bool doSmoothMask = false;

    Mat inputFrame, frame, foregroundMask, foreground, background;
    String sample_name = "bgfg_segm";
    // 创建子目录
    if (mkdir(sample_name.c_str(), 0777) == -1)
    {
        cerr << "Error :  " << strerror(errno) << endl;
        return 1;
    }

    int frame_counter = 0;
    for (;;)
    {
        // prepare input frame
        cap >> inputFrame;
        if (inputFrame.empty())
        {
            cout << "Finished reading: empty frame" << endl;
            break;
        }
        const Size scaledSize(640, 640 * inputFrame.rows / inputFrame.cols);
        resize(inputFrame, frame, scaledSize, 0, 0, INTER_LINEAR);

        // pass the frame to background model
        model->apply(frame, foregroundMask, doUpdateModel ? -1 : 0);

        // show processed frame
        // imshow("image", frame);

        // show foreground image and mask (with optional smoothing)
        if (doSmoothMask)
        {
            GaussianBlur(foregroundMask, foregroundMask, Size(11, 11), 3.5, 3.5);
            threshold(foregroundMask, foregroundMask, 10, 255, THRESH_BINARY);
        }
        if (foreground.empty())
            foreground.create(scaledSize, frame.type());
        foreground = Scalar::all(0);
        frame.copyTo(foreground, foregroundMask);
        // imshow("foreground mask", foregroundMask);
        // imshow("foreground image", foreground);

        // show background image
        model->getBackgroundImage(background);
        if (!background.empty())
            ; // imshow("mean background image", background);

        // Save the frames and masks
        String frame_filename = sample_name + "/frame_" + to_string(frame_counter) + ".png";
        String foreground_filename = sample_name + "/foreground_" + to_string(frame_counter) + ".png";
        String background_filename = sample_name + "/background_" + to_string(frame_counter) + ".png";

        imwrite(frame_filename, frame);
        imwrite(foreground_filename, foreground);
        if (!background.empty())
            imwrite(background_filename, background);

        cout << "Saved frame to " << frame_filename << endl;
        cout << "Saved foreground to " << foreground_filename << endl;
        if (!background.empty())
            cout << "Saved background to " << background_filename << endl;

        frame_counter++;

        // interact with user
        // const char key = (char)waitKey(30);
        // if (key == 27 || key == 'q') // ESC
        // {
        //     cout << "Exit requested" << endl;
        //     break;
        // }
        // else if (key == ' ')
        // {
        //     doUpdateModel = !doUpdateModel;
        //     cout << "Toggle background update: " << (doUpdateModel ? "ON" : "OFF") << endl;
        // }
        // else if (key == 's')
        // {
        //     doSmoothMask = !doSmoothMask;
        //     cout << "Toggle foreground mask smoothing: " << (doSmoothMask ? "ON" : "OFF") << endl;
        // }
    }
    return 0;
}

