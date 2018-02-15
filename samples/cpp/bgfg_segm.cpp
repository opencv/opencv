#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
    const String keys = "{c camera||use video stream from camera (default is NO)}"
                        "{fn file_name|../data/tree.avi|video file}"
                        "{m method|mog2|method: background subtraction algorithm ('knn', 'mog2')}"
                        "{h help||show help message}";
    CommandLineParser parser(argc, argv, keys);
    parser.about("This sample demonstrates background segmentation.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    bool useCamera = parser.has("camera");
    String file = parser.get<String>("file_name");
    String method = parser.get<String>("method");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    VideoCapture cap;
    if (useCamera)
        cap.open(0);
    else
        cap.open(file.c_str());
    if (!cap.isOpened())
    {
        cout << "Can not open video stream: '" << (useCamera ? "<camera 0>" : file) << "'" << endl;
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

    cout << "Press <space> to toggle background model update" << endl;
    cout << "Press 's' to toggle foreground mask smoothing" << endl;
    cout << "Press ESC or 'q' to exit" << endl;
    bool doUpdateModel = true;
    bool doSmoothMask = false;

    Mat inputFrame, frame, foregroundMask, foreground, background;
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
        imshow("image", frame);

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
        imshow("foreground mask", foregroundMask);
        imshow("foreground image", foreground);

        // show background image
        model->getBackgroundImage(background);
        if (!background.empty())
            imshow("mean background image", background );

        // interact with user
        const char key = (char)waitKey(30);
        if (key == 27 || key == 'q') // ESC
        {
            cout << "Exit requested" << endl;
            break;
        }
        else if (key == ' ')
        {
            doUpdateModel = !doUpdateModel;
            cout << "Toggle background update: " << (doUpdateModel ? "ON" : "OFF") << endl;
        }
        else if (key == 's')
        {
            doSmoothMask = !doSmoothMask;
            cout << "Toggle foreground mask smoothing: " << (doSmoothMask ? "ON" : "OFF") << endl;
        }
    }
    return 0;
}
