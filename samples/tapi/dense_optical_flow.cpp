// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <iostream>
#include <iomanip>
#include <vector>

#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"

using namespace std;
using namespace cv;

static Mat getVisibleFlow(InputArray flow)
{
    vector<UMat> flow_vec;
    split(flow, flow_vec);
    UMat magnitude, angle;
    cartToPolar(flow_vec[0], flow_vec[1], magnitude, angle, true);
    magnitude.convertTo(magnitude, CV_32F, 0.2);
    vector<UMat> hsv_vec;
    hsv_vec.push_back(angle);
    hsv_vec.push_back(UMat::ones(angle.size(), angle.type()));
    hsv_vec.push_back(magnitude);
    UMat hsv;
    merge(hsv_vec, hsv);
    Mat img;
    cvtColor(hsv, img, COLOR_HSV2BGR);
    return img;
}

static Size fitSize(const Size & sz,  const Size & bounds)
{
    CV_Assert(!sz.empty());
    if (sz.width > bounds.width || sz.height > bounds.height)
    {
        double scale = std::min((double)bounds.width / sz.width, (double)bounds.height / sz.height);
        return Size(cvRound(sz.width * scale), cvRound(sz.height * scale));
    }
    return sz;
}

int main(int argc, const char* argv[])
{
    const char* keys =
            "{ h help     |     | print help message }"
            "{ c camera   | 0   | capture video from camera (device index starting from 0) }"
            "{ a algorithm | fb | algorithm (supported: 'fb', 'dis')}"
            "{ m cpu      |     | run without OpenCL }"
            "{ v video    |     | use video as input }"
            "{ o original |     | use original frame size (do not resize to 640x480)}"
            ;
    CommandLineParser parser(argc, argv, keys);
    parser.about("This sample demonstrates using of dense optical flow algorithms.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    int camera = parser.get<int>("camera");
    string algorithm = parser.get<string>("algorithm");
    bool useCPU = parser.has("cpu");
    string filename = parser.get<string>("video");
    bool useOriginalSize = parser.has("original");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    VideoCapture cap;
    if(filename.empty())
        cap.open(camera);
    else
        cap.open(filename);
    if (!cap.isOpened())
    {
        cout << "Can not open video stream: '" << (filename.empty() ? "<camera>" : filename) << "'" << endl;
        return 2;
    }

    Ptr<DenseOpticalFlow> alg;
    if (algorithm == "fb")
        alg = FarnebackOpticalFlow::create();
    else if (algorithm == "dis")
        alg = DISOpticalFlow::create(DISOpticalFlow::PRESET_FAST);
    else
    {
        cout << "Invalid algorithm: " << algorithm << endl;
        return 3;
    }

    ocl::setUseOpenCL(!useCPU);

    cout << "Press 'm' to toggle CPU/GPU processing mode" << endl;
    cout << "Press ESC or 'q' to exit" << endl;

    UMat prevFrame, frame, input_frame, flow;
    for(;;)
    {
        if (!cap.read(input_frame) || input_frame.empty())
        {
            cout << "Finished reading: empty frame" << endl;
            break;
        }
        Size small_size = fitSize(input_frame.size(), Size(640, 480));
        if (!useOriginalSize && small_size != input_frame.size())
            resize(input_frame, frame, small_size);
        else
            frame = input_frame;
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        imshow("frame", frame);
        if (!prevFrame.empty())
        {
            int64 t = getTickCount();
            alg->calc(prevFrame, frame, flow);
            t = getTickCount() - t;
            {
                Mat img = getVisibleFlow(flow);
                ostringstream buf;
                buf << "Algo: " << algorithm << " | "
                    << "Mode: " << (useCPU ? "CPU" : "GPU") << " | "
                    << "FPS: " << fixed << setprecision(1) << (getTickFrequency() / (double)t);
                putText(img, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);
                imshow("Dense optical flow field", img);
            }
        }
        frame.copyTo(prevFrame);

        // interact with user
        const char key = (char)waitKey(30);
        if (key == 27 || key == 'q') // ESC
        {
            cout << "Exit requested" << endl;
            break;
        }
        else if (key == 'm')
        {
            useCPU = !useCPU;
            ocl::setUseOpenCL(!useCPU);
            cout << "Set processing mode to: " << (useCPU ? "CPU" : "GPU") << endl;
        }
    }

    return 0;
}
