#include <iostream>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

static const Scalar greenColor(0, 255, 0);
static const Scalar redColor(0, 0, 255);
static const Scalar yellowColor(0, 255, 255);
static Scalar randColor()
{
    RNG &rng = theRNG();
    return Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
}

//===============================================================================

struct TheApp
{
    Ptr<barcode::BarcodeDetector> bardet;
    //! [output]
    vector<Point> corners;
    vector<string> decode_info;
    vector<string> decode_type;
    //! [output]
    bool detectOnly;

    void cleanup()
    {
        corners.clear();
        decode_info.clear();
        decode_type.clear();
    }

    inline string modeString() const
    {
        return detectOnly ? "<detect>" : "<detectAndDecode>";
    }

    void drawResults(Mat &frame) const
{
    //! [visualize]
    for (size_t i = 0; i < corners.size(); i += 4)
    {
        const size_t idx = i / 4;
        const bool isDecodable = idx < decode_info.size()
            && idx < decode_type.size()
            && !decode_type[idx].empty();
        const Scalar lineColor = isDecodable ? greenColor : redColor;
        // draw barcode rectangle
        vector<Point> contour(corners.begin() + i, corners.begin() + i + 4);
        const vector< vector<Point> > contours {contour};
        // drawContours(frame, contours, 0, lineColor, 1); // 注释掉
        // draw vertices
        for (size_t j = 0; j < 4; j++)
            ;// circle(frame, contour[j], 2, randColor(), -1); // 注释掉
        // write decoded text
        if (isDecodable)
        {
            ostringstream buf;
            buf << "[" << decode_type[idx] << "] " << decode_info[idx];
            ;// putText(frame, buf.str(), contour[1], FONT_HERSHEY_COMPLEX, 0.8, yellowColor, 1); // 注释掉
        }
    }
    //! [visualize]
}


    void drawFPS(Mat &frame, double fps) const
{
    ostringstream buf;
    buf << modeString()
        << " (" << corners.size() / 4 << "/" << decode_type.size() << "/" << decode_info.size() << ") "
        << cv::format("%.2f", fps) << " FPS ";
    ;// putText(frame, buf.str(), Point(25, 25), FONT_HERSHEY_COMPLEX, 0.8, redColor, 2); // 注释掉
}

    inline void call_decode(Mat &frame)
    {
        cleanup();
        if (detectOnly)
        {
            //! [detect]
            bardet->detectMulti(frame, corners);
            //! [detect]
        }
        else
        {
            //! [detectAndDecode]
            bardet->detectAndDecodeWithType(frame, decode_info, decode_type, corners);
            //! [detectAndDecode]
        }
    }

    int liveBarCodeDetect()
{
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cout << "Cannot open a camera" << endl;
        return 2;
    }
    Mat frame;
    Mat result;
    cap >> frame;
    cout << "Image size: " << frame.size() << endl;
    cout << "Press 'd' to switch between <detect> and <detectAndDecode> modes" << endl;
    cout << "Press 'ESC' to exit" << endl;
    for (;;)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "End of video stream" << endl;
            break;
        }
        if (frame.channels() == 1)
            cvtColor(frame, frame, COLOR_GRAY2BGR);
        TickMeter timer;
        timer.start();
        call_decode(frame);
        timer.stop();
        drawResults(frame);
        drawFPS(frame, timer.getFPS());
        // imshow("barcode", frame); // 注释掉
        const char c = (char)waitKey(1);
        if (c == 'd')
        {
            detectOnly = !detectOnly;
            cout << "Mode switched to " << modeString() << endl;
        }
        else if (c == 27)
        {
            cout << "'ESC' is pressed. Exiting..." << endl;
            break;
        }
    }
    return 0;
}

    int imageBarCodeDetect(const string &in_file, const string &out_file)
{
    Mat frame = imread(in_file, IMREAD_COLOR);
    cout << "Image size: " << frame.size() << endl;
    cout << "Mode is " << modeString() << endl;
    const int count_experiments = 100;
    TickMeter timer;
    for (size_t i = 0; i < count_experiments; i++)
    {
        timer.start();
        call_decode(frame);
        timer.stop();
    }
    cout << "FPS: " << timer.getFPS() << endl;
    drawResults(frame);
    if (!out_file.empty())
    {
        cout << "Saving result: " << out_file << endl;
        imwrite(out_file, frame);
    }
    // imshow("barcode", frame); // 注释掉
    cout << "Press any key to exit ..." << endl;
    ;// waitKey(0); // 注释掉
    return 0;
    }

};


//==============================================================================

int main(int argc, char **argv)
{
    const string keys = "{h help ? |        | print help messages }"
                        "{i in     |        | input image path (also switches to image detection mode) }"
                        "{detect   | false  | detect 1D barcode only (skip decoding) }"
                        "{o out    |        | path to result file (only for single image decode) }"
                        "{sr_prototxt|      | super resolution prototxt path }"
                        "{sr_model |        | super resolution model path }";
    CommandLineParser cmd_parser(argc, argv, keys);
    cmd_parser.about("This program detects the 1D barcodes from camera or images using the OpenCV library.");
    if (cmd_parser.has("help"))
    {
        cmd_parser.printMessage();
        return 0;
    }
    const string in_file = cmd_parser.get<string>("in");
    const string out_file = cmd_parser.get<string>("out");
    const string sr_prototxt = cmd_parser.get<string>("sr_prototxt");
    const string sr_model = cmd_parser.get<string>("sr_model");
    if (!cmd_parser.check())
    {
        cmd_parser.printErrors();
        return -1;
    }

    TheApp app;
    app.detectOnly = cmd_parser.has("detect") && cmd_parser.get<bool>("detect");
    //! [initialize]
    try
    {
        app.bardet = makePtr<barcode::BarcodeDetector>(sr_prototxt, sr_model);
    }
    catch (const std::exception& e)
    {
        cout <<
             "\n---------------------------------------------------------------\n"
             "Failed to initialize super resolution.\n"
             "Please, download 'sr.*' from\n"
             "https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode\n"
             "and put them into the current directory.\n"
             "Or you can leave sr_prototxt and sr_model unspecified.\n"
             "---------------------------------------------------------------\n";
        cout << e.what() << endl;
        return -1;
    }
    //! [initialize]

    if (in_file.empty())
        return app.liveBarCodeDetect();
    else
        return app.imageBarCodeDetect(in_file, out_file);
}

