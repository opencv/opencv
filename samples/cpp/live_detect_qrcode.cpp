#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

void getMatWithQRCodeContour(Mat &color_image, vector<Point> transform);
void getMatWithFPS(Mat &color_image, double fps);
int  liveQRCodeDetect();
int  showImageQRCodeDetect(string in, string out);

int main(int argc, char *argv[])
{
    const string keys =
        "{h help ? |        | print help messages }"
        "{i in     |        | input  path to file for detect (with parameter - show image, otherwise - camera)}"
        "{o out    |        | output path to file (save image, work with -i parameter) }";
    CommandLineParser cmd_parser(argc, argv, keys);

    cmd_parser.about("This program detects the QR-codes from camera or images using the OpenCV library.");
    if (cmd_parser.has("help"))
    {
        cmd_parser.printMessage();
        return 0;
    }

    string in_file_name  = cmd_parser.get<string>("in");    // input  path to image
    string out_file_name = cmd_parser.get<string>("out");   // output path to image

    if (!cmd_parser.check())
    {
        cmd_parser.printErrors();
        return -1;
    }

    int return_code = 0;
    if (in_file_name.empty())
    {
        return_code = liveQRCodeDetect();
    }
    else
    {
        return_code = showImageQRCodeDetect(in_file_name, out_file_name);
    }
    return return_code;
}

void getMatWithQRCodeContour(Mat &color_image, vector<Point> transform)
{
    if (!transform.empty())
    {
        double show_radius = (color_image.rows  > color_image.cols)
                   ? (2.813 * color_image.rows) / color_image.cols
                   : (2.813 * color_image.cols) / color_image.rows;
        double contour_radius = show_radius * 0.4;

        vector< vector<Point> > contours;
        contours.push_back(transform);
        drawContours(color_image, contours, 0, Scalar(211, 0, 148), cvRound(contour_radius));

        RNG rng(1000);
        for (size_t i = 0; i < 4; i++)
        {
            Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
            circle(color_image, transform[i], cvRound(show_radius), color, -1);
        }
    }
}

void getMatWithFPS(Mat &color_image, double fps)
{
    ostringstream convert;
    convert << cvRound(fps) << " FPS.";
    putText(color_image, convert.str(), Point(25, 25), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 2);
}

int liveQRCodeDetect()
{
    VideoCapture cap(0);
    if(!cap.isOpened())
    {
        cout << "Cannot open a camera" << '\n';
        return -4;
    }

    QRCodeDetector qrcode;
    TickMeter total;
    for(;;)
    {
        Mat frame, src, straight_barcode;
        string decode_info;
        vector<Point> transform;
        cap >> frame;
        if(frame.empty()) { break; }
        cvtColor(frame, src, COLOR_BGR2GRAY);

        total.start();
        bool result_detection = qrcode.detect(src, transform);
        if (result_detection)
        {
            decode_info = qrcode.decode(src, transform, straight_barcode);
            if (!decode_info.empty()) { cout << decode_info << '\n'; }
        }
        total.stop();
        double fps = 1 / total.getTimeSec();
        total.reset();

        if (result_detection) { getMatWithQRCodeContour(frame, transform); }
        getMatWithFPS(frame, fps);

        imshow("Live QR code detector", frame);
        if( waitKey(30) > 0 ) { break; }
    }
    return 0;
}

int showImageQRCodeDetect(string in, string out)
{
    Mat src = imread(in, IMREAD_GRAYSCALE), straight_barcode;
    string decoded_info;
    vector<Point> transform;
    const int count_experiments = 10;
    double transform_time = 0.0;
    bool result_detection = false;
    TickMeter total;
    QRCodeDetector qrcode;
    for (size_t i = 0; i < count_experiments; i++)
    {
        total.start();
        transform.clear();
        result_detection = qrcode.detect(src, transform);
        total.stop();
        transform_time += total.getTimeSec();
        total.reset();
        if (!result_detection) { break; }

        total.start();
        decoded_info = qrcode.decode(src, transform, straight_barcode);
        total.stop();
        transform_time += total.getTimeSec();
        total.reset();
        if (decoded_info.empty()) { break; }

    }
    double fps = count_experiments / transform_time;
    if (!result_detection) { cout << "QR code not found\n"; return -2; }
    if (decoded_info.empty()) { cout << "QR code cannot be decoded\n"; return -3; }

    Mat color_src = imread(in);
    getMatWithQRCodeContour(color_src, transform);
    getMatWithFPS(color_src, fps);

    for(;;)
    {
        imshow("Detect QR code on image", color_src);
        if( waitKey(30) > 0 ) { break; }
    }

    if (!out.empty())
    {
        getMatWithQRCodeContour(color_src, transform);
        getMatWithFPS(color_src, fps);

        cout << "Input  image file path: " << in  << '\n';
        cout << "Output image file path: " << out << '\n';
        cout << "Size: " << color_src.size() << '\n';
        cout << "FPS: " << fps << '\n';
        cout << "Decoded info: " << decoded_info << '\n';

        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);
        try
        {
            imwrite(out, color_src, compression_params);
        }
        catch (const cv::Exception& ex)
        {
            cout << "Exception converting image to PNG format: ";
            cout << ex.what() << '\n';
            return -3;
        }
    }
    return 0;
}
