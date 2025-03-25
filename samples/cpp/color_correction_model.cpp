//! [tutorial]
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ccm.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace ccm;

const char *about = "Color correction sample";
const char *keys =
    "{ help h  |                                                     | show this message }"
    "{ input   |   opencv_extra/testdata/cv/mcc/mcc_ccm_test.jpg     | Path of the image file to process }"
    "{ colors  |             samples/data/ccm_test_data.txt               | Path to the txt file containing color values }";

int main(int argc, char *argv[])
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (argc==1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string filepath = parser.get<string>("input");
    string colorFile = parser.get<string>("colors");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    Mat image = imread(filepath, IMREAD_COLOR);
    if (!image.data)
    {
        cout << "Invalid Image!" << endl;
        return 1;
    }

    ifstream infile(colorFile);
    if (!infile.is_open())
    {
        cout << "Failed to open color values file!" << endl;
        return 1;
    }

    Mat src(24, 1, CV_64FC3);
    double r, g, b;
    for (int i = 0; i < 24; i++)
    {
        infile >> r >> g >> b;
        src.at<Vec3d>(i, 0) = Vec3d(r, g, b);
    }
    infile.close();

    ColorCorrectionModel model1(src, COLORCHECKER_Macbeth);
    model1.computeCCM();
    Mat ccm = model1.getCCM();
    cout << "ccm " << ccm << endl;
    double loss = model1.getLoss();
    cout << "loss " << loss << endl;

    // Save CCM matrix and loss using OpenCV FileStorage
    FileStorage fs("ccm_output.yaml", FileStorage::WRITE);
    fs << "ccm" << ccm;
    fs << "loss" << loss;
    fs.release();

    Mat img_;
    cvtColor(image, img_, COLOR_BGR2RGB);
    img_.convertTo(img_, CV_64F);
    img_ = img_ / 255.0;

    Mat calibratedImage = model1.infer(img_);
    Mat out_ = calibratedImage * 255.0;

    out_.convertTo(out_, CV_8UC3);
    Mat img_out = min(max(out_, 0), 255);
    Mat out_img;
    cvtColor(img_out, out_img, COLOR_RGB2BGR);
    string filename = filepath.substr(filepath.find_last_of('/')+1);
    size_t dotIndex = filename.find_last_of('.');
    string baseName = filename.substr(0, dotIndex);
    string ext = filename.substr(dotIndex+1, filename.length()-dotIndex);
    string calibratedFilePath = baseName + ".calibrated." + ext;
    imwrite(calibratedFilePath, out_img);

    return 0;
}
//! [tutorial]
