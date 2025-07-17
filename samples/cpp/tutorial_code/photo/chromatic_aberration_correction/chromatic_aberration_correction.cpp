#include "opencv2/core.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

static const char* usage =
    "Chromatic Aberration Correction Sample\n"
    "Usage:\n"
    "  ca_correction <input_image> <calibration_file> [bayer_pattern] [output_image]\n"
    "\n"
    "Arguments:\n"
    "  input_image       Path to the input image. Can be:\n"
    "                      • a 3-channel BGR image, or\n"
    "                      • a 1-channel raw Bayer image (see bayer_pattern)\n"
    "  calibration_file  OpenCV YAML/XML file with chromatic aberration calibration:\n"
    "                      image_width, image_height, red_channel/coeffs_x, coeffs_y,\n"
    "                      blue_channel/coeffs_x, coeffs_y.\n"
    "  output_image      (optional) Path to save the corrected image. Default: corrected.png\n"
    "  bayer_pattern     (optional) integer code for demosaicing a 1-channel raw image:\n"
    "                      cv::COLOR_BayerBG2BGR = 46\n"
    "                      cv::COLOR_BayerGB2BGR = 47\n"
    "                      cv::COLOR_BayerGR2BGR = 48\n"
    "                      cv::COLOR_BayerRG2BGR = 49\n"
    "                    If omitted or <0, input is assumed 3-channel BGR.\n"
    "\n"
    "Example:\n"
    "  ca_correction input.png calib.yaml 46 corrected.png\n"
    "\n";

int main(int argc, char** argv)
{
    const string keys =
        "{help h       |      | show this help message }"
        "{@input       |      | input image (BGR or Bayer)}"
        "{@calibration |      | calibration file (YAML/XML) }"
        "{output       |corrected.png| output image file }"
        "{bayer        |-1    | Bayer pattern code for demosaic }"
        ;

    CommandLineParser parser(argc, argv, keys);
    parser.about("Chromatic Aberration Correction Sample");
    if (parser.has("help") || argc < 3)
    {
        parser.printMessage();
        return 0;
    }

    string inputPath      = parser.get<string>("@input");
    string calibPath      = parser.get<string>("@calibration");
    string outputPath     = parser.get<string>("output");
    int bayerPattern      = parser.get<int>("bayer");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    // Load the input image
    Mat input = imread(inputPath, IMREAD_UNCHANGED);
    if (input.empty())
    {
        cerr << "ERROR: Could not load input image: " << inputPath << endl;
        return 1;
    }

    try
    {
        // Apply chromatic aberration correction
        Mat corrected = correctChromaticAberration(input, calibPath, bayerPattern);

        // Show results
        namedWindow("Original",    WINDOW_AUTOSIZE);
        namedWindow("Corrected",   WINDOW_AUTOSIZE);
        imshow("Original",  input);
        imshow("Corrected", corrected);
        cout << "Press any key to continue..." << endl;
        waitKey();

        // Save corrected image
        if (!imwrite(outputPath, corrected))
        {
            cerr << "WARNING: Could not write output image: " << outputPath << endl;
        }
        else
        {
            cout << "Saved corrected image to: " << outputPath << endl;
        }
    }
    catch (const Exception& e)
    {
        cerr << "OpenCV error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
