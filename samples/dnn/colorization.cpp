// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// the 313 ab cluster centers from pts_in_hull.npy (already transposed)
static float hull_pts[] = {
    -90., -90., -90., -90., -90., -80., -80., -80., -80., -80., -80., -80., -80., -70., -70., -70., -70., -70., -70., -70., -70.,
    -70., -70., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -50., -50., -50., -50., -50., -50., -50., -50.,
    -50., -50., -50., -50., -50., -50., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -30.,
    -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -20., -20., -20., -20., -20., -20., -20.,
    -20., -20., -20., -20., -20., -20., -20., -20., -20., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10.,
    -10., -10., -10., -10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 10., 10., 10., 10., 10., 10.,
    10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.,
    20., 20., 20., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 40., 40., 40., 40.,
    40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
    50., 50., 50., 50., 50., 50., 50., 50., 50., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60.,
    60., 60., 60., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 80., 80., 80.,
    80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 90., 90., 90., 90., 90., 90., 90., 90., 90., 90.,
    90., 90., 90., 90., 90., 90., 90., 90., 90., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 50., 60., 70., 80., 90.,
    20., 30., 40., 50., 60., 70., 80., 90., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -20., -10., 0., 10., 20., 30., 40., 50.,
    60., 70., 80., 90., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -40., -30., -20., -10., 0., 10., 20.,
    30., 40., 50., 60., 70., 80., 90., 100., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -50.,
    -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -60., -50., -40., -30., -20., -10., 0., 10., 20.,
    30., 40., 50., 60., 70., 80., 90., 100., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90.,
    100., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -80., -70., -60., -50.,
    -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -90., -80., -70., -60., -50., -40., -30., -20., -10.,
    0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30.,
    40., 50., 60., 70., 80., 90., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70.,
    80., -110., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., -110., -100.,
    -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., -110., -100., -90., -80., -70.,
    -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., -110., -100., -90., -80., -70., -60., -50., -40., -30.,
    -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0.
};

int main(int argc, char **argv)
{
    const string about =
        "This sample demonstrates recoloring grayscale images with dnn.\n"
        "This program is based on:\n"
        "  http://richzhang.github.io/colorization\n"
        "  https://github.com/richzhang/colorization\n"
        "Download caffemodel and prototxt files:\n"
        "  http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel\n"
        "  https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt\n";
    const string keys =
        "{ h help |                                    | print this help message }"
        "{ proto  | colorization_deploy_v2.prototxt    | model configuration }"
        "{ model  | colorization_release_v2.caffemodel | model weights }"
        "{ image  | space_shuttle.jpg                  | path to image file }"
        "{ opencl |                                    | enable OpenCL }";
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    string modelTxt = parser.get<string>("proto");
    string modelBin = parser.get<string>("model");
    string imageFile = parser.get<string>("image");
    bool useOpenCL = parser.has("opencl");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    Mat img = imread(imageFile);
    if (img.empty())
    {
        cout << "Can't read image from file: " << imageFile << endl;
        return 2;
    }

    // fixed input size for the pretrained network
    const int W_in = 224;
    const int H_in = 224;
    Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
    if (useOpenCL)
        net.setPreferableTarget(DNN_TARGET_OPENCL);

    // setup additional layers:
    int sz[] = {2, 313, 1, 1};
    const Mat pts_in_hull(4, sz, CV_32F, hull_pts);
    Ptr<dnn::Layer> class8_ab = net.getLayer("class8_ab");
    class8_ab->blobs.push_back(pts_in_hull);
    Ptr<dnn::Layer> conv8_313_rh = net.getLayer("conv8_313_rh");
    conv8_313_rh->blobs.push_back(Mat(1, 313, CV_32F, Scalar(2.606)));

    // extract L channel and subtract mean
    Mat lab, L, input;
    img.convertTo(img, CV_32F, 1.0/255);
    cvtColor(img, lab, COLOR_BGR2Lab);
    extractChannel(lab, L, 0);
    resize(L, input, Size(W_in, H_in));
    input -= 50;

    // run the L channel through the network
    Mat inputBlob = blobFromImage(input);
    net.setInput(inputBlob);
    Mat result = net.forward();

    // retrieve the calculated a,b channels from the network output
    Size siz(result.size[2], result.size[3]);
    Mat a = Mat(siz, CV_32F, result.ptr(0,0));
    Mat b = Mat(siz, CV_32F, result.ptr(0,1));
    resize(a, a, img.size());
    resize(b, b, img.size());

    // merge, and convert back to BGR
    Mat color, chn[] = {L, a, b};
    merge(chn, 3, lab);
    cvtColor(lab, color, COLOR_Lab2BGR);

    imshow("color", color);
    imshow("original", img);
    waitKey();
    return 0;
}
