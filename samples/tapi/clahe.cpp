#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

Ptr<CLAHE> pFilter;
int tilesize;
int cliplimit;

static void TSize_Callback(int pos, void* /*data*/)
{
    if(pos==0)
        pFilter->setTilesGridSize(Size(1,1));
    else
        pFilter->setTilesGridSize(Size(tilesize,tilesize));
}

static void Clip_Callback(int, void* /*data*/)
{
    pFilter->setClipLimit(cliplimit);
}

int main(int argc, char** argv)
{
    const char* keys =
        "{ i input    |                    | specify input image }"
        "{ c camera   |  0                 | specify camera id   }"
        "{ o output   | clahe_output.jpg   | specify output save path}"
        "{ h help     |                    | print help message }";

    cv::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help"))
    {
        cout << "Usage : clahe [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printMessage();
        return EXIT_SUCCESS;
    }

    string infile = cmd.get<string>("i"), outfile = cmd.get<string>("o");
    int camid = cmd.get<int>("c");
    VideoCapture capture;

    namedWindow("CLAHE");
    createTrackbar("Tile Size", "CLAHE", &tilesize, 32, (TrackbarCallback)TSize_Callback);
    createTrackbar("Clip Limit", "CLAHE", &cliplimit, 20, (TrackbarCallback)Clip_Callback);

    UMat frame, outframe;

    int cur_clip;
    Size cur_tilesize;
    pFilter = createCLAHE();

    cur_clip = (int)pFilter->getClipLimit();
    cur_tilesize = pFilter->getTilesGridSize();
    setTrackbarPos("Tile Size", "CLAHE", cur_tilesize.width);
    setTrackbarPos("Clip Limit", "CLAHE", cur_clip);

    if(infile != "")
    {
        imread(infile).copyTo(frame);
        if(frame.empty())
        {
            cout << "error read image: " << infile << endl;
            return EXIT_FAILURE;
        }
    }
    else
        capture.open(camid);

    cout << "\nControls:\n"
         << "\to - save output image\n"
         << "\tm - switch OpenCL <-> CPU mode"
         << "\tESC - exit\n";

    for (;;)
    {
        if(capture.isOpened())
            capture.read(frame);
        else
            imread(infile).copyTo(frame);
        if(frame.empty())
            continue;

        cvtColor(frame, frame, COLOR_BGR2GRAY);
        pFilter->apply(frame, outframe);

        imshow("CLAHE", outframe);

        char key = (char)waitKey(3);
        if(key == 'o')
            imwrite(outfile, outframe);
        else if(key == 27)
            break;
        else if(key == 'm')
        {
            ocl::setUseOpenCL(!cv::ocl::useOpenCL());
            cout << "Switched to " << (ocl::useOpenCL() ? "OpenCL enabled" : "CPU") << " mode\n";
        }
    }
    return EXIT_SUCCESS;
}
