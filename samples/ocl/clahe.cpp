#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"
using namespace cv;
using namespace std;

Ptr<CLAHE> pFilter;
int tilesize;
int cliplimit;
string outfile;

static void TSize_Callback(int pos)
{
    if(pos==0)
    {
        pFilter->setTilesGridSize(Size(1,1));
    }
    pFilter->setTilesGridSize(Size(tilesize,tilesize));
}

static void Clip_Callback(int)
{
    pFilter->setClipLimit(cliplimit);
}

int main(int argc, char** argv)
{
    const char* keys =
        "{ i input   |                    | specify input image }"
        "{ c camera  |    0               | specify camera id   }"
        "{ s use_cpu |    false           | use cpu algorithm   }"
        "{ o output  | clahe_output.jpg   | specify output save path}";

    CommandLineParser cmd(argc, argv, keys);
    string infile = cmd.get<string>("i");
    outfile = cmd.get<string>("o");
    int camid = cmd.get<int>("c");
    bool use_cpu = cmd.get<bool>("s");
    VideoCapture capture;
    bool running = true;

    namedWindow("CLAHE");
    createTrackbar("Tile Size", "CLAHE", &tilesize, 32, (TrackbarCallback)TSize_Callback);
    createTrackbar("Clip Limit", "CLAHE", &cliplimit, 20, (TrackbarCallback)Clip_Callback);

    Mat frame, outframe;
    ocl::oclMat d_outframe;

    int cur_clip;
    Size cur_tilesize;
    if(use_cpu)
    {
        pFilter = createCLAHE();
    }
    else
    {
        pFilter = ocl::createCLAHE();
    }
    cur_clip = (int)pFilter->getClipLimit();
    cur_tilesize = pFilter->getTilesGridSize();
    setTrackbarPos("Tile Size", "CLAHE", cur_tilesize.width);
    setTrackbarPos("Clip Limit", "CLAHE", cur_clip);
    if(infile != "")
    {
        frame = imread(infile);
        if(frame.empty())
        {
            cout << "error read image: " << infile << endl;
            return -1;
        }
    }
    else
    {
        capture.open(camid);
    }
    cout << "\nControls:\n"
         << "\to - save output image\n"
         << "\tESC - exit\n";
    while(running)
    {
        if(capture.isOpened())
            capture.read(frame);
        else
            frame = imread(infile);
        if(frame.empty())
        {
            continue;
        }
        if(use_cpu)
        {
            cvtColor(frame, frame, COLOR_BGR2GRAY);
            pFilter->apply(frame, outframe);
        }
        else
        {
            ocl::oclMat d_frame(frame);
            ocl::cvtColor(d_frame, d_outframe, COLOR_BGR2GRAY);
            pFilter->apply(d_outframe, d_outframe);
            d_outframe.download(outframe);
        }
        imshow("CLAHE", outframe);
        char key = (char)waitKey(3);
        if(key == 'o') imwrite(outfile, outframe);
        else if(key == 27) running = false;
    }
    return 0;
}
