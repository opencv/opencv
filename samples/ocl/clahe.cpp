#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"
using namespace cv;
using namespace std;

Ptr<CLAHE> pFilter;
int tilesize;
int cliplimit;

static void TSize_Callback(int pos)
{
    if(pos==0)
        pFilter->setTilesGridSize(Size(1,1));
    else
        pFilter->setTilesGridSize(Size(tilesize,tilesize));
}

static void Clip_Callback(int)
{
    pFilter->setClipLimit(cliplimit);
}

int main(int argc, char** argv)
{
    const char* keys =
        "{ i | input   |                    | specify input image }"
        "{ c | camera  |    0               | specify camera id   }"
        "{ s | use_cpu |    false           | use cpu algorithm   }"
        "{ o | output  | clahe_output.jpg   | specify output save path}"
        "{ h | help    | false              | print help message }";

    cv::CommandLineParser cmd(argc, argv, keys);
    if (cmd.get<bool>("help"))
    {
        cout << "Usage : clahe [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printParams();
        return EXIT_SUCCESS;
    }

    string infile = cmd.get<string>("i"), outfile = cmd.get<string>("o");
    int camid = cmd.get<int>("c");
    bool use_cpu = cmd.get<bool>("s");
    CvCapture* capture = 0;

    namedWindow("CLAHE");
    createTrackbar("Tile Size", "CLAHE", &tilesize, 32, (TrackbarCallback)TSize_Callback);
    createTrackbar("Clip Limit", "CLAHE", &cliplimit, 20, (TrackbarCallback)Clip_Callback);

    Mat frame, outframe;
    ocl::oclMat d_outframe, d_frame;

    int cur_clip;
    Size cur_tilesize;
    pFilter = use_cpu ? createCLAHE() : ocl::createCLAHE();

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
            return EXIT_FAILURE;
        }
    }
    else
        capture = cvCaptureFromCAM(camid);

    cout << "\nControls:\n"
         << "\to - save output image\n"
         << "\tESC - exit\n";

    for (;;)
    {
        if(capture)
            frame = cvQueryFrame(capture);
        else
            frame = imread(infile);
        if(frame.empty())
            continue;

        if(use_cpu)
        {
            cvtColor(frame, frame, COLOR_BGR2GRAY);
            pFilter->apply(frame, outframe);
        }
        else
        {
            ocl::cvtColor(d_frame = frame, d_outframe, COLOR_BGR2GRAY);
            pFilter->apply(d_outframe, d_outframe);
            d_outframe.download(outframe);
        }

        imshow("CLAHE", outframe);

        char key = (char)cvWaitKey(3);
        if(key == 'o')
            imwrite(outfile, outframe);
        else if(key == 27)
            break;
    }
    return EXIT_SUCCESS;
}
