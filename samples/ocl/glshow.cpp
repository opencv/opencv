#include "cvconfig.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ocl/ocl.hpp"
#include <iostream>

using namespace std;
using namespace cv;

static void help()
{
    cout << "Display oclMat using OpenCL-OpenGL interoperation.\n"
         << "\nUsage: glshow\n"
         << "  [<image>|--camera <camera_id>] # frames source\n";
}

void run(bool src_is_image, VideoCapture &cap, string &imsrc)
{
    cout << "press ESC to exit\n";
    bool running = true;
    cvNamedWindow("OpenCL-OpenGL interop", CV_WINDOW_OPENGL);

    ocl::oclMat dimg;
    ocl::oclMat dwarpimg;
    if (src_is_image)
    {
        Mat frame;
        frame = imread(imsrc);
        if (frame.empty())
            throw runtime_error(string("can't open image file: " + imsrc));
        dimg.upload(frame);
    }
    while (running)
    {
        if (!src_is_image)
        {
            Mat frame;
            cap >> frame;
            dimg.upload(frame);
        }

        float w = (float) dimg.cols;
        float h = (float) dimg.rows;
        Point2f src[4];
        Point2f dst[4];
        src[0] = Point2f( 0.f, 0.f );
        src[1] = Point2f( w - 1, 0.f );
        src[2] = Point2f( w - 1, h - 1 );
        src[3] = Point2f( 0.f, h - 1 );
        dst[0] = Point2f( 0.f, 0.f );
        dst[1] = Point2f( w/4*3, h/4 );
        dst[2] = Point2f( w/4*3, h/4*3 );
        dst[3] = Point2f( 0.f, h - 1  );
        Mat pers_mat = getPerspectiveTransform( src, dst );

        warpPerspective(dimg,dwarpimg,pers_mat,dimg.size());

#ifdef HAVE_OPENGL
        ocl::imshow("OpenCL-OpenGL interop",dwarpimg);
#else
        Mat warpimg;
        dwarpimg.download(warpimg);
        imshow("result",warpimg);
#endif
        running = cvWaitKey(3) != 27;
    }
}

int main(int argc, const char **argv)
{
    help();
    vector<ocl::Info> oclinfo;
    int devnums = ocl::getDevice(oclinfo);

    if (devnums < 1)
    {
        cout << "no OpenCL device found\n";
        return -1;
    }

    VideoCapture cap;
    string imsrc;
    bool src_is_image = true;
    if( argc < 2 )
    {
        cap.open(0);
        if (!cap.isOpened())
        {
            stringstream msg;
            msg << "can't open camera: " << 0;
            throw runtime_error(msg.str());
        }
        cout << "\ncamera 0\n";
        src_is_image = false;
    }
    else
    {
        if (string(argv[1]) == "--camera")
        {
            int camera_id = atoi(argv[2]);
            cap.open(camera_id);
            if (!cap.isOpened())
            {
                stringstream msg;
                msg << "can't open camera: " << camera_id;
                throw runtime_error(msg.str());
            }
            cout << "\ncamera " << camera_id << endl;
            src_is_image = false;
        }
        else
            imsrc = argv[1];
    }

    try
    {
        run(src_is_image, cap, imsrc);
    }
    catch (const Exception& e) { return cout << "error: "  << e.what() << endl, 1; }
    catch (const exception& e) { return cout << "error: "  << e.what() << endl, 1; }
    catch(...) { return cout << "unknown exception" << endl, 1; }
    return 0;
}