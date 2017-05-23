#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help(char** argv)
{
    cout << "\nDemonstrate the OpenCL mean-shift based segmentation.\n"
    << "Call:\n   " << argv[0] << " image\n"
    << "This program allows you to set the spatial and color radius\n"
    << "of the mean shift window as well as the minimal size of blocks\n"
    << endl;
}


string winName = "meanshift";
int spatialRad, colorRad, minSize;
Mat img, res;

static void meanShiftSegmentation( int, void* )
{
    cout << "spatialRad=" << spatialRad << "; "
         << "colorRad=" << colorRad << "; "
         << "minSize=" << minSize << endl;
    
    ocl::meanShiftSegmentation(ocl::oclMat(img), res, spatialRad, colorRad, minSize);
    imshow( winName, res );
}

int main(int argc, char** argv)
{
    if( argc !=2 )
    {
        help(argv);
        return -1;
    }

    img = imread( argv[1] );
    if( img.empty() )
        return -1;

    std::vector<ocl::Info> infos;
    ocl::getDevice(infos);

    spatialRad = 10;
    colorRad = 10;
    minSize = 30;

    namedWindow( winName, CV_WINDOW_AUTOSIZE );

    createTrackbar( "spatialRad", winName, &spatialRad, 80, meanShiftSegmentation );
    createTrackbar( "colorRad", winName, &colorRad, 60, meanShiftSegmentation );
    createTrackbar( "minSize", winName, &minSize, 50, meanShiftSegmentation );

    cvtColor(img, img, CV_BGR2BGRA);
    meanShiftSegmentation(0, 0);
    waitKey();
    return 0;
}
