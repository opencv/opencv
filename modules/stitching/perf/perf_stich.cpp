#include "perf_precomp.hpp"

#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
using namespace perf;


/*
// Stitcher::Status Stitcher::stitch(InputArray imgs, OutputArray pano)
*/
PERF_TEST( stitch3, a123 )
{
    Mat pano;
    
    vector<Mat> imgs;
    imgs.push_back( imread( getDataPath("stitching/a1.jpg") ) );
    imgs.push_back( imread( getDataPath("stitching/a2.jpg") ) );
    imgs.push_back( imread( getDataPath("stitching/a3.jpg") ) );

    Stitcher stitcher = Stitcher::createDefault();
    Stitcher::Status status;

    declare.time(30 * 20);

    TEST_CYCLE(20) { status = stitcher.stitch(imgs, pano); }
}

PERF_TEST( stitch2, b12 )
{
    Mat pano;
    
    vector<Mat> imgs;
    imgs.push_back( imread( getDataPath("stitching/b1.jpg") ) );
    imgs.push_back( imread( getDataPath("stitching/b2.jpg") ) );

    Stitcher stitcher = Stitcher::createDefault();
    Stitcher::Status status;

    declare.time(30 * 20);

    TEST_CYCLE(20) { status = stitcher.stitch(imgs, pano); }
}
