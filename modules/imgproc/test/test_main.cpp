#include "test_precomp.hpp"

CV_TEST_MAIN("cv")

#if 0
using namespace cv;
using namespace std;

int main(int, char**)
{
#if 0
    Mat src = imread("/Users/vp/Downloads/resize/original.png"), dst;
    //resize(src, dst, Size(), 0.25, 0.25, INTER_NEAREST);
    //imwrite("/Users/vp/Downloads/resize/xnview_nn_opencv.png", dst);
    printf("\n\n\n\n\n\n***************************************\n");
    //resize(src, dst, Size(), 0.25, 0.25, INTER_AREA);
    //int nsteps = 4;
    //double rate = pow(0.25,1./nsteps);
    //for( int i = 0; i < nsteps; i++ )
    //    resize(src, src, Size(), rate, rate, INTER_LINEAR );
    //GaussianBlur(src, src, Size(5, 5), 2, 2);
    resize(src, src, Size(), 0.25, 0.25, INTER_NEAREST);
    imwrite("/Users/vp/Downloads/resize/xnview_bilinear_opencv.png", src);
    //resize(src, dst, Size(), 0.25, 0.25, INTER_LANCZOS4);
    //imwrite("/Users/vp/Downloads/resize/xnview_lanczos3_opencv.png", dst);
#else
    float data[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    Mat src(5,5,CV_32FC1, data);
    Mat dst;
    resize(src, dst, Size(), 3, 3, INTER_NEAREST); 
    cout << src << endl;
    cout << dst << endl;
#endif
    return 0;
}
#endif


