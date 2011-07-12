#include "test_precomp.hpp"

//CV_TEST_MAIN("cv")

#if 1
using namespace cv;

int main(int, char**)
{
    Mat prevImg = imread("/Users/vp/work/ocv/opencv_extra/testdata/cv/optflow/rock_1.bmp", 0);
    Mat nextImg = imread("/Users/vp/work/ocv/opencv_extra/testdata/cv/optflow/rock_2.bmp", 0);
    FileStorage fs("/Users/vp/work/ocv/opencv_extra/testdata/cv/optflow/lk_prev.dat", FileStorage::READ);
    vector<Point2f> u, v;
    Mat _u;
    fs["points"] >> _u;
    _u.reshape(2, 0).copyTo(u);
    vector<uchar> status;
    vector<float> err;
    double tmin = 0;
    
    for( int k = 0; k < 3; k++ )
    {
        double t = (double)getTickCount();
#if 1
        calcOpticalFlowPyrLK(prevImg, nextImg, u, v, status, err, Size(11,11),
                             5, TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 30, 0.01), 100);
#else
        {
            CvMat _prevImg = prevImg;
            CvMat _nextImg = nextImg;
            v.resize(u.size());
            status.resize(u.size());
            err.resize(u.size());
            cvCalcOpticalFlowPyrLK(&_prevImg, &_nextImg, 0, 0, (CvPoint2D32f*)&u[0], (CvPoint2D32f*)&v[0], (int)u.size(),
                                   cvSize(21, 21), 4, (char*)&status[0],
                                   &err[0], cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.01), 0);
        }
#endif
        t = (double)getTickCount() - t;
        tmin = k == 0 ? t : std::min(tmin, t);
    }
    printf("time = %gms\n", tmin*1000./getTickFrequency());
        
    Mat color;
    cvtColor(prevImg, color, CV_GRAY2BGR);
    for( size_t i = 0; i < u.size(); i++ )
    {
        Point2f ui = u[i], vi = v[i];
        if( cvIsNaN(v[i].x) || cvIsNaN(v[i].y) || !status[i] )
        {
            const float r = 2.f;
            line(color, Point2f(u[i].x-r,u[i].y-r), Point2f(u[i].x+r,u[i].y+r), Scalar(0, 0, 255), 1);
            line(color, Point2f(u[i].x-r,u[i].y+r), Point2f(u[i].x+r,u[i].y-r), Scalar(0, 0, 255), 1);
            continue;
        }
        line(color, ui, vi, Scalar(0, 255, 0), 1);
    }
    imshow("flow", color);
    waitKey();
    return 0;
}
#endif

