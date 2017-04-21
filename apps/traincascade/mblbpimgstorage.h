#ifndef _OPENCV_MBLBP_IMAGESTORAGE_H_
#define _OPENCV_MBLBP_IMAGESTORAGE_H_

#include <opencv/highgui.h>
#include "common.h"

using namespace cv;

class PosImageReader{
    short* vec;
    FILE*  file;
    int    count;
    int    vecSize;
    int    last;
    int    base;
 public:
    PosImageReader();
    ~PosImageReader();

    bool create(const string _posFilename);
    bool get(Mat &_img);
    void restart();

};

class NegImageReader{
 public:
    NegImageReader();
    bool create( const string _filename, Size _winSize );
    //bool get( Mat& _img, Mat & _imgsum );
    bool get( Mat& _img);
    bool get_good_negative( Mat& _img, MBLBPCascadef * pCascade, size_t &testCount);
    bool nextImg();
    bool nextImg2(MBLBPCascadef * pCascade);

    
    
    Mat src;
    Mat img;
    Mat sum;

    vector<String> imgFilenames;
    Point   offset, point;
    float   scale;
    float   scaleFactor;
    float   stepFactor;
    size_t  last, round;
    Size    winSize;
};

#endif
