#ifndef _OPENCV_IMAGESTORAGE_H_
#define _OPENCV_IMAGESTORAGE_H_

#include <highgui.h>

using namespace cv;

class CvCascadeImageReader
{
public:
    bool create( const String _posFilename, const String _negFilename, Size _winSize );
    void restart() { posReader.restart(); }
    bool getNeg(Mat &_img) { return negReader.get( _img ); }
    bool getPos(Mat &_img) { return posReader.get( _img ); }

private:
    class PosReader
    {
    public:
        PosReader();
        virtual ~PosReader();
        bool create( const String _filename );
        bool get( Mat &_img );
        void restart();

        short* vec;
        FILE*  file;
        int    count;
        int    vecSize;
        int    last;
        int    base;
    } posReader;

    class NegReader
    {
    public:
        NegReader();
        bool create( const String _filename, Size _winSize );
        bool get( Mat& _img );
        bool nextImg();

        Mat     src, img;
        vector<String> imgFilenames;
        Point   offset, point;
        float   scale;
        float   scaleFactor;
        float   stepFactor;
        size_t  last, round;
        Size    winSize;
    } negReader;
};

#endif
