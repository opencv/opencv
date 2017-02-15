// Yu's file

#ifndef _OPENCV_IMAGESTORAGE_H_
#define _OPENCV_IMAGESTORAGE_H_

#include <stdio.h>
#include <string>
#include <vector>
#include "highgui.h"
#include "common.h"
class CvCascadeImageReader
{
public:
    bool create(const std::string _posFilename,const std::string _negFilename, cv::Size _winSize);
    void restart() { posReader.restart(); }
    bool getNeg(cv::Mat &_img) { return negReader.get( _img ); }
    bool getPos(cv::Mat &_img) { return posReader.get( _img ); }
private:
    class PosReader
    {
    public:
        PosReader();
        ~PosReader();
        bool create(const std::string _filename);
        bool get(cv::Mat &_img);
        void restart();

        short *vec;
        FILE * file;
        int count;
        int vecSize;
        int last;
        int base;
    } posReader;
    
    class NegReader
    {
    public:
        NegReader();
        bool create(const std::string _filename, cv::Size _winSize);
        bool get(cv::Mat& _img);
        bool get_good_negative( cv::Mat& _img, MBLBPCascadef * pCascade, size_t &testCount);
        bool nextImg();
        bool nextImg2(MBLBPCascadef * pCascade);
        
        cv::Mat src,img,sum;
        std::vector<std::string> imgFilenames;
        cv::Point offset, point;
        float scale;
        float scaleFactor;
        float stepFactor;
        size_t last,round;
        cv::Size winSize;
    } negReader;
};

#endif
