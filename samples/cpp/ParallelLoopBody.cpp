#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// This class use ParallelLoopBody to process image using thread
// ParallelMeanLine process image to have for each line a mean value of 128 and standard deviation of 64
class ParallelMeanLine: public cv::ParallelLoopBody
{
private:
    cv::Mat &imgSrc;
    vector<double> &meanLine;
    vector<double> &stdLine;
    bool verbose;

public:
    ParallelMeanLine(Mat& img, vector<double> &m,vector<double> &s):
        imgSrc(img),
        meanLine(m),
        stdLine(s),
        verbose(false)
    {}
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const
    {

        int h = imgSrc.cols;
        if (verbose)
            cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << endl;
    // First mean and variance for each line are computed
        for (int y = range.start; y < range.end; y++)
        {

            uchar *ptr=imgSrc.ptr(y);
            int m = 0;
            int s2=0;
            for (int x = 0; x<h; x++,ptr++)
            {
                    m += *ptr;
                    s2 += *ptr * *ptr;
            }
            meanLine[y] = m / double(h);
            stdLine[y] = s2/h-meanLine[y]*meanLine[y];
        }
    // We want : all lines must have a mean value of 128 and standard deviation of 64. Negative value or value greater than 255 are threshold
        for (int y = range.start; y < range.end; y++)
        {

            uchar *ptr=imgSrc.ptr(y);
            int v;
            for (int x = 0; x<h; x++,ptr++)
            {
                   v = int((*ptr-meanLine[y])*sqrt(64*64/stdLine[y])+128);
                   if (v>255)
                       *ptr=255;
                   else if (v<0)
                       *ptr=0;
                   else
                       *ptr=(uchar)v;
            }
        }

    }
    ParallelMeanLine& operator=(const ParallelMeanLine &) {
        CV_Assert(false);
        // We can not remove this implementation because Visual Studio warning C4822.
        return *this;
    };
};




const String keys =
"{help h usage ? |      | This program demonstrated the use of class ParallelLoopBody for threading\n ./ParallelLoopBody [image_name -- default ../data/lena.jpg]  --thread <int> default 16}"
"{@image         |../data/lena.jpg      | image to process     }"
    "{@thread        |16    | number o fthread     }"
    "{@verbose       |true   | verbose mode     }"
    ;



int main (int argc,char **argv)
{
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    String fileName = parser.get<String>(0);
    int nbThreads = parser.get<int>(1);
    bool verbose = parser.get<bool>(2);
    cout <<"default value for thread number : "<< getNumThreads() << "\n";
    setNumThreads(nbThreads);
    cout <<"This value is fixed now to  : "<< getNumThreads() << "\n";
    Mat m=imread(fileName,CV_LOAD_IMAGE_GRAYSCALE);
    if (m.empty())
    {
        cout << fileName<<" : Image does not exist!\n";
        return -1;
    }
    cout << "Image size : "<< m.size()<<"\n";
    imshow("Original image",m);
    double  tps;

    vector<double> meanLine,stdLine;
    meanLine.resize(m.rows);
    stdLine.resize(m.rows);

    ParallelMeanLine x(m,meanLine,stdLine);
    x.Verbose(verbose);
    int64 tpsIni = getTickCount();
    parallel_for_(cv::Range(0,m.rows), x,nbThreads);
    int64  tpsFin = getTickCount();
    tps=(tpsFin - tpsIni) / getTickFrequency();
    cout << "For " << nbThreads << " thread times is " << tps << "s\n";
    cout << "*****************************************************************************************\n";
    imshow("Processed Image",m);
    waitKey();
    return 0;
}
