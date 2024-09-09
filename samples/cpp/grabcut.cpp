#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

using namespace std;
using namespace cv;

static void help(char** argv)
{
    cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
            "and then grabcut will attempt to segment it out.\n"
            "Call:\n"
        <<  argv[0] << " <image_name>\n"
            "\nSelect a rectangular area around the object you want to segment\n" <<
            "\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tr - restore the original image\n"
            "\tn - next iteration\n"
            "\n"
            "\tleft mouse button - set rectangle\n"
            "\n"
            "\tCTRL+left mouse button - set GC_BGD pixels\n"
            "\tSHIFT+left mouse button - set GC_FGD pixels\n"
            "\n"
            "\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
            "\tSHIFT+right mouse button - set GC_PR_FGD pixels\n" << endl;
}

const Scalar RED = Scalar(0,0,255);
const Scalar PINK = Scalar(230,130,255);
const Scalar BLUE = Scalar(255,0,0);
const Scalar LIGHTBLUE = Scalar(255,255,160);
const Scalar GREEN = Scalar(0,255,0);

const int BGD_KEY = EVENT_FLAG_CTRLKEY;
const int FGD_KEY = EVENT_FLAG_SHIFTKEY;

static void getBinMask( const Mat& comMask, Mat& binMask )
{
    if( comMask.empty() || comMask.type()!=CV_8UC1 )
        CV_Error( Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
    if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
        binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}

class GCApplication
{
public:
    enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
    static const int radius = 2;
    static const int thickness = -1;

    void reset();
    void setImageAndWinName( const Mat& _image, const string& _winName );
    void saveImage(const string& filename) const;
    void process();
    int nextIter();
    int getIterCount() const { return iterCount; }
private:
    void setRectInMask();

    const string* winName;
    const Mat* image;
    Mat mask;
    Mat bgdModel, fgdModel;

    uchar rectState;
    bool isInitialized;

    Rect rect;
    int iterCount;
};

void GCApplication::reset()
{
    if( !mask.empty() )
        mask.setTo(Scalar::all(GC_BGD));
    isInitialized = false;
    rectState = NOT_SET;
    iterCount = 0;
}

void GCApplication::setImageAndWinName( const Mat& _image, const string& _winName  )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = &_image;
    winName = &_winName;
    mask.create( image->size(), CV_8UC1);
    reset();
}

void GCApplication::saveImage(const string& filename) const
{
    if( image->empty() || winName->empty() )
        return;

    Mat res;
    Mat binMask;
    image->copyTo( res );
    if( isInitialized ){
        getBinMask( mask, binMask);

        Mat black (binMask.rows, binMask.cols, CV_8UC3, cv::Scalar(0,0,0));
        black.setTo(Scalar::all(255), binMask);

        addWeighted(black, 0.5, res, 0.5, 0.0, res);
    }

    if( rectState == IN_PROCESS || rectState == SET )
        rectangle( res, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), GREEN, 2);

    string outputDir = "grabcut";
    system(("mkdir -p " + outputDir).c_str());
    string outputPath = outputDir + "/" + filename;
    imwrite(outputPath, res);
    cout << "Processed image saved at: " << outputPath << endl;
}

void GCApplication::setRectInMask()
{
    CV_Assert( !mask.empty() );
    mask.setTo( GC_BGD );
    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, image->cols-rect.x);
    rect.height = min(rect.height, image->rows-rect.y);
    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

void GCApplication::process()
{
    // Assuming the rectangle is set for now for demo purposes
    rect = Rect(10, 10, image->cols - 20, image->rows - 20);
    rectState = SET;
    setRectInMask();
    nextIter();
    saveImage("result.png");
}

int GCApplication::nextIter()
{
    if( isInitialized )
        grabCut( *image, mask, rect, bgdModel, fgdModel, 1 );
    else
    {
        if( rectState != SET )
            return iterCount;

        grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT );

        isInitialized = true;
    }
    iterCount++;

    return iterCount;
}

GCApplication gcapp;

int main( int argc, char** argv )
{
    cv::CommandLineParser parser(argc, argv, "{@input|../../samples/data/messi5.jpg|}");
    help(argv);

    string filename = parser.get<string>("@input");
    if( filename.empty() )
    {
        cout << "\nDurn, empty filename" << endl;
        return 1;
    }
    Mat image = imread(samples::findFile(filename), IMREAD_COLOR);
    if( image.empty() )
    {
        cout << "\n Durn, couldn't read image filename " << filename << endl;
        return 1;
    }

    const string winName = "image";

    gcapp.setImageAndWinName( image, winName );
    gcapp.process();

    return 0;
}

