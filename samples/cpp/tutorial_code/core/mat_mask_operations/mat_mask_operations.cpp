#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

static void help(char* progName)
{
    cout << endl
        <<  "This program shows how to filter images with mask: the use of filter2D function" << endl
        <<  "for a hand-written approximation of a 3x3 sharpening filter." << endl
        <<  "Usage:"                                                               << endl
        <<  progName << " [image_path -- default starry_night.jpg] [G -- grayscale]" << endl << endl;
}

void Sharpen(const Mat& myImage, Mat& Result);

int main( int argc, char* argv[])
{
    help(argv[0]);

    // Modernization: Switched default from 'lena.jpg' to 'starry_night.jpg' [#25635]
    string filename = "starry_night.jpg";
    if (argc >= 2)
    {
        filename = argv[1];
    }

    // Robustness: Use false for 'required' to prevent C++ exception crashes
    string image_path = samples::findFile(filename, false);

    if (image_path.empty())
    {
        cout << "Can't find sample image: [" << filename << "]" << endl;
        return -1;
    }

    Mat src, dst0, dst1;

    if (argc >= 3 && !strcmp("G", argv[2]))
        src = imread( image_path, IMREAD_GRAYSCALE);
    else
        src = imread( image_path, IMREAD_COLOR);

    if (src.empty())
    {
        cerr << "Can't open image ["  << image_path << "]" << endl;
        return -1;
    }

    namedWindow("Input", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);

    imshow( "Input", src );
    double t = (double)getTickCount();

    Sharpen(src, dst0);

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Hand written function time passed in seconds: " << t << endl;

    imshow( "Output", dst0 );
    waitKey();

    // [filter2D]
    Mat kernel = (Mat_<char>(3,3) <<  0, -1,  0,
                                   -1,  5, -1,
                                    0, -1,  0);
    t = (double)getTickCount();

    filter2D( src, dst1, src.depth(), kernel );

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Built-in filter2D time passed in seconds:     " << t << endl;

    imshow( "Output", dst1 );

    waitKey();
    return 0;
}

void Sharpen(const Mat& myImage, Mat& Result)
{
    // [sharpen_hand_written]
    CV_Assert(myImage.depth() == CV_8U);  // accept only uchar images

    const int nChannels = myImage.channels();
    Result.create(myImage.size(), myImage.type());

    for(int j = 1 ; j < myImage.rows-1; ++j)
    {
        const uchar* previous = myImage.ptr<uchar>(j - 1);
        const uchar* current  = myImage.ptr<uchar>(j    );
        const uchar* next     = myImage.ptr<uchar>(j + 1);

        uchar* output = Result.ptr<uchar>(j);

        for(int i= nChannels; i < nChannels*(myImage.cols-1); ++i)
        {
            *output++ = saturate_cast<uchar>(5*current[i]
                         -current[i-nChannels] - current[i+nChannels] - previous[i] - next[i]);
        }
    }
    // [sharpen_hand_written]

    Result.row(0).setTo(Scalar(0));
    Result.row(Result.rows-1).setTo(Scalar(0));
    Result.col(0).setTo(Scalar(0));
    Result.col(Result.cols-1).setTo(Scalar(0));
}
