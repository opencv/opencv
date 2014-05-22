#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;

typedef tr1::tuple<string, int, int, int, int, int, int> Size_Source_t;
typedef TestBaseWithParam<Size_Source_t> Size_Source;

PERF_TEST_P(Size_Source, floodFill1, Combine(
    testing::Values("perf/fruits.jpg", "perf/rubberwhale1.png"), //images
            testing::Values(120, 200 ), //seed point x
            testing::Values(82, 140),//, seed point y
            testing::Values(4,8), //connectivity
            testing::Values(IMREAD_COLOR, IMREAD_GRAYSCALE), //color image, or not
            testing::Values(0, 1, 2), //use fixed(1), gradient (2) or simple(0) mode
            testing::Values(CV_8U, CV_32F, CV_32S) //image depth
            ))
{
    //test given image(s)
    string filename = getDataPath(get<0>(GetParam()));
    Point pseed;
    pseed.x = get<1>(GetParam());
    pseed.y = get<2>(GetParam());

    int connectivity = get<3>(GetParam());
    int colorType = get<4>(GetParam());
    int modeType = get<5>(GetParam());
    int imdepth = get<6>(GetParam());

    Mat image0 = imread(filename, colorType);

    Mat source;
    if (imdepth == CV_8U)
    {
        image0.copyTo(source);
    }
    else if (imdepth == CV_32F)
    {
        image0.convertTo(source, CV_32F);
    }
    else if (imdepth == CV_32S)
    {
        image0.convertTo(source, CV_32S);
    }

    int newMaskVal = 255;

    Scalar newval, loVal, upVal;
    int b = 152;//(unsigned)theRNG() & 255;
    int g = 136;//(unsigned)theRNG() & 255;
    int r = 53;//(unsigned)theRNG() & 255;
    if (modeType == 0)
    {
        loVal = Scalar(0, 0, 0);
        upVal = Scalar(0, 0, 0);
    }
    else
    {
        loVal = Scalar(4, 4, 4);
        upVal = Scalar(20, 20, 20);
    }
    newval = (colorType == IMREAD_COLOR) ? Scalar(b, g, r) : Scalar(r*0.299 + g*0.587 + b*0.114);
    Rect prect;
    int flags = connectivity + (newMaskVal << 8) + (modeType == 1 ? FLOODFILL_FIXED_RANGE : 0);
    int numpix;

    declare.in(source);
    declare.out(source);

    for (;  next(); )
    {
        startTimer();
        numpix = cv::floodFill(source, pseed, newval, &prect, loVal, upVal, flags);
        stopTimer();
        if (imdepth == CV_8U)
        {
            image0.copyTo(source);
        }
        else if (imdepth == CV_32F)
        {
            image0.convertTo(source, CV_32F);
        }
        else if (imdepth == CV_32S)
        {
            image0.convertTo(source, CV_32S);
        }
    }

    SANITY_CHECK(source);
}
