#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;

typedef tr1::tuple<string, int, int, int, bool, int> Size_Source_t;
typedef TestBaseWithParam<Size_Source_t> Size_Source;

PERF_TEST_P(Size_Source, floodFill1, Combine(
    testing::Values("perf/fruits.jpg", "perf/rubberwhale1.png"), //images
            testing::Values(120, 200 ), //seed point x
            testing::Values(82, 140),//, seed point y
            testing::Values(4,8), //connectivity
            testing::Bool(), //color image, or not
            testing::Values(0, 1, 2) //use fixed(2), gradient mode (1) or simple(0)
            ))
{
    //test given image(s)
    string filename = getDataPath(get<0>(GetParam()));
    Mat image0 = imread(filename, 1);

    Point pseed;
    pseed.x = get<1>(GetParam());
    pseed.y = get<2>(GetParam());

    int connectivity = get<3>(GetParam());
    bool isColored = get<4>(GetParam());
    int isGradient = get<5>(GetParam());

    Mat source;
    if (isColored)
    {
        image0.copyTo(source);
    }
    else
    {
        //convert to grayscale
        cvtColor(image0, source, COLOR_BGR2GRAY);
    }

    int newMaskVal = 255;

    Scalar newval, loVal, upVal;
    int b = 152;//(unsigned)theRNG() & 255;
    int g = 136;//(unsigned)theRNG() & 255;
    int r = 53;//(unsigned)theRNG() & 255;
    if (isGradient)
    {
        loVal = Scalar(4, 4, 4);
        upVal = Scalar(20, 20, 20);
    }
    else
    {
        loVal = Scalar(0, 0, 0);
        upVal = Scalar(0, 0, 0);
    }
    newval = isColored ? Scalar(b, g, r) : Scalar(r*0.299 + g*0.587 + b*0.114);
    Rect prect;
    int flags = connectivity + (newMaskVal << 8);
    if (isGradient == 2)
        flags += (FLOODFILL_FIXED_RANGE);
    int numpix;

    declare.in(source);
    declare.out(source);

    for (;  next(); )
    {
        startTimer();
        numpix = cv::floodFill(source, pseed, newval, &prect, loVal, upVal, flags);
        stopTimer();
        if (isColored)
        {
            image0.copyTo(source);
        }
        else
        {
            //convert to grayscale
            cvtColor(image0, source, COLOR_BGR2GRAY);
        }
    }

    SANITY_CHECK(source);
}
