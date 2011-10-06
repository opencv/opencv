#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <cstdio>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    if( argc != 2 )
    {
        cout << "Colors count should be passed." << endl;
        return -1;
    }

    int colorsCount = atoi(argv[1]);
    vector<Scalar> colors;
    theRNG() = (uint64)time(0);
    generateColors( colors, colorsCount );

    int stripWidth = 20;
    Mat strips(300, colorsCount*stripWidth, CV_8UC3);
    for( int i = 0; i < colorsCount; i++ )
    {
        strips.colRange(i*stripWidth, (i+1)*stripWidth) = colors[i];
    }

    imshow( "strips", strips );
    waitKey();

    return 0;
}
