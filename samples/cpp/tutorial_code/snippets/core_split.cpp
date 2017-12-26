/**
 * @file core_split.cpp
 * @brief It demonstrates the usage of cv::split .
 *
 * It shows how to split a 3-channel matrix into a 3 single channel matrices.
 *
 * @author KUANG Fangjun
 * @date August 2017
 */

#include <iostream>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

int main()
{
    //! [example]
    char d[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    Mat m(2, 2, CV_8UC3, d);
    Mat channels[3];
    split(m, channels);

    /*
    channels[0] =
    [  1,   4;
       7,  10]
    channels[1] =
    [  2,   5;
       8,  11]
    channels[2] =
    [  3,   6;
       9,  12]
    */
    //! [example]

    return 0;
}
