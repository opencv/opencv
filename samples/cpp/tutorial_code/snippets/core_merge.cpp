/**
 * @file core_merge.cpp
 * @brief It demonstrates the usage of cv::merge.
 *
 * It shows how to merge 3 single channel matrices into a 3-channel matrix.
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
    Mat m1 = (Mat_<uchar>(2,2) << 1,4,7,10);
    Mat m2 = (Mat_<uchar>(2,2) << 2,5,8,11);
    Mat m3 = (Mat_<uchar>(2,2) << 3,6,9,12);

    Mat channels[3] = {m1, m2, m3};
    Mat m;
    merge(channels, 3, m);
    /*
    m =
    [  1,   2,   3,   4,   5,   6;
       7,   8,   9,  10,  11,  12]
    m.channels() = 3
    */
    //! [example]

    return 0;
}
