/**
 * @file core_reduce.cpp
 * @brief It demonstrates the usage of cv::reduce .
 *
 * It shows how to compute the row sum, column sum, row average,
 * column average, row minimum, column minimum, row maximum
 * and column maximum of a cv::Mat.
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
    {
        //! [example]
        Mat m = (Mat_<uchar>(3,2) << 1,2,3,4,5,6);
        Mat col_sum, row_sum;

        reduce(m, col_sum, 0, REDUCE_SUM, CV_32F);
        reduce(m, row_sum, 1, REDUCE_SUM, CV_32F);
        /*
        m =
        [  1,   2;
           3,   4;
           5,   6]
        col_sum =
        [9, 12]
        row_sum =
        [3;
         7;
         11]
         */
        //! [example]

        Mat col_average, row_average, col_min, col_max, row_min, row_max;
        reduce(m, col_average, 0, REDUCE_AVG, CV_32F);
        cout << "col_average =\n" << col_average << endl;

        reduce(m, row_average, 1, REDUCE_AVG, CV_32F);
        cout << "row_average =\n" << row_average << endl;

        reduce(m, col_min, 0, REDUCE_MIN, CV_8U);
        cout << "col_min =\n" << col_min << endl;

        reduce(m, row_min, 1, REDUCE_MIN, CV_8U);
        cout << "row_min =\n" << row_min << endl;

        reduce(m, col_max, 0, REDUCE_MAX, CV_8U);
        cout << "col_max =\n" << col_max << endl;

        reduce(m, row_max, 1, REDUCE_MAX, CV_8U);
        cout << "row_max =\n" << row_max << endl;

        /*
        col_average =
        [3, 4]
        row_average =
        [1.5;
         3.5;
         5.5]
        col_min =
        [  1,   2]
        row_min =
        [  1;
           3;
           5]
        col_max =
        [  5,   6]
        row_max =
        [  2;
           4;
           6]
        */
    }

    {
        //! [example2]
        // two channels
        char d[] = {1,2,3,4,5,6};
        Mat m(3, 1, CV_8UC2, d);
        Mat col_sum_per_channel;
        reduce(m, col_sum_per_channel, 0, REDUCE_SUM, CV_32F);
        /*
        col_sum_per_channel =
        [9, 12]
        */
        //! [example2]
    }

    return 0;
}
