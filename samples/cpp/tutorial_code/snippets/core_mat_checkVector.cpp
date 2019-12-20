/**
 * @brief It demonstrates the usage of cv::Mat::checkVector.
 */

#include <opencv2/core.hpp>

int main()
{
    //! [example-2d]
    cv::Mat mat(20, 1, CV_32FC2);
    int n = mat.checkVector(2);
    CV_Assert(n == 20); // mat has 20 elements

    mat.create(20, 2, CV_32FC1);
    n = mat.checkVector(1);
    CV_Assert(n == -1); // mat is neither a column nor a row vector

    n = mat.checkVector(2);
    CV_Assert(n == 20); // the 2 columns are considered as 1 element
    //! [example-2d]

    mat.create(1, 5, CV_32FC1);
    n = mat.checkVector(1);
    CV_Assert(n == 5); // mat has 5 elements

    n = mat.checkVector(5);
    CV_Assert(n == 1); // the 5 columns are considered as 1 element

    //! [example-3d]
    int dims[] = {1, 3, 5}; // 1 plane, every plane has 3 rows and 5 columns
    mat.create(3, dims, CV_32FC1); // for 3-d mat, it MUST have only 1 channel
    n = mat.checkVector(5); // the 5 columns are considered as 1 element
    CV_Assert(n == 3);

    int dims2[] = {3, 1, 5}; // 3 planes, every plane has 1 row and 5 columns
    mat.create(3, dims2, CV_32FC1);
    n = mat.checkVector(5); // the 5 columns are considered as 1 element
    CV_Assert(n == 3);
    //! [example-3d]

    return 0;
}
