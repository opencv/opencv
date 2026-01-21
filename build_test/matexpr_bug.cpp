#include <opencv2/core.hpp>
#include <iostream>
#include "matexpr_bug.h"

int main()
{
    cv::Mat1f a = cv::Mat1f::ones(3, 3);
    NewFunction(a);

    std::cout << "a(0,0) = "
              << static_cast<cv::Mat1f>(a)(0, 0)
              << std::endl;

    return 0;
}

void NewFunction(cv::Mat1f &a)
{
    a += cv::Mat1f::ones(3, 3);
}