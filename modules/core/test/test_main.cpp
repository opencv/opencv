#include "test_precomp.hpp"

//CV_TEST_MAIN("cv")

#include <iostream>
#include <cxcore.hpp>

using namespace cv;
using namespace std;

int main(int,char**)
{
    /*cv::Mat img3x3(3, 3, CV_8UC1);
    cv::Mat img2x2 = img3x3(cv::Rect(0, 0, 2, 2));
    
    for(cv::MatIterator_<uchar> it = img3x3.begin<uchar>(); it != img3x3.end<uchar>(); ++it)
        *it = 1;
    
    int sum = 0;
    for(cv::MatConstIterator_<uchar> it = img2x2.begin<uchar>(); it != img2x2.end<uchar>(); ++it)
        sum += *it;
    std::cout << "sum = " << sum << " (should be 4)\n";*/
    FileStorage fs("empty.yml", CV_STORAGE_READ);
    cout << fs["a"].type() << endl;
    return 0;
}
