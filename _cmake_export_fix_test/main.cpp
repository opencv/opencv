#include <opencv2/core.hpp>

int main()
{
    cv::Mat m(1, 1, CV_8U);
    return m.empty() ? 1 : 0;
}
