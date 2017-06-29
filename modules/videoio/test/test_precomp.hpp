#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  if defined __clang__ || defined __APPLE__
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#    pragma GCC diagnostic ignored "-Wextra"
#  endif
#endif

#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include <iostream>
#include "opencv2/ts.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#include "opencv2/core/private.hpp"

inline std::string fourccToString(int fourcc)
{
    return cv::format("%c%c%c%c", fourcc & 255, (fourcc >> 8) & 255, (fourcc >> 16) & 255, (fourcc >> 24) & 255);
}

inline int fourccFromString(const std::string &fourcc)
{
    if (fourcc.size() != 4) return 0;
    return cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
}

inline void generateFrame(int i, int FRAME_COUNT, cv::Mat & frame)
{
    using namespace cv;
    using namespace std;
    frame = Scalar(30, 140, 10);
    ostringstream buf; buf << "Frame " << setw(2) << setfill('0') << i + 1;
    int baseLine = 0;
    Size box = getTextSize(buf.str(), FONT_HERSHEY_COMPLEX, 2, 5, &baseLine);
    putText(frame, buf.str(), Point((frame.cols - box.width) / 2, (frame.rows - box.height) / 2 + baseLine),
            FONT_HERSHEY_COMPLEX, 2, Scalar(255, 255, 255), 5, LINE_AA);
    Point p(i * frame.cols / (FRAME_COUNT - 1), i * frame.rows / (FRAME_COUNT - 1));
    circle(frame, p, 20, Scalar(200, 25, 55), 5, LINE_AA);
}

#endif
