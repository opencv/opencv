// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_SLOT_QUEUE_HPP
#define OPENCV_CORE_SLOT_QUEUE_HPP

#include "opencv2/core.hpp"
#include <vector>

namespace cv {

class CV_EXPORTS_W SlotQueue
{
public:
    SlotQueue();
    explicit SlotQueue(int capacity);

    CV_WRAP void push(InputArray frame);
    CV_WRAP int getSize() const;
    CV_WRAP int getCapacity() const { return n; }
    CV_WRAP void getFrame(int index, OutputArray out) const;
    CV_WRAP void clear();

private:
    std::vector<Mat> ring;
    int head;
    int tail;
    int n;
};

} // namespace cv

#endif
