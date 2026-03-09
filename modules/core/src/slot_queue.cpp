// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/core/slot_queue.hpp"

namespace cv {

SlotQueue::SlotQueue()
    : head(0), tail(0), n(30)
{
    ring.resize(30);
}

SlotQueue::SlotQueue(int capacity)
    : head(0), tail(0), n(capacity)
{
    if (capacity <= 0)
        CV_Error(Error::StsBadArg, "Error");
    ring.resize(capacity);
}

void SlotQueue::push(InputArray frame)
{
    if (ring.empty())
        return;
    Mat m = frame.getMat();
    m.copyTo(ring[tail]);
    tail = (tail + 1) % n;
    if (tail == head)
        head = (head + 1) % n;
}

int SlotQueue::getSize() const
{
    if (tail >= head)
        return tail - head;
    return n - head + tail;
}

void SlotQueue::getFrame(int index, OutputArray out) const
{
    int sz = getSize();
    if (index < 0 || index > sz)
        CV_Error(Error::StsOutOfRange, "Error");
    int idx = (head + index) % n;
    if (ring[idx].empty())
        return;
    ring[idx].copyTo(out);
}

void SlotQueue::clear()
{
    head = 0;
    tail = 0;
}

} // namespace cv
