// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef BYTETRACKER_HPP
#define BYTETRACKER_HPP

#include "opencv2/video/detail/tracking.detail.hpp"
#include "tracker_mil_state.hpp"
#include <map>
#include <unordered_map>

using namespace cv;

namespace cv {
inline namespace tracking {
namespace impl {

using namespace cv::detail::tracking;
std::map<int, int> lapjv(std::vector<std::vector<float>>, std::vector<int>& , std::vector<int>&);


class CV_EXPORTS_W ByteTracker : public Tracker {
protected:
    ByteTracker();
public:
    virtual ~ByteTracker() CV_OVERRIDE;

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();
        CV_PROP_RW int frameRate;
        CV_PROP_RW int frameBuffer;
    };
    
    static CV_WRAP
    Ptr<ByteTracker> create(const ByteTracker::Params& parameters = ByteTracker::Params());

};
}
}
}

#endif // BYTETRACKER_STRACK_HPP
