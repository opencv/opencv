// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_VIDEO_DETAIL_TRACKING_ONLINE_MIL_HPP
#define OPENCV_VIDEO_DETAIL_TRACKING_ONLINE_MIL_HPP

#include <limits>

namespace cv {
namespace detail {
inline namespace tracking {

//! @addtogroup tracking_detail
//! @{

//TODO based on the original implementation
//http://vision.ucsd.edu/~bbabenko/project_miltrack.shtml

class ClfOnlineStump;

class CV_EXPORTS ClfMilBoost
{
public:
    struct CV_EXPORTS Params
    {
        Params();
        int _numSel;
        int _numFeat;
        float _lRate;
    };

    ClfMilBoost();
    ~ClfMilBoost();
    void init(const ClfMilBoost::Params& parameters = ClfMilBoost::Params());
    void update(const Mat& posx, const Mat& negx);
    std::vector<float> classify(const Mat& x, bool logR = true);

    inline float sigmoid(float x)
    {
        return 1.0f / (1.0f + exp(-x));
    }

private:
    uint _numsamples;
    ClfMilBoost::Params _myParams;
    std::vector<int> _selectors;
    std::vector<ClfOnlineStump*> _weakclf;
    uint _counter;
};

class ClfOnlineStump
{
public:
    float _mu0, _mu1, _sig0, _sig1;
    float _q;
    int _s;
    float _log_n1, _log_n0;
    float _e1, _e0;
    float _lRate;

    ClfOnlineStump();
    ClfOnlineStump(int ind);
    void init();
    void update(const Mat& posx, const Mat& negx, const cv::Mat_<float>& posw = cv::Mat_<float>(), const cv::Mat_<float>& negw = cv::Mat_<float>());
    bool classify(const Mat& x, int i);
    float classifyF(const Mat& x, int i);
    std::vector<float> classifySetF(const Mat& x);

private:
    bool _trained;
    int _ind;
};

//! @}

}}}  // namespace cv::detail::tracking

#endif
