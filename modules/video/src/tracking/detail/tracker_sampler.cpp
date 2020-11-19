// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#include "opencv2/video/detail/tracking.private.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

TrackerSampler::TrackerSampler()
{
    blockAddTrackerSampler = false;
}

TrackerSampler::~TrackerSampler()
{
    // nothing
}

void TrackerSampler::sampling(const Mat& image, Rect boundingBox)
{
    clearSamples();

    for (size_t i = 0; i < samplers.size(); i++)
    {
        CV_DbgAssert(samplers[i]);
        std::vector<Mat> current_samples;
        samplers[i]->sampling(image, boundingBox, current_samples);

        //push in samples all current_samples
        for (size_t j = 0; j < current_samples.size(); j++)
        {
            std::vector<Mat>::iterator it = samples.end();
            samples.insert(it, current_samples.at(j));
        }
    }

    blockAddTrackerSampler = true;
}

bool TrackerSampler::addTrackerSamplerAlgorithm(const Ptr<TrackerSamplerAlgorithm>& sampler)
{
    CV_Assert(!blockAddTrackerSampler);
    CV_Assert(sampler);

    samplers.push_back(sampler);
    return true;
}

const std::vector<Ptr<TrackerSamplerAlgorithm>>& TrackerSampler::getSamplers() const
{
    return samplers;
}

const std::vector<Mat>& TrackerSampler::getSamples() const
{
    return samples;
}

void TrackerSampler::clearSamples()
{
    samples.clear();
}

}}}  // namespace cv::detail::tracking
