#ifndef __OPENCV_PERF_PRECOMP_HPP__
#define __OPENCV_PERF_PRECOMP_HPP__

#include "opencv2/ts.hpp"
#include "opencv2/stitching.hpp"

namespace cv
{

static inline Ptr<detail::FeaturesFinder> getFeatureFinder(const std::string& name)
{
    if (name == "orb")
        return makePtr<detail::OrbFeaturesFinder>();
    else if (name == "surf")
        return makePtr<detail::SurfFeaturesFinder>();
    else if (name == "akaze")
        return makePtr<detail::AKAZEFeaturesFinder>();
    else
        return Ptr<detail::FeaturesFinder>();
}

} // namespace cv

#endif
