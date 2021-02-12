#ifndef __OPENCV_PERF_PRECOMP_HPP__
#define __OPENCV_PERF_PRECOMP_HPP__

#include "opencv2/ts.hpp"
#include "opencv2/stitching.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

namespace cv
{

static inline Ptr<Feature2D> getFeatureFinder(const std::string& name)
{
    if (name == "orb")
        return ORB::create();
#if defined(HAVE_OPENCV_XFEATURES2D) && defined(OPENCV_ENABLE_NONFREE)
    else if (name == "surf")
        return xfeatures2d::SURF::create();
#endif
    else if (name == "akaze")
        return AKAZE::create();
    else
        return Ptr<Feature2D>();
}

} // namespace cv

#endif
