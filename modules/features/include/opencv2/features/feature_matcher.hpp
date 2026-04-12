// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_FEATURES_FEATURE_MATCHER_HPP
#define OPENCV_FEATURES_FEATURE_MATCHER_HPP

#include "opencv2/core.hpp"
#include "opencv2/features.hpp"

namespace cv
{
namespace features
{

    class CV_EXPORTS_W FeatureMatcher : public Algorithm
    {
    public:
        virtual ~FeatureMatcher();

        CV_WRAP virtual void match(InputArray queryDesc,
                                   InputArray trainDesc,
                                   CV_OUT std::vector<DMatch> &matches,
                                   InputArray mask = noArray()) const = 0;

        CV_WRAP virtual void match(InputArray queryKpts,
                                   InputArray queryDesc,
                                   InputArray trainKpts,
                                   InputArray trainDesc,
                                   CV_OUT std::vector<DMatch> &matches,
                                   InputArray mask = noArray(),
                                   Size queryImageSize = Size(),
                                   Size trainImageSize = Size()) const;
    };

    class CV_EXPORTS_W TraditionalFeatureMatcher : public FeatureMatcher
    {
    public:
        CV_WRAP static Ptr<TraditionalFeatureMatcher> create(const Ptr<DescriptorMatcher> &descriptorMatcher);

        CV_WRAP virtual void setDescriptorMatcher(const Ptr<DescriptorMatcher> &descriptorMatcher) = 0;
        CV_WRAP virtual Ptr<DescriptorMatcher> getDescriptorMatcher() const = 0;
    };

    class CV_EXPORTS_W LightGlue : public FeatureMatcher
    {
    public:
        struct Params
        {
            Params();

            String modelPath;
            int dnnEngine;
            int dnnBackend;
            int dnnTarget;
            bool disableWinograd;
            float scoreThreshold;
        };

        CV_WRAP static Ptr<LightGlue> create();
        static Ptr<LightGlue> create(const LightGlue::Params &params);

        CV_WRAP virtual void setModel(const String &modelPath) = 0;
        CV_WRAP virtual String getModel() const = 0;
    };

} // namespace features
} // namespace cv

#endif
