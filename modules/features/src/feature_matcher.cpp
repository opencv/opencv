// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/features/feature_matcher.hpp"

namespace cv
{
namespace features
{

        FeatureMatcher::~FeatureMatcher() {}

        void FeatureMatcher::match(InputArray,
                                   InputArray queryDesc,
                                   InputArray,
                                   InputArray trainDesc,
                                   std::vector<DMatch> &matches,
                                   InputArray mask,
                                   Size,
                                   Size) const
        {
            match(queryDesc, trainDesc, matches, mask);
        }

        namespace
        {

            class TraditionalFeatureMatcherImpl CV_FINAL : public TraditionalFeatureMatcher
            {
            public:
                explicit TraditionalFeatureMatcherImpl(const Ptr<DescriptorMatcher> &descriptorMatcher)
                    : descriptorMatcher_(descriptorMatcher)
                {
                }

                void match(InputArray queryDesc,
                           InputArray trainDesc,
                           std::vector<DMatch> &matches,
                           InputArray mask) const CV_OVERRIDE
                {
                    CV_Assert(!descriptorMatcher_.empty());
                    descriptorMatcher_->match(queryDesc, trainDesc, matches, mask);
                }

                void setBackend(const Ptr<DescriptorMatcher> &descriptorMatcher) CV_OVERRIDE
                {
                    descriptorMatcher_ = descriptorMatcher;
                }

                Ptr<DescriptorMatcher> getBackend() const CV_OVERRIDE
                {
                    return descriptorMatcher_;
                }

            private:
                Ptr<DescriptorMatcher> descriptorMatcher_;
            };

        } // namespace

        Ptr<TraditionalFeatureMatcher> TraditionalFeatureMatcher::create(const Ptr<DescriptorMatcher> &descriptorMatcher)
        {
            return makePtr<TraditionalFeatureMatcherImpl>(descriptorMatcher);
        }

} // namespace features
} // namespace cv
