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
                explicit TraditionalFeatureMatcherImpl(const Ptr<DescriptorMatcher> &backend)
                    : backend_(backend)
                {
                }

                void match(InputArray queryDesc,
                           InputArray trainDesc,
                           std::vector<DMatch> &matches,
                           InputArray mask) const CV_OVERRIDE
                {
                    CV_Assert(!backend_.empty());
                    backend_->match(queryDesc, trainDesc, matches, mask);
                }

                void setBackend(const Ptr<DescriptorMatcher> &backend) CV_OVERRIDE
                {
                    backend_ = backend;
                }

                Ptr<DescriptorMatcher> getBackend() const CV_OVERRIDE
                {
                    return backend_;
                }

            private:
                Ptr<DescriptorMatcher> backend_;
            };

        } // namespace

        Ptr<TraditionalFeatureMatcher> TraditionalFeatureMatcher::create(const Ptr<DescriptorMatcher> &backend)
        {
            return makePtr<TraditionalFeatureMatcherImpl>(backend);
        }

    } // namespace features
} // namespace cv
