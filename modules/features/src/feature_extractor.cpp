// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/features/feature_extractor.hpp"

namespace cv
{
namespace features
{

        FeatureExtractor::~FeatureExtractor() {}

        namespace
        {

            class TraditionalFeatureExtractorImpl CV_FINAL : public TraditionalFeatureExtractor
            {
            public:
                explicit TraditionalFeatureExtractorImpl(const Ptr<Feature2D> &feature2D)
                    : feature2D_(feature2D)
                {
                }

                void extract(InputArray image,
                             std::vector<KeyPoint> &keypoints,
                             OutputArray descriptors,
                             InputArray mask) const CV_OVERRIDE
                {
                    CV_Assert(!feature2D_.empty());
                    feature2D_->detectAndCompute(image, mask, keypoints, descriptors, false);
                }

                void setBackend(const Ptr<Feature2D> &feature2D) CV_OVERRIDE
                {
                    feature2D_ = feature2D;
                }

                Ptr<Feature2D> getBackend() const CV_OVERRIDE
                {
                    return feature2D_;
                }

            private:
                Ptr<Feature2D> feature2D_;
            };

        } // namespace

        Ptr<TraditionalFeatureExtractor> TraditionalFeatureExtractor::create(const Ptr<Feature2D> &feature2D)
        {
            return makePtr<TraditionalFeatureExtractorImpl>(feature2D);
        }

} // namespace features
} // namespace cv
