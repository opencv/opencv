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
                explicit TraditionalFeatureExtractorImpl(const Ptr<Feature2D> &backend)
                    : backend_(backend)
                {
                }

                void extract(InputArray image,
                             std::vector<KeyPoint> &keypoints,
                             OutputArray descriptors,
                             InputArray mask) const CV_OVERRIDE
                {
                    CV_Assert(!backend_.empty());
                    backend_->detectAndCompute(image, mask, keypoints, descriptors, false);
                }

                void setBackend(const Ptr<Feature2D> &backend) CV_OVERRIDE
                {
                    backend_ = backend;
                }

                Ptr<Feature2D> getBackend() const CV_OVERRIDE
                {
                    return backend_;
                }

            private:
                Ptr<Feature2D> backend_;
            };

        } // namespace

        Ptr<TraditionalFeatureExtractor> TraditionalFeatureExtractor::create(const Ptr<Feature2D> &backend)
        {
            return makePtr<TraditionalFeatureExtractorImpl>(backend);
        }

    } // namespace features
} // namespace cv
