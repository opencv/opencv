// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_FEATURES_FEATURE_EXTRACTOR_HPP
#define OPENCV_FEATURES_FEATURE_EXTRACTOR_HPP

#include "opencv2/core.hpp"
#include "opencv2/features.hpp"

namespace cv
{
    namespace features
    {

        class CV_EXPORTS_W FeatureExtractor : public Algorithm
        {
        public:
            virtual ~FeatureExtractor();

            CV_WRAP virtual void extract(InputArray image,
                                         CV_OUT std::vector<KeyPoint> &keypoints,
                                         OutputArray descriptors,
                                         InputArray mask = noArray()) const = 0;
        };

        class CV_EXPORTS_W TraditionalFeatureExtractor : public FeatureExtractor
        {
        public:
            CV_WRAP static Ptr<TraditionalFeatureExtractor> create(const Ptr<Feature2D> &backend);

            CV_WRAP virtual void setBackend(const Ptr<Feature2D> &backend) = 0;
            CV_WRAP virtual Ptr<Feature2D> getBackend() const = 0;
        };

    } // namespace features
} // namespace cv

#endif
