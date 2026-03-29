// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_FEATURES_SUPERPOINT_HPP
#define OPENCV_FEATURES_SUPERPOINT_HPP

#include "opencv2/features/feature_extractor.hpp"

namespace cv
{

    namespace features
    {

        class CV_EXPORTS_W SuperPoint : public FeatureExtractor
        {
        public:
            struct Params
            {
                Params();

                String modelPath;
                int engine;
                int backend;
                int target;
                Size inputSize;
                bool swapRB;

                String inputName;
                String keypointsName;
                String descriptorsName;
                String scoresName;
            };

            static Ptr<SuperPoint> create(const SuperPoint::Params &params = SuperPoint::Params());

            CV_WRAP virtual void setModel(const String &modelPath) = 0;
            CV_WRAP virtual String getModel() const = 0;
        };

    } // namespace features
} // namespace cv

#endif
