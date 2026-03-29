// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_FEATURES_LIGHTGLUE_HPP
#define OPENCV_FEATURES_LIGHTGLUE_HPP

#include "opencv2/features/feature_matcher.hpp"

namespace cv
{

    namespace features
    {

        class CV_EXPORTS_W LightGlue : public FeatureMatcher
        {
        public:
            struct Params
            {
                Params();

                String modelPath;
                int engine;
                int backend;
                int target;
                bool disableWinograd;

                String kpts0Name;
                String desc0Name;
                String kpts1Name;
                String desc1Name;

                String matches0Name;
                String mscores0Name;
            };

            static Ptr<LightGlue> create(const LightGlue::Params &params = LightGlue::Params());

            CV_WRAP virtual void setModel(const String &modelPath) = 0;
            CV_WRAP virtual String getModel() const = 0;
        };

    } // namespace features
} // namespace cv

#endif
