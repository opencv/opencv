// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#ifndef OPENCV_3D_VOLUME_SETTINGS_HPP
#define OPENCV_3D_VOLUME_SETTINGS_HPP

#include <opencv2/core/cvstd.hpp>

namespace cv
{

    class VolumeSettings
    {
    public:
        VolumeSettings();
        ~VolumeSettings();

        class Impl;
    private:
        Ptr<Impl> impl;
    };

}

#endif // !OPENCV_3D_VOLUME_SETTINGS_HPP
