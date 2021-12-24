// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_VOLUME_HPP
#define OPENCV_3D_VOLUME_HPP

#include "volume_settings.hpp"

#include "opencv2/core/affine.hpp"

namespace cv
{

class CV_EXPORTS_W Volume
{
public:
    Volume();
    Volume(VolumeType vtype, const VolumeSettings& settings);
    ~Volume();

    void integrate(const OdometryFrame& frame, InputArray pose);
    void integrate(InputArray depth, InputArray pose);
    void integrate(InputArray depth, InputArray image, InputArray pose);
    void raycast(InputArray cameraPose, int height, int width, OdometryFrame& outFrame) const;
    void raycast(InputArray cameraPose, int height, int width, OutputArray points, OutputArray normals) const;
    void raycast(InputArray cameraPose, int height, int width, OutputArray points, OutputArray normals, OutputArray colors) const;

    void fetchNormals(InputArray points, OutputArray normals) const;
    void fetchPointsNormals(OutputArray points, OutputArray normals) const;
    void fetchPointsNormalsColors(OutputArray points, OutputArray normals, OutputArray colors) const;

    void reset();
    int getVisibleBlocks() const;
    size_t getTotalVolumeUnits() const;

    class Impl;
private:
    Ptr<Impl> impl;
};

}  // namespace cv
#endif // include guard
