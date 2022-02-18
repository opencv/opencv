// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_ODOMETRY_FRAME_HPP
#define OPENCV_3D_ODOMETRY_FRAME_HPP

#include <opencv2/core.hpp>

namespace cv
{
/** Indicates what pyramid is to access using get/setPyramid... methods:
* @param PYR_IMAGE The pyramid of RGB images
* @param PYR_DEPTH The pyramid of depth images
* @param PYR_MASK  The pyramid of masks
* @param PYR_CLOUD The pyramid of point clouds, produced from the pyramid of depths
* @param PYR_DIX   The pyramid of dI/dx derivative images
* @param PYR_DIY   The pyramid of dI/dy derivative images
* @param PYR_TEXMASK The pyramid of textured masks
* @param PYR_NORM  The pyramid of normals
* @param PYR_NORMMASK The pyramid of normals masks
*/

enum OdometryFramePyramidType
{
    PYR_IMAGE = 0,
    PYR_DEPTH = 1,
    PYR_MASK = 2,
    PYR_CLOUD = 3,
    PYR_DIX = 4,
    PYR_DIY = 5,
    PYR_TEXMASK = 6,
    PYR_NORM = 7,
    PYR_NORMMASK = 8,
    N_PYRAMIDS
};

enum class OdometryFrameStoreType
{
    MAT  = 0,
    UMAT = 1
};

class CV_EXPORTS_W OdometryFrame
{
public:
    OdometryFrame();
    OdometryFrame(OdometryFrameStoreType matType);
    ~OdometryFrame() {};
    void setImage(InputArray  image);
    void getImage(OutputArray image) const;
    void getGrayImage(OutputArray image) const;
    void setDepth(InputArray  depth);
    void getDepth(OutputArray depth) const;
    void getScaledDepth(OutputArray depth) const;
    void setMask(InputArray  mask);
    void getMask(OutputArray mask) const;
    void setNormals(InputArray  normals);
    void getNormals(OutputArray normals) const;
    void setPyramidLevel(size_t _nLevels, OdometryFramePyramidType oftype);
    void setPyramidLevels(size_t _nLevels);
    size_t getPyramidLevels(OdometryFramePyramidType oftype) const;
    void setPyramidAt(InputArray  img, OdometryFramePyramidType pyrType, size_t level);
    void getPyramidAt(OutputArray img, OdometryFramePyramidType pyrType, size_t level) const;

    class Impl;
private:
    Ptr<Impl> impl;
};
}
#endif // !OPENCV_3D_ODOMETRY_FRAME_HPP
