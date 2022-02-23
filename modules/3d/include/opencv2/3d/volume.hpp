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
    /** @brief Constructor of default volume - TSDF.
    */
    Volume();
    /** @brief Constructor of custom volume.
    * @param vtype the volume type [TSDF, HashTSDF, ColorTSDF].
    * @param settings the custom settings for volume.
    */
    Volume(VolumeType vtype, const VolumeSettings& settings);
    ~Volume();

    /** @brief Integrates the input data to the volume.

    Camera intrinsics are taken from volume settings structure.

    * @param frame the object from which to take depth and image data.
      For color TSDF a depth data should be registered with color data, i.e. have the same intrinsics & camera pose.
      This can be done using function registerDepth() from 3d module.
    * @param pose the pose of camera in global coordinates.
    */
    void integrate(const OdometryFrame& frame, InputArray pose);

    /** @brief Integrates the input data to the volume.

    Camera intrinsics are taken from volume settings structure.

    * @param depth the depth image.
    * @param pose the pose of camera in global coordinates.
    */
    void integrate(InputArray depth, InputArray pose);

    /** @brief Integrates the input data to the volume.

    Camera intrinsics are taken from volume settings structure.

    * @param depth the depth image.
    * @param image the color image (only for ColorTSDF).
      For color TSDF a depth data should be registered with color data, i.e. have the same intrinsics & camera pose.
      This can be done using function registerDepth() from 3d module.
    * @param pose the pose of camera in global coordinates.
    */
    void integrate(InputArray depth, InputArray image, InputArray pose);

    /** @brief Renders the volume contents into an image. The resulting points and normals are in camera's coordinate system.

    Rendered image size and camera intrinsics are taken from volume settings structure.

    * @param cameraPose the pose of camera in global coordinates.
    * @param outFrame the object where to store rendered points and normals.
    */
    void raycast(InputArray cameraPose, OdometryFrame& outFrame) const;

    /** @brief Renders the volume contents into an image. The resulting points and normals are in camera's coordinate system.

    Rendered image size and camera intrinsics are taken from volume settings structure.

    * @param cameraPose the pose of camera in global coordinates.
    * @param points image to store rendered points.
    * @param normals image to store rendered normals corresponding to points.
    * @param colors image to store rendered colors corresponding to points (only for ColorTSDF).
    */
    void raycast(InputArray cameraPose, OutputArray points, OutputArray normals, OutputArray colors = noArray()) const;

    /** @brief Renders the volume contents into an image. The resulting points and normals are in camera's coordinate system.

    Rendered image size and camera intrinsics are taken from volume settings structure.

    * @param cameraPose the pose of camera in global coordinates.
    * @param height the height of result image.
    * @param width the width of result image.
    * @param outFrame the object where to store rendered points and normals.
    */
    void raycast(InputArray cameraPose, int height, int width, OdometryFrame& outFrame) const;

    /** @brief Renders the volume contents into an image. The resulting points and normals are in camera's coordinate system.

    Rendered image size and camera intrinsics are taken from volume settings structure.

    * @param cameraPose the pose of camera in global coordinates.
    * @param height the height of result image.
    * @param width the width of result image.
    * @param points image to store rendered points.
    * @param normals image to store rendered normals corresponding to points.
    * @param colors image to store rendered colors corresponding to points (only for ColorTSDF).
    */
    void raycast(InputArray cameraPose, int height, int width, OutputArray points, OutputArray normals, OutputArray colors = noArray()) const;

    /** @brief Extract the all data from volume.
    * @param points the input exist point.
    * @param normals the storage of normals (corresponding to input points) in the image.
    */
    void fetchNormals(InputArray points, OutputArray normals) const;
    /** @brief Extract the all data from volume.
    * @param points the storage of all points.
    * @param normals the storage of all normals, corresponding to points.
    */
    void fetchPointsNormals(OutputArray points, OutputArray normals) const;
    /** @brief Extract the all data from volume.
    * @param points the storage of all points.
    * @param normals the storage of all normals, corresponding to points.
    * @param colors the storage of all colors, corresponding to points (only for ColorTSDF).
    */
    void fetchPointsNormalsColors(OutputArray points, OutputArray normals, OutputArray colors) const;

    /** @brief clear all data in volume.
    */
    void reset();

    /** @brief return visible blocks in volume.
    */
    int getVisibleBlocks() const;

    /** @brief return number of vulmeunits in volume.
    */
    size_t getTotalVolumeUnits() const;

    class Impl;
private:
    Ptr<Impl> impl;
};

}  // namespace cv
#endif // include guard
