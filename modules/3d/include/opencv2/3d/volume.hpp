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

    /** @brief Interrates the input data to the volume.
    * @param frame the object which consist depth and image etc.
    * @param pose the pose of camera.
    */
    void integrate(const OdometryFrame& frame, InputArray pose);
    /** @brief Interrates the input data to the volume.
    * @param depth the depth frame.
    * @param pose the pose of camera.
    */
    void integrate(InputArray depth, InputArray pose);
    /** @brief Interrates the input data to the volume.
    * @param depth the depth frame.
    * @param image the depth frame (only for ColorTSDF).
    * @param pose the pose of camera.
    */
    void integrate(InputArray depth, InputArray image, InputArray pose);

    /** @brief Extract the points of volume from set position.
    * @param cameraPose the pose of camera (camera in volume environment, not real).
    * @param height the height of result image.
    * @param width the width of result image.
    * @param outFrame the object, which store the result data.
    */
    void raycast(InputArray cameraPose, int height, int width, OdometryFrame& outFrame) const;
    /** @brief Extract the points of volume from set position.
    * @param cameraPose the pose of camera (camera in volume environment, not real).
    * @param height the height of result image.
    * @param width the width of result image.
    * @param points the storage of points in the image.
    * @param normals the storage of normals (corresponding to points) in the image.
    */
    void raycast(InputArray cameraPose, int height, int width, OutputArray points, OutputArray normals) const;
    /** @brief Extract the points of volume from set position.
    * @param cameraPose the pose of camera (camera in volume environment, not real).
    * @param height the height of result image.
    * @param width the width of result image.
    * @param points the storage of points in the image.
    * @param normals the storage of normals (corresponding to points) in the image.
    * @param colors the storage of colors (corresponding to points) in the image (only for ColorTSDF).
    */
    void raycast(InputArray cameraPose, int height, int width, OutputArray points, OutputArray normals, OutputArray colors) const;

    /** @brief Extract the all data from volume.
    * @param points the input exist point.
    * @param normals the storage of normals (corresponding to input points) in the image.
    */
    void fetchNormals(InputArray points, OutputArray normals) const;
    /** @brief Extract the all data from volume.
    * @param points the storage of all points.
    * @param normals the storage of all normals, corresponding to points.
    * @param colors the storage of all colors, corresponding to points (only for ColorTSDF).
    */
    void fetchPointsNormals(OutputArray points, OutputArray normals) const;
    /** @brief Extract the all data from volume.
    * @param points the storage of all points.
    * @param normals the storage of all normals, corresponding to points.
    * @param colors the storage of all colors, corresponding to points (only for ColorTSDF).
    */
    void fetchPointsNormalsColors(OutputArray points, OutputArray normals, OutputArray colors) const;

    /** @brief clear all data in volume
    */
    void reset();

    /** @brief return visible blocks in volume
    */
    int getVisibleBlocks() const;

    /** @brief return number of vulmeunits in volume
    */
    size_t getTotalVolumeUnits() const;

    class Impl;
private:
    Ptr<Impl> impl;
};

}  // namespace cv
#endif // include guard
