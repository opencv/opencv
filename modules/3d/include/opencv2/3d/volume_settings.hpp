// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#ifndef OPENCV_3D_VOLUME_SETTINGS_HPP
#define OPENCV_3D_VOLUME_SETTINGS_HPP

#include <opencv2/core.hpp>
#include <opencv2/3d/volume.hpp>

namespace cv
{

enum class VolumeType
{
    TSDF = 0,
    HashTSDF = 1,
    ColorTSDF = 2
};


class CV_EXPORTS_W VolumeSettings
{
public:
    /** @brief Constructor of settings for common TSDF volume type.
    */
    VolumeSettings();

    /** @brief Constructor of settings for custom Volume type.
    * @param volumeType volume type.
    */
    VolumeSettings(VolumeType volumeType);
    ~VolumeSettings();

    /** @brief Sets the width of the image for integration.
    * @param val input value.
    */
    void  setIntegrateWidth(int val);

    /** @brief Returns the width of the image for integration.
    */
    int   getIntegrateWidth() const;

    /** @brief Sets the height of the image for integration.
    * @param val input value.
    */
    void  setIntegrateHeight(int val);

    /** @brief Returns the height of the image for integration.
    */
    int   getIntegrateHeight() const;


    /** @brief Sets the width of the raycasted image.
    * @param val input value.
    */
    void  setRaycastWidth(int val);

    /** @brief Returns the width of the raycasted image.
    */
    int   getRaycastWidth() const;

    /** @brief Sets the height of the raycasted image.
    * @param val input value.
    */
    void  setRaycastHeight(int val);

    /** @brief Returns the height of the raycasted image.
    */
    int   getRaycastHeight() const;


    /** @brief Sets depth factor, witch is the number for depth scaling.
    * @param val input value.
    */
    void  setDepthFactor(float val);

    /** @brief Returns depth factor, witch is the number for depth scaling.
    */
    float getDepthFactor() const;

    /** @brief Sets the size of voxel.
    * @param val input value.
    */
    void  setVoxelSize(float val);

    /** @brief Returns the size of voxel.
    */
    float getVoxelSize() const;


    /** @brief Sets TSDF truncation distance. Distances greater than value from surface will be truncated to 1.0.
    * @param val input value.
    */
    void  setTsdfTruncateDistance(float val);

    /** @brief Returns TSDF truncation distance. Distances greater than value from surface will be truncated to 1.0.
    */
    float getTsdfTruncateDistance() const;

    /** @brief Sets threshold for depth truncation in meters. Truncates the depth greater than threshold to 0.
    * @param val input value.
    */
    void  setMaxDepth(float val);

    /** @brief Returns threshold for depth truncation in meters. Truncates the depth greater than threshold to 0.
    */
    float getMaxDepth() const;

    /** @brief Sets max number of frames to integrate per voxel.
        Represents the max number of frames over which a running average of the TSDF is calculated for a voxel.
    * @param val input value.
    */
    void  setMaxWeight(int val);

    /** @brief Returns max number of frames to integrate per voxel.
        Represents the max number of frames over which a running average of the TSDF is calculated for a voxel.
    */
    int   getMaxWeight() const;

    /** @brief Sets length of single raycast step.
        Describes the percentage of voxel length that is skipped per march.
    * @param val input value.
    */
    void  setRaycastStepFactor(float val);

    /** @brief Returns length of single raycast step.
        Describes the percentage of voxel length that is skipped per march.
    */
    float getRaycastStepFactor() const;

    /** @brief Sets volume pose.
    * @param val input value.
    */
    void setVolumePose(InputArray val);

    /** @brief Sets volume pose.
    * @param val output value.
    */
    void getVolumePose(OutputArray val) const;

    /** @brief Resolution of voxel space.
        Number of voxels in each dimension.
        Applicable only for TSDF Volume.
        HashTSDF volume only supports equal resolution in all three dimensions.
    * @param val input value.
    */
    void setVolumeResolution(InputArray val);

    /** @brief Resolution of voxel space.
        Number of voxels in each dimension.
        Applicable only for TSDF Volume.
        HashTSDF volume only supports equal resolution in all three dimensions.
    * @param val output value.
    */
    void getVolumeResolution(OutputArray val) const;

    /** @brief Returns 3 integers representing strides by x, y and z dimension.
        Can be used to iterate over volume unit raw data.
    * @param val output value.
    */
    void getVolumeDimensions(OutputArray val) const;

    /** @brief Sets intrinsics of camera for integrations.
    * Format of input:
    * [ fx  0 cx ]
    * [  0 fy cy ]
    * [  0  0  1 ]
    * where fx and fy are focus points of Ox and Oy axises, and cx and cy are central points of Ox and Oy axises.
    * @param val input value.
    */
    void setCameraIntegrateIntrinsics(InputArray val);

    /** @brief Returns intrinsics of camera for integrations.
    * Format of output:
    * [ fx  0 cx ]
    * [  0 fy cy ]
    * [  0  0  1 ]
    * where fx and fy are focus points of Ox and Oy axises, and cx and cy are central points of Ox and Oy axises.
    * @param val output value.
    */
    void getCameraIntegrateIntrinsics(OutputArray val) const;

    /** @brief Sets intrinsics of camera for raycast image.
    * Format of input:
    * [ fx  0 cx ]
    * [  0 fy cy ]
    * [  0  0  1 ]
    * where fx and fy are focus points of Ox and Oy axises, and cx and cy are central points of Ox and Oy axises.
    * @param val input value.
    */
    void setCameraRaycastIntrinsics(InputArray val);

    /** @brief Returns intrinsics of camera for raycast image.
    * Format of output:
    * [ fx  0 cx ]
    * [  0 fy cy ]
    * [  0  0  1 ]
    * where fx and fy are focus points of Ox and Oy axises, and cx and cy are central points of Ox and Oy axises.
    * @param val output value.
    */
    void getCameraRaycastIntrinsics(OutputArray val) const;


    class Impl;
private:
    Ptr<Impl> impl;
};

}

#endif // !OPENCV_3D_VOLUME_SETTINGS_HPP
