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


class CV_EXPORTS_W_SIMPLE VolumeSettings
{
public:
    /** @brief Constructor of settings for custom Volume type.
    * @param volumeType volume type.
    */
    CV_WRAP explicit VolumeSettings(VolumeType volumeType = VolumeType::TSDF);

    VolumeSettings(const VolumeSettings& vs);
    VolumeSettings& operator=(const VolumeSettings&);
    ~VolumeSettings();

    /** @brief Sets the width of the image for integration.
    * @param val input value.
    */
    CV_WRAP void setIntegrateWidth(int val);

    /** @brief Returns the width of the image for integration.
    */
    CV_WRAP int getIntegrateWidth() const;

    /** @brief Sets the height of the image for integration.
    * @param val input value.
    */
    CV_WRAP void setIntegrateHeight(int val);

    /** @brief Returns the height of the image for integration.
    */
    CV_WRAP int getIntegrateHeight() const;

    /** @brief Sets the width of the raycasted image, used when user does not provide it at raycast() call.
    * @param val input value.
    */
    CV_WRAP void setRaycastWidth(int val);

    /** @brief Returns the width of the raycasted image, used when user does not provide it at raycast() call.
    */
    CV_WRAP int getRaycastWidth() const;

    /** @brief Sets the height of the raycasted image, used when user does not provide it at raycast() call.
    * @param val input value.
    */
    CV_WRAP void setRaycastHeight(int val);

    /** @brief Returns the height of the raycasted image, used when user does not provide it at raycast() call.
    */
    CV_WRAP int getRaycastHeight() const;

    /** @brief Sets depth factor, witch is the number for depth scaling.
    * @param val input value.
    */
    CV_WRAP void setDepthFactor(float val);

    /** @brief Returns depth factor, witch is the number for depth scaling.
    */
    CV_WRAP float getDepthFactor() const;

    /** @brief Sets the size of voxel.
    * @param val input value.
    */
    CV_WRAP void setVoxelSize(float val);

    /** @brief Returns the size of voxel.
    */
    CV_WRAP float getVoxelSize() const;

    /** @brief Sets TSDF truncation distance. Distances greater than value from surface will be truncated to 1.0.
    * @param val input value.
    */
    CV_WRAP void setTsdfTruncateDistance(float val);

    /** @brief Returns TSDF truncation distance. Distances greater than value from surface will be truncated to 1.0.
    */
    CV_WRAP float getTsdfTruncateDistance() const;

    /** @brief Sets threshold for depth truncation in meters. Truncates the depth greater than threshold to 0.
    * @param val input value.
    */
    CV_WRAP void setMaxDepth(float val);

    /** @brief Returns threshold for depth truncation in meters. Truncates the depth greater than threshold to 0.
    */
    CV_WRAP float getMaxDepth() const;

    /** @brief Sets max number of frames to integrate per voxel.
        Represents the max number of frames over which a running average of the TSDF is calculated for a voxel.
    * @param val input value.
    */
    CV_WRAP void setMaxWeight(int val);

    /** @brief Returns max number of frames to integrate per voxel.
        Represents the max number of frames over which a running average of the TSDF is calculated for a voxel.
    */
    CV_WRAP int getMaxWeight() const;

    /** @brief Sets length of single raycast step.
        Describes the percentage of voxel length that is skipped per march.
    * @param val input value.
    */
    CV_WRAP void setRaycastStepFactor(float val);

    /** @brief Returns length of single raycast step.
        Describes the percentage of voxel length that is skipped per march.
    */
    CV_WRAP float getRaycastStepFactor() const;

    /** @brief Sets volume pose.
    * @param val input value.
    */
    CV_WRAP void setVolumePose(InputArray val);

    /** @brief Sets volume pose.
    * @param val output value.
    */
    CV_WRAP void getVolumePose(OutputArray val) const;

    /** @brief Resolution of voxel space.
        Number of voxels in each dimension.
        Applicable only for TSDF Volume.
        HashTSDF volume only supports equal resolution in all three dimensions.
    * @param val input value.
    */
    CV_WRAP void setVolumeResolution(InputArray val);

    /** @brief Resolution of voxel space.
        Number of voxels in each dimension.
        Applicable only for TSDF Volume.
        HashTSDF volume only supports equal resolution in all three dimensions.
    * @param val output value.
    */
    CV_WRAP void getVolumeResolution(OutputArray val) const;

    /** @brief Returns 3 integers representing strides by x, y and z dimension.
        Can be used to iterate over raw volume unit data.
    * @param val output value.
    */
    CV_WRAP void getVolumeStrides(OutputArray val) const;

    /** @brief Sets intrinsics of camera for integrations.
    * Format of input:
    * [ fx  0 cx ]
    * [  0 fy cy ]
    * [  0  0  1 ]
    * where fx and fy are focus points of Ox and Oy axises, and cx and cy are central points of Ox and Oy axises.
    * @param val input value.
    */
    CV_WRAP void setCameraIntegrateIntrinsics(InputArray val);

    /** @brief Returns intrinsics of camera for integrations.
    * Format of output:
    * [ fx  0 cx ]
    * [  0 fy cy ]
    * [  0  0  1 ]
    * where fx and fy are focus points of Ox and Oy axises, and cx and cy are central points of Ox and Oy axises.
    * @param val output value.
    */
    CV_WRAP void getCameraIntegrateIntrinsics(OutputArray val) const;

    /** @brief Sets camera intrinsics for raycast image which, used when user does not provide them at raycast() call.
    * Format of input:
    * [ fx  0 cx ]
    * [  0 fy cy ]
    * [  0  0  1 ]
    * where fx and fy are focus points of Ox and Oy axises, and cx and cy are central points of Ox and Oy axises.
    * @param val input value.
    */
    CV_WRAP void setCameraRaycastIntrinsics(InputArray val);

    /** @brief Returns camera intrinsics for raycast image, used when user does not provide them at raycast() call.
    * Format of output:
    * [ fx  0 cx ]
    * [  0 fy cy ]
    * [  0  0  1 ]
    * where fx and fy are focus points of Ox and Oy axises, and cx and cy are central points of Ox and Oy axises.
    * @param val output value.
    */
    CV_WRAP void getCameraRaycastIntrinsics(OutputArray val) const;

    class Impl;
private:
    Ptr<Impl> impl;
};

}

#endif // !OPENCV_3D_VOLUME_SETTINGS_HPP
