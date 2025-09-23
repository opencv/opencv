#ifndef OPENCV_3D_COLOR_HASH_VOLUME_HPP
#define OPENCV_3D_COLOR_HASH_VOLUME_HPP

#include <opencv2/3d/volume.hpp>

namespace cv {

class ColorHashTSDFVolumeImpl : public Volume
{
public:
    ColorHashTSDFVolumeImpl(const VolumeSettings& settings);
    
    virtual void integrate(InputArray _depth, InputArray _image, InputArray _pose, InputArray _cameraMat) CV_OVERRIDE;
    
    virtual void raycast(InputArray cameraPose, InputArray cameraMat, 
                        OutputArray points, OutputArray normals, 
                        Size frameSize = Size()) const CV_OVERRIDE;

private:
    // Hash table structure for sparse storage
    struct VoxelUnit {
        float tsdf;
        float weight;
        Vec3b color;
    };
    
    std::unordered_map<size_t, VoxelUnit> hashMap;
    float voxelSize;
    float truncDist;
    Matx44f pose;

    Point3f transformPoint(const Point3f& pt, const Matx44f& transform)
    {
        Point3f result;
        result.x = transform(0,0) * pt.x + transform(0,1) * pt.y + transform(0,2) * pt.z + transform(0,3);
        result.y = transform(1,0) * pt.x + transform(1,1) * pt.y + transform(1,2) * pt.z + transform(1,3);
        result.z = transform(2,0) * pt.x + transform(2,1) * pt.y + transform(2,2) * pt.z + transform(2,3);
        return result;
    }
    
    Point3i worldToVoxel(const Point3f& pt)
    {
        return Point3i(static_cast<int>(pt.x / voxelSize),
                      static_cast<int>(pt.y / voxelSize),
                      static_cast<int>(pt.z / voxelSize));
    }
    
    float computeSignedDistance(const Point3f& ptWorld, const Point3f& ptCam)
    {
        return norm(ptWorld - ptCam);
    }
    
    size_t generateHashKey(const Point3i& voxelCoords)
    {
        // Simple spatial hash function
        return ((size_t)(voxelCoords.x) * 73856093) ^
               ((size_t)(voxelCoords.y) * 19349663) ^
               ((size_t)(voxelCoords.z) * 83492791);
    }
};

} // namespace cv

#endif 