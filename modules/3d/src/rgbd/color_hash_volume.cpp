#include "precomp.hpp"
#include "color_hash_volume.hpp"

namespace cv {

ColorHashTSDFVolumeImpl::ColorHashTSDFVolumeImpl(const VolumeSettings& settings)
{
    voxelSize = settings.getVoxelSize();
    truncDist = settings.getTsdfTruncateDistance();
    settings.getVolumePose(pose);
}

void ColorHashTSDFVolumeImpl::integrate(InputArray depth, InputArray pose, InputArray cameraMat)
{
    Mat depthMat = depth.getMat();
    Matx44f poseMat = pose.getMat();
    Matx33f intrinsics = cameraMat.getMat();
    

    const float fx = intrinsics(0, 0);
    const float fy = intrinsics(1, 1);
    const float cx = intrinsics(0, 2);
    const float cy = intrinsics(1, 2);

    Matx44f invPose = poseMat.inv();

     for(int y = 0; y < depthMat.rows; y++)
    {
        for(int x = 0; x < depthMat.cols; x++)
        {
            float depth = depthMat.at<float>(y, x);
            if(depth <= 0 || depth > maxDepth)
                continue;
                
            Point3f ptCam;
            ptCam.x = (x - cx) * depth / fx;
            ptCam.y = (y - cy) * depth / fy;
            ptCam.z = depth;
            
            Point3f ptWorld = transformPoint(ptCam, invPose);
            Point3i voxelCoords = worldToVoxel(ptWorld);
            
            float sdf = computeSignedDistance(ptWorld, ptCam);
            float tsdf = std::min(1.0f, sdf / truncDist);
            tsdf = std::max(-1.0f, tsdf);
            
            size_t hashKey = generateHashKey(voxelCoords);
            
            auto& voxel = hashMap[hashKey];
            if(voxel.weight == 0)
            {
                voxel.tsdf = tsdf;
                voxel.weight = 1;
            }
            else
            {
                float oldW = voxel.weight;
                float newW = std::min(oldW + 1, maxWeight);
                voxel.tsdf = (voxel.tsdf * oldW + tsdf) / newW;
                voxel.weight = newW;
            }
            
            if(std::abs(sdf) <= truncDist)
            {
                Vec3b color = colorImage.at<Vec3b>(y, x);
                if(voxel.weight == 1)
                {
                    voxel.color = color;
                }
                else
                {
                    float alpha = 1.0f / voxel.weight;
                    voxel.color = (1 - alpha) * voxel.color + alpha * color;
                }
            }
        }
    }
    
}

void ColorHashTSDFVolumeImpl::raycast(InputArray cameraPose, InputArray cameraMat,
                                    OutputArray points, OutputArray normals,
                                    Size frameSize) const
{
    Matx44f viewPose = cameraPose.getMat();
    Matx33f intrinsics = cameraMat.getMat();
    
    // TODO: Implement raycasting with color interpolation
}

} 