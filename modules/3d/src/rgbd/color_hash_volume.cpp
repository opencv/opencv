// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Partially rewritten from https://github.com/Nerei/kinfu_remake
// Copyright(c) 2012, Anatoly Baksheev. All rights reserved.

#include "../precomp.hpp"
#include "color_hash_volume.hpp"
#include <opencv2/3d/volume.hpp>

namespace cv {

inline Point3f nanPoint3f() {
    return Point3f(std::numeric_limits<float>::quiet_NaN(),
                  std::numeric_limits<float>::quiet_NaN(),
                  std::numeric_limits<float>::quiet_NaN());
}
inline Point3f normalizePoint3f(const Point3f& p) {
    const float epsilon = std::numeric_limits<float>::epsilon();
    float n = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (n > epsilon) {
        return Point3f(p.x/n, p.y/n, p.z/n);
    } else {
        // Return zero vector or handle error appropriately
        return Point3f(0.0f, 0.0f, 0.0f);
}

// Voxel structure with color
struct ColorVoxel {
    float tsdf;
    int weight;
    Vec3b color;
    
    ColorVoxel() : tsdf(1.0f), weight(0), color(Vec3b(0,0,0)) {}
};

class ColorHashTSDFVolumeImpl : public ColorHashTSDFVolume {
private:
    typedef std::pair<Point3i, ColorVoxel> HashMapElement;
    std::unordered_map<size_t, HashMapElement> hashMap;
    float voxelSize;
    float truncDist;
    Matx44f pose;
    float maxDepth;
    int maxWeight;

    size_t computeHash(const Point3i& p) const {
        const size_t p0 = static_cast<size_t>(p.x);
        const size_t p1 = static_cast<size_t>(p.y);
        const size_t p2 = static_cast<size_t>(p.z);
        return (p0 * 73856093) ^ (p1 * 19349669) ^ (p2 * 83492791);
    }

public:
    ColorHashTSDFVolumeImpl(const VolumeSettings& settings)
    {
        voxelSize = settings.getVoxelSize();
        truncDist = settings.getTsdfTruncateDistance();
        settings.getVolumePose(pose);
        

        try {
            maxDepth = settings.getMaxDepth();
        } catch (...) {
            maxDepth = 3.0f;
        }
        
        try {
            maxWeight = settings.getMaxWeight();
        } catch (...) {
            maxWeight = 100;
        }
    }

    void integrate(InputArray _depth, InputArray _color, InputArray _pose, InputArray _cameraMat) override
    {
        Mat depthMat = _depth.getMat();
        Mat colorMat = _color.getMat();
        Matx44f poseMat = _pose.getMat();
        Matx33f intrinsics = _cameraMat.getMat();

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

                size_t hashKey = computeHash(voxelCoords);
                auto it = hashMap.find(hashKey);
                ColorVoxel* voxel;
                
                if(it == hashMap.end()) {
                    hashMap[hashKey] = HashMapElement(voxelCoords, ColorVoxel());
                    voxel = &hashMap[hashKey].second;
                    voxel->tsdf = tsdf;
                    voxel->weight = 1;
                    voxel->color = colorMat.at<Vec3b>(y, x);
                }
                else {
                    voxel = &it->second.second;
                    float oldW = voxel->weight;
                    float newW = std::min(oldW + 1.0f, static_cast<float>(maxWeight));
                    
                    voxel->tsdf = (voxel->tsdf * oldW + tsdf) / newW;
                    voxel->weight = newW;

                    // Update color with running average when within truncation band
                    if(std::abs(sdf) <= truncDist)
                    {
                        Vec3b newColor = colorMat.at<Vec3b>(y, x);
                        float alpha = 1.0f / newW;
                        voxel->color = (1.0f - alpha) * voxel->color + alpha * newColor;
                    }
                }
            }
        }
    }

    void raycast(InputArray cameraPose, InputArray cameraMat, 
                 OutputArray points, OutputArray normals, OutputArray colors,
                 Size frameSize) override
    {
        points.create(frameSize, CV_32FC3);
        normals.create(frameSize, CV_32FC3);
        colors.create(frameSize, CV_8UC3);

        Mat pointsMat = points.getMat();
        Mat normalsMat = normals.getMat();
        Mat colorsMat = colors.getMat();
        
        const float raycastStepFactor = 0.8f;
        const float tstep = truncDist * raycastStepFactor;

        Affine3f cameraPoseAff(cameraPose.getMat());
        Affine3f volumePoseInv(pose.inv());
        Matx33f Rinv = cameraPoseAff.rotation().inv();

        parallel_for_(Range(0, frameSize.height), [&](const Range& range)
        {
            const Intr::Reprojector reproj(Intr(cameraMat.getMat()).makeReprojector());

            for (int y = range.start; y < range.end; y++)
            {
                Point3f* pointsRow = pointsMat.ptr<Point3f>(y);
                Point3f* normalsRow = normalsMat.ptr<Point3f>(y);
                Vec3b* colorsRow = colorsMat.ptr<Vec3b>(y);

                for (int x = 0; x < frameSize.width; x++)
                {
                    Point3f point = nanPoint3f();
                    Point3f normal = nanPoint3f();
                    Vec3b color = Vec3b(0, 0, 0);

                    Point3f orig = volumePoseInv.translation();
                    Point3f dir = normalizePoint3f(volumePoseInv.rotation() * reproj(Point3f(x, y, 1.0f)));

                    float tmin = 0;
                    float tmax = maxDepth;
                    
                    // Ray marching
                    float t = tmin;
                    float prevTsdf = truncDist;
                    Point3i prevVoxel = Point3i(0, 0, 0);

                    while (t < tmax)
                    {
                        Point3f p = orig + dir * t;
                        Point3i voxelCoords = worldToVoxel(p);
                        
                        auto it = hashMap.find(computeHash(voxelCoords));
                        if (it != hashMap.end() && it->second.first == voxelCoords) {
                            const ColorVoxel& voxel = it->second.second;
                            
                            if (voxel.tsdf * prevTsdf < 0) // Zero crossing
                            {
                                // Interpolate position and normal
                                t = t - tstep * voxel.tsdf / (prevTsdf - voxel.tsdf);
                                point = orig + dir * t;
                                normal = computeNormal(point);
                                color = voxel.color;
                                break;
                            }
                            
                            prevTsdf = voxel.tsdf;
                            prevVoxel = voxelCoords;
                        }
                        
                        t += tstep;
                    }

                    pointsRow[x] = point;
                    normalsRow[x] = Rinv * normal;
                    colorsRow[x] = color;
                }
            }
        });
    }

protected:
    Point3i worldToVoxel(const Point3f& p) const {
        Point3f voxelPt = p / voxelSize;
        return Point3i(cvRound(voxelPt.x), cvRound(voxelPt.y), cvRound(voxelPt.z));
    }

    Point3f voxelToWorld(const Point3i& v) const {
        return Point3f(v.x * voxelSize, v.y * voxelSize, v.z * voxelSize);
    }

    Point3f transformPoint(const Point3f& point, const Matx44f& transform)
    {
        Point3f result;
        result.x = transform(0,0) * point.x + transform(0,1) * point.y + transform(0,2) * point.z + transform(0,3);
        result.y = transform(1,0) * point.x + transform(1,1) * point.y + transform(1,2) * point.z + transform(1,3);
        result.z = transform(2,0) * point.x + transform(2,1) * point.y + transform(2,2) * point.z + transform(2,3);
        return result;
    }

    float computeSignedDistance(const Point3f& pt, const Point3f& camera) {
        return norm(pt - camera);
    }

    Point3f computeNormal(const Point3f& p)
    {
        const float eps = voxelSize;
        Point3f n;
        
        Point3f vx(eps, 0, 0);
        Point3f vy(0, eps, 0);
        Point3f vz(0, 0, eps);

        n.x = getTSDFValue(p + vx) - getTSDFValue(p - vx);
        n.y = getTSDFValue(p + vy) - getTSDFValue(p - vy);
        n.z = getTSDFValue(p + vz) - getTSDFValue(p - vz);
        
        float norm = sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
        return norm > 0 ? n * (1.0f/norm) : Point3f(0, 0, 0);
    }

    float getTSDFValue(const Point3f& p)
    {
        Point3i voxelCoords = worldToVoxel(p);
        auto it = hashMap.find(computeHash(voxelCoords));
        return it != hashMap.end() ? it->second.second.tsdf : truncDist;
    }
};

CV_EXPORTS_W Ptr<ColorHashTSDFVolume> ColorHashTSDFVolume::create(const VolumeSettings& settings){
    return makePtr<ColorHashTSDFVolumeImpl>(settings);
}
};

} // namespace cv
