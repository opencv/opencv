// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include "color_hash_tsdf_functions.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/ocl.hpp"

namespace cv {

template<>
class DataType<RGBTsdfVoxel>
{
public:
    typedef float channel_type;
    enum { 
        channels = 4,
        type = CV_32FC4 
    };
};

}

namespace cv
{

void integrateColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int& lastVolIndex, const int frameId, const int volumeUnitDegree, bool enableGrowth,
    InputArray _depth, InputArray _rgb, InputArray _pixNorms, InputOutputArray _volUnitsData, VolumeUnitIndexes& volumeUnits)
{
    CV_TRACE_FUNCTION();
    
    Mat depth = _depth.getMat();
    Mat rgb = _rgb.getMat();
    Mat pixNorms = _pixNorms.getMat();
    Mat volUnitsData = _volUnitsData.getMat();
    
    CV_Assert(!depth.empty());
    CV_Assert(!rgb.empty());
    CV_Assert(depth.size() == rgb.size());
    
    float voxelSize = settings.getVoxelSize();
    float trancDist = settings.getTsdfTruncateDistance();
    int maxWeight = settings.getMaxWeight();
    Vec3i volResolution;
    settings.getVolumeResolution(volResolution);
    
    Matx33f intrMat;
    settings.getCameraIntegrateIntrinsics(intrMat);
    Intr intrinsics(intrMat);
    
    const int volUnitSize = 1 << volumeUnitDegree;
    const Point3i volUnitDims = Point3i(volUnitSize, volUnitSize, volUnitSize);
    const int unitResolution = volUnitSize * volUnitSize * volUnitSize;
    
    // Iterate through all existing volume units and integrate depth data
    std::vector<Vec3i> unitsToRemove;
    
    for (auto it = volumeUnits.begin(); it != volumeUnits.end(); ++it)
    {
        auto& unitPair = *it;
        const Vec3i& unitIdx = unitPair.first;
        VolumeUnit& unit = unitPair.second;
        
        bool visible = integrateColorVolumeUnit(unit, settings, cameraPose, depth, rgb, pixNorms, 
                                              unitIdx, volUnitDims, voxelSize, trancDist, maxWeight, 
                                              intrinsics);
        
        unit.lastVisibleIndex = frameId;
        if (!visible && frameId - unit.lastVisibleIndex > settings.getVolumeUnitHideThreshold())
        {
            unitsToRemove.push_back(unitIdx);
        }
    }
    
    // Remove invisible units if growth is disabled
    if (!enableGrowth)
    {
        for (const Vec3i& unitIdx : unitsToRemove)
        {
            volumeUnits.erase(unitIdx);
        }
    }
    
    // Find new visible volume units and create them
    if (enableGrowth)
    {
        std::vector<Vec3i> newUnits = findNewVisibleVolumeUnits(settings, cameraPose, depth, intrinsics, 
                                                               volumeUnits, volUnitDims, voxelSize);
        
        for (const Vec3i& unitIdx : newUnits)
        {
            if (volumeUnits.find(unitIdx) == volumeUnits.end())
            {
                VolumeUnit newUnit;
                newUnit.index = lastVolIndex++;
                newUnit.lastVisibleIndex = frameId;
                
                int dataStart = newUnit.index * unitResolution;
                CV_Assert(dataStart + unitResolution <= volUnitsData.rows * volUnitsData.cols);
                
                Mat unitData = volUnitsData.rowRange(dataStart, dataStart + unitResolution);
                unitData.forEach<ColorHashTsdfVoxel>([](ColorHashTsdfVoxel& voxel, const int*) {
                    voxel.tsdf = floatToTsdf(0.0f);
                    voxel.weight = 0;
                    voxel.r = voxel.g = voxel.b = 0;
                });
                
                volumeUnits[unitIdx] = newUnit;
                
                integrateColorVolumeUnit(volumeUnits[unitIdx], settings, cameraPose, depth, rgb, pixNorms,
                                       unitIdx, volUnitDims, voxelSize, trancDist, maxWeight, intrinsics);
            }
        }
    }
}

void raycastColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int height, int width, InputArray intr, const int volumeUnitDegree,
    InputArray _volUnitsData, const CustomHashSet& hashTable, OutputArray _points, OutputArray _normals, OutputArray _colors)
{
    CV_TRACE_FUNCTION();

    Mat volUnitsData = _volUnitsData.getMat();
    Matx33f cameraIntr = intr.getMat();

    _points.create(height, width, CV_32FC3);
    _normals.create(height, width, CV_32FC3);
    _colors.create(height, width, CV_8UC3);

    Mat points = _points.getMat();
    Mat normals = _normals.getMat();
    Mat colors = _colors.getMat();

    points.setTo(0);
    normals.setTo(0);
    colors.setTo(0);

    float voxelSize = settings.getVoxelSize();
    float trancDist = settings.getTsdfTruncateDistance();
    Vec3i volResolution;
    settings.getVolumeResolution(volResolution);

    const int volUnitSize = 1 << volumeUnitDegree;
    const Point3i volUnitDims = Point3i(volUnitSize, volUnitSize, volUnitSize);
    const int unitResolution = volUnitSize * volUnitSize * volUnitSize;

    Matx44f invCameraPose = cameraPose.inv();
    float voxelSizeInv = 1.0f / voxelSize;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Point3f rayDirVec(
                (x - cameraIntr(0, 2)) / cameraIntr(0, 0),
                (y - cameraIntr(1, 2)) / cameraIntr(1, 1),
                1.0f
            );
            float rayDirNorm = cv::norm(rayDirVec);
            Point3f rayDir = rayDirNorm > 0 ? rayDirVec / rayDirNorm : Point3f(0, 0, 0);

            Point3f rayOrigin(0, 0, 0);
            Point3f rayDirVol = transformDirection(rayDir, invCameraPose);
            Point3f rayOriginVol = transformPoint(rayOrigin, invCameraPose) * voxelSizeInv;

            float tmin = 0.0f;
            float tmax = settings.getRaycastStepFactor() * voxelSizeInv;

            Point3f hitPoint, hitNormal;
            Vec3b hitColor;
            bool hit = raycastColorHashTsdf(settings, rayOriginVol, rayDirVol, tmin, tmax, hashTable, volUnitsData,
                                          volUnitDims, unitResolution, voxelSize, trancDist,
                                          hitPoint, hitNormal, hitColor);

            if (hit)
            {
                // Transform hit point back to camera space
                Point3f worldPoint = hitPoint * voxelSize;
                Point3f cameraPoint = transformPoint(worldPoint, cameraPose);

                points.at<Point3f>(y, x) = cameraPoint;
                normals.at<Point3f>(y, x) = hitNormal;
                colors.at<Vec3b>(y, x) = hitColor;
            }
        }
    }
}

void fetchPointsNormalsColorsFromColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, InputArray _volUnitsData, const CustomHashSet& hashTable,
    const int volumeUnitDegree, OutputArray _points, OutputArray _normals, OutputArray _colors)
{
    CV_TRACE_FUNCTION();

    Mat volUnitsData = _volUnitsData.getMat();
    
    float voxelSize = settings.getVoxelSize();
    float trancDist = settings.getTsdfTruncateDistance();
    float gradientDeltaFactor = settings.getGradientDeltaFactor();
    
    const int volUnitSize = 1 << volumeUnitDegree;
    const Point3i volUnitDims = Point3i(volUnitSize, volUnitSize, volUnitSize);
    const int unitResolution = volUnitSize * volUnitSize * volUnitSize;
    
    std::vector<Point3f> pointsVec;
    std::vector<Point3f> normalsVec;
    std::vector<Vec3b> colorsVec;
    
    for (const auto& unitPair : hashTable)
    {
        const Vec3i& unitIdx = unitPair.first;
        const VolumeUnit& unit = unitPair.second;
        
        Point3f unitOrigin = Point3f(unitIdx[0], unitIdx[1], unitIdx[2]) * volUnitSize * voxelSize;
        
        int dataStart = unit.index * unitResolution;
        CV_Assert(dataStart + unitResolution <= volUnitsData.rows * volUnitsData.cols);
        
        Mat unitData = volUnitsData.rowRange(dataStart, dataStart + unitResolution);
        
        // Extract surface points from this volume unit
        for (int z = 1; z < volUnitSize - 1; z++)
        {
            for (int y = 1; y < volUnitSize - 1; y++)
            {
                for (int x = 1; x < volUnitSize - 1; x++)
                {
                    int voxelIndex = z * volUnitSize * volUnitSize + y * volUnitSize + x;
                    const ColorHashTsdfVoxel& voxel = unitData.ptr<ColorHashTsdfVoxel>()[voxelIndex];
                    
                    if (voxel.weight > 0 && std::abs(tsdfToFloat(voxel.tsdf)) < trancDist)
                    {
                        Point3f voxelPos = unitOrigin + Point3f(x, y, z) * voxelSize;
                        
                        Point3f normal = computeColorVoxelNormal(unitData, x, y, z, volUnitDims.x, voxelSize, 1.0f);
                        
                        if (!isNaN(normal))
                        {
                            pointsVec.push_back(voxelPos);
                            normalsVec.push_back(normal / cv::norm(normal));
                            colorsVec.push_back(Vec3b(voxel.r, voxel.g, voxel.b));
                        }
                    }
                }
            }
        }
    }
    
    // Convert vectors to output arrays
    if (!pointsVec.empty())
    {
        Mat(pointsVec).reshape(1, (int)pointsVec.size()).copyTo(_points);
        Mat(normalsVec).reshape(1, (int)normalsVec.size()).copyTo(_normals);
        Mat(colorsVec).reshape(1, (int)colorsVec.size()).copyTo(_colors);
    }
    else
    {
        _points.release();
        _normals.release();
        _colors.release();
    }
}

// Helper functions
bool integrateColorVolumeUnit(const VolumeSettings& settings, const Matx44f& cameraPose,
                            InputArray _depth, InputArray _rgb, InputArray _pixNorms,
                            const Vec3i& unitIdx, const Point3i& volUnitDims,
                            float voxelSize, float trancDist, const Intr& intrinsics)
{
    CV_TRACE_FUNCTION();
    Mat depth = _depth.getMat();
    Mat rgb = _rgb.getMat();
    Mat pixNorms = _pixNorms.getMat();

    // Calculate world coordinates of the volume unit
    Point3f unitOrigin = Point3f(unitIdx[0], unitIdx[1], unitIdx[2]) * volUnitDims.x * voxelSize;

    // Invert the camera pose to transform world points to camera coordinates
    Matx44f worldToCamera = cameraPose.inv();

    bool isVisible = false;

    // Iterate through all voxels in the volume unit
    for (int z = 0; z < volUnitDims.z; z++)
    {
        for (int y = 0; y < volUnitDims.y; y++)
        {
            for (int x = 0; x < volUnitDims.x; x++)
            {
                // Calculate world position of this voxel
                Point3f worldPos = unitOrigin + Point3f(x, y, z) * voxelSize;

                // Transform world position to camera coordinates
                Point3f cameraPos = transformPoint(worldPos, worldToCamera);
                if (cameraPos.z <= 0) continue;

                // Project the 3D point to 2D image coordinates
                Point2f pixelPos;
                pixelPos.x = cameraPos.x * intrinsics.fx / cameraPos.z + intrinsics.cx;
                pixelPos.y = cameraPos.y * intrinsics.fy / cameraPos.z + intrinsics.cy;

                // Check if the projected point is within the image bounds
                if (pixelPos.x >= 0 && pixelPos.x < depth.cols && pixelPos.y >= 0 && pixelPos.y < depth.rows)
                {
                    int px = static_cast<int>(pixelPos.x);
                    int py = static_cast<int>(pixelPos.y);

                    // Get the observed depth from the depth map
                    float observedDepth = depth.at<float>(py, px);

                    // Check if the observed depth is valid (not zero or too far)
                    if (observedDepth > 0 && observedDepth < settings.getMaxDepth())
                    {
                        // Calculate the signed distance (SDF)
                        float sdf = observedDepth - cameraPos.z;

                        // Clamp the SDF to the truncation distance
                        sdf = std::max(std::min(sdf, trancDist), -trancDist);

                        // Simplified visibility check: if a pixel projects onto the unit's area
                        // and has valid depth within influence range, mark as visible.
                        if (std::abs(sdf) < trancDist * 2.0f) {
                            isVisible = true;
                        }
                    }
                }
            }
        }
    }
    return isVisible;
}

std::vector<Vec3i> findNewVisibleVolumeUnits(const VolumeSettings& settings, const Matx44f& cameraPose,
                                           InputArray depth, const Intr& intrinsics,
                                           const VolumeUnitIndexes& existingUnits,
                                           const Point3i& volUnitDims, float voxelSize)
{
    CV_TRACE_FUNCTION();
    std::vector<Vec3i> newUnits;
    Mat depthMat = depth.getMat();
    

    for (int y = 0; y < depthMat.rows; y += 10) {
        for (int x = 0; x < depthMat.cols; x += 10) {
            float d = depthMat.at<float>(y, x);
            if (d > 0 && d < settings.getMaxDepth()) {
                // Convert pixel coordinates to 3D point in camera space
                Point3f cameraPoint;
                cameraPoint.z = d;
                cameraPoint.x = (x - intrinsics.cx) * d / intrinsics.fx;
                cameraPoint.y = (y - intrinsics.cy) * d / intrinsics.fy;

                // Transform the 3D point from camera space to world space
                Point3f worldPoint = transformPoint(cameraPoint, cameraPose);

                // Calculate which volume unit this world point belongs to
                Vec3i potentialUnitIdx(
                    static_cast<int>(std::floor(worldPoint.x / (volUnitDims.x * voxelSize))),
                    static_cast<int>(std::floor(worldPoint.y / (volUnitDims.y * voxelSize))),
                    static_cast<int>(std::floor(worldPoint.z / (volUnitDims.z * voxelSize)))
                );

                // Check if this unit already exists in the map
                if (existingUnits.find(potentialUnitIdx) == existingUnits.end()) {
                    // If not found, add it as a new unit to be created
                    newUnits.push_back(potentialUnitIdx);
                }
            }
        }
    }

    // Remove duplicate unit indices
    std::sort(newUnits.begin(), newUnits.end(), [](const Vec3i& a, const Vec3i& b) {
        // Lexicographic comparison: compare z first, then y, then x
        if (a[2] != b[2]) return a[2] < b[2];
        if (a[1] != b[1]) return a[1] < b[1];
        return a[0] < b[0];
    });
    newUnits.erase(std::unique(newUnits.begin(), newUnits.end()), newUnits.end());
    
    return newUnits;
}

Point3f transformDirection(const Point3f& dir, const Matx44f& mat)
{
    return Point3f(mat(0,0)*dir.x + mat(0,1)*dir.y + mat(0,2)*dir.z,
                  mat(1,0)*dir.x + mat(1,1)*dir.y + mat(1,2)*dir.z,
                  mat(2,0)*dir.x + mat(2,1)*dir.y + mat(2,2)*dir.z);
}

Point3f transformPoint(const Point3f& pt, const Matx44f& mat) 
{
    return Point3f(mat(0,0)*pt.x + mat(0,1)*pt.y + mat(0,2)*pt.z + mat(0,3),
                  mat(1,0)*pt.x + mat(1,1)*pt.y + mat(1,2)*pt.z + mat(1,3),
                  mat(2,0)*pt.x + mat(2,1)*pt.y + mat(2,2)*pt.z + mat(2,3));
}

bool raycastColorHashTsdf(const VolumeSettings& settings, const Point3f& rayOrigin, const Point3f& rayDir, float tmin,
                              float tmax, const VolumeUnitIndexes& volumeUnits, InputArray _volUnitsData,
                              const Point3i& volUnitDims, int unitResolution, float voxelSize, float trancDist,
                              Point3f& hitPoint, Point3f& hitNormal, Vec3b& hitColor)
    {
        CV_TRACE_FUNCTION();
        Mat volUnitsData = _volUnitsData.getMat();

        float t = tmin;
        const float stepSize = voxelSize * settings.getRaycastStepFactor();

        while (t < tmax) {
            Point3f currentPos = rayOrigin + t * rayDir;

            // Determine which volume unit this point might be in
            Vec3i unitIdx(
                static_cast<int>(std::floor(currentPos.x / (volUnitDims.x * voxelSize))),
                static_cast<int>(std::floor(currentPos.y / (volUnitDims.y * voxelSize))),
                static_cast<int>(std::floor(currentPos.z / (volUnitDims.z * voxelSize)))
            );

            // Check if the potential unit exists in our hash map
            auto unitIt = volumeUnits.find(unitIdx);
            if (unitIt != volumeUnits.end()) {
                const VolumeUnit& unit = unitIt->second;

                // Calculate the position within the unit's coordinate system
                Point3i localCoord(
                    static_cast<int>(std::floor((currentPos.x - unitIdx[0] * volUnitDims.x * voxelSize) / voxelSize)),
                    static_cast<int>(std::floor((currentPos.y - unitIdx[1] * volUnitDims.y * voxelSize) / voxelSize)),
                    static_cast<int>(std::floor((currentPos.z - unitIdx[2] * volUnitDims.z * voxelSize) / voxelSize))
                );

                // Check if local coordinates are within the unit's bounds
                if (localCoord.x >= 0 && localCoord.x < volUnitDims.x &&
                    localCoord.y >= 0 && localCoord.y < volUnitDims.y &&
                    localCoord.z >= 0 && localCoord.z < volUnitDims.z) {

                    // Calculate the index within the unit's data array
                    int voxelIndex = localCoord.z * volUnitDims.x * volUnitDims.y +
                                     localCoord.y * volUnitDims.x + localCoord.x;

                    // Ensure the unit's data slice is accessible
                    int dataStart = unit.index * unitResolution;
                    if (dataStart + unitResolution <= volUnitsData.rows * volUnitsData.cols) {
                        Mat unitData = volUnitsData.rowRange(dataStart, dataStart + unitResolution);
                        const ColorHashTsdfVoxel& voxel = unitData.ptr<ColorHashTsdfVoxel>()[voxelIndex];

                        // Check if this voxel represents the surface (TSDF near zero)
                        if (voxel.weight > 0 && std::abs(tsdfToFloat(voxel.tsdf)) < trancDist * 0.1f) {
                            hitPoint = currentPos;
                            hitColor = Vec3b(voxel.b, voxel.g, voxel.r);

                            // Compute normal using gradient of TSDF values in the neighborhood
                            hitNormal = computeColorVoxelNormal(unitData, localCoord.x, localCoord.y, localCoord.z,
                                                              volUnitDims.x, voxelSize, 1.0f);

                            // Normalize the normal vector
                            float normLength = cv::norm(hitNormal);
                            if (normLength > 1e-6f) {
                                hitNormal /= normLength;
                            } else {
                                // Fallback if gradient calculation failed
                                hitNormal = -rayDir;
                            }
                            return true;
                        }
                    }
                }
            }
            t += stepSize;
        }
        return false;
}

Point3f computeColorVoxelNormal(InputArray _unitData, int x, int y, int z, int volUnitSize)
{
    CV_TRACE_FUNCTION();
    Mat unitData = _unitData.getMat();

    // Ensure the voxel is not on the boundary where gradient calculation is impossible
    if (x <= 0 || x >= volUnitSize - 1 || y <= 0 || y >= volUnitSize - 1 || z <= 0 || z >= volUnitSize - 1) {
        return Point3f(0, 0, 0);
    }

    // Calculate indices for neighboring voxels in x, y, z directions
    int idx_xp = z * volUnitSize * volUnitSize + y * volUnitSize + (x + 1);
    int idx_xm = z * volUnitSize * volUnitSize + y * volUnitSize + (x - 1);
    int idx_yp = z * volUnitSize * volUnitSize + (y + 1) * volUnitSize + x;
    int idx_ym = z * volUnitSize * volUnitSize + (y - 1) * volUnitSize + x;
    int idx_zp = (z + 1) * volUnitSize * volUnitSize + y * volUnitSize + x;
    int idx_zm = (z - 1) * volUnitSize * volUnitSize + y * volUnitSize + x;

    // Get pointers to the voxel data
    const ColorHashTsdfVoxel* ptr = unitData.ptr<ColorHashTsdfVoxel>();

    // Check if indices are valid (within the unit data range)
    int unitResolution = volUnitSize * volUnitSize * volUnitSize;
    if (idx_xp >= unitResolution || idx_xm < 0 || idx_yp >= unitResolution ||
        idx_ym < 0 || idx_zp >= unitResolution || idx_zm < 0) {
        return Point3f(0, 0, 0);
    }

    // Calculate gradients using central differences (finite differences)
    float grad_x = tsdfToFloat(ptr[idx_xp].tsdf) - tsdfToFloat(ptr[idx_xm].tsdf);
    float grad_y = tsdfToFloat(ptr[idx_yp].tsdf) - tsdfToFloat(ptr[idx_ym].tsdf);
    float grad_z = tsdfToFloat(ptr[idx_zp].tsdf) - tsdfToFloat(ptr[idx_zm].tsdf);

    // Return the gradient vector (which is the normal direction)
    // The normal points in the direction of increasing SDF (away from the surface)
    Point3f normal(grad_x, grad_y, grad_z);
    return normal;
}

#ifdef HAVE_OPENCL

void ocl_integrateColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int& lastVolIndex, const int frameId,
    const int volumeUnitDegree, bool enableGrowth, InputArray _depth, InputArray _rgb,
    InputArray _pixNorms, InputOutputArray _volUnitsData, VolumeUnitIndexes& volumeUnits)
{
    integrateColorHashTsdfVolumeUnit(
        settings, cameraPose, lastVolIndex, frameId, volumeUnitDegree, enableGrowth,
        _depth, _rgb, _pixNorms, _volUnitsData, volumeUnits);
}

void ocl_raycastColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int height, int width,
    InputArray intr, const int volumeUnitDegree, InputArray _volUnitsData,
    const CustomHashSet& hashTable, OutputArray _points, OutputArray _normals,
    OutputArray _colors)
{
    raycastColorHashTsdfVolumeUnit(
        settings, cameraPose, height, width, intr, volumeUnitDegree,
        _volUnitsData, hashTable, _points, _normals, _colors);
}

void ocl_fetchPointsNormalsColorsFromColorHashTsdfVolumeUnit(const VolumeSettings& settings,
    InputArray _volUnits, const CustomHashSet& hashTable,
    int volumeUnitDegree, OutputArray points, OutputArray normals, OutputArray colors)
{
    fetchPointsNormalsColorsFromColorHashTsdfVolumeUnit(
        settings, _volUnits, hashTable, volumeUnitDegree, points, normals, colors);
}

#endif

} // namespace cv