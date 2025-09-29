// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include "color_hash_tsdf_functions.hpp"
#include "opencv2/core/hal/intrin.hpp"

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
    
    for (auto& unitPair : volumeUnits)
    {
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
                unitData.forEach<ColorHashTsdfVoxel>([](ColorHashTsdfVoxel& voxel, const int* pos) {
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
    InputArray _volUnitsData, const VolumeUnitIndexes& volumeUnits, OutputArray _points, OutputArray _normals, OutputArray _colors)
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
            Point3f rayDir = normalize(Point3f(
                (x - cameraIntr(0, 2)) / cameraIntr(0, 0),
                (y - cameraIntr(1, 2)) / cameraIntr(1, 1),
                1.0f
            ));
            
            Point3f rayOrigin(0, 0, 0);
            Point3f rayDirVol = transformDirection(rayDir, invCameraPose);
            Point3f rayOriginVol = transformPoint(rayOrigin, invCameraPose) * voxelSizeInv;
            
            float tmin = 0.0f;
            float tmax = settings.getRaycastMaxDistance() * voxelSizeInv;
            
            Point3f hitPoint, hitNormal;
            Vec3b hitColor;
            bool hit = raycastColorHashTsdf(rayOriginVol, rayDirVol, tmin, tmax, volumeUnits, volUnitsData,
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
    const VolumeSettings& settings, InputArray _volUnitsData, const VolumeUnitIndexes& volumeUnits,
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
    
    for (const auto& unitPair : volumeUnits)
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
                        
                        Point3f normal = computeColorVoxelNormal(unitData, x, y, z, volUnitSize, voxelSize, gradientDeltaFactor);
                        
                        if (!isNaN(normal))
                        {
                            pointsVec.push_back(voxelPos);
                            normalsVec.push_back(normalize(normal));
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
bool integrateColorVolumeUnit(VolumeUnit& unit, const VolumeSettings& settings, const Matx44f& cameraPose,
                            InputArray _depth, InputArray _rgb, InputArray _pixNorms,
                            const Vec3i& unitIdx, const Point3i& volUnitDims,
                            float voxelSize, float trancDist, int maxWeight, const Intr& intrinsics)
{
    return true; 
}

std::vector<Vec3i> findNewVisibleVolumeUnits(const VolumeSettings& settings, const Matx44f& cameraPose,
                                           InputArray _depth, const Intr& intrinsics,
                                           const VolumeUnitIndexes& existingUnits,
                                           const Point3i& volUnitDims, float voxelSize)
{
    std::vector<Vec3i> newUnits;
    return newUnits; 
}

bool raycastColorHashTsdf(const Point3f& rayOrigin, const Point3f& rayDir, float tmin, float tmax,
                         const VolumeUnitIndexes& volumeUnits, InputArray _volUnitsData,
                         const Point3i& volUnitDims, int unitResolution, float voxelSize, float trancDist,
                         Point3f& hitPoint, Point3f& hitNormal, Vec3b& hitColor)
{
    return false; 
}

Point3f computeColorVoxelNormal(InputArray _unitData, int x, int y, int z, int volUnitSize, float voxelSize, float deltaFactor)
{
    return Point3f(0, 0, 0); 
}

#ifdef HAVE_OPENCL

void ocl_integrateColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int& lastVolIndex, const int frameId, int& bufferSizeDegree, const int volumeUnitDegree, bool enableGrowth,
    InputArray _depth, InputArray _rgb, InputArray _pixNorms, InputArray _lastVisibleIndices, InputOutputArray _volUnitsDataCopy, InputOutputArray _volUnitsData, CustomHashSet& hashTable, InputArray _isActiveFlags)
{
    CV_TRACE_FUNCTION();
    CV_Error(cv::Error::StsNotImplemented, "OpenCL implementation for Color Hash TSDF not implemented yet");
}

void ocl_raycastColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int height, int width, InputArray intr, const int volumeUnitDegree,
    const CustomHashSet& hashTable, InputArray _volUnitsData, OutputArray _points, OutputArray _normals, OutputArray _colors)
{
    CV_TRACE_FUNCTION();
    CV_Error(cv::Error::StsNotImplemented, "OpenCL implementation for Color Hash TSDF raycast not implemented yet");
}

void ocl_fetchPointsNormalsColorsFromColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const int volumeUnitDegree, InputArray _volUnitsData, InputArray _volUnitsDataCopy,
    const CustomHashSet& hashTable, OutputArray _points, OutputArray _normals, OutputArray _colors)
{
    CV_TRACE_FUNCTION();
    CV_Error(cv::Error::StsNotImplemented, "OpenCL implementation for Color Hash TSDF fetch not implemented yet");
}

#endif 

} // namespace cv