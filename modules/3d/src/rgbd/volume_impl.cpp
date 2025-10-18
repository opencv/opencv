// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include <iostream>
#include "volume_impl.hpp"
#include "tsdf_functions.hpp"
#include "hash_tsdf_functions.hpp"
#include "color_tsdf_functions.hpp"
#include "color_hash_tsdf_functions.hpp"
#include "opencv2/imgproc.hpp"

namespace cv
{

Volume::Impl::Impl(const VolumeSettings& _settings) :
    settings(_settings)
#ifdef HAVE_OPENCL
    , useGPU(ocl::useOpenCL())
#endif
{}

// TSDF

TsdfVolume::TsdfVolume(const VolumeSettings& _settings) :
    Volume::Impl(_settings)
{
    Vec3i volResolution;
    settings.getVolumeResolution(volResolution);
#ifndef HAVE_OPENCL
    volume = Mat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<TsdfVoxel>());
#else
    if (useGPU)
        gpu_volume = UMat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<TsdfVoxel>());
    else
        cpu_volume = Mat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<TsdfVoxel>());
#endif

    reset();
}
TsdfVolume::~TsdfVolume() {}

void TsdfVolume::integrate(const OdometryFrame& frame, InputArray _cameraPose)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENCL
    Mat depth;
#else
    UMat depth;
#endif
    frame.getDepth(depth);
    integrate(depth, _cameraPose);
}

void TsdfVolume::integrate(InputArray _depth, InputArray _cameraPose)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENCL
    Mat depth = _depth.getMat();
#else
    UMat depth = _depth.getUMat();
#endif
    CV_Assert(!depth.empty());

    Matx33f intr;
    settings.getCameraIntegrateIntrinsics(intr);
    Intr intrinsics(intr);
    Vec6f newParams((float)depth.rows, (float)depth.cols,
                    intrinsics.fx, intrinsics.fy,
                    intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
#ifndef HAVE_OPENCL
        preCalculationPixNorm(depth.size(), intrinsics, pixNorms);
#else
        if (useGPU)
            ocl_preCalculationPixNorm(depth.size(), intrinsics, gpu_pixNorms);
        else
            preCalculationPixNorm(depth.size(), intrinsics, cpu_pixNorms);
#endif
    }
    const Matx44f cameraPose = _cameraPose.getMat();

#ifndef HAVE_OPENCL
    integrateTsdfVolumeUnit(settings, cameraPose, depth, pixNorms, volume);
#else
    if (useGPU)
        ocl_integrateTsdfVolumeUnit(settings, cameraPose, depth, gpu_pixNorms, gpu_volume);
    else
        integrateTsdfVolumeUnit(settings, cameraPose, depth, cpu_pixNorms, cpu_volume);
#endif
}
void TsdfVolume::integrate(InputArray, InputArray, InputArray)
{
    CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");
}


void TsdfVolume::raycast(InputArray cameraPose, OutputArray points, OutputArray normals, OutputArray colors)  const
{
    Matx33f intr;
    settings.getCameraRaycastIntrinsics(intr);
    raycast(cameraPose, settings.getRaycastHeight(), settings.getRaycastWidth(), intr, points, normals, colors);
}


void TsdfVolume::raycast(InputArray _cameraPose, int height, int width, InputArray intr, OutputArray _points, OutputArray _normals, OutputArray _colors) const
{
    if (_colors.needed())
        CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");

    CV_Assert(height > 0);
    CV_Assert(width > 0);

    const Matx44f cameraPose = _cameraPose.getMat();
#ifndef HAVE_OPENCL
    raycastTsdfVolumeUnit(settings, cameraPose, height, width, intr, volume, _points, _normals);
#else
    if (useGPU)
        ocl_raycastTsdfVolumeUnit(settings, cameraPose, height, width, intr, gpu_volume, _points, _normals);
    else
        raycastTsdfVolumeUnit(settings, cameraPose, height, width, intr, cpu_volume, _points, _normals);
#endif
}

void TsdfVolume::fetchNormals(InputArray points, OutputArray normals) const
{
#ifndef HAVE_OPENCL
    fetchNormalsFromTsdfVolumeUnit(settings, volume, points, normals);
#else
    if (useGPU)
        ocl_fetchNormalsFromTsdfVolumeUnit(settings, gpu_volume, points, normals);
    else
        fetchNormalsFromTsdfVolumeUnit(settings, cpu_volume, points, normals);
#endif
}

void TsdfVolume::fetchPointsNormals(OutputArray points, OutputArray normals) const
{
#ifndef HAVE_OPENCL
    fetchPointsNormalsFromTsdfVolumeUnit(settings, volume, points, normals);
#else
    if (useGPU)
        ocl_fetchPointsNormalsFromTsdfVolumeUnit(settings, gpu_volume, points, normals);
    else
        fetchPointsNormalsFromTsdfVolumeUnit(settings, cpu_volume, points, normals);
#endif
}

void TsdfVolume::fetchPointsNormalsColors(OutputArray, OutputArray, OutputArray) const
{
    CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");
}

void TsdfVolume::reset()
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENCL
    //TODO: use setTo(Scalar(0, 0))
    volume.forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
        {
            TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
            v.tsdf = floatToTsdf(0.0f); v.weight = 0;
        });
#else
    if (useGPU)
        gpu_volume.setTo(Scalar(floatToTsdf(0.0f), 0));
    else
        //TODO: use setTo(Scalar(0, 0))
        cpu_volume.forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
            {
                TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
                v.tsdf = floatToTsdf(0.0f); v.weight = 0;
            });
#endif
}
int TsdfVolume::getVisibleBlocks() const { return 1; }
size_t TsdfVolume::getTotalVolumeUnits() const { return 1; }


void TsdfVolume::getBoundingBox(OutputArray bb, int precision) const
{
    if (precision == Volume::BoundingBoxPrecision::VOXEL)
    {
        CV_Error(Error::StsNotImplemented, "Voxel mode is not implemented yet");
    }
    else
    {
        float sz = this->settings.getVoxelSize();
        Vec3f res;
        this->settings.getVolumeResolution(res);
        Vec3f volSize = res * sz;
        Vec6f(0, 0, 0, volSize[0], volSize[1], volSize[2]).copyTo(bb);
    }
}

void TsdfVolume::setEnableGrowth(bool /*v*/) { }

bool TsdfVolume::getEnableGrowth() const
{
    return false;
}


// HASH_TSDF

HashTsdfVolume::HashTsdfVolume(const VolumeSettings& _settings) :
    Volume::Impl(_settings)
{
    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const Point3i volResolution = Point3i(resolution);
    volumeUnitDegree = calcVolumeUnitDegree(volResolution);

#ifndef HAVE_OPENCL
    volUnitsData = cv::Mat(VOLUMES_SIZE, resolution[0] * resolution[1] * resolution[2], rawType<TsdfVoxel>());
    reset();
#else
    if (useGPU)
    {
        reset();
    }
    else
    {
        cpu_volUnitsData = cv::Mat(VOLUMES_SIZE, resolution[0] * resolution[1] * resolution[2], rawType<TsdfVoxel>());
        reset();
    }
#endif
}

HashTsdfVolume::~HashTsdfVolume() {}

void HashTsdfVolume::integrate(const OdometryFrame& frame, InputArray _cameraPose)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENCL
    Mat depth;
#else
    UMat depth;
#endif
    frame.getDepth(depth);
    integrate(depth, _cameraPose);
}

void HashTsdfVolume::integrate(InputArray _depth, InputArray _cameraPose)
{
#ifndef HAVE_OPENCL
    Mat depth = _depth.getMat();
#else
    UMat depth = _depth.getUMat();
#endif
    const Matx44f cameraPose = _cameraPose.getMat();
    Matx33f intr;
    settings.getCameraIntegrateIntrinsics(intr);
    Intr intrinsics(intr);
    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
#ifndef HAVE_OPENCL
        preCalculationPixNorm(depth.size(), intrinsics, pixNorms);
#else
        if (useGPU)
            ocl_preCalculationPixNorm(depth.size(), intrinsics, gpu_pixNorms);
        else
            preCalculationPixNorm(depth.size(), intrinsics, cpu_pixNorms);
#endif
    }
#ifndef HAVE_OPENCL
    integrateHashTsdfVolumeUnit(settings, cameraPose, lastVolIndex, lastFrameId, volumeUnitDegree, enableGrowth, depth, pixNorms, volUnitsData, volumeUnits);
    lastFrameId++;
#else
    if (useGPU)
    {
        ocl_integrateHashTsdfVolumeUnit(settings, cameraPose, lastVolIndex, lastFrameId, bufferSizeDegree, volumeUnitDegree, enableGrowth, depth, gpu_pixNorms,
                                        lastVisibleIndices, volUnitsDataCopy, gpu_volUnitsData, hashTable, isActiveFlags);
    }
    else
    {
        integrateHashTsdfVolumeUnit(settings, cameraPose, lastVolIndex, lastFrameId, volumeUnitDegree, enableGrowth, depth,
                                    cpu_pixNorms, cpu_volUnitsData, cpu_volumeUnits);
        lastFrameId++;
    }
#endif
}

void HashTsdfVolume::integrate(InputArray, InputArray, InputArray)
{
    CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");
}


void HashTsdfVolume::raycast(InputArray cameraPose, OutputArray points, OutputArray normals, OutputArray colors)  const
{
    Matx33f intr;
    settings.getCameraRaycastIntrinsics(intr);
    raycast(cameraPose, settings.getRaycastHeight(), settings.getRaycastWidth(), intr, points, normals, colors);
}


void HashTsdfVolume::raycast(InputArray _cameraPose, int height, int width, InputArray intr, OutputArray _points, OutputArray _normals, OutputArray _colors) const
{
    if (_colors.needed())
        CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");

    const Matx44f cameraPose = _cameraPose.getMat();

#ifndef HAVE_OPENCL
    raycastHashTsdfVolumeUnit(settings, cameraPose, height, width, intr, volumeUnitDegree, volUnitsData, volumeUnits, _points, _normals);
#else
    if (useGPU)
        ocl_raycastHashTsdfVolumeUnit(settings, cameraPose, height, width, intr, volumeUnitDegree, hashTable, gpu_volUnitsData, _points, _normals);
    else
        raycastHashTsdfVolumeUnit(settings, cameraPose, height, width, intr, volumeUnitDegree, cpu_volUnitsData, cpu_volumeUnits, _points, _normals);
#endif
}

void HashTsdfVolume::fetchNormals(InputArray points, OutputArray normals) const
{
#ifndef HAVE_OPENCL
    fetchNormalsFromHashTsdfVolumeUnit(settings, volUnitsData, volumeUnits, volumeUnitDegree, points, normals);
#else
    if (useGPU)
        ocl_fetchNormalsFromHashTsdfVolumeUnit(settings, volumeUnitDegree, gpu_volUnitsData, volUnitsDataCopy, hashTable, points, normals);
    else
        fetchNormalsFromHashTsdfVolumeUnit(settings, cpu_volUnitsData, cpu_volumeUnits, volumeUnitDegree, points, normals);

#endif
}
void HashTsdfVolume::fetchPointsNormals(OutputArray points, OutputArray normals) const
{
#ifndef HAVE_OPENCL
    fetchPointsNormalsFromHashTsdfVolumeUnit(settings, volUnitsData, volumeUnits, volumeUnitDegree, points, normals);
#else
    if (useGPU)
        ocl_fetchPointsNormalsFromHashTsdfVolumeUnit(settings, volumeUnitDegree, gpu_volUnitsData, volUnitsDataCopy, hashTable, points, normals);
    else
        fetchPointsNormalsFromHashTsdfVolumeUnit(settings, cpu_volUnitsData, cpu_volumeUnits, volumeUnitDegree, points, normals);
#endif
}

void HashTsdfVolume::fetchPointsNormalsColors(OutputArray, OutputArray, OutputArray) const
{
    CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");
};

void HashTsdfVolume::reset()
{
    CV_TRACE_FUNCTION();
    lastVolIndex = 0;
    lastFrameId = 0;
    enableGrowth = true;
#ifndef HAVE_OPENCL
    volUnitsData.forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
        {
            TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
            v.tsdf = floatToTsdf(0.0f); v.weight = 0;
        });
    volumeUnits = VolumeUnitIndexes();
#else
    if (useGPU)
    {
        Vec3i resolution;
        settings.getVolumeResolution(resolution);

        bufferSizeDegree = 15;
        int buff_lvl = (int)(1 << bufferSizeDegree);
        int volCubed = resolution[0] * resolution[1] * resolution[2];

        volUnitsDataCopy = cv::Mat(buff_lvl, volCubed, rawType<TsdfVoxel>());
        gpu_volUnitsData = cv::UMat(buff_lvl, volCubed, CV_8UC2);
        lastVisibleIndices = cv::UMat(buff_lvl, 1, CV_32S);
        isActiveFlags = cv::UMat(buff_lvl, 1, CV_8U);
        hashTable = CustomHashSet();
        frameParams = Vec6f();
        gpu_pixNorms = UMat();
    }
    else
    {
        cpu_volUnitsData.forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
            {
                TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
                v.tsdf = floatToTsdf(0.0f); v.weight = 0;
            });
        cpu_volumeUnits = VolumeUnitIndexes();
    }
#endif
}

int HashTsdfVolume::getVisibleBlocks() const { return 1; }
size_t HashTsdfVolume::getTotalVolumeUnits() const { return 1; }

void HashTsdfVolume::setEnableGrowth(bool v)
{
    enableGrowth = v;
}

bool HashTsdfVolume::getEnableGrowth() const
{
    return enableGrowth;
}

void HashTsdfVolume::getBoundingBox(OutputArray boundingBox, int precision) const
{
    if (precision == Volume::BoundingBoxPrecision::VOXEL)
    {
        CV_Error(Error::StsNotImplemented, "Voxel mode is not implemented yet");
    }
    else
    {
        Vec3i res;
        this->settings.getVolumeResolution(res);
        float voxelSize = this->settings.getVoxelSize();
        float side = res[0] * voxelSize;

        std::vector<Vec3i> vi;
#ifndef HAVE_OPENCL
        for (const auto& keyvalue : volumeUnits)
        {
            vi.push_back(keyvalue.first);
        }
#else
        if (useGPU)
        {
            for (int row = 0; row < hashTable.last; row++)
            {
                Vec4i idx4 = hashTable.data[row];
                vi.push_back(Vec3i(idx4[0], idx4[1], idx4[2]));
            }
        }
        else
        {
            for (const auto& keyvalue : cpu_volumeUnits)
            {
                vi.push_back(keyvalue.first);
            }
        }
#endif

        if (vi.empty())
        {
            boundingBox.setZero();
        }
        else
        {
            std::vector<Point3f> pts;
            for (Vec3i idx : vi)
            {
                Point3f base = Point3f((float)idx[0], (float)idx[1], (float)idx[2]) * side;
                pts.push_back(base);
                pts.push_back(base + Point3f(side, 0, 0));
                pts.push_back(base + Point3f(0, side, 0));
                pts.push_back(base + Point3f(0, 0, side));
                pts.push_back(base + Point3f(side, side, 0));
                pts.push_back(base + Point3f(side, 0, side));
                pts.push_back(base + Point3f(0, side, side));
                pts.push_back(base + Point3f(side, side, side));
            }

            const float mval = std::numeric_limits<float>::max();
            Vec6f bb(mval, mval, mval, -mval, -mval, -mval);
            for (auto p : pts)
            {
                // pt in local coords
                Point3f pg = p;
                bb[0] = min(bb[0], pg.x);
                bb[1] = min(bb[1], pg.y);
                bb[2] = min(bb[2], pg.z);
                bb[3] = max(bb[3], pg.x);
                bb[4] = max(bb[4], pg.y);
                bb[5] = max(bb[5], pg.z);
            }

            bb.copyTo(boundingBox);
        }
    }
}

// COLOR_TSDF

ColorTsdfVolume::ColorTsdfVolume(const VolumeSettings& _settings) :
    Volume::Impl(_settings)
{
    Vec3i volResolution;
    settings.getVolumeResolution(volResolution);
    volume = Mat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<RGBTsdfVoxel>());
    reset();
}

ColorTsdfVolume::~ColorTsdfVolume() {}

void ColorTsdfVolume::integrate(const OdometryFrame& frame, InputArray cameraPose)
{
    CV_TRACE_FUNCTION();
    Mat depth;
    frame.getDepth(depth);
    Mat rgb;
    frame.getImage(rgb);

    integrate(depth, rgb, cameraPose);
}

void ColorTsdfVolume::integrate(InputArray, InputArray)
{
    CV_Error(cv::Error::StsBadFunc, "Color data should be passed for this volume type");
}

void ColorTsdfVolume::integrate(InputArray _depth, InputArray _image, InputArray _cameraPose)
{
    Mat depth = _depth.getMat();
    Colors image = _image.getMat();
    const Matx44f cameraPose = _cameraPose.getMat();
    Matx33f intr;
    settings.getCameraIntegrateIntrinsics(intr);
    Intr intrinsics(intr);
    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
        preCalculationPixNorm(depth.size(), intrinsics, pixNorms);
    }
    integrateColorTsdfVolumeUnit(settings, cameraPose, depth, image, pixNorms, volume);
}

void ColorTsdfVolume::raycast(InputArray cameraPose, OutputArray points, OutputArray normals, OutputArray colors)  const
{
    Matx33f intr;
    settings.getCameraRaycastIntrinsics(intr);
    raycast(cameraPose, settings.getRaycastHeight(), settings.getRaycastWidth(), intr, points, normals, colors);
}

void ColorTsdfVolume::raycast(InputArray _cameraPose, int height, int width, InputArray intr, OutputArray _points, OutputArray _normals, OutputArray _colors) const
{
    const Matx44f cameraPose = _cameraPose.getMat();
    raycastColorTsdfVolumeUnit(settings, cameraPose, height, width, intr, volume, _points, _normals, _colors);
}

void ColorTsdfVolume::fetchNormals(InputArray points, OutputArray normals) const
{
    fetchNormalsFromColorTsdfVolumeUnit(settings, volume, points, normals);
}

void ColorTsdfVolume::fetchPointsNormals(OutputArray points, OutputArray normals) const
{
    fetchPointsNormalsFromColorTsdfVolumeUnit(settings, volume, points, normals);
}

void ColorTsdfVolume::fetchPointsNormalsColors(OutputArray points, OutputArray normals, OutputArray colors) const
{
    fetchPointsNormalsColorsFromColorTsdfVolumeUnit(settings, volume, points, normals, colors);
}

void ColorTsdfVolume::reset()
{
    CV_TRACE_FUNCTION();

    volume.forEach<VecRGBTsdfVoxel>([](VecRGBTsdfVoxel& vv, const int* /* position */)
        {
            RGBTsdfVoxel& v = reinterpret_cast<RGBTsdfVoxel&>(vv);
            v.tsdf = floatToTsdf(0.0f); v.weight = 0;
            v.r = v.g = v.b = 0;
        });
}

int ColorTsdfVolume::getVisibleBlocks() const { return 1; }
size_t ColorTsdfVolume::getTotalVolumeUnits() const { return 1; }

void ColorTsdfVolume::getBoundingBox(OutputArray bb, int precision) const
{
    if (precision == Volume::BoundingBoxPrecision::VOXEL)
    {
        CV_Error(Error::StsNotImplemented, "Voxel mode is not implemented yet");
    }
    else
    {
        float sz = this->settings.getVoxelSize();
        Vec3f res;
        this->settings.getVolumeResolution(res);
        Vec3f volSize = res * sz;
        Vec6f(0, 0, 0, volSize[0], volSize[1], volSize[2]).copyTo(bb);
    }
}

void ColorTsdfVolume::setEnableGrowth(bool /*v*/) { }

bool ColorTsdfVolume::getEnableGrowth() const
{
    return false;
}

// COLOR_HASH_TSDF

ColorHashTsdfVolume::ColorHashTsdfVolume(const VolumeSettings& _settings) :
    ColorTsdfVolume(_settings),
    lastVolIndex(0),
    lastFrameId(0),
    volumeUnitDegree(0),
    enableGrowth(true)
{
    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const Point3i volResolution = Point3i(resolution);
    volumeUnitDegree = calcVolumeUnitDegree(volResolution);

#ifndef HAVE_OPENCL
    volUnitsData = cv::Mat(VOLUMES_SIZE, resolution[0] * resolution[1] * resolution[2], rawType<ColorHashTsdfVoxel>());
    reset();
#else
    if (useGPU)
    {
        reset();
    }
    else
    {
        cpu_volUnitsData = cv::Mat(VOLUMES_SIZE, resolution[0] * resolution[1] * resolution[2], rawType<ColorHashTsdfVoxel>());
        reset();
    }
#endif
}

ColorHashTsdfVolume::~ColorHashTsdfVolume() {}

void ColorHashTsdfVolume::integrate(const OdometryFrame& frame, InputArray _cameraPose)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENCL
    Mat depth, image;
#else
    UMat depth, image;
#endif
    frame.getDepth(depth);
    frame.getImage(image);
    integrate(depth, image, _cameraPose);
}

void ColorHashTsdfVolume::integrate(InputArray /*_depth*/, InputArray /*_cameraPose*/)
{
    CV_Error(cv::Error::StsBadFunc, "Color data should be passed for this volume type");
}

void ColorHashTsdfVolume::integrate(InputArray _depth, InputArray _image, InputArray _cameraPose)
{
#ifndef HAVE_OPENCL
    Mat depth = _depth.getMat();
    Mat image = _image.getMat();
#else
    UMat depth = _depth.getUMat();
    UMat image = _image.getUMat();
#endif
    const Matx44f cameraPose = _cameraPose.getMat();
    Matx33f intr;
    settings.getCameraIntegrateIntrinsics(intr);
    Intr intrinsics(intr);
    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
#ifndef HAVE_OPENCL
        preCalculationPixNorm(depth.size(), intrinsics, pixNorms);
#else
        if (useGPU)
            ocl_preCalculationPixNorm(depth.size(), intrinsics, gpu_pixNorms);
        else
            preCalculationPixNorm(depth.size(), intrinsics, cpu_pixNorms);
#endif
    }
#ifndef HAVE_OPENCL
    integrateColorHashTsdfVolumeUnit(settings, cameraPose, lastVolIndex, lastFrameId, volumeUnitDegree, enableGrowth, depth, image, pixNorms, volUnitsData, volumeUnits);
    lastFrameId++;
#else
    if (useGPU)
    {
        VolumeUnitIndexes volumeUnits; 
        ocl_integrateColorHashTsdfVolumeUnit(
            settings, cameraPose, lastVolIndex, lastFrameId,
            volumeUnitDegree, enableGrowth,
            depth, image, gpu_pixNorms, gpu_volUnitsData, volumeUnits);
    }
    else
    {
        integrateColorHashTsdfVolumeUnit(settings, cameraPose, lastVolIndex, lastFrameId, volumeUnitDegree, enableGrowth, depth, image,
                                    cpu_pixNorms, cpu_volUnitsData, cpu_volumeUnits);
        lastFrameId++;
    }
#endif
}

void ColorHashTsdfVolume::raycast(InputArray cameraPose, OutputArray points, OutputArray normals, OutputArray colors)  const
{
    Matx33f intr;
    settings.getCameraRaycastIntrinsics(intr);
    raycast(cameraPose, settings.getRaycastHeight(), settings.getRaycastWidth(), intr, points, normals, colors);
}

void ColorHashTsdfVolume::raycast(InputArray _cameraPose, int height, int width, InputArray intr, OutputArray _points, OutputArray _normals, OutputArray _colors) const
{
    const Matx44f cameraPose = _cameraPose.getMat();

#ifndef HAVE_OPENCL
    raycastColorHashTsdfVolumeUnit(settings, cameraPose, height, width, intr, volumeUnitDegree, volUnitsData, volumeUnits, _points, _normals, _colors);
#else
    if (useGPU)
        ocl_raycastColorHashTsdfVolumeUnit(settings, cameraPose, height, width, intr, volumeUnitDegree, gpu_volUnitsData, hashTable, _points, _normals, _colors);
    else
        raycastColorHashTsdfVolumeUnit(settings, cameraPose, height, width, intr, volumeUnitDegree, cpu_volUnitsData, hashTable, _points, _normals, _colors);
#endif
}

void ColorHashTsdfVolume::fetchNormals(InputArray points, OutputArray normals) const
{
    if(points.empty())
    {
        normals.release();
        return;
    }
    
    fetchPointsNormalsColors(OutputArray(points.getMat()), normals, noArray());
}

void ColorHashTsdfVolume::fetchPointsNormals(OutputArray points, OutputArray normals) const
{
    // This is valid, it just fetches points and normals, ignoring colors
    fetchPointsNormalsColors(points, normals, noArray());
}

void ColorHashTsdfVolume::fetchPointsNormalsColors(OutputArray points, OutputArray normals, OutputArray colors) const
{
#ifndef HAVE_OPENCL
    fetchPointsNormalsColorsFromColorHashTsdfVolumeUnit(settings, volUnitsData, volumeUnits, volumeUnitDegree, points, normals, colors);
#else
    if (useGPU)
        ocl_fetchPointsNormalsColorsFromColorHashTsdfVolumeUnit(settings, gpu_volUnitsData, hashTable, volumeUnitDegree, points, normals, colors);
    else
        fetchPointsNormalsColorsFromColorHashTsdfVolumeUnit(settings, cpu_volUnitsData, hashTable, volumeUnitDegree, points, normals, colors);
#endif
}

void ColorHashTsdfVolume::reset()
{
    CV_TRACE_FUNCTION();
    lastVolIndex = 0;
    lastFrameId = 0;
    enableGrowth = true;
#ifndef HAVE_OPENCL
    volUnitsData.forEach<VecRGBTsdfVoxel>([](VecRGBTsdfVoxel& vv, const int* /* position */)
        {
            ColorHashTsdfVoxel& v = reinterpret_cast<ColorHashTsdfVoxel&>(vv);
            v.tsdf = floatToTsdf(0.0f); v.weight = 0;
            v.r = v.g = v.b = 0;
        });
    volumeUnits = VolumeUnitIndexes();
#else
    if (useGPU)
    {
        Vec3i resolution;
        settings.getVolumeResolution(resolution);

        bufferSizeDegree = 15;
        int buff_lvl = (int)(1 << bufferSizeDegree);
        int volCubed = resolution[0] * resolution[1] * resolution[2];

        volUnitsDataCopy = cv::Mat(buff_lvl, volCubed, rawType<ColorHashTsdfVoxel>());
        gpu_volUnitsData = cv::UMat(buff_lvl, volCubed, rawType<ColorHashTsdfVoxel>());
        lastVisibleIndices = cv::UMat(buff_lvl, 1, CV_32S);
        isActiveFlags = cv::UMat(buff_lvl, 1, CV_8U);
        hashTable = CustomHashSet();
        frameParams = Vec6f();
        gpu_pixNorms = UMat();
    }
    else
    {
        cpu_volUnitsData.forEach<VecRGBTsdfVoxel>([](VecRGBTsdfVoxel& vv, const int* /* position */)
            {
                ColorHashTsdfVoxel& v = reinterpret_cast<ColorHashTsdfVoxel&>(vv);
                v.tsdf = floatToTsdf(0.0f); v.weight = 0;
                v.r = v.g = v.b = 0;
            });
        cpu_volumeUnits = VolumeUnitIndexes();
    }
#endif
}

int ColorHashTsdfVolume::getVisibleBlocks() const { return 1; }
size_t ColorHashTsdfVolume::getTotalVolumeUnits() const { return 1; }

void ColorHashTsdfVolume::setEnableGrowth(bool v)
{
    enableGrowth = v;
}

bool ColorHashTsdfVolume::getEnableGrowth() const
{
    return enableGrowth;
}

void ColorHashTsdfVolume::getBoundingBox(OutputArray boundingBox, int precision) const
{
    // Reusing HashTsdfVolume's implementation
    if (precision == Volume::BoundingBoxPrecision::VOXEL)
    {
        CV_Error(Error::StsNotImplemented, "Voxel mode is not implemented yet");
    }
    else
    {
        Vec3i res;
        this->settings.getVolumeResolution(res);
        float voxelSize = this->settings.getVoxelSize();
        float side = res[0] * voxelSize;

        std::vector<Vec3i> vi;
#ifndef HAVE_OPENCL
        for (const auto& keyvalue : volumeUnits)
        {
            vi.push_back(keyvalue.first);
        }
#else
        if (useGPU)
        {
            for (int row = 0; row < hashTable.last; row++)
            {
                Vec4i idx4 = hashTable.data[row];
                vi.push_back(Vec3i(idx4[0], idx4[1], idx4[2]));
            }
        }
        else
        {
            for (const auto& keyvalue : cpu_volumeUnits)
            {
                vi.push_back(keyvalue.first);
            }
        }
#endif

        if (vi.empty())
        {
            boundingBox.setZero();
        }
        else
        {
            std::vector<Point3f> pts;
            for (Vec3i idx : vi)
            {
                Point3f base = Point3f((float)idx[0], (float)idx[1], (float)idx[2]) * side;
                pts.push_back(base);
                pts.push_back(base + Point3f(side, 0, 0));
                pts.push_back(base + Point3f(0, side, 0));
                pts.push_back(base + Point3f(0, 0, side));
                pts.push_back(base + Point3f(side, side, 0));
                pts.push_back(base + Point3f(side, 0, side));
                pts.push_back(base + Point3f(0, side, side));
                pts.push_back(base + Point3f(side, side, side));
            }

            const float mval = std::numeric_limits<float>::max();
            Vec6f bb(mval, mval, mval, -mval, -mval, -mval);
            for (auto p : pts)
            {
                Point3f pg = p;
                bb[0] = min(bb[0], pg.x);
                bb[1] = min(bb[1], pg.y);
                bb[2] = min(bb[2], pg.z);
                bb[3] = max(bb[3], pg.x);
                bb[4] = max(bb[4], pg.y);
                bb[5] = max(bb[5], pg.z);
            }

            bb.copyTo(boundingBox);
        }
    }
}

}
