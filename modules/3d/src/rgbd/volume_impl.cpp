// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include <iostream>
#include "volume_impl.hpp"
#include "tsdf_functions.hpp"
#include "hash_tsdf_functions.hpp"
#include "color_tsdf_functions.hpp"
#include "opencv2/imgproc.hpp"

namespace cv
{

Volume::Impl::Impl(const VolumeSettings& _settings) :
    settings(_settings)
{}

// TSDF

TsdfVolume::TsdfVolume(const VolumeSettings& settings) :
    Volume::Impl(settings)
{
    std::cout << "TsdfVolume::TsdfVolume()" << std::endl;

    Vec3i volResolution;
    settings.getVolumeResolution(volResolution);
#ifndef HAVE_OPENCL
    volume = Mat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<TsdfVoxel>());
#else
    if (ocl::useOpenCL())
        gpu_volume = UMat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<TsdfVoxel>());
    else
        cpu_volume = Mat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<TsdfVoxel>());
#endif

    reset();
}
TsdfVolume::~TsdfVolume() {}

void TsdfVolume::integrate(const OdometryFrame& frame, InputArray _cameraPose)
{
    std::cout << "TsdfVolume::integrate(OdometryFrame)" << std::endl;

    CV_TRACE_FUNCTION();
    Depth depth;
    frame.getDepth(depth);
    // dependence from OdometryFrame
    depth = depth * settings.getDepthFactor();

    integrate(depth, _cameraPose);
}

void TsdfVolume::integrate(InputArray _depth, InputArray _cameraPose)
{
    std::cout << "TsdfVolume::integrate(Mat)" << std::endl;

    CV_TRACE_FUNCTION();
    Depth depth = _depth.getMat();
    CV_Assert(depth.type() == DEPTH_TYPE);
    CV_Assert(!depth.empty());

    Matx33f intr;
    settings.getCameraIntrinsics(intr);
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
        if (ocl::useOpenCL())
            ocl_preCalculationPixNorm(depth.size(), intrinsics, gpu_pixNorms);
        else
            preCalculationPixNorm(depth.size(), intrinsics, cpu_pixNorms);
#endif
    }
    const Matx44f cameraPose = _cameraPose.getMat();

#ifndef HAVE_OPENCL
    integrateTsdfVolumeUnit(settings, cameraPose, depth, pixNorms, volume);
#else
    if (ocl::useOpenCL())
        ocl_integrateTsdfVolumeUnit(settings, cameraPose, depth, gpu_pixNorms, gpu_volume);
    else
        integrateTsdfVolumeUnit(settings, cameraPose, depth, cpu_pixNorms, cpu_volume);
#endif
}
void TsdfVolume::integrate(InputArray depth, InputArray image, InputArray pose)
{
    CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");
}


void TsdfVolume::raycast(InputArray cameraPose, int height, int width, OdometryFrame& outFrame) const
{
    std::cout << "TsdfVolume::raycast(OdometryFrame)" << std::endl;
#ifndef HAVE_OPENCL
    Mat points, normals;
    raycast(cameraPose, height, width, points, normals);
    outFrame.setPyramidLevel(1, OdometryFramePyramidType::PYR_CLOUD);
    outFrame.setPyramidLevel(1, OdometryFramePyramidType::PYR_NORM);
    outFrame.setPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
    outFrame.setPyramidAt(normals, OdometryFramePyramidType::PYR_NORM, 0);
#else
    if (ocl::useOpenCL())
    {
        UMat points, normals;
        raycast(cameraPose, height, width, points, normals);
        outFrame.setPyramidLevel(1, OdometryFramePyramidType::PYR_CLOUD);
        outFrame.setPyramidLevel(1, OdometryFramePyramidType::PYR_NORM);
        outFrame.setPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
        outFrame.setPyramidAt(normals, OdometryFramePyramidType::PYR_NORM, 0);
    }
    else
    {
        Mat points, normals;
        raycast(cameraPose, height, width, points, normals);
        outFrame.setPyramidLevel(1, OdometryFramePyramidType::PYR_CLOUD);
        outFrame.setPyramidLevel(1, OdometryFramePyramidType::PYR_NORM);
        outFrame.setPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
        outFrame.setPyramidAt(normals, OdometryFramePyramidType::PYR_NORM, 0);
    }
#endif
}


void TsdfVolume::raycast(InputArray _cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const
{
    std::cout << "TsdfVolume::raycast(Mat)" << std::endl;

    CV_Assert(height > 0);
    CV_Assert(width > 0);

    const Matx44f cameraPose = _cameraPose.getMat();
#ifndef HAVE_OPENCL
    raycastTsdfVolumeUnit(settings, cameraPose, height, width, volume, _points, _normals);
#else
    if (ocl::useOpenCL())
        ocl_raycastTsdfVolumeUnit(settings, cameraPose, height, width, gpu_volume, _points, _normals);
    else
        raycastTsdfVolumeUnit(settings, cameraPose, height, width, cpu_volume, _points, _normals);
#endif
}

void TsdfVolume::raycast(InputArray cameraPose, int height, int width, OutputArray _points, OutputArray _normals, OutputArray _colors) const
{
    CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");
}

void TsdfVolume::fetchNormals(InputArray points, OutputArray normals) const
{
    std::cout << "TsdfVolume::fetchNormals(Mat)" << std::endl;
#ifndef HAVE_OPENCL
    fetchNormalsFromTsdfVolumeUnit(settings, volume, points, normals);
#else
    if (ocl::useOpenCL())
        ocl_fetchNormalsFromTsdfVolumeUnit(settings, gpu_volume, points, normals);
    else
        fetchNormalsFromTsdfVolumeUnit(settings, cpu_volume, points, normals);
#endif
}

void TsdfVolume::fetchPointsNormals(OutputArray points, OutputArray normals) const
{
    std::cout << "TsdfVolume::fetchPointsNormals(Mat)" << std::endl;
#ifndef HAVE_OPENCL
    fetchPointsNormalsFromTsdfVolumeUnit(settings, volume, points, normals);
#else
    if (ocl::useOpenCL())
        ocl_fetchPointsNormalsFromTsdfVolumeUnit(settings, gpu_volume, points, normals);
    else
        fetchPointsNormalsFromTsdfVolumeUnit(settings, cpu_volume, points, normals);

#endif
}

void TsdfVolume::fetchPointsNormalsColors(OutputArray points, OutputArray normals, OutputArray colors) const
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
    if (ocl::useOpenCL())
        gpu_volume.setTo(Scalar(0, 0));
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



// HASH_TSDF

HashTsdfVolume::HashTsdfVolume(const VolumeSettings& settings) :
    Volume::Impl(settings)
{
#ifndef HAVE_OPENCL
    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    volUnitsData = cv::Mat(VOLUMES_SIZE, resolution[0] * resolution[1] * resolution[2], rawType<TsdfVoxel>());
    reset();
#else
    if (ocl::useOpenCL())
    {
        reset();
    }
    else
    {
        Vec3i resolution;
        settings.getVolumeResolution(resolution);
        cpu_volUnitsData = cv::Mat(VOLUMES_SIZE, resolution[0] * resolution[1] * resolution[2], rawType<TsdfVoxel>());
        reset();
    }
#endif
}

HashTsdfVolume::~HashTsdfVolume() {}

void HashTsdfVolume::integrate(const OdometryFrame& frame, InputArray _cameraPose)
{
    std::cout << "HashTsdfVolume::integrate(OdometryFrame)" << std::endl;
    CV_TRACE_FUNCTION();
    Depth depth;
    frame.getDepth(depth);
    // dependence from OdometryFrame
    depth = depth * settings.getDepthFactor();

    integrate(depth, _cameraPose);
}

void HashTsdfVolume::integrate(InputArray _depth, InputArray _cameraPose)
{
    std::cout << "HashTsdfVolume::integrate(Mat)" << std::endl;
    Depth depth = _depth.getMat();
    const Matx44f cameraPose = _cameraPose.getMat();
    Matx33f intr;
    settings.getCameraIntrinsics(intr);
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
        if (ocl::useOpenCL())
            ocl_preCalculationPixNorm(depth.size(), intrinsics, gpu_pixNorms);
        else
            preCalculationPixNorm(depth.size(), intrinsics, cpu_pixNorms);
#endif
    }
#ifndef HAVE_OPENCL
    integrateHashTsdfVolumeUnit(settings, cameraPose, lastVolIndex, lastFrameId, depth, pixNorms, volUnitsData, volumeUnits);
    lastFrameId++;
#else
    if (ocl::useOpenCL())
    {
        ocl_integrateHashTsdfVolumeUnit(settings, cameraPose, lastVolIndex, lastFrameId, bufferSizeDegree, depth, gpu_pixNorms, lastVisibleIndices, volUnitsDataCopy, gpu_volUnitsData, hashTable, isActiveFlags);
    }
    else
    {
        integrateHashTsdfVolumeUnit(settings, cameraPose, lastVolIndex, lastFrameId, depth, cpu_pixNorms, cpu_volUnitsData, cpu_volumeUnits);
        lastFrameId++;
    }
#endif
}

void HashTsdfVolume::integrate(InputArray depth, InputArray image, InputArray pose)
{
    CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");
}


void HashTsdfVolume::raycast(InputArray cameraPose, int height, int width, OdometryFrame& outFrame) const
{
    std::cout << "HashTsdfVolume::raycast()" << std::endl;
#ifndef HAVE_OPENCL
    Mat points, normals;
    raycast(cameraPose, height, width, points, normals);
    outFrame.setPyramidLevel(1, OdometryFramePyramidType::PYR_CLOUD);
    outFrame.setPyramidLevel(1, OdometryFramePyramidType::PYR_NORM);
    outFrame.setPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
    outFrame.setPyramidAt(normals, OdometryFramePyramidType::PYR_NORM, 0);
#else
    if (ocl::useOpenCL())
    {
        UMat points, normals;
        raycast(cameraPose, height, width, points, normals);
        outFrame.setPyramidLevel(1, OdometryFramePyramidType::PYR_CLOUD);
        outFrame.setPyramidLevel(1, OdometryFramePyramidType::PYR_NORM);
        outFrame.setPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
        outFrame.setPyramidAt(normals, OdometryFramePyramidType::PYR_NORM, 0);
    }
    else
    {
        Mat points, normals;
        raycast(cameraPose, height, width, points, normals);
        outFrame.setPyramidLevel(1, OdometryFramePyramidType::PYR_CLOUD);
        outFrame.setPyramidLevel(1, OdometryFramePyramidType::PYR_NORM);
        outFrame.setPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
        outFrame.setPyramidAt(normals, OdometryFramePyramidType::PYR_NORM, 0);

    }
#endif
}
void HashTsdfVolume::raycast(InputArray _cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const
{
    std::cout << "HashTsdfVolume::raycast()" << std::endl;
    const Matx44f cameraPose = _cameraPose.getMat();

#ifndef HAVE_OPENCL
    raycastHashTsdfVolumeUnit(settings, cameraPose, height, width, volUnitsData, volumeUnits, _points, _normals);
#else
    if (ocl::useOpenCL())
        ocl_raycastHashTsdfVolumeUnit(settings, cameraPose, height, width, hashTable, gpu_volUnitsData, _points, _normals);
    else
        raycastHashTsdfVolumeUnit(settings, cameraPose, height, width, cpu_volUnitsData, cpu_volumeUnits, _points, _normals);
#endif
}
void HashTsdfVolume::raycast(InputArray cameraPose, int height, int width, OutputArray _points, OutputArray _normals, OutputArray _colors) const
{
    CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");
}

void HashTsdfVolume::fetchNormals(InputArray points, OutputArray normals) const
{
    std::cout << "HashTsdfVolume::fetchNormals()" << std::endl;
#ifndef HAVE_OPENCL
    fetchNormalsFromHashTsdfVolumeUnit(settings, volUnitsData, volumeUnits, points, normals);
#else
    if (ocl::useOpenCL())
        olc_fetchNormalsFromHashTsdfVolumeUnit(settings, gpu_volUnitsData, volUnitsDataCopy, hashTable, points, normals);
    else
        fetchNormalsFromHashTsdfVolumeUnit(settings, cpu_volUnitsData, cpu_volumeUnits, points, normals);

#endif
}
void HashTsdfVolume::fetchPointsNormals(OutputArray points, OutputArray normals) const
{
    std::cout << "fetchPointsNormals()" << std::endl;
#ifndef HAVE_OPENCL
    fetchPointsNormalsFromHashTsdfVolumeUnit(settings, volUnitsData, volumeUnits, points, normals);
#else
    if (ocl::useOpenCL())
        ocl_fetchPointsNormalsFromHashTsdfVolumeUnit(settings, gpu_volUnitsData, volUnitsDataCopy, hashTable, points, normals);
    else
        fetchPointsNormalsFromHashTsdfVolumeUnit(settings, cpu_volUnitsData, cpu_volumeUnits, points, normals);
#endif
}

void HashTsdfVolume::fetchPointsNormalsColors(OutputArray points, OutputArray normals, OutputArray colors) const {};

void HashTsdfVolume::reset()
{
    CV_TRACE_FUNCTION();
    lastVolIndex = 0;
    lastFrameId = 0;
#ifndef HAVE_OPENCL
    volUnitsData.forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
        {
            TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
            v.tsdf = floatToTsdf(0.0f); v.weight = 0;
        });
    volumeUnits = VolumeUnitIndexes();
#else
    if (ocl::useOpenCL())
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

// COLOR_TSDF

ColorTsdfVolume::ColorTsdfVolume(const VolumeSettings& settings) :
    Volume::Impl(settings)
{
    Vec3i volResolution;
    settings.getVolumeResolution(volResolution);
    volume = Mat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<RGBTsdfVoxel>());
    reset();
}

ColorTsdfVolume::~ColorTsdfVolume() {}

void ColorTsdfVolume::integrate(const OdometryFrame& frame, InputArray pose)
{
    std::cout << "ColorTsdfVolume::integrate(OdometryFrame)" << std::endl;
}

void ColorTsdfVolume::integrate(InputArray depth, InputArray pose)
{
    std::cout << "ColorTsdfVolume::integrate(Mat)" << std::endl;
    CV_Error(cv::Error::StsBadFunc, "There is no color data");
}

void ColorTsdfVolume::integrate(InputArray _depth, InputArray _image, InputArray _cameraPose)
{
    std::cout << "ColorTsdfVolume::integrate(Mat, Mat)" << std::endl;
    Depth depth = _depth.getMat();
    Colors image = _image.getMat();
    const Matx44f cameraPose = _cameraPose.getMat();
    Matx33f intr;
    settings.getCameraIntrinsics(intr);
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

void ColorTsdfVolume::raycast(InputArray cameraPose, int height, int width, OdometryFrame& outFrame) const
{
    std::cout << "ColorTsdfVolume::raycast()" << std::endl;
}

void ColorTsdfVolume::raycast(InputArray _cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const
{
    std::cout << "ColorTsdfVolume::raycast()" << std::endl;
}
void ColorTsdfVolume::raycast(InputArray _cameraPose, int height, int width, OutputArray _points, OutputArray _normals, OutputArray _colors) const
{
    std::cout << "ColorTsdfVolume::raycast()" << std::endl;

    const Matx44f cameraPose = _cameraPose.getMat();
    raycastColorTsdfVolumeUnit(settings, cameraPose, height, width, volume, _points, _normals, _colors);
}

void ColorTsdfVolume::fetchNormals(InputArray points, OutputArray normals) const
{
    std::cout << "ColorTsdfVolume::fetchNormals()" << std::endl;

    fetchNormalsFromColorTsdfVolumeUnit(settings, volume, points, normals);
}

void ColorTsdfVolume::fetchPointsNormals(OutputArray points, OutputArray normals) const
{
    std::cout << "ColorTsdfVolume::fetchPointsNormals()" << std::endl;

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
        });
}

int ColorTsdfVolume::getVisibleBlocks() const { return 1; }
size_t ColorTsdfVolume::getTotalVolumeUnits() const { return 1; }

}
