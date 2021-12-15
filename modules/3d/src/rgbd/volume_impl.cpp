// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include <iostream>
#include "volume_impl.hpp"
#include "tsdf_functions.hpp"
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
        volume = UMat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<TsdfVoxel>());
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
            ocl_preCalculationPixNorm(depth.size(), intrinsics, pixNorms);
        else
            preCalculationPixNorm(depth.size(), intrinsics, cpu_pixNorms);
#endif
    }
    const Matx44f cameraPose = _cameraPose.getMat();

#ifndef HAVE_OPENCL
    integrateVolumeUnit(settings, cameraPose, depth, pixNorms, volume);
#else
    if (ocl::useOpenCL())
        ocl_integrateVolumeUnit(settings, cameraPose, depth, pixNorms, volume);
    else
        integrateVolumeUnit(settings, cameraPose, depth, cpu_pixNorms, cpu_volume);
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
    raycastVolumeUnit(settings, cameraPose, height, width, volume, _points, _normals);
#else
    if (ocl::useOpenCL())
        ocl_raycastVolumeUnit(settings, cameraPose, height, width, volume, _points, _normals);
    else
        raycastVolumeUnit(settings, cameraPose, height, width, cpu_volume, _points, _normals);
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
        ocl_fetchNormalsFromTsdfVolumeUnit(settings, volume, points, normals);
    else
        fetchNormalsFromTsdfVolumeUnit(settings, cpu_volume, points, normals);
#endif
}

void TsdfVolume::fetchPointsNormals(OutputArray points, OutputArray normals) const {}
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
        volume.setTo(Scalar(0, 0));
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
{ }
HashTsdfVolume::~HashTsdfVolume() {}

void HashTsdfVolume::integrate(const OdometryFrame& frame, InputArray pose) { std::cout << "HashTsdfVolume::integrate()" << std::endl; }
void HashTsdfVolume::integrate(InputArray depth, InputArray pose) { std::cout << "HashTsdfVolume::integrate()" << std::endl; }
void HashTsdfVolume::integrate(InputArray depth, InputArray image, InputArray pose) {}
void HashTsdfVolume::raycast(InputArray cameraPose, int height, int width, OdometryFrame& outFrame) const { std::cout << "HashTsdfVolume::raycast()" << std::endl; }
void HashTsdfVolume::raycast(InputArray _cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const { std::cout << "HashTsdfVolume::raycast()" << std::endl; }
void HashTsdfVolume::raycast(InputArray cameraPose, int height, int width, OutputArray _points, OutputArray _normals, OutputArray _colors) const {}

void HashTsdfVolume::fetchNormals(InputArray points, OutputArray normals) const {}
void HashTsdfVolume::fetchPointsNormals(OutputArray points, OutputArray normals) const {}
void HashTsdfVolume::fetchPointsNormalsColors(OutputArray points, OutputArray normals, OutputArray colors) const {};

void HashTsdfVolume::reset() {}
int HashTsdfVolume::getVisibleBlocks() const { return 1; }
size_t HashTsdfVolume::getTotalVolumeUnits() const { return 1; }

// COLOR_TSDF

ColorTsdfVolume::ColorTsdfVolume(const VolumeSettings& settings) :
    Volume::Impl(settings)
{}
ColorTsdfVolume::~ColorTsdfVolume() {}

void ColorTsdfVolume::integrate(const OdometryFrame& frame, InputArray pose) { std::cout << "ColorTsdfVolume::integrate()" << std::endl; }
void ColorTsdfVolume::integrate(InputArray depth, InputArray pose) { std::cout << "ColorTsdfVolume::integrate()" << std::endl; }
void ColorTsdfVolume::integrate(InputArray depth, InputArray image, InputArray pose) {}
void ColorTsdfVolume::raycast(InputArray cameraPose, int height, int width, OdometryFrame& outFrame) const { std::cout << "ColorTsdfVolume::raycast()" << std::endl; }
void ColorTsdfVolume::raycast(InputArray _cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const { std::cout << "ColorTsdfVolume::raycast()" << std::endl; }
void ColorTsdfVolume::raycast(InputArray cameraPose, int height, int width, OutputArray _points, OutputArray _normals, OutputArray _colors) const {}

void ColorTsdfVolume::fetchNormals(InputArray points, OutputArray normals) const {}
void ColorTsdfVolume::fetchPointsNormals(OutputArray points, OutputArray normals) const {}
void ColorTsdfVolume::fetchPointsNormalsColors(OutputArray points, OutputArray normals, OutputArray colors) const {};

void ColorTsdfVolume::reset() {}
int ColorTsdfVolume::getVisibleBlocks() const { return 1; }
size_t ColorTsdfVolume::getTotalVolumeUnits() const { return 1; }

}
