/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "opencv2/core/gpumat.hpp"
#include <iostream>

#if defined(HAVE_CUDA)
# include <cuda_runtime.h>
# include <npp.h>

# define CUDART_MINIMUM_REQUIRED_VERSION 4020
# define NPP_MINIMUM_REQUIRED_VERSION 4200

# if (CUDART_VERSION < CUDART_MINIMUM_REQUIRED_VERSION)
#  error "Insufficient Cuda Runtime library version, please update it."
# endif

# if (NPP_VERSION_MAJOR * 1000 + NPP_VERSION_MINOR * 100 + NPP_VERSION_BUILD < NPP_MINIMUM_REQUIRED_VERSION)
#  error "Insufficient NPP version, please update it."
# endif
#endif

#ifdef DYNAMIC_CUDA_SUPPORT
# include <dlfcn.h>
# include <sys/types.h>
# include <sys/stat.h>
# include <dirent.h>
#endif

#ifdef ANDROID
# ifdef LOG_TAG
#  undef LOG_TAG
# endif
# ifdef LOGE
#  undef LOGE
# endif
# ifdef LOGD
#  undef LOGD
# endif
# ifdef LOGI
#  undef LOGI
# endif

# include <android/log.h>

# define LOG_TAG "OpenCV::CUDA"
# define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))
# define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
# define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#endif

using namespace std;
using namespace cv;
using namespace cv::gpu;

#define throw_nogpu CV_Error(CV_GpuNotSupported, "The library is compiled without CUDA support")

#include "opencv2/dynamicuda/dynamicuda.hpp"

#ifdef DYNAMIC_CUDA_SUPPORT

typedef GpuFuncTable* (*GpuFactoryType)();
typedef DeviceInfoFuncTable* (*DeviceInfoFactoryType)();

static GpuFactoryType gpuFactory = NULL;
static DeviceInfoFactoryType deviceInfoFactory = NULL;

# if defined(__linux__) || defined(__APPLE__) || defined (ANDROID)

const std::string DYNAMIC_CUDA_LIB_NAME = "libopencv_dynamicuda.so";

#  ifdef ANDROID
static const std::string getCudaSupportLibName()
{
    Dl_info dl_info;
    if(0 != dladdr((void *)getCudaSupportLibName, &dl_info))
    {
        LOGD("Library name: %s", dl_info.dli_fname);
        LOGD("Library base address: %p", dl_info.dli_fbase);

        const char* libName=dl_info.dli_fname;
        while( ((*libName)=='/') || ((*libName)=='.') )
        libName++;

        char lineBuf[2048];
        FILE* file = fopen("/proc/self/smaps", "rt");

        if(file)
        {
            while (fgets(lineBuf, sizeof lineBuf, file) != NULL)
            {
                //verify that line ends with library name
                int lineLength = strlen(lineBuf);
                int libNameLength = strlen(libName);

                //trim end
                for(int i = lineLength - 1; i >= 0 && isspace(lineBuf[i]); --i)
                {
                    lineBuf[i] = 0;
                    --lineLength;
                }

                if (0 != strncmp(lineBuf + lineLength - libNameLength, libName, libNameLength))
                {
                //the line does not contain the library name
                    continue;
                }

                //extract path from smaps line
                char* pathBegin = strchr(lineBuf, '/');
                if (0 == pathBegin)
                {
                    LOGE("Strange error: could not find path beginning in lin \"%s\"", lineBuf);
                    continue;
                }

                char* pathEnd = strrchr(pathBegin, '/');
                pathEnd[1] = 0;

                LOGD("Libraries folder found: %s", pathBegin);

                fclose(file);
                return std::string(pathBegin) + DYNAMIC_CUDA_LIB_NAME;
            }
            fclose(file);
            LOGE("Could not find library path");
        }
        else
        {
            LOGE("Could not read /proc/self/smaps");
        }
    }
    else
    {
        LOGE("Could not get library name and base address");
    }

    return string();
}

#  else
static const std::string getCudaSupportLibName()
{
    return DYNAMIC_CUDA_LIB_NAME;
}
#  endif

static bool loadCudaSupportLib()
{
    void* handle;
    const std::string name = getCudaSupportLibName();
    dlerror();
    handle = dlopen(name.c_str(), RTLD_LAZY);
    if (!handle)
    {
        LOGE("Cannot dlopen %s: %s", name.c_str(), dlerror());
        return false;
    }

    deviceInfoFactory = (DeviceInfoFactoryType)dlsym(handle, "deviceInfoFactory");
    if (!deviceInfoFactory)
    {
        LOGE("Cannot dlsym deviceInfoFactory: %s", dlerror());
        dlclose(handle);
        return false;
    }

    gpuFactory = (GpuFactoryType)dlsym(handle, "gpuFactory");
    if (!gpuFactory)
    {
        LOGE("Cannot dlsym gpuFactory: %s", dlerror());
        dlclose(handle);
        return false;
    }

    return true;
}

# else
#  error "Dynamic CUDA support is not implemented for this platform!"
# endif

#endif

static GpuFuncTable* gpuFuncTable()
{
#ifdef DYNAMIC_CUDA_SUPPORT
   static EmptyFuncTable stub;
   static GpuFuncTable* libFuncTable = loadCudaSupportLib() ? gpuFactory(): (GpuFuncTable*)&stub;
   static GpuFuncTable *funcTable = libFuncTable ? libFuncTable : (GpuFuncTable*)&stub;
#else
# ifdef USE_CUDA
   static CudaFuncTable impl;
   static GpuFuncTable* funcTable = &impl;
#else
   static EmptyFuncTable stub;
   static GpuFuncTable* funcTable = &stub;
#endif
#endif
   return funcTable;
}

static DeviceInfoFuncTable* deviceInfoFuncTable()
{
#ifdef DYNAMIC_CUDA_SUPPORT
   static EmptyDeviceInfoFuncTable stub;
   static DeviceInfoFuncTable* libFuncTable = loadCudaSupportLib() ? deviceInfoFactory(): (DeviceInfoFuncTable*)&stub;
   static DeviceInfoFuncTable* funcTable = libFuncTable ? libFuncTable : (DeviceInfoFuncTable*)&stub;
#else
# ifdef USE_CUDA
   static CudaDeviceInfoFuncTable impl;
   static DeviceInfoFuncTable* funcTable = &impl;
#else
   static EmptyDeviceInfoFuncTable stub;
   static DeviceInfoFuncTable* funcTable = &stub;
#endif
#endif
   return funcTable;
}


//////////////////////////////// Initialization & Info ////////////////////////

int cv::gpu::getCudaEnabledDeviceCount() { return deviceInfoFuncTable()->getCudaEnabledDeviceCount(); }

void cv::gpu::setDevice(int device) { deviceInfoFuncTable()->setDevice(device); }
int cv::gpu::getDevice() { return deviceInfoFuncTable()->getDevice(); }

void cv::gpu::resetDevice() { deviceInfoFuncTable()->resetDevice(); }

bool cv::gpu::deviceSupports(FeatureSet feature_set) { return deviceInfoFuncTable()->deviceSupports(feature_set); }

bool cv::gpu::TargetArchs::builtWith(FeatureSet feature_set) { return deviceInfoFuncTable()->builtWith(feature_set); }
bool cv::gpu::TargetArchs::has(int major, int minor) { return deviceInfoFuncTable()->has(major, minor); }
bool cv::gpu::TargetArchs::hasPtx(int major, int minor) {  return deviceInfoFuncTable()->hasPtx(major, minor); }
bool cv::gpu::TargetArchs::hasBin(int major, int minor) { return deviceInfoFuncTable()->hasBin(major, minor);  }
bool cv::gpu::TargetArchs::hasEqualOrLessPtx(int major, int minor) { return deviceInfoFuncTable()->hasEqualOrLessPtx(major, minor); }
bool cv::gpu::TargetArchs::hasEqualOrGreater(int major, int minor) { return deviceInfoFuncTable()->hasEqualOrGreater(major, minor); }
bool cv::gpu::TargetArchs::hasEqualOrGreaterPtx(int major, int minor) { return deviceInfoFuncTable()->hasEqualOrGreaterPtx(major, minor); }
bool cv::gpu::TargetArchs::hasEqualOrGreaterBin(int major, int minor) { return deviceInfoFuncTable()->hasEqualOrGreaterBin(major, minor); }

size_t cv::gpu::DeviceInfo::sharedMemPerBlock() const { return deviceInfoFuncTable()->sharedMemPerBlock(device_id_); }
void cv::gpu::DeviceInfo::queryMemory(size_t& total_memory, size_t& free_memory) const { deviceInfoFuncTable()->queryMemory(device_id_, total_memory, free_memory); }
size_t cv::gpu::DeviceInfo::freeMemory() const { return deviceInfoFuncTable()->freeMemory(device_id_); }
size_t cv::gpu::DeviceInfo::totalMemory() const { return deviceInfoFuncTable()->totalMemory(device_id_); }
bool cv::gpu::DeviceInfo::supports(FeatureSet feature_set) const { return deviceInfoFuncTable()->supports(device_id_, feature_set); }
bool cv::gpu::DeviceInfo::isCompatible() const { return deviceInfoFuncTable()->isCompatible(device_id_); }

void cv::gpu::DeviceInfo::query()
{
    name_ = deviceInfoFuncTable()->name(device_id_);
    multi_processor_count_ = deviceInfoFuncTable()->multiProcessorCount(device_id_);
    majorVersion_ = deviceInfoFuncTable()->majorVersion(device_id_);
    minorVersion_ = deviceInfoFuncTable()->minorVersion(device_id_);
}

void cv::gpu::printCudaDeviceInfo(int device) { deviceInfoFuncTable()->printCudaDeviceInfo(device); }
void cv::gpu::printShortCudaDeviceInfo(int device) { deviceInfoFuncTable()->printShortCudaDeviceInfo(device); }

namespace cv { namespace gpu
{
    CV_EXPORTS void copyWithMask(const cv::gpu::GpuMat&, cv::gpu::GpuMat&, const cv::gpu::GpuMat&, cudaStream_t);
    CV_EXPORTS void convertTo(const cv::gpu::GpuMat&, cv::gpu::GpuMat&);
    CV_EXPORTS void convertTo(const cv::gpu::GpuMat&, cv::gpu::GpuMat&, double, double, cudaStream_t = 0);
    CV_EXPORTS void setTo(cv::gpu::GpuMat&, cv::Scalar, cudaStream_t);
    CV_EXPORTS void setTo(cv::gpu::GpuMat&, cv::Scalar, const cv::gpu::GpuMat&, cudaStream_t);
    CV_EXPORTS void setTo(cv::gpu::GpuMat&, cv::Scalar);
    CV_EXPORTS void setTo(cv::gpu::GpuMat&, cv::Scalar, const cv::gpu::GpuMat&);
}}

//////////////////////////////// GpuMat ///////////////////////////////

cv::gpu::GpuMat::GpuMat(const GpuMat& m)
    : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend)
{
    if (refcount)
        CV_XADD(refcount, 1);
}

cv::gpu::GpuMat::GpuMat(int rows_, int cols_, int type_, void* data_, size_t step_) :
    flags(Mat::MAGIC_VAL + (type_ & TYPE_MASK)), rows(rows_), cols(cols_),
    step(step_), data((uchar*)data_), refcount(0),
    datastart((uchar*)data_), dataend((uchar*)data_)
{
    size_t minstep = cols * elemSize();

    if (step == Mat::AUTO_STEP)
    {
        step = minstep;
        flags |= Mat::CONTINUOUS_FLAG;
    }
    else
    {
        if (rows == 1)
            step = minstep;

        CV_DbgAssert(step >= minstep);

        flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
    }
    dataend += step * (rows - 1) + minstep;
}

cv::gpu::GpuMat::GpuMat(Size size_, int type_, void* data_, size_t step_) :
    flags(Mat::MAGIC_VAL + (type_ & TYPE_MASK)), rows(size_.height), cols(size_.width),
    step(step_), data((uchar*)data_), refcount(0),
    datastart((uchar*)data_), dataend((uchar*)data_)
{
    size_t minstep = cols * elemSize();

    if (step == Mat::AUTO_STEP)
    {
        step = minstep;
        flags |= Mat::CONTINUOUS_FLAG;
    }
    else
    {
        if (rows == 1)
            step = minstep;

        CV_DbgAssert(step >= minstep);

        flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
    }
    dataend += step * (rows - 1) + minstep;
}

cv::gpu::GpuMat::GpuMat(const GpuMat& m, Range _rowRange, Range _colRange)
{
    flags = m.flags;
    step = m.step; refcount = m.refcount;
    data = m.data; datastart = m.datastart; dataend = m.dataend;

    if (_rowRange == Range::all())
        rows = m.rows;
    else
    {
        CV_Assert(0 <= _rowRange.start && _rowRange.start <= _rowRange.end && _rowRange.end <= m.rows);

        rows = _rowRange.size();
        data += step*_rowRange.start;
    }

    if (_colRange == Range::all())
        cols = m.cols;
    else
    {
        CV_Assert(0 <= _colRange.start && _colRange.start <= _colRange.end && _colRange.end <= m.cols);

        cols = _colRange.size();
        data += _colRange.start*elemSize();
        flags &= cols < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
    }

    if (rows == 1)
        flags |= Mat::CONTINUOUS_FLAG;

    if (refcount)
        CV_XADD(refcount, 1);

    if (rows <= 0 || cols <= 0)
        rows = cols = 0;
}

cv::gpu::GpuMat::GpuMat(const GpuMat& m, Rect roi) :
    flags(m.flags), rows(roi.height), cols(roi.width),
    step(m.step), data(m.data + roi.y*step), refcount(m.refcount),
    datastart(m.datastart), dataend(m.dataend)
{
    flags &= roi.width < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
    data += roi.x * elemSize();

    CV_Assert(0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows);

    if (refcount)
        CV_XADD(refcount, 1);

    if (rows <= 0 || cols <= 0)
        rows = cols = 0;
}

cv::gpu::GpuMat::GpuMat(const Mat& m) :
    flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{
    upload(m);
}

GpuMat& cv::gpu::GpuMat::operator = (const GpuMat& m)
{
    if (this != &m)
    {
        GpuMat temp(m);
        swap(temp);
    }

    return *this;
}

void cv::gpu::GpuMat::swap(GpuMat& b)
{
    std::swap(flags, b.flags);
    std::swap(rows, b.rows);
    std::swap(cols, b.cols);
    std::swap(step, b.step);
    std::swap(data, b.data);
    std::swap(datastart, b.datastart);
    std::swap(dataend, b.dataend);
    std::swap(refcount, b.refcount);
}

void cv::gpu::GpuMat::locateROI(Size& wholeSize, Point& ofs) const
{
    size_t esz = elemSize();
    ptrdiff_t delta1 = data - datastart;
    ptrdiff_t delta2 = dataend - datastart;

    CV_DbgAssert(step > 0);

    if (delta1 == 0)
        ofs.x = ofs.y = 0;
    else
    {
        ofs.y = static_cast<int>(delta1 / step);
        ofs.x = static_cast<int>((delta1 - step * ofs.y) / esz);

        CV_DbgAssert(data == datastart + ofs.y * step + ofs.x * esz);
    }

    size_t minstep = (ofs.x + cols) * esz;

    wholeSize.height = std::max(static_cast<int>((delta2 - minstep) / step + 1), ofs.y + rows);
    wholeSize.width = std::max(static_cast<int>((delta2 - step * (wholeSize.height - 1)) / esz), ofs.x + cols);
}

GpuMat& cv::gpu::GpuMat::adjustROI(int dtop, int dbottom, int dleft, int dright)
{
    Size wholeSize;
    Point ofs;
    locateROI(wholeSize, ofs);

    size_t esz = elemSize();

    int row1 = std::max(ofs.y - dtop, 0);
    int row2 = std::min(ofs.y + rows + dbottom, wholeSize.height);

    int col1 = std::max(ofs.x - dleft, 0);
    int col2 = std::min(ofs.x + cols + dright, wholeSize.width);

    data += (row1 - ofs.y) * step + (col1 - ofs.x) * esz;
    rows = row2 - row1;
    cols = col2 - col1;

    if (esz * cols == step || rows == 1)
        flags |= Mat::CONTINUOUS_FLAG;
    else
        flags &= ~Mat::CONTINUOUS_FLAG;

    return *this;
}

GpuMat cv::gpu::GpuMat::reshape(int new_cn, int new_rows) const
{
    GpuMat hdr = *this;

    int cn = channels();
    if (new_cn == 0)
        new_cn = cn;

    int total_width = cols * cn;

    if ((new_cn > total_width || total_width % new_cn != 0) && new_rows == 0)
        new_rows = rows * total_width / new_cn;

    if (new_rows != 0 && new_rows != rows)
    {
        int total_size = total_width * rows;

        if (!isContinuous())
            CV_Error(CV_BadStep, "The matrix is not continuous, thus its number of rows can not be changed");

        if ((unsigned)new_rows > (unsigned)total_size)
            CV_Error(CV_StsOutOfRange, "Bad new number of rows");

        total_width = total_size / new_rows;

        if (total_width * new_rows != total_size)
            CV_Error(CV_StsBadArg, "The total number of matrix elements is not divisible by the new number of rows");

        hdr.rows = new_rows;
        hdr.step = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;

    if (new_width * new_cn != total_width)
        CV_Error(CV_BadNumChannels, "The total width is not divisible by the new number of channels");

    hdr.cols = new_width;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn - 1) << CV_CN_SHIFT);

    return hdr;
}

cv::Mat::Mat(const GpuMat& m) : flags(0), dims(0), rows(0), cols(0), data(0), refcount(0), datastart(0), dataend(0), datalimit(0), allocator(0), size(&rows)
{
    m.download(*this);
}

void cv::gpu::createContinuous(int rows, int cols, int type, GpuMat& m)
{
    int area = rows * cols;
    if (m.empty() || m.type() != type || !m.isContinuous() || m.size().area() < area)
        m.create(1, area, type);

    m.cols = cols;
    m.rows = rows;
    m.step = m.elemSize() * cols;
    m.flags |= Mat::CONTINUOUS_FLAG;
}

void cv::gpu::ensureSizeIsEnough(int rows, int cols, int type, GpuMat& m)
{
    if (m.empty() || m.type() != type || m.data != m.datastart)
        m.create(rows, cols, type);
    else
    {
        const size_t esz = m.elemSize();
        const ptrdiff_t delta2 = m.dataend - m.datastart;

        const size_t minstep = m.cols * esz;

        Size wholeSize;
        wholeSize.height = std::max(static_cast<int>((delta2 - minstep) / m.step + 1), m.rows);
        wholeSize.width = std::max(static_cast<int>((delta2 - m.step * (wholeSize.height - 1)) / esz), m.cols);

        if (wholeSize.height < rows || wholeSize.width < cols)
            m.create(rows, cols, type);
        else
        {
            m.cols = cols;
            m.rows = rows;
        }
    }
}

GpuMat cv::gpu::allocMatFromBuf(int rows, int cols, int type, GpuMat &mat)
{
    if (!mat.empty() && mat.type() == type && mat.rows >= rows && mat.cols >= cols)
        return mat(Rect(0, 0, cols, rows));
    return mat = GpuMat(rows, cols, type);
}

void cv::gpu::GpuMat::upload(const Mat& m)
{
    CV_DbgAssert(!m.empty());

    create(m.size(), m.type());

    gpuFuncTable()->copy(m, *this);
}

void cv::gpu::GpuMat::download(Mat& m) const
{
    CV_DbgAssert(!empty());

    m.create(size(), type());

    gpuFuncTable()->copy(*this, m);
}

void cv::gpu::GpuMat::copyTo(GpuMat& m) const
{
    CV_DbgAssert(!empty());

    m.create(size(), type());

    gpuFuncTable()->copy(*this, m);
}

void cv::gpu::GpuMat::copyTo(GpuMat& mat, const GpuMat& mask) const
{
    if (mask.empty())
        copyTo(mat);
    else
    {
        mat.create(size(), type());

        gpuFuncTable()->copyWithMask(*this, mat, mask);
    }
}

void cv::gpu::GpuMat::convertTo(GpuMat& dst, int rtype, double alpha, double beta) const
{
    bool noScale = fabs(alpha - 1) < numeric_limits<double>::epsilon() && fabs(beta) < numeric_limits<double>::epsilon();

    if (rtype < 0)
        rtype = type();
    else
        rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());

    int sdepth = depth();
    int ddepth = CV_MAT_DEPTH(rtype);
    if (sdepth == ddepth && noScale)
    {
        copyTo(dst);
        return;
    }

    GpuMat temp;
    const GpuMat* psrc = this;
    if (sdepth != ddepth && psrc == &dst)
    {
        temp = *this;
        psrc = &temp;
    }

    dst.create(size(), rtype);

    if (noScale)
        cv::gpu::convertTo(*psrc, dst);
    else
        cv::gpu::convertTo(*psrc, dst, alpha, beta);
}

GpuMat& cv::gpu::GpuMat::setTo(Scalar s, const GpuMat& mask)
{
    CV_Assert(mask.empty() || mask.type() == CV_8UC1);
    CV_DbgAssert(!empty());

    gpu::setTo(*this, s, mask);

    return *this;
}

void cv::gpu::GpuMat::create(int _rows, int _cols, int _type)
{
    _type &= TYPE_MASK;

    if (rows == _rows && cols == _cols && type() == _type && data)
        return;

    if (data)
        release();

    CV_DbgAssert(_rows >= 0 && _cols >= 0);

    if (_rows > 0 && _cols > 0)
    {
        flags = Mat::MAGIC_VAL + _type;
        rows = _rows;
        cols = _cols;

        size_t esz = elemSize();

        void* devPtr;
        gpuFuncTable()->mallocPitch(&devPtr, &step, esz * cols, rows);

        // Single row must be continuous
        if (rows == 1)
            step = esz * cols;

        if (esz * cols == step)
            flags |= Mat::CONTINUOUS_FLAG;

        int64 _nettosize = static_cast<int64>(step) * rows;
        size_t nettosize = static_cast<size_t>(_nettosize);

        datastart = data = static_cast<uchar*>(devPtr);
        dataend = data + nettosize;

        refcount = static_cast<int*>(fastMalloc(sizeof(*refcount)));
        *refcount = 1;
    }
}

void cv::gpu::GpuMat::release()
{
    if (refcount && CV_XADD(refcount, -1) == 1)
    {
        fastFree(refcount);

        gpuFuncTable()->free(datastart);
    }

    data = datastart = dataend = 0;
    step = rows = cols = 0;
    refcount = 0;
}

namespace cv { namespace gpu
{
    void convertTo(const GpuMat& src, GpuMat& dst)
    {
        gpuFuncTable()->convert(src, dst);
    }

    void convertTo(const GpuMat& src, GpuMat& dst, double alpha, double beta, cudaStream_t stream)
    {
        gpuFuncTable()->convert(src, dst, alpha, beta, stream);
    }

    void setTo(GpuMat& src, Scalar s, cudaStream_t stream)
    {
        gpuFuncTable()->setTo(src, s, cv::gpu::GpuMat(), stream);
    }

    void setTo(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream)
    {
        gpuFuncTable()->setTo(src, s, mask, stream);
    }

    void setTo(GpuMat& src, Scalar s)
    {
        setTo(src, s, 0);
    }

    void setTo(GpuMat& src, Scalar s, const GpuMat& mask)
    {
        setTo(src, s, mask, 0);
    }
}}

////////////////////////////////////////////////////////////////////////
// Error handling

void cv::gpu::error(const char *error_string, const char *file, const int line, const char *func)
{
    int code = CV_GpuApiCallError;

    if (uncaught_exception())
    {
        const char* errorStr = cvErrorStr(code);
        const char* function = func ? func : "unknown function";

        cerr << "OpenCV Error: " << errorStr << "(" << error_string << ") in " << function << ", file " << file << ", line " << line;
        cerr.flush();
    }
    else
        cv::error( cv::Exception(code, error_string, func, file, line) );
}
