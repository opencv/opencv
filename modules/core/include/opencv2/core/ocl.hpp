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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_OPENCL_HPP__
#define __OPENCV_OPENCL_HPP__

#include "opencv2/core.hpp"

#ifndef OCL_FALLBACK_FN_CODE
#define OCL_FALLBACK_FN_CODE() ;
#define OCL_FALLBACK_FN_CODE_(...) ;
#define OCL_FALLBACK_METHOD_CODE() ;
#define OCL_FALLBACK_METHOD_CODE_(...) ;
#endif

namespace cv { namespace ocl {

//! @addtogroup core_opencl
//! @{

CV_EXPORTS_W bool haveOpenCL() OCL_FALLBACK_FN_CODE_(return false;);
CV_EXPORTS_W bool useOpenCL() OCL_FALLBACK_FN_CODE_(return false;);
CV_EXPORTS_W bool haveAmdBlas() OCL_FALLBACK_FN_CODE_(return false;);
CV_EXPORTS_W bool haveAmdFft() OCL_FALLBACK_FN_CODE_(return false;)
CV_EXPORTS_W void setUseOpenCL(bool flag) OCL_FALLBACK_FN_CODE()
CV_EXPORTS_W void finish() OCL_FALLBACK_FN_CODE()

CV_EXPORTS bool haveSVM() OCL_FALLBACK_FN_CODE_(return false;)

class CV_EXPORTS Context;
class CV_EXPORTS Device;
class CV_EXPORTS Kernel;
class CV_EXPORTS Program;
class CV_EXPORTS ProgramSource;
class CV_EXPORTS Queue;
class CV_EXPORTS PlatformInfo;
class CV_EXPORTS Image2D;

class CV_EXPORTS Device
{
public:
    Device()  OCL_FALLBACK_METHOD_CODE()
    explicit Device(void* d) OCL_FALLBACK_METHOD_CODE()
    Device(const Device& d) OCL_FALLBACK_METHOD_CODE()
    Device& operator = (const Device& d) OCL_FALLBACK_METHOD_CODE()
    ~Device() OCL_FALLBACK_METHOD_CODE_()

    void set(void* d) OCL_FALLBACK_METHOD_CODE()

    enum
    {
        TYPE_DEFAULT     = (1 << 0),
        TYPE_CPU         = (1 << 1),
        TYPE_GPU         = (1 << 2),
        TYPE_ACCELERATOR = (1 << 3),
        TYPE_DGPU        = TYPE_GPU + (1 << 16),
        TYPE_IGPU        = TYPE_GPU + (1 << 17),
        TYPE_ALL         = 0xFFFFFFFF
    };

    String name() const OCL_FALLBACK_METHOD_CODE()
    String extensions() const OCL_FALLBACK_METHOD_CODE()
    String version() const OCL_FALLBACK_METHOD_CODE()
    String vendorName() const OCL_FALLBACK_METHOD_CODE()
    String OpenCL_C_Version() const OCL_FALLBACK_METHOD_CODE()
    String OpenCLVersion() const OCL_FALLBACK_METHOD_CODE()
    int deviceVersionMajor() const OCL_FALLBACK_METHOD_CODE()
    int deviceVersionMinor() const OCL_FALLBACK_METHOD_CODE()
    String driverVersion() const OCL_FALLBACK_METHOD_CODE()
    void* ptr() const OCL_FALLBACK_METHOD_CODE()

    int type() const OCL_FALLBACK_METHOD_CODE()

    int addressBits() const OCL_FALLBACK_METHOD_CODE()
    bool available() const OCL_FALLBACK_METHOD_CODE()
    bool compilerAvailable() const OCL_FALLBACK_METHOD_CODE()
    bool linkerAvailable() const OCL_FALLBACK_METHOD_CODE()

    enum
    {
        FP_DENORM=(1 << 0),
        FP_INF_NAN=(1 << 1),
        FP_ROUND_TO_NEAREST=(1 << 2),
        FP_ROUND_TO_ZERO=(1 << 3),
        FP_ROUND_TO_INF=(1 << 4),
        FP_FMA=(1 << 5),
        FP_SOFT_FLOAT=(1 << 6),
        FP_CORRECTLY_ROUNDED_DIVIDE_SQRT=(1 << 7)
    };
    int doubleFPConfig() const OCL_FALLBACK_METHOD_CODE()
    int singleFPConfig() const OCL_FALLBACK_METHOD_CODE()
    int halfFPConfig() const OCL_FALLBACK_METHOD_CODE()

    bool endianLittle() const OCL_FALLBACK_METHOD_CODE()
    bool errorCorrectionSupport() const OCL_FALLBACK_METHOD_CODE()

    enum
    {
        EXEC_KERNEL=(1 << 0),
        EXEC_NATIVE_KERNEL=(1 << 1)
    };
    int executionCapabilities() const OCL_FALLBACK_METHOD_CODE()

    size_t globalMemCacheSize() const OCL_FALLBACK_METHOD_CODE()

    enum
    {
        NO_CACHE=0,
        READ_ONLY_CACHE=1,
        READ_WRITE_CACHE=2
    };
    int globalMemCacheType() const OCL_FALLBACK_METHOD_CODE()
    int globalMemCacheLineSize() const OCL_FALLBACK_METHOD_CODE()
    size_t globalMemSize() const OCL_FALLBACK_METHOD_CODE()

    size_t localMemSize() const OCL_FALLBACK_METHOD_CODE()
    enum
    {
        NO_LOCAL_MEM=0,
        LOCAL_IS_LOCAL=1,
        LOCAL_IS_GLOBAL=2
    };
    int localMemType() const OCL_FALLBACK_METHOD_CODE()
    bool hostUnifiedMemory() const OCL_FALLBACK_METHOD_CODE()

    bool imageSupport() const OCL_FALLBACK_METHOD_CODE()

    bool imageFromBufferSupport() const OCL_FALLBACK_METHOD_CODE()
    uint imagePitchAlignment() const OCL_FALLBACK_METHOD_CODE()
    uint imageBaseAddressAlignment() const OCL_FALLBACK_METHOD_CODE()

    size_t image2DMaxWidth() const OCL_FALLBACK_METHOD_CODE()
    size_t image2DMaxHeight() const OCL_FALLBACK_METHOD_CODE()

    size_t image3DMaxWidth() const OCL_FALLBACK_METHOD_CODE()
    size_t image3DMaxHeight() const OCL_FALLBACK_METHOD_CODE()
    size_t image3DMaxDepth() const OCL_FALLBACK_METHOD_CODE()

    size_t imageMaxBufferSize() const OCL_FALLBACK_METHOD_CODE()
    size_t imageMaxArraySize() const OCL_FALLBACK_METHOD_CODE()

    enum
    {
        UNKNOWN_VENDOR=0,
        VENDOR_AMD=1,
        VENDOR_INTEL=2,
        VENDOR_NVIDIA=3
    };
    int vendorID() const OCL_FALLBACK_METHOD_CODE()
    // FIXIT
    // dev.isAMD() doesn't work for OpenCL CPU devices from AMD OpenCL platform.
    // This method should use platform name instead of vendor name.
    // After fix restore code in arithm.cpp: ocl_compare()
    inline bool isAMD() const { return vendorID() == VENDOR_AMD; }
    inline bool isIntel() const { return vendorID() == VENDOR_INTEL; }
    inline bool isNVidia() const { return vendorID() == VENDOR_NVIDIA; }

    int maxClockFrequency() const OCL_FALLBACK_METHOD_CODE()
    int maxComputeUnits() const OCL_FALLBACK_METHOD_CODE()
    int maxConstantArgs() const OCL_FALLBACK_METHOD_CODE()
    size_t maxConstantBufferSize() const OCL_FALLBACK_METHOD_CODE()

    size_t maxMemAllocSize() const OCL_FALLBACK_METHOD_CODE()
    size_t maxParameterSize() const OCL_FALLBACK_METHOD_CODE()

    int maxReadImageArgs() const OCL_FALLBACK_METHOD_CODE()
    int maxWriteImageArgs() const OCL_FALLBACK_METHOD_CODE()
    int maxSamplers() const OCL_FALLBACK_METHOD_CODE()

    size_t maxWorkGroupSize() const OCL_FALLBACK_METHOD_CODE()
    int maxWorkItemDims() const OCL_FALLBACK_METHOD_CODE()
    void maxWorkItemSizes(size_t*) const OCL_FALLBACK_METHOD_CODE()

    int memBaseAddrAlign() const OCL_FALLBACK_METHOD_CODE()

    int nativeVectorWidthChar() const OCL_FALLBACK_METHOD_CODE()
    int nativeVectorWidthShort() const OCL_FALLBACK_METHOD_CODE()
    int nativeVectorWidthInt() const OCL_FALLBACK_METHOD_CODE()
    int nativeVectorWidthLong() const OCL_FALLBACK_METHOD_CODE()
    int nativeVectorWidthFloat() const OCL_FALLBACK_METHOD_CODE()
    int nativeVectorWidthDouble() const OCL_FALLBACK_METHOD_CODE()
    int nativeVectorWidthHalf() const OCL_FALLBACK_METHOD_CODE()

    int preferredVectorWidthChar() const OCL_FALLBACK_METHOD_CODE()
    int preferredVectorWidthShort() const OCL_FALLBACK_METHOD_CODE()
    int preferredVectorWidthInt() const OCL_FALLBACK_METHOD_CODE()
    int preferredVectorWidthLong() const OCL_FALLBACK_METHOD_CODE()
    int preferredVectorWidthFloat() const OCL_FALLBACK_METHOD_CODE()
    int preferredVectorWidthDouble() const OCL_FALLBACK_METHOD_CODE()
    int preferredVectorWidthHalf() const OCL_FALLBACK_METHOD_CODE()

    size_t printfBufferSize() const OCL_FALLBACK_METHOD_CODE()
    size_t profilingTimerResolution() const OCL_FALLBACK_METHOD_CODE()

    static const Device& getDefault() OCL_FALLBACK_METHOD_CODE()

protected:
    struct Impl;
    Impl* p;
};


class CV_EXPORTS Context
{
public:
    Context() OCL_FALLBACK_METHOD_CODE()
    explicit Context(int dtype) OCL_FALLBACK_METHOD_CODE()
    ~Context() OCL_FALLBACK_METHOD_CODE_()
    Context(const Context& c) OCL_FALLBACK_METHOD_CODE()
    Context& operator = (const Context& c) OCL_FALLBACK_METHOD_CODE()

    bool create() OCL_FALLBACK_METHOD_CODE()
    bool create(int dtype) OCL_FALLBACK_METHOD_CODE()
    size_t ndevices() const OCL_FALLBACK_METHOD_CODE()
    const Device& device(size_t idx) const OCL_FALLBACK_METHOD_CODE()
    Program getProg(const ProgramSource& prog,
                    const String& buildopt, String& errmsg); /*CL_FALLBACK_CODE()*/

    static Context& getDefault(bool initialize = true) OCL_FALLBACK_METHOD_CODE()
    void* ptr() const OCL_FALLBACK_METHOD_CODE()

    friend void initializeContextFromHandle(Context& ctx, void* platform, void* context, void* device) OCL_FALLBACK_METHOD_CODE()

    bool useSVM() const OCL_FALLBACK_METHOD_CODE()
    void setUseSVM(bool enabled) OCL_FALLBACK_METHOD_CODE()

    struct Impl;
    Impl* p;
};

class CV_EXPORTS Platform
{
public:
    Platform() OCL_FALLBACK_METHOD_CODE()
    ~Platform() OCL_FALLBACK_METHOD_CODE_()
    Platform(const Platform& p) OCL_FALLBACK_METHOD_CODE()
    Platform& operator = (const Platform& p) OCL_FALLBACK_METHOD_CODE()

    void* ptr() const OCL_FALLBACK_METHOD_CODE()
    static Platform& getDefault() OCL_FALLBACK_METHOD_CODE()

    friend void initializeContextFromHandle(Context& ctx, void* platform, void* context, void* device);
protected:
    struct Impl;
    Impl* p;
};

/*
//! @brief Attaches OpenCL context to OpenCV
//
//! @note Note:
//    OpenCV will check if available OpenCL platform has platformName name,
//    then assign context to OpenCV and call clRetainContext function.
//    The deviceID device will be used as target device and new command queue
//    will be created.
//
// Params:
//! @param platformName - name of OpenCL platform to attach,
//!                       this string is used to check if platform is available
//!                       to OpenCV at runtime
//! @param platfromID   - ID of platform attached context was created for
//! @param context      - OpenCL context to be attached to OpenCV
//! @param deviceID     - ID of device, must be created from attached context
*/
CV_EXPORTS void attachContext(const String& platformName, void* platformID, void* context, void* deviceID) OCL_FALLBACK_FN_CODE()

/*
//! @brief Convert OpenCL buffer to UMat
//
//! @note Note:
//   OpenCL buffer (cl_mem_buffer) should contain 2D image data, compatible with OpenCV.
//   Memory content is not copied from clBuffer to UMat. Instead, buffer handle assigned
//   to UMat and clRetainMemObject is called.
//
// Params:
//! @param  cl_mem_buffer - source clBuffer handle
//! @param  step          - num of bytes in single row
//! @param  rows          - number of rows
//! @param  cols          - number of cols
//! @param  type          - OpenCV type of image
//! @param  dst           - destination UMat
*/
CV_EXPORTS void convertFromBuffer(void* cl_mem_buffer, size_t step, int rows, int cols, int type, UMat& dst) OCL_FALLBACK_FN_CODE()

/*
//! @brief Convert OpenCL image2d_t to UMat
//
//! @note Note:
//   OpenCL image2d_t (cl_mem_image), should be compatible with OpenCV
//   UMat formats.
//   Memory content is copied from image to UMat with
//   clEnqueueCopyImageToBuffer function.
//
// Params:
//! @param  cl_mem_image - source image2d_t handle
//! @param  dst          - destination UMat
*/
CV_EXPORTS void convertFromImage(void* cl_mem_image, UMat& dst) OCL_FALLBACK_FN_CODE()

// TODO Move to internal header
void initializeContextFromHandle(Context& ctx, void* platform, void* context, void* device);

class CV_EXPORTS Queue
{
public:
    Queue() OCL_FALLBACK_METHOD_CODE()
    explicit Queue(const Context& c, const Device& d=Device()) OCL_FALLBACK_METHOD_CODE()
    ~Queue() OCL_FALLBACK_METHOD_CODE_()
    Queue(const Queue& q) OCL_FALLBACK_METHOD_CODE()
    Queue& operator = (const Queue& q) OCL_FALLBACK_METHOD_CODE()

    bool create(const Context& c=Context(), const Device& d=Device()) OCL_FALLBACK_METHOD_CODE()
    void finish() OCL_FALLBACK_METHOD_CODE()
    void* ptr() const OCL_FALLBACK_METHOD_CODE()
    static Queue& getDefault() OCL_FALLBACK_METHOD_CODE()

protected:
    struct Impl;
    Impl* p;
};


class CV_EXPORTS KernelArg
{
public:
    KernelArg(int _flags, UMat* _m, int wscale=1, int iwscale=1, const void* _obj=0, size_t _sz=0) OCL_FALLBACK_METHOD_CODE()
    KernelArg() OCL_FALLBACK_METHOD_CODE()

    enum
    {
        LOCAL=1, READ_ONLY=2, WRITE_ONLY=4, READ_WRITE=6, CONSTANT=8, PTR_ONLY = 16, NO_SIZE=256
    };

    static KernelArg Local() { return KernelArg(LOCAL, 0); }
    static KernelArg PtrWriteOnly(const UMat& m)
    { return KernelArg(PTR_ONLY+WRITE_ONLY, (UMat*)&m); }
    static KernelArg PtrReadOnly(const UMat& m)
    { return KernelArg(PTR_ONLY+READ_ONLY, (UMat*)&m); }
    static KernelArg PtrReadWrite(const UMat& m)
    { return KernelArg(PTR_ONLY+READ_WRITE, (UMat*)&m); }
    static KernelArg ReadWrite(const UMat& m, int wscale=1, int iwscale=1)
    { return KernelArg(READ_WRITE, (UMat*)&m, wscale, iwscale); }
    static KernelArg ReadWriteNoSize(const UMat& m, int wscale=1, int iwscale=1)
    { return KernelArg(READ_WRITE+NO_SIZE, (UMat*)&m, wscale, iwscale); }
    static KernelArg ReadOnly(const UMat& m, int wscale=1, int iwscale=1)
    { return KernelArg(READ_ONLY, (UMat*)&m, wscale, iwscale); }
    static KernelArg WriteOnly(const UMat& m, int wscale=1, int iwscale=1)
    { return KernelArg(WRITE_ONLY, (UMat*)&m, wscale, iwscale); }
    static KernelArg ReadOnlyNoSize(const UMat& m, int wscale=1, int iwscale=1)
    { return KernelArg(READ_ONLY+NO_SIZE, (UMat*)&m, wscale, iwscale); }
    static KernelArg WriteOnlyNoSize(const UMat& m, int wscale=1, int iwscale=1)
    { return KernelArg(WRITE_ONLY+NO_SIZE, (UMat*)&m, wscale, iwscale); }
    static KernelArg Constant(const Mat& m) OCL_FALLBACK_METHOD_CODE()
    template<typename _Tp> static KernelArg Constant(const _Tp* arr, size_t n)
    { return KernelArg(CONSTANT, 0, 1, 1, (void*)arr, n); }

    int flags;
    UMat* m;
    const void* obj;
    size_t sz;
    int wscale, iwscale;
};


class CV_EXPORTS Kernel
{
public:
    Kernel() OCL_FALLBACK_METHOD_CODE()
    Kernel(const char* kname, const Program& prog) OCL_FALLBACK_METHOD_CODE()
    Kernel(const char* kname, const ProgramSource& prog,
           const String& buildopts = String(), String* errmsg=0) OCL_FALLBACK_METHOD_CODE()
    ~Kernel() OCL_FALLBACK_METHOD_CODE_()
    Kernel(const Kernel& k) OCL_FALLBACK_METHOD_CODE()
    Kernel& operator = (const Kernel& k) OCL_FALLBACK_METHOD_CODE()

    bool empty() const OCL_FALLBACK_METHOD_CODE()
    bool create(const char* kname, const Program& prog) OCL_FALLBACK_METHOD_CODE()
    bool create(const char* kname, const ProgramSource& prog,
                const String& buildopts, String* errmsg=0) OCL_FALLBACK_METHOD_CODE()

    int set(int i, const void* value, size_t sz) OCL_FALLBACK_METHOD_CODE()
    int set(int i, const Image2D& image2D) OCL_FALLBACK_METHOD_CODE()
    int set(int i, const UMat& m) OCL_FALLBACK_METHOD_CODE()
    int set(int i, const KernelArg& arg) OCL_FALLBACK_METHOD_CODE()
    template<typename _Tp> int set(int i, const _Tp& value)
    { return set(i, &value, sizeof(value)); }

    template<typename _Tp0>
    Kernel& args(const _Tp0& a0)
    {
        set(0, a0); return *this;
    }

    template<typename _Tp0, typename _Tp1>
    Kernel& args(const _Tp0& a0, const _Tp1& a1)
    {
        int i = set(0, a0); set(i, a1); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2)
    {
        int i = set(0, a0); i = set(i, a1); set(i, a2); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3, typename _Tp4>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2,
                 const _Tp3& a3, const _Tp4& a4)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2);
        i = set(i, a3); set(i, a4); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2,
             typename _Tp3, typename _Tp4, typename _Tp5>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2,
                 const _Tp3& a3, const _Tp4& a4, const _Tp5& a5)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2);
        i = set(i, a3); i = set(i, a4); set(i, a5); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3);
        i = set(i, a4); i = set(i, a5); set(i, a6); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6, typename _Tp7>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3);
        i = set(i, a4); i = set(i, a5); i = set(i, a6); set(i, a7); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3, typename _Tp4,
             typename _Tp5, typename _Tp6, typename _Tp7, typename _Tp8>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4);
        i = set(i, a5); i = set(i, a6); i = set(i, a7); set(i, a8); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3, typename _Tp4,
             typename _Tp5, typename _Tp6, typename _Tp7, typename _Tp8, typename _Tp9>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8, const _Tp9& a9)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4); i = set(i, a5);
        i = set(i, a6); i = set(i, a7); i = set(i, a8); set(i, a9); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6, typename _Tp7,
             typename _Tp8, typename _Tp9, typename _Tp10>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8, const _Tp9& a9, const _Tp10& a10)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4); i = set(i, a5);
        i = set(i, a6); i = set(i, a7); i = set(i, a8); i = set(i, a9); set(i, a10); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6, typename _Tp7,
             typename _Tp8, typename _Tp9, typename _Tp10, typename _Tp11>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8, const _Tp9& a9, const _Tp10& a10, const _Tp11& a11)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4); i = set(i, a5);
        i = set(i, a6); i = set(i, a7); i = set(i, a8); i = set(i, a9); i = set(i, a10); set(i, a11); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6, typename _Tp7,
             typename _Tp8, typename _Tp9, typename _Tp10, typename _Tp11, typename _Tp12>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8, const _Tp9& a9, const _Tp10& a10, const _Tp11& a11,
                 const _Tp12& a12)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4); i = set(i, a5);
        i = set(i, a6); i = set(i, a7); i = set(i, a8); i = set(i, a9); i = set(i, a10); i = set(i, a11);
        set(i, a12); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6, typename _Tp7,
             typename _Tp8, typename _Tp9, typename _Tp10, typename _Tp11, typename _Tp12,
             typename _Tp13>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8, const _Tp9& a9, const _Tp10& a10, const _Tp11& a11,
                 const _Tp12& a12, const _Tp13& a13)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4); i = set(i, a5);
        i = set(i, a6); i = set(i, a7); i = set(i, a8); i = set(i, a9); i = set(i, a10); i = set(i, a11);
        i = set(i, a12); set(i, a13); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6, typename _Tp7,
             typename _Tp8, typename _Tp9, typename _Tp10, typename _Tp11, typename _Tp12,
             typename _Tp13, typename _Tp14>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8, const _Tp9& a9, const _Tp10& a10, const _Tp11& a11,
                 const _Tp12& a12, const _Tp13& a13, const _Tp14& a14)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4); i = set(i, a5);
        i = set(i, a6); i = set(i, a7); i = set(i, a8); i = set(i, a9); i = set(i, a10); i = set(i, a11);
        i = set(i, a12); i = set(i, a13); set(i, a14); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6, typename _Tp7,
             typename _Tp8, typename _Tp9, typename _Tp10, typename _Tp11, typename _Tp12,
             typename _Tp13, typename _Tp14, typename _Tp15>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8, const _Tp9& a9, const _Tp10& a10, const _Tp11& a11,
                 const _Tp12& a12, const _Tp13& a13, const _Tp14& a14, const _Tp15& a15)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4); i = set(i, a5);
        i = set(i, a6); i = set(i, a7); i = set(i, a8); i = set(i, a9); i = set(i, a10); i = set(i, a11);
        i = set(i, a12); i = set(i, a13); i = set(i, a14); set(i, a15); return *this;
    }

    bool run(int dims, size_t globalsize[],
             size_t localsize[], bool sync, const Queue& q=Queue()) OCL_FALLBACK_METHOD_CODE()
    bool runTask(bool sync, const Queue& q=Queue()) OCL_FALLBACK_METHOD_CODE()

    size_t workGroupSize() const OCL_FALLBACK_METHOD_CODE()
    size_t preferedWorkGroupSizeMultiple() const OCL_FALLBACK_METHOD_CODE()
    bool compileWorkGroupSize(size_t wsz[]) const OCL_FALLBACK_METHOD_CODE()
    size_t localMemSize() const OCL_FALLBACK_METHOD_CODE()

    void* ptr() const OCL_FALLBACK_METHOD_CODE()
    struct Impl;

protected:
    Impl* p;
};

class CV_EXPORTS Program
{
public:
    Program() OCL_FALLBACK_METHOD_CODE()
    Program(const ProgramSource& src,
            const String& buildflags, String& errmsg) OCL_FALLBACK_METHOD_CODE()
    explicit Program(const String& buf) OCL_FALLBACK_METHOD_CODE()
    Program(const Program& prog) OCL_FALLBACK_METHOD_CODE()

    Program& operator = (const Program& prog) OCL_FALLBACK_METHOD_CODE()
    ~Program() OCL_FALLBACK_METHOD_CODE_()

    bool create(const ProgramSource& src,
                const String& buildflags, String& errmsg) OCL_FALLBACK_METHOD_CODE()
    bool read(const String& buf, const String& buildflags) OCL_FALLBACK_METHOD_CODE()
    bool write(String& buf) const OCL_FALLBACK_METHOD_CODE()

    const ProgramSource& source() const OCL_FALLBACK_METHOD_CODE()
    void* ptr() const OCL_FALLBACK_METHOD_CODE()

    String getPrefix() const OCL_FALLBACK_METHOD_CODE()
    static String getPrefix(const String& buildflags) OCL_FALLBACK_METHOD_CODE()

protected:
    struct Impl;
    Impl* p;
};


class CV_EXPORTS ProgramSource
{
public:
    typedef uint64 hash_t;

    ProgramSource() OCL_FALLBACK_METHOD_CODE()
    explicit ProgramSource(const String& prog) OCL_FALLBACK_METHOD_CODE()
    explicit ProgramSource(const char* prog) OCL_FALLBACK_METHOD_CODE()
    ~ProgramSource() OCL_FALLBACK_METHOD_CODE_()
    ProgramSource(const ProgramSource& prog) OCL_FALLBACK_METHOD_CODE()
    ProgramSource& operator = (const ProgramSource& prog) OCL_FALLBACK_METHOD_CODE()

    const String& source() const OCL_FALLBACK_METHOD_CODE()
    hash_t hash() const OCL_FALLBACK_METHOD_CODE()

protected:
    struct Impl;
    Impl* p;
};

class CV_EXPORTS PlatformInfo
{
public:
    PlatformInfo() OCL_FALLBACK_METHOD_CODE()
    explicit PlatformInfo(void* id) OCL_FALLBACK_METHOD_CODE()
    ~PlatformInfo() OCL_FALLBACK_METHOD_CODE_()

    PlatformInfo(const PlatformInfo& i) OCL_FALLBACK_METHOD_CODE()
    PlatformInfo& operator =(const PlatformInfo& i) OCL_FALLBACK_METHOD_CODE()

    String name() const OCL_FALLBACK_METHOD_CODE()
    String vendor() const OCL_FALLBACK_METHOD_CODE()
    String version() const OCL_FALLBACK_METHOD_CODE()
    int deviceNumber() const OCL_FALLBACK_METHOD_CODE()
    void getDevice(Device& device, int d) const OCL_FALLBACK_METHOD_CODE()

protected:
    struct Impl;
    Impl* p;
};

CV_EXPORTS const char* convertTypeStr(int sdepth, int ddepth, int cn, char* buf) OCL_FALLBACK_FN_CODE()
CV_EXPORTS const char* typeToStr(int t) OCL_FALLBACK_FN_CODE()
CV_EXPORTS const char* memopTypeToStr(int t) OCL_FALLBACK_FN_CODE()
CV_EXPORTS const char* vecopTypeToStr(int t) OCL_FALLBACK_FN_CODE()
CV_EXPORTS String kernelToStr(InputArray _kernel, int ddepth = -1, const char * name = NULL) OCL_FALLBACK_FN_CODE()
CV_EXPORTS void getPlatfomsInfo(std::vector<PlatformInfo>& platform_info) OCL_FALLBACK_FN_CODE()


enum OclVectorStrategy
{
    // all matrices have its own vector width
    OCL_VECTOR_OWN = 0,
    // all matrices have maximal vector width among all matrices
    // (useful for cases when matrices have different data types)
    OCL_VECTOR_MAX = 1,

    // default strategy
    OCL_VECTOR_DEFAULT = OCL_VECTOR_OWN
};

CV_EXPORTS int predictOptimalVectorWidth(InputArray src1, InputArray src2 = noArray(), InputArray src3 = noArray(),
                                         InputArray src4 = noArray(), InputArray src5 = noArray(), InputArray src6 = noArray(),
                                         InputArray src7 = noArray(), InputArray src8 = noArray(), InputArray src9 = noArray(),
                                         OclVectorStrategy strat = OCL_VECTOR_DEFAULT) OCL_FALLBACK_FN_CODE()

CV_EXPORTS int checkOptimalVectorWidth(const int *vectorWidths,
                                       InputArray src1, InputArray src2 = noArray(), InputArray src3 = noArray(),
                                       InputArray src4 = noArray(), InputArray src5 = noArray(), InputArray src6 = noArray(),
                                       InputArray src7 = noArray(), InputArray src8 = noArray(), InputArray src9 = noArray(),
                                       OclVectorStrategy strat = OCL_VECTOR_DEFAULT) OCL_FALLBACK_FN_CODE()

// with OCL_VECTOR_MAX strategy
CV_EXPORTS int predictOptimalVectorWidthMax(InputArray src1, InputArray src2 = noArray(), InputArray src3 = noArray(),
                                            InputArray src4 = noArray(), InputArray src5 = noArray(), InputArray src6 = noArray(),
                                            InputArray src7 = noArray(), InputArray src8 = noArray(), InputArray src9 = noArray()) OCL_FALLBACK_FN_CODE()

CV_EXPORTS void buildOptionsAddMatrixDescription(String& buildOptions, const String& name, InputArray _m) OCL_FALLBACK_FN_CODE()

class CV_EXPORTS Image2D
{
public:
    Image2D() OCL_FALLBACK_METHOD_CODE()

    // src:     The UMat from which to get image properties and data
    // norm:    Flag to enable the use of normalized channel data types
    // alias:   Flag indicating that the image should alias the src UMat.
    //          If true, changes to the image or src will be reflected in
    //          both objects.
    explicit Image2D(const UMat &src, bool norm = false, bool alias = false) OCL_FALLBACK_METHOD_CODE()
    Image2D(const Image2D & i) OCL_FALLBACK_METHOD_CODE()
    ~Image2D() OCL_FALLBACK_METHOD_CODE_()

    Image2D & operator = (const Image2D & i) OCL_FALLBACK_METHOD_CODE()

    // Indicates if creating an aliased image should succeed.  Depends on the
    // underlying platform and the dimensions of the UMat.
    static bool canCreateAlias(const UMat &u) OCL_FALLBACK_METHOD_CODE()

    // Indicates if the image format is supported.
    static bool isFormatSupported(int depth, int cn, bool norm) OCL_FALLBACK_METHOD_CODE()

    void* ptr() const OCL_FALLBACK_METHOD_CODE()
protected:
    struct Impl;
    Impl* p;
};


CV_EXPORTS MatAllocator* getOpenCLAllocator() OCL_FALLBACK_FN_CODE()


#ifdef __OPENCV_BUILD
namespace internal {

CV_EXPORTS bool isPerformanceCheckBypassed();
#define OCL_PERFORMANCE_CHECK(condition) (cv::ocl::internal::isPerformanceCheckBypassed() || (condition))

CV_EXPORTS bool isCLBuffer(UMat& u);

} // namespace internal
#endif

//! @}

}}

#undef OCL_FALLBACK_FN_CODE
#undef OCL_FALLBACK_FN_CODE_
#undef OCL_FALLBACK_METHOD_CODE
#undef OCL_FALLBACK_METHOD_CODE_

#endif
