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

namespace cv { namespace ocl {

CV_EXPORTS_W bool haveOpenCL();
CV_EXPORTS_W bool useOpenCL();
CV_EXPORTS_W bool haveAmdBlas();
CV_EXPORTS_W bool haveAmdFft();
CV_EXPORTS_W void setUseOpenCL(bool flag);
CV_EXPORTS_W void finish();

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
    Device();
    explicit Device(void* d);
    Device(const Device& d);
    Device& operator = (const Device& d);
    ~Device();

    void set(void* d);

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

    String name() const;
    String extensions() const;
    String version() const;
    String vendorName() const;
    String OpenCL_C_Version() const;
    String OpenCLVersion() const;
    int deviceVersionMajor() const;
    int deviceVersionMinor() const;
    String driverVersion() const;
    void* ptr() const;

    int type() const;

    int addressBits() const;
    bool available() const;
    bool compilerAvailable() const;
    bool linkerAvailable() const;

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
    int doubleFPConfig() const;
    int singleFPConfig() const;
    int halfFPConfig() const;

    bool endianLittle() const;
    bool errorCorrectionSupport() const;

    enum
    {
        EXEC_KERNEL=(1 << 0),
        EXEC_NATIVE_KERNEL=(1 << 1)
    };
    int executionCapabilities() const;

    size_t globalMemCacheSize() const;

    enum
    {
        NO_CACHE=0,
        READ_ONLY_CACHE=1,
        READ_WRITE_CACHE=2
    };
    int globalMemCacheType() const;
    int globalMemCacheLineSize() const;
    size_t globalMemSize() const;

    size_t localMemSize() const;
    enum
    {
        NO_LOCAL_MEM=0,
        LOCAL_IS_LOCAL=1,
        LOCAL_IS_GLOBAL=2
    };
    int localMemType() const;
    bool hostUnifiedMemory() const;

    bool imageSupport() const;

    bool imageFromBufferSupport() const;
    uint imagePitchAlignment() const;
    uint imageBaseAddressAlignment() const;

    size_t image2DMaxWidth() const;
    size_t image2DMaxHeight() const;

    size_t image3DMaxWidth() const;
    size_t image3DMaxHeight() const;
    size_t image3DMaxDepth() const;

    size_t imageMaxBufferSize() const;
    size_t imageMaxArraySize() const;

    enum
    {
        UNKNOWN_VENDOR=0,
        VENDOR_AMD=1,
        VENDOR_INTEL=2,
        VENDOR_NVIDIA=3
    };
    int vendorID() const;
    // FIXIT
    // dev.isAMD() doesn't work for OpenCL CPU devices from AMD OpenCL platform.
    // This method should use platform name instead of vendor name.
    // After fix restore code in arithm.cpp: ocl_compare()
    inline bool isAMD() const { return vendorID() == VENDOR_AMD; }
    inline bool isIntel() const { return vendorID() == VENDOR_INTEL; }

    int maxClockFrequency() const;
    int maxComputeUnits() const;
    int maxConstantArgs() const;
    size_t maxConstantBufferSize() const;

    size_t maxMemAllocSize() const;
    size_t maxParameterSize() const;

    int maxReadImageArgs() const;
    int maxWriteImageArgs() const;
    int maxSamplers() const;

    size_t maxWorkGroupSize() const;
    int maxWorkItemDims() const;
    void maxWorkItemSizes(size_t*) const;

    int memBaseAddrAlign() const;

    int nativeVectorWidthChar() const;
    int nativeVectorWidthShort() const;
    int nativeVectorWidthInt() const;
    int nativeVectorWidthLong() const;
    int nativeVectorWidthFloat() const;
    int nativeVectorWidthDouble() const;
    int nativeVectorWidthHalf() const;

    int preferredVectorWidthChar() const;
    int preferredVectorWidthShort() const;
    int preferredVectorWidthInt() const;
    int preferredVectorWidthLong() const;
    int preferredVectorWidthFloat() const;
    int preferredVectorWidthDouble() const;
    int preferredVectorWidthHalf() const;

    size_t printfBufferSize() const;
    size_t profilingTimerResolution() const;

    static const Device& getDefault();

protected:
    struct Impl;
    Impl* p;
};


class CV_EXPORTS Context
{
public:
    Context();
    explicit Context(int dtype);
    ~Context();
    Context(const Context& c);
    Context& operator = (const Context& c);

    bool create();
    bool create(int dtype);
    size_t ndevices() const;
    const Device& device(size_t idx) const;
    Program getProg(const ProgramSource& prog,
                    const String& buildopt, String& errmsg);

    static Context& getDefault(bool initialize = true);
    void* ptr() const;

    friend void initializeContextFromHandle(Context& ctx, void* platform, void* context, void* device);
protected:
    struct Impl;
    Impl* p;
};

class CV_EXPORTS Platform
{
public:
    Platform();
    ~Platform();
    Platform(const Platform& p);
    Platform& operator = (const Platform& p);

    void* ptr() const;
    static Platform& getDefault();

    friend void initializeContextFromHandle(Context& ctx, void* platform, void* context, void* device);
protected:
    struct Impl;
    Impl* p;
};

// TODO Move to internal header
void initializeContextFromHandle(Context& ctx, void* platform, void* context, void* device);

class CV_EXPORTS Queue
{
public:
    Queue();
    explicit Queue(const Context& c, const Device& d=Device());
    ~Queue();
    Queue(const Queue& q);
    Queue& operator = (const Queue& q);

    bool create(const Context& c=Context(), const Device& d=Device());
    void finish();
    void* ptr() const;
    static Queue& getDefault();

protected:
    struct Impl;
    Impl* p;
};


class CV_EXPORTS KernelArg
{
public:
    enum { LOCAL=1, READ_ONLY=2, WRITE_ONLY=4, READ_WRITE=6, CONSTANT=8, PTR_ONLY = 16, NO_SIZE=256 };
    KernelArg(int _flags, UMat* _m, int wscale=1, int iwscale=1, const void* _obj=0, size_t _sz=0);
    KernelArg();

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
    static KernelArg Constant(const Mat& m);
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
    Kernel();
    Kernel(const char* kname, const Program& prog);
    Kernel(const char* kname, const ProgramSource& prog,
           const String& buildopts = String(), String* errmsg=0);
    ~Kernel();
    Kernel(const Kernel& k);
    Kernel& operator = (const Kernel& k);

    bool empty() const;
    bool create(const char* kname, const Program& prog);
    bool create(const char* kname, const ProgramSource& prog,
                const String& buildopts, String* errmsg=0);

    int set(int i, const void* value, size_t sz);
    int set(int i, const Image2D& image2D);
    int set(int i, const UMat& m);
    int set(int i, const KernelArg& arg);
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
             size_t localsize[], bool sync, const Queue& q=Queue());
    bool runTask(bool sync, const Queue& q=Queue());

    size_t workGroupSize() const;
    size_t preferedWorkGroupSizeMultiple() const;
    bool compileWorkGroupSize(size_t wsz[]) const;
    size_t localMemSize() const;

    void* ptr() const;
    struct Impl;

protected:
    Impl* p;
};

class CV_EXPORTS Program
{
public:
    Program();
    Program(const ProgramSource& src,
            const String& buildflags, String& errmsg);
    explicit Program(const String& buf);
    Program(const Program& prog);

    Program& operator = (const Program& prog);
    ~Program();

    bool create(const ProgramSource& src,
                const String& buildflags, String& errmsg);
    bool read(const String& buf, const String& buildflags);
    bool write(String& buf) const;

    const ProgramSource& source() const;
    void* ptr() const;

    String getPrefix() const;
    static String getPrefix(const String& buildflags);

protected:
    struct Impl;
    Impl* p;
};


class CV_EXPORTS ProgramSource
{
public:
    typedef uint64 hash_t;

    ProgramSource();
    explicit ProgramSource(const String& prog);
    explicit ProgramSource(const char* prog);
    ~ProgramSource();
    ProgramSource(const ProgramSource& prog);
    ProgramSource& operator = (const ProgramSource& prog);

    const String& source() const;
    hash_t hash() const;

protected:
    struct Impl;
    Impl* p;
};

class CV_EXPORTS PlatformInfo
{
public:
    PlatformInfo();
    explicit PlatformInfo(void* id);
    ~PlatformInfo();

    PlatformInfo(const PlatformInfo& i);
    PlatformInfo& operator =(const PlatformInfo& i);

    String name() const;
    String vendor() const;
    String version() const;
    int deviceNumber() const;
    void getDevice(Device& device, int d) const;

protected:
    struct Impl;
    Impl* p;
};

CV_EXPORTS const char* convertTypeStr(int sdepth, int ddepth, int cn, char* buf);
CV_EXPORTS const char* typeToStr(int t);
CV_EXPORTS const char* memopTypeToStr(int t);
CV_EXPORTS String kernelToStr(InputArray _kernel, int ddepth = -1, const char * name = NULL);
CV_EXPORTS void getPlatfomsInfo(std::vector<PlatformInfo>& platform_info);
CV_EXPORTS int predictOptimalVectorWidth(InputArray src1, InputArray src2 = noArray(), InputArray src3 = noArray(),
                                         InputArray src4 = noArray(), InputArray src5 = noArray(), InputArray src6 = noArray(),
                                         InputArray src7 = noArray(), InputArray src8 = noArray(), InputArray src9 = noArray());

CV_EXPORTS void buildOptionsAddMatrixDescription(String& buildOptions, const String& name, InputArray _m);

class CV_EXPORTS Image2D
{
public:
    Image2D();

    // src:     The UMat from which to get image properties and data
    // norm:    Flag to enable the use of normalized channel data types
    // alias:   Flag indicating that the image should alias the src UMat.
    //          If true, changes to the image or src will be reflected in
    //          both objects.
    explicit Image2D(const UMat &src, bool norm = false, bool alias = false);
    Image2D(const Image2D & i);
    ~Image2D();

    Image2D & operator = (const Image2D & i);

    // Indicates if creating an aliased image should succeed.  Depends on the
    // underlying platform and the dimensions of the UMat.
    static bool canCreateAlias(const UMat &u);

    // Indicates if the image format is supported.
    static bool isFormatSupported(int depth, int cn, bool norm);

    void* ptr() const;
protected:
    struct Impl;
    Impl* p;
};


CV_EXPORTS MatAllocator* getOpenCLAllocator();

CV_EXPORTS_W bool isPerformanceCheckBypassed();
#define OCL_PERFORMANCE_CHECK(condition) (cv::ocl::isPerformanceCheckBypassed() || (condition))

}}

#endif
