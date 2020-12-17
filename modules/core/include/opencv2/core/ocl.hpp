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

#ifndef OPENCV_OPENCL_HPP
#define OPENCV_OPENCL_HPP

#include "opencv2/core.hpp"

namespace cv { namespace ocl {

//! @addtogroup core_opencl
//! @{

CV_EXPORTS_W bool haveOpenCL();
CV_EXPORTS_W bool useOpenCL();
CV_EXPORTS_W bool haveAmdBlas();
CV_EXPORTS_W bool haveAmdFft();
CV_EXPORTS_W void setUseOpenCL(bool flag);
CV_EXPORTS_W void finish();

CV_EXPORTS bool haveSVM();

class CV_EXPORTS Context;
class CV_EXPORTS_W_SIMPLE Device;
class CV_EXPORTS Kernel;
class CV_EXPORTS Program;
class CV_EXPORTS ProgramSource;
class CV_EXPORTS Queue;
class CV_EXPORTS PlatformInfo;
class CV_EXPORTS Image2D;

class CV_EXPORTS_W_SIMPLE Device
{
public:
    CV_WRAP Device();
    explicit Device(void* d);
    Device(const Device& d);
    Device& operator = (const Device& d);
    CV_WRAP ~Device();

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

    CV_WRAP String name() const;
    CV_WRAP String extensions() const;
    CV_WRAP bool isExtensionSupported(const String& extensionName) const;
    CV_WRAP String version() const;
    CV_WRAP String vendorName() const;
    CV_WRAP String OpenCL_C_Version() const;
    CV_WRAP String OpenCLVersion() const;
    CV_WRAP int deviceVersionMajor() const;
    CV_WRAP int deviceVersionMinor() const;
    CV_WRAP String driverVersion() const;
    void* ptr() const;

    CV_WRAP int type() const;

    CV_WRAP int addressBits() const;
    CV_WRAP bool available() const;
    CV_WRAP bool compilerAvailable() const;
    CV_WRAP bool linkerAvailable() const;

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
    CV_WRAP int doubleFPConfig() const;
    CV_WRAP int singleFPConfig() const;
    CV_WRAP int halfFPConfig() const;

    CV_WRAP bool endianLittle() const;
    CV_WRAP bool errorCorrectionSupport() const;

    enum
    {
        EXEC_KERNEL=(1 << 0),
        EXEC_NATIVE_KERNEL=(1 << 1)
    };
    CV_WRAP int executionCapabilities() const;

    CV_WRAP size_t globalMemCacheSize() const;

    enum
    {
        NO_CACHE=0,
        READ_ONLY_CACHE=1,
        READ_WRITE_CACHE=2
    };
    CV_WRAP int globalMemCacheType() const;
    CV_WRAP int globalMemCacheLineSize() const;
    CV_WRAP size_t globalMemSize() const;

    CV_WRAP size_t localMemSize() const;
    enum
    {
        NO_LOCAL_MEM=0,
        LOCAL_IS_LOCAL=1,
        LOCAL_IS_GLOBAL=2
    };
    CV_WRAP int localMemType() const;
    CV_WRAP bool hostUnifiedMemory() const;

    CV_WRAP bool imageSupport() const;

    CV_WRAP bool imageFromBufferSupport() const;
    uint imagePitchAlignment() const;
    uint imageBaseAddressAlignment() const;

    /// deprecated, use isExtensionSupported() method (probably with "cl_khr_subgroups" value)
    CV_WRAP bool intelSubgroupsSupport() const;

    CV_WRAP size_t image2DMaxWidth() const;
    CV_WRAP size_t image2DMaxHeight() const;

    CV_WRAP size_t image3DMaxWidth() const;
    CV_WRAP size_t image3DMaxHeight() const;
    CV_WRAP size_t image3DMaxDepth() const;

    CV_WRAP size_t imageMaxBufferSize() const;
    CV_WRAP size_t imageMaxArraySize() const;

    enum
    {
        UNKNOWN_VENDOR=0,
        VENDOR_AMD=1,
        VENDOR_INTEL=2,
        VENDOR_NVIDIA=3
    };
    CV_WRAP int vendorID() const;
    // FIXIT
    // dev.isAMD() doesn't work for OpenCL CPU devices from AMD OpenCL platform.
    // This method should use platform name instead of vendor name.
    // After fix restore code in arithm.cpp: ocl_compare()
    CV_WRAP inline bool isAMD() const { return vendorID() == VENDOR_AMD; }
    CV_WRAP inline bool isIntel() const { return vendorID() == VENDOR_INTEL; }
    CV_WRAP inline bool isNVidia() const { return vendorID() == VENDOR_NVIDIA; }

    CV_WRAP int maxClockFrequency() const;
    CV_WRAP int maxComputeUnits() const;
    CV_WRAP int maxConstantArgs() const;
    CV_WRAP size_t maxConstantBufferSize() const;

    CV_WRAP size_t maxMemAllocSize() const;
    CV_WRAP size_t maxParameterSize() const;

    CV_WRAP int maxReadImageArgs() const;
    CV_WRAP int maxWriteImageArgs() const;
    CV_WRAP int maxSamplers() const;

    CV_WRAP size_t maxWorkGroupSize() const;
    CV_WRAP int maxWorkItemDims() const;
    void maxWorkItemSizes(size_t*) const;

    CV_WRAP int memBaseAddrAlign() const;

    CV_WRAP int nativeVectorWidthChar() const;
    CV_WRAP int nativeVectorWidthShort() const;
    CV_WRAP int nativeVectorWidthInt() const;
    CV_WRAP int nativeVectorWidthLong() const;
    CV_WRAP int nativeVectorWidthFloat() const;
    CV_WRAP int nativeVectorWidthDouble() const;
    CV_WRAP int nativeVectorWidthHalf() const;

    CV_WRAP int preferredVectorWidthChar() const;
    CV_WRAP int preferredVectorWidthShort() const;
    CV_WRAP int preferredVectorWidthInt() const;
    CV_WRAP int preferredVectorWidthLong() const;
    CV_WRAP int preferredVectorWidthFloat() const;
    CV_WRAP int preferredVectorWidthDouble() const;
    CV_WRAP int preferredVectorWidthHalf() const;

    CV_WRAP size_t printfBufferSize() const;
    CV_WRAP size_t profilingTimerResolution() const;

    CV_WRAP static const Device& getDefault();

    /**
     * @param d OpenCL handle (cl_device_id). clRetainDevice() is called on success.
     */
    static Device fromHandle(void* d);

    struct Impl;
    inline Impl* getImpl() const { return (Impl*)p; }
    inline bool empty() const { return !p; }
protected:
    Impl* p;
};


class CV_EXPORTS Context
{
public:
    Context();
    explicit Context(int dtype);  //!< @deprecated
    ~Context();
    Context(const Context& c);
    Context& operator= (const Context& c);

    /** @deprecated */
    bool create();
    /** @deprecated */
    bool create(int dtype);

    size_t ndevices() const;
    Device& device(size_t idx) const;
    Program getProg(const ProgramSource& prog,
                    const String& buildopt, String& errmsg);
    void unloadProg(Program& prog);


    /** Get thread-local OpenCL context (initialize if necessary) */
#if 0  // OpenCV 5.0
    static Context& getDefault();
#else
    static Context& getDefault(bool initialize = true);
#endif

    /** @returns cl_context value */
    void* ptr() const;


    bool useSVM() const;
    void setUseSVM(bool enabled);

    /**
     * @param context OpenCL handle (cl_context). clRetainContext() is called on success
     */
    static Context fromHandle(void* context);
    static Context fromDevice(const ocl::Device& device);
    static Context create(const std::string& configuration);

    void release();

    struct Impl;
    inline Impl* getImpl() const { return (Impl*)p; }
    inline bool empty() const { return !p; }
// TODO OpenCV 5.0
//protected:
    Impl* p;
};

/** @deprecated */
class CV_EXPORTS Platform
{
public:
    Platform();
    ~Platform();
    Platform(const Platform& p);
    Platform& operator = (const Platform& p);

    void* ptr() const;

    /** @deprecated */
    static Platform& getDefault();

    struct Impl;
    inline Impl* getImpl() const { return (Impl*)p; }
    inline bool empty() const { return !p; }
protected:
    Impl* p;
};

/** @brief Attaches OpenCL context to OpenCV
@note
  OpenCV will check if available OpenCL platform has platformName name, then assign context to
  OpenCV and call `clRetainContext` function. The deviceID device will be used as target device and
  new command queue will be created.
@param platformName name of OpenCL platform to attach, this string is used to check if platform is available to OpenCV at runtime
@param platformID ID of platform attached context was created for
@param context OpenCL context to be attached to OpenCV
@param deviceID ID of device, must be created from attached context
*/
CV_EXPORTS void attachContext(const String& platformName, void* platformID, void* context, void* deviceID);

/** @brief Convert OpenCL buffer to UMat
@note
  OpenCL buffer (cl_mem_buffer) should contain 2D image data, compatible with OpenCV. Memory
  content is not copied from `clBuffer` to UMat. Instead, buffer handle assigned to UMat and
  `clRetainMemObject` is called.
@param cl_mem_buffer source clBuffer handle
@param step num of bytes in single row
@param rows number of rows
@param cols number of cols
@param type OpenCV type of image
@param dst destination UMat
*/
CV_EXPORTS void convertFromBuffer(void* cl_mem_buffer, size_t step, int rows, int cols, int type, UMat& dst);

/** @brief Convert OpenCL image2d_t to UMat
@note
  OpenCL `image2d_t` (cl_mem_image), should be compatible with OpenCV UMat formats. Memory content
  is copied from image to UMat with `clEnqueueCopyImageToBuffer` function.
@param cl_mem_image source image2d_t handle
@param dst destination UMat
*/
CV_EXPORTS void convertFromImage(void* cl_mem_image, UMat& dst);

// TODO Move to internal header
/// @deprecated
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

    /// @brief Returns OpenCL command queue with enable profiling mode support
    const Queue& getProfilingQueue() const;

    struct Impl; friend struct Impl;
    inline Impl* getImpl() const { return p; }
    inline bool empty() const { return !p; }
protected:
    Impl* p;
};


class CV_EXPORTS KernelArg
{
public:
    enum { LOCAL=1, READ_ONLY=2, WRITE_ONLY=4, READ_WRITE=6, CONSTANT=8, PTR_ONLY = 16, NO_SIZE=256 };
    KernelArg(int _flags, UMat* _m, int wscale=1, int iwscale=1, const void* _obj=0, size_t _sz=0);
    KernelArg();

    static KernelArg Local(size_t localMemSize)
    { return KernelArg(LOCAL, 0, 1, 1, 0, localMemSize); }
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


protected:
    template<typename _Tp0> inline
    int set_args_(int i, const _Tp0& a0) { return set(i, a0); }
    template<typename _Tp0, typename... _Tps> inline
    int set_args_(int i, const _Tp0& a0, const _Tps&... rest_args) { i = set(i, a0); return set_args_(i, rest_args...); }
public:
    /** @brief Setup OpenCL Kernel arguments.
    Avoid direct using of set(i, ...) methods.
    @code
    bool ok = kernel
        .args(
            srcUMat, dstUMat,
            (float)some_float_param
        ).run(ndims, globalSize, localSize);
    if (!ok) return false;
    @endcode
    */
    template<typename... _Tps> inline
    Kernel& args(const _Tps&... kernel_args) { set_args_(0, kernel_args...); return *this; }


    /** @brief Run the OpenCL kernel.
    @param dims the work problem dimensions. It is the length of globalsize and localsize. It can be either 1, 2 or 3.
    @param globalsize work items for each dimension. It is not the final globalsize passed to
      OpenCL. Each dimension will be adjusted to the nearest integer divisible by the corresponding
      value in localsize. If localsize is NULL, it will still be adjusted depending on dims. The
      adjusted values are greater than or equal to the original values.
    @param localsize work-group size for each dimension.
    @param sync specify whether to wait for OpenCL computation to finish before return.
    @param q command queue
    */
    bool run(int dims, size_t globalsize[],
             size_t localsize[], bool sync, const Queue& q=Queue());
    bool runTask(bool sync, const Queue& q=Queue());

    /** @brief Similar to synchronized run() call with returning of kernel execution time
     * Separate OpenCL command queue may be used (with CL_QUEUE_PROFILING_ENABLE)
     * @return Execution time in nanoseconds or negative number on error
     */
    int64 runProfiling(int dims, size_t globalsize[], size_t localsize[], const Queue& q=Queue());

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
    Program(const Program& prog);

    Program& operator = (const Program& prog);
    ~Program();

    bool create(const ProgramSource& src,
                const String& buildflags, String& errmsg);

    void* ptr() const;

    /**
     * @brief Query device-specific program binary.
     *
     * Returns RAW OpenCL executable binary without additional attachments.
     *
     * @sa ProgramSource::fromBinary
     *
     * @param[out] binary output buffer
     */
    void getBinary(std::vector<char>& binary) const;

    struct Impl; friend struct Impl;
    inline Impl* getImpl() const { return (Impl*)p; }
    inline bool empty() const { return !p; }
protected:
    Impl* p;
public:
#ifndef OPENCV_REMOVE_DEPRECATED_API
    // TODO Remove this
    CV_DEPRECATED bool read(const String& buf, const String& buildflags); // removed, use ProgramSource instead
    CV_DEPRECATED bool write(String& buf) const; // removed, use getBinary() method instead (RAW OpenCL binary)
    CV_DEPRECATED const ProgramSource& source() const; // implementation removed
    CV_DEPRECATED String getPrefix() const; // deprecated, implementation replaced
    CV_DEPRECATED static String getPrefix(const String& buildflags); // deprecated, implementation replaced
#endif
};


class CV_EXPORTS ProgramSource
{
public:
    typedef uint64 hash_t; // deprecated

    ProgramSource();
    explicit ProgramSource(const String& module, const String& name, const String& codeStr, const String& codeHash);
    explicit ProgramSource(const String& prog); // deprecated
    explicit ProgramSource(const char* prog); // deprecated
    ~ProgramSource();
    ProgramSource(const ProgramSource& prog);
    ProgramSource& operator = (const ProgramSource& prog);

    const String& source() const; // deprecated
    hash_t hash() const; // deprecated


    /** @brief Describe OpenCL program binary.
     * Do not call clCreateProgramWithBinary() and/or clBuildProgram().
     *
     * Caller should guarantee binary buffer lifetime greater than ProgramSource object (and any of its copies).
     *
     * This kind of binary is not portable between platforms in general - it is specific to OpenCL vendor / device / driver version.
     *
     * @param module name of program owner module
     * @param name unique name of program (module+name is used as key for OpenCL program caching)
     * @param binary buffer address. See buffer lifetime requirement in description.
     * @param size buffer size
     * @param buildOptions additional program-related build options passed to clBuildProgram()
     * @return created ProgramSource object
     */
    static ProgramSource fromBinary(const String& module, const String& name,
            const unsigned char* binary, const size_t size,
            const cv::String& buildOptions = cv::String());

    /** @brief Describe OpenCL program in SPIR format.
     * Do not call clCreateProgramWithBinary() and/or clBuildProgram().
     *
     * Supports SPIR 1.2 by default (pass '-spir-std=X.Y' in buildOptions to override this behavior)
     *
     * Caller should guarantee binary buffer lifetime greater than ProgramSource object (and any of its copies).
     *
     * Programs in this format are portable between OpenCL implementations with 'khr_spir' extension:
     * https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/cl_khr_spir.html
     * (but they are not portable between different platforms: 32-bit / 64-bit)
     *
     * Note: these programs can't support vendor specific extensions, like 'cl_intel_subgroups'.
     *
     * @param module name of program owner module
     * @param name unique name of program (module+name is used as key for OpenCL program caching)
     * @param binary buffer address. See buffer lifetime requirement in description.
     * @param size buffer size
     * @param buildOptions additional program-related build options passed to clBuildProgram()
     *        (these options are added automatically: '-x spir' and '-spir-std=1.2')
     * @return created ProgramSource object.
     */
    static ProgramSource fromSPIR(const String& module, const String& name,
            const unsigned char* binary, const size_t size,
            const cv::String& buildOptions = cv::String());

    //OpenCL 2.1+ only
    //static Program fromSPIRV(const String& module, const String& name,
    //        const unsigned char* binary, const size_t size,
    //        const cv::String& buildOptions = cv::String());

    struct Impl; friend struct Impl;
    inline Impl* getImpl() const { return (Impl*)p; }
    inline bool empty() const { return !p; }
protected:
    Impl* p;
};

class CV_EXPORTS PlatformInfo
{
public:
    PlatformInfo();
    /**
     * @param id pointer cl_platform_id (cl_platform_id*)
     */
    explicit PlatformInfo(void* id);
    ~PlatformInfo();

    PlatformInfo(const PlatformInfo& i);
    PlatformInfo& operator =(const PlatformInfo& i);

    String name() const;
    String vendor() const;

    /// See CL_PLATFORM_VERSION
    String version() const;
    int versionMajor() const;
    int versionMinor() const;

    int deviceNumber() const;
    void getDevice(Device& device, int d) const;

    struct Impl;
    bool empty() const { return !p; }
protected:
    Impl* p;
};

CV_EXPORTS const char* convertTypeStr(int sdepth, int ddepth, int cn, char* buf);
CV_EXPORTS const char* typeToStr(int t);
CV_EXPORTS const char* memopTypeToStr(int t);
CV_EXPORTS const char* vecopTypeToStr(int t);
CV_EXPORTS const char* getOpenCLErrorString(int errorCode);
CV_EXPORTS String kernelToStr(InputArray _kernel, int ddepth = -1, const char * name = NULL);
CV_EXPORTS void getPlatfomsInfo(std::vector<PlatformInfo>& platform_info);


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
                                         OclVectorStrategy strat = OCL_VECTOR_DEFAULT);

CV_EXPORTS int checkOptimalVectorWidth(const int *vectorWidths,
                                       InputArray src1, InputArray src2 = noArray(), InputArray src3 = noArray(),
                                       InputArray src4 = noArray(), InputArray src5 = noArray(), InputArray src6 = noArray(),
                                       InputArray src7 = noArray(), InputArray src8 = noArray(), InputArray src9 = noArray(),
                                       OclVectorStrategy strat = OCL_VECTOR_DEFAULT);

// with OCL_VECTOR_MAX strategy
CV_EXPORTS int predictOptimalVectorWidthMax(InputArray src1, InputArray src2 = noArray(), InputArray src3 = noArray(),
                                            InputArray src4 = noArray(), InputArray src5 = noArray(), InputArray src6 = noArray(),
                                            InputArray src7 = noArray(), InputArray src8 = noArray(), InputArray src9 = noArray());

CV_EXPORTS void buildOptionsAddMatrixDescription(String& buildOptions, const String& name, InputArray _m);

class CV_EXPORTS Image2D
{
public:
    Image2D();

    /**
    @param src UMat object from which to get image properties and data
    @param norm flag to enable the use of normalized channel data types
    @param alias flag indicating that the image should alias the src UMat. If true, changes to the
        image or src will be reflected in both objects.
    */
    explicit Image2D(const UMat &src, bool norm = false, bool alias = false);
    Image2D(const Image2D & i);
    ~Image2D();

    Image2D & operator = (const Image2D & i);

    /** Indicates if creating an aliased image should succeed.
    Depends on the underlying platform and the dimensions of the UMat.
    */
    static bool canCreateAlias(const UMat &u);

    /** Indicates if the image format is supported.
    */
    static bool isFormatSupported(int depth, int cn, bool norm);

    void* ptr() const;
protected:
    struct Impl;
    Impl* p;
};

class CV_EXPORTS Timer
{
public:
    Timer(const Queue& q);
    ~Timer();
    void start();
    void stop();

    uint64 durationNS() const; //< duration in nanoseconds

protected:
    struct Impl;
    Impl* const p;

private:
    Timer(const Timer&); // disabled
    Timer& operator=(const Timer&); // disabled
};

CV_EXPORTS MatAllocator* getOpenCLAllocator();


class CV_EXPORTS_W OpenCLExecutionContext
{
public:
    OpenCLExecutionContext() = default;
    ~OpenCLExecutionContext() = default;

    OpenCLExecutionContext(const OpenCLExecutionContext&) = default;
    OpenCLExecutionContext(OpenCLExecutionContext&&) = default;

    OpenCLExecutionContext& operator=(const OpenCLExecutionContext&) = default;
    OpenCLExecutionContext& operator=(OpenCLExecutionContext&&) = default;

    /** Get associated ocl::Context */
    Context& getContext() const;
    /** Get the single default associated ocl::Device */
    Device& getDevice() const;
    /** Get the single ocl::Queue that is associated with the ocl::Context and
     *  the single default ocl::Device
     */
    Queue& getQueue() const;

    bool useOpenCL() const;
    void setUseOpenCL(bool flag);

    /** Get OpenCL execution context of current thread.
     *
     * Initialize OpenCL execution context if it is empty
     * - create new
     * - reuse context of the main thread (threadID = 0)
     */
    static OpenCLExecutionContext& getCurrent();

    /** Get OpenCL execution context of current thread (can be empty) */
    static OpenCLExecutionContext& getCurrentRef();

    /** Bind this OpenCL execution context to current thread.
     *
     * Context can't be empty.
     *
     * @note clFinish is not called for queue of previous execution context
     */
    void bind() const;

    /** Creates new execution context with same OpenCV context and device
     *
     * @param q OpenCL queue
     */
    OpenCLExecutionContext cloneWithNewQueue(const ocl::Queue& q) const;
    /** @overload */
    OpenCLExecutionContext cloneWithNewQueue() const;

    /** @brief Creates OpenCL execution context
     * OpenCV will check if available OpenCL platform has platformName name, then assign context to
     * OpenCV and call `clRetainContext` function. The deviceID device will be used as target device and
     * new command queue will be created.
     *
     * @note Lifetime of passed handles is transferred to OpenCV wrappers on success
     *
     * @param platformName name of OpenCL platform to attach, this string is used to check if platform is available to OpenCV at runtime
     * @param platformID ID of platform attached context was created for (cl_platform_id)
     * @param context OpenCL context to be attached to OpenCV (cl_context)
     * @param deviceID OpenCL device (cl_device_id)
     */
    static OpenCLExecutionContext create(const std::string& platformName, void* platformID, void* context, void* deviceID);

    /** @brief Creates OpenCL execution context
     *
     * @param context non-empty OpenCL context
     * @param device non-empty OpenCL device (must be a part of context)
     * @param queue non-empty OpenCL queue for provided context and device
     */
    static OpenCLExecutionContext create(const Context& context, const Device& device, const ocl::Queue& queue);
    /** @overload */
    static OpenCLExecutionContext create(const Context& context, const Device& device);

    struct Impl;
    inline bool empty() const { return !p; }
    void release();
protected:
    std::shared_ptr<Impl> p;
};

class OpenCLExecutionContextScope
{
    OpenCLExecutionContext ctx_;
public:
    inline OpenCLExecutionContextScope(const OpenCLExecutionContext& ctx)
    {
        CV_Assert(!ctx.empty());
        ctx_ = OpenCLExecutionContext::getCurrentRef();
        ctx.bind();
    }

    inline ~OpenCLExecutionContextScope()
    {
        if (!ctx_.empty())
        {
            ctx_.bind();
        }
    }
};

#ifdef __OPENCV_BUILD
namespace internal {

CV_EXPORTS bool isOpenCLForced();
#define OCL_FORCE_CHECK(condition) (cv::ocl::internal::isOpenCLForced() || (condition))

CV_EXPORTS bool isPerformanceCheckBypassed();
#define OCL_PERFORMANCE_CHECK(condition) (cv::ocl::internal::isPerformanceCheckBypassed() || (condition))

CV_EXPORTS bool isCLBuffer(UMat& u);

} // namespace internal
#endif

//! @}

}}

#endif
