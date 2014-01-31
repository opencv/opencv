#ifndef __GPUMAT_CUDA_HPP__
#define __GPUMAT_CUDA_HPP__

#ifndef HAVE_CUDA
typedef void* cudaStream_t;
#endif

class DeviceInfoFuncTable
{
public:
    // cv::DeviceInfo
    virtual size_t sharedMemPerBlock(int id) const = 0;
    virtual void queryMemory(int id, size_t&, size_t&) const = 0;
    virtual size_t freeMemory(int id) const = 0;
    virtual size_t totalMemory(int id) const = 0;
    virtual bool supports(int id, FeatureSet) const = 0;
    virtual bool isCompatible(int id) const = 0;
    virtual std::string name(int id) const = 0;
    virtual int majorVersion(int id) const = 0;
    virtual int minorVersion(int id) const = 0;
    virtual int multiProcessorCount(int id) const = 0;

    virtual int getCudaEnabledDeviceCount() const = 0;
    virtual void setDevice(int) const = 0;
    virtual int getDevice() const = 0;
    virtual void resetDevice() const  = 0;
    virtual bool deviceSupports(FeatureSet) const = 0;

    // cv::TargetArchs
    virtual bool builtWith(FeatureSet) const = 0;
    virtual bool has(int, int) const = 0;
    virtual bool hasPtx(int, int) const = 0;
    virtual bool hasBin(int, int) const = 0;
    virtual bool hasEqualOrLessPtx(int, int) const = 0;
    virtual bool hasEqualOrGreater(int, int) const = 0;
    virtual bool hasEqualOrGreaterPtx(int, int) const = 0;
    virtual bool hasEqualOrGreaterBin(int, int) const = 0;

    virtual void printCudaDeviceInfo(int) const = 0;
    virtual void printShortCudaDeviceInfo(int) const = 0;

    virtual ~DeviceInfoFuncTable() {};
};

class GpuFuncTable
{
public:
    // GpuMat routines
    virtual void copy(const Mat& src, GpuMat& dst) const = 0;
    virtual void copy(const GpuMat& src, Mat& dst) const = 0;
    virtual void copy(const GpuMat& src, GpuMat& dst) const = 0;

    virtual void copyWithMask(const GpuMat& src, GpuMat& dst, const GpuMat& mask) const = 0;

    // gpu::device::convertTo funcs
    virtual void convert(const GpuMat& src, GpuMat& dst, double alpha, double beta, cudaStream_t stream = 0) const = 0;
    virtual void convert(const GpuMat& src, GpuMat& dst) const = 0;

    // for gpu::device::setTo funcs
    virtual void setTo(cv::gpu::GpuMat&, cv::Scalar, const cv::gpu::GpuMat&, cudaStream_t) const = 0;

    virtual void mallocPitch(void** devPtr, size_t* step, size_t width, size_t height) const = 0;
    virtual void free(void* devPtr) const = 0;

    virtual ~GpuFuncTable() {}
};

class EmptyDeviceInfoFuncTable: public DeviceInfoFuncTable
{
public:
    size_t sharedMemPerBlock(int) const { throw_nogpu; return 0; }
    void queryMemory(int, size_t&, size_t&) const { throw_nogpu; }
    size_t freeMemory(int) const { throw_nogpu; return 0; }
    size_t totalMemory(int) const { throw_nogpu; return 0; }
    bool supports(int, FeatureSet) const { throw_nogpu; return false; }
    bool isCompatible(int) const { throw_nogpu; return false; }
    std::string name(int) const { throw_nogpu; return std::string(); }
    int majorVersion(int) const { throw_nogpu; return -1; }
    int minorVersion(int) const { throw_nogpu; return -1; }
    int multiProcessorCount(int) const { throw_nogpu; return -1; }

    int getCudaEnabledDeviceCount() const { return 0; }

    void setDevice(int) const { throw_nogpu; }
    int getDevice() const { throw_nogpu; return 0; }

    void resetDevice() const { throw_nogpu; }

    bool deviceSupports(FeatureSet) const { throw_nogpu; return false; }

    bool builtWith(FeatureSet) const { throw_nogpu; return false; }
    bool has(int, int) const { throw_nogpu; return false; }
    bool hasPtx(int, int) const { throw_nogpu; return false; }
    bool hasBin(int, int) const { throw_nogpu; return false; }
    bool hasEqualOrLessPtx(int, int) const { throw_nogpu; return false; }
    bool hasEqualOrGreater(int, int) const { throw_nogpu; return false; }
    bool hasEqualOrGreaterPtx(int, int) const { throw_nogpu; return false; }
    bool hasEqualOrGreaterBin(int, int) const { throw_nogpu; return false; }

    void printCudaDeviceInfo(int) const
    {
        printf("The library is compiled without CUDA support\n");
    }

    void printShortCudaDeviceInfo(int) const
    {
        printf("The library is compiled without CUDA support\n");
    }
};

class EmptyFuncTable : public GpuFuncTable
{
public:

    void copy(const Mat&, GpuMat&) const { throw_nogpu; }
    void copy(const GpuMat&, Mat&) const { throw_nogpu; }
    void copy(const GpuMat&, GpuMat&) const { throw_nogpu; }

    void copyWithMask(const GpuMat&, GpuMat&, const GpuMat&) const { throw_nogpu; }

    void convert(const GpuMat&, GpuMat&) const { throw_nogpu; }
    void convert(const GpuMat&, GpuMat&, double, double, cudaStream_t stream = 0) const { (void)stream; throw_nogpu; }

    virtual void setTo(cv::gpu::GpuMat&, cv::Scalar, const cv::gpu::GpuMat&, cudaStream_t) const { throw_nogpu; }

    void mallocPitch(void**, size_t*, size_t, size_t) const { throw_nogpu; }
    void free(void*) const {}
};

#if defined(USE_CUDA)

// Disable NPP for this file
//#define USE_NPP
#undef USE_NPP

#define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__, CV_Func)
inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
{
    if (cudaSuccess != err)
        cv::gpu::error(cudaGetErrorString(err), file, line, func);
}

#ifdef USE_NPP

#define nppSafeCall(expr)  ___nppSafeCall(expr, __FILE__, __LINE__, CV_Func)
inline void ___nppSafeCall(int err, const char *file, const int line, const char *func = "")
{
    if (err < 0)
    {
        std::ostringstream msg;
        msg << "NPP API Call Error: " << err;
        cv::gpu::error(msg.str().c_str(), file, line, func);
    }
}

#endif

namespace cv { namespace gpu { namespace device
{
    void copyToWithMask_gpu(PtrStepSzb src, PtrStepSzb dst, size_t elemSize1, int cn, PtrStepSzb mask, bool colorMask, cudaStream_t stream);

    template <typename T>
    void set_to_gpu(PtrStepSzb mat, const T* scalar, int channels, cudaStream_t stream);

    template <typename T>
    void set_to_gpu(PtrStepSzb mat, const T* scalar, PtrStepSzb mask, int channels, cudaStream_t stream);

    void convert_gpu(PtrStepSzb src, int sdepth, PtrStepSzb dst, int ddepth, double alpha, double beta, cudaStream_t stream);
}}}

template <typename T> void kernelSetCaller(GpuMat& src, Scalar s, cudaStream_t stream)
{
    Scalar_<T> sf = s;
    cv::gpu::device::set_to_gpu(src, sf.val, src.channels(), stream);
}

template <typename T> void kernelSetCaller(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream)
{
    Scalar_<T> sf = s;
    cv::gpu::device::set_to_gpu(src, sf.val, mask, src.channels(), stream);
}

#ifdef USE_NPP

template<int n> struct NPPTypeTraits;
template<> struct NPPTypeTraits<CV_8U>  { typedef Npp8u npp_type; };
template<> struct NPPTypeTraits<CV_8S>  { typedef Npp8s npp_type; };
template<> struct NPPTypeTraits<CV_16U> { typedef Npp16u npp_type; };
template<> struct NPPTypeTraits<CV_16S> { typedef Npp16s npp_type; };
template<> struct NPPTypeTraits<CV_32S> { typedef Npp32s npp_type; };
template<> struct NPPTypeTraits<CV_32F> { typedef Npp32f npp_type; };
template<> struct NPPTypeTraits<CV_64F> { typedef Npp64f npp_type; };

#endif

//////////////////////////////////////////////////////////////////////////
// Convert

#ifdef USE_NPP

template<int SDEPTH, int DDEPTH> struct NppConvertFunc
{
    typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
    typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

    typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, dst_t* pDst, int nDstStep, NppiSize oSizeROI);
};
template<int DDEPTH> struct NppConvertFunc<CV_32F, DDEPTH>
{
    typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

    typedef NppStatus (*func_ptr)(const Npp32f* pSrc, int nSrcStep, dst_t* pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);
};

template<int SDEPTH, int DDEPTH, typename NppConvertFunc<SDEPTH, DDEPTH>::func_ptr func> struct NppCvt
{
    typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
    typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

    static void call(const GpuMat& src, GpuMat& dst)
    {
        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;

        nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), dst.ptr<dst_t>(), static_cast<int>(dst.step), sz) );

        cudaSafeCall( cudaDeviceSynchronize() );
    }
};

template<int DDEPTH, typename NppConvertFunc<CV_32F, DDEPTH>::func_ptr func> struct NppCvt<CV_32F, DDEPTH, func>
{
    typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

    static void call(const GpuMat& src, GpuMat& dst)
    {
        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;

        nppSafeCall( func(src.ptr<Npp32f>(), static_cast<int>(src.step), dst.ptr<dst_t>(), static_cast<int>(dst.step), sz, NPP_RND_NEAR) );

        cudaSafeCall( cudaDeviceSynchronize() );
    }
};

#endif

//////////////////////////////////////////////////////////////////////////
// Set

#ifdef USE_NPP

template<int SDEPTH, int SCN> struct NppSetFunc
{
    typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

    typedef NppStatus (*func_ptr)(const src_t values[], src_t* pSrc, int nSrcStep, NppiSize oSizeROI);
};
template<int SDEPTH> struct NppSetFunc<SDEPTH, 1>
{
    typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

    typedef NppStatus (*func_ptr)(src_t val, src_t* pSrc, int nSrcStep, NppiSize oSizeROI);
};
template<int SCN> struct NppSetFunc<CV_8S, SCN>
{
    typedef NppStatus (*func_ptr)(Npp8s values[], Npp8s* pSrc, int nSrcStep, NppiSize oSizeROI);
};
template<> struct NppSetFunc<CV_8S, 1>
{
    typedef NppStatus (*func_ptr)(Npp8s val, Npp8s* pSrc, int nSrcStep, NppiSize oSizeROI);
};

template<int SDEPTH, int SCN, typename NppSetFunc<SDEPTH, SCN>::func_ptr func> struct NppSet
{
    typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

    static void call(GpuMat& src, Scalar s)
    {
        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;

        Scalar_<src_t> nppS = s;

        nppSafeCall( func(nppS.val, src.ptr<src_t>(), static_cast<int>(src.step), sz) );

        cudaSafeCall( cudaDeviceSynchronize() );
    }
};
template<int SDEPTH, typename NppSetFunc<SDEPTH, 1>::func_ptr func> struct NppSet<SDEPTH, 1, func>
{
    typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

    static void call(GpuMat& src, Scalar s)
    {
        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;

        Scalar_<src_t> nppS = s;

        nppSafeCall( func(nppS[0], src.ptr<src_t>(), static_cast<int>(src.step), sz) );

        cudaSafeCall( cudaDeviceSynchronize() );
    }
};

template<int SDEPTH, int SCN> struct NppSetMaskFunc
{
    typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

    typedef NppStatus (*func_ptr)(const src_t values[], src_t* pSrc, int nSrcStep, NppiSize oSizeROI, const Npp8u* pMask, int nMaskStep);
};
template<int SDEPTH> struct NppSetMaskFunc<SDEPTH, 1>
{
    typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

    typedef NppStatus (*func_ptr)(src_t val, src_t* pSrc, int nSrcStep, NppiSize oSizeROI, const Npp8u* pMask, int nMaskStep);
};

template<int SDEPTH, int SCN, typename NppSetMaskFunc<SDEPTH, SCN>::func_ptr func> struct NppSetMask
{
    typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

    static void call(GpuMat& src, Scalar s, const GpuMat& mask)
    {
        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;

        Scalar_<src_t> nppS = s;

        nppSafeCall( func(nppS.val, src.ptr<src_t>(), static_cast<int>(src.step), sz, mask.ptr<Npp8u>(), static_cast<int>(mask.step)) );

        cudaSafeCall( cudaDeviceSynchronize() );
    }
};
template<int SDEPTH, typename NppSetMaskFunc<SDEPTH, 1>::func_ptr func> struct NppSetMask<SDEPTH, 1, func>
{
    typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

    static void call(GpuMat& src, Scalar s, const GpuMat& mask)
    {
        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;

        Scalar_<src_t> nppS = s;

        nppSafeCall( func(nppS[0], src.ptr<src_t>(), static_cast<int>(src.step), sz, mask.ptr<Npp8u>(), static_cast<int>(mask.step)) );

        cudaSafeCall( cudaDeviceSynchronize() );
    }
};

#endif

//////////////////////////////////////////////////////////////////////////
// CopyMasked

#ifdef USE_NPP

template<int SDEPTH> struct NppCopyMaskedFunc
{
    typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

    typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, src_t* pDst, int nDstStep, NppiSize oSizeROI, const Npp8u* pMask, int nMaskStep);
};

template<int SDEPTH, typename NppCopyMaskedFunc<SDEPTH>::func_ptr func> struct NppCopyMasked
{
    typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

    static void call(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t /*stream*/)
    {
        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;

        nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), dst.ptr<src_t>(), static_cast<int>(dst.step), sz, mask.ptr<Npp8u>(), static_cast<int>(mask.step)) );

        cudaSafeCall( cudaDeviceSynchronize() );
    }
};

#endif

template <typename T> static inline bool isAligned(const T* ptr, size_t size)
{
    return reinterpret_cast<size_t>(ptr) % size == 0;
}

namespace cv { namespace gpu { namespace device
{
    void copyWithMask(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t stream = 0);
    void convertTo(const GpuMat& src, GpuMat& dst);
    void convertTo(const GpuMat& src, GpuMat& dst, double alpha, double beta, cudaStream_t stream = 0);
    void setTo(GpuMat& src, Scalar s, cudaStream_t stream);
    void setTo(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream);
    void setTo(GpuMat& src, Scalar s);
    void setTo(GpuMat& src, Scalar s, const GpuMat& mask);

    void copyWithMask(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t stream)
    {
        CV_Assert(src.size() == dst.size() && src.type() == dst.type());
        CV_Assert(src.size() == mask.size() && mask.depth() == CV_8U && (mask.channels() == 1 || mask.channels() == src.channels()));

        cv::gpu::device::copyToWithMask_gpu(src.reshape(1), dst.reshape(1), src.elemSize1(), src.channels(), mask.reshape(1), mask.channels() != 1, stream);
    }

    void convertTo(const GpuMat& src, GpuMat& dst)
    {
        cv::gpu::device::convert_gpu(src.reshape(1), src.depth(), dst.reshape(1), dst.depth(), 1.0, 0.0, 0);
    }

    void convertTo(const GpuMat& src, GpuMat& dst, double alpha, double beta, cudaStream_t stream)
    {
        cv::gpu::device::convert_gpu(src.reshape(1), src.depth(), dst.reshape(1), dst.depth(), alpha, beta, stream);
    }

    void setTo(GpuMat& src, Scalar s, cudaStream_t stream)
    {
        typedef void (*caller_t)(GpuMat& src, Scalar s, cudaStream_t stream);

        static const caller_t callers[] =
        {
            kernelSetCaller<uchar>, kernelSetCaller<schar>, kernelSetCaller<ushort>, kernelSetCaller<short>, kernelSetCaller<int>,
            kernelSetCaller<float>, kernelSetCaller<double>
        };

        callers[src.depth()](src, s, stream);
    }

    void setTo(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream)
    {
        typedef void (*caller_t)(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream);

        static const caller_t callers[] =
        {
            kernelSetCaller<uchar>, kernelSetCaller<schar>, kernelSetCaller<ushort>, kernelSetCaller<short>, kernelSetCaller<int>,
            kernelSetCaller<float>, kernelSetCaller<double>
        };

        callers[src.depth()](src, s, mask, stream);
    }

    void setTo(GpuMat& src, Scalar s)
    {
        setTo(src, s, 0);
    }

    void setTo(GpuMat& src, Scalar s, const GpuMat& mask)
    {
        setTo(src, s, mask, 0);
    }
}}}

class CudaArch
{
public:
    CudaArch()
    {
        fromStr(CUDA_ARCH_BIN, bin);
        fromStr(CUDA_ARCH_PTX, ptx);
        fromStr(CUDA_ARCH_FEATURES, features);
    }

    bool builtWith(FeatureSet feature_set) const
    {
        return !features.empty() && (features.back() >= feature_set);
    }

    bool hasPtx(int major, int minor) const
    {
        return find(ptx.begin(), ptx.end(), major * 10 + minor) != ptx.end();
    }

    bool hasBin(int major, int minor) const
    {
        return find(bin.begin(), bin.end(), major * 10 + minor) != bin.end();
    }

    bool hasEqualOrLessPtx(int major, int minor) const
    {
        return !ptx.empty() && (ptx.front() <= major * 10 + minor);
    }

    bool hasEqualOrGreaterPtx(int major, int minor) const
    {
        return !ptx.empty() && (ptx.back() >= major * 10 + minor);
    }

    bool hasEqualOrGreaterBin(int major, int minor) const
    {
        return !bin.empty() && (bin.back() >= major * 10 + minor);
    }


private:
    void fromStr(const string& set_as_str, vector<int>& arr)
    {
        if (set_as_str.find_first_not_of(" ") == string::npos)
            return;

        istringstream stream(set_as_str);
        int cur_value;

        while (!stream.eof())
        {
            stream >> cur_value;
            arr.push_back(cur_value);
        }

        sort(arr.begin(), arr.end());
    }

    vector<int> bin;
    vector<int> ptx;
    vector<int> features;
};

class DeviceProps
{
public:
    DeviceProps()
    {
        props_.resize(10, 0);
    }

    ~DeviceProps()
    {
        for (size_t i = 0; i < props_.size(); ++i)
        {
            if (props_[i])
                delete props_[i];
        }
        props_.clear();
    }

    cudaDeviceProp* get(int devID)
    {
        if (devID >= (int) props_.size())
            props_.resize(devID + 5, 0);

        if (!props_[devID])
        {
            props_[devID] = new cudaDeviceProp;
            cudaSafeCall( cudaGetDeviceProperties(props_[devID], devID) );
        }

        return props_[devID];
    }
private:
    std::vector<cudaDeviceProp*> props_;
};

DeviceProps deviceProps;
const CudaArch cudaArch;

class CudaDeviceInfoFuncTable : public DeviceInfoFuncTable
{
public:
    size_t sharedMemPerBlock(int id) const
    {
        return deviceProps.get(id)->sharedMemPerBlock;
    }

    void queryMemory(int id, size_t& _totalMemory, size_t& _freeMemory) const
    {
        int prevDeviceID = getDevice();
        if (prevDeviceID != id)
            setDevice(id);

        cudaSafeCall( cudaMemGetInfo(&_freeMemory, &_totalMemory) );

        if (prevDeviceID != id)
            setDevice(prevDeviceID);
    }

    size_t freeMemory(int id) const
    {
        size_t _totalMemory, _freeMemory;
        queryMemory(id, _totalMemory, _freeMemory);
        return _freeMemory;
    }

    size_t totalMemory(int id) const
    {
        size_t _totalMemory, _freeMemory;
        queryMemory(id, _totalMemory, _freeMemory);
        return _totalMemory;
    }

    bool supports(int id, FeatureSet feature_set) const
    {
        int version = majorVersion(id) * 10 + minorVersion(id);
        return version >= feature_set;
    }

    bool isCompatible(int id) const
    {
        // Check PTX compatibility
        if (hasEqualOrLessPtx(majorVersion(id), minorVersion(id)))
            return true;

        // Check BIN compatibility
            for (int i = minorVersion(id); i >= 0; --i)
                if (hasBin(majorVersion(id), i))
                    return true;

                return false;
    }

    std::string name(int id) const
    {
        const cudaDeviceProp* prop = deviceProps.get(id);
        return prop->name;
    }

    int majorVersion(int id) const
    {
        const cudaDeviceProp* prop = deviceProps.get(id);
        return prop->major;
    }

    int minorVersion(int id) const
    {
        const cudaDeviceProp* prop = deviceProps.get(id);
        return prop->minor;
    }

    int multiProcessorCount(int id) const
    {
        const cudaDeviceProp* prop = deviceProps.get(id);
        return prop->multiProcessorCount;
    }

    int getCudaEnabledDeviceCount() const
    {
        int count;
        cudaError_t error = cudaGetDeviceCount( &count );

        if (error == cudaErrorInsufficientDriver)
            return -1;

        if (error == cudaErrorNoDevice)
            return 0;

        cudaSafeCall( error );
        return count;
    }

    void setDevice(int device) const
    {
        cudaSafeCall( cudaSetDevice( device ) );
    }

    int getDevice() const
    {
        int device;
        cudaSafeCall( cudaGetDevice( &device ) );
        return device;
    }

    void resetDevice() const
    {
        cudaSafeCall( cudaDeviceReset() );
    }

    bool builtWith(FeatureSet feature_set) const
    {
        return cudaArch.builtWith(feature_set);
    }

    bool has(int major, int minor) const
    {
        return hasPtx(major, minor) || hasBin(major, minor);
    }

    bool hasPtx(int major, int minor) const
    {
        return cudaArch.hasPtx(major, minor);
    }

    bool hasBin(int major, int minor) const
    {
        return cudaArch.hasBin(major, minor);
    }

    bool hasEqualOrLessPtx(int major, int minor) const
    {
        return cudaArch.hasEqualOrLessPtx(major, minor);
    }

    bool hasEqualOrGreater(int major, int minor) const
    {
        return hasEqualOrGreaterPtx(major, minor) || hasEqualOrGreaterBin(major, minor);
    }

    bool hasEqualOrGreaterPtx(int major, int minor) const
    {
        return cudaArch.hasEqualOrGreaterPtx(major, minor);
    }

    bool hasEqualOrGreaterBin(int major, int minor) const
    {
        return cudaArch.hasEqualOrGreaterBin(major, minor);
    }

    bool deviceSupports(FeatureSet feature_set) const
    {
        static int versions[] =
        {
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        };
        static const int cache_size = static_cast<int>(sizeof(versions) / sizeof(versions[0]));

        const int devId = getDevice();

        int version;

        if (devId < cache_size && versions[devId] >= 0)
            version = versions[devId];
        else
        {
            DeviceInfo dev(devId);
            version = dev.majorVersion() * 10 + dev.minorVersion();
            if (devId < cache_size)
                versions[devId] = version;
        }

        return TargetArchs::builtWith(feature_set) && (version >= feature_set);
    }

    void printCudaDeviceInfo(int device) const
    {
        int count = getCudaEnabledDeviceCount();
        bool valid = (device >= 0) && (device < count);

        int beg = valid ? device   : 0;
        int end = valid ? device+1 : count;

        printf("*** CUDA Device Query (Runtime API) version (CUDART static linking) *** \n\n");
        printf("Device count: %d\n", count);

        int driverVersion = 0, runtimeVersion = 0;
        cudaSafeCall( cudaDriverGetVersion(&driverVersion) );
        cudaSafeCall( cudaRuntimeGetVersion(&runtimeVersion) );

        const char *computeMode[] = {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
               "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
               "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
               "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
               "Unknown",
               NULL
        };

        for(int dev = beg; dev < end; ++dev)
        {
            cudaDeviceProp prop;
            cudaSafeCall( cudaGetDeviceProperties(&prop, dev) );

            printf("\nDevice %d: \"%s\"\n", dev, prop.name);
            printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, driverVersion%100, runtimeVersion/1000, runtimeVersion%100);
            printf("  CUDA Capability Major/Minor version number:    %d.%d\n", prop.major, prop.minor);
            printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n", (float)prop.totalGlobalMem/1048576.0f, (unsigned long long) prop.totalGlobalMem);

        int cores = convertSMVer2Cores(prop.major, prop.minor);
        if (cores > 0)
            printf("  (%2d) Multiprocessors x (%2d) CUDA Cores/MP:     %d CUDA Cores\n", prop.multiProcessorCount, cores, cores * prop.multiProcessorCount);

        printf("  GPU Clock Speed:                               %.2f GHz\n", prop.clockRate * 1e-6f);

        printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
               prop.maxTexture1D, prop.maxTexture2D[0], prop.maxTexture2D[1],
               prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
        printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n",
               prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1],
               prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);

        printf("  Total amount of constant memory:               %u bytes\n", (int)prop.totalConstMem);
        printf("  Total amount of shared memory per block:       %u bytes\n", (int)prop.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", prop.regsPerBlock);
        printf("  Warp size:                                     %d\n", prop.warpSize);
        printf("  Maximum number of threads per block:           %d\n", prop.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1],  prop.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %u bytes\n", (int)prop.memPitch);
        printf("  Texture alignment:                             %u bytes\n", (int)prop.textureAlignment);

        printf("  Concurrent copy and execution:                 %s with %d copy engine(s)\n", (prop.deviceOverlap ? "Yes" : "No"), prop.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n", prop.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", prop.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", prop.canMapHostMemory ? "Yes" : "No");

        printf("  Concurrent kernel execution:                   %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", prop.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support enabled:                %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("  Device is using TCC driver mode:               %s\n", prop.tccDriver ? "Yes" : "No");
        printf("  Device supports Unified Addressing (UVA):      %s\n", prop.unifiedAddressing ? "Yes" : "No");
        printf("  Device PCI Bus ID / PCI location ID:           %d / %d\n", prop.pciBusID, prop.pciDeviceID );
        printf("  Compute Mode:\n");
        printf("      %s \n", computeMode[prop.computeMode]);
        }

        printf("\n");
        printf("deviceQuery, CUDA Driver = CUDART");
        printf(", CUDA Driver Version  = %d.%d", driverVersion / 1000, driverVersion % 100);
        printf(", CUDA Runtime Version = %d.%d", runtimeVersion/1000, runtimeVersion%100);
        printf(", NumDevs = %d\n\n", count);
        fflush(stdout);
    }

    void printShortCudaDeviceInfo(int device) const
    {
        int count = getCudaEnabledDeviceCount();
        bool valid = (device >= 0) && (device < count);

        int beg = valid ? device   : 0;
        int end = valid ? device+1 : count;

        int driverVersion = 0, runtimeVersion = 0;
        cudaSafeCall( cudaDriverGetVersion(&driverVersion) );
        cudaSafeCall( cudaRuntimeGetVersion(&runtimeVersion) );

        for(int dev = beg; dev < end; ++dev)
        {
            cudaDeviceProp prop;
            cudaSafeCall( cudaGetDeviceProperties(&prop, dev) );

            const char *arch_str = prop.major < 2 ? " (not Fermi)" : "";
            printf("Device %d:  \"%s\"  %.0fMb", dev, prop.name, (float)prop.totalGlobalMem/1048576.0f);
            printf(", sm_%d%d%s", prop.major, prop.minor, arch_str);

            int cores = convertSMVer2Cores(prop.major, prop.minor);
            if (cores > 0)
                printf(", %d cores", cores * prop.multiProcessorCount);

            printf(", Driver/Runtime ver.%d.%d/%d.%d\n", driverVersion/1000, driverVersion%100, runtimeVersion/1000, runtimeVersion%100);
        }
        fflush(stdout);
    }

private:
    int convertSMVer2Cores(int major, int minor) const
    {
        // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
        typedef struct {
            int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
            int Cores;
        } SMtoCores;

        SMtoCores gpuArchCoresPerSM[] =  { { 0x10,  8 }, { 0x11,  8 }, { 0x12,  8 }, { 0x13,  8 }, { 0x20, 32 }, { 0x21, 48 }, {0x30, 192}, {0x35, 192}, { -1, -1 }  };

        int index = 0;
        while (gpuArchCoresPerSM[index].SM != -1)
        {
            if (gpuArchCoresPerSM[index].SM == ((major << 4) + minor) )
                return gpuArchCoresPerSM[index].Cores;
            index++;
        }

        return -1;
    }
};

class CudaFuncTable : public GpuFuncTable
{
public:

    void copy(const Mat& src, GpuMat& dst) const
    {
        cudaSafeCall( cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyHostToDevice) );
    }

    void copy(const GpuMat& src, Mat& dst) const
    {
        cudaSafeCall( cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyDeviceToHost) );
    }

    void copy(const GpuMat& src, GpuMat& dst) const
    {
        cudaSafeCall( cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyDeviceToDevice) );
    }

    void copyWithMask(const GpuMat& src, GpuMat& dst, const GpuMat& mask) const
    {
        CV_Assert(src.depth() <= CV_64F && src.channels() <= 4);
        CV_Assert(src.size() == dst.size() && src.type() == dst.type());
        CV_Assert(src.size() == mask.size() && mask.depth() == CV_8U && (mask.channels() == 1 || mask.channels() == src.channels()));

        if (src.depth() == CV_64F)
        {
            if (!TargetArchs::builtWith(NATIVE_DOUBLE) || !DeviceInfo().supports(NATIVE_DOUBLE))
                CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
        }

        typedef void (*func_t)(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t stream);

#ifdef USE_NPP
        static const func_t funcs[7][4] =
        {
            /*  8U */ {NppCopyMasked<CV_8U , nppiCopy_8u_C1MR >::call, cv::gpu::device::copyWithMask, NppCopyMasked<CV_8U , nppiCopy_8u_C3MR >::call, NppCopyMasked<CV_8U , nppiCopy_8u_C4MR >::call},
            /*  8S */ {cv::gpu::device::copyWithMask                ,  cv::gpu::device::copyWithMask, cv::gpu::device::copyWithMask                 , cv::gpu::device::copyWithMask                         },
            /* 16U */ {NppCopyMasked<CV_16U, nppiCopy_16u_C1MR>::call, cv::gpu::device::copyWithMask, NppCopyMasked<CV_16U, nppiCopy_16u_C3MR>::call, NppCopyMasked<CV_16U, nppiCopy_16u_C4MR>::call},
            /* 16S */ {NppCopyMasked<CV_16S, nppiCopy_16s_C1MR>::call, cv::gpu::device::copyWithMask, NppCopyMasked<CV_16S, nppiCopy_16s_C3MR>::call, NppCopyMasked<CV_16S, nppiCopy_16s_C4MR>::call},
            /* 32S */ {NppCopyMasked<CV_32S, nppiCopy_32s_C1MR>::call, cv::gpu::device::copyWithMask, NppCopyMasked<CV_32S, nppiCopy_32s_C3MR>::call, NppCopyMasked<CV_32S, nppiCopy_32s_C4MR>::call},
            /* 32F */ {NppCopyMasked<CV_32F, nppiCopy_32f_C1MR>::call, cv::gpu::device::copyWithMask, NppCopyMasked<CV_32F, nppiCopy_32f_C3MR>::call, NppCopyMasked<CV_32F, nppiCopy_32f_C4MR>::call},
            /* 64F */ {cv::gpu::device::copyWithMask                ,  cv::gpu::device::copyWithMask, cv::gpu::device::copyWithMask                 , cv::gpu::device::copyWithMask                         }
         };

         const func_t func =  mask.channels() == src.channels() ? funcs[src.depth()][src.channels() - 1] : cv::gpu::device::copyWithMask;
#else
        const func_t func = cv::gpu::device::copyWithMask;
#endif

         func(src, dst, mask, 0);
    }

    void convert(const GpuMat& src, GpuMat& dst) const
    {
        typedef void (*func_t)(const GpuMat& src, GpuMat& dst);

#ifdef USE_NPP
        static const func_t funcs[7][7][4] =
        {
            {
                /*  8U ->  8U */ {0, 0, 0, 0},
                /*  8U ->  8S */ {cv::gpu::device::convertTo                        , cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo                                },
                /*  8U -> 16U */ {NppCvt<CV_8U, CV_16U, nppiConvert_8u16u_C1R>::call, cv::gpu::device::convertTo, cv::gpu::device::convertTo, NppCvt<CV_8U, CV_16U, nppiConvert_8u16u_C4R>::call},
                /*  8U -> 16S */ {NppCvt<CV_8U, CV_16S, nppiConvert_8u16s_C1R>::call, cv::gpu::device::convertTo, cv::gpu::device::convertTo, NppCvt<CV_8U, CV_16S, nppiConvert_8u16s_C4R>::call},
                /*  8U -> 32S */ {cv::gpu::device::convertTo                        , cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo                                },
                /*  8U -> 32F */ {NppCvt<CV_8U, CV_32F, nppiConvert_8u32f_C1R>::call, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo                                },
                /*  8U -> 64F */ {cv::gpu::device::convertTo                        , cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo                                }
            },
            {
                /*  8S ->  8U */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /*  8S ->  8S */ {0,0,0,0},
                /*  8S -> 16U */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /*  8S -> 16S */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /*  8S -> 32S */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /*  8S -> 32F */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /*  8S -> 64F */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo}
            },
            {
                /* 16U ->  8U */ {NppCvt<CV_16U, CV_8U , nppiConvert_16u8u_C1R >::call, cv::gpu::device::convertTo, cv::gpu::device::convertTo, NppCvt<CV_16U, CV_8U, nppiConvert_16u8u_C4R>::call},
                /* 16U ->  8S */ {cv::gpu::device::convertTo                                  , cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo                                },
                /* 16U -> 16U */ {0,0,0,0},
                /* 16U -> 16S */ {cv::gpu::device::convertTo                                  , cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo                                },
                /* 16U -> 32S */ {NppCvt<CV_16U, CV_32S, nppiConvert_16u32s_C1R>::call, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo                                },
                /* 16U -> 32F */ {NppCvt<CV_16U, CV_32F, nppiConvert_16u32f_C1R>::call, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo                                },
                /* 16U -> 64F */ {cv::gpu::device::convertTo                                  , cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo                                }
            },
            {
                /* 16S ->  8U */ {NppCvt<CV_16S, CV_8U , nppiConvert_16s8u_C1R >::call, cv::gpu::device::convertTo, cv::gpu::device::convertTo, NppCvt<CV_16S, CV_8U, nppiConvert_16s8u_C4R>::call},
                /* 16S ->  8S */ {cv::gpu::device::convertTo                                  , cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo                                },
                /* 16S -> 16U */ {cv::gpu::device::convertTo                                  , cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo                                },
                /* 16S -> 16S */ {0,0,0,0},
                /* 16S -> 32S */ {NppCvt<CV_16S, CV_32S, nppiConvert_16s32s_C1R>::call, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo                                },
                /* 16S -> 32F */ {NppCvt<CV_16S, CV_32F, nppiConvert_16s32f_C1R>::call, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo                                },
                /* 16S -> 64F */ {cv::gpu::device::convertTo                                  , cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo                                }
            },
            {
                /* 32S ->  8U */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 32S ->  8S */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 32S -> 16U */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 32S -> 16S */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 32S -> 32S */ {0,0,0,0},
                /* 32S -> 32F */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 32S -> 64F */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo}
            },
            {
                /* 32F ->  8U */ {NppCvt<CV_32F, CV_8U , nppiConvert_32f8u_C1R >::call, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 32F ->  8S */ {cv::gpu::device::convertTo                          , cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 32F -> 16U */ {NppCvt<CV_32F, CV_16U, nppiConvert_32f16u_C1R>::call, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 32F -> 16S */ {NppCvt<CV_32F, CV_16S, nppiConvert_32f16s_C1R>::call, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 32F -> 32S */ {cv::gpu::device::convertTo                          , cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 32F -> 32F */ {0,0,0,0},
                /* 32F -> 64F */ {cv::gpu::device::convertTo                          , cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo}
            },
            {
                /* 64F ->  8U */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 64F ->  8S */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 64F -> 16U */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 64F -> 16S */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 64F -> 32S */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 64F -> 32F */ {cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo, cv::gpu::device::convertTo},
                /* 64F -> 64F */ {0,0,0,0}
            }
        };
#endif

        CV_Assert(src.depth() <= CV_64F && src.channels() <= 4);
        CV_Assert(dst.depth() <= CV_64F);
        CV_Assert(src.size() == dst.size() && src.channels() == dst.channels());

        if (src.depth() == CV_64F || dst.depth() == CV_64F)
        {
            if (!TargetArchs::builtWith(NATIVE_DOUBLE) || !DeviceInfo().supports(NATIVE_DOUBLE))
                CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
        }

        bool aligned = isAligned(src.data, 16) && isAligned(dst.data, 16);
        if (!aligned)
        {
            cv::gpu::device::convertTo(src, dst);
            return;
        }

#ifdef USE_NPP
        const func_t func = funcs[src.depth()][dst.depth()][src.channels() - 1];
        CV_DbgAssert(func != 0);
#else
        const func_t func = cv::gpu::device::convertTo;
#endif

        func(src, dst);
    }

    void convert(const GpuMat& src, GpuMat& dst, double alpha, double beta, cudaStream_t stream) const
    {
        CV_Assert(src.depth() <= CV_64F && src.channels() <= 4);
        CV_Assert(dst.depth() <= CV_64F);

        if (src.depth() == CV_64F || dst.depth() == CV_64F)
        {
            if (!TargetArchs::builtWith(NATIVE_DOUBLE) || !DeviceInfo().supports(NATIVE_DOUBLE))
                CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
        }

        cv::gpu::device::convertTo(src, dst, alpha, beta, stream);
    }

    void setTo(GpuMat& m, Scalar s, const GpuMat& mask, cudaStream_t stream) const
    {
        if (mask.empty())
        {
            if (s[0] == 0.0 && s[1] == 0.0 && s[2] == 0.0 && s[3] == 0.0)
            {
                cudaSafeCall( cudaMemset2D(m.data, m.step, 0, m.cols * m.elemSize(), m.rows) );
                return;
            }

            if (m.depth() == CV_8U)
            {
                int cn = m.channels();

                if (cn == 1 || (cn == 2 && s[0] == s[1]) || (cn == 3 && s[0] == s[1] && s[0] == s[2]) || (cn == 4 && s[0] == s[1] && s[0] == s[2] && s[0] == s[3]))
                {
                    int val = saturate_cast<uchar>(s[0]);
                    cudaSafeCall( cudaMemset2D(m.data, m.step, val, m.cols * m.elemSize(), m.rows) );
                    return;
                }
            }

            typedef void (*func_t)(GpuMat& src, Scalar s);

#ifdef USE_NPP
            static const func_t funcs[7][4] =
            {
                {NppSet<CV_8U , 1, nppiSet_8u_C1R >::call, cv::gpu::device::setTo                  , cv::gpu::device::setTo                        , NppSet<CV_8U , 4, nppiSet_8u_C4R >::call},
                {cv::gpu::device::setTo                  , cv::gpu::device::setTo                  , cv::gpu::device::setTo                        , cv::gpu::device::setTo                          },
                {NppSet<CV_16U, 1, nppiSet_16u_C1R>::call, NppSet<CV_16U, 2, nppiSet_16u_C2R>::call, cv::gpu::device::setTo                        , NppSet<CV_16U, 4, nppiSet_16u_C4R>::call},
                {NppSet<CV_16S, 1, nppiSet_16s_C1R>::call, NppSet<CV_16S, 2, nppiSet_16s_C2R>::call, cv::gpu::device::setTo                        , NppSet<CV_16S, 4, nppiSet_16s_C4R>::call},
                {NppSet<CV_32S, 1, nppiSet_32s_C1R>::call, cv::gpu::device::setTo                  , cv::gpu::device::setTo                        , NppSet<CV_32S, 4, nppiSet_32s_C4R>::call},
                {NppSet<CV_32F, 1, nppiSet_32f_C1R>::call, cv::gpu::device::setTo                  , cv::gpu::device::setTo                        , NppSet<CV_32F, 4, nppiSet_32f_C4R>::call},
                {cv::gpu::device::setTo                  , cv::gpu::device::setTo                  , cv::gpu::device::setTo                        , cv::gpu::device::setTo                          }
            };
#endif

            CV_Assert(m.depth() <= CV_64F && m.channels() <= 4);

            if (m.depth() == CV_64F)
            {
                if (!TargetArchs::builtWith(NATIVE_DOUBLE) || !DeviceInfo().supports(NATIVE_DOUBLE))
                    CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
            }

#ifdef USE_NPP
        const func_t func = funcs[m.depth()][m.channels() - 1];
#else
        const func_t func = cv::gpu::device::setTo;
#endif

            if (stream)
                cv::gpu::device::setTo(m, s, stream);
            else
                func(m, s);
        }
        else
        {
            typedef void (*func_t)(GpuMat& src, Scalar s, const GpuMat& mask);

#ifdef USE_NPP
            static const func_t funcs[7][4] =
            {
                {NppSetMask<CV_8U , 1, nppiSet_8u_C1MR >::call, cv::gpu::device::setTo, cv::gpu::device::setTo, NppSetMask<CV_8U , 4, nppiSet_8u_C4MR >::call},
                {cv::gpu::device::setTo                       , cv::gpu::device::setTo, cv::gpu::device::setTo, cv::gpu::device::setTo                               },
                {NppSetMask<CV_16U, 1, nppiSet_16u_C1MR>::call, cv::gpu::device::setTo, cv::gpu::device::setTo, NppSetMask<CV_16U, 4, nppiSet_16u_C4MR>::call},
                {NppSetMask<CV_16S, 1, nppiSet_16s_C1MR>::call, cv::gpu::device::setTo, cv::gpu::device::setTo, NppSetMask<CV_16S, 4, nppiSet_16s_C4MR>::call},
                {NppSetMask<CV_32S, 1, nppiSet_32s_C1MR>::call, cv::gpu::device::setTo, cv::gpu::device::setTo, NppSetMask<CV_32S, 4, nppiSet_32s_C4MR>::call},
                {NppSetMask<CV_32F, 1, nppiSet_32f_C1MR>::call, cv::gpu::device::setTo, cv::gpu::device::setTo, NppSetMask<CV_32F, 4, nppiSet_32f_C4MR>::call},
                {cv::gpu::device::setTo                       , cv::gpu::device::setTo, cv::gpu::device::setTo, cv::gpu::device::setTo                               }
            };
#endif

            CV_Assert(m.depth() <= CV_64F && m.channels() <= 4);

            if (m.depth() == CV_64F)
            {
                if (!TargetArchs::builtWith(NATIVE_DOUBLE) || !DeviceInfo().supports(NATIVE_DOUBLE))
                    CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
            }

#ifdef USE_NPP
        const func_t func = funcs[m.depth()][m.channels() - 1];
#else
        const func_t func = cv::gpu::device::setTo;
#endif

            if (stream)
                cv::gpu::device::setTo(m, s, mask, stream);
            else
                func(m, s, mask);
        }
    }

    void mallocPitch(void** devPtr, size_t* step, size_t width, size_t height) const
    {
        cudaSafeCall( cudaMallocPitch(devPtr, step, width, height) );
    }

    void free(void* devPtr) const
    {
        cudaFree(devPtr);
    }
};
#endif
#endif
