// =============================================================
// cuda_backend.cpp
// GPU HAL CUDA backend plugin.
//
// Implements cv::hal::Backend for CUDA.
// Loaded at runtime by hal_backend.cpp via dlopen.
// Wraps existing cv::cuda:: functions internally.
// User code never calls this file directly.
// =============================================================

#include "opencv2/core.hpp"
#include "opencv2/core/hal/backend.hpp"
#include "opencv2/core/hal/backend_registry.hpp"
#include "opencv2/core/cuda.hpp"

#ifdef HAVE_OPENCV_CUDAWARPING
#include "opencv2/cudawarping.hpp"
#endif

#ifdef HAVE_OPENCV_CUDAARITHM
#include "opencv2/cudaarithm.hpp"
#endif

#ifdef HAVE_OPENCV_CUDAFILTERS
#include "opencv2/cudafilters.hpp"
#endif

#ifdef HAVE_OPENCV_CUDAIMGPROC
#include "opencv2/cudaimgproc.hpp"
#endif

namespace cv { namespace hal {

// =============================================================
// Helper — UMat → GpuMat (zero copy)
//
// Extracts the CUDA device pointer and step from UMatData
// and wraps them in a GpuMat header.
// No cudaMemcpy — both point to the same VRAM bytes.
// =============================================================
static cuda::GpuMat extractGpuMat(const UMat& u)
{
    CV_Assert(u.u != nullptr);
    CV_Assert(u.u->data != nullptr);

    // GpuMat wraps the same device pointer
    // that the CUDA allocator put in UMatData::data
    cuda::GpuMat gpu;
    gpu.rows    = u.rows;
    gpu.cols    = u.cols;
    gpu.step    = u.step[0];
    gpu.data    = u.u->data;
    gpu.flags   = u.flags;
    // refcount not managed here —
    // UMat owns the memory lifetime
    gpu.refcount  = nullptr;
    gpu.datastart = u.u->data;
    gpu.dataend   = u.u->data + u.u->size;
    return gpu;
}

// =============================================================
// Helper — GpuMat result → OutputArray
//
// Downloads the CUDA result into the caller's OutputArray.
// gpu.download(dst) handles both Mat and UMat destinations.
// Full zero-copy output requires a CudaAllocator (Phase 4
// extension) — download is correct behaviour for now.
// =============================================================
static void wrapResultIntoUMat(const cuda::GpuMat& gpu,
                                OutputArray dst,
                                Backend* /*backend*/)
{
    gpu.download(dst);
}

// =============================================================
// CudaBackend
// Implements the Backend interface for CUDA.
// =============================================================
class CudaBackend : public Backend
{
public:

    // ---------------------------------------------------------
    // support()
    // Returns true for operations this backend can handle.
    // Phase 4: resize only.
    // Phase 7 will add more operations.
    // ---------------------------------------------------------
    bool support(int op_id) const CV_OVERRIDE
    {
        switch (op_id)
        {
#ifdef HAVE_OPENCV_CUDAWARPING
            case GPU_OP_RESIZE:
                return true;
#endif
            default:
                return false;
        }
    }

    // ---------------------------------------------------------
    // run()
    // Dispatches to the correct cuda:: function.
    // Parameters passed from CV_GPU_RUN:
    //   param1  = dsize.width
    //   param2  = dsize.height
    //   fparam1 = inv_scale_x
    //   fparam2 = inv_scale_y
    // ---------------------------------------------------------
    bool run(int          op_id,
             InputArray   src,
             OutputArray  dst,
             int          param1  = 0,
             int          param2  = 0,
             double       fparam1 = 0.0,
             double       fparam2 = 0.0) CV_OVERRIDE
    {
        switch (op_id)
        {

#ifdef HAVE_OPENCV_CUDAWARPING
        case GPU_OP_RESIZE:
        {
            // extract source GpuMat from UMat — zero copy
            UMat src_umat = src.getUMat();
            cuda::GpuMat gpu_src = extractGpuMat(src_umat);

            // build destination size
            Size dsize(param1, param2);

            // if dsize is empty use scale factors
            if (dsize.empty())
            {
                dsize = Size(
                    saturate_cast<int>(
                        gpu_src.cols * fparam1),
                    saturate_cast<int>(
                        gpu_src.rows * fparam2));
            }

            // run cuda::resize
            cuda::GpuMat gpu_dst;
            cuda::resize(gpu_src, gpu_dst, dsize,
                         fparam1, fparam2,
                         INTER_LINEAR);

            // wrap result back into UMat
            wrapResultIntoUMat(gpu_dst, dst, this);
            return true;
        }
#endif

        default:
            return false;
        }
    }

    // ---------------------------------------------------------
    // allocator()
    // Returns nullptr for now.
    // Phase 4 extension: return a CudaAllocator that
    // calls cudaMallocPitch for new UMat allocations.
    // ---------------------------------------------------------
    MatAllocator* allocator() const CV_OVERRIDE
    {
        return nullptr;
    }
};

}} // cv::hal

// =============================================================
// Factory function — dlopen entry point
//
// Called by hal_backend.cpp after dlopen loads this plugin.
// Must be extern "C" to prevent C++ name mangling.
// dlsym looks for exactly: "cv_hal_createCudaBackend"
// =============================================================
// Forward declaration suppresses -Wmissing-declarations
extern "C" CV_EXPORTS cv::hal::Backend* cv_hal_createCudaBackend();

cv::hal::Backend* cv_hal_createCudaBackend()
{
    return new cv::hal::CudaBackend();
}
