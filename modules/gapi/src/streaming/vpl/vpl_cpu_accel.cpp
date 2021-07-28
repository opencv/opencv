#ifdef HAVE_ONEVPL
#include <stdlib.h>
#include <exception>

#include "streaming/vpl/vpl_cpu_accel.hpp"
#include "streaming/vpl/vpl_utils.hpp"
#include "logger.hpp"

#ifdef _WIN32
    #include <windows.h>
    #include <sysinfoapi.h>
#endif
namespace cv {
namespace gapi {
namespace wip {

class VPLCPUAccelerationPolicy::MediaFrameAdapter : public cv::MediaFrame::IAdapter {
public:
    MediaFrameAdapter(mfxFrameSurface1* parent);
    ~MediaFrameAdapter();
    cv::GFrameDesc meta() const override;
    MediaFrame::View access(MediaFrame::Access) override;
    
    // FIXME: design a better solution
    // The default implementation does nothing
    cv::util::any blobParams() const override;
    void serialize(cv::gapi::s11n::IOStream&) override;
    void deserialize(cv::gapi::s11n::IIStream&) override;
private:
    mfxFrameSurface1* parent_surface_ptr;
};


VPLCPUAccelerationPolicy::MediaFrameAdapter::MediaFrameAdapter(mfxFrameSurface1* parent):
    parent_surface_ptr(parent) {

    GAPI_Assert(parent_surface_ptr && "Surface is nullptr");
    parent_surface_ptr->Data.Locked++;
    GAPI_LOG_DEBUG(nullptr, "surface: " << parent_surface_ptr <<
                            ", locked times: " << parent_surface_ptr->Data.Locked + 1);
}

VPLCPUAccelerationPolicy::MediaFrameAdapter::~MediaFrameAdapter() {
    GAPI_Assert(parent_surface_ptr && "Surface is nullptr");
    parent_surface_ptr->Data.Locked--;
    GAPI_LOG_DEBUG(nullptr, "surface: " << parent_surface_ptr <<
                            ", locked times: " << parent_surface_ptr->Data.Locked);
}

cv::GFrameDesc VPLCPUAccelerationPolicy::MediaFrameAdapter::meta() const {
    GFrameDesc desc;
    switch(parent_surface_ptr->Info.FourCC)
    {
        case MFX_FOURCC_I420:
            throw std::runtime_error("MediaFrame doesn't support I420 type");
            break;
        case MFX_FOURCC_NV12:
            desc.fmt = MediaFormat::NV12;
            break;
        default:
            throw std::runtime_error("MediaFrame unknown 'fmt' type: " + std::to_string(parent_surface_ptr->Info.FourCC));
    }
    
    desc.size = cv::Size{parent_surface_ptr->Info.Width, parent_surface_ptr->Info.Height};
    return desc;
}

MediaFrame::View VPLCPUAccelerationPolicy::MediaFrameAdapter::access(MediaFrame::Access mode) {
    (void)mode;

    using stride_t = typename cv::MediaFrame::View::Strides::value_type;
    GAPI_Assert(parent_surface_ptr->Data.Pitch >= 0 && "Pitch is less 0");

    stride_t pitch = static_cast<stride_t>(parent_surface_ptr->Data.Pitch);
    switch(parent_surface_ptr->Info.FourCC) {
        case MFX_FOURCC_I420:
        {
            cv::MediaFrame::View::Ptrs pp = {
                parent_surface_ptr->Data.Y,
                parent_surface_ptr->Data.U,
                parent_surface_ptr->Data.V,
                nullptr
                };
            cv::MediaFrame::View::Strides ss = {
                    pitch,
                    pitch / 2,
                    pitch / 2, 0u
                };
            return cv::MediaFrame::View(std::move(pp), std::move(ss));
        }
        case MFX_FOURCC_NV12:
        {
            cv::MediaFrame::View::Ptrs pp = {
                parent_surface_ptr->Data.Y,
                parent_surface_ptr->Data.UV, nullptr, nullptr
                };
            cv::MediaFrame::View::Strides ss = {
                    pitch,
                    pitch, 0u, 0u
                };
            return cv::MediaFrame::View(std::move(pp), std::move(ss));
        }
            break;
        default:
            throw std::runtime_error("MediaFrame unknown 'fmt' type: " + std::to_string(parent_surface_ptr->Info.FourCC));
    }
}

cv::util::any VPLCPUAccelerationPolicy::MediaFrameAdapter::blobParams() const {
    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
}

void VPLCPUAccelerationPolicy::MediaFrameAdapter::serialize(cv::gapi::s11n::IOStream&) {
    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
}
void VPLCPUAccelerationPolicy::MediaFrameAdapter::deserialize(cv::gapi::s11n::IIStream&) {
    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
}

    
VPLCPUAccelerationPolicy::VPLCPUAccelerationPolicy(mfxSession session) {
    (void)session;
    //MFXVideoCORE_SetFrameAllocator(session, mfxFrameAllocator instance)
    GAPI_LOG_INFO(nullptr, "VPLCPUAccelerationPolicy initialized, session: " << session);
}

VPLCPUAccelerationPolicy::~VPLCPUAccelerationPolicy() {
    GAPI_LOG_INFO(nullptr, "VPLCPUAccelerationPolicy deinitialized");
}

VPLCPUAccelerationPolicy::pool_key_t
VPLCPUAccelerationPolicy::create_surface_pool(size_t pool_size, size_t surface_size_bytes,
                                              surface_ptr_ctr_t creator) {
    GAPI_LOG_DEBUG(nullptr, "pool size: " << pool_size << ", surface size bytes: " << surface_size_bytes);

    // create empty pool
    pool_t pool;
    pool.reserve(pool_size);

    // allocate workplace memory area
    size_t preallocated_raw_bytes = pool_size * surface_size_bytes;
    size_t page_size_bytes = 4 * 1024;
    void *preallocated_pool_memory_ptr = nullptr;

#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    page_size_bytes = sysInfo.dwPageSize;

    GAPI_LOG_DEBUG(nullptr, "page size: " << page_size_bytes << ", preallocated_raw_bytes: " << preallocated_raw_bytes);
    preallocated_pool_memory_ptr = _aligned_malloc(preallocated_raw_bytes, page_size_bytes);
#endif

    if (!preallocated_pool_memory_ptr) {
        throw std::runtime_error("VPLCPUAccelerationPolicy::create_surface_pool - failed: not enough memory."
                                 "Requested surface count: " + std::to_string(pool_size) +
                                 ", surface bytes: " + std::to_string(surface_size_bytes));
    }

    // fill pool with surfaces
    size_t i = 0;
    try {
        for (; i < pool_size; i++) {
            size_t preallocated_mem_offset = static_cast<size_t>(i) * surface_size_bytes;

            surface_ptr_t surf_ptr = creator(preallocated_pool_memory_ptr,
                                             preallocated_mem_offset,
                                             preallocated_raw_bytes);
            pool.push_back(std::move(surf_ptr));
        }
    } catch (const std::exception& ex) {
        free(preallocated_pool_memory_ptr);
        throw std::runtime_error(std::string("VPLCPUAccelerationPolicy::create_surface_pool - ") +
                                 "cannot construct surface index: " + std::to_string(i) + ", error: " +
                                 ex.what() +
                                 "Requested surface count: " + std::to_string(pool_size) +
                                 ", surface bytes: " + std::to_string(surface_size_bytes));
    }

    // remember pool by key
    GAPI_LOG_INFO(nullptr, "New pool allocated, key: " << preallocated_pool_memory_ptr <<
                           ", surface count: " << pool.size() <<
                           ", surface size bytes: " << surface_size_bytes);
    try {
        if (!pool_table.emplace(preallocated_pool_memory_ptr, std::move(pool)).second) {
            throw std::runtime_error(std::string("VPLCPUAccelerationPolicy::create_surface_pool - ") +
                                     "cannot insert pool, table size: " + std::to_string(pool_table.size()));
        }
    } catch (const std::exception&) {
        free(preallocated_pool_memory_ptr);
        throw;
    }

    return preallocated_pool_memory_ptr;
}

VPLCPUAccelerationPolicy::surface_weak_ptr_t VPLCPUAccelerationPolicy::get_free_surface(pool_key_t key) const {
    auto pool_it = pool_table.find(key);
    if (pool_it == pool_table.end()) {
        std::stringstream ss;
        ss << "key is not found: " << key << ", table size: " << pool_table.size();
        const std::string& str = ss.str();
        GAPI_LOG_WARNING(nullptr, str);
        throw std::runtime_error(std::string(__FUNCTION__) + " - " + str);
    }

    const pool_t& requested_pool = pool_it->second;
    auto it =
        std::find_if(requested_pool.begin(), requested_pool.end(),
                     [](const surface_ptr_t& val) {
            GAPI_Assert(val && "Pool contains empty surface");
            return !val->Data.Locked;
        });

    // TODO realloc pool
    if (it == requested_pool.end()) {
        std::stringstream ss;
        ss << "cannot get free surface from pool, key: " << key << ", size: " << requested_pool.size();
        const std::string& str = ss.str();
        GAPI_LOG_WARNING(nullptr, str);
        throw std::runtime_error(std::string(__FUNCTION__) + " - " + str);
    }

    return *it;
}

cv::MediaFrame::AdapterPtr VPLCPUAccelerationPolicy::create_frame_adapter(surface_raw_ptr_t surface) {

    return cv::MediaFrame::AdapterPtr{new VPLCPUAccelerationPolicy::MediaFrameAdapter(surface)};
}
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
