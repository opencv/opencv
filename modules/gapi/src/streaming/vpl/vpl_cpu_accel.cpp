#ifdef HAVE_ONEVPL
#include <exception>

#include "streaming/vpl/vpl_cpu_accel.hpp"
#include "logger.hpp"

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

    parent_surface_ptr->Data.Locked++;
}

VPLCPUAccelerationPolicy::MediaFrameAdapter::~MediaFrameAdapter()
{
    parent_surface_ptr->Data.Locked--;
}

cv::GFrameDesc VPLCPUAccelerationPolicy::MediaFrameAdapter::meta() const
{
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

MediaFrame::View VPLCPUAccelerationPolicy::MediaFrameAdapter::access(MediaFrame::Access mode)
{
    (void)mode;

    using stride_t = typename cv::MediaFrame::View::Strides::value_type;
    assert(parent_surface_ptr->Data.Pitch >= 0 && "Pitch is less 0");

    stride_t pitch = static_cast<stride_t>(parent_surface_ptr->Data.Pitch);
    switch(parent_surface_ptr->Info.FourCC)
    {
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

cv::util::any VPLCPUAccelerationPolicy::MediaFrameAdapter::blobParams() const
{
    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
}

void VPLCPUAccelerationPolicy::MediaFrameAdapter::serialize(cv::gapi::s11n::IOStream&)
{
    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
}
void VPLCPUAccelerationPolicy::MediaFrameAdapter::deserialize(cv::gapi::s11n::IIStream&)
{
    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
}

    
VPLCPUAccelerationPolicy::VPLCPUAccelerationPolicy(mfxSession session)
{
    (void)session;
    //MFXVideoCORE_SetFrameAllocator(session, mfxFrameAllocator instance)
    GAPI_LOG_INFO(nullptr, "VPLCPUAccelerationPolicy initialized");
}

VPLCPUAccelerationPolicy::~VPLCPUAccelerationPolicy()
{
    GAPI_LOG_INFO(nullptr, "VPLCPUAccelerationPolicy release ID3D11Device");
}

cv::MediaFrame::AdapterPtr VPLCPUAccelerationPolicy::create_frame_adapter(mfxFrameSurface1* surface_ptr) {

    cv::MediaFrame::AdapterPtr ret(new VPLCPUAccelerationPolicy::MediaFrameAdapter(surface_ptr));
    return ret;
}
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
