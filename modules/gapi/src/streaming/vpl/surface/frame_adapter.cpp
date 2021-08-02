#include "streaming/vpl/surface/frame_adapter.hpp"
#include "streaming/vpl/surface/surface.hpp"
#include "logger.hpp"

#ifdef HAVE_ONEVPL

#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif

#include <vpl/mfx.h>

namespace cv {
namespace gapi {
namespace wip {

MediaFrameAdapter::MediaFrameAdapter(std::shared_ptr<Surface> surface):
    parent_surface_ptr(surface) {

    GAPI_Assert(parent_surface_ptr && "Surface is nullptr");
    parent_surface_ptr->obtain_lock();


    const Surface::info_t& info = parent_surface_ptr->get_info();
    const Surface::data_t& data = parent_surface_ptr->get_data();

    GAPI_LOG_DEBUG(nullptr, "surface: " << parent_surface_ptr->get_handle() <<
                            ", w: " << info.Width << ", h: " << info.Height <<
                            ", p: " << data.Pitch);
}

MediaFrameAdapter::~MediaFrameAdapter() {

    // Each MediaFrameAdapter releases mfx surface counter
    // The last MediaFrameAdapter releases shared Surface pointer
    // The last surface pointer releases workspace memory
    parent_surface_ptr->release_lock();
}

cv::GFrameDesc MediaFrameAdapter::meta() const {
    GFrameDesc desc;
    const Surface::info_t info = parent_surface_ptr->get_info();
    switch(info.FourCC)
    {
        case MFX_FOURCC_I420:
            throw std::runtime_error("MediaFrame doesn't support I420 type");
            break;
        case MFX_FOURCC_NV12:
            desc.fmt = MediaFormat::NV12;
            break;
        default:
            throw std::runtime_error("MediaFrame unknown 'fmt' type: " + std::to_string(info.FourCC));
    }
    
    desc.size = cv::Size{info.Width, info.Height};
    return desc;
}

MediaFrame::View MediaFrameAdapter::access(MediaFrame::Access mode) {
    (void)mode;

    const Surface::data_t& data = parent_surface_ptr->get_data();
    const Surface::info_t info = parent_surface_ptr->get_info();
    using stride_t = typename cv::MediaFrame::View::Strides::value_type;
    GAPI_Assert(data.Pitch >= 0 && "Pitch is less 0");

    stride_t pitch = static_cast<stride_t>(data.Pitch);
    switch(info.FourCC) {
        case MFX_FOURCC_I420:
        {
            cv::MediaFrame::View::Ptrs pp = {
                data.Y,
                data.U,
                data.V,
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
                data.Y,
                data.UV, nullptr, nullptr
                };
            cv::MediaFrame::View::Strides ss = {
                    pitch,
                    pitch, 0u, 0u
                };
            return cv::MediaFrame::View(std::move(pp), std::move(ss));
        }
            break;
        default:
            throw std::runtime_error("MediaFrame unknown 'fmt' type: " + std::to_string(info.FourCC));
    }
}

cv::util::any MediaFrameAdapter::blobParams() const {
    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
}

void MediaFrameAdapter::serialize(cv::gapi::s11n::IOStream&) {
    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
}
void MediaFrameAdapter::deserialize(cv::gapi::s11n::IIStream&) {
    throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented");
}

#endif HAVE_ONEVPL
} // namespace wip
} // namespace gapi
} // namespace cv
