#ifndef GAPI_STREAMING_ONE_VPL_FRAME_ADAPTER_HPP
#define GAPI_STREAMING_ONE_VPL_FRAME_ADAPTER_HPP


#ifdef HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {

class Surface;
class MediaFrameAdapter : public cv::MediaFrame::IAdapter {
public:
    MediaFrameAdapter(std::shared_ptr<Surface> assoc_surface);
    ~MediaFrameAdapter();
    cv::GFrameDesc meta() const override;
    MediaFrame::View access(MediaFrame::Access) override;
    
    // FIXME: design a better solution
    // The default implementation does nothing
    cv::util::any blobParams() const override;
    void serialize(cv::gapi::s11n::IOStream&) override;
    void deserialize(cv::gapi::s11n::IIStream&) override;
private:
    std::shared_ptr<Surface> parent_surface_ptr;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL


#endif // GAPI_STREAMING_ONE_VPL_FRAME_ADAPTER_HPP
