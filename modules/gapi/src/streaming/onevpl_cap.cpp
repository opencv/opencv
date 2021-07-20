#include <opencv2/gapi/streaming/onevpl_cap.hpp>

#include "streaming/onevpl_priv_interface.hpp"

namespace cv {
namespace gapi {
namespace wip {

OneVPLCapture::OneVPLCapture(const std::string& filepath) : IStreamSource(),
    m_priv(OneVPLCapture::IPriv::make_priv(filepath))
{
}

OneVPLCapture::~OneVPLCapture()
{
}

bool OneVPLCapture::pull(cv::gapi::wip::Data& data)
{
    return m_priv->pull(data);
}

GMetaArg OneVPLCapture::descr_of() const
{
    return m_priv->descr_of();
}

} // namespace wip
} // namespace gapi
} // namespace cv
