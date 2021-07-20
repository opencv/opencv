#ifndef ONEVPL_PRIV_INTERFACE_HPP
#define ONEVPL_PRIV_INTERFACE_HPP

#ifdef HAVE_ONEVPL

#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif

#include <vpl/mfx.h>
#endif // HAVE_ONEVPL

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/streaming/meta.hpp>
#include <opencv2/gapi/streaming/onevpl_cap.hpp>

#include "streaming/onevpl_builder.hpp"

namespace cv {
namespace gapi {
namespace wip {
struct OneVPLCapture::IPriv
{
    virtual ~IPriv() {}
    
    virtual bool pull(cv::gapi::wip::Data& data) = 0;
    virtual GMetaArg descr_of() const = 0;

    template<typename... Args>
    static std::unique_ptr<OneVPLCapture::IPriv> make_priv(Args&& ...args)
    {
        oneVPLBulder builder;
        builder.set(std::forward<Args>(args)...);
        return builder.build();
    }
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // ONEVPL_PRIV_INTERFACE_HPP
