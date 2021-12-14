#ifndef GAPI_STREAMING_ONEVPL_DATA_PROVIDER_DEFINES_HPP
#define GAPI_STREAMING_ONEVPL_DATA_PROVIDER_DEFINES_HPP

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"
#endif // HAVE_ONEVPL

#include <opencv2/gapi/own/assert.hpp>
#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

#ifdef HAVE_ONEVPL
struct IDataProvider::mfx_bitstream : public mfxBitstream {};
#else // HAVE_ONEVPL
struct IDataProvider::mfx_bitstream {
    mfx_bitstream() {
        GAPI_Assert(false && "Reject to create `mfxBitstream` because library compiled without VPL/MFX support");
    }
};
#endif // HAVE_ONEVPL

} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_DATA_PROVIDER_DEFINES_HPP
