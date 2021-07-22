#ifndef OPENCV_GAPI_STREAMING_VPL_ENGINE_HPP
#define OPENCV_GAPI_STREAMING_VPL_ENGINE_HPP
#include <stdio.h>
#include <memory>

#include "streaming/engine/base_engine.hpp"
#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>

namespace cv {
namespace gapi {
namespace wip {

mfxStatus ReadEncodedStream(mfxBitstream &bs, FILE *f);

class VPLDecodeEngine : public VPLProcessingEngine {
public:
    using file_ptr = std::unique_ptr<FILE, decltype(&fclose)>;

    VPLDecodeEngine(file_ptr&& src_ptr);
    void initialize_session(mfxSession mfx_session, mfxBitstream mfx_session_bitstream);

private:
    file_ptr source_handle;
    mfxFrameSurface1 *dec_surface_out;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // OPENCV_GAPI_STREAMING_VPL_ENGINE_HPP
