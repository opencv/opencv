#ifndef GAPI_STREAMING_ENGINE_SESSION_HPP
#define GAPI_STREAMING_ENGINE_SESSION_HPP

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>

namespace cv {
namespace gapi {
namespace wip {

struct DecoderParams {
    mfxBitstream stream;
    mfxVideoParam param;
};

struct EngineSession {
    mfxSession session;
    mfxBitstream stream;
    mfxSyncPoint sync;
    mfxStatus last_status;

    EngineSession(mfxSession sess, mfxBitstream&& str);
    virtual ~EngineSession();
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ENGINE_SESSION_HPP
