#include "streaming/vpl/vpl_source_engine.hpp"

namespace cv {
namespace gapi {
namespace wip {

// Read encoded stream from file
mfxStatus ReadEncodedStream(mfxBitstream &bs, FILE *f) {
    mfxU8 *p0 = bs.Data;
    mfxU8 *p1 = bs.Data + bs.DataOffset;
    if (bs.DataOffset > bs.MaxLength - 1) {
        return MFX_ERR_NOT_ENOUGH_BUFFER;
    }
    if (bs.DataLength + bs.DataOffset > bs.MaxLength) {
        return MFX_ERR_NOT_ENOUGH_BUFFER;
    }
    for (mfxU32 i = 0; i < bs.DataLength; i++) {
        *(p0++) = *(p1++);
    }
    bs.DataOffset = 0;
    bs.DataLength += (mfxU32)fread(bs.Data + bs.DataLength, 1, bs.MaxLength - bs.DataLength, f);
    if (bs.DataLength == 0)
        return MFX_ERR_MORE_DATA;

    return MFX_ERR_NONE;
}
//////////


VPLDecodeEngine::VPLDecodeEngine(file_ptr&& src_ptr) :
    source_handle(std::move(src_ptr)),
    dec_surface_out()
{
}

void VPLDecodeEngine::initialize_session(mfxSession mfx_session, mfxBitstream mfx_session_bitstream)
{
    register_session(mfx_session, mfx_session_bitstream,
        [this] (EngineSession& sess)
            {
                sess.last_status = ReadEncodedStream(sess.stream, source_handle.get());
            },
            [this] (EngineSession& sess)
            {
                if (sess.last_status != MFX_ERR_NONE) // No more data to read, start decode draining mode
                {
                    sess.last_status = MFXVideoDECODE_DecodeFrameAsync(sess.session,
                                          nullptr,
                                          nullptr,
                                          &dec_surface_out,
                                          &sess.sync);
                } else {
                    sess.last_status = MFXVideoDECODE_DecodeFrameAsync(sess.session,
                                          &sess.stream,
                                          nullptr,
                                          &dec_surface_out,
                                          &sess.sync);
                }
            },
            [] (EngineSession& sess)
            {
                if (sess.last_status == MFX_ERR_NONE) // Got 1 decoded frame
                {
                    do {
                        sess.last_status = MFXVideoCORE_SyncOperation(sess.session, sess.sync, 100);
                        if (MFX_ERR_NONE == sess.last_status) {
                            // -S- TODO Consume Data
                            //framenum++;
                        }
                    } while (sess.last_status == MFX_WRN_IN_EXECUTION);
                }
            });
}

} // namespace wip
} // namespace gapi
} // namespace cv
