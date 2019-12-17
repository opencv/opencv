// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "cap_mfx_writer.hpp"
#include "opencv2/core/base.hpp"
#include "cap_mfx_common.hpp"
#include "opencv2/imgproc/hal/hal.hpp"

using namespace std;
using namespace cv;

static size_t getBitrateDivisor()
{
    static const size_t res = utils::getConfigurationParameterSizeT("OPENCV_VIDEOIO_MFX_BITRATE_DIVISOR", 300);
    return res;
}

static mfxU32 getWriterTimeoutMS()
{
    static const size_t res = utils::getConfigurationParameterSizeT("OPENCV_VIDEOIO_MFX_WRITER_TIMEOUT", 1);
    return saturate_cast<mfxU32>(res * 1000); // convert from seconds
}

inline mfxU32 codecIdByFourCC(int fourcc)
{
    const int CC_MPG2 = FourCC('M', 'P', 'G', '2').vali32;
    const int CC_H264 = FourCC('H', '2', '6', '4').vali32;
    const int CC_X264 = FourCC('X', '2', '6', '4').vali32;
    const int CC_AVC = FourCC('A', 'V', 'C', ' ').vali32;
    const int CC_H265 = FourCC('H', '2', '6', '5').vali32;
    const int CC_HEVC = FourCC('H', 'E', 'V', 'C').vali32;

    if (fourcc == CC_X264 || fourcc == CC_H264 || fourcc == CC_AVC)
        return MFX_CODEC_AVC;
    else if (fourcc == CC_H265 || fourcc == CC_HEVC)
        return MFX_CODEC_HEVC;
    else if (fourcc == CC_MPG2)
        return MFX_CODEC_MPEG2;
    else
        return (mfxU32)-1;
}

VideoWriter_IntelMFX::VideoWriter_IntelMFX(const String &filename, int _fourcc, double fps, Size frameSize_, bool)
    : session(0), plugin(0), deviceHandler(0), bs(0), encoder(0), pool(0), outSurface(NULL), frameSize(frameSize_), good(false)
{
    mfxStatus res = MFX_ERR_NONE;

    if (frameSize.width % 2 || frameSize.height % 2)
    {
        MSG(cerr << "MFX: Invalid frame size passed to encoder" << endl);
        return;
    }

    if (fps <= 0)
    {
        MSG(cerr << "MFX: Invalid FPS passed to encoder" << endl);
        return;
    }

    // Init device and session
    deviceHandler = createDeviceHandler();
    session = new MFXVideoSession();
    if (!deviceHandler->init(*session))
    {
        MSG(cerr << "MFX: Can't initialize session" << endl);
        return;
    }

    // Load appropriate plugin

    mfxU32 codecId = codecIdByFourCC(_fourcc);
    if (codecId == (mfxU32)-1)
    {
        MSG(cerr << "MFX: Unsupported FourCC: " << FourCC(_fourcc) << endl);
        return;
    }
    plugin = Plugin::loadEncoderPlugin(*session, codecId);
    if (plugin && !plugin->isGood())
    {
        MSG(cerr << "MFX: LoadPlugin failed for codec: " << codecId << " (" << FourCC(_fourcc) << ")" << endl);
        return;
    }

    // Init encoder

    encoder = new MFXVideoENCODE(*session);
    mfxVideoParam params;
    memset(&params, 0, sizeof(params));
    params.mfx.CodecId = codecId;
    params.mfx.TargetUsage = MFX_TARGETUSAGE_BALANCED;
    params.mfx.TargetKbps = saturate_cast<mfxU16>((frameSize.area() * fps) / (42.6666 * getBitrateDivisor())); // TODO: set in options
    params.mfx.RateControlMethod = MFX_RATECONTROL_VBR;
    params.mfx.FrameInfo.FrameRateExtN = cvRound(fps * 1000);
    params.mfx.FrameInfo.FrameRateExtD = 1000;
    params.mfx.FrameInfo.FourCC = MFX_FOURCC_NV12;
    params.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
    params.mfx.FrameInfo.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
    params.mfx.FrameInfo.CropX = 0;
    params.mfx.FrameInfo.CropY = 0;
    params.mfx.FrameInfo.CropW = (mfxU16)frameSize.width;
    params.mfx.FrameInfo.CropH = (mfxU16)frameSize.height;
    params.mfx.FrameInfo.Width = (mfxU16)alignSize(frameSize.width, 32);
    params.mfx.FrameInfo.Height = (mfxU16)alignSize(frameSize.height, 32);
    params.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY;
    res = encoder->Query(&params, &params);
    DBG(cout << "MFX Query: " << res << endl << params.mfx << params.mfx.FrameInfo);
    if (res < MFX_ERR_NONE)
    {
        MSG(cerr << "MFX: Query failed: " << res << endl);
        return;
    }

    // Init surface pool
    pool = SurfacePool::create(encoder, params);
    if (!pool)
    {
        MSG(cerr << "MFX: Failed to create surface pool" << endl);
        return;
    }

    // Init encoder
    res = encoder->Init(&params);
    DBG(cout << "MFX Init: " << res << endl << params.mfx.FrameInfo);
    if (res < MFX_ERR_NONE)
    {
        MSG(cerr << "MFX: Failed to init encoder: " << res << endl);
        return;
    }

    // Open output bitstream
    {
        mfxVideoParam par;
        memset(&par, 0, sizeof(par));
        res = encoder->GetVideoParam(&par);
        DBG(cout << "MFX GetVideoParam: " << res << endl << "requested " << par.mfx.BufferSizeInKB << " kB" << endl);
        CV_Assert(res >= MFX_ERR_NONE);
        bs = new WriteBitstream(filename.c_str(), par.mfx.BufferSizeInKB * 1024 * 2);
        if (!bs->isOpened())
        {
            MSG(cerr << "MFX: Failed to open output file: " << filename << endl);
            return;
        }
    }

    good = true;
}

VideoWriter_IntelMFX::~VideoWriter_IntelMFX()
{
    if (isOpened())
    {
        DBG(cout << "====== Drain bitstream..." << endl);
        Mat dummy;
        while (write_one(dummy)) {}
        DBG(cout << "====== Drain Finished" << endl);
    }
    cleanup(bs);
    cleanup(pool);
    cleanup(encoder);
    cleanup(plugin);
    cleanup(session);
    cleanup(deviceHandler);
}

double VideoWriter_IntelMFX::getProperty(int) const
{
    MSG(cerr << "MFX: getProperty() is not implemented" << endl);
    return 0;
}

bool VideoWriter_IntelMFX::setProperty(int, double)
{
    MSG(cerr << "MFX: setProperty() is not implemented" << endl);
    return false;
}

bool VideoWriter_IntelMFX::isOpened() const
{
    return good;
}

void VideoWriter_IntelMFX::write(cv::InputArray input)
{
    write_one(input);
}

bool VideoWriter_IntelMFX::write_one(cv::InputArray bgr)
{
    mfxStatus res;
    mfxFrameSurface1 *workSurface = 0;
    mfxSyncPoint sync;

    if (!bgr.empty() && (bgr.dims() != 2 || bgr.type() != CV_8UC3 || bgr.size() != frameSize))
    {
        MSG(cerr << "MFX: invalid frame passed to encoder: "
            << "dims/depth/cn=" << bgr.dims() << "/" << bgr.depth() << "/" << bgr.channels()
            << ", size=" << bgr.size() << endl);
        return false;

    }
    if (!bgr.empty())
    {
        workSurface = pool->getFreeSurface();
        if (!workSurface)
        {
            // not enough surfaces
            MSG(cerr << "MFX: Failed to get free surface" << endl);
            return false;
        }
        Mat src = bgr.getMat();
        hal::cvtBGRtoTwoPlaneYUV(src.data, src.step,
                                 workSurface->Data.Y, workSurface->Data.UV, workSurface->Data.Pitch,
                                 workSurface->Info.CropW, workSurface->Info.CropH,
                                 3, false, 1);
    }

    while (true)
    {
        outSurface = 0;
        DBG(cout << "Calling with surface: " << workSurface << endl);
        res = encoder->EncodeFrameAsync(NULL, workSurface, &bs->stream, &sync);
        if (res == MFX_ERR_NONE)
        {
            res = session->SyncOperation(sync, getWriterTimeoutMS()); // TODO: provide interface to modify timeout
            if (res == MFX_ERR_NONE)
            {
                // ready to write
                if (!bs->write())
                {
                    MSG(cerr << "MFX: Failed to write bitstream" << endl);
                    return false;
                }
                else
                {
                    DBG(cout << "Write bitstream" << endl);
                    return true;
                }
            }
            else
            {
                MSG(cerr << "MFX: Sync error: " << res << endl);
                return false;
            }
        }
        else if (res == MFX_ERR_MORE_DATA)
        {
            DBG(cout << "ERR_MORE_DATA" << endl);
            return false;
        }
        else if (res == MFX_WRN_DEVICE_BUSY)
        {
            DBG(cout << "Waiting for device" << endl);
            sleep_ms(1000);
            continue;
        }
        else
        {
            MSG(cerr << "MFX: Bad status: " << res << endl);
            return false;
        }
    }
}

Ptr<VideoWriter_IntelMFX> VideoWriter_IntelMFX::create(const String &filename, int _fourcc, double fps, Size frameSize, bool isColor)
{
    if (codecIdByFourCC(_fourcc) > 0)
    {
        Ptr<VideoWriter_IntelMFX> a = makePtr<VideoWriter_IntelMFX>(filename, _fourcc, fps, frameSize, isColor);
        if (a->isOpened())
            return a;
    }
    return Ptr<VideoWriter_IntelMFX>();
}
