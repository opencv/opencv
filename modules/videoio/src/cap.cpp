/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#include "opencv2/videoio/registry.hpp"
#include "videoio_registry.hpp"

namespace cv {

static bool param_VIDEOIO_DEBUG = utils::getConfigurationParameterBool("OPENCV_VIDEOIO_DEBUG", false);
static bool param_VIDEOCAPTURE_DEBUG = utils::getConfigurationParameterBool("OPENCV_VIDEOCAPTURE_DEBUG", false);
static bool param_VIDEOWRITER_DEBUG = utils::getConfigurationParameterBool("OPENCV_VIDEOWRITER_DEBUG", false);

#define CV_CAPTURE_LOG_DEBUG(tag, ...)                   \
    if (param_VIDEOIO_DEBUG || param_VIDEOCAPTURE_DEBUG) \
    {                                                    \
        CV_LOG_WARNING(nullptr, __VA_ARGS__);            \
    }

#define CV_WRITER_LOG_DEBUG(tag, ...)                   \
    if (param_VIDEOIO_DEBUG || param_VIDEOWRITER_DEBUG) \
    {                                                   \
        CV_LOG_WARNING(nullptr, __VA_ARGS__)            \
    }

IStreamReader::~IStreamReader()
{
    // nothing
}

VideoCapture::VideoCapture() : throwOnFail(false)
{}

VideoCapture::VideoCapture(const String& filename, int apiPreference, double target_fps) : throwOnFail(false)
{
    CV_TRACE_FUNCTION();
    open(filename, apiPreference, target_fps);
}

VideoCapture::VideoCapture(const String& filename, int apiPreference, const std::vector<int>& params)
    : throwOnFail(false)
{
    CV_TRACE_FUNCTION();
    open(filename, apiPreference, params);
}

VideoCapture::VideoCapture(const Ptr<IStreamReader>& source, int apiPreference, const std::vector<int>& params)
    : throwOnFail(false)
{
    CV_TRACE_FUNCTION();
    open(source, apiPreference, params);
}

VideoCapture::VideoCapture(int index, int apiPreference, double target_fps) : throwOnFail(false)
{
    CV_TRACE_FUNCTION();
    open(index, apiPreference, target_fps);
}

VideoCapture::VideoCapture(int index, int apiPreference, const std::vector<int>& params)
    : throwOnFail(false)
{
    CV_TRACE_FUNCTION();
    open(index, apiPreference, params);
}

VideoCapture::~VideoCapture()
{
    CV_TRACE_FUNCTION();
    icap.release();
}

bool VideoCapture::open(const String& filename, int apiPreference, double target_fps)
{
    CV_INSTRUMENT_REGION();
    bool ok = open(filename, apiPreference, std::vector<int>());
    if (ok)
        enableFpsControl(target_fps);
    return ok;
}

bool VideoCapture::open(const String& filename, int apiPreference, const std::vector<int>& params)
{
    CV_INSTRUMENT_REGION();

    if (isOpened())
    {
        release();
    }

    const VideoCaptureParameters parameters(params);
    const std::vector<VideoBackendInfo> backends = cv::videoio_registry::getAvailableBackends_CaptureByFilename();
    for (size_t i = 0; i < backends.size(); i++)
    {
        const VideoBackendInfo& info = backends[i];
        if (apiPreference == CAP_ANY || apiPreference == info.id)
        {
            if (!info.backendFactory)
            {
                CV_LOG_DEBUG(NULL, "VIDEOIO(" << info.name << "): factory is not available (plugins require filesystem support)");
                continue;
            }
            CV_CAPTURE_LOG_DEBUG(NULL,
                                 cv::format("VIDEOIO(%s): trying capture filename='%s' ...",
                                            info.name, filename.c_str()));
            CV_Assert(!info.backendFactory.empty());
            const Ptr<IBackend> backend = info.backendFactory->getBackend();
            if (!backend.empty())
            {
                try
                {
                    icap = backend->createCapture(filename, parameters);
                    if (!icap.empty())
                    {
                        CV_CAPTURE_LOG_DEBUG(NULL,
                                             cv::format("VIDEOIO(%s): created, isOpened=%d",
                                                        info.name, icap->isOpened()));
                        if (icap->isOpened())
                        {
                            return true;
                        }
                        icap.release();
                    }
                    else
                    {
                        CV_CAPTURE_LOG_DEBUG(NULL,
                                             cv::format("VIDEOIO(%s): can't create capture",
                                                        info.name));
                    }
                }
                catch (const cv::Exception& e)
                {
                    if (throwOnFail && apiPreference != CAP_ANY)
                    {
                        throw;
                    }
                    CV_LOG_WARNING(NULL,
                                   cv::format("VIDEOIO(%s): raised OpenCV exception:\n\n%s\n",
                                              info.name, e.what()));
                }
                catch (const std::exception& e)
                {
                    if (throwOnFail && apiPreference != CAP_ANY)
                    {
                        throw;
                    }
                    CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): raised C++ exception:\n\n%s\n",
                                                    info.name, e.what()));
                }
                catch (...)
                {
                    if (throwOnFail && apiPreference != CAP_ANY)
                    {
                        throw;
                    }
                    CV_LOG_WARNING(NULL,
                                   cv::format("VIDEOIO(%s): raised unknown C++ exception!\n\n",
                                              info.name));
                }
            }
            else
            {
                CV_CAPTURE_LOG_DEBUG(NULL,
                                     cv::format("VIDEOIO(%s): backend is not available "
                                                "(plugin is missing, or can't be loaded due "
                                                "dependencies or it is not compatible)",
                                                info.name));
            }
        }
    }

    if(apiPreference != CAP_ANY)
    {
        bool found = cv::videoio_registry::isBackendBuiltIn(static_cast<VideoCaptureAPIs>(apiPreference));
        if (found)
        {
            CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): backend is generally available "
                                            "but can't be used to capture by name",
                                            cv::videoio_registry::getBackendName(static_cast<VideoCaptureAPIs>(apiPreference)).c_str()));
        }
    }

    if (throwOnFail)
    {
        CV_Error_(Error::StsError, ("could not open '%s'", filename.c_str()));
    }

    if (cv::videoio_registry::checkDeprecatedBackend(apiPreference))
    {
        CV_LOG_DEBUG(NULL,
            cv::format("VIDEOIO(%s): backend is removed from OpenCV",
                cv::videoio_registry::getBackendName((VideoCaptureAPIs) apiPreference).c_str()));
    }
    else
    {
        CV_LOG_DEBUG(NULL, "VIDEOIO: chosen backend does not work or wrong. "
            "Please make sure that your computer support chosen backend and OpenCV built "
            "with right flags.");
    }

    return false;
}

bool VideoCapture::open(const Ptr<IStreamReader>& stream, int apiPreference, const std::vector<int>& params)

{
    CV_INSTRUMENT_REGION();

    if (apiPreference == CAP_ANY)
    {
        CV_Error_(Error::StsBadArg, ("Avoid CAP_ANY - explicit backend expected to avoid read data stream reset"));
    }

    if (isOpened())
    {
        release();
    }

    const VideoCaptureParameters parameters(params);
    const std::vector<VideoBackendInfo> backends = cv::videoio_registry::getAvailableBackends_CaptureByStream();
    for (size_t i = 0; i < backends.size(); i++)
    {
        const VideoBackendInfo& info = backends[i];
        if (apiPreference != info.id)
            continue;

        if (!info.backendFactory)
        {
            CV_LOG_DEBUG(NULL, "VIDEOIO(" << info.name << "): factory is not available (plugins require filesystem support)");
            continue;
        }
        CV_CAPTURE_LOG_DEBUG(NULL,
                                cv::format("VIDEOIO(%s): trying capture buffer ...",
                                        info.name));
        CV_Assert(!info.backendFactory.empty());
        const Ptr<IBackend> backend = info.backendFactory->getBackend();
        if (!backend.empty())
        {
            try
            {
                icap = backend->createCapture(stream, parameters);
                if (!icap.empty())
                {
                    CV_CAPTURE_LOG_DEBUG(NULL,
                                            cv::format("VIDEOIO(%s): created, isOpened=%d",
                                                    info.name, icap->isOpened()));
                    if (icap->isOpened())
                    {
                        return true;
                    }
                    icap.release();
                }
                else
                {
                    CV_CAPTURE_LOG_DEBUG(NULL,
                                            cv::format("VIDEOIO(%s): can't create capture",
                                                    info.name));
                }
            }
            catch (const cv::Exception& e)
            {
                if (throwOnFail)
                {
                    throw;
                }
                CV_LOG_WARNING(NULL,
                                cv::format("VIDEOIO(%s): raised OpenCV exception:\n\n%s\n",
                                            info.name, e.what()));
            }
            catch (const std::exception& e)
            {
                if (throwOnFail)
                {
                    throw;
                }
                CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): raised C++ exception:\n\n%s\n",
                                                info.name, e.what()));
            }
            catch (...)
            {
                if (throwOnFail)
                {
                    throw;
                }
                CV_LOG_WARNING(NULL,
                                cv::format("VIDEOIO(%s): raised unknown C++ exception!\n\n",
                                            info.name));
            }
        }
        else
        {
            CV_CAPTURE_LOG_DEBUG(NULL,
                                 cv::format("VIDEOIO(%s): backend is not available "
                                            "(plugin is missing, or can't be loaded due "
                                            "dependencies or it is not compatible)",
                                            info.name));
        }
    }

    bool found = cv::videoio_registry::isBackendBuiltIn(static_cast<VideoCaptureAPIs>(apiPreference));
    if (found)
    {
        CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): backend is generally available "
                                        "but can't be used to capture by read data stream",
                                        cv::videoio_registry::getBackendName(static_cast<VideoCaptureAPIs>(apiPreference)).c_str()));
    }

    if (throwOnFail)
    {
        CV_Error_(Error::StsError, ("could not open read data stream"));
    }

    if (cv::videoio_registry::checkDeprecatedBackend(apiPreference))
    {
        CV_LOG_DEBUG(NULL,
            cv::format("VIDEOIO(%s): backend is removed from OpenCV",
                cv::videoio_registry::getBackendName((VideoCaptureAPIs) apiPreference).c_str()));
    }
    else
    {
        CV_LOG_DEBUG(NULL, "VIDEOIO: chosen backend does not work or wrong. "
            "Please make sure that your computer support chosen backend and OpenCV built "
            "with right flags.");
    }

    return false;
}

bool VideoCapture::open(int cameraNum, int apiPreference, double target_fps)
{
    CV_INSTRUMENT_REGION();
    bool ok = open(cameraNum, apiPreference, std::vector<int>());
    if (ok)
        enableFpsControl(target_fps);
    return ok;
}

bool VideoCapture::open(int cameraNum, int apiPreference, const std::vector<int>& params)
{
    CV_TRACE_FUNCTION();

    if (isOpened())
    {
        release();
    }

    if (apiPreference == CAP_ANY)
    {
        // interpret preferred interface (0 = autodetect)
        int backendID = (cameraNum / 100) * 100;
        if (backendID)
        {
            cameraNum %= 100;
            apiPreference = backendID;
        }
    }

    const VideoCaptureParameters parameters(params);
    const std::vector<VideoBackendInfo> backends = cv::videoio_registry::getAvailableBackends_CaptureByIndex();
    for (size_t i = 0; i < backends.size(); i++)
    {
        const VideoBackendInfo& info = backends[i];
        if (apiPreference == CAP_ANY || apiPreference == info.id)
        {
            if (!info.backendFactory)
            {
                CV_LOG_DEBUG(NULL, "VIDEOIO(" << info.name << "): factory is not available (plugins require filesystem support)");
                continue;
            }
            CV_CAPTURE_LOG_DEBUG(NULL,
                                 cv::format("VIDEOIO(%s): trying capture cameraNum=%d ...",
                                            info.name, cameraNum));
            CV_Assert(!info.backendFactory.empty());
            const Ptr<IBackend> backend = info.backendFactory->getBackend();
            if (!backend.empty())
            {
                try
                {
                    icap = backend->createCapture(cameraNum, parameters);
                    if (!icap.empty())
                    {
                        CV_CAPTURE_LOG_DEBUG(NULL,
                                             cv::format("VIDEOIO(%s): created, isOpened=%d",
                                                        info.name, icap->isOpened()));
                        if (icap->isOpened())
                        {
                            return true;
                        }
                        icap.release();
                    }
                    else
                    {
                        CV_CAPTURE_LOG_DEBUG(NULL,
                                             cv::format("VIDEOIO(%s): can't create capture",
                                                        info.name));
                    }
                }
                catch (const cv::Exception& e)
                {
                    if (throwOnFail && apiPreference != CAP_ANY)
                    {
                        throw;
                    }
                    CV_LOG_WARNING(NULL,
                                   cv::format("VIDEOIO(%s): raised OpenCV exception:\n\n%s\n",
                                              info.name, e.what()));
                }
                catch (const std::exception& e)
                {
                    if (throwOnFail && apiPreference != CAP_ANY)
                    {
                        throw;
                    }
                    CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): raised C++ exception:\n\n%s\n",
                                                    info.name, e.what()));
                }
                catch (...)
                {
                    if (throwOnFail && apiPreference != CAP_ANY)
                    {
                        throw;
                    }
                    CV_LOG_WARNING(NULL,
                                   cv::format("VIDEOIO(%s): raised unknown C++ exception!\n\n",
                                              info.name));
                }
            }
            else
            {
                CV_CAPTURE_LOG_DEBUG(NULL,
                                     cv::format("VIDEOIO(%s): backend is not available "
                                                "(plugin is missing, or can't be loaded due "
                                                "dependencies or it is not compatible)",
                                                info.name));
            }
        }
    }

    if(apiPreference != CAP_ANY)
    {
        bool found = cv::videoio_registry::isBackendBuiltIn(static_cast<VideoCaptureAPIs>(apiPreference));
        if (found)
        {
            CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): backend is generally available "
                                            "but can't be used to capture by index",
                                            cv::videoio_registry::getBackendName(static_cast<VideoCaptureAPIs>(apiPreference)).c_str()));
        }
    }

    if (throwOnFail)
    {
        CV_Error_(Error::StsError, ("could not open camera %d", cameraNum));
    }

    if (cv::videoio_registry::checkDeprecatedBackend(apiPreference))
    {
        CV_LOG_DEBUG(NULL,
            cv::format("VIDEOIO(%s): backend is removed from OpenCV",
                cv::videoio_registry::getBackendName((VideoCaptureAPIs) apiPreference).c_str()));
    }
    else
    {
        CV_LOG_DEBUG(NULL, "VIDEOIO: chosen backend does not work or wrong."
            "Please make sure that your computer support chosen backend and OpenCV built "
            "with right flags.");
    }

    return false;
}

bool VideoCapture::isOpened() const
{
    return !icap.empty() ? icap->isOpened() : false;
}

String VideoCapture::getBackendName() const
{
    int api = 0;
    if (icap)
    {
        api = icap->isOpened() ? icap->getCaptureDomain() : 0;
    }
    CV_Assert(api != 0);
    return cv::videoio_registry::getBackendName(static_cast<VideoCaptureAPIs>(api));
}

void VideoCapture::release()
{
    CV_TRACE_FUNCTION();
    icap.release();
    fpsCtl = FpsControlState(); // don't leak fps-control state (clock/buffers) across reopen
}

// target_fps <= 0 disables frame-rate control entirely (plain passthrough).
void VideoCapture::enableFpsControl(double target_fps)
{
    fpsCtl = FpsControlState();
    if (target_fps > 0 && !icap.empty())
    {
        fpsCtl.enabled = true;
        fpsCtl.outFrameDurationMs = 1000.0 / target_fps;
    }
}

// Pull exactly one real frame from the backend into the next free lookahead-buffer slot.
// Always clones: some backends reuse a single internal decode buffer across grabFrame() calls,
// and we need two buffered frames to stay valid and independent at the same time.
bool VideoCapture::fpsControlReadOne()
{
    CV_Assert(fpsCtl.bufCount < 2);
    if (!icap->grabFrame())
        return false;
    Mat frame;
    if (!icap->retrieveFrame(0, frame) || frame.empty())
        return false;
    fpsCtl.bufFrame[fpsCtl.bufCount] = frame.clone();
    fpsCtl.bufPts[fpsCtl.bufCount] = icap->getProperty(CAP_PROP_POS_MSEC);
    fpsCtl.bufCount++;
    return true;
}

// Drop-only version of FFmpeg's vf_fps.c decision rule: keep 2 frames buffered and compare the
// *next* frame's timestamp against the output clock (nextOutPts). If the next frame has already
// reached or passed the current tick, the buffered frame is stale -- drop it and re-check.
// Otherwise it's the answer for this tick -- consume it and advance the clock by one output-frame
// duration. Since target_fps is always <= the source rate here, a buffered frame is never held
// over for a second tick, so there is no duplicate case to handle.
//
// At end of stream, only one frame may be left buffered with no lookahead to compare it against --
// in that case its own timestamp is checked directly against the clock instead: still due -> emit
// it (the last available data is the best answer for this tick); not yet due -> the schedule never
// caught up before the source ran out, so there's nothing to emit for this tick.
//
// kFpsControlEpsMs (see videoio.hpp, next to FpsControlState) absorbs floating-point rounding
// noise in backend-reported timestamps so an exact schedule boundary doesn't get flipped by it.
const double VideoCapture::kFpsControlEpsMs = 1e-6;

bool VideoCapture::fpsControlGrab()
{
    FpsControlState& s = fpsCtl;

    for (;;)
    {
        while (s.bufCount < 2)
        {
            if (fpsControlReadOne())
                continue;
            break; // source exhausted -- bufCount may now be 0 or 1
        }

        if (s.bufCount == 0)
        {
            // Nothing buffered and nothing left to emit -- invalidate the previous answer so a
            // retrieve() call made without checking this grab()'s return value can't silently
            // re-serve a stale frame instead of failing.
            s.pendingFrame.release();
            s.pendingPts = -1.0;
            return false;
        }

        if (s.nextOutPts < 0)
            s.nextOutPts = s.bufPts[0]; // anchor the output clock to the first real frame's timestamp

        if (s.bufCount == 2)
        {
            if (s.bufPts[1] <= s.nextOutPts + kFpsControlEpsMs)
            {
                // buf[0] is stale -- drop it, shift, and re-check against the new pair.
                s.bufFrame[0] = s.bufFrame[1];
                s.bufPts[0] = s.bufPts[1];
                s.bufCount = 1;
                continue;
            }
        }
        else // bufCount == 1: source is exhausted, no lookahead frame left to compare against
        {
            if (s.bufPts[0] < s.nextOutPts - kFpsControlEpsMs)
            {
                s.pendingFrame.release(); // same reasoning as the bufCount==0 case above
                s.pendingPts = -1.0;
                return false; // schedule never caught up before the source ran out
            }
        }

        // buf[0] is the answer for this tick.
        s.pendingFrame = s.bufFrame[0];
        s.pendingPts = s.bufPts[0];
        s.nextOutPts += s.outFrameDurationMs;
        if (s.bufCount == 2)
        {
            s.bufFrame[0] = s.bufFrame[1];
            s.bufPts[0] = s.bufPts[1];
            s.bufCount = 1;
        }
        else
        {
            s.bufCount = 0;
        }
        return true;
    }
}

bool VideoCapture::grab()
{
    CV_INSTRUMENT_REGION();
    bool ret = false;
    if (!icap.empty())
    {
        ret = fpsCtl.enabled ? fpsControlGrab() : icap->grabFrame();
    }
    if (!ret && throwOnFail)
    {
        CV_Error(Error::StsError, "");
    }
    return ret;
}

bool VideoCapture::retrieve(OutputArray image, int channel)
{
    CV_INSTRUMENT_REGION();

    bool ret = false;
    if (!icap.empty())
    {
        if (fpsCtl.enabled)
        {
            // target_fps only buffers/decodes channel 0 (fpsControlReadOne() always calls
            // retrieveFrame(0, ...)) -- it does not support multi-head sources (stereo camera,
            // Kinect, etc: see the multi-head note on grab() above). Fail loudly rather than
            // silently handing back channel 0's data for a different requested channel.
            if (channel != 0)
            {
                CV_LOG_WARNING(NULL, "VIDEOIO: target_fps does not support multi-head capture "
                                      "(channel != 0); use target_fps <= 0 for multi-head sources");
            }
            else if (!fpsCtl.pendingFrame.empty())
            {
                // grab() already fully decoded this frame via fpsControlGrab() -- hand back a copy.
                fpsCtl.pendingFrame.copyTo(image);
                ret = true;
            }
        }
        else
        {
            ret = icap->retrieveFrame(channel, image);
        }
    }
    if (!ret && throwOnFail)
    {
        CV_Error_(Error::StsError, ("could not retrieve channel %d", channel));
    }
    return ret;
}

bool VideoCapture::read(OutputArray image)
{
    CV_INSTRUMENT_REGION();

    if (grab())
    {
        retrieve(image);
    } else {
        image.release();
    }
    return !image.empty();
}

VideoCapture& VideoCapture::operator >> (Mat& image)
{
#ifdef WINRT_VIDEO
    // FIXIT grab/retrieve methods() should work too
    if (grab())
    {
        if (retrieve(image))
        {
            std::lock_guard<std::mutex> lock(VideoioBridge::getInstance().inputBufferMutex);
            VideoioBridge& bridge = VideoioBridge::getInstance();

            // double buffering
            bridge.swapInputBuffers();
            auto p = bridge.frontInputPtr;

            bridge.bIsFrameNew = false;

            // needed here because setting Mat 'image' is not allowed by OutputArray in read()
            Mat m(bridge.getHeight(), bridge.getWidth(), CV_8UC3, p);
            image = m;
        }
    }
#else
    read(image);
#endif
    return *this;
}

VideoCapture& VideoCapture::operator >> (UMat& image)
{
    CV_INSTRUMENT_REGION();

    read(image);
    return *this;
}

bool VideoCapture::set(int propId, double value)
{
    CV_CheckNE(propId, (int)CAP_PROP_BACKEND, "Can't set read-only property");
    bool ret = !icap.empty() ? icap->setProperty(propId, value) : false;
    if (!ret && throwOnFail)
    {
        CV_Error_(Error::StsError, ("could not set prop %d = %f", propId, value));
    }
    return ret;
}

double VideoCapture::get(int propId) const
{
    if (propId == CAP_PROP_BACKEND)
    {
        int api = 0;
        if (icap && icap->isOpened())
        {
            api = icap->getCaptureDomain();
        }
        if (api <= 0)
        {
            return CAP_PROP_UNKNOWN;
        }
        return static_cast<double>(api);
    }
    if (propId == CAP_PROP_POS_MSEC && fpsCtl.enabled && fpsCtl.pendingPts >= 0)
    {
        // Report the timestamp of the frame actually handed back by read()/retrieve(), not the
        // backend's own internal position -- which, under fps control, sits on whichever
        // lookahead frame the algorithm most recently had to pull to make its drop/keep decision
        // (one frame ahead of the emitted one in steady state, or an inconsistent
        // backend-specific value right at end-of-stream).
        return fpsCtl.pendingPts;
    }
    return !icap.empty() ? icap->getProperty(propId) : static_cast<double>(CAP_PROP_UNKNOWN);
}


bool VideoCapture::waitAny(const std::vector<VideoCapture>& streams,
                           CV_OUT std::vector<int>& readyIndex, int64 timeoutNs)
{
    CV_Assert(!streams.empty());

    VideoCaptureAPIs backend = (VideoCaptureAPIs)streams[0].icap->getCaptureDomain();

    for (size_t i = 1; i < streams.size(); ++i)
    {
        VideoCaptureAPIs backend_i = (VideoCaptureAPIs)streams[i].icap->getCaptureDomain();
        CV_CheckEQ((int)backend, (int)backend_i, "All captures must have the same backend");
    }

#if (defined HAVE_CAMV4L2 || defined HAVE_VIDEOIO) // see cap_v4l.cpp guard
    if (backend == CAP_V4L2)
    {
        return VideoCapture_V4L_waitAny(streams, readyIndex, timeoutNs);
    }
#else
    CV_UNUSED(readyIndex);
    CV_UNUSED(timeoutNs);
#endif
    CV_Error(Error::StsNotImplemented, "VideoCapture::waitAny() is supported by V4L backend only");
}


//=================================================================================================



VideoWriter::VideoWriter()
{}

VideoWriter::VideoWriter(const String& filename, int _fourcc, double fps, Size frameSize,
                         bool isColor)
{
    open(filename, _fourcc, fps, frameSize, isColor);
}


VideoWriter::VideoWriter(const String& filename, int apiPreference, int _fourcc, double fps,
                         Size frameSize, bool isColor)
{
    open(filename, apiPreference, _fourcc, fps, frameSize, isColor);
}

VideoWriter::VideoWriter(const cv::String& filename, int fourcc, double fps,
                         const cv::Size& frameSize, const std::vector<int>& params)
{
    open(filename, fourcc, fps, frameSize, params);
}

VideoWriter::VideoWriter(const cv::String& filename, int apiPreference, int fourcc, double fps,
                         const cv::Size& frameSize, const std::vector<int>& params)
{
    open(filename, apiPreference, fourcc, fps, frameSize, params);
}

void VideoWriter::release()
{
    iwriter.release();
}

VideoWriter::~VideoWriter()
{
    release();
}

bool VideoWriter::open(const String& filename, int _fourcc, double fps, Size frameSize,
                       bool isColor)
{
    return open(filename, CAP_ANY, _fourcc, fps, frameSize,
                std::vector<int> { VIDEOWRITER_PROP_IS_COLOR, static_cast<int>(isColor) });
}

bool VideoWriter::open(const String& filename, int apiPreference, int _fourcc, double fps,
                       Size frameSize, bool isColor)
{
    return open(filename, apiPreference, _fourcc, fps, frameSize,
                std::vector<int> { VIDEOWRITER_PROP_IS_COLOR, static_cast<int>(isColor) });
}


bool VideoWriter::open(const String& filename, int fourcc, double fps, const Size& frameSize,
                       const std::vector<int>& params)
{
    return open(filename, CAP_ANY, fourcc, fps, frameSize, params);
}

bool VideoWriter::open(const String& filename, int apiPreference, int fourcc, double fps,
                       const Size& frameSize, const std::vector<int>& params)
{
    CV_INSTRUMENT_REGION();

    if (isOpened())
    {
        release();
    }

    const VideoWriterParameters parameters(params);
    for (const auto& info : videoio_registry::getAvailableBackends_Writer())
    {
        if (apiPreference == CAP_ANY || apiPreference == info.id)
        {
            CV_WRITER_LOG_DEBUG(NULL,
                                cv::format("VIDEOIO(%s): trying writer with filename='%s' "
                                           "fourcc=0x%08x fps=%g sz=%dx%d isColor=%d...",
                                           info.name, filename.c_str(), (unsigned)fourcc, fps,
                                           frameSize.width, frameSize.height,
                                           parameters.get(VIDEOWRITER_PROP_IS_COLOR, true)));
            CV_Assert(!info.backendFactory.empty());
            const Ptr<IBackend> backend = info.backendFactory->getBackend();
            if (!backend.empty())
            {
                try
                {
                    iwriter = backend->createWriter(filename, fourcc, fps, frameSize, parameters);
                    if (!iwriter.empty())
                    {

                        CV_WRITER_LOG_DEBUG(NULL,
                                            cv::format("VIDEOIO(%s): created, isOpened=%d",
                                                       info.name, iwriter->isOpened()));
                        if (param_VIDEOIO_DEBUG || param_VIDEOWRITER_DEBUG)
                        {
                            for (int key: parameters.getUnused())
                            {
                                CV_LOG_WARNING(NULL,
                                               cv::format("VIDEOIO(%s): parameter with key '%d' was unused",
                                                          info.name, key));
                            }
                        }
                        if (iwriter->isOpened())
                        {
                            return true;
                        }
                        iwriter.release();
                    }
                    else
                    {
                        CV_WRITER_LOG_DEBUG(NULL, cv::format("VIDEOIO(%s): can't create writer",
                                                             info.name));
                    }
                }
                catch (const cv::Exception& e)
                {
                    CV_LOG_WARNING(NULL,
                                   cv::format("VIDEOIO(%s): raised OpenCV exception:\n\n%s\n",
                                              info.name, e.what()));
                }
                catch (const std::exception& e)
                {
                    CV_LOG_WARNING(NULL, cv::format("VIDEOIO(%s): raised C++ exception:\n\n%s\n",
                                                    info.name, e.what()));
                }
                catch (...)
                {
                    CV_LOG_WARNING(NULL,
                                   cv::format("VIDEOIO(%s): raised unknown C++ exception!\n\n",
                                              info.name));
                }
            }
            else
            {
                CV_WRITER_LOG_DEBUG(NULL,
                                    cv::format("VIDEOIO(%s): backend is not available "
                                               "(plugin is missing, or can't be loaded due "
                                               "dependencies or it is not compatible)",
                                               info.name));
            }
        }
    }

    if (cv::videoio_registry::checkDeprecatedBackend(apiPreference))
    {
        CV_LOG_DEBUG(NULL,
            cv::format("VIDEOIO(%s): backend is removed from OpenCV",
                cv::videoio_registry::getBackendName((VideoCaptureAPIs) apiPreference).c_str()));
    }
    else
    {
        CV_LOG_DEBUG(NULL, "VIDEOIO: chosen backend does not work or wrong."
            "Please make sure that your computer support chosen backend and OpenCV built "
            "with right flags.");
    }

    return false;
}

bool VideoWriter::isOpened() const
{
    return !iwriter.empty();
}


bool VideoWriter::set(int propId, double value)
{
    CV_CheckNE(propId, (int)CAP_PROP_BACKEND, "Can't set read-only property");

    if (!iwriter.empty())
    {
        return iwriter->setProperty(propId, value);
    }
    return false;
}

double VideoWriter::get(int propId) const
{
    if (propId == CAP_PROP_BACKEND)
    {
        int api = 0;
        if (iwriter)
        {
            api = iwriter->getCaptureDomain();
        }
        return (api <= 0) ?  -1. : static_cast<double>(api);
    }
    if (!iwriter.empty())
    {
        return iwriter->getProperty(propId);
    }
    return CAP_PROP_UNKNOWN;
}

String VideoWriter::getBackendName() const
{
    int api = 0;
    if (iwriter)
    {
        api = iwriter->getCaptureDomain();
    }
    CV_Assert(api != 0);
    return cv::videoio_registry::getBackendName(static_cast<VideoCaptureAPIs>(api));
}

bool VideoWriter::write(InputArray image)
{
    CV_INSTRUMENT_REGION();

    if (iwriter)
    {
        return iwriter->write(image);
    }
    return false;
}

VideoWriter& VideoWriter::operator << (const Mat& image)
{
    CV_INSTRUMENT_REGION();

    write(image);
    return *this;
}

VideoWriter& VideoWriter::operator << (const UMat& image)
{
    CV_INSTRUMENT_REGION();
    write(image);
    return *this;
}

// FIXIT OpenCV 4.0: make inline
int VideoWriter::fourcc(char c1, char c2, char c3, char c4)
{
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24);
}

} // namespace cv
