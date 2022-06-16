// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_OAK_HPP
#define OPENCV_GAPI_OAK_HPP

#include <opencv2/gapi/garg.hpp>       // IStreamSource
#include <opencv2/gapi/gkernel.hpp>    // GKernelPackage
#include <opencv2/gapi/gstreaming.hpp> // GOptRunArgsP

namespace cv {
namespace gapi {
namespace oak {

// FIXME: copypasted from dai library
struct EncoderConfig {
    /**
     * Rate control mode specifies if constant or variable bitrate should be used (H264 / H265)
     */
    enum class RateControlMode: int { CBR, VBR };

    /**
     * Encoding profile, H264, H265 or MJPEG
     */
    enum class Profile: int { H264_BASELINE, H264_HIGH, H264_MAIN, H265_MAIN, MJPEG };
    /**
     * Specifies preferred bitrate (kb) of compressed output bitstream
     */
    std::int32_t bitrate = 8000;
    /**
     * Every x number of frames a keyframe will be inserted
     */
    std::int32_t keyframeFrequency = 30;
    /**
     * Specifies maximum bitrate (kb) of compressed output bitstream
     */
    std::int32_t maxBitrate = 8000;
    /**
     * Specifies number of B frames to be inserted
     */
    std::int32_t numBFrames = 0;
    /**
     * This options specifies how many frames are available in this nodes pool (can help if
     * receiver node is slow at consuming
     */
    std::uint32_t numFramesPool = 4;
    /**
     * Encoding profile, H264, H265 or MJPEG
     */
    Profile profile = Profile::H265_MAIN;
    /**
     * Value between 0-100% (approximates quality)
     */
    std::int32_t quality = 80;
    /**
     * Lossless mode ([M]JPEG only)
     */
    bool lossless = false;
    /**
     * Rate control mode specifies if constant or variable bitrate should be used (H264 / H265)
     */
    RateControlMode rateCtrlMode = RateControlMode::CBR;
    /**
     * Input and compressed output frame width
     */
    std::int32_t width = 1920;
    /**
     * Input and compressed output frame height
     */
    std::int32_t height = 1080;
    /**
     * Frame rate
     */
    float frameRate = 30.0f;
};

G_API_OP(GEncFrame, <GArray<uint8_t>(GFrame, EncoderConfig)>, "org.opencv.oak.enc_frame") {
    static GArrayDesc outMeta(const GFrameDesc&, const EncoderConfig&) {
        return cv::empty_array_desc();
    }
};

G_API_OP(GSobelXY, <GFrame(GFrame, const cv::Mat&, const cv::Mat&)>, "org.opencv.oak.sobelxy") {
    static GFrameDesc outMeta(const GFrameDesc& in, const cv::Mat&, const cv::Mat&) {
        return in;
    }
};

G_API_OP(GCopy, <GFrame(GFrame)>, "org.opencv.oak.copy") {
    static GFrameDesc outMeta(const GFrameDesc& in) {
        return in;
    }
};

// FIXME: add documentation on operations below

GAPI_EXPORTS GArray<uint8_t> encode(const GFrame& in, const EncoderConfig&);

GAPI_EXPORTS GFrame sobelXY(const GFrame& in,
                            const cv::Mat& hk,
                            const cv::Mat& vk);

GAPI_EXPORTS GFrame copy(const GFrame& in);

// OAK backend & kernels ////////////////////////////////////////////////////////
GAPI_EXPORTS cv::gapi::GBackend backend();
GAPI_EXPORTS cv::gapi::GKernelPackage kernels();

// Camera object ///////////////////////////////////////////////////////////////

struct GAPI_EXPORTS ColorCameraParams {
    /**
     * Format of the frame one gets from the camera
     */
    bool interleaved = false;

    // FIXME: extend
    enum class BoardSocket: int { RGB, BGR };

    BoardSocket board_socket = BoardSocket::RGB;

    // FIXME: extend
    enum class Resolution: int { THE_1080_P };

    Resolution resolution = Resolution::THE_1080_P;
};

class GAPI_EXPORTS ColorCamera: public cv::gapi::wip::IStreamSource {
    cv::MediaFrame m_dummy;
    ColorCameraParams m_params;

    virtual bool pull(cv::gapi::wip::Data &data) override;
    virtual GMetaArg descr_of() const override;

public:
    ColorCamera();
    explicit ColorCamera(const ColorCameraParams& params);
};

} // namespace oak
} // namespace gapi

namespace detail {
template<> struct CompileArgTag<gapi::oak::ColorCameraParams> {
    static const char* tag() { return "gapi.oak.colorCameraParams"; }
};

template<> struct CompileArgTag<gapi::oak::EncoderConfig> {
    static const char* tag() { return "gapi.oak.encoderConfig"; }
};
} // namespace detail

} // namespace cv

#endif // OPENCV_GAPI_OAK_HPP
