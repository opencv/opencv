// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <android/log.h>

#include "media/NdkMediaCodec.h"
#include "media/NdkMediaExtractor.h"

#define INPUT_TIMEOUT_MS 2000

#define COLOR_FormatYUV420Planar 19
#define COLOR_FormatYUV420SemiPlanar 21

using namespace cv;

#define TAG "NativeCodec"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)


static inline void deleter_AMediaExtractor(AMediaExtractor *extractor) {
    AMediaExtractor_delete(extractor);
}

static inline void deleter_AMediaCodec(AMediaCodec *codec) {
    AMediaCodec_stop(codec);
    AMediaCodec_delete(codec);
}

static inline void deleter_AMediaFormat(AMediaFormat *format) {
    AMediaFormat_delete(format);
}

class AndroidMediaNdkCapture : public IVideoCapture
{

public:
    AndroidMediaNdkCapture():
        sawInputEOS(false), sawOutputEOS(false),
        frameWidth(0), frameHeight(0), colorFormat(0) {}
    std::shared_ptr<AMediaExtractor> mediaExtractor;
    std::shared_ptr<AMediaCodec> mediaCodec;
    bool sawInputEOS;
    bool sawOutputEOS;
    int32_t frameWidth;
    int32_t frameHeight;
    int32_t colorFormat;
    std::vector<uint8_t> buffer;

    ~AndroidMediaNdkCapture() { cleanUp(); }

    bool decodeFrame() {
        while (!sawInputEOS || !sawOutputEOS) {
            if (!sawInputEOS) {
                auto bufferIndex = AMediaCodec_dequeueInputBuffer(mediaCodec.get(), INPUT_TIMEOUT_MS);
                LOGV("input buffer %zd", bufferIndex);
                if (bufferIndex >= 0) {
                    size_t bufferSize;
                    auto inputBuffer = AMediaCodec_getInputBuffer(mediaCodec.get(), bufferIndex, &bufferSize);
                    auto sampleSize = AMediaExtractor_readSampleData(mediaExtractor.get(), inputBuffer, bufferSize);
                    if (sampleSize < 0) {
                        sampleSize = 0;
                        sawInputEOS = true;
                        LOGV("EOS");
                    }
                    auto presentationTimeUs = AMediaExtractor_getSampleTime(mediaExtractor.get());

                    AMediaCodec_queueInputBuffer(mediaCodec.get(), bufferIndex, 0, sampleSize,
                        presentationTimeUs, sawInputEOS ? AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM : 0);
                    AMediaExtractor_advance(mediaExtractor.get());
                }
            }

            if (!sawOutputEOS) {
                AMediaCodecBufferInfo info;
                auto bufferIndex = AMediaCodec_dequeueOutputBuffer(mediaCodec.get(), &info, 0);
                if (bufferIndex >= 0) {
                    size_t bufferSize = 0;
                    auto mediaFormat = std::shared_ptr<AMediaFormat>(AMediaCodec_getOutputFormat(mediaCodec.get()), deleter_AMediaFormat);
                    AMediaFormat_getInt32(mediaFormat.get(), AMEDIAFORMAT_KEY_WIDTH, &frameWidth);
                    AMediaFormat_getInt32(mediaFormat.get(), AMEDIAFORMAT_KEY_HEIGHT, &frameHeight);
                    AMediaFormat_getInt32(mediaFormat.get(), AMEDIAFORMAT_KEY_COLOR_FORMAT, &colorFormat);
                    uint8_t* codecBuffer = AMediaCodec_getOutputBuffer(mediaCodec.get(), bufferIndex, &bufferSize);
                    buffer = std::vector<uint8_t>(codecBuffer + info.offset, codecBuffer + bufferSize);
                    LOGV("colorFormat: %d", colorFormat);
                    LOGV("buffer size: %zu", bufferSize);
                    LOGV("width (frame): %d", frameWidth);
                    LOGV("height (frame): %d", frameHeight);
                    if (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
                        LOGV("output EOS");
                        sawOutputEOS = true;
                    }
                    AMediaCodec_releaseOutputBuffer(mediaCodec.get(), bufferIndex, info.size != 0);
                    return true;
                } else if (bufferIndex == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
                    LOGV("output buffers changed");
                } else if (bufferIndex == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
                    auto format = AMediaCodec_getOutputFormat(mediaCodec.get());
                    LOGV("format changed to: %s", AMediaFormat_toString(format));
                    AMediaFormat_delete(format);
                } else if (bufferIndex == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
                    LOGV("no output buffer right now");
                } else {
                    LOGV("unexpected info code: %zd", bufferIndex);
                }
            }
        }
        return false;
    }

    bool isOpened() const CV_OVERRIDE { return mediaCodec.get() != nullptr; }

    int getCaptureDomain() CV_OVERRIDE { return CAP_ANDROID; }

    bool grabFrame() CV_OVERRIDE
    {
        // clear the previous frame
        buffer.clear();
        return decodeFrame();
    }

    bool retrieveFrame(int, OutputArray out) CV_OVERRIDE
    {
        if (buffer.empty()) {
            return false;
        }
        Mat yuv(frameHeight + frameHeight/2, frameWidth, CV_8UC1, buffer.data());
        if (colorFormat == COLOR_FormatYUV420Planar) {
            cv::cvtColor(yuv, out, cv::COLOR_YUV2BGR_YV12);
        } else if (colorFormat == COLOR_FormatYUV420SemiPlanar) {
            cv::cvtColor(yuv, out, cv::COLOR_YUV2BGR_NV21);
        } else {
            LOGE("Unsupported video format: %d", colorFormat);
            return false;
        }
        return true;
    }

    double getProperty(int property_id) const CV_OVERRIDE
    {
        switch (property_id)
        {
            case CV_CAP_PROP_FRAME_WIDTH: return frameWidth;
            case CV_CAP_PROP_FRAME_HEIGHT: return frameHeight;
        }
        return 0;
    }

    bool setProperty(int /* property_id */, double /* value */) CV_OVERRIDE
    {
        return false;
    }

    bool initCapture(const char * filename)
    {
        struct stat statBuffer;
        if (stat(filename, &statBuffer) != 0) {
            LOGE("failed to stat file: %s (%s)", filename, strerror(errno));
            return false;
        }

        int fd = open(filename, O_RDONLY);

        if (fd < 0) {
            LOGE("failed to open file: %s %d (%s)", filename, fd, strerror(errno));
            return false;
        }

        mediaExtractor = std::shared_ptr<AMediaExtractor>(AMediaExtractor_new(), deleter_AMediaExtractor);
        if (!mediaExtractor) {
            return false;
        }
        media_status_t err = AMediaExtractor_setDataSourceFd(mediaExtractor.get(), fd, 0, statBuffer.st_size);
        close(fd);
        if (err != AMEDIA_OK) {
            LOGV("setDataSource error: %d", err);
            return false;
        }

        int numtracks = AMediaExtractor_getTrackCount(mediaExtractor.get());

        LOGV("input has %d tracks", numtracks);
        for (int i = 0; i < numtracks; i++) {
            auto format = std::shared_ptr<AMediaFormat>(AMediaExtractor_getTrackFormat(mediaExtractor.get(), i), deleter_AMediaFormat);
            if (!format) {
                continue;
            }
            const char *s = AMediaFormat_toString(format.get());
            LOGV("track %d format: %s", i, s);
            const char *mime;
            if (!AMediaFormat_getString(format.get(), AMEDIAFORMAT_KEY_MIME, &mime)) {
                LOGV("no mime type");
            } else if (!strncmp(mime, "video/", 6)) {
                int32_t trackWidth, trackHeight;
                AMediaFormat_getInt32(format.get(), AMEDIAFORMAT_KEY_WIDTH, &trackWidth);
                AMediaFormat_getInt32(format.get(), AMEDIAFORMAT_KEY_HEIGHT, &trackHeight);
                LOGV("width (track): %d", trackWidth);
                LOGV("height (track): %d", trackHeight);
                if (AMediaExtractor_selectTrack(mediaExtractor.get(), i) != AMEDIA_OK) {
                    continue;
                }
                mediaCodec = std::shared_ptr<AMediaCodec>(AMediaCodec_createDecoderByType(mime), deleter_AMediaCodec);
                if (!mediaCodec) {
                    continue;
                }
                if (AMediaCodec_configure(mediaCodec.get(), format.get(), NULL, NULL, 0) != AMEDIA_OK) {
                    continue;
                }
                sawInputEOS = false;
                sawOutputEOS = false;
                if (AMediaCodec_start(mediaCodec.get()) != AMEDIA_OK) {
                    continue;
                }
                return true;
            }
        }

        return false;
    }

    void cleanUp() {
        sawInputEOS = true;
        sawOutputEOS = true;
        frameWidth = 0;
        frameHeight = 0;
        colorFormat = 0;
    }
};

/****************** Implementation of interface functions ********************/

Ptr<IVideoCapture> cv::createAndroidCapture_file(const std::string &filename) {
    Ptr<AndroidMediaNdkCapture> res = makePtr<AndroidMediaNdkCapture>();
    if (res && res->initCapture(filename.c_str()))
        return res;
    return Ptr<IVideoCapture>();
}
