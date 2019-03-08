/*
 *
 */

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

class AndroidMediaNdkCapture : public IVideoCapture
{

public:
    AndroidMediaNdkCapture():
        mediaExtractor(NULL), mediaCodec(NULL), sawInputEOS(false), sawOutputEOS(false),
        frameWidth(0), frameHeight(0), colorFormat(0), buffer(NULL), bufferSize(0) {}
    AMediaExtractor* mediaExtractor;
    AMediaCodec *mediaCodec;
    bool sawInputEOS;
    bool sawOutputEOS;
    int32_t frameWidth;
    int32_t frameHeight;
    int32_t colorFormat;
    uint8_t* buffer;
    size_t bufferSize;

    ~AndroidMediaNdkCapture() { cleanUp(); }

    void allocateBuffer(size_t bufSize) {
        buffer = (uint8_t*)malloc(bufSize);
        bufferSize = bufSize;
    }

    bool decodeFrame() {
        while (!sawInputEOS || !sawOutputEOS) {
            if (!sawInputEOS) {
                auto bufferIndex = AMediaCodec_dequeueInputBuffer(mediaCodec, INPUT_TIMEOUT_MS);
                LOGV("input buffer %zd", bufferIndex);
                if (bufferIndex >= 0) {
                    size_t bufsize;
                    auto buf = AMediaCodec_getInputBuffer(mediaCodec, bufferIndex, &bufsize);
                    auto sampleSize = AMediaExtractor_readSampleData(mediaExtractor, buf, bufsize);
                    if (sampleSize < 0) {
                        sampleSize = 0;
                        sawInputEOS = true;
                        LOGV("EOS");
                    }
                    auto presentationTimeUs = AMediaExtractor_getSampleTime(mediaExtractor);

                    AMediaCodec_queueInputBuffer(mediaCodec, bufferIndex, 0, sampleSize,
                        presentationTimeUs, sawInputEOS ? AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM : 0);
                    AMediaExtractor_advance(mediaExtractor);
                }
            }

            if (!sawOutputEOS) {
                AMediaCodecBufferInfo info;
                auto bufferIndex = AMediaCodec_dequeueOutputBuffer(mediaCodec, &info, 0);
                if (bufferIndex >= 0) {
                    size_t bufSize = 0;
                    AMediaFormat* mediaFormat = AMediaCodec_getOutputFormat(mediaCodec);
                    AMediaFormat_getInt32(mediaFormat, AMEDIAFORMAT_KEY_WIDTH, &frameWidth);
                    AMediaFormat_getInt32(mediaFormat, AMEDIAFORMAT_KEY_HEIGHT, &frameHeight);
                    AMediaFormat_getInt32(mediaFormat, AMEDIAFORMAT_KEY_COLOR_FORMAT, &colorFormat);
                    uint8_t* codecBuffer = AMediaCodec_getOutputBuffer(mediaCodec, bufferIndex, &bufSize);
                    if (buffer == NULL) {
                        allocateBuffer(bufSize);
                    } else if (bufferSize < bufSize) {
                        free(buffer);
                        allocateBuffer(bufSize);
                    }
                    memcpy(buffer, codecBuffer, bufferSize);
                    LOGV("colorFormat: %d", colorFormat);
                    LOGV("buffer size: %zu", bufferSize);
                    LOGV("width (frame): %d", frameWidth);
                    LOGV("height (frame): %d", frameHeight);
                    if (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
                        LOGV("output EOS");
                        sawOutputEOS = true;
                    }
                    AMediaCodec_releaseOutputBuffer(mediaCodec, bufferIndex, info.size != 0);
                    return true;
                } else if (bufferIndex == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
                    LOGV("output buffers changed");
                } else if (bufferIndex == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
                    auto format = AMediaCodec_getOutputFormat(mediaCodec);
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

    bool isOpened() const CV_OVERRIDE { return mediaCodec != NULL; }

    int getCaptureDomain() CV_OVERRIDE { return CAP_ANDROID; }

    bool grabFrame() CV_OVERRIDE
    {
        // clear the previous frame
        buffer = NULL;
        bufferSize = 0;
        return decodeFrame();
    }

    bool retrieveFrame(int, OutputArray out) CV_OVERRIDE
    {
        if (buffer == NULL) {
            return false;
        }
        Mat yuv(frameHeight + frameHeight/2, frameWidth, CV_8UC1, buffer);
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

        AMediaExtractor *extractor = AMediaExtractor_new();
        media_status_t err = AMediaExtractor_setDataSourceFd(extractor, fd, 0, statBuffer.st_size);
        close(fd);
        if (err != AMEDIA_OK) {
            LOGV("setDataSource error: %d", err);
            return false;
        }

        int numtracks = AMediaExtractor_getTrackCount(extractor);

        AMediaCodec *codec = NULL;

        LOGV("input has %d tracks", numtracks);
        for (int i = 0; i < numtracks; i++) {
            AMediaFormat *format = AMediaExtractor_getTrackFormat(extractor, i);
            const char *s = AMediaFormat_toString(format);
            LOGV("track %d format: %s", i, s);
            const char *mime;
            if (!AMediaFormat_getString(format, AMEDIAFORMAT_KEY_MIME, &mime)) {
                LOGV("no mime type");
                return false;
            } else if (!strncmp(mime, "video/", 6)) {
                int32_t trackWidth, trackHeight;
                AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_WIDTH, &trackWidth);
                AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_HEIGHT, &trackHeight);
                LOGV("width (track): %d", trackWidth);
                LOGV("height (track): %d", trackHeight);
                AMediaExtractor_selectTrack(extractor, i);
                codec = AMediaCodec_createDecoderByType(mime);
                AMediaCodec_configure(codec, format, NULL, NULL, 0);
                this->mediaExtractor = extractor;
                this->mediaCodec = codec;
                this->sawInputEOS = false;
                this->sawOutputEOS = false;
                AMediaCodec_start(codec);
            }
            AMediaFormat_delete(format);
        }
        return true;
    }

    void cleanUp() {
        if (mediaCodec != NULL) {
            AMediaCodec_stop(mediaCodec);
            AMediaCodec_delete(mediaCodec);
            mediaCodec = NULL;
        }
        if (mediaExtractor != NULL) {
            AMediaExtractor_delete(mediaExtractor);
            mediaExtractor = NULL;
        }
        buffer = NULL;
        bufferSize = 0;
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
