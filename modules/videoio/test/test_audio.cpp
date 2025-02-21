// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

//file name, number of audio channels, epsilon, video type, weight, height, number of frame, number of audio samples, fps, psnr Threshold, backend
typedef std::tuple<std::string, int, double, int, int, int, int, int, int, double, VideoCaptureAPIs> paramCombination;
//file name, number of audio channels, number of audio samples, epsilon, backend
typedef std::tuple<std::string, int, int, double, VideoCaptureAPIs> param;

class AudioBaseTest
{
protected:
    AudioBaseTest(){}
    void getValidAudioData()
    {
        const double step = 3.14/22050;
        double value = 0;
        validAudioData.resize(expectedNumAudioCh);
        for (int nCh = 0; nCh < expectedNumAudioCh; nCh++)
        {
            value = 0;
            for(unsigned int i = 0; i < numberOfSamples; i++)
            {
                if (i != 0 && i % 44100 == 0)
                    value = 0;
                validAudioData[nCh].push_back(sin(value));
                value += step;
            }
        }
    }
    void checkAudio()
    {
        getValidAudioData();

        ASSERT_EQ(expectedNumAudioCh, (int)audioData.size());
        for (unsigned int nCh = 0; nCh < audioData.size(); nCh++)
        {
#ifdef _WIN32
            if (audioData[nCh].size() == 132924 && numberOfSamples == 131819 && fileName == "test_audio.mp4")
                throw SkipTestException("Detected failure observed on legacy Windows versions. SKIP");
#endif
            ASSERT_EQ(numberOfSamples, audioData[nCh].size()) << "nCh=" << nCh;
            for (unsigned int i = 0; i < numberOfSamples; i++)
            {
                EXPECT_NEAR(validAudioData[nCh][i], audioData[nCh][i], epsilon) << "sample index=" << i << " nCh=" << nCh;
            }
        }
    }
protected:
    int expectedNumAudioCh;
    unsigned int numberOfSamples;
    double epsilon;
    VideoCaptureAPIs backend;
    std::string root;
    std::string fileName;

    std::vector<std::vector<double>> validAudioData;
    std::vector<std::vector<double>> audioData;
    std::vector<int> params;

    Mat audioFrame;
    VideoCapture cap;
};

class AudioTestFixture : public AudioBaseTest, public testing::TestWithParam <param>
{
public:
    AudioTestFixture()
    {
        fileName = get<0>(GetParam());
        expectedNumAudioCh = get<1>(GetParam());
        numberOfSamples = get<2>(GetParam());
        epsilon = get<3>(GetParam());
        backend = get<4>(GetParam());
        root = "audio/";
        params = {  CAP_PROP_AUDIO_STREAM, 0,
                    CAP_PROP_VIDEO_STREAM, -1,
                    CAP_PROP_AUDIO_DATA_DEPTH, CV_16S };
    }

    void doTest()
    {
        ASSERT_TRUE(cap.open(findDataFile(root + fileName), backend, params));
        const int audioBaseIndex = static_cast<int>(cap.get(cv::CAP_PROP_AUDIO_BASE_INDEX));
        const int numberOfChannels = (int)cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS);
        ASSERT_EQ(expectedNumAudioCh, numberOfChannels);
        double f = 0;
        audioData.resize(numberOfChannels);
        for (;;)
        {
            if (cap.grab())
            {
                for (int nCh = 0; nCh < numberOfChannels; nCh++)
                {
                    ASSERT_TRUE(cap.retrieve(audioFrame, audioBaseIndex + nCh));
                    ASSERT_EQ(CV_16SC1, audioFrame.type()) << audioData[nCh].size();
                    for (int i = 0; i < audioFrame.cols; i++)
                    {
                        f = ((double) audioFrame.at<signed short>(0,i)) / (double) 32768;
                        audioData[nCh].push_back(f);
                    }
                }
            }
            else { break; }
        }
        ASSERT_FALSE(audioData.empty());

        checkAudio();
    }
};

const param audioParams[] =
{
#ifdef _WIN32
    param("test_audio.wav", 1, 132300, 0.0001, cv::CAP_MSMF),
    param("test_mono_audio.mp3", 1, 133104, 0.12, cv::CAP_MSMF),
    param("test_stereo_audio.mp3", 2, 133104, 0.12, cv::CAP_MSMF),
    param("test_audio.mp4", 1, 133104, 0.15, cv::CAP_MSMF),
#endif
    param("test_audio.wav", 1, 132300, 0.0001, cv::CAP_GSTREAMER),
    param("test_audio.mp4", 1, 132522, 0.15, cv::CAP_GSTREAMER),
};

class Audio : public AudioTestFixture{};

TEST_P(Audio, audio)
{
    if (!videoio_registry::hasBackend(cv::VideoCaptureAPIs(backend)))
        throw SkipTestException(cv::videoio_registry::getBackendName(backend) + " backend was not found");

    doTest();
}

inline static std::string Audio_name_printer(const testing::TestParamInfo<Audio::ParamType>& info)
{
    std::ostringstream out;
    out << getExtensionSafe(get<0>(info.param)) << "_"
        << get<1>(info.param) << "CN" << "_"
        << getBackendNameSafe(get<4>(info.param));
    return out.str();
}

INSTANTIATE_TEST_CASE_P(/**/, Audio, testing::ValuesIn(audioParams), Audio_name_printer);

class MediaTestFixture : public AudioBaseTest, public testing::TestWithParam <paramCombination>
{
public:
    MediaTestFixture():
        videoType(get<3>(GetParam())),
        height(get<4>(GetParam())),
        width(get<5>(GetParam())),
        numberOfFrames(get<6>(GetParam())),
        fps(get<8>(GetParam())),
        psnrThreshold(get<9>(GetParam()))
        {
            fileName = get<0>(GetParam());
            expectedNumAudioCh = get<1>(GetParam());
            numberOfSamples = get<7>(GetParam());
            epsilon = get<2>(GetParam());
            backend = get<10>(GetParam());
            root = "audio/";
            params = {  CAP_PROP_AUDIO_STREAM, 0,
                        CAP_PROP_VIDEO_STREAM, 0,
                        CAP_PROP_AUDIO_DATA_DEPTH, CV_16S };
        }

    void doTest()
    {
        ASSERT_TRUE(cap.open(findDataFile(root + fileName), backend, params));

        const int audioBaseIndex = static_cast<int>(cap.get(cv::CAP_PROP_AUDIO_BASE_INDEX));
        const int numberOfChannels = (int)cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS);
        ASSERT_EQ(expectedNumAudioCh, numberOfChannels);

        const int samplePerSecond = (int)cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND);
        ASSERT_EQ(44100, samplePerSecond);
        int samplesPerFrame = (int)(1./fps*samplePerSecond);

        double audio0_timestamp = 0;

        Mat videoFrame;
        Mat img(height, width, videoType);
        audioData.resize(numberOfChannels);
        for (int frame = 0; frame < numberOfFrames; frame++)
        {
            SCOPED_TRACE(cv::format("frame=%d", frame));

            ASSERT_TRUE(cap.grab());
            if (frame == 0)
            {
                double audio_shift = cap.get(CAP_PROP_AUDIO_SHIFT_NSEC);
                double video0_timestamp = cap.get(CAP_PROP_POS_MSEC) * 1e-3;
                audio0_timestamp = video0_timestamp + audio_shift * 1e-9;

                std::cout << "video0 timestamp: " << video0_timestamp << "  audio0 timestamp: " << audio0_timestamp << " (audio shift nanoseconds: " << audio_shift << " , seconds: " << audio_shift * 1e-9 << ")" << std::endl;
            }
            ASSERT_TRUE(cap.retrieve(videoFrame));
            if (epsilon >= 0)
            {
                generateFrame(frame, numberOfFrames, img);
                ASSERT_EQ(img.size, videoFrame.size);
                double psnr = cvtest::PSNR(img, videoFrame);
                EXPECT_GE(psnr, psnrThreshold);
            }

            int audioFrameCols = 0;
            for (int nCh = 0; nCh < numberOfChannels; nCh++)
            {
                ASSERT_TRUE(cap.retrieve(audioFrame, audioBaseIndex+nCh));
                if (audioFrame.empty())
                    continue;
                ASSERT_EQ(CV_16SC1, audioFrame.type());
                if (nCh == 0)
                    audioFrameCols = audioFrame.cols;
                else
                    ASSERT_EQ(audioFrameCols, audioFrame.cols) << "channel "<< nCh;
                for (int i = 0; i < audioFrame.cols; i++)
                {
                    double f = audioFrame.at<signed short>(0,i) / 32768.0;
                    audioData[nCh].push_back(f);
                }
            }

            if (frame < 5 || frame >= numberOfFrames-5)
                std::cout << "frame=" << frame << ":  audioFrameSize=" << audioFrameCols << "  videoTimestamp=" << cap.get(CAP_PROP_POS_MSEC) << " ms" << std::endl;
            else if (frame == 6)
                std::cout << "frame..." << std::endl;

            if (audioFrameCols == 0)
                continue;
            if (frame != 0 && frame != numberOfFrames-1)
            {
                // validate audio position
                EXPECT_NEAR(
                        cap.get(CAP_PROP_AUDIO_POS) / samplePerSecond + audio0_timestamp,
                        cap.get(CAP_PROP_POS_MSEC) * 1e-3,
                        (1.0 / fps) * 0.6)
                    << "CAP_PROP_AUDIO_POS=" << cap.get(CAP_PROP_AUDIO_POS) << " CAP_PROP_POS_MSEC=" << cap.get(CAP_PROP_POS_MSEC);
            }
            if (frame != 0 && frame != numberOfFrames-1 && audioData[0].size() != (size_t)numberOfSamples)
            {
                if (backend == cv::CAP_MSMF)
                {
                    int audioSamplesTolerance = samplesPerFrame / 2;
                    // validate audio frame size
                    EXPECT_NEAR(audioFrame.cols, samplesPerFrame, audioSamplesTolerance);
                }
            }
        }
        ASSERT_FALSE(cap.grab());
        ASSERT_FALSE(audioData.empty());

        std::cout << "Total audio samples=" << audioData[0].size() << std::endl;

        if (epsilon >= 0)
            checkAudio();
    }
protected:
    const int videoType;
    const int height;
    const int width;
    const int numberOfFrames;
    const int fps;
    const double psnrThreshold;
};

class Media : public MediaTestFixture{};

TEST_P(Media, audio)
{
    if (!videoio_registry::hasBackend(cv::VideoCaptureAPIs(backend)))
        throw SkipTestException(cv::videoio_registry::getBackendName(backend) + " backend was not found");
    if (cvtest::skipUnstableTests && backend == CAP_GSTREAMER)
        throw SkipTestException("Unstable GStreamer test");

    doTest();
}

const paramCombination mediaParams[] =
{
    paramCombination("test_audio.mp4", 1, 0.15, CV_8UC3, 240, 320, 90, 132299, 30, 30., cv::CAP_GSTREAMER)
#ifdef _WIN32
    , paramCombination("test_audio.mp4", 1, 0.15, CV_8UC3, 240, 320, 90, 131819, 30, 30., cv::CAP_MSMF)
#if 0
    // https://filesamples.com/samples/video/mp4/sample_960x400_ocean_with_audio.mp4
    , paramCombination("sample_960x400_ocean_with_audio.mp4", 2, -1/*eplsilon*/, CV_8UC3, 400, 960, 1116, 2056588, 30, 30., cv::CAP_MSMF)
#endif
#endif  // _WIN32
};

inline static std::string Media_name_printer(const testing::TestParamInfo<Media::ParamType>& info)
{
    std::ostringstream out;
    out << getExtensionSafe(get<0>(info.param)) << "_"
        << get<1>(info.param) << "CN" << "_"
        << getBackendNameSafe(get<10>(info.param));
    return out.str();
}

INSTANTIATE_TEST_CASE_P(/**/, Media, testing::ValuesIn(mediaParams), Media_name_printer);

TEST(AudioOpenCheck, bad_arg_invalid_audio_stream)
{
    if (!videoio_registry::hasBackend(cv::VideoCaptureAPIs(cv::CAP_MSMF)))
        throw SkipTestException("CAP_MSMF backend was not found");

    std::string fileName = "audio/test_audio.wav";
    std::vector<int> params {
         CAP_PROP_AUDIO_STREAM, 1,
         CAP_PROP_VIDEO_STREAM, -1,  // disabled
         CAP_PROP_AUDIO_DATA_DEPTH, CV_16S
    };
    VideoCapture cap;
    cap.open(findDataFile(fileName), cv::CAP_MSMF, params);
    ASSERT_FALSE(cap.isOpened());
}

TEST(AudioOpenCheck, bad_arg_invalid_audio_stream_video)
{
    if (!videoio_registry::hasBackend(cv::VideoCaptureAPIs(cv::CAP_MSMF)))
        throw SkipTestException("CAP_MSMF backend was not found");

    std::string fileName = "audio/test_audio.mp4";
    std::vector<int> params {
         CAP_PROP_AUDIO_STREAM, 1,
         CAP_PROP_VIDEO_STREAM, 0,
         CAP_PROP_AUDIO_DATA_DEPTH, CV_16S
    };
    VideoCapture cap;
    cap.open(findDataFile(fileName), cv::CAP_MSMF, params);
    ASSERT_FALSE(cap.isOpened());
}


TEST(AudioOpenCheck, MSMF_bad_arg_invalid_audio_sample_per_second)
{
    if (!videoio_registry::hasBackend(cv::VideoCaptureAPIs(cv::CAP_MSMF)))
        throw SkipTestException("CAP_MSMF backend was not found");

    std::string fileName = "audio/test_audio.mp4";
    std::vector<int> params {
        CAP_PROP_AUDIO_STREAM, 0,
        CAP_PROP_VIDEO_STREAM, -1,  // disabled
        CAP_PROP_AUDIO_SAMPLES_PER_SECOND, (int)1e9
    };
    VideoCapture cap;
    cap.open(findDataFile(fileName), cv::CAP_MSMF, params);
    ASSERT_FALSE(cap.isOpened());
}

TEST(AudioOpenCheck, bad_arg_invalid_audio_sample_per_second)
{
    std::string fileName = "audio/test_audio.mp4";
    std::vector<int> params {
        CAP_PROP_AUDIO_STREAM, 0,
        CAP_PROP_VIDEO_STREAM, -1,  // disabled
        CAP_PROP_AUDIO_SAMPLES_PER_SECOND, -1000
    };
    VideoCapture cap;
    cap.open(findDataFile(fileName), cv::CAP_ANY, params);
    ASSERT_FALSE(cap.isOpened());
}

}} //namespace
