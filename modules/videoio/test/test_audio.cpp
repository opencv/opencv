// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

//file name, number of audio channels, epsilon, video type, weight, height, number of frame, number of audio samples, fps, psnr Threshold, backend
typedef std::tuple<std::string, int, double, int, int, int, int, int, int, double, VideoCaptureAPIs> paramCombination;
//file name, number of audio channels, epsilon, backend
typedef std::tuple<std::string, int, double, VideoCaptureAPIs> param;

class AudioBaseTest
{
protected:
    AudioBaseTest(){};
    void getValidAudioData()
    {
        const double step = 3.14/22050;
        double value = 0;
        validAudioData.resize(expectedNumAudioCh);
        for (int nCh = 0; nCh < expectedNumAudioCh; nCh++)
        {
            for (int j = 0; j < 3; j++)
            {
                value = 0;
                for(int i = 0; i < 44100; i++)
                {
                    validAudioData[nCh].push_back(sin(value));
                    value += step;
                }
            }
        }
    }
protected:
    int expectedNumAudioCh;
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
        epsilon = get<2>(GetParam());
        backend = get<3>(GetParam());
        root = "audio/";
        params = {  CAP_PROP_AUDIO_STREAM, 0,
                    CAP_PROP_VIDEO_STREAM, -1,
                    CAP_PROP_AUDIO_DATA_DEPTH, CV_16S };
    }

    void doTest()
    {
        getValidAudioData();

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
                    ASSERT_TRUE(cap.retrieve(audioFrame, audioBaseIndex));
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
    void checkAudio()
    {
        for (unsigned int nCh = 0; nCh < audioData.size(); nCh++)
            for (unsigned int i = 0; i < validAudioData[nCh].size(); i++)
            {
                EXPECT_LE(fabs(validAudioData[nCh][i] - audioData[nCh][i]), epsilon) << "sample index " << i;
            }
    }
};

const param audioParams[] =
{
    param("test_audio.wav", 1, 0.0001, cv::CAP_MSMF),
    param("test_mono_audio.mp3", 1, 0.1, cv::CAP_MSMF),
    param("test_stereo_audio.mp3", 2, 0.1, cv::CAP_MSMF),
    param("test_audio.mp4", 1, 0.15, cv::CAP_MSMF)
};

class Audio : public AudioTestFixture{};

TEST_P(Audio, audio)
{
    if (!videoio_registry::hasBackend(cv::VideoCaptureAPIs(backend)))
        throw SkipTestException(cv::videoio_registry::getBackendName(backend) + " backend was not found");

    doTest();
}

INSTANTIATE_TEST_CASE_P(/**/, Audio, testing::ValuesIn(audioParams));

class MediaTestFixture : public AudioBaseTest, public testing::TestWithParam <paramCombination>
{
public:
    MediaTestFixture():
        videoType(get<3>(GetParam())),
        height(get<4>(GetParam())),
        width(get<5>(GetParam())),
        numberOfFrames(get<6>(GetParam())),
        numberOfSamples(get<7>(GetParam())),
        fps(get<8>(GetParam())),
        psnrThreshold(get<9>(GetParam()))
        {
            fileName = get<0>(GetParam());
            expectedNumAudioCh = get<1>(GetParam());
            epsilon = get<2>(GetParam());
            backend = get<10>(GetParam());
            root = "audio/";
            params = {  CAP_PROP_AUDIO_STREAM, 0,
                        CAP_PROP_VIDEO_STREAM, 0,
                        CAP_PROP_AUDIO_DATA_DEPTH, CV_16S };
        };

    void doTest()
    {
        getValidAudioData();

        ASSERT_TRUE(cap.open(findDataFile(root + fileName), backend, params));

        const int audioBaseIndex = static_cast<int>(cap.get(cv::CAP_PROP_AUDIO_BASE_INDEX));
        const int numberOfChannels = (int)cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS);
        ASSERT_EQ(expectedNumAudioCh, numberOfChannels);

        const int samplePerSecond = (int)cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND);
        ASSERT_EQ(44100, samplePerSecond);
        int samplesPerFrame = (int)(1./fps*samplePerSecond);
        int audioSamplesTolerance = samplesPerFrame / 2;

        double f = 0;
        Mat videoFrame;
        Mat img(height, width, videoType);
        audioData.resize(numberOfChannels);
        for (int frame = 0; frame < numberOfFrames; frame++)
        {
            SCOPED_TRACE(cv::format("frame=%d", frame));

            ASSERT_TRUE(cap.grab());

            ASSERT_TRUE(cap.retrieve(videoFrame));
            generateFrame(frame, numberOfFrames, img);
            ASSERT_EQ(img.size, videoFrame.size);
            double psnr = cvtest::PSNR(img, videoFrame);
            EXPECT_GE(psnr, psnrThreshold);

            int audioFrameCols = 0;
            for (int nCh = 0; nCh < numberOfChannels; nCh++)
            {
                ASSERT_TRUE(cap.retrieve(audioFrame, audioBaseIndex+nCh));
                if (!audioFrame.empty())
                    ASSERT_EQ(CV_16SC1, audioFrame.type());
                if (nCh == 0)
                    audioFrameCols = audioFrame.cols;
                else
                    ASSERT_EQ(audioFrameCols, audioFrame.cols) << "channel "<< nCh;
                for (int i = 0; i < audioFrame.cols; i++)
                {
                    f = audioFrame.at<signed short>(0,i) / 32768.0;
                    audioData[nCh].push_back(f);
                }
            }
            if (!audioFrame.empty())
                if (frame != 0 && frame != numberOfFrames-1)
                {

                    EXPECT_NEAR(cap.get(CAP_PROP_AUDIO_POS), ((cap.get(CAP_PROP_POS_MSEC) - cap.get(CAP_PROP_AUDIO_SHIFT_NSEC) / 1e9) / 1000) * samplePerSecond, audioSamplesTolerance) << "CAP_PROP_POS_MSEC = " << cap.get(CAP_PROP_POS_MSEC);
                    EXPECT_LT(abs(audioFrame.cols - samplesPerFrame), audioSamplesTolerance);
                }
        }
        ASSERT_FALSE(cap.grab());
        ASSERT_FALSE(audioData.empty());

        checkAudio();
    }
    void checkAudio()
    {
        for (unsigned int nCh = 0; nCh < audioData.size(); nCh++)
        {
            ASSERT_EQ(numberOfSamples, audioData[nCh].size());
            for (int i = 0; i < numberOfSamples; i++)
            {
                EXPECT_LE(fabs(validAudioData[nCh][i] - audioData[nCh][i]), epsilon) << "sample index " << i;
            }
        }
    }
protected:
    const int videoType;
    const int height;
    const int width;
    const int numberOfFrames;
    const int numberOfSamples;
    const int fps;
    const double psnrThreshold;
};

const paramCombination mediaParams[] =
{
    paramCombination("test_audio.mp4", 1, 0.15, CV_8UC3, 240, 320, 90, 131819, 30, 30., cv::CAP_MSMF)
};

class Media : public MediaTestFixture{};

TEST_P(Media, audio)
{
    if (!videoio_registry::hasBackend(cv::VideoCaptureAPIs(backend)))
        throw SkipTestException(cv::videoio_registry::getBackendName(backend) + " backend was not found");

    doTest();
}

INSTANTIATE_TEST_CASE_P(/**/, Media, testing::ValuesIn(mediaParams));

}} //namespace
