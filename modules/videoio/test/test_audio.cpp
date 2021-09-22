// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

//file format, epsilon, video type, weight, height, number of frame, fps, psnr Threshold, backend
typedef std::tuple<std::string, double, int, int, int, int, int, double, std::pair<std::string, int> > paramCombination;
typedef std::tuple<std::string, double, std::pair<std::string, int> > param;

class baseAudio
{
protected:
    baseAudio(){};
    void getValidAudioData()
    {
        const double step = 3.14/22050;
        double value = 0;
        for(int j = 0; j < 3; j++)
        {
            value = 0;
            for(int i = 0; i < 44100; i++)
            {
                validAudioData.push_back(sin(value));
                value += step;
            }
        }
    }
protected:
    std::string format;
    double epsilon;
    std::pair<std::string, int> backend;
    std::string root;
    std::string fileName;

    std::vector<double> validAudioData;
    std::vector<double> audioData;
    std::vector<int> params;

    Mat audioFrame;
    VideoCapture cap;
};

class AudioTestFixture : public baseAudio, public testing::TestWithParam <param>
{
public:
    AudioTestFixture()
    {
        format = get<0>(GetParam());
        epsilon = get<1>(GetParam());
        backend = get<2>(GetParam());
        root = "audio/";
        fileName = "test_audio";
        params = {  CAP_PROP_AUDIO_STREAM, 0,
                    CAP_PROP_VIDEO_STREAM, -1,
                    CAP_PROP_AUDIO_DATA_DEPTH, CV_16S };
    }
    void doTest()
    {
        getValidAudioData();
        getFileData();
        checkAudio();
    }

private:
    void getValidAudioData()
    {
        const double step = 3.14/22050;
        double value = 0;
        for(int j = 0; j < 3; j++)
        {
            value = 0;
            for(int i = 0; i < 44100; i++)
            {
                validAudioData.push_back(sin(value));
                value += step;
            }
        }
    }
    void getFileData()
    {
        ASSERT_TRUE(cap.open(findDataFile(root + fileName + "." + format), backend.second, params));
        const int audioBaseIndex = static_cast<int>(cap.get(cv::CAP_PROP_AUDIO_BASE_INDEX));
        double f = 0;
        for (;;)
        {
            if(cap.grab())
            {
                ASSERT_TRUE(cap.retrieve(audioFrame, audioBaseIndex));
                ASSERT_EQ(audioFrame.type(), CV_16SC1);
                for(int i = 0; i < audioFrame.cols; i++)
                {
                    f = ((double) audioFrame.at<signed short>(0,i)) / (double) 32768;
                    audioData.push_back(f);
                }
            }
            else
            {
                break;
            }
        }
        ASSERT_FALSE(audioData.empty());
    }
    void checkAudio()
    {
        for(unsigned int i = 0; i < validAudioData.size(); i++)
        {
            EXPECT_LE(fabs(validAudioData[i] - audioData[i]), epsilon) << "sample index " << i;
        }
    }
};

const param audioParams[] =
{
    param("wav", 0.0001, {"CAP_MSMF", cv::CAP_MSMF}),
    param("mp3", 0.1, {"CAP_MSMF", cv::CAP_MSMF}),
    param("mp4", 0.15, {"CAP_MSMF", cv::CAP_MSMF})
};

class Audio : public AudioTestFixture{};

TEST_P(Audio, audio)
{
    if (!videoio_registry::hasBackend(cv::VideoCaptureAPIs(backend.second)))
        throw SkipTestException(backend.first + " backend was not found");

    doTest();
}

INSTANTIATE_TEST_CASE_P(/**/, Audio, testing::ValuesIn(audioParams));

class MediaTestFixture : public baseAudio, public testing::TestWithParam <paramCombination>
{
public:
    MediaTestFixture():
        videoType(get<2>(GetParam())),
        height(get<3>(GetParam())),
        width(get<4>(GetParam())),
        numberOfFrames(get<5>(GetParam())),
        fps(get<6>(GetParam())),
        psnrThreshold(get<7>(GetParam()))
        {
            format = get<0>(GetParam());
            epsilon = get<1>(GetParam());
            backend = get<8>(GetParam());
            root = "audio/";
            fileName = "test_audio";
            params = {  CAP_PROP_AUDIO_STREAM, 0,
                        CAP_PROP_VIDEO_STREAM, 0,
                        CAP_PROP_AUDIO_DATA_DEPTH, CV_16S };
        };
    void doTest()
    {
        getValidAudioData();
        getValidVideoData();
        getFileData();
        comparisonAudio();
        comparisonVideo();
    }

private:
    void getValidAudioData()
    {
        const double step = 3.14/22050;
        double value = 0;
        for (int j = 0; j < 3; j++)
        {
            value = 0;
            for (int i = 0; i < 44100; i++)
            {
                validAudioData.push_back(sin(value));
                value += step;
            }
        }
    }
    void getValidVideoData()
    {
        Mat img(height, width, videoType);
        for (int i = 0; i < numberOfFrames; ++i)
        {
            generateFrame(i, numberOfFrames, img);
            validVideoData.push_back(img);
        }
    }
    void getFileData()
    {
        ASSERT_TRUE(cap.open(findDataFile(root + fileName + "." + format), backend.second, params));
        const int audioBaseIndex = static_cast<int>(cap.get(cv::CAP_PROP_AUDIO_BASE_INDEX));
        double f = 0;
        int samplePerSecond = (int)cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND);
        ASSERT_EQ(samplePerSecond, 44100);
        int samplesPerFrame = (int)(1./fps*samplePerSecond);
        int audioSamplesTolerance = (int)(samplesPerFrame / 2.);
        for (;;)
        {
            if (cap.grab())
            {
                SCOPED_TRACE(cv::format("frame=%d", (int)(videoData.size()+1)));
                ASSERT_TRUE(cap.retrieve(videoFrame));
                ASSERT_TRUE(cap.retrieve(audioFrame, audioBaseIndex));
                ASSERT_LT(abs(cap.get(CAP_PROP_AUDIO_POS) -  (cap.get(CAP_PROP_POS_MSEC)/ 1000  + 1./fps) * samplePerSecond), audioSamplesTolerance);
                ASSERT_LT(abs(audioFrame.cols - samplesPerFrame), audioSamplesTolerance);
                ASSERT_EQ(audioFrame.type(), CV_16SC1);
                if (!videoFrame.empty())
                    videoData.push_back(videoFrame);
                for (int i = 0; i < audioFrame.cols; i++)
                {
                    f = audioFrame.at<signed short>(0,i) / 32768.0;
                    audioData.push_back(f);
                }
            }
            else
            {
                break;
            }
        }
        ASSERT_FALSE(audioData.empty());
        ASSERT_FALSE(videoData.empty());
    }
    void comparisonAudio()
    {
        for (unsigned int i = 0; i < audioData.size(); i++)
        {
            EXPECT_LE(fabs(validAudioData[i] - audioData[i]), epsilon) << "sample index " << i;
        }
    }
    void comparisonVideo()
    {
        ASSERT_EQ(validVideoData.size(), videoData.size());
        double minPsnrOriginal = 1000;
        for (unsigned int i = 0; i < validVideoData.size(); i++)
        {
            ASSERT_EQ(videoData[i].rows, validVideoData[i].rows) << "The dimension of the rows does not match. Frame index: " << i;
            ASSERT_EQ(videoData[i].cols, validVideoData[i].cols) << "The dimension of the cols does not match. Frame index: " << i;
            double psnr = cvtest::PSNR(validVideoData[i], videoData[i]);
            if (psnr < minPsnrOriginal)
                minPsnrOriginal = psnr;
        }
        EXPECT_GE(minPsnrOriginal, psnrThreshold);
    }
protected:
    const int videoType;
    const int height;
    const int width;
    const int numberOfFrames;
    const int fps;
    const double psnrThreshold;

    std::vector<Mat> validVideoData;
    std::vector<Mat> videoData;

    Mat videoFrame;
};

const paramCombination mediaParams[] =
{
    paramCombination("mp4", 0.15, CV_8UC3, 240, 320, 90, 30, 30., {"CAP_MSMF", cv::CAP_MSMF})
};

class Media : public MediaTestFixture{};

TEST_P(Media, audio)
{
    if (!videoio_registry::hasBackend(cv::VideoCaptureAPIs(backend.second)))
        throw SkipTestException(backend.first + " backend was not found");

    doTest();
}

INSTANTIATE_TEST_CASE_P(/**/, Media, testing::ValuesIn(mediaParams));

}} //namespace
