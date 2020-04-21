// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/videoio/container_avi.private.hpp"
#include <cstdio>

using namespace cv;

namespace opencv_test
{

TEST(videoio_avi, good_MJPG) {
    String filename = BunnyParameters::getFilename(".mjpg.avi");
    AVIReadContainer in;
    in.initStream(filename);
    frame_list frames;
    ASSERT_TRUE(in.parseRiff(frames));
    EXPECT_EQ(frames.size(), static_cast<unsigned>(BunnyParameters::getCount()));
    EXPECT_EQ(in.getWidth(), static_cast<unsigned>(BunnyParameters::getWidth()));
    EXPECT_EQ(in.getHeight(), static_cast<unsigned>(BunnyParameters::getHeight()));
    EXPECT_EQ(in.getFps(), static_cast<unsigned>(BunnyParameters::getFps()));
}

TEST(videoio_avi, bad_MJPG) {
    String filename = BunnyParameters::getFilename(".avi");
    AVIReadContainer in;
    in.initStream(filename);
    frame_list frames;
    EXPECT_FALSE(in.parseRiff(frames));
    EXPECT_EQ(frames.size(), static_cast<unsigned>(0));
}

TEST(videoio_avi, basic)
{
    const String filename = cv::tempfile("test.avi");
    const double fps = 100;
    const Size sz(800, 600);
    const size_t count = 10;
    const uchar data[count] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0xA};
    const Codecs codec = MJPEG;
    {
        AVIWriteContainer out;
        ASSERT_TRUE(out.initContainer(filename, fps, sz, true));
        ASSERT_TRUE(out.isOpenedStream());
        EXPECT_EQ(out.getWidth(), sz.width);
        EXPECT_EQ(out.getHeight(), sz.height);
        EXPECT_EQ(out.getChannels(), 3);

        out.startWriteAVI(1);
        {
            out.writeStreamHeader(codec); // starts LIST chunk
            size_t chunkPointer = out.getStreamPos();
            int avi_index = out.getAVIIndex(0, dc);
            {
                out.startWriteChunk(avi_index);
                out.putStreamBytes(data, count);
                size_t tempChunkPointer = out.getStreamPos();
                size_t moviPointer = out.getMoviPointer();
                out.pushFrameOffset(chunkPointer - moviPointer);
                out.pushFrameSize(tempChunkPointer - chunkPointer - 8);
                out.endWriteChunk();
            }
            out.endWriteChunk(); // ends LIST chunk
        }
        out.writeIndex(0, dc);
        out.finishWriteAVI();
    }
    {
        AVIReadContainer in;
        in.initStream(filename);
        frame_list frames;
        ASSERT_TRUE(in.parseRiff(frames));
        EXPECT_EQ(in.getFps(), fps);
        EXPECT_EQ(in.getWidth(), static_cast<unsigned>(sz.width));
        EXPECT_EQ(in.getHeight(), static_cast<unsigned>(sz.height));
        ASSERT_EQ(frames.size(), static_cast<unsigned>(1));
        std::vector<char> actual = in.readFrame(frames.begin());
        ASSERT_EQ(actual.size(), count);
        for (size_t i = 0; i < count; ++i)
            EXPECT_EQ(actual.at(i), data[i]) << "at index " << i;
    }
    remove(filename.c_str());
}

}
