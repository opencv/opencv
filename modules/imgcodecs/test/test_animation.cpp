// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static void readFileBytes(const std::string& fname, std::vector<unsigned char>& buf)
{
    FILE * wfile = fopen(fname.c_str(), "rb");
    if (wfile != NULL)
    {
        fseek(wfile, 0, SEEK_END);
        size_t wfile_size = ftell(wfile);
        fseek(wfile, 0, SEEK_SET);

        buf.resize(wfile_size);

        size_t data_size = fread(&buf[0], 1, wfile_size, wfile);

        if(wfile)
        {
            fclose(wfile);
        }

        EXPECT_EQ(data_size, wfile_size);
    }
}

static bool fillFrames(Animation& animation, bool hasAlpha)
{
    // Set the path to the test image directory and filename for loading.
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "pngsuite/tp1n3p08.png";
    Mat image = imread(filename, hasAlpha ? IMREAD_UNCHANGED : IMREAD_COLOR);
    if (image.empty())
        return false;

    animation.loop_count = 0xffff; // 0xffff is the maximum value to set.
    animation.bgcolor = Scalar(50, 100, 150, 128); // different values for test purpose.

    // Add the first frame with a duration value of 500 milliseconds.
    int duration = 80;
    animation.durations.push_back(duration * 5);
    animation.frames.push_back(image.clone());
    putText(animation.frames[0], "0", Point(5, 28), FONT_HERSHEY_SIMPLEX, .5, Scalar(100, 255, 0, 255), 2);

    // Define a region of interest (ROI)
    Rect roi(2, 16, 26, 16);

    // Modify the ROI in 13 iterations to simulate slight changes in animation frames.
    for (int i = 1; i < 14; i++)
    {
        roi.x++;
        roi.width -= 2;
        RNG rng = theRNG();
        for (int x = roi.x; x < roi.x + roi.width; x++)
            for (int y = roi.y; y < roi.y + roi.height; y++)
            {
                if (hasAlpha)
                {
                    Vec4b& pixel = image.at<Vec4b>(y, x);
                    if (pixel[3] > 0)
                    {
                        if (pixel[0] > 10) pixel[0] -= (uchar)rng.uniform(2, 5);
                        if (pixel[1] > 10) pixel[1] -= (uchar)rng.uniform(2, 5);
                        if (pixel[2] > 10) pixel[2] -= (uchar)rng.uniform(2, 5);
                        pixel[3] -= (uchar)rng.uniform(2, 5);
                    }
                }
                else
                {
                    Vec3b& pixel = image.at<Vec3b>(y, x);
                    if (pixel[0] > 50) pixel[0] -= (uchar)rng.uniform(2, 5);
                    if (pixel[1] > 50) pixel[1] -= (uchar)rng.uniform(2, 5);
                    if (pixel[2] > 50) pixel[2] -= (uchar)rng.uniform(2, 5);
                }
            }

        // Update the duration and add the modified frame to the animation.
        duration += rng.uniform(2, 10);  // Increase duration with random value (to be sure different duration values saved correctly).
        animation.frames.push_back(image.clone());
        putText(animation.frames[i], format("%d", i), Point(5, 28), FONT_HERSHEY_SIMPLEX, .5, Scalar(100, 255, 0, 255), 2);
        animation.durations.push_back(duration);
    }

    // Add two identical frames with the same duration.
    animation.durations.push_back(++duration);
    animation.frames.push_back(animation.frames.back());
    animation.durations.push_back(++duration);
    animation.frames.push_back(animation.frames.back());

    return true;
}

#ifdef HAVE_WEBP

TEST(Imgcodecs_WebP, imwriteanimation_rgba)
{
    Animation s_animation, l_animation;
    EXPECT_TRUE(fillFrames(s_animation, true));

    // Create a temporary output filename for saving the animation.
    string output = cv::tempfile(".webp");

    // Write the animation to a .webp file and verify success.
    EXPECT_TRUE(imwriteanimation(output, s_animation));

    // Read the animation back and compare with the original.
    EXPECT_TRUE(imreadanimation(output, l_animation));

    // Since the last frames are identical, WebP optimizes by storing only one of them,
    // and the duration value for the last frame is handled by libwebp.
    size_t expected_frame_count = s_animation.frames.size() - 2;

    // Verify that the number of frames matches the expected count.
    EXPECT_EQ(imcount(output), expected_frame_count);
    EXPECT_EQ(l_animation.frames.size(), expected_frame_count);

    // Check that the background color and loop count match between saved and loaded animations.
    EXPECT_EQ(l_animation.bgcolor, s_animation.bgcolor); // written as BGRA order
    EXPECT_EQ(l_animation.loop_count, s_animation.loop_count);

    // Verify that the durations of frames match.
    for (size_t i = 0; i < l_animation.frames.size() - 1; i++)
        EXPECT_EQ(s_animation.durations[i], l_animation.durations[i]);

    EXPECT_TRUE(imreadanimation(output, l_animation, 5, 3));
    EXPECT_EQ(l_animation.frames.size(), expected_frame_count + 3);
    EXPECT_EQ(l_animation.frames.size(), l_animation.durations.size());
    EXPECT_EQ(0, cvtest::norm(l_animation.frames[5], l_animation.frames[14], NORM_INF));
    EXPECT_EQ(0, cvtest::norm(l_animation.frames[6], l_animation.frames[15], NORM_INF));
    EXPECT_EQ(0, cvtest::norm(l_animation.frames[7], l_animation.frames[16], NORM_INF));

    // Verify whether the imread function successfully loads the first frame
    Mat frame = imread(output, IMREAD_UNCHANGED);
    EXPECT_EQ(0, cvtest::norm(l_animation.frames[0], frame, NORM_INF));

    std::vector<uchar> buf;
    readFileBytes(output, buf);
    vector<Mat> webp_frames;

    EXPECT_TRUE(imdecodemulti(buf, IMREAD_UNCHANGED, webp_frames));
    EXPECT_EQ(expected_frame_count, webp_frames.size());

    // Test encoding and decoding the images in memory (without saving to disk).
    webp_frames.clear();
    EXPECT_TRUE(imencode(".webp", s_animation.frames, buf));
    EXPECT_TRUE(imdecodemulti(buf, IMREAD_UNCHANGED, webp_frames));
    EXPECT_EQ(expected_frame_count, webp_frames.size());

    // Clean up by removing the temporary file.
    EXPECT_EQ(0, remove(output.c_str()));
}

TEST(Imgcodecs_WebP, imwriteanimation_rgb)
{
    Animation s_animation, l_animation;
    EXPECT_TRUE(fillFrames(s_animation, false));

    // Create a temporary output filename for saving the animation.
    string output = cv::tempfile(".webp");

    // Write the animation to a .webp file and verify success.
    EXPECT_TRUE(imwriteanimation(output, s_animation));

    // Read the animation back and compare with the original.
    EXPECT_TRUE(imreadanimation(output, l_animation));

    // Since the last frames are identical, WebP optimizes by storing only one of them,
    // and the duration value for the last frame is handled by libwebp.
    size_t expected_frame_count = s_animation.frames.size() - 2;

    // Verify that the number of frames matches the expected count.
    EXPECT_EQ(imcount(output), expected_frame_count);
    EXPECT_EQ(l_animation.frames.size(), expected_frame_count);

    // Check that the background color and loop count match between saved and loaded animations.
    EXPECT_EQ(l_animation.bgcolor, s_animation.bgcolor); // written as BGRA order
    EXPECT_EQ(l_animation.loop_count, s_animation.loop_count);

    // Verify that the durations of frames match.
    for (size_t i = 0; i < l_animation.frames.size() - 1; i++)
        EXPECT_EQ(s_animation.durations[i], l_animation.durations[i]);

    EXPECT_TRUE(imreadanimation(output, l_animation, 5, 3));
    EXPECT_EQ(l_animation.frames.size(), expected_frame_count + 3);
    EXPECT_EQ(l_animation.frames.size(), l_animation.durations.size());
    EXPECT_TRUE(cvtest::norm(l_animation.frames[5], l_animation.frames[14], NORM_INF) == 0);
    EXPECT_TRUE(cvtest::norm(l_animation.frames[6], l_animation.frames[15], NORM_INF) == 0);
    EXPECT_TRUE(cvtest::norm(l_animation.frames[7], l_animation.frames[16], NORM_INF) == 0);

    // Verify whether the imread function successfully loads the first frame
    Mat frame = imread(output, IMREAD_COLOR);
    EXPECT_TRUE(cvtest::norm(l_animation.frames[0], frame, NORM_INF) == 0);

    std::vector<uchar> buf;
    readFileBytes(output, buf);

    vector<Mat> webp_frames;
    EXPECT_TRUE(imdecodemulti(buf, IMREAD_UNCHANGED, webp_frames));
    EXPECT_EQ(webp_frames.size(), expected_frame_count);

    // Clean up by removing the temporary file.
    EXPECT_EQ(0, remove(output.c_str()));
}

TEST(Imgcodecs_WebP, imwritemulti_rgb)
{
    Animation s_animation;
    EXPECT_TRUE(fillFrames(s_animation, false));

    string output = cv::tempfile(".webp");
    ASSERT_TRUE(imwrite(output, s_animation.frames));
    vector<Mat> read_frames;
    ASSERT_TRUE(imreadmulti(output, read_frames));
    EXPECT_EQ(s_animation.frames.size() - 2, read_frames.size());
    EXPECT_EQ(0, remove(output.c_str()));
}

#endif // HAVE_WEBP

#ifdef HAVE_PNG

TEST(Imgcodecs_APNG, imwriteanimation_rgba)
{
    Animation s_animation, l_animation;
    EXPECT_TRUE(fillFrames(s_animation, true));

    // Create a temporary output filename for saving the animation.
    string output = cv::tempfile(".png");

    // Write the animation to a .png file and verify success.
    EXPECT_TRUE(imwriteanimation(output, s_animation));

    // Read the animation back and compare with the original.
    EXPECT_TRUE(imreadanimation(output, l_animation));

    size_t expected_frame_count = s_animation.frames.size();

    // Verify that the number of frames matches the expected count.
    EXPECT_EQ(imcount(output), expected_frame_count);
    EXPECT_EQ(l_animation.frames.size(), expected_frame_count);

    // Check that the background color and loop count match between saved and loaded animations.
    EXPECT_EQ(l_animation.bgcolor, Scalar()/*s_animation.bgcolor*/); // TO DO not implemented yet
    EXPECT_EQ(l_animation.loop_count, s_animation.loop_count);

    for (size_t i = 0; i < l_animation.frames.size(); i++)
    {
        EXPECT_EQ(s_animation.durations[i], l_animation.durations[i]);
        EXPECT_EQ(0, cvtest::norm(s_animation.frames[i], l_animation.frames[i], NORM_INF));
    }

    EXPECT_TRUE(imreadanimation(output, l_animation, 5, 3));
    EXPECT_EQ(l_animation.frames.size(), expected_frame_count + 3);
    EXPECT_EQ(l_animation.frames.size(), l_animation.durations.size());
    EXPECT_EQ(0, cvtest::norm(l_animation.frames[5], l_animation.frames[16], NORM_INF));
    EXPECT_EQ(0, cvtest::norm(l_animation.frames[6], l_animation.frames[17], NORM_INF));
    EXPECT_EQ(0, cvtest::norm(l_animation.frames[7], l_animation.frames[18], NORM_INF));

    // Verify whether the imread function successfully loads the first frame
    Mat frame = imread(output, IMREAD_UNCHANGED);
    EXPECT_EQ(0, cvtest::norm(l_animation.frames[0], frame, NORM_INF));

    std::vector<uchar> buf;
    readFileBytes(output, buf);
    vector<Mat> apng_frames;

    EXPECT_TRUE(imdecodemulti(buf, IMREAD_UNCHANGED, apng_frames));
    EXPECT_EQ((size_t)1/*expected_frame_count*/, apng_frames.size()); // TO DO not implemented yet

    apng_frames.clear();
    // Test saving the animation frames as individual still images.
    EXPECT_TRUE(imwrite(output, s_animation.frames));

    // Read back the still images into a vector of Mats.
    EXPECT_TRUE(imreadmulti(output, apng_frames));

    // Expect all frames written as multi-page image
    expected_frame_count = 16;
    EXPECT_EQ(expected_frame_count, apng_frames.size());

    // Test encoding and decoding the images in memory (without saving to disk).
    apng_frames.clear();
    EXPECT_TRUE(imencode(".png", s_animation.frames, buf));
    EXPECT_TRUE(imdecodemulti(buf, IMREAD_UNCHANGED, apng_frames));
    EXPECT_EQ((size_t)1/*expected_frame_count*/, apng_frames.size());  // TO DO not implemented yet

    // Clean up by removing the temporary file.
    EXPECT_EQ(0, remove(output.c_str()));
}

TEST(Imgcodecs_APNG, imwriteanimation_rgb)
{
    Animation s_animation, l_animation;
    EXPECT_TRUE(fillFrames(s_animation, false));

    string output = cv::tempfile(".png");

    // Write the animation to a .png file and verify success.
    EXPECT_TRUE(imwriteanimation(output, s_animation));

    // Read the animation back and compare with the original.
    EXPECT_TRUE(imreadanimation(output, l_animation));
    EXPECT_EQ(s_animation.frames.size(), l_animation.frames.size());
    for (size_t i = 0; i < l_animation.frames.size(); i++)
    {
        EXPECT_EQ(0, cvtest::norm(s_animation.frames[i], l_animation.frames[i], NORM_INF));
    }
    EXPECT_EQ(0, remove(output.c_str()));
}

TEST(Imgcodecs_APNG, imwritemulti_rgba)
{
    Animation s_animation;
    EXPECT_TRUE(fillFrames(s_animation, true));

    string output = cv::tempfile(".png");
    EXPECT_EQ(true, imwrite(output, s_animation.frames));
    vector<Mat> read_frames;
    EXPECT_EQ(true, imreadmulti(output, read_frames, IMREAD_UNCHANGED));
    EXPECT_EQ(s_animation.frames.size(), read_frames.size());
    EXPECT_EQ(read_frames.size(), imcount(output));
    EXPECT_EQ(0, remove(output.c_str()));
}

TEST(Imgcodecs_APNG, imwritemulti_rgb)
{
    Animation s_animation;
    EXPECT_TRUE(fillFrames(s_animation, false));

    string output = cv::tempfile(".png");
    ASSERT_TRUE(imwrite(output, s_animation.frames));
    vector<Mat> read_frames;
    ASSERT_TRUE(imreadmulti(output, read_frames));
    EXPECT_EQ(s_animation.frames.size(), read_frames.size());
    EXPECT_EQ(0, remove(output.c_str()));

    for (size_t i = 0; i < read_frames.size(); i++)
    {
        EXPECT_EQ(0, cvtest::norm(s_animation.frames[i], read_frames[i], NORM_INF));
    }
}

TEST(Imgcodecs_APNG, imwritemulti_gray)
{
    Animation s_animation;
    EXPECT_TRUE(fillFrames(s_animation, false));

    for (size_t i = 0; i < s_animation.frames.size(); i++)
    {
        cvtColor(s_animation.frames[i], s_animation.frames[i], COLOR_BGR2GRAY);
    }

    string output = cv::tempfile(".png");
    EXPECT_TRUE(imwrite(output, s_animation.frames));
    vector<Mat> read_frames;
    EXPECT_TRUE(imreadmulti(output, read_frames));
    EXPECT_EQ(1, read_frames[0].channels());
    read_frames.clear();
    EXPECT_TRUE(imreadmulti(output, read_frames, IMREAD_UNCHANGED));
    EXPECT_EQ(1, read_frames[0].channels());
    read_frames.clear();
    EXPECT_TRUE(imreadmulti(output, read_frames, IMREAD_COLOR));
    EXPECT_EQ(3, read_frames[0].channels());
    read_frames.clear();
    EXPECT_TRUE(imreadmulti(output, read_frames, IMREAD_GRAYSCALE));
    EXPECT_EQ(0, remove(output.c_str()));

    for (size_t i = 0; i < s_animation.frames.size(); i++)
    {
        EXPECT_EQ(0, cvtest::norm(s_animation.frames[i], read_frames[i], NORM_INF));
    }
}

#endif // HAVE_PNG

}} // namespace
