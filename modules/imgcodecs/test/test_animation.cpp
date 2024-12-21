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

#ifdef HAVE_WEBP

TEST(Imgcodecs_Animation, webp_load_save_rgba)
{
    RNG rng = theRNG();

    // Set the path to the test image directory and filename for loading.
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "pngsuite/tp1n3p08.png";

    // Create an Animation object using the default constructor.
    // This initializes the loop count to 0 (infinite looping), background color to 0 (transparent)
    Animation l_animation;

    // Create an Animation object with custom parameters.
    int loop_count = 0xffff; // 0xffff is the maximum value to set.
    Scalar bgcolor(125, 126, 127, 128); // different values for test purpose.
    Animation s_animation(loop_count, bgcolor);

    // Load the image file with alpha channel (IMREAD_UNCHANGED).
    Mat image = imread(filename, IMREAD_UNCHANGED);
    ASSERT_FALSE(image.empty()) << "Failed to load image: " << filename;

    // Add the first frame with a duration value of 500 milliseconds.
    int duration = 100;
    s_animation.durations.push_back(duration * 5);
    s_animation.frames.push_back(image.clone());  // Store the first frame.
    putText(s_animation.frames[0], "0", Point(5, 28), FONT_HERSHEY_SIMPLEX, .5, Scalar(100, 255, 0, 255), 2);

    // Define a region of interest (ROI) in the loaded image for manipulation.
    Mat roi = image(Rect(0, 16, 32, 16));  // Select a subregion of the image.

    // Modify the ROI in 13 iterations to simulate slight changes in animation frames.
    for (int i = 1; i < 14; i++)
    {
        for (int x = 0; x < roi.rows; x++)
            for (int y = 0; y < roi.cols; y++)
            {
                // Apply random changes to pixel values to create animation variations.
                Vec4b& pixel = roi.at<Vec4b>(x, y);
                if (pixel[3] > 0)
                {
                    if (pixel[0] > 10) pixel[0] -= (uchar)rng.uniform(3, 10);  // Reduce blue channel.
                    if (pixel[1] > 10) pixel[1] -= (uchar)rng.uniform(3, 10);  // Reduce green channel.
                    if (pixel[2] > 10) pixel[2] -= (uchar)rng.uniform(3, 10);  // Reduce red channel.
                    pixel[3] -= (uchar)rng.uniform(2, 5);  // Reduce alpha channel.
                }
            }

        // Update the duration and add the modified frame to the animation.
        duration += rng.uniform(2, 10);  // Increase duration with random value (to be sure different duration values saved correctly).
        s_animation.frames.push_back(image.clone());
        putText(s_animation.frames[i], format("%d", i), Point(5, 28), FONT_HERSHEY_SIMPLEX, .5, Scalar(100, 255, 0, 255), 2);
        s_animation.durations.push_back(duration);
    }

    // Add two identical frames with the same duration.
    s_animation.durations.push_back(duration);
    s_animation.frames.push_back(s_animation.frames[13].clone());
    s_animation.durations.push_back(duration);
    s_animation.frames.push_back(s_animation.frames[13].clone());

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

    webp_frames.clear();
    // Test saving the animation frames as individual still images.
    EXPECT_TRUE(imwrite(output, s_animation.frames));

    // Read back the still images into a vector of Mats.
    EXPECT_TRUE(imreadmulti(output, webp_frames));

    // Expect all frames written as multi-page image
    expected_frame_count = 14;
    EXPECT_EQ(expected_frame_count, webp_frames.size());

    // Test encoding and decoding the images in memory (without saving to disk).
    webp_frames.clear();
    EXPECT_TRUE(imencode(".webp", s_animation.frames, buf));
    EXPECT_TRUE(imdecodemulti(buf, IMREAD_UNCHANGED, webp_frames));
    EXPECT_EQ(expected_frame_count, webp_frames.size());

    // Clean up by removing the temporary file.
    EXPECT_EQ(0, remove(output.c_str()));
}

TEST(Imgcodecs_Animation, webp_load_save_rgb)
{
    RNG rng = theRNG();

    // Set the path to the test image directory and filename for loading.
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "pngsuite/tp1n3p08.png";

    // Create an Animation object using the default constructor.
    // This initializes the loop count to 0 (infinite looping), background color to 0 (transparent)
    Animation l_animation;

    // Create an Animation object with custom parameters.
    int loop_count = 0xffff; // 0xffff is the maximum value to set.
    Scalar bgcolor(125, 126, 127, 128); // different values for test purpose.
    Animation s_animation(loop_count, bgcolor);

    // Load the image file without alpha channel
    Mat image = imread(filename);
    ASSERT_FALSE(image.empty()) << "Failed to load image: " << filename;

    // Add the first frame with a duration value of 500 milliseconds.
    int duration = 100;
    s_animation.durations.push_back(duration * 5);
    s_animation.frames.push_back(image.clone());  // Store the first frame.
    putText(s_animation.frames[0], "0", Point(5, 28), FONT_HERSHEY_SIMPLEX, .5, Scalar(100, 255, 0, 255), 2);

    // Define a region of interest (ROI) in the loaded image for manipulation.
    Mat roi = image(Rect(0, 16, 32, 16));  // Select a subregion of the image.

    // Modify the ROI in 13 iterations to simulate slight changes in animation frames.
    for (int i = 1; i < 14; i++)
    {
        for (int x = 0; x < roi.rows; x++)
            for (int y = 0; y < roi.cols; y++)
            {
                // Apply random changes to pixel values to create animation variations.
                Vec3b& pixel = roi.at<Vec3b>(x, y);
                if (pixel[0] > 50) pixel[0] -= (uchar)rng.uniform(3, 10);  // Reduce blue channel.
                if (pixel[1] > 50) pixel[1] -= (uchar)rng.uniform(3, 10);  // Reduce green channel.
                if (pixel[2] > 50) pixel[2] -= (uchar)rng.uniform(3, 10);  // Reduce red channel.
            }

        // Update the duration and add the modified frame to the animation.
        duration += rng.uniform(2, 10);  // Increase duration with random value (to be sure different duration values saved correctly).
        s_animation.frames.push_back(image.clone());
        putText(s_animation.frames[i], format("%d", i), Point(5, 28), FONT_HERSHEY_SIMPLEX, .5, Scalar(100, 255, 0, 255), 2);
        s_animation.durations.push_back(duration);
    }

    // Add two identical frames with the same duration.
    s_animation.durations.push_back(duration);
    s_animation.frames.push_back(s_animation.frames[13].clone());
    s_animation.durations.push_back(duration);
    s_animation.frames.push_back(s_animation.frames[13].clone());

    // Create a temporary output filename for saving the animation.
    string output = cv::tempfile(".webp");

    // Write the animation to a .webp file and verify success.
    EXPECT_EQ(true, imwriteanimation(output, s_animation));

    // Read the animation back and compare with the original.
    EXPECT_EQ(true, imreadanimation(output, l_animation));

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

    EXPECT_EQ(true, imreadanimation(output, l_animation, 5, 3));
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

#endif // HAVE_WEBP

#ifdef HAVE_PNG

TEST(Imgcodecs_Animation, apng_load_save_rgba)
{
    RNG rng = theRNG();

    // Set the path to the test image directory and filename for loading.
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "pngsuite/tp1n3p08.png";

    // Create an Animation object using the default constructor.
    // This initializes the loop count to 0 (infinite looping), background color to 0 (transparent)
    Animation l_animation;

    // Create an Animation object with custom parameters.
    int loop_count = 0xffff; // 0xffff is the maximum value to set.
    Scalar bgcolor(125, 126, 127, 128); // different values for test purpose.
    Animation s_animation(loop_count, bgcolor);

    // Load the image file with alpha channel (IMREAD_UNCHANGED).
    Mat image = imread(filename, IMREAD_UNCHANGED);
    ASSERT_FALSE(image.empty()) << "Failed to load image: " << filename;

    // Add the first frame with a duration value of 500 milliseconds.
    int duration = 100;
    s_animation.durations.push_back(duration * 5);
    s_animation.frames.push_back(image.clone());  // Store the first frame.
    putText(s_animation.frames[0], "0", Point(5, 28), FONT_HERSHEY_SIMPLEX, .5, Scalar(100, 255, 0, 255), 2);

    // Define a region of interest (ROI) in the loaded image for manipulation.
    Mat roi = image(Rect(0, 16, 32, 16));  // Select a subregion of the image.

    // Modify the ROI in 13 iterations to simulate slight changes in animation frames.
    for (int i = 1; i < 14; i++)
    {
        for (int x = 0; x < roi.rows; x++)
            for (int y = 0; y < roi.cols; y++)
            {
                // Apply random changes to pixel values to create animation variations.
                Vec4b& pixel = roi.at<Vec4b>(x, y);
                if (pixel[3] > 0)
                {
                    if (pixel[0] > 10) pixel[0] -= (uchar)rng.uniform(3, 10);  // Reduce blue channel.
                    if (pixel[1] > 10) pixel[1] -= (uchar)rng.uniform(3, 10);  // Reduce green channel.
                    if (pixel[2] > 10) pixel[2] -= (uchar)rng.uniform(3, 10);  // Reduce red channel.
                    pixel[3] -= (uchar)rng.uniform(2, 5);  // Reduce alpha channel.
                }
            }

        // Update the duration and add the modified frame to the animation.
        duration += rng.uniform(2, 10);  // Increase duration with random value (to be sure different duration values saved correctly).
        s_animation.frames.push_back(image.clone());
        putText(s_animation.frames[i], format("%d", i), Point(5, 28), FONT_HERSHEY_SIMPLEX, .5, Scalar(100, 255, 0, 255), 2);
        s_animation.durations.push_back(duration);
    }

    // Add two identical frames with the same duration.
    s_animation.durations.push_back(duration);
    s_animation.frames.push_back(s_animation.frames[13].clone());
    s_animation.durations.push_back(duration);
    s_animation.frames.push_back(s_animation.frames[13].clone());

    // Create a temporary output filename for saving the animation.
    string output = cv::tempfile(".png");

    // Write the animation to a .webp file and verify success.
    EXPECT_TRUE(imwriteanimation(output, s_animation));

    // Read the animation back and compare with the original.
    EXPECT_TRUE(imreadanimation(output, l_animation));

    size_t expected_frame_count = s_animation.frames.size();

    // Verify that the number of frames matches the expected count.
    EXPECT_EQ(imcount(output), expected_frame_count);
    EXPECT_EQ(l_animation.frames.size(), expected_frame_count);

    // Check that the background color and loop count match between saved and loaded animations.
    //EXPECT_EQ(l_animation.bgcolor, s_animation.bgcolor); // written as BGRA order
    EXPECT_EQ(l_animation.loop_count, s_animation.loop_count);

    // Verify that the durations of frames match.
    for (size_t i = 0; i < l_animation.frames.size() - 1; i++)
        EXPECT_EQ(s_animation.durations[i], l_animation.durations[i]);

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
    EXPECT_EQ(expected_frame_count, apng_frames.size());

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
    EXPECT_EQ(expected_frame_count, apng_frames.size());

    // Clean up by removing the temporary file.
    EXPECT_EQ(0, remove(output.c_str()));
}

TEST(Imgcodecs_Animation, apng_load_save_multiframes_rgba)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "pngsuite/tp1n3p08.png";
    vector<Mat> png_frames;
    RNG rng = theRNG();

    Mat image = imread(filename, IMREAD_UNCHANGED);
    png_frames.push_back(image.clone());
    Mat roi = image(Rect(0, 16, 32, 16));

    // Modify the ROI in 13 iterations to simulate slight changes in animation frames.
    for (int i = 1; i < 14; i++)
    {
        for (int x = 0; x < roi.rows; x++)
            for (int y = 0; y < roi.cols; y++)
            {
                // Apply random changes to pixel values to create animation variations.
                Vec4b& pixel = roi.at<Vec4b>(x, y);
                if (pixel[3] > 0)
                {
                    if (pixel[0] > 10) pixel[0] -= (uchar)rng.uniform(3, 10);  // Reduce blue channel.
                    if (pixel[1] > 10) pixel[1] -= (uchar)rng.uniform(3, 10);  // Reduce green channel.
                    if (pixel[2] > 10) pixel[2] -= (uchar)rng.uniform(3, 10);  // Reduce red channel.
                    pixel[3] -= (uchar)rng.uniform(2, 5);  // Reduce alpha channel.
                }
            }

        png_frames.push_back(image.clone());
        putText(png_frames[i], format("%d", i), Point(5, 28), FONT_HERSHEY_SIMPLEX, .5, Scalar(100, 255, 0, 255), 2);
    }

    string output = cv::tempfile(".png");
    EXPECT_EQ(true, imwrite(output, png_frames));
    vector<Mat> read_frames;
    EXPECT_EQ(true, imreadmulti(output, read_frames, IMREAD_UNCHANGED));
    EXPECT_EQ(png_frames.size(), read_frames.size());
    EXPECT_EQ(read_frames.size(), imcount(output));
    EXPECT_EQ(0, remove(output.c_str()));
    std::vector<uchar> buf;
    EXPECT_EQ(true, imencode(".png", png_frames, buf));
    EXPECT_EQ(true, imdecodemulti(buf, IMREAD_COLOR_RGB, read_frames));
}

TEST(Imgcodecs_Animation, apng_load_save_multiframes_rgb)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "pngsuite/tp1n3p08.png";
    vector<Mat> png_frames;
    RNG rng = theRNG();

    Mat image = imread(filename);
    png_frames.push_back(image.clone());
    Mat roi = image(Rect(0, 16, 32, 16));

    // Modify the ROI in 13 iterations to simulate slight changes in animation frames.
    for (int i = 1; i < 14; i++)
    {
        for (int x = 0; x < roi.rows; x++)
            for (int y = 0; y < roi.cols; y++)
            {
                // Apply random changes to pixel values to create animation variations.
                Vec3b& pixel = roi.at<Vec3b>(x, y);
                if (pixel[0] > 10) pixel[0] -= (uchar)rng.uniform(3, 10);  // Reduce blue channel.
                if (pixel[1] > 10) pixel[1] -= (uchar)rng.uniform(3, 10);  // Reduce green channel.
                if (pixel[2] > 10) pixel[2] -= (uchar)rng.uniform(3, 10);  // Reduce red channel.
            }

        png_frames.push_back(image.clone());
        putText(png_frames[i], format("%d", i), Point(5, 28), FONT_HERSHEY_SIMPLEX, .5, Scalar(100, 255, 0, 255), 2);
    }

    string output = cv::tempfile(".png");
    ASSERT_TRUE(imwrite(output, png_frames));
    vector<Mat> read_frames;
    ASSERT_TRUE(imreadmulti(output, read_frames));
    EXPECT_EQ(png_frames.size(), read_frames.size());
    EXPECT_EQ(read_frames.size(), imcount(output));
    EXPECT_EQ(0, remove(output.c_str()));

    for (size_t i = 0; i < png_frames.size(); i++)
    {
        EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), png_frames[i], read_frames[i]);
    }
}

TEST(Imgcodecs_Animation, apng_load_save_multiframes_gray)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "pngsuite/tp1n3p08.png";
    vector<Mat> png_frames;

    Mat image = imread(filename, IMREAD_GRAYSCALE);
    png_frames.push_back(image.clone());
    Mat roi = image(Rect(0, 16, 32, 16));

    for (size_t i = 0; i < 15; i++)
    {
        roi = roi - Scalar(10);
        png_frames.push_back(image.clone());
    }

    string output = cv::tempfile(".png");
    EXPECT_EQ(true, imwrite(output, png_frames));
    vector<Mat> read_frames;
    EXPECT_EQ(true, imreadmulti(output, read_frames));
    EXPECT_EQ(1, read_frames[0].channels());
    read_frames.clear();
    EXPECT_EQ(true, imreadmulti(output, read_frames, IMREAD_UNCHANGED));
    EXPECT_EQ(1, read_frames[0].channels());
    read_frames.clear();
    EXPECT_EQ(true, imreadmulti(output, read_frames, IMREAD_COLOR));
    EXPECT_EQ(3, read_frames[0].channels());
    read_frames.clear();
    EXPECT_EQ(true, imreadmulti(output, read_frames, IMREAD_GRAYSCALE));
    EXPECT_EQ(png_frames.size(), read_frames.size());
    EXPECT_EQ(read_frames.size(), imcount(output));
    EXPECT_EQ(0, remove(output.c_str()));

    for (size_t i = 0; i < png_frames.size(); i++)
    {
        EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), png_frames[i], read_frames[i]);
    }
}

#endif // HAVE_PNG

}} // namespace
