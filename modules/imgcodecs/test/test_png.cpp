// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"

namespace opencv_test { namespace {

#if defined(HAVE_PNG) || defined(HAVE_SPNG)

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

TEST(Imgcodecs_Png, write_big)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/read.png";
    const string dst_file = cv::tempfile(".png");
    Mat img;
    ASSERT_NO_THROW(img = imread(filename));
    ASSERT_FALSE(img.empty());
    EXPECT_EQ(13043, img.cols);
    EXPECT_EQ(13917, img.rows);
    ASSERT_NO_THROW(imwrite(dst_file, img));
    EXPECT_EQ(0, remove(dst_file.c_str()));
}

TEST(Imgcodecs_Png, encode)
{
    vector<uchar> buff;
    Mat img_gt = Mat::zeros(1000, 1000, CV_8U);
    vector<int> param;
    param.push_back(IMWRITE_PNG_COMPRESSION);
    param.push_back(3); //default(3) 0-9.
    EXPECT_NO_THROW(imencode(".png", img_gt, buff, param));
    Mat img;
    EXPECT_NO_THROW(img = imdecode(buff, IMREAD_ANYDEPTH)); // hang
    EXPECT_FALSE(img.empty());
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img, img_gt);
}

TEST(Imgcodecs_Png, regression_ImreadVSCvtColor)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string imgName = root + "../cv/shared/lena.png";
    Mat original_image = imread(imgName);
    Mat gray_by_codec = imread(imgName, IMREAD_GRAYSCALE);
    Mat gray_by_cvt;
    cvtColor(original_image, gray_by_cvt, COLOR_BGR2GRAY);

    Mat diff;
    absdiff(gray_by_codec, gray_by_cvt, diff);
    EXPECT_LT(cvtest::mean(diff)[0], 1.);
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(10, 0), gray_by_codec, gray_by_cvt);
}

// Test OpenCV issue 3075 is solved
TEST(Imgcodecs_Png, read_color_palette_with_alpha)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    Mat img;

    // First Test : Read PNG with alpha, imread flag -1
    img = imread(root + "readwrite/color_palette_alpha.png", IMREAD_UNCHANGED);
    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(img.channels() == 4);

    // pixel is red in BGRA
    EXPECT_EQ(img.at<Vec4b>(0, 0), Vec4b(0, 0, 255, 255));
    EXPECT_EQ(img.at<Vec4b>(0, 1), Vec4b(0, 0, 255, 255));

    // Second Test : Read PNG without alpha, imread flag -1
    img = imread(root + "readwrite/color_palette_no_alpha.png", IMREAD_UNCHANGED);
    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(img.channels() == 3);

    // pixel is red in BGR
    EXPECT_EQ(img.at<Vec3b>(0, 0), Vec3b(0, 0, 255));
    EXPECT_EQ(img.at<Vec3b>(0, 1), Vec3b(0, 0, 255));

    // Third Test : Read PNG with alpha, imread flag 1
    img = imread(root + "readwrite/color_palette_alpha.png", IMREAD_COLOR);
    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(img.channels() == 3);

    // pixel is red in BGR
    EXPECT_EQ(img.at<Vec3b>(0, 0), Vec3b(0, 0, 255));
    EXPECT_EQ(img.at<Vec3b>(0, 1), Vec3b(0, 0, 255));

    img = imread(root + "readwrite/color_palette_alpha.png", IMREAD_COLOR_RGB);
    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(img.channels() == 3);

    // pixel is red in RGB
    EXPECT_EQ(img.at<Vec3b>(0, 0), Vec3b(255, 0, 0));
    EXPECT_EQ(img.at<Vec3b>(0, 1), Vec3b(255, 0, 0));

    // Fourth Test : Read PNG without alpha, imread flag 1
    img = imread(root + "readwrite/color_palette_no_alpha.png", IMREAD_COLOR);
    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(img.channels() == 3);

    // pixel is red in BGR
    EXPECT_EQ(img.at<Vec3b>(0, 0), Vec3b(0, 0, 255));
    EXPECT_EQ(img.at<Vec3b>(0, 1), Vec3b(0, 0, 255));

    img = imread(root + "readwrite/color_palette_no_alpha.png", IMREAD_COLOR_RGB);
    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(img.channels() == 3);

    // pixel is red in RGB
    EXPECT_EQ(img.at<Vec3b>(0, 0), Vec3b(255, 0, 0));
    EXPECT_EQ(img.at<Vec3b>(0, 1), Vec3b(255, 0, 0));
}

typedef testing::TestWithParam<string> Imgcodecs_Png_PngSuite;

TEST_P(Imgcodecs_Png_PngSuite, decode)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "pngsuite/" + GetParam() + ".png";
    const string xml_filename = root + "pngsuite/" + GetParam() + ".xml";
    FileStorage fs(xml_filename, FileStorage::READ);
    EXPECT_TRUE(fs.isOpened());

    Mat src = imread(filename, IMREAD_UNCHANGED);
    Mat gt;
    fs.getFirstTopLevelNode() >> gt;

    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), src, gt);
}

const string pngsuite_files[] =
{
    "basi0g01",
    "basi0g02",
    "basi0g04",
    "basi0g08",
    "basi0g16",
    "basi2c08",
    "basi2c16",
    "basi3p01",
    "basi3p02",
    "basi3p04",
    "basi3p08",
    "basi4a08",
    "basi4a16",
    "basi6a08",
    "basi6a16",
    "basn0g01",
    "basn0g02",
    "basn0g04",
    "basn0g08",
    "basn0g16",
    "basn2c08",
    "basn2c16",
    "basn3p01",
    "basn3p02",
    "basn3p04",
    "basn3p08",
    "basn4a08",
    "basn4a16",
    "basn6a08",
    "basn6a16",
    "bgai4a08",
    "bgai4a16",
    "bgan6a08",
    "bgan6a16",
    "bgbn4a08",
    "bggn4a16",
    "bgwn6a08",
    "bgyn6a16",
    "ccwn2c08",
    "ccwn3p08",
    "cdfn2c08",
    "cdhn2c08",
    "cdsn2c08",
    "cdun2c08",
    "ch1n3p04",
    "ch2n3p08",
    "cm0n0g04",
    "cm7n0g04",
    "cm9n0g04",
    "cs3n2c16",
    "cs3n3p08",
    "cs5n2c08",
    "cs5n3p08",
    "cs8n2c08",
    "cs8n3p08",
    "ct0n0g04",
    "ct1n0g04",
    "cten0g04",
    "ctfn0g04",
    "ctgn0g04",
    "cthn0g04",
    "ctjn0g04",
    "ctzn0g04",
    "exif2c08",
    "f00n0g08",
    "f00n2c08",
    "f01n0g08",
    "f01n2c08",
    "f02n0g08",
    "f02n2c08",
    "f03n0g08",
    "f03n2c08",
    "f04n0g08",
    "f04n2c08",
    "f99n0g04",
    "g03n0g16",
    "g03n2c08",
    "g03n3p04",
    "g04n0g16",
    "g04n2c08",
    "g04n3p04",
    "g05n0g16",
    "g05n2c08",
    "g05n3p04",
    "g07n0g16",
    "g07n2c08",
    "g07n3p04",
    "g10n0g16",
    "g10n2c08",
    "g10n3p04",
    "g25n0g16",
    "g25n2c08",
    "g25n3p04",
    "oi1n0g16",
    "oi1n2c16",
    "oi2n0g16",
    "oi2n2c16",
    "oi4n0g16",
    "oi4n2c16",
    "oi9n0g16",
    "oi9n2c16",
    "pp0n2c16",
    "pp0n6a08",
    "ps1n0g08",
    "ps1n2c16",
    "ps2n0g08",
    "ps2n2c16",
    "s01i3p01",
    "s01n3p01",
    "s02i3p01",
    "s02n3p01",
    "s03i3p01",
    "s03n3p01",
    "s04i3p01",
    "s04n3p01",
    "s05i3p02",
    "s05n3p02",
    "s06i3p02",
    "s06n3p02",
    "s07i3p02",
    "s07n3p02",
    "s08i3p02",
    "s08n3p02",
    "s09i3p02",
    "s09n3p02",
    "s32i3p04",
    "s32n3p04",
    "s33i3p04",
    "s33n3p04",
    "s34i3p04",
    "s34n3p04",
    "s35i3p04",
    "s35n3p04",
    "s36i3p04",
    "s36n3p04",
    "s37i3p04",
    "s37n3p04",
    "s38i3p04",
    "s38n3p04",
    "s39i3p04",
    "s39n3p04",
    "s40i3p04",
    "s40n3p04",
    "tbbn0g04",
    "tbbn2c16",
    "tbbn3p08",
    "tbgn2c16",
    "tbgn3p08",
    "tbrn2c08",
    "tbwn0g16",
    "tbwn3p08",
    "tbyn3p08",
    "tm3n3p02",
    "tp0n0g08",
    "tp0n2c08",
    "tp0n3p08",
    "tp1n3p08",
    "z00n2c08",
    "z03n2c08",
    "z06n2c08",
    "z09n2c08",
};

INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgcodecs_Png_PngSuite,
                        testing::ValuesIn(pngsuite_files));

typedef testing::TestWithParam<string> Imgcodecs_Png_PngSuite_Corrupted;

TEST_P(Imgcodecs_Png_PngSuite_Corrupted, decode)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "pngsuite/" + GetParam() + ".png";

    Mat src = imread(filename, IMREAD_UNCHANGED);

    // Corrupted files should not be read
    EXPECT_TRUE(src.empty());
}

const string pngsuite_files_corrupted[] = {
    "xc1n0g08",
    "xc9n2c08",
    "xcrn0g04",
    "xcsn0g01",
    "xd0n2c08",
    "xd3n2c08",
    "xd9n2c08",
    "xdtn0g01",
    "xhdn0g08",
    "xlfn0g04",
    "xs1n0g01",
    "xs2n0g01",
    "xs4n0g01",
    "xs7n0g01",
};

INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgcodecs_Png_PngSuite_Corrupted,
                        testing::ValuesIn(pngsuite_files_corrupted));

TEST(Imgcodecs_APNG, load_save_animation_rgba)
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
    vector<Mat> webp_frames;

    EXPECT_TRUE(imdecodemulti(buf, IMREAD_UNCHANGED, webp_frames));
    EXPECT_EQ(expected_frame_count, webp_frames.size());

    webp_frames.clear();
    // Test saving the animation frames as individual still images.
    EXPECT_TRUE(imwrite(output, s_animation.frames));

    // Read back the still images into a vector of Mats.
    EXPECT_TRUE(imreadmulti(output, webp_frames));

    // Expect all frames written as multi-page image
    expected_frame_count = 16;
    EXPECT_EQ(expected_frame_count, webp_frames.size());

    // Test encoding and decoding the images in memory (without saving to disk).
    webp_frames.clear();
    EXPECT_TRUE(imencode(".png", s_animation.frames, buf));
    EXPECT_TRUE(imdecodemulti(buf, IMREAD_UNCHANGED, webp_frames));
    EXPECT_EQ(expected_frame_count, webp_frames.size());

    // Clean up by removing the temporary file.
    EXPECT_EQ(0, remove(output.c_str()));
}

TEST(Imgcodecs_Png, load_save_multiframes_rgba)
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

TEST(Imgcodecs_Png, load_save_multiframes_rgb)
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

TEST(Imgcodecs_Png, load_save_multiframes_gray)
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
