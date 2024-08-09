// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#include <cstdint>
#include <fstream>

#include "test_precomp.hpp"

#ifdef HAVE_WEBP

namespace opencv_test {
namespace {

class Imgcodecs_WebP_RoundTripSuite
    : public testing::TestWithParam<std::tuple<int, int, int, ImreadModes>> {
 protected:
  static cv::Mat modifyImage(const cv::Mat& img_original, int channels,
                             int bit_depth) {
    cv::Mat img;
    if (channels == 1) {
      cv::cvtColor(img_original, img, cv::COLOR_BGR2GRAY);
    } else if (channels == 4) {
      std::vector<cv::Mat> imgs;
      cv::split(img_original, imgs);
      imgs.push_back(cv::Mat(imgs[0]));
      imgs[imgs.size() - 1] = cv::Scalar::all(128);
      cv::merge(imgs, img);
    } else {
      img = img_original.clone();
    }

    cv::Mat img_final = img;
    // Convert image to CV_16U for some bit depths.
    if (bit_depth > 8) img.convertTo(img_final, CV_16U, 1 << (bit_depth - 8));

    return img_final;
  }

  void SetUp() {
    bit_depth_ = std::get<0>(GetParam());
    channels_ = std::get<1>(GetParam());
    quality_ = std::get<2>(GetParam());
    imread_mode_ = std::get<3>(GetParam());
    encoding_params_ = {cv::IMWRITE_WEBP_QUALITY, quality_};
  }

  bool IsBitDepthValid() const {
    return (bit_depth_ == 8 || bit_depth_ == 10 || bit_depth_ == 12);
  }

  // Makes sure images are close enough after encode/decode roundtrip.
  void ValidateRead(const cv::Mat& img_original, const cv::Mat& img) const {
    EXPECT_EQ(img_original.size(), img.size());
    if (imread_mode_ == IMREAD_UNCHANGED) {
      ASSERT_EQ(img_original.type(), img.type());
      // Lossless.
      if (quality_ == 100) {
        EXPECT_EQ(0, cvtest::norm(img, img_original, NORM_INF));
      } else {
        const double norm = cvtest::norm(img, img_original, NORM_L2) /
                           img.channels() / img.cols / img.rows /
                           (1 << (bit_depth_ - 8));
        if (quality_ == 50) {
          EXPECT_LE(norm, 10);
        } else if (quality_ == 0) {
          EXPECT_LE(norm, 13);
        } else {
          EXPECT_FALSE(true);
        }
      }
    }
  }

 public:
  int bit_depth_;
  int channels_;
  int quality_;
  int imread_mode_;
  std::vector<int> encoding_params_;
};

////////////////////////////////////////////////////////////////////////////////

class Imgcodecs_WebP_Image_RoundTripSuite
    : public Imgcodecs_WebP_RoundTripSuite {
 public:
  const cv::Mat& get_img_original() {
    const Key key = {channels_, (bit_depth_ < 8) ? 8 : bit_depth_};
    return imgs_[key];
  }

  // Prepare the original image modified for different number of channels and
  // bit depth.
  static void SetUpTestCase() {
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "../cv/shared/lena.png";
    const cv::Mat img_original = cv::imread(filename);
    cv::Mat img_resized;
    cv::resize(img_original, img_resized, cv::Size(kWidth, kHeight), 0, 0);
    for (int channels : {1, 3, 4}) {
      for (int bit_depth : {8, 10, 12}) {
        const Key key{channels, bit_depth};
        imgs_[key] = modifyImage(img_resized, channels, bit_depth);
      }
    }
  }

  static const int kWidth;
  static const int kHeight;

 private:
  typedef std::tuple<int, int> Key;
  static std::map<Key, cv::Mat> imgs_;
};
std::map<std::tuple<int, int>, cv::Mat>
    Imgcodecs_WebP_Image_RoundTripSuite::imgs_;
const int Imgcodecs_WebP_Image_RoundTripSuite::kWidth = 51;
const int Imgcodecs_WebP_Image_RoundTripSuite::kHeight = 31;

class Imgcodecs_WebP_Image_WriteReadSuite
    : public Imgcodecs_WebP_Image_RoundTripSuite {};

TEST_P(Imgcodecs_WebP_Image_WriteReadSuite, imwrite_imread) {
  const cv::Mat& img_original = get_img_original();
  ASSERT_FALSE(img_original.empty());

  // Encode.
  const string output = cv::tempfile(".webp");
  if (!IsBitDepthValid()) {
    EXPECT_NO_FATAL_FAILURE(
        cv::imwrite(output, img_original, encoding_params_));
    EXPECT_EQ(0, remove(output.c_str()));
    return;
  }
  EXPECT_NO_THROW(cv::imwrite(output, img_original, encoding_params_));

  // Read from file.
  const cv::Mat img = cv::imread(output, imread_mode_);

  ValidateRead(img_original, img);

  EXPECT_EQ(0, remove(output.c_str()));
}

INSTANTIATE_TEST_CASE_P(
    Imgcodecs_WebP, Imgcodecs_WebP_Image_WriteReadSuite,
    ::testing::Combine(::testing::ValuesIn({6, 8, 10, 12}),
                       ::testing::ValuesIn({1, 3, 4}),
                       ::testing::ValuesIn({0, 50, 100}),
                       ::testing::ValuesIn({IMREAD_UNCHANGED, IMREAD_GRAYSCALE,
                                            IMREAD_COLOR, IMREAD_COLOR_RGB})));

class Imgcodecs_WebP_Image_EncodeDecodeSuite
    : public Imgcodecs_WebP_Image_RoundTripSuite {};

TEST_P(Imgcodecs_WebP_Image_EncodeDecodeSuite, imencode_imdecode) {
  const cv::Mat& img_original = get_img_original();
  ASSERT_FALSE(img_original.empty());

  // Encode.
  std::vector<unsigned char> buf;
  if (!IsBitDepthValid()) {
    EXPECT_THROW(cv::imencode(".webp", img_original, buf, encoding_params_),
                 cv::Exception);
    return;
  }
  bool result = true;
  EXPECT_NO_THROW(
      result = cv::imencode(".webp", img_original, buf, encoding_params_););
  EXPECT_TRUE(result);

  // Read back.
  const cv::Mat img = cv::imdecode(buf, imread_mode_);

  ValidateRead(img_original, img);
}

INSTANTIATE_TEST_CASE_P(
    Imgcodecs_WebP, Imgcodecs_WebP_Image_EncodeDecodeSuite,
    ::testing::Combine(::testing::ValuesIn({6, 8, 10, 12}),
                       ::testing::ValuesIn({1, 3, 4}),
                       ::testing::ValuesIn({0, 50, 100}),
                       ::testing::ValuesIn({IMREAD_UNCHANGED, IMREAD_GRAYSCALE,
                                            IMREAD_COLOR, IMREAD_COLOR_RGB})));

////////////////////////////////////////////////////////////////////////////////

TEST(Imgcodecs_WebP, encode_decode_lossless_webp)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    string filename = root + "../cv/shared/lena.png";
    cv::Mat img = cv::imread(filename);
    ASSERT_FALSE(img.empty());

    string output = cv::tempfile(".webp");
    EXPECT_NO_THROW(cv::imwrite(output, img)); // lossless

    cv::Mat img_webp = cv::imread(output);

    std::vector<unsigned char> buf;

    FILE * wfile = NULL;

    wfile = fopen(output.c_str(), "rb");
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

        if (data_size != wfile_size)
        {
            EXPECT_TRUE(false);
        }
    }

    EXPECT_EQ(0, remove(output.c_str()));

    cv::Mat decode = cv::imdecode(buf, IMREAD_COLOR);
    ASSERT_FALSE(decode.empty());
    EXPECT_TRUE(cvtest::norm(decode, img_webp, NORM_INF) == 0);

    cv::Mat decode_rgb = cv::imdecode(buf, IMREAD_COLOR_RGB);
    ASSERT_FALSE(decode_rgb.empty());

    cvtColor(decode_rgb, decode_rgb, COLOR_RGB2BGR);
    EXPECT_TRUE(cvtest::norm(decode_rgb, img_webp, NORM_INF) == 0);

    ASSERT_FALSE(img_webp.empty());

    EXPECT_TRUE(cvtest::norm(img, img_webp, NORM_INF) == 0);
}

TEST(Imgcodecs_WebP, encode_decode_lossy_webp)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    std::string input = root + "../cv/shared/lena.png";
    cv::Mat img = cv::imread(input);
    ASSERT_FALSE(img.empty());

    for(int q = 100; q>=0; q-=20)
    {
        std::vector<int> params;
        params.push_back(IMWRITE_WEBP_QUALITY);
        params.push_back(q);
        string output = cv::tempfile(".webp");

        EXPECT_NO_THROW(cv::imwrite(output, img, params));
        cv::Mat img_webp = cv::imread(output);
        EXPECT_EQ(0, remove(output.c_str()));
        EXPECT_FALSE(img_webp.empty());
        EXPECT_EQ(3,   img_webp.channels());
        EXPECT_EQ(512, img_webp.cols);
        EXPECT_EQ(512, img_webp.rows);
    }
}

TEST(Imgcodecs_WebP, encode_decode_with_alpha_webp)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    std::string input = root + "../cv/shared/lena.png";
    cv::Mat img = cv::imread(input);
    ASSERT_FALSE(img.empty());

    std::vector<cv::Mat> imgs;
    cv::split(img, imgs);
    imgs.push_back(cv::Mat(imgs[0]));
    imgs[imgs.size() - 1] = cv::Scalar::all(128);
    cv::merge(imgs, img);

    string output = cv::tempfile(".webp");

    EXPECT_NO_THROW(cv::imwrite(output, img));
    cv::Mat img_webp = cv::imread(output, IMREAD_UNCHANGED);
    cv::Mat img_webp_bgr = cv::imread(output); // IMREAD_COLOR by default
    EXPECT_EQ(0, remove(output.c_str()));
    EXPECT_FALSE(img_webp.empty());
    EXPECT_EQ(4,   img_webp.channels());
    EXPECT_EQ(512, img_webp.cols);
    EXPECT_EQ(512, img_webp.rows);
    EXPECT_FALSE(img_webp_bgr.empty());
    EXPECT_EQ(3,   img_webp_bgr.channels());
    EXPECT_EQ(512, img_webp_bgr.cols);
    EXPECT_EQ(512, img_webp_bgr.rows);
}

TEST(Imgcodecs_WebP, load_save_multiframes)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/OpenCV_logo_white.png";
    vector<Mat> png_frames;

    Mat image = imread(filename, IMREAD_UNCHANGED);
    png_frames.push_back(image.clone());
    Mat roi = image(Rect(0, 680, 680, 220));

    for (int i = 0; i < 15; i++)
    {
        roi = roi - Scalar(0, 0, 0, 20);
        png_frames.push_back(image.clone());
    }

    string output = cv::tempfile(".webp");
    EXPECT_EQ(true, imwrite(output, png_frames));
    vector<Mat> webp_frames;
    EXPECT_EQ(true, imreadmulti(output, webp_frames, IMREAD_UNCHANGED));
    EXPECT_EQ(png_frames.size() - 2, webp_frames.size()); // because last 3 images are identical so 1 image inserted as last frame and its duration calculated by libwebP
    //EXPECT_EQ(14, imcount(output)); //TO DO : actual return value is 1. should be frames count
    EXPECT_EQ(0, remove(output.c_str()));
}

TEST(Imgcodecs_WebP, load_save_animation)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/OpenCV_logo_white.png";
    Animation l_animation, s_animation;

    Mat image = imread(filename, IMREAD_UNCHANGED);
    s_animation.frames.push_back(image.clone());
    Mat roi = image(Rect(0, 680, 680, 220));
    int timestamp = 0;
    s_animation.timestamps.push_back(timestamp);
    s_animation.bgcolor = 0xff00ff;
    s_animation.loop_count = 5;

    for (int i = 0; i < 15; i++)
    {
        roi = roi - Scalar(0, 0, 0, 20);
        s_animation.frames.push_back(image.clone());
        timestamp += 100;
        s_animation.timestamps.push_back(timestamp);
    }

    string output = cv::tempfile(".webp");

    EXPECT_EQ(true, imwriteanimation(output, s_animation));
    EXPECT_EQ(true, imreadanimation(output, l_animation));
    EXPECT_EQ(l_animation.frames.size(), s_animation.frames.size() - 2); // because last 3 images are identical so 1 image inserted as last frame and its duration calculated by libwebP
    EXPECT_EQ(l_animation.bgcolor, s_animation.bgcolor);
    EXPECT_EQ(l_animation.loop_count, s_animation.loop_count);
    //EXPECT_EQ(0, remove(output.c_str()));
}

#endif // HAVE_WEBP

}} // namespace
