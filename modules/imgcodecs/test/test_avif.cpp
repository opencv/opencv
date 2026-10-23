// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#include <cstdint>
#include <fstream>

#include "test_precomp.hpp"

#ifdef HAVE_AVIF

namespace opencv_test {
namespace {

class Imgcodecs_Avif_RoundTripSuite
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
      cv::Mat alpha(imgs[0].size, CV_8UC1, cv::Scalar(128)); // Do not use opaque alpha plane. it will be dropped by libavif to optimize.
      imgs.push_back(alpha);
      cv::merge(imgs, img);
    } else {
      img = img_original.clone();
    }

    cv::Mat img_final;
    // Convert image to CV_16U for some bit depths.
    if (bit_depth == 10) {
      img.convertTo(img_final, CV_16U, 1023.0 / 255.0);
    } else if (bit_depth == 12) {
      img.convertTo(img_final, CV_16U, 4095.0 / 255.0);
    } else if (bit_depth == 16) {
      img.convertTo(img_final, CV_16U, 65535.0 / 255.0);
    } else {
      img_final = img;
    }

    return img_final;
  }

  void SetUp() {
    bit_depth_ = std::get<0>(GetParam());
    channels_ = std::get<1>(GetParam());
    quality_ = std::get<2>(GetParam());
    imread_mode_ = std::get<3>(GetParam());
    encoding_params_ = {cv::IMWRITE_AVIF_QUALITY, quality_,
                        cv::IMWRITE_AVIF_DEPTH, bit_depth_};
  }

  // Memo: this is used only for animation test. AVIF animation does not support 16 bit_depth.
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
        const float num_samples = static_cast<float>(img.channels() * img.cols * img.rows);
        const float norm_ = cvtest::norm(img, img_original, NORM_L2) / std::sqrt(num_samples);
        const float norm = (bit_depth_ == 16)? norm_ / 65535.0f * 255.0f :
                           (bit_depth_ == 12)? norm_ / 4095.0f * 255.0f :
                           (bit_depth_ == 10)? norm_ / 1023.0f * 255.0f :
                                               norm_; // scale to 8 bit

        if (quality_ == 50) {
          EXPECT_LE(norm, 20); // Maximum = 18.15 -> ceil(18.15 * 1.1) = 20
        } else if (quality_ == 0) {
          EXPECT_LE(norm, 60); // Maximum = 54.0416 -> ceil(54.0416 * 1.1) = 60
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

class Imgcodecs_Avif_Image_RoundTripSuite
    : public Imgcodecs_Avif_RoundTripSuite {
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
      for (int bit_depth : {8, 10, 12, 16}) {
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
    Imgcodecs_Avif_Image_RoundTripSuite::imgs_;
const int Imgcodecs_Avif_Image_RoundTripSuite::kWidth = 51;
const int Imgcodecs_Avif_Image_RoundTripSuite::kHeight = 31;

class Imgcodecs_Avif_Image_WriteReadSuite
    : public Imgcodecs_Avif_Image_RoundTripSuite {};

TEST_P(Imgcodecs_Avif_Image_WriteReadSuite, imwrite_imread) {
#ifndef OPENCV_IMGCODECS_AVIF_SUPPORT_16BIT
  if(bit_depth_ == 16) {
    throw SkipTestException("Test is skipped(libavif 1.4.0+ is required to test 16 bit supporting)");
  }
#endif

  const cv::Mat& img_original = get_img_original();
  ASSERT_FALSE(img_original.empty());

  // Encode.
  const string output = cv::tempfile(".avif");
  EXPECT_NO_THROW(cv::imwrite(output, img_original, encoding_params_));

  // Read from file.
  const cv::Mat img = cv::imread(output, imread_mode_);

  ValidateRead(img_original, img);

  EXPECT_EQ(0, remove(output.c_str()));
}

INSTANTIATE_TEST_CASE_P(
    Imgcodecs_AVIF, Imgcodecs_Avif_Image_WriteReadSuite,
    ::testing::Combine(::testing::ValuesIn({6, 8, 10, 12, 16}),
                       ::testing::ValuesIn({1, 3, 4}),
                       ::testing::ValuesIn({0, 50, 100}),
                       ::testing::ValuesIn({IMREAD_UNCHANGED, IMREAD_GRAYSCALE,
                                            IMREAD_COLOR, IMREAD_COLOR_RGB})));

class Imgcodecs_Avif_Image_EncodeDecodeSuite
    : public Imgcodecs_Avif_Image_RoundTripSuite {};

TEST_P(Imgcodecs_Avif_Image_EncodeDecodeSuite, imencode_imdecode) {
#ifndef OPENCV_IMGCODECS_AVIF_SUPPORT_16BIT
  if(bit_depth_ == 16) {
    throw SkipTestException("Test is skipped(libavif 1.4.0+ is required to test 16 bit supporting)");
  }
#endif

  const cv::Mat& img_original = get_img_original();
  ASSERT_FALSE(img_original.empty());

  // Encode.
  std::vector<unsigned char> buf;
  bool result = true;
  EXPECT_NO_THROW(
      result = cv::imencode(".avif", img_original, buf, encoding_params_););
  EXPECT_TRUE(result);

  // Read back.
  const cv::Mat img = cv::imdecode(buf, imread_mode_);

  ValidateRead(img_original, img);
}

INSTANTIATE_TEST_CASE_P(
    Imgcodecs_AVIF, Imgcodecs_Avif_Image_EncodeDecodeSuite,
    ::testing::Combine(::testing::ValuesIn({6, 8, 10, 12, 16}),
                       ::testing::ValuesIn({1, 3, 4}),
                       ::testing::ValuesIn({0, 50, 100}),
                       ::testing::ValuesIn({IMREAD_UNCHANGED, IMREAD_GRAYSCALE,
                                            IMREAD_COLOR, IMREAD_COLOR_RGB})));

////////////////////////////////////////////////////////////////////////////////

class Imgcodecs_Avif_Animation_RoundTripSuite
    : public Imgcodecs_Avif_RoundTripSuite {
 public:
  const std::vector<cv::Mat>& get_anim_original() {
    const Key key = {channels_, bit_depth_};
    return anims_[key];
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
        const cv::Mat img = modifyImage(img_resized, channels, bit_depth);
        cv::Mat img2, img3;
        cv::flip(img, img2, 0);
        cv::flip(img, img3, -1);
        anims_[key] = {img, img2, img3};
      }
    }
  }

  void ValidateRead(const std::vector<cv::Mat>& anim_original,
                    const std::vector<cv::Mat>& anim) const {
    ASSERT_EQ(anim_original.size(), anim.size());
    for (size_t i = 0; i < anim.size(); ++i) {
      Imgcodecs_Avif_RoundTripSuite::ValidateRead(anim_original[i], anim[i]);
    }
  }

  static const int kWidth;
  static const int kHeight;

 private:
  typedef std::tuple<int, int> Key;
  static std::map<Key, std::vector<cv::Mat>> anims_;
};
std::map<std::tuple<int, int>, std::vector<cv::Mat>>
    Imgcodecs_Avif_Animation_RoundTripSuite::anims_;
const int Imgcodecs_Avif_Animation_RoundTripSuite::kWidth = 5;
const int Imgcodecs_Avif_Animation_RoundTripSuite::kHeight = 5;

class Imgcodecs_Avif_Animation_WriteReadSuite
    : public Imgcodecs_Avif_Animation_RoundTripSuite {};

TEST_P(Imgcodecs_Avif_Animation_WriteReadSuite, encode_decode) {
  const std::vector<cv::Mat>& anim_original = get_anim_original();
  ASSERT_FALSE(anim_original.empty());

  // Encode.
  const string output = cv::tempfile(".avif");
  if (!IsBitDepthValid()) {
    EXPECT_THROW(cv::imwritemulti(output, anim_original, encoding_params_),
                 cv::Exception);
    EXPECT_NE(0, remove(output.c_str()));
    return;
  }
  EXPECT_NO_THROW(cv::imwritemulti(output, anim_original, encoding_params_));
  EXPECT_EQ(anim_original.size(), imcount(output));

  // Read from file.
  std::vector<cv::Mat> anim;
  ASSERT_TRUE(cv::imreadmulti(output, anim, imread_mode_));

  ValidateRead(anim_original, anim);

  EXPECT_EQ(0, remove(output.c_str()));
}

INSTANTIATE_TEST_CASE_P(
    Imgcodecs_AVIF, Imgcodecs_Avif_Animation_WriteReadSuite,
    ::testing::Combine(::testing::ValuesIn({8, 10, 12}),
                       ::testing::ValuesIn({1, 3}), ::testing::ValuesIn({50}),
                       ::testing::ValuesIn({IMREAD_UNCHANGED, IMREAD_GRAYSCALE,
                                            IMREAD_COLOR, IMREAD_COLOR_RGB})));
class Imgcodecs_Avif_Animation_WriteDecodeSuite
    : public Imgcodecs_Avif_Animation_RoundTripSuite {};

TEST_P(Imgcodecs_Avif_Animation_WriteDecodeSuite, encode_decode) {
  const std::vector<cv::Mat>& anim_original = get_anim_original();
  ASSERT_FALSE(anim_original.empty());

  // Encode.
  const string output = cv::tempfile(".avif");
  if (!IsBitDepthValid()) {
    EXPECT_THROW(cv::imwritemulti(output, anim_original, encoding_params_),
                 cv::Exception);
    EXPECT_NE(0, remove(output.c_str()));
    return;
  }
  EXPECT_NO_THROW(cv::imwritemulti(output, anim_original, encoding_params_));

  // Put file into buffer and read from buffer.
  std::ifstream file(output, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> buf(size);
  EXPECT_TRUE(file.read(reinterpret_cast<char*>(buf.data()), size));
  file.close();
  std::vector<cv::Mat> anim;
  ASSERT_TRUE(cv::imdecodemulti(buf, imread_mode_, anim));

  ValidateRead(anim_original, anim);

  if (imread_mode_ == IMREAD_UNCHANGED) {
    ImageCollection collection(output, IMREAD_UNCHANGED);
    anim.clear();
    for (auto&& i : collection)
      anim.push_back(i);
    ValidateRead(anim_original, anim);
  }

  EXPECT_EQ(0, remove(output.c_str()));
}

INSTANTIATE_TEST_CASE_P(
    Imgcodecs_AVIF, Imgcodecs_Avif_Animation_WriteDecodeSuite,
    ::testing::Combine(::testing::ValuesIn({8, 10, 12}),
                       ::testing::ValuesIn({1, 3}), ::testing::ValuesIn({50}),
                       ::testing::ValuesIn({IMREAD_UNCHANGED, IMREAD_GRAYSCALE,
                                            IMREAD_COLOR, IMREAD_COLOR_RGB})));

////////////////////////////////////////////////////////////////////////////////

#ifdef OPENCV_IMGCODECS_AVIF_SUPPORT_16BIT
TEST(Imgcodecs_Avif, unsupported_animation_with_16bit)
#else
TEST(Imgcodecs_Avif, DISABLED_unsupported_animation_with_16bit)
#endif
{
  cv::Mat frame(cv::Size(160, 120), CV_16UC3, cv::Scalar::all(32));
  std::vector<cv::Mat> frames = {frame, frame};
  std::vector<uchar> buf;
  // Encoding 16-bit AVIF animations should fail due to a libavif limitation.
  EXPECT_NO_THROW(EXPECT_FALSE(cv::imencodemulti(".avif", frames, buf, {IMWRITE_AVIF_DEPTH, 16})));
}

}  // namespace
}  // namespace opencv_test

#endif  // HAVE_AVIF
