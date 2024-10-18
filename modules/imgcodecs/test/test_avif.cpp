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
    encoding_params_ = {cv::IMWRITE_AVIF_QUALITY, quality_,
                        cv::IMWRITE_AVIF_DEPTH, bit_depth_};
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
        const float norm = cvtest::norm(img, img_original, NORM_L2) /
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
    Imgcodecs_Avif_Image_RoundTripSuite::imgs_;
const int Imgcodecs_Avif_Image_RoundTripSuite::kWidth = 51;
const int Imgcodecs_Avif_Image_RoundTripSuite::kHeight = 31;

class Imgcodecs_Avif_Image_WriteReadSuite
    : public Imgcodecs_Avif_Image_RoundTripSuite {};

TEST_P(Imgcodecs_Avif_Image_WriteReadSuite, imwrite_imread) {
  const cv::Mat& img_original = get_img_original();
  ASSERT_FALSE(img_original.empty());

  // Encode.
  const string output = cv::tempfile(".avif");
  if (!IsBitDepthValid()) {
    EXPECT_NO_FATAL_FAILURE(
        cv::imwrite(output, img_original, encoding_params_));
    EXPECT_NE(0, remove(output.c_str()));
    return;
  }
  EXPECT_NO_THROW(cv::imwrite(output, img_original, encoding_params_));

  // Read from file.
  const cv::Mat img = cv::imread(output, imread_mode_);

  ValidateRead(img_original, img);

  EXPECT_EQ(0, remove(output.c_str()));
}

INSTANTIATE_TEST_CASE_P(
    Imgcodecs_AVIF, Imgcodecs_Avif_Image_WriteReadSuite,
    ::testing::Combine(::testing::ValuesIn({6, 8, 10, 12}),
                       ::testing::ValuesIn({1, 3, 4}),
                       ::testing::ValuesIn({0, 50, 100}),
                       ::testing::ValuesIn({IMREAD_UNCHANGED, IMREAD_GRAYSCALE,
                                            IMREAD_COLOR, IMREAD_COLOR_RGB})));

class Imgcodecs_Avif_Image_EncodeDecodeSuite
    : public Imgcodecs_Avif_Image_RoundTripSuite {};

TEST_P(Imgcodecs_Avif_Image_EncodeDecodeSuite, imencode_imdecode) {
  const cv::Mat& img_original = get_img_original();
  ASSERT_FALSE(img_original.empty());

  // Encode.
  std::vector<unsigned char> buf;
  bool result = true;
  EXPECT_NO_THROW(
      result = cv::imencode(".avif", img_original, buf, encoding_params_););

  if (!IsBitDepthValid()) {
    EXPECT_FALSE(result);
    return;
  }
  EXPECT_TRUE(result);

  // Read back.
  const cv::Mat img = cv::imdecode(buf, imread_mode_);

  ValidateRead(img_original, img);
}

INSTANTIATE_TEST_CASE_P(
    Imgcodecs_AVIF, Imgcodecs_Avif_Image_EncodeDecodeSuite,
    ::testing::Combine(::testing::ValuesIn({6, 8, 10, 12}),
                       ::testing::ValuesIn({1, 3, 4}),
                       ::testing::ValuesIn({0, 50, 100}),
                       ::testing::ValuesIn({IMREAD_UNCHANGED, IMREAD_GRAYSCALE,
                                            IMREAD_COLOR, IMREAD_COLOR_RGB})));

////////////////////////////////////////////////////////////////////////////////

typedef testing::TestWithParam<string> Imgcodecs_AVIF_Exif;

TEST_P(Imgcodecs_AVIF_Exif, exif_orientation) {
  const string root = cvtest::TS::ptr()->get_data_path();
  const string filename = root + GetParam();
  const int colorThresholdHigh = 250;
  const int colorThresholdLow = 5;

  Mat m_img = imread(filename);
  ASSERT_FALSE(m_img.empty());
  Vec3b vec;

  // Checking the first quadrant (with supposed red)
  vec = m_img.at<Vec3b>(2, 2);  // some point inside the square
  EXPECT_LE(vec.val[0], colorThresholdLow);
  EXPECT_LE(vec.val[1], colorThresholdLow);
  EXPECT_GE(vec.val[2], colorThresholdHigh);

  // Checking the second quadrant (with supposed green)
  vec = m_img.at<Vec3b>(2, 7);  // some point inside the square
  EXPECT_LE(vec.val[0], colorThresholdLow);
  EXPECT_GE(vec.val[1], colorThresholdHigh);
  EXPECT_LE(vec.val[2], colorThresholdLow);

  // Checking the third quadrant (with supposed blue)
  vec = m_img.at<Vec3b>(7, 2);  // some point inside the square
  EXPECT_GE(vec.val[0], colorThresholdHigh);
  EXPECT_LE(vec.val[1], colorThresholdLow);
  EXPECT_LE(vec.val[2], colorThresholdLow);
}

const string exif_files[] = {"readwrite/testExifOrientation_1.avif",
                             "readwrite/testExifOrientation_2.avif",
                             "readwrite/testExifOrientation_3.avif",
                             "readwrite/testExifOrientation_4.avif",
                             "readwrite/testExifOrientation_5.avif",
                             "readwrite/testExifOrientation_6.avif",
                             "readwrite/testExifOrientation_7.avif",
                             "readwrite/testExifOrientation_8.avif"};

INSTANTIATE_TEST_CASE_P(ExifFiles, Imgcodecs_AVIF_Exif,
                        testing::ValuesIn(exif_files));

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

}  // namespace
}  // namespace opencv_test

#endif  // HAVE_AVIF
