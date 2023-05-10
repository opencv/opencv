// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

#include <fstream>

#ifdef HAVE_AVIF

namespace opencv_test {
namespace {

class Imgcodecs_Avif_EncodeDecodeSuite
    : public testing::TestWithParam<std::tuple<int, int, int, ImreadModes>> {
 protected:
  static cv::Mat modifyImage(const cv::Mat& img_original, int channels,
                             bool depth_is_8) {
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
    // Convert image to CV_32F for some bit depths.
    if (!depth_is_8) img.convertTo(img_final, CV_32F, 1. / 255.);

    return img_final;
  }
};

class Imgcodecs_Avif_Image_EncodeDecodeSuite
    : public Imgcodecs_Avif_EncodeDecodeSuite {
 public:
  static const cv::Mat& get_img_original(int channels, bool depth_is_8) {
    const Key key = {channels, depth_is_8};
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
      for (bool depth_is_8 : {false, true}) {
        const Key key{channels, depth_is_8};
        imgs_[key] = modifyImage(img_resized, channels, depth_is_8);
      }
    }
  }

  static int kWidth;
  static int kHeight;

 private:
  typedef std::tuple<int, bool> Key;
  static std::map<Key, cv::Mat> imgs_;
};
std::map<std::tuple<int, bool>, cv::Mat>
    Imgcodecs_Avif_Image_EncodeDecodeSuite::imgs_;
int Imgcodecs_Avif_Image_EncodeDecodeSuite::kWidth = 50;
int Imgcodecs_Avif_Image_EncodeDecodeSuite::kHeight = 50;

TEST_P(Imgcodecs_Avif_Image_EncodeDecodeSuite, encode_decode) {
  // Get parameters and image to encode.
  const int bit_depth = std::get<0>(GetParam());
  const int channels = std::get<1>(GetParam());
  const int quality = std::get<2>(GetParam());
  const int imread_mode = std::get<3>(GetParam());
  const cv::Mat& img_original = get_img_original(channels, bit_depth == 8);
  ASSERT_FALSE(img_original.empty());

  // Encode.
  const string output = cv::tempfile(".avif");
  const std::vector<int> params = {cv::IMWRITE_AVIF_QUALITY, quality,
                                   cv::IMWRITE_AVIF_DEPTH, bit_depth};
  if (bit_depth != 8 && bit_depth != 10 && bit_depth != 12) {
    EXPECT_NO_FATAL_FAILURE(cv::imwrite(output, img_original, params));
    return;
  } else {
    EXPECT_NO_THROW(cv::imwrite(output, img_original, params));
  }

  // Read from file.
  const cv::Mat img_imread = cv::imread(output, imread_mode);

  // Put file into buffer and read from buffer.
  std::ifstream file(output, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> buf(size);
  EXPECT_TRUE(file.read(reinterpret_cast<char*>(buf.data()), size));

  EXPECT_EQ(0, remove(output.c_str()));

  const cv::Mat img_decode = cv::imdecode(buf, imread_mode);

  // Test resulting images.
  ASSERT_EQ(img_imread.type(), img_decode.type());
  for (const cv::Mat& img : {img_imread, img_decode}) {
    ASSERT_FALSE(img.empty());
    EXPECT_EQ(Imgcodecs_Avif_Image_EncodeDecodeSuite::kWidth, img.cols);
    EXPECT_EQ(Imgcodecs_Avif_Image_EncodeDecodeSuite::kHeight, img.rows);

    if (imread_mode == IMREAD_UNCHANGED) {
      ASSERT_EQ(img_original.type(), img.type());
      // Lossless.
      if (quality == 100) {
        if (img_original.depth() == CV_8U) {
          EXPECT_EQ(0, cvtest::norm(img, img_original, NORM_INF));
        } else {
          // For CV_32F, results can be slightly off due to normalization.
          EXPECT_LE(cvtest::norm(img, img_original, NORM_INF), 1);
        }
      }
    }
  }
}

INSTANTIATE_TEST_CASE_P(
    Imgcodecs_AVIF, Imgcodecs_Avif_Image_EncodeDecodeSuite,
    ::testing::Combine(::testing::ValuesIn({6, 8, 10, 12}),
                       ::testing::ValuesIn({1, 3, 4}),
                       ::testing::ValuesIn({0, 50, 100}),
                       ::testing::ValuesIn({IMREAD_COLOR, IMREAD_UNCHANGED})));

////////////////////////////////////////////////////////////////////////////////

class Imgcodecs_Avif_Animation_EncodeDecodeSuite
    : public Imgcodecs_Avif_EncodeDecodeSuite {
 public:
  static const std::vector<cv::Mat>& get_anim_original(int channels,
                                                       bool depth_is_8) {
    const Key key = {channels, depth_is_8};
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
      for (bool depth_is_8 : {false, true}) {
        const Key key{channels, depth_is_8};
        const cv::Mat img = modifyImage(img_resized, channels, depth_is_8);
        anims_[key].push_back(img);
        cv::Mat img2;
        cv::flip(img, img2, 0);
        anims_[key].push_back(img2);
      }
    }
  }

  static int kWidth;
  static int kHeight;

 private:
  typedef std::tuple<int, bool> Key;
  static std::map<Key, std::vector<cv::Mat>> anims_;
};
std::map<std::tuple<int, bool>, std::vector<cv::Mat>>
    Imgcodecs_Avif_Animation_EncodeDecodeSuite::anims_;
int Imgcodecs_Avif_Animation_EncodeDecodeSuite::kWidth = 5;
int Imgcodecs_Avif_Animation_EncodeDecodeSuite::kHeight = 5;

TEST_P(Imgcodecs_Avif_Animation_EncodeDecodeSuite, encode_decode) {
  // Get parameters and image to encode.
  const int bit_depth = std::get<0>(GetParam());
  const int channels = std::get<1>(GetParam());
  const int quality = std::get<2>(GetParam());
  const int imread_mode = std::get<3>(GetParam());
  const std::vector<cv::Mat>& anim_original =
      get_anim_original(channels, bit_depth == 8);
  ASSERT_FALSE(anim_original.empty());

  // Encode.
  const string output = cv::tempfile(".avif");
  const std::vector<int> params = {cv::IMWRITE_AVIF_DEPTH, bit_depth,
                                   cv::IMWRITE_AVIF_QUALITY, quality};
  if (bit_depth != 8 && bit_depth != 10 && bit_depth != 12) {
    EXPECT_NO_FATAL_FAILURE(cv::imwritemulti(output, anim_original, params));
    return;
  } else {
    EXPECT_NO_THROW(cv::imwritemulti(output, anim_original, params));
  }

  // Read from file.
  std::vector<cv::Mat> anim_read;
  ASSERT_TRUE(cv::imreadmulti(output, anim_read, imread_mode));

  // Put file into buffer and read from buffer.
  std::ifstream file(output, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> buf(size);
  EXPECT_TRUE(file.read(reinterpret_cast<char*>(buf.data()), size));

  EXPECT_EQ(0, remove(output.c_str()));

  // Test resulting images.
  ASSERT_EQ(anim_original.size(), anim_read.size());
  for (size_t i = 0; i < anim_read.size(); ++i) {
    const cv::Mat img = anim_read[i];
    ASSERT_FALSE(img.empty());
    EXPECT_EQ(Imgcodecs_Avif_Animation_EncodeDecodeSuite::kWidth, img.cols);
    EXPECT_EQ(Imgcodecs_Avif_Animation_EncodeDecodeSuite::kHeight, img.rows);

    if (imread_mode == IMREAD_UNCHANGED) {
      ASSERT_EQ(anim_original[i].type(), img.type());
    }
  }
}

INSTANTIATE_TEST_CASE_P(
    Imgcodecs_AVIF, Imgcodecs_Avif_Animation_EncodeDecodeSuite,
    ::testing::Combine(::testing::ValuesIn({8, 10, 12}),
                       ::testing::ValuesIn({1, 3, 4}), ::testing::ValuesIn({50}),
                       ::testing::ValuesIn({IMREAD_COLOR, IMREAD_UNCHANGED})));

}  // namespace
}  // namespace opencv_test

#endif  // HAVE_AVIF
