// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"

namespace opencv_test { namespace {
const string gifsuite_files_multi[]={
        "basi3p01",
        "basi3p02",
        "basi3p04",
        "basn3p01",
        "basn3p02",
        "basn3p04",
        "ccwn3p08",
        "ch1n3p04",
        "cs3n3p08",
        "cs5n3p08",
        "cs8n3p08",
        "g03n3p04",
        "g04n3p04",
        "g05n3p04",
        "g07n3p04",
        "g10n3p04",
        "g25n3p04",
        "s32i3p04",
        "s32n3p04",
        "tp0n3p08",
};
const string gifsuite_files_single[] = {
    "basi3p01",
    "basi3p02",
    "basi3p04",
    "basn3p01",
    "basn3p02",
    "basn3p04",
    "ccwn3p08",
    "cdfn2c08",
    "cdhn2c08",
    "cdsn2c08",
    "cdun2c08",
    "ch1n3p04",
    "cs3n3p08",
    "cs5n2c08",
    "cs5n3p08",
    "cs8n2c08",
    "cs8n3p08",
    "exif2c08",
    "g03n2c08",
    "g03n3p04",
    "g04n2c08",
    "g04n3p04",
    "g05n2c08",
    "g05n3p04",
    "g07n2c08",
    "g07n3p04",
    "g10n2c08",
    "g10n3p04",
    "g25n2c08",
    "g25n3p04",
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
    "tp0n3p08",
};
const std::pair<string,int> gifsuite_files_bgra[]={
        make_pair("gif_bgra1",53287),
        make_pair("gif_bgra2",52651),
        make_pair("gif_bgra3",54809),
        make_pair("gif_bgra4",57562),
        make_pair("gif_bgra5",56733),
        make_pair("gif_bgra6",52110),
};
TEST(Imgcodecs_Gif, read_gif_multi)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "gifsuite/gif_multi.gif";
    vector<cv::Mat> img_vec_8UC4;
    ASSERT_NO_THROW(cv::imreadmulti(filename, img_vec_8UC4,0,20));
    EXPECT_EQ(img_vec_8UC4.size(), imcount(filename));
    vector<cv::Mat> img_vec_8UC3;
    for(const auto & i : img_vec_8UC4){
        cv::Mat img_tmp;
        cvtColor(i,img_tmp,COLOR_BGRA2BGR);
        img_vec_8UC3.push_back(img_tmp);
    }
    const long unsigned int expected_size=20;
    EXPECT_EQ(img_vec_8UC3.size(),expected_size);
    for(long unsigned int i=0;i<img_vec_8UC3.size();i++){
        cv::Mat img=img_vec_8UC3[i];
        const string png_filename = root + "pngsuite/" + gifsuite_files_multi[i] + ".png";
        cv::Mat img_png;
        ASSERT_NO_THROW(img_png = imread(png_filename,IMREAD_UNCHANGED));
        ASSERT_FALSE(img_png.empty());
        EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img, img_png);
    }
}
typedef testing::TestWithParam<string> Imgcodecs_Gif_GifSuite_SingleFrame;
TEST_P(Imgcodecs_Gif_GifSuite_SingleFrame,read_gif_single)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "gifsuite/" + GetParam() + ".gif";
    const string png_filename=root + "pngsuite/" + GetParam() + ".png";
    const long unsigned int expected_size = 1;
    EXPECT_EQ(expected_size, imcount(filename));
    cv::Mat img_8UC4;
    ASSERT_NO_THROW(img_8UC4 = cv::imread(filename,IMREAD_UNCHANGED));
    ASSERT_FALSE(img_8UC4.empty());
    cv::Mat img_8UC3;
    cvtColor(img_8UC4,img_8UC3,COLOR_BGRA2BGR);
    cv::Mat img_png;
    ASSERT_NO_THROW(img_png = cv::imread(png_filename,IMREAD_UNCHANGED));
    ASSERT_FALSE(img_png.empty());
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img_8UC3, img_png);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgcodecs_Gif_GifSuite_SingleFrame,
                        testing::ValuesIn(gifsuite_files_single));
TEST(Imgcodecs_Gif, read_gif_big){
    const string root = cvtest::TS::ptr()->get_data_path();
    const string gif_filename = root + "gifsuite/gif_big.gif";
    const string png_filename = root + "gifsuite/gif_big.png";
    cv::Mat img_8UC4;
    ASSERT_NO_THROW(img_8UC4 = imread(gif_filename, IMREAD_UNCHANGED));
    ASSERT_FALSE(img_8UC4.empty());
    cv::Mat img_8UC3;
    const int expected_col=1303;
    const int expected_row=1391;
    EXPECT_EQ(expected_col, img_8UC4.cols);
    EXPECT_EQ(expected_row, img_8UC4.rows);
    ASSERT_NO_THROW(cvtColor(img_8UC4,img_8UC3,COLOR_BGRA2BGR));
    EXPECT_EQ(expected_col, img_8UC3.cols);
    EXPECT_EQ(expected_row, img_8UC3.rows);
    cv::Mat img_png;
    ASSERT_NO_THROW(img_png=imread(png_filename,IMREAD_UNCHANGED));
    ASSERT_FALSE(img_png.empty());
    cv::Mat img_png_8UC3;
    ASSERT_NO_THROW(cvtColor(img_png,img_png_8UC3,COLOR_BGRA2BGR));
    EXPECT_EQ(img_8UC3.size,img_png_8UC3.size);
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img_8UC3, img_png_8UC3);
}
typedef testing::TestWithParam<std::pair<string,int>> Imgcodecs_Gif_GifSuite_SingleFrame_BGRA;
TEST_P(Imgcodecs_Gif_GifSuite_SingleFrame_BGRA , read_gif_single_bgra){
    const string root = cvtest::TS::ptr()->get_data_path();
    const string gif_filename = root + "gifsuite/" + GetParam().first + ".gif";
    const string png_filename = root + "gifsuite/" + GetParam().first + ".png";
    cv::Mat gif_img;
    cv::Mat png_img;
    ASSERT_NO_THROW(gif_img = cv::imread(gif_filename,IMREAD_UNCHANGED));
    ASSERT_FALSE(gif_img.empty());
    ASSERT_NO_THROW(png_img = cv::imread(png_filename,IMREAD_UNCHANGED));
    ASSERT_FALSE(png_img.empty());
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), gif_img, png_img);
    int transparent_count = 0;
    for(int i=0;i<gif_img.rows;i++){
        for(int j=0;j<gif_img.cols;j++){
            cv::Vec4b pixel1 = gif_img.at<cv::Vec4b>(i,j);
            if((int)(pixel1[3]) == 0){
                transparent_count++;
            }
        }
    }
    EXPECT_EQ(transparent_count,GetParam().second);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgcodecs_Gif_GifSuite_SingleFrame_BGRA ,
                                     testing::ValuesIn(gifsuite_files_bgra));
TEST(Imgcodecs_Gif,read_gif_multi_bgra){
            const string root = cvtest::TS::ptr()->get_data_path();
            const string gif_filename = root + "gifsuite/gif_multi_bgra.gif";
    vector<cv::Mat> img_vec;
    ASSERT_NO_THROW(cv::imreadmulti(gif_filename,img_vec));
    EXPECT_EQ(imcount(gif_filename),img_vec.size());
    const int fixed_transparent_count = 53211;
    for(auto & frame_count : img_vec){
        int transparent_count=0;
        for(int i=0;i<frame_count.rows;i++){
            for(int j=0;j<frame_count.cols;j++){
                cv::Vec4b pixel1 = frame_count.at<cv::Vec4b>(i,j);
                if((int)(pixel1[3]) == 0){
                    transparent_count++;
                }
            }
        }
        EXPECT_EQ(fixed_transparent_count,transparent_count);
    }
}
TEST(Imgcodecs_Gif, read_gif_special){
    const string root = cvtest::TS::ptr()->get_data_path();
    const string gif_filename1 = root + "gifsuite/special1.gif";
    const string png_filename1 = root + "gifsuite/special1.png";
    const string gif_filename2 = root + "gifsuite/special2.gif";
    const string png_filename2 = root + "gifsuite/special2.png";
    cv::Mat gif_img1;
    ASSERT_NO_THROW(gif_img1 = cv::imread(gif_filename1,IMREAD_UNCHANGED));
    ASSERT_FALSE(gif_img1.empty());
    cv::Mat png_img1;
    ASSERT_NO_THROW(png_img1 = cv::imread(png_filename1,IMREAD_UNCHANGED));
    ASSERT_FALSE(png_img1.empty());
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), gif_img1, png_img1);
    cv::Mat gif_img2;
    ASSERT_NO_THROW(gif_img2 = cv::imread(gif_filename2,IMREAD_UNCHANGED));
    ASSERT_FALSE(gif_img2.empty());
    cv::Mat png_img2;
    ASSERT_NO_THROW(png_img2 = cv::imread(png_filename2,IMREAD_UNCHANGED));
    ASSERT_FALSE(png_img2.empty());
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), gif_img2, png_img2);
}
}//opencv_test
}//namespace
