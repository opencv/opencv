// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"

#ifdef HAVE_IMGCODEC_GIF

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

const string gifsuite_files_read_single[] = {
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
    "g10n3p04"
};

const string gifsuite_files_read_write_suite[]={
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
    ASSERT_NO_THROW(cv::imreadmulti(filename, img_vec_8UC4,0,20,IMREAD_UNCHANGED));
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

TEST_P(Imgcodecs_Gif_GifSuite_SingleFrame, read_gif_single)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "gifsuite/" + GetParam() + ".gif";
    const string png_filename=root + "pngsuite/" + GetParam() + ".png";
    const long unsigned int expected_size = 1;

    EXPECT_EQ(expected_size, imcount(filename));
    cv::Mat img_8UC4;
    ASSERT_NO_THROW(img_8UC4 = cv::imread(filename, IMREAD_UNCHANGED));
    ASSERT_FALSE(img_8UC4.empty());
    cv::Mat img_8UC3;
    ASSERT_NO_THROW(cvtColor(img_8UC4, img_8UC3, COLOR_BGRA2BGR));
    cv::Mat img_png;
    ASSERT_NO_THROW(img_png = cv::imread(png_filename, IMREAD_UNCHANGED));
    ASSERT_FALSE(img_png.empty());
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img_8UC3, img_png);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgcodecs_Gif_GifSuite_SingleFrame,
                        testing::ValuesIn(gifsuite_files_read_single));

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
    ASSERT_NO_THROW(cvtColor(img_8UC4, img_8UC3,COLOR_BGRA2BGR));
    EXPECT_EQ(expected_col, img_8UC3.cols);
    EXPECT_EQ(expected_row, img_8UC3.rows);
    cv::Mat img_png;
    ASSERT_NO_THROW(img_png=imread(png_filename, IMREAD_UNCHANGED));
    ASSERT_FALSE(img_png.empty());
    cv::Mat img_png_8UC3;
    ASSERT_NO_THROW(cvtColor(img_png,img_png_8UC3, COLOR_BGRA2BGR));
    EXPECT_EQ(img_8UC3.size, img_png_8UC3.size);
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img_8UC3, img_png_8UC3);
}

typedef testing::TestWithParam<std::pair<string,int>> Imgcodecs_Gif_GifSuite_SingleFrame_BGRA;

TEST_P(Imgcodecs_Gif_GifSuite_SingleFrame_BGRA, read_gif_single_bgra){
    const string root = cvtest::TS::ptr()->get_data_path();
    const string gif_filename = root + "gifsuite/" + GetParam().first + ".gif";
    const string png_filename = root + "gifsuite/" + GetParam().first + ".png";
    cv::Mat gif_img;
    cv::Mat png_img;
    ASSERT_NO_THROW(gif_img = cv::imread(gif_filename, IMREAD_UNCHANGED));
    ASSERT_FALSE(gif_img.empty());
    ASSERT_NO_THROW(png_img = cv::imread(png_filename, IMREAD_UNCHANGED));
    ASSERT_FALSE(png_img.empty());
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), gif_img, png_img);
    int transparent_count = 0;
    for(int i=0; i<gif_img.rows; i++){
        for(int j=0; j<gif_img.cols; j++){
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
    ASSERT_NO_THROW(cv::imreadmulti(gif_filename, img_vec, IMREAD_UNCHANGED));
    EXPECT_EQ(imcount(gif_filename), img_vec.size());
    const int fixed_transparent_count = 53211;
    for(auto & frame_count : img_vec){
        int transparent_count=0;
        for(int i=0; i<frame_count.rows; i++){
            for(int j=0; j<frame_count.cols; j++){
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
    ASSERT_NO_THROW(gif_img1 = cv::imread(gif_filename1,IMREAD_COLOR));
    ASSERT_FALSE(gif_img1.empty());
    cv::Mat png_img1;
    ASSERT_NO_THROW(png_img1 = cv::imread(png_filename1,IMREAD_COLOR));
    ASSERT_FALSE(png_img1.empty());
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), gif_img1, png_img1);
    cv::Mat gif_img2;
    ASSERT_NO_THROW(gif_img2 = cv::imread(gif_filename2,IMREAD_COLOR));
    ASSERT_FALSE(gif_img2.empty());
    cv::Mat png_img2;
    ASSERT_NO_THROW(png_img2 = cv::imread(png_filename2,IMREAD_COLOR));
    ASSERT_FALSE(png_img2.empty());
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), gif_img2, png_img2);
}

TEST(Imgcodecs_Gif,write_gif_flags){
    const string root = cvtest::TS::ptr()->get_data_path();
    const string png_filename = root + "gifsuite/special1.png";
    vector<uchar> buff;
    const int expected_rows=611;
    const int expected_cols=293;
    Mat img_gt = Mat::ones(expected_rows, expected_cols, CV_8UC1);
    vector<int> param;
    param.push_back(IMWRITE_GIF_QUALITY);
    param.push_back(7);
    param.push_back(IMWRITE_GIF_DITHER);
    param.push_back(2);
    EXPECT_NO_THROW(imencode(".png", img_gt, buff, param));
    Mat img;
    EXPECT_NO_THROW(img = imdecode(buff, IMREAD_ANYDEPTH)); // hang
    EXPECT_FALSE(img.empty());
    EXPECT_EQ(img.cols, expected_cols);
    EXPECT_EQ(img.rows, expected_rows);
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img, img_gt);
}

TEST(Imgcodecs_Gif, write_gif_big) {
    const string root = cvtest::TS::ptr()->get_data_path();
    const string png_filename = root + "gifsuite/gif_big.png";
    const string gif_filename = cv::tempfile(".png");
    cv::Mat img;
    ASSERT_NO_THROW(img = cv::imread(png_filename, IMREAD_UNCHANGED));
    ASSERT_FALSE(img.empty());
    EXPECT_EQ(1303, img.cols);
    EXPECT_EQ(1391, img.rows);
    ASSERT_NO_THROW(imwrite(gif_filename, img));
    cv::Mat img_gif;
    ASSERT_NO_THROW(img_gif = cv::imread(gif_filename, IMREAD_UNCHANGED));
    ASSERT_FALSE(img_gif.empty());
    EXPECT_EQ(1303, img_gif.cols);
    EXPECT_EQ(1391, img_gif.rows);
    EXPECT_EQ(0, remove(gif_filename.c_str()));
}

typedef testing::TestWithParam<string> Imgcodecs_Gif_GifSuite_Read_Write_Suite;

TEST_P(Imgcodecs_Gif_GifSuite_Read_Write_Suite ,read_gif_single)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string png_filename = root + "pngsuite/"+GetParam()+".png";
    const string gif_filename = cv::tempfile(".gif");
    cv::Mat img;
    ASSERT_NO_THROW(img = cv::imread(png_filename, IMREAD_UNCHANGED));
    ASSERT_FALSE(img.empty());
    vector<int> param;
    param.push_back(IMWRITE_GIF_QUALITY);
    param.push_back(8);
    param.push_back(IMWRITE_GIF_DITHER);
    param.push_back(3);
    ASSERT_NO_THROW(imwrite(gif_filename, img, param));
    cv::Mat img_gif;
    ASSERT_NO_THROW(img_gif = cv::imread(gif_filename, IMREAD_UNCHANGED));
    ASSERT_FALSE(img_gif.empty());
    cv::Mat img_8UC3;
    ASSERT_NO_THROW(cv::cvtColor(img_gif, img_8UC3, COLOR_BGRA2BGR));
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(29, 0), img, img_8UC3);
    EXPECT_EQ(0, remove(gif_filename.c_str()));
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgcodecs_Gif_GifSuite_Read_Write_Suite ,
                                     testing::ValuesIn(gifsuite_files_read_write_suite));

TEST(Imgcodecs_Gif, write_gif_multi) {
    const string root = cvtest::TS::ptr()->get_data_path();
    const string gif_filename = cv::tempfile(".gif");
    vector<cv::Mat> img_vec;
    for (long unsigned int i = 0; i < 20; i++) {
        const string png_filename = root + "pngsuite/" + gifsuite_files_multi[i] + ".png";
        cv::Mat img;
        ASSERT_NO_THROW(img = cv::imread(png_filename, IMREAD_UNCHANGED));
        ASSERT_FALSE(img.empty());
        img_vec.push_back(img);
    }
    vector<int> param;
    param.push_back(IMWRITE_GIF_QUALITY);
    param.push_back(8);
    param.push_back(IMWRITE_GIF_DITHER);
    param.push_back(3);
    ASSERT_NO_THROW(cv::imwritemulti(gif_filename, img_vec, param));
    vector<cv::Mat> img_vec_gif;
    ASSERT_NO_THROW(cv::imreadmulti(gif_filename, img_vec_gif));
    EXPECT_EQ(img_vec.size(), img_vec_gif.size());
    for (long unsigned int i = 0; i < img_vec.size(); i++) {
        cv::Mat img_8UC3;
        ASSERT_NO_THROW(cv::cvtColor(img_vec_gif[i], img_8UC3, COLOR_BGRA2BGR));
        EXPECT_PRED_FORMAT2(cvtest::MatComparator(29, 0), img_vec[i], img_8UC3);
    }
    EXPECT_EQ(0, remove(gif_filename.c_str()));
}

TEST(Imgcodecs_Gif, encode_IMREAD_GRAYSCALE) {
    cv::Mat src;
    cv::Mat decoded;
    vector<uint8_t> buf;
    vector<int> param;
    bool ret = false;

    src = cv::Mat(240,240,CV_8UC3,cv::Scalar(128,64,32));
    EXPECT_NO_THROW(ret = imencode(".gif", src, buf, param));
    EXPECT_TRUE(ret);
    EXPECT_NO_THROW(decoded = imdecode(buf, cv::IMREAD_GRAYSCALE));
    EXPECT_FALSE(decoded.empty());
    EXPECT_EQ(decoded.channels(), 1);
}

// See https://github.com/opencv/opencv/issues/26924
TEST(Imgcodecs_Gif, decode_disposal_method)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "gifsuite/disposalMethod.gif";
    cv::Animation anim;
    bool ret = false;
    EXPECT_NO_THROW(ret = imreadanimation(filename, anim, cv::IMREAD_UNCHANGED));
    EXPECT_TRUE(ret);

    /* [Detail of this test]
     * disposalMethod.gif has 5 frames to draw 8x8 rectangles with each color index, offsets and disposal method.
     *   frame 1 draws {1, ... ,1} rectangle at (1,1) with DisposalMethod = 0.
     *   frame 2 draws {2, ... ,2} rectangle at (2,2) with DisposalMethod = 3.
     *   frame 3 draws {3, ... ,3} rectangle at (3,3) with DisposalMethod = 1.
     *   frame 4 draws {4, ... ,4} rectangle at (4,4) with DisposalMethod = 2.
     *   frame 5 draws {5, ... ,5} rectangle at (5,5) with DisposalMethod = 1.
     *
     * To convenience to test, color[N] in the color table has RGB(32*N, some, some).
     *   color[0] = RGB(0,0,0) (background color).
     *   color[1] = RGB(32,0,0)
     *   color[2] = RGB(64,0,255)
     *   color[3] = RGB(96,255,0)
     *   color[4] = RGB(128,128,128)
     *   color[5] = RGB(160,255,255)
     */
    const int refIds[5][6] =
    {// { 0, 0, 0, 0, 0, 0}     0 is background color.
        { 0, 1, 1, 1, 1, 1}, // 1 is to be not disposed.
        { 0, 1, 2, 2, 2, 2}, // 2 is to be restored to previous.
        { 0, 1, 1, 3, 3, 3}, // 3 is to be left in place.
        { 0, 1, 1, 3, 4, 4}, // 4 is to be restored to the background color.
        { 0, 1, 1, 3, 0, 5}, // 5 is to be left in place.
    };

    for(int i = 0 ; i < 5; i++)
    {
        cv::Mat frame = anim.frames[i];
        EXPECT_FALSE(frame.empty());
        EXPECT_EQ(frame.type(), CV_8UC4);
        for(int j = 0; j < 6; j ++ )
        {
            const cv::Scalar p = frame.at<Vec4b>(j,j);
            EXPECT_EQ( p[2], refIds[i][j] * 32 ) << "  i = " << i << " j = " << j << " pixels = " << p;
        }
    }
}

// See https://github.com/opencv/opencv/issues/26970
typedef testing::TestWithParam<int> Imgcodecs_Gif_loop_count;
TEST_P(Imgcodecs_Gif_loop_count, imwriteanimation)
{
    const string gif_filename = cv::tempfile(".gif");

    int loopCount = GetParam();
    cv::Animation anim(loopCount);

    vector<cv::Mat> src;
    for(int n = 1; n <= 5 ; n ++ )
    {
        cv::Mat frame(64, 64, CV_8UC3, cv::Scalar::all(0));
        cv::putText(frame, cv::format("%d", n), cv::Point(0,64), cv::FONT_HERSHEY_PLAIN, 4.0, cv::Scalar::all(255));
        anim.frames.push_back(frame);
        anim.durations.push_back(1000 /* ms */);
    }

    bool ret = false;
#if 0
    // To output gif image for test.
    EXPECT_NO_THROW(ret = imwriteanimation(cv::format("gif_loop-%d.gif", loopCount), anim));
    EXPECT_TRUE(ret);
#endif
    EXPECT_NO_THROW(ret = imwriteanimation(gif_filename, anim));
    EXPECT_TRUE(ret);

    // Read raw GIF data.
    std::ifstream ifs(gif_filename);
    std::stringstream ss;
    ss << ifs.rdbuf();
    string tmp = ss.str();
    std::vector<uint8_t> buf(tmp.begin(), tmp.end());

    std::vector<uint8_t> netscape = {0x21, 0xFF, 0x0B, 'N','E','T','S','C','A','P','E','2','.','0'};
    auto pos = std::search(buf.begin(), buf.end(), netscape.begin(), netscape.end());
    if(loopCount == 1) {
        EXPECT_EQ(pos, buf.end()) << "Netscape Application Block should not be included if Animation.loop_count == 1";
    } else {
        EXPECT_NE(pos, buf.end()) << "Netscape Application Block should be included if Animation.loop_count != 1";
    }

    remove(gif_filename.c_str());
}

INSTANTIATE_TEST_CASE_P(/*nothing*/,
    Imgcodecs_Gif_loop_count,
    testing::Values(
        -1,
         0, // Default, loop-forever
         1,
         2,
     65534,
     65535, // Maximum Limit
     65536
    )
);

typedef testing::TestWithParam<int> Imgcodecs_Gif_duration;
TEST_P(Imgcodecs_Gif_duration, imwriteanimation)
{
    const string gif_filename = cv::tempfile(".gif");

    cv::Animation anim;

    int duration = GetParam();
    vector<cv::Mat> src;
    for(int n = 1; n <= 5 ; n ++ )
    {
        cv::Mat frame(64, 64, CV_8UC3, cv::Scalar::all(0));
        cv::putText(frame, cv::format("%d", n), cv::Point(0,64), cv::FONT_HERSHEY_PLAIN, 4.0, cv::Scalar::all(255));
        anim.frames.push_back(frame);
        anim.durations.push_back(duration /* ms */);
    }

    bool ret = false;
#if 0
    // To output gif image for test.
    EXPECT_NO_THROW(ret = imwriteanimation(cv::format("gif_duration-%d.gif", duration), anim));
    EXPECT_EQ(ret, ( (0 <= duration) && (duration <= 655350) ) );
#endif
    EXPECT_NO_THROW(ret = imwriteanimation(gif_filename, anim));
    EXPECT_EQ(ret, ( (0 <= duration) && (duration <= 655350) ) );

    remove(gif_filename.c_str());
}

INSTANTIATE_TEST_CASE_P(/*nothing*/,
    Imgcodecs_Gif_duration,
    testing::Values(
         -1, // Unsupported
          0, // Undefined Behaviour
          1,
          9,
         10,
         50,
        100, // 10 FPS
       1000, //  1 FPS
     655340,
     655350, // Maximum Limit
     655360  // Unsupported
    )
);

}//opencv_test
}//namespace

#endif
