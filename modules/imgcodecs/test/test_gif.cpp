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
TEST(Imgcodecs_Gif, read_gif_multi)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root+"gifsuite/output.gif";
    vector<cv::Mat> img_vec;
    cv::imreadmulti(filename, img_vec,0,20);
    const long unsigned int expected_size=20;
    EXPECT_EQ(img_vec.size(),expected_size);
    for(long unsigned int i=0;i<img_vec.size();i++){
        cv::Mat img=img_vec[i];
        const string xml_filename=root+"pngsuite/"+gifsuite_files_multi[i]+".xml";
        FileStorage fs(xml_filename, FileStorage::READ);
        EXPECT_TRUE(fs.isOpened());
        Mat gt;
        fs.getFirstTopLevelNode() >> gt;
        EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img, gt);
    }
}
typedef testing::TestWithParam<string> Imgcodecs_Gif_GifSuite_SingleFrame;
TEST_P(Imgcodecs_Gif_GifSuite_SingleFrame,read_gif_single)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "gifsuite/" + GetParam() + ".gif";
    const string xml_filename =  root + "pngsuite/" + GetParam() + ".xml";
    cv::Mat img;
    img=cv::imread(filename,IMREAD_UNCHANGED);
    FileStorage fs(xml_filename, FileStorage::READ);
    EXPECT_TRUE(fs.isOpened());
    Mat gt;
    fs.getFirstTopLevelNode() >> gt;
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img, gt);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgcodecs_Gif_GifSuite_SingleFrame,
                        testing::ValuesIn(gifsuite_files_single));
}//opencv_test
}//namespace

#endif //HAVE_IMGCODEC_GIF