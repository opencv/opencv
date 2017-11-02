/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

using namespace cv;
using namespace std;
using namespace std::tr1;

typedef tuple<string, int> File_Mode;
typedef testing::TestWithParam<File_Mode> Imgcodecs_FileMode;

TEST_P(Imgcodecs_FileMode, regression)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + get<0>(GetParam());
    const int mode = get<1>(GetParam());

    const Mat single = imread(filename, mode);
    ASSERT_FALSE(single.empty());

    vector<Mat> pages;
    ASSERT_TRUE(imreadmulti(filename, pages, mode));
    ASSERT_FALSE(pages.empty());
    const Mat page = pages[0];
    ASSERT_FALSE(page.empty());

    EXPECT_EQ(page.channels(), single.channels());
    EXPECT_EQ(page.depth(), single.depth());
    EXPECT_EQ(page.size().height, single.size().height);
    EXPECT_EQ(page.size().width, single.size().width);
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), page, single);
}

const string all_images[] =
{
#ifdef HAVE_JASPER
    "readwrite/Rome.jp2",
    "readwrite/Bretagne2.jp2",
    "readwrite/Bretagne2.jp2",
    "readwrite/Grey.jp2",
    "readwrite/Grey.jp2",
#endif
#ifdef HAVE_GDCM
    "readwrite/int16-mono1.dcm",
    "readwrite/uint8-mono2.dcm",
    "readwrite/uint16-mono2.dcm",
    "readwrite/uint8-rgb.dcm",
#endif
    "readwrite/color_palette_alpha.png",
    "readwrite/multipage.tif",
    "readwrite/ordinary.bmp",
    "readwrite/rle8.bmp",
    "readwrite/test_1_c1.jpg",
    "readwrite/rle.hdr"
};

const int basic_modes[] =
{
    IMREAD_UNCHANGED,
    IMREAD_GRAYSCALE,
    IMREAD_COLOR,
    IMREAD_ANYDEPTH,
    IMREAD_ANYCOLOR
};

INSTANTIATE_TEST_CASE_P(All, Imgcodecs_FileMode,
                        testing::Combine(
                            testing::ValuesIn(all_images),
                            testing::ValuesIn(basic_modes)));

// GDAL does not support "hdr", "dcm" and have problems with "jp2"
struct notForGDAL {
    bool operator()(const string &name) const {
        const string &ext = name.substr(name.size() - 3, 3);
        return ext == "hdr" || ext == "dcm" || ext == "jp2";
    }
};

inline vector<string> gdal_images()
{
    vector<string> res;
    back_insert_iterator< vector<string> > it(res);
    remove_copy_if(all_images, all_images + sizeof(all_images)/sizeof(all_images[0]), it, notForGDAL());
    return res;
}

INSTANTIATE_TEST_CASE_P(GDAL, Imgcodecs_FileMode,
                        testing::Combine(
                            testing::ValuesIn(gdal_images()),
                            testing::Values(IMREAD_LOAD_GDAL)));

//==================================================================================================

typedef tuple<string, Size> Ext_Size;
typedef testing::TestWithParam<Ext_Size> Imgcodecs_ExtSize;

TEST_P(Imgcodecs_ExtSize, write_imageseq)
{
    const string ext = get<0>(GetParam());
    const Size size = get<1>(GetParam());
    const Point2i center = Point2i(size.width / 2, size.height / 2);
    const int radius = std::min(size.height, size.width / 4);

    for (int cn = 1; cn <= 4; cn++)
    {
        SCOPED_TRACE(format("channels %d", cn));
        if (cn == 2)
            continue;
        if (cn == 4 && ext != ".tiff")
            continue;
        string filename = cv::tempfile(format("%d%s", cn, ext.c_str()).c_str());

        Mat img_gt(size, CV_MAKETYPE(CV_8U, cn), Scalar::all(0));
        circle(img_gt, center, radius, Scalar::all(255));
        ASSERT_TRUE(imwrite(filename, img_gt));

        Mat img = imread(filename, IMREAD_UNCHANGED);
        ASSERT_FALSE(img.empty());
        EXPECT_EQ(img.size(), img.size());
        EXPECT_EQ(img.type(), img.type());
        EXPECT_EQ(cn, img.channels());

        if (ext == ".jpg")
        {
            // JPEG format does not provide 100% accuracy
            // using fuzzy image comparison
            double n = cvtest::norm(img, img_gt, NORM_L1);
            double expected = 0.07 * img.size().area();
            EXPECT_LT(n, expected);
            EXPECT_PRED_FORMAT2(cvtest::MatComparator(10, 0), img, img_gt);
        }
        else
        {
            double n = cvtest::norm(img, img_gt, NORM_L2);
            EXPECT_LT(n, 1.);
            EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img, img_gt);
        }
        EXPECT_EQ(0, remove(filename.c_str()));
    }
}

const string all_exts[] =
{
#ifdef HAVE_PNG
    ".png",
#endif
#ifdef HAVE_TIFF
    ".tiff",
#endif
#ifdef HAVE_JPEG
    ".jpg",
#endif
    ".bmp",
    ".pgm",
    ".pam"
};

vector<Size> all_sizes()
{
    vector<Size> res;
    for (int k = 1; k <= 5; ++k)
        res.push_back(Size(640 * k, 480 * k));
    return res;
}

INSTANTIATE_TEST_CASE_P(All, Imgcodecs_ExtSize,
                        testing::Combine(
                            testing::ValuesIn(all_exts),
                            testing::ValuesIn(all_sizes())));

//==================================================================================================

TEST(Imgcodecs_Bmp, read_rle8)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    Mat rle = imread(root + "readwrite/rle8.bmp");
    ASSERT_FALSE(rle.empty());
    Mat ord = imread(root + "readwrite/ordinary.bmp");
    ASSERT_FALSE(ord.empty());
    EXPECT_LE(cvtest::norm(rle, ord, NORM_L2), 1.e-10);
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), rle, ord);
}

TEST(Imgcodecs_Hdr, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "/readwrite/";
    string name_rle = folder + "rle.hdr";
    string name_no_rle = folder + "no_rle.hdr";
    Mat img_rle = imread(name_rle, -1);
    ASSERT_FALSE(img_rle.empty()) << "Could not open " << name_rle;
    Mat img_no_rle = imread(name_no_rle, -1);
    ASSERT_FALSE(img_no_rle.empty()) << "Could not open " << name_no_rle;

    double min = 0.0, max = 1.0;
    minMaxLoc(abs(img_rle - img_no_rle), &min, &max);
    ASSERT_FALSE(max > DBL_EPSILON);
    string tmp_file_name = tempfile(".hdr");
    vector<int>param(1);
    for(int i = 0; i < 2; i++) {
        param[0] = i;
        imwrite(tmp_file_name, img_rle, param);
        Mat written_img = imread(tmp_file_name, -1);
        ASSERT_FALSE(written_img.empty()) << "Could not open " << tmp_file_name;
        minMaxLoc(abs(img_rle - written_img), &min, &max);
        ASSERT_FALSE(max > DBL_EPSILON);
    }
    remove(tmp_file_name.c_str());
}

TEST(Imgcodecs_Pam, read_write)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "readwrite/";
    string filepath = folder + "lena.pam";

    cv::Mat img = cv::imread(filepath);
    ASSERT_FALSE(img.empty());

    std::vector<int> params;
    params.push_back(IMWRITE_PAM_TUPLETYPE);
    params.push_back(IMWRITE_PAM_FORMAT_RGB);

    string writefile = cv::tempfile(".pam");
    EXPECT_NO_THROW(cv::imwrite(writefile, img, params));
    cv::Mat reread = cv::imread(writefile);

    string writefile_no_param = cv::tempfile(".pam");
    EXPECT_NO_THROW(cv::imwrite(writefile_no_param, img));
    cv::Mat reread_no_param = cv::imread(writefile_no_param);

    EXPECT_EQ(0, cvtest::norm(reread, reread_no_param, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(img, reread, NORM_INF));

    remove(writefile.c_str());
    remove(writefile_no_param.c_str());
}
